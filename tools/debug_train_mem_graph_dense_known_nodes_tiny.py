#!/usr/bin/env python3
"""Tiny MEM known-node dense graph training/debug wrapper for D3G.

This command is intentionally not a full training entrypoint. It is a capped
smoke/debug helper for the Option 2A `MemGraphDenseKnownNodes` prototype.

Default safety boundary:
- explicit records JSON/JSONL only;
- max 5 records;
- batch size 1 only;
- CPU only in this first implementation;
- checkpoint/model-output writes disabled;
- no HDF5 packing or full dataset export;
- requires explicit acknowledgement flags before running backward/optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from detectron2.config import get_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.mem_observed_gt_dataset import load_mem_observed_gt_records
from data.mem_observed_gt_mapper import MemObservedGtMapper
from models.mem_graph_dense import MemGraphDenseKnownNodes
from utils.configs import add_dep_graph_config, add_detr_config


MAX_RECORDS = 5
MAX_ITER_CAP = 200
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "mem" / "option2a_known_nodes_graph_smoke.yaml"
SCHEMA = "mem_d3g_known_node_tiny_train_debug_summary_v0"


def _as_path(path: Optional[Path]) -> Optional[Path]:
    return Path(path).expanduser() if path is not None else None


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype).cpu()
    return torch.tensor(value, dtype=dtype)


def _float_or_none(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None or value.numel() == 0:
        return None
    scalar = float(value.detach().cpu().item())
    if math.isnan(scalar) or math.isinf(scalar):
        return scalar
    return scalar


def _shape(tensor: Any) -> List[int]:
    return [int(dim) for dim in tensor.shape]


def compute_non_diagonal_mask(target: Any) -> torch.Tensor:
    """Return a boolean mask selecting non-self directed pairs in a square graph."""

    target_tensor = _as_tensor(target)
    if target_tensor.dim() != 2 or target_tensor.shape[0] != target_tensor.shape[1]:
        raise ValueError("target graph must be a square [N,N] matrix")
    num_nodes = int(target_tensor.shape[0])
    return ~torch.eye(num_nodes, dtype=torch.bool)


def compute_average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Compute binary average precision without requiring sklearn.

    Returns None if no positive labels are present, because AP is undefined in
    that case.
    """

    pairs = [(float(score), int(label)) for score, label in zip(scores, labels)]
    num_positive = sum(label for _, label in pairs)
    if num_positive == 0:
        return None
    pairs.sort(key=lambda item: item[0], reverse=True)
    true_positive_count = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            true_positive_count += 1
            precision_sum += true_positive_count / float(rank)
    return precision_sum / float(num_positive)


def summarize_edge_targets(target: Any) -> Dict[str, Any]:
    target_tensor = _as_tensor(target)
    mask = compute_non_diagonal_mask(target_tensor)
    selected = target_tensor[mask]
    num_pairs = int(selected.numel())
    num_positive = int(selected.sum().item())
    return {
        "num_nodes": int(target_tensor.shape[0]),
        "graph_gt_shape": _shape(target_tensor),
        "num_non_diagonal_pairs": num_pairs,
        "num_positive_directed_edges": num_positive,
        "positive_edge_ratio": float(num_positive / num_pairs) if num_pairs else None,
    }


def summarize_logits_by_target(logits: Any, target: Any) -> Dict[str, Any]:
    logits_tensor = _as_tensor(logits)
    target_tensor = _as_tensor(target)
    if logits_tensor.shape != target_tensor.shape:
        raise ValueError("logits and target must have matching shapes")
    mask = compute_non_diagonal_mask(target_tensor)
    selected_logits = logits_tensor[mask]
    selected_target = target_tensor[mask]
    positive_logits = selected_logits[selected_target == 1]
    negative_logits = selected_logits[selected_target == 0]
    positive_probs = torch.sigmoid(positive_logits) if positive_logits.numel() else positive_logits
    negative_probs = torch.sigmoid(negative_logits) if negative_logits.numel() else negative_logits
    return {
        "graph_logits_shape": _shape(logits_tensor),
        "graph_logits_finite": bool(torch.isfinite(logits_tensor).all().item()),
        "mean_positive_logit": _float_or_none(positive_logits.mean() if positive_logits.numel() else None),
        "mean_negative_logit": _float_or_none(negative_logits.mean() if negative_logits.numel() else None),
        "mean_positive_probability": _float_or_none(positive_probs.mean() if positive_probs.numel() else None),
        "mean_negative_probability": _float_or_none(negative_probs.mean() if negative_probs.numel() else None),
    }


def compute_binary_metrics_at_threshold(logits: Any, target: Any, threshold: float = 0.5) -> Dict[str, Any]:
    logits_tensor = _as_tensor(logits)
    target_tensor = _as_tensor(target)
    if logits_tensor.shape != target_tensor.shape:
        raise ValueError("logits and target must have matching shapes")
    mask = compute_non_diagonal_mask(target_tensor)
    probabilities = torch.sigmoid(logits_tensor[mask])
    labels = target_tensor[mask].long()
    predictions = (probabilities >= float(threshold)).long()

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    ap = compute_average_precision(probabilities.detach().cpu().tolist(), labels.detach().cpu().tolist())
    return {
        "threshold": float(threshold),
        "num_pairs": int(labels.numel()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": ap,
    }


def build_mem_graph_dense_known_nodes_tiny_train_cfg(
    config_file: Optional[Path] = DEFAULT_CONFIG_FILE,
    *,
    device: str = "cpu",
    cfg_overrides: Optional[Sequence[str]] = None,
):
    if device != "cpu":
        raise ValueError("tiny train/debug wrapper is CPU-only in this first implementation")
    cfg = get_cfg()
    add_dep_graph_config(cfg)
    add_detr_config(cfg)
    config_file = _as_path(config_file)
    if config_file is not None:
        cfg.merge_from_file(str(config_file))
    if cfg_overrides:
        cfg.merge_from_list(list(cfg_overrides))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = ""
    if cfg.MODEL.META_ARCHITECTURE != "MemGraphDenseKnownNodes":
        raise ValueError("expected MODEL.META_ARCHITECTURE='MemGraphDenseKnownNodes'")
    if cfg.MODEL.GRAPH_HEAD.NAME != "GraphTransformerDense":
        raise ValueError("expected MODEL.GRAPH_HEAD.NAME='GraphTransformerDense'")
    return cfg


def _validate_safety_inputs(
    *,
    max_records: int,
    max_iter: int,
    device: str,
    output_dir: Optional[Path],
    no_checkpoint: bool,
    acknowledge_backward: bool,
    acknowledge_optimizer_step: bool,
) -> None:
    if max_records < 1 or max_records > MAX_RECORDS:
        raise ValueError("max_records must be between 1 and {}".format(MAX_RECORDS))
    if max_iter < 1 or max_iter > MAX_ITER_CAP:
        raise ValueError("max_iter must be between 1 and {} for this tiny debug wrapper".format(MAX_ITER_CAP))
    if device != "cpu":
        raise ValueError("tiny train/debug wrapper is CPU-only in this first implementation")
    if not no_checkpoint:
        raise ValueError("checkpoint/model-output writes are disabled; pass no_checkpoint=True for this wrapper")
    if not acknowledge_backward or not acknowledge_optimizer_step:
        raise ValueError("acknowledge_backward and acknowledge_optimizer_step must both be true")
    if output_dir is not None and output_dir.exists():
        raise FileExistsError("refusing to write into existing output_dir: {}".format(output_dir))


def _parse_sample_ids(sample_ids: Optional[Iterable[str]]) -> List[str]:
    if sample_ids is None:
        return []
    return [str(item).strip() for item in sample_ids if str(item).strip()]


def _select_records(records: Sequence[Dict[str, Any]], sample_ids: Sequence[str], *, split_name: str) -> List[Dict[str, Any]]:
    ids = _parse_sample_ids(sample_ids)
    if not ids:
        return [dict(records[0])] if split_name == "train" and records else []
    by_id = {str(record.get("sample_id")): record for record in records}
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError("{} sample ids not found in records: {}".format(split_name, missing))
    return [dict(by_id[sample_id]) for sample_id in ids]


def _total_parameter_norm(model: torch.nn.Module) -> Dict[str, Any]:
    total_sq = 0.0
    finite = True
    for parameter in model.parameters():
        value = parameter.detach()
        norm = float(value.norm(2).cpu().item())
        finite = finite and math.isfinite(norm)
        total_sq += norm * norm
    total = math.sqrt(total_sq)
    return {"value": float(total), "finite": bool(math.isfinite(total) and finite)}


def _total_gradient_norm(model: torch.nn.Module) -> Dict[str, Any]:
    total_sq = 0.0
    finite = True
    saw_grad = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        saw_grad = True
        value = parameter.grad.detach()
        norm = float(value.norm(2).cpu().item())
        finite = finite and math.isfinite(norm)
        total_sq += norm * norm
    total = math.sqrt(total_sq) if saw_grad else 0.0
    return {"value": float(total), "finite": bool(math.isfinite(total) and finite), "saw_grad": bool(saw_grad)}


def _eval_one_record(model: MemGraphDenseKnownNodes, mapped: Dict[str, Any], *, threshold: float = 0.5) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model([mapped])
    if was_training:
        model.train()
    if len(outputs) != 1:
        raise ValueError("expected one eval output, got {}".format(len(outputs)))
    output = outputs[0]
    logits = output["graph_logits"].detach().cpu()
    target = mapped["graph_gt"].detach().cpu()
    return {
        "image_id": mapped.get("image_id"),
        "target_summary": summarize_edge_targets(target),
        "logit_summary": summarize_logits_by_target(logits, target),
        "metrics_at_threshold_0_5": compute_binary_metrics_at_threshold(logits, target, threshold=threshold),
    }


def _write_summary_json(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mem_graph_dense_known_nodes_tiny_train_summary(
    records_json: Path,
    data_root: str = ".",
    train_sample_ids: Optional[Sequence[str]] = None,
    val_sample_ids: Optional[Sequence[str]] = None,
    max_records: int = 1,
    device: str = "cpu",
    config_file: Optional[Path] = DEFAULT_CONFIG_FILE,
    expected_height: int = 140,
    expected_width: int = 200,
    validate_semantic_range: bool = True,
    cfg_overrides: Optional[Sequence[str]] = None,
    seed: int = 0,
    max_iter: int = 1,
    lr: float = 1e-4,
    output_dir: Optional[Path] = None,
    no_checkpoint: bool = True,
    acknowledge_backward: bool = False,
    acknowledge_optimizer_step: bool = False,
) -> Dict[str, Any]:
    """Run a capped tiny train/debug loop and return a compact JSON summary."""

    records_json = Path(records_json).expanduser()
    output_dir = _as_path(output_dir)
    _validate_safety_inputs(
        max_records=int(max_records),
        max_iter=int(max_iter),
        device=str(device),
        output_dir=output_dir,
        no_checkpoint=bool(no_checkpoint),
        acknowledge_backward=bool(acknowledge_backward),
        acknowledge_optimizer_step=bool(acknowledge_optimizer_step),
    )
    if lr <= 0:
        raise ValueError("lr must be positive")

    torch.manual_seed(int(seed))
    cfg = build_mem_graph_dense_known_nodes_tiny_train_cfg(config_file, device=device, cfg_overrides=cfg_overrides)
    records = load_mem_observed_gt_records(records_json, max_records=max_records)
    if not records:
        raise ValueError("records_json produced no records")
    train_records = _select_records(records, _parse_sample_ids(train_sample_ids), split_name="train")
    val_records = _select_records(records, _parse_sample_ids(val_sample_ids), split_name="val")
    if not train_records:
        raise ValueError("at least one train record is required")

    mapper = MemObservedGtMapper(
        data_root=data_root,
        is_train=True,
        graph_gt_type="dense",
        expected_height=expected_height,
        expected_width=expected_width,
        validate_semantic_range=validate_semantic_range,
    )
    train_mapped = [mapper(record) for record in train_records]
    val_mapped = [mapper(record) for record in val_records]

    model = MemGraphDenseKnownNodes(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    train_target_summaries = [
        {"image_id": mapped.get("image_id"), **summarize_edge_targets(mapped["graph_gt"])} for mapped in train_mapped
    ]
    train_iterations: List[Dict[str, Any]] = []

    for iteration in range(int(max_iter)):
        mapped = train_mapped[iteration % len(train_mapped)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model([mapped])
        if "loss_mem_dense_graph" not in losses:
            raise ValueError("model did not return loss_mem_dense_graph")
        loss = losses["loss_mem_dense_graph"]
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError("non-finite loss_mem_dense_graph at iteration {}".format(iteration))
        loss.backward()
        grad_norm = _total_gradient_norm(model)
        optimizer.step()
        param_norm = _total_parameter_norm(model)
        eval_summary = _eval_one_record(model, mapped)
        train_iterations.append(
            {
                "iteration": int(iteration),
                "image_id": mapped.get("image_id"),
                "loss_mem_dense_graph": float(loss.detach().cpu().item()),
                "loss_mem_dense_graph_finite": bool(torch.isfinite(loss.detach()).all().item()),
                "grad_norm_total": grad_norm["value"],
                "grad_norm_total_finite": grad_norm["finite"],
                "grad_norm_saw_grad": grad_norm["saw_grad"],
                "parameter_norm_total": param_norm["value"],
                "parameter_norm_total_finite": param_norm["finite"],
                "post_step_logit_summary": eval_summary["logit_summary"],
                "post_step_metrics_at_threshold_0_5": eval_summary["metrics_at_threshold_0_5"],
            }
        )

    validation_summaries = [_eval_one_record(model, mapped) for mapped in val_mapped]

    summary = {
        "schema": SCHEMA,
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "records_json": str(records_json),
        "data_root": str(data_root),
        "config_file": str(_as_path(config_file)) if config_file is not None else None,
        "device": str(device),
        "seed": int(seed),
        "optimizer": "AdamW",
        "lr": float(lr),
        "max_iter": int(max_iter),
        "max_records": int(max_records),
        "num_records_loaded": len(records),
        "num_train_records": len(train_mapped),
        "num_val_records": len(val_mapped),
        "train_sample_ids": [str(record.get("sample_id")) for record in train_records],
        "val_sample_ids": [str(record.get("sample_id")) for record in val_records],
        "train_target_summaries": train_target_summaries,
        "train_iterations": train_iterations,
        "validation_summaries": validation_summaries,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "checkpoint_policy": "disabled/no_checkpoint_required",
        "safety": {
            "runs_model_forward": True,
            "runs_model_train_mode": True,
            "runs_backward": True,
            "runs_optimizer_step": True,
            "runs_training_loop": True,
            "max_records_cap": MAX_RECORDS,
            "max_iter_cap": MAX_ITER_CAP,
            "batch_size": 1,
            "device": str(device),
            "writes_hdf5_or_full_dataset": False,
            "writes_checkpoints_or_model_outputs": False,
            "writes_summary_json": output_dir is not None,
            "loads_model_weights": False,
        },
    }

    if output_dir is not None:
        _write_summary_json(output_dir, summary)
    return summary


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None or value == "":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True, help="Explicit MEM Option 2A records JSON/JSONL")
    parser.add_argument("--data-root", default=".", help="Root for relative sample_dir values")
    parser.add_argument("--train-sample-ids", required=True, help="Comma-separated train sample ids")
    parser.add_argument("--val-sample-ids", default="", help="Comma-separated validation sample ids")
    parser.add_argument("--max-records", type=int, default=1, help="Safety cap; 1..5")
    parser.add_argument("--device", default="cpu", help="CPU only in this first implementation")
    parser.add_argument("--max-iter", type=int, default=1, help="Tiny iteration cap; 1..200")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Torch manual seed")
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE, help="D3G MEM known-node config")
    parser.add_argument("--expected-height", type=int, default=140, help="Expected MEM tensor height")
    parser.add_argument("--expected-width", type=int, default=200, help="Expected MEM tensor width")
    parser.add_argument(
        "--no-validate-semantic-range",
        action="store_true",
        help="Disable semantic_hms integer/range validation for synthetic debugging records",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New output dir for small summary.json; must not exist")
    parser.add_argument("--no-checkpoint", action="store_true", help="Required: keep checkpoint/model-output writes disabled")
    parser.add_argument(
        "--i-understand-this-runs-backward",
        action="store_true",
        help="Required acknowledgement for this tiny debug wrapper",
    )
    parser.add_argument(
        "--i-understand-this-runs-optimizer-step",
        action="store_true",
        help="Required acknowledgement for this tiny debug wrapper",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional Detectron2-style CFG overrides after --",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    opts = list(args.opts or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    summary = build_mem_graph_dense_known_nodes_tiny_train_summary(
        records_json=args.records_json,
        data_root=args.data_root,
        train_sample_ids=_split_csv(args.train_sample_ids),
        val_sample_ids=_split_csv(args.val_sample_ids),
        max_records=args.max_records,
        device=args.device,
        config_file=args.config_file,
        expected_height=args.expected_height,
        expected_width=args.expected_width,
        validate_semantic_range=not args.no_validate_semantic_range,
        cfg_overrides=opts,
        seed=args.seed,
        max_iter=args.max_iter,
        lr=args.lr,
        output_dir=args.output_dir,
        no_checkpoint=args.no_checkpoint,
        acknowledge_backward=args.i_understand_this_runs_backward,
        acknowledge_optimizer_step=args.i_understand_this_runs_optimizer_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
