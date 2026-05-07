#!/usr/bin/env python3
"""Forward-only MEM known-node dense graph smoke for D3G.

This command loads a tiny explicit MEM Option 2A records JSON/JSONL file,
materializes mapper outputs, and runs the `MemGraphDenseKnownNodes` model in
CPU eval mode under `torch.no_grad()`.

Safety boundary: this is a forward-only diagnostic. It does not run backward,
optimizer steps, training loops, checkpoint/model-weight loading or writing,
HDF5 packing, or full dataset export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "mem" / "option2a_known_nodes_graph_smoke.yaml"


def _shape(tensor: Any) -> List[int]:
    return [int(dim) for dim in tensor.shape]


def _float_or_none(value: torch.Tensor) -> Optional[float]:
    if value.numel() == 0:
        return None
    return float(value.item())


def _tensor_min(tensor: torch.Tensor) -> Optional[float]:
    return _float_or_none(tensor.detach().min().cpu())


def _tensor_max(tensor: torch.Tensor) -> Optional[float]:
    return _float_or_none(tensor.detach().max().cpu())


def _build_cfg(config_file: Optional[Path], device: str, cfg_overrides: Optional[Sequence[str]]):
    if device != "cpu":
        raise ValueError("this forward-smoke wrapper is CPU-only unless a separate GPU run is explicitly implemented")
    cfg = get_cfg()
    add_dep_graph_config(cfg)
    add_detr_config(cfg)
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


def _summarize_forward_output(mapped: Dict[str, Any], model_output: Dict[str, Any]) -> Dict[str, Any]:
    image = mapped["image"]
    instances = mapped["instances"]
    graph_gt = mapped["graph_gt"]
    dense_gt = mapped["dense_gt"]
    graph_logits = model_output["graph_logits"].detach().cpu()
    graph_probs = model_output["graph_probs"].detach().cpu()
    metadata = mapped.get("mem_metadata", {}) or {}
    return {
        "image_id": mapped.get("image_id"),
        "height": int(mapped.get("height")),
        "width": int(mapped.get("width")),
        "image_shape_chw": _shape(image),
        "image_dtype": str(image.dtype).replace("torch.", ""),
        "image_finite": bool(torch.isfinite(image).all().item()),
        "image_min": _tensor_min(image),
        "image_max": _tensor_max(image),
        "num_nodes": int(len(instances)),
        "instances_gt_boxes_shape": _shape(instances.gt_boxes.tensor),
        "instances_gt_classes_shape": _shape(instances.gt_classes),
        "graph_gt_shape": _shape(graph_gt),
        "dense_gt_shape": _shape(dense_gt),
        "num_directed_gt_edges": int(graph_gt.sum().item()),
        "dense_gt_matches_graph_gt": bool(torch.equal(dense_gt, graph_gt)),
        "graph_logits_shape": _shape(graph_logits),
        "graph_probs_shape": _shape(graph_probs),
        "graph_logits_finite": bool(torch.isfinite(graph_logits).all().item()),
        "graph_probs_finite": bool(torch.isfinite(graph_probs).all().item()),
        "graph_logits_min": _tensor_min(graph_logits),
        "graph_logits_max": _tensor_max(graph_logits),
        "graph_probs_min": _tensor_min(graph_probs),
        "graph_probs_max": _tensor_max(graph_probs),
        "selected_view_indices": metadata.get("selected_view_indices"),
        "node_order_instance_ids": metadata.get("node_order_instance_ids"),
        "edge_type": metadata.get("edge_type"),
        "graph_gt_convention": metadata.get("graph_gt_convention"),
        "direct_mem_dense_graph_gt": bool(metadata.get("direct_mem_dense_graph_gt", False)),
    }


def build_mem_graph_dense_known_nodes_forward_smoke_summary(
    records_json: Path,
    data_root: str = ".",
    max_records: int = 1,
    device: str = "cpu",
    config_file: Optional[Path] = DEFAULT_CONFIG_FILE,
    expected_height: int = 140,
    expected_width: int = 200,
    validate_semantic_range: bool = True,
    cfg_overrides: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a capped CPU eval forward smoke and return compact diagnostics."""

    if max_records < 1 or max_records > MAX_RECORDS:
        raise ValueError("max_records must be between 1 and {}".format(MAX_RECORDS))
    records_json = Path(records_json).expanduser()
    config_file = Path(config_file).expanduser() if config_file is not None else None

    torch.manual_seed(int(seed))
    cfg = _build_cfg(config_file, device=device, cfg_overrides=cfg_overrides)
    records = load_mem_observed_gt_records(records_json, max_records=max_records)
    mapper = MemObservedGtMapper(
        data_root=data_root,
        is_train=False,
        graph_gt_type="dense",
        expected_height=expected_height,
        expected_width=expected_width,
        validate_semantic_range=validate_semantic_range,
    )
    model = MemGraphDenseKnownNodes(cfg)
    model.eval()

    validation_errors: List[str] = []
    record_summaries: List[Dict[str, Any]] = []
    with torch.no_grad():
        for record_index, record in enumerate(records):
            try:
                mapped = mapper(record)
                outputs = model([mapped])
                if len(outputs) != 1:
                    raise ValueError("expected one model output, got {}".format(len(outputs)))
                record_summaries.append(_summarize_forward_output(mapped, outputs[0]))
            except Exception as exc:  # pragma: no cover - used by bad smoke inputs
                validation_errors.append(
                    "record {} failed known-node forward smoke: {}: {}".format(
                        record_index, type(exc).__name__, exc
                    )
                )

    return {
        "schema": "mem_d3g_known_node_forward_smoke_cli_summary_v0",
        "records_json": str(records_json),
        "data_root": str(data_root),
        "config_file": str(config_file) if config_file is not None else None,
        "device": device,
        "model_architecture": "MemGraphDenseKnownNodes",
        "model_mode": "eval",
        "gradient_enabled": False,
        "seed": int(seed),
        "max_records": int(max_records),
        "num_records_loaded": len(records),
        "num_records_mapped": len(record_summaries),
        "num_records_forwarded": len(record_summaries),
        "record_summaries": record_summaries,
        "total_nodes": sum(int(item["num_nodes"]) for item in record_summaries),
        "total_edges": sum(int(item["num_directed_gt_edges"]) for item in record_summaries),
        "validation_errors": validation_errors,
        "safety": {
            "runs_model_forward": True,
            "forward_only": True,
            "model_mode": "eval",
            "device": device,
            "runs_backward": False,
            "runs_optimizer_step": False,
            "runs_training_loop": False,
            "writes_hdf5_or_full_dataset": False,
            "writes_checkpoints_or_training_outputs": False,
            "loads_or_writes_model_weights": False,
            "max_records_cap": MAX_RECORDS,
        },
    }


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True, help="JSON list/object or JSONL MEM Option 2A records")
    parser.add_argument("--data-root", default=".", help="Root for relative sample_dir values; absolute sample_dir values ignore it")
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE, help="D3G MEM known-node graph smoke config")
    parser.add_argument("--max-records", type=int, default=1, help="Safety cap; 1..5")
    parser.add_argument("--device", choices=["cpu"], default="cpu", help="CPU only for this smoke wrapper")
    parser.add_argument("--expected-height", type=int, default=140, help="Expected materialized map height; use 0 to disable")
    parser.add_argument("--expected-width", type=int, default=200, help="Expected materialized map width; use 0 to disable")
    parser.add_argument(
        "--no-validate-semantic-range",
        action="store_true",
        help="Disable semantic_hms integer/range validation for synthetic debugging records",
    )
    parser.add_argument("--seed", type=int, default=0, help="Manual torch seed for deterministic random scaffold weights")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional small diagnostics JSON path; refuses to overwrite")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional Detectron2-style CFG overrides after --, e.g. -- MODEL.MEM_GRAPH.HIDDEN_DIM 16",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    opts = list(args.opts or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    summary = build_mem_graph_dense_known_nodes_forward_smoke_summary(
        records_json=args.records_json,
        data_root=args.data_root,
        max_records=args.max_records,
        device=args.device,
        config_file=args.config_file,
        expected_height=args.expected_height,
        expected_width=args.expected_width,
        validate_semantic_range=not args.no_validate_semantic_range,
        cfg_overrides=opts,
        seed=args.seed,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output_path = args.output_json.expanduser()
        if output_path.exists():
            raise FileExistsError("refusing to overwrite existing diagnostics JSON: {}".format(output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return 0 if not summary.get("validation_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
