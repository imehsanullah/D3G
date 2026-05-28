#!/usr/bin/env python3
"""Config-driven MEM dense graph trainer/evaluator for D3G Stage 6.

This is the proper MEM training/evaluation entrypoint for thesis Stage 6 runs.
It uses explicit records/split manifests, the existing MemObservedGtMapper, and
MemGraphDenseKnownNodes with GraphTransformerDense.  The old tiny debug wrapper
remains separate and unchanged.

Safety defaults:
- output_dir must not already exist;
- checkpoints/model weights are not written unless --enable-checkpoints is used;
- MODEL.WEIGHTS is cleared by default, so this path does not load checkpoints;
- data is loaded only from explicit records/split manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.mem_observed_gt_dataset import load_mem_observed_gt_records
from data.mem_observed_gt_mapper import MemObservedGtMapper
from models.mem_graph_dense import MemGraphDenseKnownNodes, node_binary_bce_loss
from utils.configs import add_dep_graph_config, add_detr_config


SCHEMA = "mem_d3g_train_eval_summary_v0"
PREDICTION_DUMP_SCHEMA = "mem_d3g_prediction_dump_v0"
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "mem" / "option2a_gt_known_nodes.yaml"
LOSS_MODES = ("bce", "fixed_pos_weighted_bce", "train_pos_weighted_bce")
DEFAULT_LOSS_MODE = "train_pos_weighted_bce"
DEFAULT_THRESHOLDS = (0.03, 0.05, 0.07, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50)
DEFAULT_PREDICTION_DUMP_THRESHOLD = 0.5
DEFAULT_PREDICTION_DUMP_TOP_K = 20
SPLIT_NAMES = ("train", "val", "test")


PathLike = Any
Record = Dict[str, Any]


def _as_path(path: Optional[PathLike]) -> Optional[Path]:
    if path is None or str(path) == "":
        return None
    return Path(path).expanduser()


def _normalise_training_device(device: str) -> str:
    value = str(device or "cpu").strip().lower()
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False")
        index_text = value.split(":", 1)[1]
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError("CUDA device index must be an integer; got {!r}".format(index_text)) from exc
        device_count = int(torch.cuda.device_count())
        if index < 0 or index >= device_count:
            raise ValueError("CUDA device index {} unavailable; torch reports {} CUDA device(s)".format(index, device_count))
        return "cuda:{}".format(index)
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'; got {!r}".format(device))


def _cuda_device_summary(device: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
    }
    if str(device).startswith("cuda:"):
        index = int(str(device).split(":", 1)[1])
        summary["cuda_device_index"] = index
        summary["cuda_device_name"] = torch.cuda.get_device_name(index)
    return summary


def setup_mem_graph_cfg(
    config_file: Optional[PathLike] = DEFAULT_CONFIG_FILE,
    *,
    device: str = "cpu",
    cfg_overrides: Optional[Sequence[str]] = None,
    clear_model_weights: bool = True,
):
    """Build a D3G MEM config for MemGraphDenseKnownNodes runs."""

    device = _normalise_training_device(device)
    cfg = get_cfg()
    add_dep_graph_config(cfg)
    add_detr_config(cfg)
    config_path = _as_path(config_file)
    if config_path is not None:
        cfg.merge_from_file(str(config_path))
    if cfg_overrides:
        cfg.merge_from_list(list(cfg_overrides))
    cfg.MODEL.DEVICE = device
    if clear_model_weights:
        cfg.MODEL.WEIGHTS = ""
    if cfg.MODEL.META_ARCHITECTURE != "MemGraphDenseKnownNodes":
        raise ValueError("expected MODEL.META_ARCHITECTURE='MemGraphDenseKnownNodes'")
    if cfg.MODEL.GRAPH_HEAD.NAME != "GraphTransformerDense":
        raise ValueError("expected MODEL.GRAPH_HEAD.NAME='GraphTransformerDense'")
    if int(cfg.SOLVER.IMS_PER_BATCH) != 1:
        raise ValueError("MemGraphDenseKnownNodes v0 requires SOLVER.IMS_PER_BATCH=1 until node padding is added")
    return cfg


def _read_json(path: PathLike) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _coerce_str_list(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def load_mem_graph_split_manifest(split_json: PathLike) -> Dict[str, Any]:
    """Load and normalize a train/val/test split manifest.

    Supported schemas include the existing Stage 5B manifest with top-level
    train_sample_ids / val_sample_ids and a generic nested {"splits": {...}}
    form. Missing split ids normalize to empty lists.
    """

    path = Path(split_json).expanduser()
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("split manifest must be a JSON object")
    nested_splits = raw.get("splits") if isinstance(raw.get("splits"), dict) else {}
    manifest = dict(raw)
    manifest["split_json"] = str(path)
    for split_name in SPLIT_NAMES:
        top_level_name = "{}_sample_ids".format(split_name)
        ids = raw.get(top_level_name, nested_splits.get(split_name, []))
        manifest[top_level_name] = _coerce_str_list(ids)
    return manifest


def _record_scene(record: Mapping[str, Any]) -> str:
    scene = record.get("scene")
    if scene is not None and str(scene) != "":
        return str(scene)
    sample_id = str(record.get("sample_id") or record.get("id") or record.get("image_id") or "")
    if "/" in sample_id:
        return sample_id.split("/", 1)[0]
    return sample_id


def _select_records_by_ids(records: Sequence[Record], sample_ids: Sequence[str], *, split_name: str) -> List[Record]:
    by_id: Dict[str, Record] = {}
    duplicates: List[str] = []
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_id in by_id:
            duplicates.append(sample_id)
        by_id[sample_id] = record
    if duplicates:
        raise ValueError("records_json contains duplicate sample ids: {}".format(sorted(set(duplicates))))
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError("{} sample ids not found in records_json: {}".format(split_name, missing))
    return [dict(by_id[sample_id]) for sample_id in sample_ids]


def _check_scene_disjoint(records_by_split: Mapping[str, Sequence[Record]]) -> Dict[str, Any]:
    scenes_by_split = {
        split_name: sorted({_record_scene(record) for record in records})
        for split_name, records in records_by_split.items()
    }
    overlaps: Dict[str, List[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(scenes_by_split.get(left, [])) & set(scenes_by_split.get(right, [])))
        if overlap:
            overlaps["{}_{}".format(left, right)] = overlap
    if overlaps:
        raise ValueError("split manifest is not scene-disjoint; scene overlaps: {}".format(overlaps))
    return {"scenes_by_split": scenes_by_split, "scene_overlaps": overlaps}


def load_mem_graph_records_for_splits(
    records_json: PathLike,
    split_manifest: Mapping[str, Any],
    *,
    require_scene_disjoint: bool = True,
    max_records_per_split: Optional[int] = None,
) -> Dict[str, List[Record]]:
    """Load explicit MEM records and select split-specific record lists."""

    records = load_mem_observed_gt_records(records_json)
    records_by_split: Dict[str, List[Record]] = {}
    for split_name in SPLIT_NAMES:
        ids = _coerce_str_list(split_manifest.get("{}_sample_ids".format(split_name), []))
        selected = _select_records_by_ids(records, ids, split_name=split_name) if ids else []
        if max_records_per_split is not None:
            selected = selected[: int(max_records_per_split)]
        records_by_split[split_name] = selected
    if not records_by_split["train"]:
        raise ValueError("split manifest must provide at least one train sample id")
    if require_scene_disjoint:
        _check_scene_disjoint(records_by_split)
    return records_by_split


def register_mem_graph_datasets(
    cfg,
    records_by_split: Mapping[str, Sequence[Record]],
    *,
    dataset_prefix: Optional[str] = None,
) -> Dict[str, str]:
    """Register explicit MEM split records with Detectron2 DatasetCatalog."""

    base_names = list(getattr(cfg.DATASETS, "TRAIN", [])) or ["mem_graph_train_eval"]
    base_name = str(base_names[0])
    prefix = str(dataset_prefix or base_name)
    if not prefix.startswith("mem_"):
        prefix = "mem_" + prefix
    registered: Dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        name = "{}_{}".format(prefix, split_name)
        if name in DatasetCatalog.list():
            raise ValueError("dataset name already registered; refusing to overwrite: {}".format(name))
        split_records = [dict(record) for record in records_by_split.get(split_name, [])]
        DatasetCatalog.register(name, func=lambda records=split_records: [dict(record) for record in records])
        MetadataCatalog.get(name).set(
            evaluator_type="mem_observed_gt_graph",
            graph_gt_type="dense",
            edge_type="blocks_access_to",
            mem_node_source=str(cfg.INPUT.MEM_NODE_SOURCE),
            mem_graph_target_scope=str(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE),
            split_name=split_name,
        )
        registered[split_name] = name
    return registered


def unregister_mem_graph_datasets(dataset_names: Mapping[str, str]) -> None:
    for name in dataset_names.values():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)


def _build_mapper(cfg, *, data_root: str, is_train: bool) -> MemObservedGtMapper:
    return MemObservedGtMapper(
        data_root=data_root,
        is_train=is_train,
        graph_gt_type=cfg.INPUT.GRAPH_GT_TYPE,
        observed_view_protocol=cfg.INPUT.MEM_OBSERVED_VIEW_PROTOCOL,
        expected_height=int(cfg.INPUT.MEM_EXPECTED_HEIGHT),
        expected_width=int(cfg.INPUT.MEM_EXPECTED_WIDTH),
        max_selected_views=int(cfg.INPUT.MEM_MAX_SELECTED_VIEWS),
        semantic_class_min=int(cfg.INPUT.MEM_SEMANTIC_CLASS_MIN),
        semantic_class_max=int(cfg.INPUT.MEM_SEMANTIC_CLASS_MAX),
        validate_semantic_range=bool(cfg.INPUT.MEM_VALIDATE_SEMANTIC_RANGE),
        mem_node_source=cfg.INPUT.MEM_NODE_SOURCE,
        mem_graph_target_scope=cfg.INPUT.MEM_GRAPH_TARGET_SCOPE,
        mem_box_mode=cfg.INPUT.MEM_BOX_MODE,
        mem_map_feature_source=cfg.INPUT.MEM_MAP_FEATURE_SOURCE,
        cnabu_derived_root=cfg.DATASETS.MEM_CNABU_DERIVED_ROOT,
        cnabu_pad_mode=cfg.INPUT.MEM_CNABU_PAD_MODE,
    )


def map_mem_graph_records(cfg, records_by_split: Mapping[str, Sequence[Record]], *, data_root: str) -> Dict[str, List[Dict[str, Any]]]:
    mapped_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in SPLIT_NAMES:
        mapper = _build_mapper(cfg, data_root=data_root, is_train=(split_name == "train"))
        mapped_by_split[split_name] = [mapper(record) for record in records_by_split.get(split_name, [])]
    return mapped_by_split


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype).cpu()
    return torch.tensor(value, dtype=dtype)


def compute_non_diagonal_mask(target: Any) -> torch.Tensor:
    target_tensor = _as_tensor(target)
    if target_tensor.dim() != 2 or target_tensor.shape[0] != target_tensor.shape[1]:
        raise ValueError("target graph must be a square [N,N] matrix")
    num_nodes = int(target_tensor.shape[0])
    return ~torch.eye(num_nodes, dtype=torch.bool)


def summarize_edge_targets(target: Any) -> Dict[str, Any]:
    target_tensor = _as_tensor(target)
    mask = compute_non_diagonal_mask(target_tensor)
    selected = target_tensor[mask]
    num_pairs = int(selected.numel())
    num_positive = int(selected.sum().item())
    num_negative = int(num_pairs - num_positive)
    return {
        "num_nodes": int(target_tensor.shape[0]),
        "num_non_diagonal_pairs": num_pairs,
        "num_positive_directed_edges": num_positive,
        "num_negative_directed_edges": num_negative,
        "positive_edge_base_rate": float(num_positive / num_pairs) if num_pairs else None,
    }


def aggregate_target_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    num_records = len(summaries)
    num_nodes = int(sum(int(summary.get("num_nodes", 0)) for summary in summaries))
    num_pairs = int(sum(int(summary.get("num_non_diagonal_pairs", 0)) for summary in summaries))
    num_positive = int(sum(int(summary.get("num_positive_directed_edges", 0)) for summary in summaries))
    num_negative = int(num_pairs - num_positive)
    return {
        "num_records": int(num_records),
        "num_nodes": num_nodes,
        "num_non_diagonal_pairs": num_pairs,
        "num_positive_directed_edges": num_positive,
        "num_negative_directed_edges": num_negative,
        "positive_edge_base_rate": float(num_positive / num_pairs) if num_pairs else None,
        "negative_positive_ratio": float(num_negative / num_positive) if num_positive else None,
    }


def _normalise_thresholds(thresholds: Optional[Sequence[float]]) -> List[float]:
    values = [float(value) for value in (thresholds or DEFAULT_THRESHOLDS)]
    if not values:
        raise ValueError("at least one threshold is required")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError("thresholds must be in [0,1]")
    if 0.5 not in values:
        values.append(0.5)
    return values


def _normalise_score_histogram_bin_edges(bin_edges: Optional[Sequence[float]]) -> Optional[List[float]]:
    if bin_edges is None:
        return None
    values = [float(value) for value in bin_edges]
    if len(values) < 2:
        raise ValueError("score histogram bin edges must contain at least two values")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("score histogram bin edges must be finite")
    if values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("score histogram bin edges must stay within [0,1]")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("score histogram bin edges must be strictly increasing")
    return values


def compute_average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    score_list = [float(score) for score in scores]
    label_list = [int(label) for label in labels]
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels must have matching lengths")
    if any(label not in (0, 1) for label in label_list):
        raise ValueError("labels must be binary 0/1")
    num_positive = sum(label_list)
    if num_positive == 0:
        return None
    sorted_indices = sorted(
        range(len(score_list)),
        key=lambda index: score_list[index],
        reverse=True,
    )
    true_positive_count = 0
    false_positive_count = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(sorted_indices):
        tied_score = score_list[sorted_indices[index]]
        tied_true_positive_count = 0
        tied_false_positive_count = 0
        while (
            index < len(sorted_indices)
            and score_list[sorted_indices[index]] == tied_score
        ):
            label = label_list[sorted_indices[index]]
            if label:
                tied_true_positive_count += 1
            else:
                tied_false_positive_count += 1
            index += 1
        true_positive_count += tied_true_positive_count
        false_positive_count += tied_false_positive_count
        if tied_true_positive_count:
            recall = true_positive_count / float(num_positive)
            precision = true_positive_count / float(
                true_positive_count + false_positive_count
            )
            average_precision += (recall - previous_recall) * precision
            previous_recall = recall
    return average_precision


def compute_binary_metrics_from_scores(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, Any]:
    score_list = [float(score) for score in scores]
    label_list = [int(label) for label in labels]
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels must have matching lengths")
    if any(label not in (0, 1) for label in label_list):
        raise ValueError("labels must be binary 0/1")
    threshold = float(threshold)
    predictions = [1 if score >= threshold else 0 for score in score_list]
    tp = sum(1 for pred, label in zip(predictions, label_list) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, label_list) if pred == 1 and label == 0)
    tn = sum(1 for pred, label in zip(predictions, label_list) if pred == 0 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, label_list) if pred == 0 and label == 1)
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "num_pairs": int(len(label_list)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_score_histogram(
    scores: Sequence[float],
    labels: Sequence[int],
    bin_edges: Sequence[float],
) -> Dict[str, Any]:
    score_list = [float(score) for score in scores]
    label_list = [int(label) for label in labels]
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels must have matching lengths")
    if any(label not in (0, 1) for label in label_list):
        raise ValueError("labels must be binary 0/1")
    edges = _normalise_score_histogram_bin_edges(bin_edges)
    if edges is None:
        raise ValueError("score histogram bin edges are required")
    positive_counts = [0 for _ in range(len(edges) - 1)]
    negative_counts = [0 for _ in range(len(edges) - 1)]
    for score, label in zip(score_list, label_list):
        if score < edges[0] or score > edges[-1]:
            raise ValueError("score {} falls outside histogram range [{},{}]".format(score, edges[0], edges[-1]))
        bin_index = len(edges) - 2
        for index, (left, right) in enumerate(zip(edges, edges[1:])):
            if left <= score < right or (index == len(edges) - 2 and score <= right):
                bin_index = index
                break
        if label == 1:
            positive_counts[bin_index] += 1
        else:
            negative_counts[bin_index] += 1
    return {
        "schema": "mem_d3g_score_histogram_v0",
        "bin_edges": edges,
        "positive_counts": [int(value) for value in positive_counts],
        "negative_counts": [int(value) for value in negative_counts],
        "total_counts": [int(pos + neg) for pos, neg in zip(positive_counts, negative_counts)],
        "positive_total": int(sum(positive_counts)),
        "negative_total": int(sum(negative_counts)),
    }


def compute_mem_graph_score_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    thresholds: Optional[Sequence[float]] = None,
    score_histogram_bin_edges: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute consistent MEM graph AP/F1/probability metrics from pair scores."""

    score_list = [float(score) for score in scores]
    label_list = [int(label) for label in labels]
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels must have matching lengths")
    if any(label not in (0, 1) for label in label_list):
        raise ValueError("labels must be binary 0/1")
    thresholds = _normalise_thresholds(thresholds)
    positive_scores = [score for score, label in zip(score_list, label_list) if label == 1]
    negative_scores = [score for score, label in zip(score_list, label_list) if label == 0]
    num_pairs = len(label_list)
    num_positive = len(positive_scores)
    num_negative = len(negative_scores)
    base_rate = num_positive / float(num_pairs) if num_pairs else None
    ap = compute_average_precision(score_list, label_list)
    threshold_sweep = [compute_binary_metrics_from_scores(score_list, label_list, threshold) for threshold in thresholds]
    best = max(
        threshold_sweep,
        key=lambda item: (item["f1"], item["recall"], item["precision"], -item["threshold"]),
    ) if threshold_sweep else None
    metrics_at_05 = compute_binary_metrics_from_scores(score_list, label_list, 0.5)
    mean_positive = sum(positive_scores) / num_positive if num_positive else None
    mean_negative = sum(negative_scores) / num_negative if num_negative else None
    histogram_edges = _normalise_score_histogram_bin_edges(score_histogram_bin_edges)
    result = {
        "num_pairs": int(num_pairs),
        "num_positive": int(num_positive),
        "num_negative": int(num_negative),
        "positive_edge_base_rate": base_rate,
        "validation_ap": ap,
        "ap_over_base_rate": (ap / base_rate) if ap is not None and base_rate else None,
        "metrics_at_threshold_0_5": metrics_at_05,
        "precision_at_threshold_0_5": metrics_at_05["precision"],
        "recall_at_threshold_0_5": metrics_at_05["recall"],
        "f1_at_threshold_0_5": metrics_at_05["f1"],
        "threshold_sweep": threshold_sweep,
        "best_threshold": best["threshold"] if best else None,
        "best_f1": best["f1"] if best else None,
        "best_threshold_metrics": best,
        "mean_positive_probability": mean_positive,
        "mean_negative_probability": mean_negative,
        "positive_negative_probability_gap": (mean_positive - mean_negative)
        if mean_positive is not None and mean_negative is not None
        else None,
    }
    if histogram_edges is not None:
        result["score_histogram"] = compute_score_histogram(score_list, label_list, histogram_edges)
    return result


def _extract_scores_labels_from_logits(logits: Any, target: Any) -> Dict[str, List[Any]]:
    logits_tensor = _as_tensor(logits)
    target_tensor = _as_tensor(target)
    if logits_tensor.shape != target_tensor.shape:
        raise ValueError("logits and target must have matching shapes")
    mask = compute_non_diagonal_mask(target_tensor)
    scores = torch.sigmoid(logits_tensor[mask]).detach().cpu().tolist()
    labels = target_tensor[mask].long().detach().cpu().tolist()
    return {"scores": [float(score) for score in scores], "labels": [int(label) for label in labels]}


def _extract_node_scores_labels_from_logits(logits: Any, target: Any) -> Dict[str, List[Any]]:
    logits_tensor = _as_tensor(logits)
    target_tensor = _as_tensor(target)
    if logits_tensor.dim() != 1 or target_tensor.dim() != 1:
        raise ValueError("node logits and target must have shape [N]")
    if logits_tensor.shape != target_tensor.shape:
        raise ValueError("node logits and target must have matching shapes")
    scores = torch.sigmoid(logits_tensor).detach().cpu().tolist()
    labels = target_tensor.long().detach().cpu().tolist()
    return {"scores": [float(score) for score in scores], "labels": [int(label) for label in labels]}


def compute_node_binary_score_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    loss_value: Optional[float] = None,
    scaled_loss_value: Optional[float] = None,
) -> Dict[str, Any]:
    score_list = [float(score) for score in scores]
    label_list = [int(label) for label in labels]
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels must have matching lengths")
    if any(label not in (0, 1) for label in label_list):
        raise ValueError("labels must be binary 0/1")
    num_nodes = len(label_list)
    num_positive = sum(label_list)
    num_negative = num_nodes - num_positive
    at_05 = compute_binary_metrics_from_scores(score_list, label_list, 0.5)
    ap = compute_average_precision(score_list, label_list)
    base_rate = float(num_positive / num_nodes) if num_nodes else None
    return {
        "num_nodes": int(num_nodes),
        "num_positive": int(num_positive),
        "num_negative": int(num_negative),
        "positive_rate": base_rate,
        "base_rate": base_rate,
        "ap": ap,
        "average_precision": ap,
        "ap_over_base_rate": (ap / base_rate) if ap is not None and base_rate else None,
        "metrics_at_threshold_0_5": at_05,
        "precision_at_threshold_0_5": at_05["precision"],
        "recall_at_threshold_0_5": at_05["recall"],
        "f1_at_threshold_0_5": at_05["f1"],
        "bce_loss": loss_value,
        "scaled_bce_loss": scaled_loss_value,
    }


def _compute_visible_blocks_hidden_aux_metrics(
    logits: Any,
    target: Any,
    *,
    loss_weight: float,
    pos_weight: float,
) -> Dict[str, Any]:
    logits_tensor = _as_tensor(logits)
    target_tensor = _as_tensor(target)
    selected = _extract_node_scores_labels_from_logits(logits_tensor, target_tensor)
    loss = node_binary_bce_loss(logits_tensor, target_tensor, pos_weight=pos_weight)
    loss_value = float(loss.detach().cpu().item())
    metrics = compute_node_binary_score_metrics(
        selected["scores"],
        selected["labels"],
        loss_value=loss_value,
        scaled_loss_value=float(loss_weight) * loss_value,
    )
    metrics["pos_weight"] = float(pos_weight)
    return metrics


def _empty_visible_blocks_hidden_aux_metrics(
    *,
    enabled: bool,
    loss_weight: float,
    pos_weight: float,
) -> Dict[str, Any]:
    metrics = compute_node_binary_score_metrics([], [], loss_value=None, scaled_loss_value=None)
    metrics["pos_weight"] = float(pos_weight)
    return {
        "enabled": bool(enabled),
        "loss_weight": float(loss_weight),
        "pos_weight": float(pos_weight),
        "available_records": 0,
        **metrics,
    }


def _aggregate_visible_blocks_hidden_aux_metrics(
    records: Sequence[Mapping[str, Dict[str, List[Any]]]],
    *,
    enabled: bool,
    loss_weight: float,
    pos_weight: float,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "loss_weight": float(loss_weight),
            "pos_weight": float(pos_weight),
            "available_records": 0,
        }
    if not records:
        return _empty_visible_blocks_hidden_aux_metrics(
            enabled=True,
            loss_weight=loss_weight,
            pos_weight=pos_weight,
        )

    logits: List[float] = []
    labels: List[int] = []
    for record in records:
        logits.extend(float(value) for value in record.get("logits", []))
        labels.extend(int(value) for value in record.get("labels", []))
    if not logits:
        return _empty_visible_blocks_hidden_aux_metrics(
            enabled=True,
            loss_weight=loss_weight,
            pos_weight=pos_weight,
        )
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    loss = node_binary_bce_loss(logits_tensor, labels_tensor, pos_weight=pos_weight)
    loss_value = float(loss.detach().cpu().item())
    scores = torch.sigmoid(logits_tensor).tolist()
    result = compute_node_binary_score_metrics(
        scores,
        labels,
        loss_value=loss_value,
        scaled_loss_value=float(loss_weight) * loss_value,
    )
    result["enabled"] = True
    result["loss_weight"] = float(loss_weight)
    result["pos_weight"] = float(pos_weight)
    result["available_records"] = int(len(records))
    return result


def _eval_output_and_target(model: MemGraphDenseKnownNodes, mapped: Mapping[str, Any]) -> Tuple[Mapping[str, Any], torch.Tensor]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model([dict(mapped)])
    if was_training:
        model.train()
    if len(outputs) != 1:
        raise ValueError("expected one eval output, got {}".format(len(outputs)))
    target = mapped["graph_gt"].detach().cpu()
    return outputs[0], target


def _eval_logits_and_target(model: MemGraphDenseKnownNodes, mapped: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    output, target = _eval_output_and_target(model, mapped)
    logits = output["graph_logits"].detach().cpu()
    return logits, target


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _node_fields_from_mapped(mapped: Mapping[str, Any], *, num_nodes: int) -> Dict[str, Any]:
    metadata = dict(mapped.get("mem_metadata", {}) or {})
    node_ids = _coerce_str_list(metadata.get("node_order_instance_ids", []))
    if not node_ids:
        node_ids = [str(index) for index in range(int(num_nodes))]
    node_ids_json: List[Any] = []
    for node_id in node_ids:
        try:
            node_ids_json.append(int(node_id))
        except ValueError:
            node_ids_json.append(node_id)

    instances = mapped.get("instances")
    if instances is None:
        raise ValueError("mapped item must include Detectron2 Instances under key 'instances'")
    if not hasattr(instances, "gt_boxes") or not hasattr(instances.gt_boxes, "tensor"):
        raise ValueError("mapped instances must include gt_boxes")
    if not hasattr(instances, "gt_classes"):
        raise ValueError("mapped instances must include gt_classes")
    boxes_tensor = instances.gt_boxes.tensor.detach().cpu().to(dtype=torch.float32)
    classes_tensor = instances.gt_classes.detach().cpu().to(dtype=torch.long)
    if boxes_tensor.shape != (int(num_nodes), 4):
        raise ValueError("node boxes shape {} does not match num_nodes {}".format(tuple(boxes_tensor.shape), num_nodes))
    if int(classes_tensor.numel()) != int(num_nodes):
        raise ValueError("node class count {} does not match num_nodes {}".format(int(classes_tensor.numel()), num_nodes))

    return {
        "node_ids": node_ids_json,
        "node_boxes_xyxy_abs": [[float(value) for value in row] for row in boxes_tensor.tolist()],
        "node_classes": [int(value) for value in classes_tensor.tolist()],
    }


def _option2b_metadata_from_mapped(mapped: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(mapped.get("mem_metadata", {}) or {})
    keys = (
        "observed_visible_instance_ids",
        "gt_aligned_instance_ids",
        "hidden_gt_instance_ids",
        "unmatched_observed_instance_ids",
        "visible_gt_instance_ids",
        "dropped_visible_gt_instance_ids",
        "invalid_class_observed_instance_ids",
        "observed_induced_source_gt_indices",
        "observed_node_records",
        "visible_blocks_hidden_target",
        "num_visible_blocks_hidden_positive",
        "num_observed_visible_instances",
        "num_gt_aligned_instances",
        "num_hidden_gt_instances",
        "num_unmatched_observed_instances",
        "target_scope_note",
    )
    return {key: _to_jsonable(metadata.get(key, [] if key.endswith("ids") or key.endswith("records") else None)) for key in keys if key in metadata}


def _edge_dump_record(
    *,
    source_index: int,
    target_index: int,
    probability_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    predicted_matrix: torch.Tensor,
    node_ids: Sequence[Any],
    node_classes: Sequence[int],
    node_boxes: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    reverse_probability = float(probability_matrix[target_index, source_index].item())
    reverse_gt_label = int(target_matrix[target_index, source_index].item())
    reverse_predicted_label = int(predicted_matrix[target_index, source_index].item())
    return {
        "source_index": int(source_index),
        "target_index": int(target_index),
        "source_node_id": node_ids[source_index],
        "target_node_id": node_ids[target_index],
        "source_class": int(node_classes[source_index]),
        "target_class": int(node_classes[target_index]),
        "source_box_xyxy_abs": [float(value) for value in node_boxes[source_index]],
        "target_box_xyxy_abs": [float(value) for value in node_boxes[target_index]],
        "probability": float(probability_matrix[source_index, target_index].item()),
        "gt_label": int(target_matrix[source_index, target_index].item()),
        "predicted_label": int(predicted_matrix[source_index, target_index].item()),
        "reverse_probability": reverse_probability,
        "reverse_gt_label": reverse_gt_label,
        "reverse_predicted_label": reverse_predicted_label,
        "reverse_direction_gt_positive": bool(reverse_gt_label == 1),
        "reverse_direction_predicted_positive": bool(reverse_predicted_label == 1),
    }


def build_mem_prediction_dump(
    mapped: Mapping[str, Any],
    logits: Any,
    target: Any,
    *,
    split_name: str,
    threshold: float = DEFAULT_PREDICTION_DUMP_THRESHOLD,
    top_k: int = DEFAULT_PREDICTION_DUMP_TOP_K,
) -> Dict[str, Any]:
    """Build a selected-sample pair-level prediction dump for MEM graph diagnostics."""

    logits_tensor = _as_tensor(logits, dtype=torch.float32)
    target_tensor = _as_tensor(target, dtype=torch.float32).long()
    if logits_tensor.shape != target_tensor.shape:
        raise ValueError("prediction dump logits and target must have matching shapes")
    if target_tensor.dim() != 2 or target_tensor.shape[0] != target_tensor.shape[1]:
        raise ValueError("prediction dump target must be a square [N,N] matrix")
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("prediction dump threshold must be in [0,1]")
    top_k = int(top_k)
    if top_k < 0:
        raise ValueError("prediction dump top_k must be non-negative")

    probability_matrix = torch.sigmoid(logits_tensor).detach().cpu()
    target_matrix = target_tensor.detach().cpu().long()
    predicted_matrix = (probability_matrix >= threshold).long()
    num_nodes = int(target_matrix.shape[0])
    node_fields = _node_fields_from_mapped(mapped, num_nodes=num_nodes)
    node_ids = node_fields["node_ids"]
    node_boxes = node_fields["node_boxes_xyxy_abs"]
    node_classes = node_fields["node_classes"]
    nodes = [
        {
            "index": int(index),
            "node_id": node_ids[index],
            "class": int(node_classes[index]),
            "box_xyxy_abs": [float(value) for value in node_boxes[index]],
        }
        for index in range(num_nodes)
    ]

    tp = fp = tn = fn = 0
    false_positive_edges: List[Dict[str, Any]] = []
    false_negative_edges: List[Dict[str, Any]] = []
    for source_index in range(num_nodes):
        for target_index in range(num_nodes):
            if source_index == target_index:
                continue
            label = int(target_matrix[source_index, target_index].item())
            pred = int(predicted_matrix[source_index, target_index].item())
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
                false_positive_edges.append(
                    _edge_dump_record(
                        source_index=source_index,
                        target_index=target_index,
                        probability_matrix=probability_matrix,
                        target_matrix=target_matrix,
                        predicted_matrix=predicted_matrix,
                        node_ids=node_ids,
                        node_classes=node_classes,
                        node_boxes=node_boxes,
                    )
                )
            elif pred == 0 and label == 0:
                tn += 1
            elif pred == 0 and label == 1:
                fn += 1
                false_negative_edges.append(
                    _edge_dump_record(
                        source_index=source_index,
                        target_index=target_index,
                        probability_matrix=probability_matrix,
                        target_matrix=target_matrix,
                        predicted_matrix=predicted_matrix,
                        node_ids=node_ids,
                        node_classes=node_classes,
                        node_boxes=node_boxes,
                    )
                )

    false_positive_edges.sort(key=lambda item: item["probability"], reverse=True)
    false_negative_edges.sort(key=lambda item: item["probability"])
    selected_fp = false_positive_edges[:top_k] if top_k else []
    selected_fn = false_negative_edges[:top_k] if top_k else []
    metadata = dict(mapped.get("mem_metadata", {}) or {})
    return {
        "schema": PREDICTION_DUMP_SCHEMA,
        "sample_id": str(mapped.get("image_id") or metadata.get("sample_id") or ""),
        "split": str(split_name),
        "threshold": threshold,
        "top_k": top_k,
        "num_nodes": int(num_nodes),
        "nodes": nodes,
        "node_ids": node_ids,
        "node_boxes_xyxy_abs": node_boxes,
        "node_classes": node_classes,
        "gt_adjacency_matrix": [[int(value) for value in row] for row in target_matrix.tolist()],
        "predicted_probability_matrix": [[float(value) for value in row] for row in probability_matrix.tolist()],
        "predicted_adjacency_matrix_at_threshold": [[int(value) for value in row] for row in predicted_matrix.tolist()],
        "edge_confusion_at_threshold": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "top_false_positive_directed_edges": selected_fp,
        "top_false_negative_directed_edges": selected_fn,
        "all_false_positive_count": int(len(false_positive_edges)),
        "all_false_negative_count": int(len(false_negative_edges)),
        "option2b_metadata": _option2b_metadata_from_mapped(mapped),
        "mem_metadata_summary": {
            "mem_node_source": metadata.get("mem_node_source"),
            "mem_graph_target_scope": metadata.get("mem_graph_target_scope"),
            "selected_view_indices": _to_jsonable(metadata.get("selected_view_indices", [])),
            "sample_dir": metadata.get("sample_dir"),
            "graph_gt_convention": metadata.get("graph_gt_convention"),
        },
    }


def _safe_prediction_dump_filename(split_name: str, sample_id: str) -> str:
    safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(sample_id).strip())
    safe_split = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(split_name).strip())
    return "{}__{}.json".format(safe_split or "split", safe_sample or "sample")


def _normalise_prediction_dump_sample_ids(values: Optional[Iterable[Any]]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values or []:
        sample_id = str(value).strip()
        if sample_id and sample_id not in seen:
            result.append(sample_id)
            seen.add(sample_id)
    return result


def _prediction_dump_request_summary(
    *,
    enabled: bool,
    dump_dir: Optional[Path],
    sample_ids: Sequence[str],
    threshold: float,
    top_k: int,
    written: Optional[Sequence[Mapping[str, Any]]] = None,
    missing_sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "schema": "mem_d3g_prediction_dump_request_v0",
        "dump_dir": str(dump_dir) if dump_dir is not None else None,
        "requested_sample_ids": list(sample_ids),
        "threshold": float(threshold),
        "top_k": int(top_k),
        "written_dumps": list(written or []),
        "written_dump_count": int(len(written or [])),
        "missing_sample_ids": list(missing_sample_ids or []),
        "scope": "selected_samples_only" if enabled else "disabled",
    }


def _write_prediction_dump(
    *,
    dump_dir: Path,
    dump: Mapping[str, Any],
) -> Dict[str, Any]:
    sample_id = str(dump.get("sample_id", "sample"))
    split_name = str(dump.get("split", "split"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / _safe_prediction_dump_filename(split_name, sample_id)
    if path.exists():
        raise FileExistsError("refusing to overwrite prediction dump: {}".format(path))
    _write_json(path, dict(dump))
    return {
        "sample_id": sample_id,
        "split": split_name,
        "path": str(path),
        "num_nodes": int(dump.get("num_nodes", 0)),
        "false_positive": int(dump.get("edge_confusion_at_threshold", {}).get("false_positive", 0)),
        "false_negative": int(dump.get("edge_confusion_at_threshold", {}).get("false_negative", 0)),
    }


def _option2b_counts(mapped_items: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    keys = (
        "num_observed_visible_instances",
        "num_gt_aligned_instances",
        "num_hidden_gt_instances",
        "num_unmatched_observed_instances",
    )
    counts = {key: 0 for key in keys}
    for mapped in mapped_items:
        metadata = dict(mapped.get("mem_metadata", {}) or {})
        for key in keys:
            counts[key] += int(metadata.get(key, 0) or 0)
    return counts


def evaluate_mem_graph_model(
    model: MemGraphDenseKnownNodes,
    mapped_items: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    thresholds: Optional[Sequence[float]] = None,
    score_histogram_bin_edges: Optional[Sequence[float]] = None,
    prediction_dump_dir: Optional[Path] = None,
    prediction_dump_sample_ids: Optional[Sequence[str]] = None,
    prediction_dump_threshold: float = DEFAULT_PREDICTION_DUMP_THRESHOLD,
    prediction_dump_top_k: int = DEFAULT_PREDICTION_DUMP_TOP_K,
) -> Dict[str, Any]:
    """Evaluate model over mapped MEM items one record at a time."""

    dump_sample_ids = set(_normalise_prediction_dump_sample_ids(prediction_dump_sample_ids))
    dump_enabled = prediction_dump_dir is not None and bool(dump_sample_ids)
    scores: List[float] = []
    labels: List[int] = []
    per_record: List[Dict[str, Any]] = []
    target_summaries: List[Dict[str, Any]] = []
    written_dumps: List[Dict[str, Any]] = []
    aux_enabled = bool(getattr(model, "visible_blocks_hidden_aux_enabled", False))
    aux_loss_weight = float(getattr(model, "visible_blocks_hidden_aux_loss_weight", 0.0))
    aux_pos_weight = float(getattr(model, "visible_blocks_hidden_aux_pos_weight", 0.0))
    aux_record_values: List[Dict[str, List[Any]]] = []
    for mapped in mapped_items:
        output, target = _eval_output_and_target(model, mapped)
        logits = output["graph_logits"].detach().cpu()
        selected = _extract_scores_labels_from_logits(logits, target)
        record_target_summary = summarize_edge_targets(target)
        record_metrics = compute_mem_graph_score_metrics(
            selected["scores"], selected["labels"], thresholds=thresholds
        ) if selected["labels"] else {}
        record_aux_metrics: Dict[str, Any] = {}
        if aux_enabled and "visible_blocks_hidden_target" in mapped and "visible_blocks_hidden_logits" in output:
            aux_logits = output["visible_blocks_hidden_logits"].detach().cpu()
            aux_target = mapped["visible_blocks_hidden_target"].detach().cpu()
            record_aux_metrics = _compute_visible_blocks_hidden_aux_metrics(
                aux_logits,
                aux_target,
                loss_weight=aux_loss_weight,
                pos_weight=aux_pos_weight,
            )
            selected_aux = _extract_node_scores_labels_from_logits(aux_logits, aux_target)
            aux_record_values.append(
                {
                    "logits": [float(value) for value in aux_logits.tolist()],
                    "labels": selected_aux["labels"],
                }
            )
        sample_id = str(mapped.get("image_id") or mapped.get("mem_metadata", {}).get("sample_id") or "")
        if dump_enabled and sample_id in dump_sample_ids:
            dump = build_mem_prediction_dump(
                mapped,
                logits,
                target,
                split_name=split_name,
                threshold=prediction_dump_threshold,
                top_k=prediction_dump_top_k,
            )
            written_dumps.append(_write_prediction_dump(dump_dir=prediction_dump_dir, dump=dump))
        per_record.append(
            {
                "image_id": mapped.get("image_id"),
                "target_summary": record_target_summary,
                "metrics": record_metrics,
                "visible_blocks_hidden_aux_metrics": record_aux_metrics,
                "mem_metadata": dict(mapped.get("mem_metadata", {}) or {}),
            }
        )
        target_summaries.append(record_target_summary)
        scores.extend(selected["scores"])
        labels.extend(selected["labels"])
    aggregate = aggregate_target_summaries(target_summaries)
    if labels:
        score_metrics = compute_mem_graph_score_metrics(
            scores,
            labels,
            thresholds=thresholds,
            score_histogram_bin_edges=score_histogram_bin_edges,
        )
    else:
        score_metrics = {
            "num_pairs": 0,
            "num_positive": 0,
            "num_negative": 0,
            "positive_edge_base_rate": None,
            "validation_ap": None,
            "ap_over_base_rate": None,
            "metrics_at_threshold_0_5": compute_binary_metrics_from_scores([], [], 0.5),
            "precision_at_threshold_0_5": 0.0,
            "recall_at_threshold_0_5": 0.0,
            "f1_at_threshold_0_5": 0.0,
            "threshold_sweep": [],
            "best_threshold": None,
            "best_f1": None,
            "best_threshold_metrics": None,
            "mean_positive_probability": None,
            "mean_negative_probability": None,
            "positive_negative_probability_gap": None,
        }
        histogram_edges = _normalise_score_histogram_bin_edges(score_histogram_bin_edges)
        if histogram_edges is not None:
            score_metrics["score_histogram"] = compute_score_histogram([], [], histogram_edges)
    return {
        "split_name": split_name,
        "num_records": int(len(mapped_items)),
        "target_summary": aggregate,
        "node_count": aggregate["num_nodes"],
        "edge_count": aggregate["num_positive_directed_edges"],
        "pair_count": aggregate["num_non_diagonal_pairs"],
        "option2b_node_counts": _option2b_counts(mapped_items),
        "visible_blocks_hidden_aux_metrics": _aggregate_visible_blocks_hidden_aux_metrics(
            aux_record_values,
            enabled=aux_enabled,
            loss_weight=aux_loss_weight,
            pos_weight=aux_pos_weight,
        ),
        "per_record": per_record,
        "prediction_dumps": written_dumps,
        **score_metrics,
    }


def _resolve_graph_loss_pos_weight(
    loss_mode: str,
    train_target_summary: Mapping[str, Any],
    *,
    fixed_graph_loss_pos_weight: Optional[float] = None,
) -> Dict[str, Any]:
    if loss_mode not in LOSS_MODES:
        raise ValueError("loss_mode must be one of {}; got {!r}".format(LOSS_MODES, loss_mode))
    if loss_mode == "bce":
        return {"value": 0.0, "source": "unweighted_bce"}
    if loss_mode == "fixed_pos_weighted_bce":
        if fixed_graph_loss_pos_weight is None:
            raise ValueError("fixed_pos_weighted_bce requires fixed_graph_loss_pos_weight")
        value = float(fixed_graph_loss_pos_weight)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fixed_graph_loss_pos_weight must be finite and positive")
        return {"value": value, "source": "fixed_graph_loss_pos_weight"}
    num_positive = int(train_target_summary["num_positive_directed_edges"])
    num_negative = int(train_target_summary["num_negative_directed_edges"])
    if num_positive <= 0:
        raise ValueError("train_pos_weighted_bce requires at least one positive train edge")
    value = num_negative / float(num_positive)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("computed graph loss pos_weight must be finite and positive")
    return {"value": float(value), "source": "train_split_negative_positive_ratio"}


def _checkpoint_metric_value(metrics: Mapping[str, Any], metric_name: str) -> Optional[float]:
    value = metrics.get(metric_name)
    if value is None:
        return None
    return float(value)


def _compact_checkpoint_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "split_name",
        "num_records",
        "node_count",
        "edge_count",
        "pair_count",
        "num_pairs",
        "num_positive",
        "num_negative",
        "positive_edge_base_rate",
        "validation_ap",
        "ap_over_base_rate",
        "best_threshold",
        "best_f1",
        "best_threshold_metrics",
        "metrics_at_threshold_0_5",
        "precision_at_threshold_0_5",
        "recall_at_threshold_0_5",
        "f1_at_threshold_0_5",
        "mean_positive_probability",
        "mean_negative_probability",
        "positive_negative_probability_gap",
        "visible_blocks_hidden_aux_metrics",
    )
    return {key: _to_jsonable(metrics.get(key)) for key in keys if key in metrics}


def _write_checkpoint(
    *,
    path: Path,
    model: MemGraphDenseKnownNodes,
    optimizer: torch.optim.Optimizer,
    cfg,
    role: str,
    iteration: int,
    validation_metric_name: str,
    validation_metric_value: Optional[float],
    validation_metrics: Optional[Mapping[str, Any]],
    include_optimizer_state: bool,
    model_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if path.exists():
        raise FileExistsError("refusing to overwrite checkpoint: {}".format(path))
    payload: Dict[str, Any] = {
        "schema": "mem_d3g_checkpoint_v1",
        "role": str(role),
        "iteration": int(iteration),
        "model": dict(model_state) if model_state is not None else model.state_dict(),
        "cfg": cfg.dump(),
        "validation_metric_name": str(validation_metric_name),
        "validation_metric_value": validation_metric_value,
        "validation_metrics": _compact_checkpoint_metrics(validation_metrics or {}),
        "contains_optimizer_state": bool(include_optimizer_state),
    }
    if include_optimizer_state:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "role": str(role),
        "iteration": int(iteration),
        "validation_metric_name": str(validation_metric_name),
        "validation_metric_value": validation_metric_value,
        "size_bytes": int(path.stat().st_size),
        "contains_optimizer_state": bool(include_optimizer_state),
    }


def _clone_model_state_dict(model: MemGraphDenseKnownNodes) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


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


def _train_model(
    model: MemGraphDenseKnownNodes,
    train_mapped: Sequence[Mapping[str, Any]],
    *,
    max_iter: int,
    lr: float,
    validation_mapped: Optional[Sequence[Mapping[str, Any]]] = None,
    thresholds: Optional[Sequence[float]] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_eval_period: Optional[int] = None,
    checkpoint_metric: str = "validation_ap",
    checkpoint_cfg=None,
    include_optimizer_state_in_checkpoints: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    history: List[Dict[str, Any]] = []
    checkpoint_enabled = checkpoint_dir is not None
    checkpoint_summary: Dict[str, Any] = {
        "enabled": bool(checkpoint_enabled),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_eval_period": int(checkpoint_eval_period) if checkpoint_eval_period is not None else None,
        "checkpoint_metric": str(checkpoint_metric),
        "include_optimizer_state": bool(include_optimizer_state_in_checkpoints),
        "validation_evaluations": [],
        "written_checkpoints": [],
        "best_checkpoint": None,
        "final_checkpoint": None,
    }
    if checkpoint_enabled:
        if checkpoint_cfg is None:
            raise ValueError("checkpoint_cfg is required when checkpointing is enabled")
        if not validation_mapped:
            raise ValueError("validation_mapped is required when checkpointing is enabled")
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            raise FileExistsError("refusing to use existing checkpoint_dir: {}".format(checkpoint_dir))
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_eval_period = int(checkpoint_eval_period or max(1, int(max_iter)))
        if checkpoint_eval_period <= 0:
            raise ValueError("checkpoint_eval_period must be positive")
        checkpoint_summary["checkpoint_eval_period"] = int(checkpoint_eval_period)
    best_metric: Optional[float] = None
    best_checkpoint_record: Optional[Dict[str, Any]] = None
    best_iteration: Optional[int] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation_metrics: Optional[Dict[str, Any]] = None
    last_validation_metrics: Optional[Dict[str, Any]] = None

    def evaluate_validation_for_checkpoint(iteration: int) -> Dict[str, Any]:
        nonlocal best_metric, best_iteration, best_model_state, best_validation_metrics, last_validation_metrics
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required")
        validation_metrics = evaluate_mem_graph_model(
            model,
            validation_mapped or [],
            split_name="val",
            thresholds=thresholds,
            score_histogram_bin_edges=None,
        )
        metric_value = _checkpoint_metric_value(validation_metrics, checkpoint_metric)
        eval_record = {
            "iteration": int(iteration),
            "validation_metric_name": str(checkpoint_metric),
            "validation_metric_value": metric_value,
            "validation_ap": validation_metrics.get("validation_ap"),
            "best_f1": validation_metrics.get("best_f1"),
            "best_threshold": validation_metrics.get("best_threshold"),
        }
        checkpoint_summary["validation_evaluations"].append(_to_jsonable(eval_record))
        last_validation_metrics = validation_metrics
        if metric_value is not None and (best_metric is None or metric_value > best_metric):
            best_metric = float(metric_value)
            best_iteration = int(iteration)
            best_validation_metrics = validation_metrics
            best_model_state = _clone_model_state_dict(model)
        return validation_metrics

    for iteration in range(int(max_iter)):
        mapped = train_mapped[iteration % len(train_mapped)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model([dict(mapped)])
        graph_loss = losses.get("loss_mem_dense_graph")
        if graph_loss is None:
            raise ValueError("model did not return loss_mem_dense_graph")
        finite_losses = {
            name: value
            for name, value in losses.items()
            if name.startswith("loss_") and isinstance(value, torch.Tensor)
        }
        if not finite_losses:
            raise ValueError("model did not return any scalar loss tensors")
        loss = sum(finite_losses.values())
        for loss_name, loss_value in finite_losses.items():
            if not torch.isfinite(loss_value).all().item():
                raise FloatingPointError("non-finite {} at iteration {}".format(loss_name, iteration))
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError("non-finite total loss at iteration {}".format(iteration))
        loss.backward()
        grad_norm = _total_gradient_norm(model)
        optimizer.step()
        history_record = {
            "iteration": int(iteration),
            "image_id": mapped.get("image_id"),
            "loss_total": float(loss.detach().cpu().item()),
            "loss_total_finite": bool(torch.isfinite(loss.detach()).all().item()),
            "loss_mem_dense_graph": float(graph_loss.detach().cpu().item()),
            "loss_mem_dense_graph_finite": bool(torch.isfinite(graph_loss.detach()).all().item()),
            "grad_norm_total": grad_norm["value"],
            "grad_norm_total_finite": grad_norm["finite"],
            "grad_norm_saw_grad": grad_norm["saw_grad"],
        }
        for loss_name, loss_value in sorted(finite_losses.items()):
            if loss_name == "loss_mem_dense_graph":
                continue
            history_record[loss_name] = float(loss_value.detach().cpu().item())
            history_record["{}_finite".format(loss_name)] = bool(torch.isfinite(loss_value.detach()).all().item())
        history.append(history_record)
        if checkpoint_enabled and (
            (iteration + 1) % int(checkpoint_eval_period) == 0 or (iteration + 1) == int(max_iter)
        ):
            evaluate_validation_for_checkpoint(iteration + 1)
    if checkpoint_enabled:
        if last_validation_metrics is None:
            last_validation_metrics = evaluate_validation_for_checkpoint(int(max_iter))
        if best_model_state is None:
            best_iteration = int(max_iter)
            best_validation_metrics = last_validation_metrics
            best_model_state = _clone_model_state_dict(model)
            best_metric = _checkpoint_metric_value(last_validation_metrics, checkpoint_metric)
        best_checkpoint_record = _write_checkpoint(
            path=Path(checkpoint_dir) / "model_best_validation.pth",
            model=model,
            optimizer=optimizer,
            cfg=checkpoint_cfg,
            role="best_validation",
            iteration=int(best_iteration if best_iteration is not None else max_iter),
            validation_metric_name=checkpoint_metric,
            validation_metric_value=best_metric,
            validation_metrics=best_validation_metrics,
            include_optimizer_state=include_optimizer_state_in_checkpoints,
            model_state=best_model_state,
        )
        checkpoint_summary["best_checkpoint"] = best_checkpoint_record
        final_checkpoint_record = _write_checkpoint(
            path=Path(checkpoint_dir) / "model_final.pth",
            model=model,
            optimizer=optimizer,
            cfg=checkpoint_cfg,
            role="final",
            iteration=int(max_iter),
            validation_metric_name=checkpoint_metric,
            validation_metric_value=_checkpoint_metric_value(last_validation_metrics, checkpoint_metric),
            validation_metrics=last_validation_metrics,
            include_optimizer_state=include_optimizer_state_in_checkpoints,
        )
        checkpoint_summary["final_checkpoint"] = final_checkpoint_record
        checkpoint_summary["written_checkpoints"] = [
            record
            for record in (checkpoint_summary.get("best_checkpoint"), checkpoint_summary.get("final_checkpoint"))
            if record
        ]
    return history, checkpoint_summary


def _inventory_files(output_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file():
            entries.append(
                {
                    "path": path.name,
                    "size_bytes": int(path.stat().st_size),
                    "checkpoint_like": path.suffix.lower() in (".pth", ".pt", ".ckpt"),
                }
            )
    return entries


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_artifacts(
    *,
    output_dir: Path,
    cfg,
    config_file: Optional[Path],
    split_json: Path,
    command: Mapping[str, Any],
    metrics: Mapping[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=output_dir.exists())
    (output_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    _write_json(output_dir / "command.json", dict(command))
    if split_json.is_file():
        shutil.copyfile(split_json, output_dir / "split_manifest.json")
    else:
        _write_json(output_dir / "split_manifest.json", {"source": str(split_json), "missing": True})
    _write_json(output_dir / "metrics.json", dict(metrics))
    if config_file is not None and Path(config_file).is_file():
        summary["source_config_file"] = str(config_file)
    artifact_inventory = {
        "schema": "mem_d3g_artifact_inventory_v0",
        "output_dir": str(output_dir),
        "inventory_timing": "after_config_command_split_metrics_before_artifact_inventory_and_summary",
        "expected_artifact_names": [
            "artifact_inventory.json",
            "command.json",
            "config.yaml",
            "metrics.json",
            "split_manifest.json",
            "summary.json",
        ],
        "inventory_excludes_finalized_files": ["artifact_inventory.json", "summary.json"],
        "files": _inventory_files(output_dir),
    }
    _write_json(output_dir / "artifact_inventory.json", artifact_inventory)
    summary["artifact_inventory"] = artifact_inventory
    _write_json(output_dir / "summary.json", summary)
    return artifact_inventory


def run_mem_graph_training(
    *,
    config_file: Optional[PathLike] = DEFAULT_CONFIG_FILE,
    records_json: Optional[PathLike] = None,
    split_json: Optional[PathLike] = None,
    output_dir: PathLike,
    data_root: Optional[str] = None,
    device: str = "cpu",
    max_iter: int = 0,
    lr: Optional[float] = None,
    seed: int = 0,
    loss_mode: str = DEFAULT_LOSS_MODE,
    fixed_graph_loss_pos_weight: Optional[float] = None,
    cfg_overrides: Optional[Sequence[str]] = None,
    thresholds: Optional[Sequence[float]] = None,
    score_histogram_bin_edges: Optional[Sequence[float]] = None,
    enable_checkpoints: bool = False,
    checkpoint_dir: Optional[PathLike] = None,
    checkpoint_eval_period: Optional[int] = None,
    checkpoint_metric: str = "validation_ap",
    include_optimizer_state_in_checkpoints: bool = False,
    require_scene_disjoint: bool = True,
    prediction_dump_dir: Optional[PathLike] = None,
    prediction_dump_sample_ids: Optional[Sequence[str]] = None,
    prediction_dump_threshold: float = DEFAULT_PREDICTION_DUMP_THRESHOLD,
    prediction_dump_top_k: int = DEFAULT_PREDICTION_DUMP_TOP_K,
    command_argv: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run a manifest-based MEM train/eval experiment and write run artifacts."""

    output_path = Path(output_dir).expanduser()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite existing output_dir: {}".format(output_path))
    loss_mode = str(loss_mode or DEFAULT_LOSS_MODE)
    if loss_mode not in LOSS_MODES:
        raise ValueError("loss_mode must be one of {}; got {!r}".format(LOSS_MODES, loss_mode))
    device = _normalise_training_device(device)
    cfg = setup_mem_graph_cfg(config_file, device=device, cfg_overrides=cfg_overrides, clear_model_weights=True)
    records_path = _as_path(records_json) or _as_path(getattr(cfg.DATASETS, "MEM_RECORDS_JSON", ""))
    split_path = _as_path(split_json) or _as_path(getattr(cfg.DATASETS, "MEM_SPLIT_JSON", ""))
    if records_path is None:
        raise ValueError("records_json must be provided by CLI/function or cfg.DATASETS.MEM_RECORDS_JSON")
    if split_path is None:
        raise ValueError("split_json must be provided by CLI/function or cfg.DATASETS.MEM_SPLIT_JSON")
    checkpoint_path = _as_path(checkpoint_dir)
    if enable_checkpoints and checkpoint_path is None:
        checkpoint_path = output_path / "checkpoints"
    data_root_value = str(data_root if data_root is not None else getattr(cfg.DATASETS, "ROOT", "."))
    lr_value = float(lr if lr is not None else cfg.SOLVER.BASE_LR)
    histogram_edges = _normalise_score_histogram_bin_edges(score_histogram_bin_edges)
    dump_sample_ids = _normalise_prediction_dump_sample_ids(prediction_dump_sample_ids)
    dump_path = _as_path(prediction_dump_dir)
    prediction_dump_enabled = dump_path is not None
    if dump_sample_ids and dump_path is None:
        raise ValueError("prediction_dump_sample_ids requires prediction_dump_dir")
    if dump_path is not None and not dump_sample_ids:
        raise ValueError("prediction_dump_dir requires at least one selected prediction_dump_sample_id")
    if dump_path is not None and not dump_path.is_absolute():
        dump_path = output_path / dump_path
    prediction_dump_threshold = float(prediction_dump_threshold)
    if prediction_dump_threshold < 0.0 or prediction_dump_threshold > 1.0:
        raise ValueError("prediction_dump_threshold must be in [0,1]")
    prediction_dump_top_k = int(prediction_dump_top_k)
    if prediction_dump_top_k < 0:
        raise ValueError("prediction_dump_top_k must be non-negative")
    cfg.DATASETS.MEM_RECORDS_JSON = str(records_path)
    cfg.DATASETS.MEM_SPLIT_JSON = str(split_path)
    cfg.DATASETS.ROOT = data_root_value
    if getattr(cfg.DATASETS, "MEM_CNABU_DERIVED_ROOT", ""):
        cfg.DATASETS.MEM_CNABU_DERIVED_ROOT = str(Path(cfg.DATASETS.MEM_CNABU_DERIVED_ROOT).expanduser())
    cfg.SOLVER.MAX_ITER = int(max_iter)
    cfg.OUTPUT_DIR = str(output_path)
    cfg.MODEL.MEM_GRAPH.LOSS_MODE = loss_mode

    torch.manual_seed(int(seed))
    if str(device).startswith("cuda:"):
        torch.cuda.manual_seed_all(int(seed))

    split_manifest = load_mem_graph_split_manifest(split_path)
    records_by_split = load_mem_graph_records_for_splits(
        records_path,
        split_manifest,
        require_scene_disjoint=require_scene_disjoint,
    )
    if prediction_dump_enabled:
        available_sample_ids = {
            str(record.get("sample_id"))
            for split_records in records_by_split.values()
            for record in split_records
        }
        missing_sample_ids = [sample_id for sample_id in dump_sample_ids if sample_id not in available_sample_ids]
        if missing_sample_ids:
            raise ValueError("prediction dump sample ids not found in selected records/split: {}".format(missing_sample_ids))
    dataset_names = register_mem_graph_datasets(cfg, records_by_split)
    try:
        mapped_by_split = map_mem_graph_records(cfg, records_by_split, data_root=data_root_value)
    finally:
        unregister_mem_graph_datasets(dataset_names)

    train_target_summaries = [summarize_edge_targets(mapped["graph_gt"]) for mapped in mapped_by_split["train"]]
    train_target_summary = aggregate_target_summaries(train_target_summaries)
    graph_loss_pos_weight = _resolve_graph_loss_pos_weight(
        loss_mode,
        train_target_summary,
        fixed_graph_loss_pos_weight=fixed_graph_loss_pos_weight,
    )
    cfg.MODEL.MEM_GRAPH.GRAPH_LOSS_POS_WEIGHT = float(graph_loss_pos_weight["value"])

    model = MemGraphDenseKnownNodes(cfg)
    train_history, checkpoint_summary = _train_model(
        model,
        mapped_by_split["train"],
        max_iter=int(max_iter),
        lr=lr_value,
        validation_mapped=mapped_by_split["val"],
        thresholds=thresholds,
        checkpoint_dir=checkpoint_path if enable_checkpoints else None,
        checkpoint_eval_period=checkpoint_eval_period,
        checkpoint_metric=checkpoint_metric,
        checkpoint_cfg=cfg,
        include_optimizer_state_in_checkpoints=include_optimizer_state_in_checkpoints,
    )

    metrics: Dict[str, Any] = {
        "schema": "mem_d3g_train_eval_metrics_v0",
        "train_target_summary": train_target_summary,
        "graph_loss_pos_weight": float(graph_loss_pos_weight["value"]),
        "graph_loss_pos_weight_source": str(graph_loss_pos_weight["source"]),
        "visible_blocks_hidden_aux_enabled": bool(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED),
        "visible_blocks_hidden_aux_loss_weight": float(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_LOSS_WEIGHT),
        "visible_blocks_hidden_aux_pos_weight": float(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT),
        "splits": {},
    }
    prediction_dump_written: List[Dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        mapped_items = mapped_by_split.get(split_name, [])
        if mapped_items:
            split_metrics = evaluate_mem_graph_model(
                model,
                mapped_items,
                split_name=split_name,
                thresholds=thresholds,
                score_histogram_bin_edges=histogram_edges,
                prediction_dump_dir=dump_path,
                prediction_dump_sample_ids=dump_sample_ids,
                prediction_dump_threshold=prediction_dump_threshold,
                prediction_dump_top_k=prediction_dump_top_k,
            )
            metrics["splits"][split_name] = split_metrics
            prediction_dump_written.extend(split_metrics.get("prediction_dumps", []))
        else:
            metrics["splits"][split_name] = {"split_name": split_name, "num_records": 0}

    prediction_dump_summary = _prediction_dump_request_summary(
        enabled=prediction_dump_enabled,
        dump_dir=dump_path,
        sample_ids=dump_sample_ids,
        threshold=prediction_dump_threshold,
        top_k=prediction_dump_top_k,
        written=prediction_dump_written,
        missing_sample_ids=[],
    )
    metrics["prediction_dump"] = prediction_dump_summary

    command = {
        "argv": list(command_argv if command_argv is not None else sys.argv),
        "config_file": str(_as_path(config_file)) if config_file is not None else None,
        "records_json": str(records_path),
        "split_json": str(split_path),
        "output_dir": str(output_path),
        "device": str(device),
        "max_iter": int(max_iter),
        "loss_mode": loss_mode,
        "fixed_graph_loss_pos_weight": float(fixed_graph_loss_pos_weight)
        if fixed_graph_loss_pos_weight is not None
        else None,
        "enable_checkpoints": bool(enable_checkpoints),
        "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_eval_period": int(checkpoint_eval_period) if checkpoint_eval_period is not None else None,
        "checkpoint_metric": str(checkpoint_metric),
        "include_optimizer_state_in_checkpoints": bool(include_optimizer_state_in_checkpoints),
        "require_scene_disjoint": bool(require_scene_disjoint),
        "prediction_dump": prediction_dump_summary,
    }
    if histogram_edges is not None:
        command["score_histogram_bin_edges"] = histogram_edges
    validation_metrics = metrics["splits"].get("val", {"split_name": "val", "num_records": 0})
    test_metrics = metrics["splits"].get("test", {"split_name": "test", "num_records": 0})
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        **_cuda_device_summary(device),
        "config_file": str(_as_path(config_file)) if config_file is not None else None,
        "records_json": str(records_path),
        "split_json": str(split_path),
        "data_root": data_root_value,
        "mem_map_feature_source": str(cfg.INPUT.MEM_MAP_FEATURE_SOURCE),
        "mem_cnabu_derived_root": str(cfg.DATASETS.MEM_CNABU_DERIVED_ROOT),
        "mem_cnabu_pad_mode": str(cfg.INPUT.MEM_CNABU_PAD_MODE),
        "mem_input_channels": int(cfg.MODEL.MEM_GRAPH.IN_CHANNELS),
        "mem_input_normalization": str(cfg.MODEL.MEM_GRAPH.INPUT_NORMALIZATION),
        "mem_node_source": str(cfg.INPUT.MEM_NODE_SOURCE),
        "mem_graph_target_scope": str(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE),
        "mem_zero_map_node_features": bool(cfg.MODEL.MEM_GRAPH.ZERO_MAP_NODE_FEATURES),
        "mem_pair_geometry_enabled": bool(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED),
        "mem_pair_geometry_features": list(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES),
        "mem_visible_blocks_hidden_aux_enabled": bool(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED),
        "mem_visible_blocks_hidden_aux_loss_weight": float(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_LOSS_WEIGHT),
        "mem_visible_blocks_hidden_aux_pos_weight": float(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT),
        "model_meta_architecture": str(cfg.MODEL.META_ARCHITECTURE),
        "graph_head_name": str(cfg.MODEL.GRAPH_HEAD.NAME),
        "device": str(device),
        "seed": int(seed),
        "optimizer": "AdamW",
        "lr": lr_value,
        "loss_mode": loss_mode,
        "fixed_graph_loss_pos_weight": float(fixed_graph_loss_pos_weight)
        if fixed_graph_loss_pos_weight is not None
        else None,
        "graph_loss_pos_weight": float(graph_loss_pos_weight["value"]),
        "graph_loss_pos_weight_source": str(graph_loss_pos_weight["source"]),
        "max_iter": int(max_iter),
        "num_train_records": len(mapped_by_split["train"]),
        "num_val_records": len(mapped_by_split["val"]),
        "num_test_records": len(mapped_by_split["test"]),
        "dataset_names": dataset_names,
        "require_scene_disjoint": bool(require_scene_disjoint),
        "scene_disjoint_summary": _check_scene_disjoint(records_by_split) if require_scene_disjoint else None,
        "prediction_dump": prediction_dump_summary,
        "checkpoint_artifacts": checkpoint_summary,
        "train_history": train_history,
        "train_target_summary": train_target_summary,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "metrics_json": str(output_path / "metrics.json"),
        "summary_json": str(output_path / "summary.json"),
        "checkpoint_policy": "best_validation_and_final" if enable_checkpoints else "disabled",
        "safety": {
            "uses_explicit_records_manifest": True,
            "uses_explicit_split_manifest": True,
            "runs_model_forward": True,
            "runs_training_loop": int(max_iter) > 0,
            "runs_backward": int(max_iter) > 0,
            "runs_optimizer_step": int(max_iter) > 0,
            "batch_size": 1,
            "device": str(device),
            "uses_cuda": str(device).startswith("cuda:"),
            "loads_model_weights": False,
            "writes_checkpoints_or_model_outputs": bool(enable_checkpoints),
            "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_count": int(len(checkpoint_summary.get("written_checkpoints", []))),
            "writes_optimizer_state": bool(include_optimizer_state_in_checkpoints),
            "writes_hdf5_or_full_dataset": False,
            "writes_selected_prediction_dumps": bool(prediction_dump_enabled),
            "writes_full_prediction_export": False,
            "prediction_dump_selected_sample_count": int(len(dump_sample_ids)),
            "refuses_existing_output_dir_by_default": True,
        },
    }
    if histogram_edges is not None:
        summary["score_histogram_bin_edges"] = histogram_edges

    artifact_inventory = _write_run_artifacts(
        output_dir=output_path,
        cfg=cfg,
        config_file=_as_path(config_file),
        split_json=split_path,
        command=command,
        metrics=metrics,
        summary=summary,
    )
    summary["artifact_inventory"] = artifact_inventory
    return summary


def _split_float_csv(value: Optional[str]) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _split_str_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE, help="D3G MEM config file")
    parser.add_argument("--records-json", type=Path, default=None, help="Explicit MEM records JSON/JSONL manifest")
    parser.add_argument("--split-json", type=Path, default=None, help="Explicit train/val/test split manifest")
    parser.add_argument("--data-root", default=None, help="Dataset root for relative sample_dir values; defaults to cfg.DATASETS.ROOT")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; must not already exist")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--max-iter", type=int, default=0, help="Training iterations; 0 runs eval with random init only")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning rate; defaults to cfg.SOLVER.BASE_LR")
    parser.add_argument("--seed", type=int, default=0, help="Torch manual seed")
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default=DEFAULT_LOSS_MODE)
    parser.add_argument(
        "--fixed-graph-loss-pos-weight",
        type=float,
        default=None,
        help="Required when --loss-mode fixed_pos_weighted_bce; sets MEM graph BCE pos_weight",
    )
    parser.add_argument("--thresholds", default="", help="Comma-separated threshold sweep; 0.5 is added if absent")
    parser.add_argument(
        "--score-histogram-bin-edges",
        default="",
        help="Opt-in comma-separated score histogram bin edges for aggregate positive/negative calibration diagnostics",
    )
    parser.add_argument("--enable-checkpoints", action="store_true", help="Explicitly write controlled best-validation and final checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for controlled checkpoints; defaults to OUTPUT_DIR/checkpoints when checkpointing is enabled",
    )
    parser.add_argument(
        "--checkpoint-eval-period",
        type=int,
        default=None,
        help="Validation-evaluation period, in iterations, for best-validation checkpoint selection",
    )
    parser.add_argument(
        "--checkpoint-metric",
        default="validation_ap",
        help="Validation metric used for best checkpoint selection; default validation_ap",
    )
    parser.add_argument(
        "--include-optimizer-state-in-checkpoints",
        action="store_true",
        help="Opt-in optimizer state in checkpoints for resumability; disabled by default",
    )
    parser.add_argument(
        "--prediction-dump-dir",
        type=Path,
        default=None,
        help="Opt-in selected-sample prediction dump directory; requires --prediction-dump-sample-ids",
    )
    parser.add_argument(
        "--prediction-dump-sample-ids",
        default="",
        help="Comma-separated sample ids to dump; required with --prediction-dump-dir; never dumps unlisted samples",
    )
    parser.add_argument(
        "--prediction-dump-threshold",
        type=float,
        default=DEFAULT_PREDICTION_DUMP_THRESHOLD,
        help="Threshold used only for selected-sample FP/FN dump diagnostics",
    )
    parser.add_argument(
        "--prediction-dump-top-k",
        type=int,
        default=DEFAULT_PREDICTION_DUMP_TOP_K,
        help="Number of top FP/FN directed edges to retain per selected sample dump",
    )
    parser.add_argument(
        "--no-require-scene-disjoint",
        action="store_true",
        help="Disable train/val/test scene-disjoint validation",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional Detectron2-style CFG overrides after --",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    opts = list(args.opts or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    summary = run_mem_graph_training(
        config_file=args.config_file,
        records_json=args.records_json,
        split_json=args.split_json,
        output_dir=args.output_dir,
        data_root=args.data_root,
        device=args.device,
        max_iter=args.max_iter,
        lr=args.lr,
        seed=args.seed,
        loss_mode=args.loss_mode,
        fixed_graph_loss_pos_weight=args.fixed_graph_loss_pos_weight,
        cfg_overrides=opts,
        thresholds=_split_float_csv(args.thresholds),
        score_histogram_bin_edges=_split_float_csv(args.score_histogram_bin_edges),
        enable_checkpoints=bool(args.enable_checkpoints),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_eval_period=args.checkpoint_eval_period,
        checkpoint_metric=args.checkpoint_metric,
        include_optimizer_state_in_checkpoints=bool(args.include_optimizer_state_in_checkpoints),
        require_scene_disjoint=not args.no_require_scene_disjoint,
        prediction_dump_dir=args.prediction_dump_dir,
        prediction_dump_sample_ids=_split_str_csv(args.prediction_dump_sample_ids),
        prediction_dump_threshold=args.prediction_dump_threshold,
        prediction_dump_top_k=args.prediction_dump_top_k,
        command_argv=sys.argv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
