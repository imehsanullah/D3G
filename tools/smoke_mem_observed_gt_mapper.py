#!/usr/bin/env python3
"""Mapper-only MEM Option 2A/2B smoke for D3G.

This command loads a tiny explicit MEM records JSON/JSONL file, invokes only the
D3G MEM mapper, and prints compact shape/target diagnostics.  It can select the
GT-known-node or observed-visible-node contract through a D3G config file.  It
does not import D3G models, run a model forward, run backward/optimizer steps,
train, write checkpoints, pack HDF5, or create a full dataset export.
"""

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
from utils.configs import add_dep_graph_config, add_detr_config


MAX_RECORDS = 5
DEFAULT_CONFIG_FILE = None


def _shape(tensor: Any) -> List[int]:
    return [int(dim) for dim in tensor.shape]


def _summarize_mapper_output(output: Dict[str, Any]) -> Dict[str, Any]:
    image = output["image"]
    instances = output["instances"]
    graph_gt = output["graph_gt"]
    dense_gt = output["dense_gt"]
    boxes = instances.gt_boxes.tensor
    classes = instances.gt_classes
    return {
        "image_id": output.get("image_id"),
        "height": int(output.get("height")),
        "width": int(output.get("width")),
        "image_shape_chw": _shape(image),
        "image_dtype": str(image.dtype).replace("torch.", ""),
        "image_finite": bool(torch.isfinite(image).all().item()),
        "instances": int(len(instances)),
        "boxes_shape": _shape(boxes),
        "classes_shape": _shape(classes),
        "graph_gt_shape": _shape(graph_gt),
        "dense_gt_shape": _shape(dense_gt),
        "graph_gt_edge_count": int(graph_gt.sum().item()),
        "dense_gt_matches_graph_gt": bool(torch.equal(dense_gt, graph_gt)),
        "selected_view_indices": output.get("mem_metadata", {}).get("selected_view_indices"),
        "node_order_instance_ids": output.get("mem_metadata", {}).get("node_order_instance_ids"),
        "mem_node_source": output.get("mem_metadata", {}).get("mem_node_source"),
        "mem_graph_target_scope": output.get("mem_metadata", {}).get("mem_graph_target_scope"),
        "observed_visible_instance_ids": output.get("mem_metadata", {}).get("observed_visible_instance_ids"),
        "gt_aligned_instance_ids": output.get("mem_metadata", {}).get("gt_aligned_instance_ids"),
        "unmatched_observed_instance_ids": output.get("mem_metadata", {}).get("unmatched_observed_instance_ids"),
        "hidden_gt_instance_ids": output.get("mem_metadata", {}).get("hidden_gt_instance_ids"),
        "runs_model_forward": False,
        "runs_training": False,
    }


def _build_cfg(config_file: Optional[Path] = None, cfg_overrides: Optional[Sequence[str]] = None):
    cfg = get_cfg()
    add_dep_graph_config(cfg)
    add_detr_config(cfg)
    config_file = Path(config_file).expanduser() if config_file is not None else None
    if config_file is not None:
        cfg.merge_from_file(str(config_file))
    if cfg_overrides:
        cfg.merge_from_list(list(cfg_overrides))
    return cfg


def build_mem_observed_gt_mapper_smoke_summary(
    records_json: Path,
    data_root: str = ".",
    max_records: int = MAX_RECORDS,
    config_file: Optional[Path] = DEFAULT_CONFIG_FILE,
    expected_height: Optional[int] = None,
    expected_width: Optional[int] = None,
    validate_semantic_range: Optional[bool] = None,
    cfg_overrides: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if max_records < 1 or max_records > MAX_RECORDS:
        raise ValueError("max_records must be between 1 and {}".format(MAX_RECORDS))

    records = load_mem_observed_gt_records(records_json, max_records=max_records)
    cfg = _build_cfg(config_file=config_file, cfg_overrides=cfg_overrides)
    effective_data_root = str(data_root)
    if effective_data_root == "." and config_file is not None:
        effective_data_root = str(cfg.DATASETS.ROOT)
    mapper = MemObservedGtMapper(
        data_root=effective_data_root,
        is_train=False,
        graph_gt_type=cfg.INPUT.GRAPH_GT_TYPE,
        observed_view_protocol=cfg.INPUT.MEM_OBSERVED_VIEW_PROTOCOL,
        expected_height=int(cfg.INPUT.MEM_EXPECTED_HEIGHT if expected_height is None else expected_height),
        expected_width=int(cfg.INPUT.MEM_EXPECTED_WIDTH if expected_width is None else expected_width),
        max_selected_views=int(cfg.INPUT.MEM_MAX_SELECTED_VIEWS),
        semantic_class_min=int(cfg.INPUT.MEM_SEMANTIC_CLASS_MIN),
        semantic_class_max=int(cfg.INPUT.MEM_SEMANTIC_CLASS_MAX),
        validate_semantic_range=bool(cfg.INPUT.MEM_VALIDATE_SEMANTIC_RANGE if validate_semantic_range is None else validate_semantic_range),
        mem_node_source=cfg.INPUT.MEM_NODE_SOURCE,
        mem_graph_target_scope=cfg.INPUT.MEM_GRAPH_TARGET_SCOPE,
    )

    validation_errors: List[str] = []
    record_summaries: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        try:
            output = mapper(record)
            record_summaries.append(_summarize_mapper_output(output))
        except Exception as exc:  # pragma: no cover - exercised manually by bad smoke inputs
            validation_errors.append("record {} failed mapper smoke: {}: {}".format(record_index, type(exc).__name__, exc))

    return {
        "schema": "mem_observed_gt_d3g_mapper_smoke_summary_v0",
        "records_json": str(records_json),
        "data_root": str(effective_data_root),
        "config_file": str(Path(config_file).expanduser()) if config_file is not None else None,
        "mem_node_source": str(cfg.INPUT.MEM_NODE_SOURCE),
        "mem_graph_target_scope": str(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE),
        "num_records_loaded": len(records),
        "num_records_mapped": len(record_summaries),
        "record_summaries": record_summaries,
        "total_nodes": sum(int(item["instances"]) for item in record_summaries),
        "total_edges": sum(int(item["graph_gt_edge_count"]) for item in record_summaries),
        "validation_errors": validation_errors,
        "safety": {
            "runs_model_forward": False,
            "runs_training": False,
            "runs_backward": False,
            "runs_optimizer_step": False,
            "writes_hdf5_or_full_dataset": False,
            "writes_checkpoints_or_training_outputs": False,
            "loads_or_writes_model_weights": False,
            "max_records_cap": MAX_RECORDS,
        },
    }


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, required=True, help="JSON list/object or JSONL MEM mapper records")
    parser.add_argument("--data-root", default=".", help="Root for relative sample_dir values; absolute sample_dir values ignore it")
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG_FILE, help="Optional D3G MEM config selecting Option 2A/2B node-source contract")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS, help="Safety cap; 1..5")
    parser.add_argument("--expected-height", type=int, default=None, help="Expected materialized map height; defaults to config value; use 0 to disable")
    parser.add_argument("--expected-width", type=int, default=None, help="Expected materialized map width; defaults to config value; use 0 to disable")
    parser.add_argument(
        "--no-validate-semantic-range",
        action="store_true",
        help="Disable semantic_hms integer/range validation for synthetic debugging records",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional Detectron2-style CFG overrides after --, e.g. -- INPUT.MEM_NODE_SOURCE observed_instance_maps INPUT.MEM_GRAPH_TARGET_SCOPE observed_induced",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    opts = list(args.opts or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    summary = build_mem_observed_gt_mapper_smoke_summary(
        records_json=args.records_json,
        data_root=args.data_root,
        max_records=args.max_records,
        config_file=args.config_file,
        expected_height=args.expected_height,
        expected_width=args.expected_width,
        validate_semantic_range=False if args.no_validate_semantic_range else None,
        cfg_overrides=opts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary.get("validation_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
