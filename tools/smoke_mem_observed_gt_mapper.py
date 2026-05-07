#!/usr/bin/env python3
"""Mapper-only MEM Option 2A smoke for D3G.

This command loads a tiny explicit MEM records JSON/JSONL file, invokes only the
D3G MEM mapper, and prints compact shape/target diagnostics.  It does not import
D3G models, run a model forward, run backward/optimizer steps, train, write
checkpoints, pack HDF5, or create a full dataset export.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.mem_observed_gt_dataset import load_mem_observed_gt_records
from data.mem_observed_gt_mapper import MemObservedGtMapper


MAX_RECORDS = 5


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
        "runs_model_forward": False,
        "runs_training": False,
    }


def build_mem_observed_gt_mapper_smoke_summary(
    records_json: Path,
    data_root: str = ".",
    max_records: int = MAX_RECORDS,
    expected_height: int = 140,
    expected_width: int = 200,
    validate_semantic_range: bool = True,
) -> Dict[str, Any]:
    if max_records < 1 or max_records > MAX_RECORDS:
        raise ValueError("max_records must be between 1 and {}".format(MAX_RECORDS))

    records = load_mem_observed_gt_records(records_json, max_records=max_records)
    mapper = MemObservedGtMapper(
        data_root=data_root,
        is_train=False,
        graph_gt_type="dense",
        expected_height=expected_height,
        expected_width=expected_width,
        validate_semantic_range=validate_semantic_range,
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
        "data_root": str(data_root),
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
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS, help="Safety cap; 1..5")
    parser.add_argument("--expected-height", type=int, default=140, help="Expected materialized map height; use 0 to disable")
    parser.add_argument("--expected-width", type=int, default=200, help="Expected materialized map width; use 0 to disable")
    parser.add_argument(
        "--no-validate-semantic-range",
        action="store_true",
        help="Disable semantic_hms integer/range validation for synthetic debugging records",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    summary = build_mem_observed_gt_mapper_smoke_summary(
        records_json=args.records_json,
        data_root=args.data_root,
        max_records=args.max_records,
        expected_height=args.expected_height,
        expected_width=args.expected_width,
        validate_semantic_range=not args.no_validate_semantic_range,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary.get("validation_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
