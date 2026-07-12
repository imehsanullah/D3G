#!/usr/bin/env python3
"""Run no-GT inference with a saved MEM CNABU component splitter checkpoint."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from train_mem_cnabu_component_splitter import (  # noqa: E402
    load_runtime_checkpoint,
    predict_record_runtime_nodes,
)
from train_mem_cnabu_node_proposal import (  # noqa: E402
    cnabu_path_from_record,
    read_json,
    read_records,
    split_records,
)


SCHEMA = "mem_cnabu_component_splitter_no_gt_inference_v0"


def select_record(
    *,
    records_json: Path,
    split_json: Optional[Path],
    split_name: str,
    sample_id: Optional[str],
) -> Dict[str, Any]:
    records = read_records(records_json)
    if sample_id is not None:
        for record in records:
            if str(record["sample_id"]) == str(sample_id):
                return dict(record)
        raise ValueError(f"sample_id not found in {records_json}: {sample_id}")
    if split_json is None:
        if not records:
            raise ValueError(f"records JSON is empty: {records_json}")
        return dict(records[0])
    split_manifest = read_json(split_json)
    records_by_split = split_records(records, split_manifest, max_train=None, max_val=None, max_test=None)
    split_records_list = records_by_split[str(split_name)]
    if not split_records_list:
        raise ValueError(f"split {split_name!r} has no records")
    return dict(split_records_list[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records-json", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--no-masks", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(str(args.device))
    model, config, checkpoint_payload = load_runtime_checkpoint(args.checkpoint, device)
    records_json = args.records_json or Path(config["DATA"]["RECORDS_JSON"])
    split_json = args.split_json or Path(config["DATA"]["SPLIT_JSON"])
    record = select_record(
        records_json=records_json,
        split_json=split_json,
        split_name=str(args.split),
        sample_id=args.sample_id,
    )
    nodes = predict_record_runtime_nodes(
        model,
        record,
        config,
        device,
        include_masks=not bool(args.no_masks),
    )
    cnabu_path = cnabu_path_from_record(record, Path(config["DATA"]["RAW_ROOT"]), Path(config["DATA"]["CNABU_ROOT"]))
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_schema": checkpoint_payload.get("schema"),
        "device": str(device),
        "sample_id": str(record["sample_id"]),
        "cnabu_path": str(cnabu_path),
        "split": str(args.split),
        "no_gt_input": True,
        "gt_loaded": False,
        "node_count": int(len(nodes)),
        "nodes": nodes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "node_count": int(len(nodes)), "sample_id": str(record["sample_id"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
