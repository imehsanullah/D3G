"""MEM observed-input / GT-graph dataset records for D3G mapper smokes.

This module intentionally only loads and normalizes tiny explicit JSON/JSONL
preview records.  It does not scan raw MEM datasets, export/pack datasets, run
models, or start training.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


MEM_DATASET_PREFIX = "mem_"
DEFAULT_MEM_DATASET_NAME = "mem_option2a_mapper_smoke"


PathLike = Union[str, Path]
Record = Dict[str, Any]


def _read_records_json(path: PathLike) -> List[Record]:
    records_path = Path(path).expanduser()
    text = records_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("expected a JSON list when records file starts with '['")
        return [dict(record) for record in records]
    if text.startswith("{"):
        return [dict(json.loads(text))]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _coerce_int_list(values: Optional[Iterable[Any]]) -> List[int]:
    if values is None:
        return []
    return [int(value) for value in values]


def _resolve_sample_dir(record: Record) -> str:
    sample_dir = record.get("sample_dir") or record.get("pre_action_dir")
    if sample_dir:
        return str(sample_dir)

    source_root = record.get("source_root")
    sample_id = record.get("sample_id")
    if source_root and sample_id:
        return str(Path(str(source_root)) / str(sample_id) / "pre_action")

    source_file = record.get("source_file") or record.get("observed_source_file")
    if source_file:
        source_path = Path(str(source_file))
        if source_path.name == "hms.npz":
            return str(source_path.parent)
    return ""


def _first_present(record: Record, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def normalise_mem_observed_gt_record(record: Record) -> Record:
    """Return the flat D3G MEM mapper record schema for one preview/flat record.

    Supported inputs:
    - already-flat records with bbox/graph fields at top level;
    - Option 2A preview records from thesis_records with nested observed_input and
      target_graph sections.
    """

    record = dict(record)
    observed = dict(record.get("observed_input") or {})
    target = dict(record.get("target_graph") or {})

    if observed or target:
        graph_gt = target.get("graph_gt")
        dense_gt = target.get("dense_gt") if target.get("dense_gt") is not None else graph_gt
        return {
            "sample_id": record.get("sample_id") or record.get("id"),
            "sample_dir": _resolve_sample_dir(record),
            "observed_source_file": observed.get("source_file"),
            "target_source_file": target.get("source_file"),
            "selected_view_indices": _coerce_int_list(observed.get("selected_view_indices")),
            "height": _first_present(record, ("height", "image_height")),
            "width": _first_present(record, ("width", "image_width")),
            "bbox_xyxy_abs": target.get("bbox_xyxy_abs", []),
            "bbox_categories": _coerce_int_list(target.get("node_classes") or target.get("bbox_categories")),
            "node_order_instance_ids": _coerce_int_list(target.get("node_order_instance_ids")),
            "graph_gt": graph_gt if graph_gt is not None else [],
            "dense_gt": dense_gt if dense_gt is not None else [],
            "edge_type": target.get("edge_type", "blocks_access_to"),
            "source_schema": record.get("schema"),
            "record_mode": record.get("record_mode"),
            "observed_input_used": bool(record.get("observed_input_used", False)),
            "is_oracle_node_conditioned": bool(record.get("is_oracle_node_conditioned", False)),
            "is_training_export": bool(record.get("is_training_export", False)),
            "is_full_dataset_export": bool(record.get("is_full_dataset_export", False)),
        }

    graph_gt = record.get("graph_gt")
    dense_gt = record.get("dense_gt") if record.get("dense_gt") is not None else graph_gt
    return {
        "sample_id": record.get("sample_id") or record.get("id") or record.get("image_id"),
        "sample_dir": _resolve_sample_dir(record),
        "observed_source_file": record.get("observed_source_file") or record.get("source_file"),
        "target_source_file": record.get("target_source_file"),
        "selected_view_indices": _coerce_int_list(record.get("selected_view_indices")),
        "height": _first_present(record, ("height", "image_height")),
        "width": _first_present(record, ("width", "image_width")),
        "bbox_xyxy_abs": _first_present(record, ("bbox_xyxy_abs", "gt_boxes_xyxy_abs"), []),
        "bbox_categories": _coerce_int_list(
            _first_present(record, ("bbox_categories", "gt_classes", "node_classes"), [])
        ),
        "node_order_instance_ids": _coerce_int_list(record.get("node_order_instance_ids")),
        "graph_gt": graph_gt if graph_gt is not None else [],
        "dense_gt": dense_gt if dense_gt is not None else [],
        "edge_type": record.get("edge_type", "blocks_access_to"),
        "source_schema": record.get("source_schema") or record.get("schema"),
        "record_mode": record.get("record_mode"),
        "observed_input_used": bool(record.get("observed_input_used", True)),
        "is_oracle_node_conditioned": bool(record.get("is_oracle_node_conditioned", True)),
        "is_training_export": bool(record.get("is_training_export", False)),
        "is_full_dataset_export": bool(record.get("is_full_dataset_export", False)),
    }


def load_mem_observed_gt_records(
    records_json: PathLike,
    max_records: Optional[int] = None,
) -> List[Record]:
    """Load tiny explicit MEM preview records and normalize them for the mapper."""

    records = [normalise_mem_observed_gt_record(record) for record in _read_records_json(records_json)]
    if max_records is not None:
        if int(max_records) < 0:
            raise ValueError("max_records must be non-negative")
        records = records[: int(max_records)]
    return records


def register_mem_observed_gt_dataset(
    name: str,
    records_json: PathLike,
    max_records: Optional[int] = None,
) -> str:
    """Register an explicit tiny MEM records file with Detectron2's DatasetCatalog."""

    from detectron2.data import DatasetCatalog, MetadataCatalog

    if not str(name).startswith(MEM_DATASET_PREFIX):
        raise ValueError("MEM D3G dataset names should start with 'mem_'")
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name,
            func=lambda: load_mem_observed_gt_records(records_json, max_records=max_records),
        )
    MetadataCatalog.get(name).set(
        evaluator_type="mem_observed_gt_graph",
        records_json=str(records_json),
        graph_gt_type="dense",
        edge_type="blocks_access_to",
    )
    return name


def register_mem_observed_gt_dataset_from_cfg(cfg, dataset_name: Optional[str] = None) -> str:
    """Register the configured MEM preview dataset for mapper-only smokes.

    This helper is deliberately explicit; importing this module does not register
    any MEM dataset automatically because MEM preview/raw paths are experiment
    specific.
    """

    records_json = str(getattr(cfg.DATASETS, "MEM_RECORDS_JSON", ""))
    if not records_json:
        raise ValueError("cfg.DATASETS.MEM_RECORDS_JSON must be set for MEM dataset registration")
    if dataset_name is None:
        train = list(getattr(cfg.DATASETS, "TRAIN", []))
        dataset_name = train[0] if train else DEFAULT_MEM_DATASET_NAME
    max_records = getattr(cfg.DATASETS, "MEM_MAX_RECORDS", None)
    if max_records is not None and int(max_records) <= 0:
        max_records = None
    return register_mem_observed_gt_dataset(dataset_name, records_json, max_records=max_records)


__all__ = [
    "DEFAULT_MEM_DATASET_NAME",
    "MEM_DATASET_PREFIX",
    "load_mem_observed_gt_records",
    "normalise_mem_observed_gt_record",
    "register_mem_observed_gt_dataset",
    "register_mem_observed_gt_dataset_from_cfg",
]
