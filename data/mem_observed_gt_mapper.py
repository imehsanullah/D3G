"""D3G mapper for MEM observed map tensors with GT dense graph targets.

The mapper implements safe D3G-side MEM data contracts only:
raw MEM hms.npz + selected observed views -> [C,H,W] float32 image tensor,
plus either GT/oracle nodes with a full GT dense graph (Option 2A) or observed
instance-map-visible nodes with an induced GT dense graph over alignable visible
nodes (Option 2B).  It does not run a model forward, training, export, HDF5
packing, checkpoint loading, or checkpoint writing.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from detectron2.config import configurable
import detectron2.structures as structures

from data.mem_observed_gt_dataset import normalise_mem_observed_gt_record


DEFAULT_OBSERVED_VIEW_PROTOCOL = "uniform10"
DEFAULT_MAX_SELECTED_VIEWS = 10
DEFAULT_SEMANTIC_CLASS_MIN = 0
DEFAULT_SEMANTIC_CLASS_MAX = 14
DEFAULT_MEM_NODE_SOURCE = "gt"
DEFAULT_MEM_GRAPH_TARGET_SCOPE = "gt_all"
SUPPORTED_MEM_NODE_SOURCES = ("gt", "observed_instance_maps")
SUPPORTED_MEM_GRAPH_TARGET_SCOPES = ("gt_all", "observed_induced")


def _coerce_int_list(values: Iterable[Any]) -> List[int]:
    return [int(value) for value in values]


def select_mem_observed_view_indices(num_available_views: int, protocol: str) -> List[int]:
    """Select observed MEM view indices deterministically for mapper smokes."""

    n_views = int(num_available_views)
    if n_views < 0:
        raise ValueError("num_available_views must be non-negative")
    if n_views == 0:
        return []

    protocol = str(protocol).strip()
    if protocol == "all":
        return list(range(n_views))
    if protocol.startswith("uniform"):
        k = int(protocol.replace("uniform", ""))
        if k <= 0:
            raise ValueError("uniform view-selection K must be positive")
        if k >= n_views:
            return list(range(n_views))
        raw = np.linspace(0, n_views - 1, num=k)
        indices = sorted({int(round(value)) for value in raw})
        cursor = 0
        while len(indices) < k and cursor < n_views:
            if cursor not in indices:
                indices.append(cursor)
            cursor += 1
        return sorted(indices[:k])
    if protocol.startswith("first"):
        k = int(protocol.replace("first", ""))
        if k <= 0:
            raise ValueError("first view-selection K must be positive")
        return list(range(min(k, n_views)))
    raise ValueError("unknown MEM observed-view protocol {!r}".format(protocol))


def materialize_mem_observed_tensor(
    hms: np.ndarray,
    semantic_hms: np.ndarray,
    selected_view_indices: Sequence[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Materialize [K * (hms_channels + 1), H, W] in view-major order."""

    hms = np.asarray(hms)
    semantic_hms = np.asarray(semantic_hms)
    selected = _coerce_int_list(selected_view_indices)

    if hms.ndim != 4:
        raise ValueError("expected hms with shape [V,H,W,C], got {}".format(hms.shape))
    if semantic_hms.ndim != 3:
        raise ValueError("expected semantic_hms with shape [V,H,W], got {}".format(semantic_hms.shape))
    if hms.shape[:3] != semantic_hms.shape:
        raise ValueError(
            "hms first three dims {} must match semantic_hms shape {}".format(
                hms.shape[:3], semantic_hms.shape
            )
        )
    if not selected:
        raise ValueError("selected_view_indices must not be empty")
    invalid = [index for index in selected if index < 0 or index >= hms.shape[0]]
    if invalid:
        raise ValueError("selected_view_indices out of range for {} views: {}".format(hms.shape[0], invalid))

    channels: List[np.ndarray] = []
    layout: List[Dict[str, Any]] = []
    channel_index = 0
    for view_index in selected:
        for source_channel in range(int(hms.shape[3])):
            channels.append(hms[view_index, :, :, source_channel].astype(np.float32, copy=False))
            layout.append(
                {
                    "channel_index": int(channel_index),
                    "view_index": int(view_index),
                    "source_field": "hms",
                    "source_channel": int(source_channel),
                }
            )
            channel_index += 1
        channels.append(semantic_hms[view_index, :, :].astype(np.float32, copy=False))
        layout.append(
            {
                "channel_index": int(channel_index),
                "view_index": int(view_index),
                "source_field": "semantic_hms",
                "source_channel": None,
            }
        )
        channel_index += 1

    return np.stack(channels, axis=0).astype(np.float32, copy=False), layout


def _resolve_sample_dir(sample_dir: str, data_root: str) -> Path:
    if not sample_dir:
        raise ValueError("MEM mapper record must include sample_dir or pre_action_dir")
    path = Path(sample_dir).expanduser()
    if not path.is_absolute():
        path = Path(data_root).expanduser() / path
    if (path / "hms.npz").is_file():
        return path
    candidate = path / "pre_action"
    if (candidate / "hms.npz").is_file():
        return candidate
    raise FileNotFoundError("could not find hms.npz under {} or {}".format(path, candidate))


def _load_mem_hms_npz(pre_action_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    hms_path = pre_action_dir / "hms.npz"
    with np.load(hms_path, allow_pickle=False) as data:
        if "hms" not in data.files:
            raise KeyError("missing required hms field in {}".format(hms_path))
        if "semantic_hms" not in data.files:
            raise KeyError("missing required semantic_hms field in {}".format(hms_path))
        hms = np.asarray(data["hms"])
        semantic_hms = np.asarray(data["semantic_hms"])
        instance_maps = np.asarray(data["instance_maps"]) if "instance_maps" in data.files else None
    return hms, semantic_hms, instance_maps


def _as_graph_tensor(value: Any, name: str, num_nodes: int) -> torch.Tensor:
    arr = np.asarray(value, dtype=np.int64)
    if arr.size == 0 and num_nodes == 0:
        arr = np.zeros((0, 0), dtype=np.int64)
    if arr.shape != (num_nodes, num_nodes):
        raise ValueError("{} must have shape ({}, {}), got {}".format(name, num_nodes, num_nodes, arr.shape))
    graph = torch.as_tensor(arr, dtype=torch.long)
    values = set(int(value) for value in graph.reshape(-1).tolist())
    if not values.issubset({0, 1}):
        raise ValueError("{} must be binary 0/1, got values {}".format(name, sorted(values)))
    if num_nodes > 0 and bool(torch.diag(graph).any()):
        raise ValueError("{} diagonal must be zero".format(name))
    return graph


def _boxes_tensor(boxes_xyxy_abs: Any, num_nodes: int, height: int, width: int) -> torch.Tensor:
    boxes = np.asarray(boxes_xyxy_abs, dtype=np.float32)
    if boxes.size == 0 and num_nodes == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if boxes.shape != (num_nodes, 4):
        raise ValueError("bbox_xyxy_abs must have shape ({}, 4), got {}".format(num_nodes, boxes.shape))
    for box_index, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy_abs[{}] must have positive xyxy width/height".format(box_index))
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(
                "bbox_xyxy_abs[{}] is outside image size (height={}, width={}): {}".format(
                    box_index, height, width, [x1, y1, x2, y2]
                )
            )
    return torch.as_tensor(boxes, dtype=torch.float32)


def _validate_semantic_hms(
    semantic_hms: np.ndarray,
    selected_view_indices: Sequence[int],
    semantic_class_min: int,
    semantic_class_max: int,
    validate_semantic_range: bool,
) -> None:
    if not validate_semantic_range:
        return
    selected = np.asarray(semantic_hms)[list(selected_view_indices)]
    if selected.size == 0:
        return
    if not np.allclose(selected, np.round(selected), equal_nan=True):
        raise ValueError("semantic_hms selected values must be integer-like class ids")
    if float(np.nanmin(selected)) < float(semantic_class_min) or float(np.nanmax(selected)) > float(semantic_class_max):
        raise ValueError(
            "semantic_hms selected values must be in [{}, {}], got [{}, {}]".format(
                semantic_class_min,
                semantic_class_max,
                float(np.nanmin(selected)),
                float(np.nanmax(selected)),
            )
        )


def _normalise_mem_node_source(value: str) -> str:
    node_source = str(value).strip()
    if node_source not in SUPPORTED_MEM_NODE_SOURCES:
        raise ValueError(
            "MEM_NODE_SOURCE must be one of {}, got {!r}".format(SUPPORTED_MEM_NODE_SOURCES, node_source)
        )
    return node_source


def _normalise_mem_graph_target_scope(value: str) -> str:
    target_scope = str(value).strip()
    if target_scope not in SUPPORTED_MEM_GRAPH_TARGET_SCOPES:
        raise ValueError(
            "MEM_GRAPH_TARGET_SCOPE must be one of {}, got {!r}".format(
                SUPPORTED_MEM_GRAPH_TARGET_SCOPES,
                target_scope,
            )
        )
    return target_scope


def _validate_mem_node_target_pair(mem_node_source: str, mem_graph_target_scope: str) -> None:
    if mem_node_source == "gt" and mem_graph_target_scope == "gt_all":
        return
    if mem_node_source == "observed_instance_maps" and mem_graph_target_scope == "observed_induced":
        return
    raise ValueError(
        "unsupported MEM node/target combination: MEM_NODE_SOURCE={!r}, MEM_GRAPH_TARGET_SCOPE={!r}".format(
            mem_node_source,
            mem_graph_target_scope,
        )
    )


def _bbox_xyxy_abs_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("cannot compute bbox for an empty observed instance mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _majority_observed_semantic_class(
    selected_semantic_hms: np.ndarray,
    selected_instance_maps: np.ndarray,
    instance_id: int,
) -> Tuple[Optional[int], Dict[str, Any]]:
    labels = np.asarray(selected_semantic_hms)[np.asarray(selected_instance_maps) == int(instance_id)]
    labels = labels[np.isfinite(labels)]
    if labels.size == 0:
        return None, {"class_source": "observed_semantic_hms_majority", "num_label_pixels": 0}
    if not np.allclose(labels, np.round(labels), equal_nan=True):
        raise ValueError("semantic_hms labels for observed instance {} are not integer-like".format(instance_id))
    labels_int = np.round(labels).astype(np.int64)
    values, counts = np.unique(labels_int, return_counts=True)
    order = np.lexsort((values, -counts))
    majority = int(values[order[0]])
    return majority, {
        "class_source": "observed_semantic_hms_majority",
        "num_label_pixels": int(labels_int.size),
        "majority_count": int(counts[order[0]]),
    }


def build_observed_induced_mem_graph_contract(
    record: Dict[str, Any],
    observed_instance_maps: Optional[np.ndarray],
    semantic_hms: np.ndarray,
    selected_view_indices: Sequence[int],
    *,
    semantic_class_min: int = DEFAULT_SEMANTIC_CLASS_MIN,
    semantic_class_max: int = DEFAULT_SEMANTIC_CLASS_MAX,
) -> Dict[str, Any]:
    """Build Option 2B nodes and induced GT graph from observed visible instances.

    The first Option 2B contract uses exact MEM instance-id alignment: an
    observed instance id is target-alignable only when the same id exists in the
    full GT graph's node order. Unmatched observed ids are reported in metadata
    and excluded from the train/eval target because no GT graph row/column exists
    for them.
    """

    if observed_instance_maps is None:
        raise ValueError("Option 2B requires hms.npz field 'instance_maps'")
    observed_instance_maps = np.asarray(observed_instance_maps)
    semantic_hms = np.asarray(semantic_hms)
    selected = _coerce_int_list(selected_view_indices)
    if observed_instance_maps.ndim != 3:
        raise ValueError("observed instance_maps must have shape [V,H,W], got {}".format(observed_instance_maps.shape))
    if semantic_hms.ndim != 3:
        raise ValueError("semantic_hms must have shape [V,H,W], got {}".format(semantic_hms.shape))
    if observed_instance_maps.shape != semantic_hms.shape:
        raise ValueError(
            "observed instance_maps shape {} must match semantic_hms shape {}".format(
                observed_instance_maps.shape,
                semantic_hms.shape,
            )
        )
    invalid = [index for index in selected if index < 0 or index >= observed_instance_maps.shape[0]]
    if invalid:
        raise ValueError("selected_view_indices out of range for observed instance_maps: {}".format(invalid))

    gt_node_order = _coerce_int_list(record.get("node_order_instance_ids", []))
    gt_index_by_instance_id = {instance_id: index for index, instance_id in enumerate(gt_node_order)}
    if len(gt_index_by_instance_id) != len(gt_node_order):
        raise ValueError("GT node_order_instance_ids must be unique for observed-induced target alignment")
    full_graph = _as_graph_tensor(record.get("graph_gt", []), "graph_gt", len(gt_node_order))

    selected_instances = observed_instance_maps[selected]
    selected_semantic = semantic_hms[selected]
    finite_values = np.unique(selected_instances[np.isfinite(selected_instances)])
    observed_visible_ids: List[int] = []
    for raw_value in finite_values.tolist():
        if not np.isclose(raw_value, round(float(raw_value))):
            raise ValueError("observed instance id values must be integer-like, got {}".format(raw_value))
        instance_id = int(round(float(raw_value)))
        if instance_id > 0:
            observed_visible_ids.append(instance_id)
    observed_visible_ids = sorted(set(observed_visible_ids))

    object_class_max_exclusive = int(semantic_class_max)
    aligned_ids: List[int] = []
    aligned_source_gt_indices: List[int] = []
    bbox_xyxy_abs: List[List[int]] = []
    bbox_categories: List[int] = []
    observed_node_records: List[Dict[str, Any]] = []
    unmatched_observed_ids: List[int] = []
    invalid_class_observed_ids: List[int] = []

    for instance_id in observed_visible_ids:
        union_mask = np.any(selected_instances == int(instance_id), axis=0)
        observed_pixels = int(union_mask.sum())
        if instance_id not in gt_index_by_instance_id:
            unmatched_observed_ids.append(instance_id)
            continue
        semantic_class, class_metadata = _majority_observed_semantic_class(
            selected_semantic,
            selected_instances,
            instance_id,
        )
        if (
            semantic_class is None
            or int(semantic_class) < int(semantic_class_min)
            or int(semantic_class) >= object_class_max_exclusive
        ):
            invalid_class_observed_ids.append(instance_id)
            continue
        xyxy = _bbox_xyxy_abs_from_mask(union_mask)
        aligned_ids.append(instance_id)
        aligned_source_gt_indices.append(int(gt_index_by_instance_id[instance_id]))
        bbox_xyxy_abs.append(xyxy)
        bbox_categories.append(int(semantic_class))
        observed_node_records.append(
            {
                "node_index": len(aligned_ids) - 1,
                "observed_instance_id": int(instance_id),
                "aligned_gt_instance_id": int(instance_id),
                "source_gt_index": int(gt_index_by_instance_id[instance_id]),
                "bbox_xyxy_abs": xyxy,
                "semantic_class_id": int(semantic_class),
                "observed_visible_pixels": observed_pixels,
                **class_metadata,
            }
        )

    if aligned_source_gt_indices:
        source_index_tensor = torch.as_tensor(aligned_source_gt_indices, dtype=torch.long)
        induced_graph = full_graph.index_select(0, source_index_tensor).index_select(1, source_index_tensor)
    else:
        induced_graph = torch.zeros((0, 0), dtype=torch.long)

    visible_aligned_set = set(aligned_ids)
    visible_gt_set = {instance_id for instance_id in observed_visible_ids if instance_id in gt_index_by_instance_id}
    hidden_gt_ids = [instance_id for instance_id in gt_node_order if instance_id not in visible_gt_set]
    hidden_gt_indices = [int(gt_index_by_instance_id[instance_id]) for instance_id in hidden_gt_ids]
    dropped_gt_visible_ids = [instance_id for instance_id in gt_node_order if instance_id in visible_gt_set and instance_id not in visible_aligned_set]
    if aligned_source_gt_indices and hidden_gt_indices:
        aligned_index_tensor = torch.as_tensor(aligned_source_gt_indices, dtype=torch.long)
        hidden_index_tensor = torch.as_tensor(hidden_gt_indices, dtype=torch.long)
        visible_blocks_hidden_target = (
            full_graph.index_select(0, aligned_index_tensor)
            .index_select(1, hidden_index_tensor)
            .any(dim=1)
            .to(dtype=torch.float32)
        )
    else:
        visible_blocks_hidden_target = torch.zeros((len(aligned_source_gt_indices),), dtype=torch.float32)

    return {
        "node_order_instance_ids": aligned_ids,
        "bbox_xyxy_abs": bbox_xyxy_abs,
        "bbox_categories": bbox_categories,
        "graph_gt": induced_graph,
        "dense_gt": induced_graph.clone(),
        "visible_blocks_hidden_target": visible_blocks_hidden_target,
        "observed_node_records": observed_node_records,
        "metadata": {
            "observed_visible_instance_ids": observed_visible_ids,
            "gt_aligned_instance_ids": aligned_ids,
            "unmatched_observed_instance_ids": unmatched_observed_ids,
            "hidden_gt_instance_ids": hidden_gt_ids,
            "visible_gt_instance_ids": sorted(visible_gt_set),
            "dropped_visible_gt_instance_ids": dropped_gt_visible_ids,
            "invalid_class_observed_instance_ids": invalid_class_observed_ids,
            "observed_induced_source_gt_indices": aligned_source_gt_indices,
            "observed_node_records": observed_node_records,
            "visible_blocks_hidden_target": [
                int(value) for value in visible_blocks_hidden_target.to(dtype=torch.long).tolist()
            ],
            "num_visible_blocks_hidden_positive": int(visible_blocks_hidden_target.sum().item()),
            "num_observed_visible_instances": len(observed_visible_ids),
            "num_gt_aligned_instances": len(aligned_ids),
            "num_unmatched_observed_instances": len(unmatched_observed_ids),
            "num_hidden_gt_instances": len(hidden_gt_ids),
            "target_scope_note": "GT graph induced over observed visible nodes with exact instance-id alignment",
        },
    }


class MemObservedGtMapper:

    """Map one MEM Option 2A/2B record into D3G's mapper item structure."""

    @configurable
    def __init__(
        self,
        data_root: str,
        is_train: bool,
        graph_gt_type: str = "dense",
        observed_view_protocol: str = DEFAULT_OBSERVED_VIEW_PROTOCOL,
        expected_height: int = 140,
        expected_width: int = 200,
        max_selected_views: int = DEFAULT_MAX_SELECTED_VIEWS,
        semantic_class_min: int = DEFAULT_SEMANTIC_CLASS_MIN,
        semantic_class_max: int = DEFAULT_SEMANTIC_CLASS_MAX,
        validate_semantic_range: bool = True,
        mem_node_source: str = DEFAULT_MEM_NODE_SOURCE,
        mem_graph_target_scope: str = DEFAULT_MEM_GRAPH_TARGET_SCOPE,
    ) -> None:
        if graph_gt_type != "dense":
            raise ValueError("MemObservedGtMapper currently supports only INPUT.GRAPH_GT_TYPE='dense'")
        self.mem_node_source = _normalise_mem_node_source(mem_node_source)
        self.mem_graph_target_scope = _normalise_mem_graph_target_scope(mem_graph_target_scope)
        _validate_mem_node_target_pair(self.mem_node_source, self.mem_graph_target_scope)
        self.data_root = data_root
        self.is_train = is_train
        self.graph_gt_type = graph_gt_type
        self.observed_view_protocol = observed_view_protocol
        self.expected_height = int(expected_height) if expected_height is not None else 0
        self.expected_width = int(expected_width) if expected_width is not None else 0
        self.max_selected_views = int(max_selected_views)
        self.semantic_class_min = int(semantic_class_min)
        self.semantic_class_max = int(semantic_class_max)
        self.validate_semantic_range = bool(validate_semantic_range)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        return {
            "data_root": cfg.DATASETS.ROOT,
            "is_train": is_train,
            "graph_gt_type": cfg.INPUT.GRAPH_GT_TYPE,
            "observed_view_protocol": getattr(cfg.INPUT, "MEM_OBSERVED_VIEW_PROTOCOL", DEFAULT_OBSERVED_VIEW_PROTOCOL),
            "expected_height": getattr(cfg.INPUT, "MEM_EXPECTED_HEIGHT", 140),
            "expected_width": getattr(cfg.INPUT, "MEM_EXPECTED_WIDTH", 200),
            "max_selected_views": getattr(cfg.INPUT, "MEM_MAX_SELECTED_VIEWS", DEFAULT_MAX_SELECTED_VIEWS),
            "semantic_class_min": getattr(cfg.INPUT, "MEM_SEMANTIC_CLASS_MIN", DEFAULT_SEMANTIC_CLASS_MIN),
            "semantic_class_max": getattr(cfg.INPUT, "MEM_SEMANTIC_CLASS_MAX", DEFAULT_SEMANTIC_CLASS_MAX),
            "validate_semantic_range": getattr(cfg.INPUT, "MEM_VALIDATE_SEMANTIC_RANGE", True),
            "mem_node_source": getattr(cfg.INPUT, "MEM_NODE_SOURCE", DEFAULT_MEM_NODE_SOURCE),
            "mem_graph_target_scope": getattr(cfg.INPUT, "MEM_GRAPH_TARGET_SCOPE", DEFAULT_MEM_GRAPH_TARGET_SCOPE),
        }

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        record = normalise_mem_observed_gt_record(dataset_dict)
        pre_action_dir = _resolve_sample_dir(record.get("sample_dir", ""), self.data_root)
        hms, semantic_hms, observed_instance_maps = _load_mem_hms_npz(pre_action_dir)

        selected_view_indices = record.get("selected_view_indices") or []
        if not selected_view_indices:
            selected_view_indices = select_mem_observed_view_indices(hms.shape[0], self.observed_view_protocol)
        selected_view_indices = _coerce_int_list(selected_view_indices)
        if len(selected_view_indices) > self.max_selected_views:
            raise ValueError(
                "selected_view_indices count {} exceeds max_selected_views {}".format(
                    len(selected_view_indices), self.max_selected_views
                )
            )

        _validate_semantic_hms(
            semantic_hms,
            selected_view_indices,
            self.semantic_class_min,
            self.semantic_class_max,
            self.validate_semantic_range,
        )
        image_np, layout = materialize_mem_observed_tensor(hms, semantic_hms, selected_view_indices)
        if not np.isfinite(image_np).all():
            raise ValueError("materialized MEM image tensor contains NaN or Inf")

        channels, height, width = [int(dim) for dim in image_np.shape]
        if self.expected_height > 0 and height != self.expected_height:
            raise ValueError("materialized MEM height {} does not match expected {}".format(height, self.expected_height))
        if self.expected_width > 0 and width != self.expected_width:
            raise ValueError("materialized MEM width {} does not match expected {}".format(width, self.expected_width))
        if record.get("height") is not None and int(record["height"]) != height:
            raise ValueError("record height {} does not match materialized height {}".format(record["height"], height))
        if record.get("width") is not None and int(record["width"]) != width:
            raise ValueError("record width {} does not match materialized width {}".format(record["width"], width))

        observed_induced_metadata: Dict[str, Any] = {}
        if self.mem_node_source == "gt":
            classes = torch.as_tensor(_coerce_int_list(record.get("bbox_categories", [])), dtype=torch.long)
            num_nodes = int(classes.numel())
            node_order = _coerce_int_list(record.get("node_order_instance_ids", []))
            if node_order and len(node_order) != num_nodes:
                raise ValueError("node_order_instance_ids length must match bbox_categories length")
            boxes = _boxes_tensor(record.get("bbox_xyxy_abs", []), num_nodes, height, width)

            graph_gt = _as_graph_tensor(record.get("graph_gt", []), "graph_gt", num_nodes)
            dense_value = record.get("dense_gt")
            dense_is_empty = isinstance(dense_value, (list, tuple)) and len(dense_value) == 0
            if dense_value is None or dense_is_empty:
                dense_gt = graph_gt.clone()
            else:
                dense_gt = _as_graph_tensor(dense_value, "dense_gt", num_nodes)
                if not torch.equal(dense_gt, graph_gt):
                    raise ValueError("dense_gt must match graph_gt for the MEM-specific dense route")
            is_oracle_node_conditioned = bool(record.get("is_oracle_node_conditioned", True))
        else:
            observed_contract = build_observed_induced_mem_graph_contract(
                record,
                observed_instance_maps,
                semantic_hms,
                selected_view_indices,
                semantic_class_min=self.semantic_class_min,
                semantic_class_max=self.semantic_class_max,
            )
            node_order = _coerce_int_list(observed_contract["node_order_instance_ids"])
            classes = torch.as_tensor(_coerce_int_list(observed_contract["bbox_categories"]), dtype=torch.long)
            num_nodes = int(classes.numel())
            boxes = _boxes_tensor(observed_contract["bbox_xyxy_abs"], num_nodes, height, width)
            graph_gt = observed_contract["graph_gt"].to(dtype=torch.long)
            dense_gt = observed_contract["dense_gt"].to(dtype=torch.long)
            observed_induced_metadata = dict(observed_contract["metadata"])
            visible_blocks_hidden_target = observed_contract["visible_blocks_hidden_target"].to(dtype=torch.float32)
            is_oracle_node_conditioned = False

        instances = structures.Instances(
            image_size=(height, width),
            gt_boxes=structures.Boxes(boxes),
            gt_classes=classes,
        )
        sample_id = record.get("sample_id") or str(pre_action_dir)

        output = {
            "width": width,
            "height": height,
            "image": torch.from_numpy(image_np.copy()).float(),
            "instances": instances,
            "graph_gt": graph_gt,
            "dense_gt": dense_gt,
            "image_id": sample_id,
            "mem_metadata": {
                "schema": "mem_observed_gt_mapper_item_v0",
                "sample_id": sample_id,
                "sample_dir": str(pre_action_dir),
                "source_file": str(pre_action_dir / "hms.npz"),
                "selected_view_indices": selected_view_indices,
                "observed_view_protocol": self.observed_view_protocol,
                "flattened_input_shape_chw": [channels, height, width],
                "channel_layout": layout,
                "node_order_instance_ids": node_order,
                "mem_node_source": self.mem_node_source,
                "mem_graph_target_scope": self.mem_graph_target_scope,
                **observed_induced_metadata,
                "edge_type": record.get("edge_type", "blocks_access_to"),
                "graph_gt_convention": "graph_gt[i, j] = 1 means ordered MEM node i blocks_access_to ordered MEM node j",
                "direct_mem_dense_graph_gt": True,
                "is_oracle_node_conditioned": is_oracle_node_conditioned,
                "is_training_export": bool(record.get("is_training_export", False)),
                "is_full_dataset_export": bool(record.get("is_full_dataset_export", False)),
                "runs_model_forward": False,
                "runs_training": False,
            },
        }
        if self.mem_node_source == "observed_instance_maps":
            output["visible_blocks_hidden_target"] = visible_blocks_hidden_target
        return output


__all__ = [
    "DEFAULT_OBSERVED_VIEW_PROTOCOL",
    "DEFAULT_MEM_GRAPH_TARGET_SCOPE",
    "DEFAULT_MEM_NODE_SOURCE",
    "MemObservedGtMapper",
    "build_observed_induced_mem_graph_contract",
    "materialize_mem_observed_tensor",
    "select_mem_observed_view_indices",
]
