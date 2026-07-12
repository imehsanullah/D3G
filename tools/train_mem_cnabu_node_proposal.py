#!/usr/bin/env python3
"""Train/evaluate a compact CNABU-to-node proposal prototype.

This tool learns node proposals from saved CNABU belief maps. GT instance masks
are used only as offline training targets and evaluation references; they are
never passed to inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import socket
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment


THESIS_ROOT = Path("/home/user/ehsanullahm1/thesis")
D3G_ROOT = THESIS_ROOT / "D3G"
MEM_ROOT = THESIS_ROOT / "manipulation_enhanced_map_prediction"
THESIS_RECORDS_ROOT = THESIS_ROOT / "thesis_records"
if str(MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEM_ROOT))


def _load_mem_utility_module(module_name: str, relative_path: str) -> Any:
    """Load MEM utility modules without importing shelf_gym/__init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    for parent in ("shelf_gym", "shelf_gym.utils"):
        sys.modules.setdefault(parent, types.ModuleType(parent))
    path = MEM_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_CNABU_SCENE_GRAPH = _load_mem_utility_module(
    "shelf_gym.utils.cnabu_scene_graph",
    "shelf_gym/utils/cnabu_scene_graph.py",
)
_CNABU_SCENE_GRAPH_VIZ = _load_mem_utility_module(
    "shelf_gym.utils.cnabu_scene_graph_viz",
    "shelf_gym/utils/cnabu_scene_graph_viz.py",
)

DEFAULT_YCB_CLASS_NAMES = _CNABU_SCENE_GRAPH.DEFAULT_YCB_CLASS_NAMES
DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS = _CNABU_SCENE_GRAPH.DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS
DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS = _CNABU_SCENE_GRAPH.DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS
DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS = _CNABU_SCENE_GRAPH.DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS
decode_binary_mask_rle = _CNABU_SCENE_GRAPH.decode_binary_mask_rle
predict_scene_graph_from_cnabu = _CNABU_SCENE_GRAPH.predict_scene_graph_from_cnabu
DEFAULT_CLASS_PALETTE_BGR = _CNABU_SCENE_GRAPH_VIZ.DEFAULT_CLASS_PALETTE_BGR
build_cnabu_map_context = _CNABU_SCENE_GRAPH_VIZ.build_cnabu_map_context


SCHEMA = "mem_cnabu_node_proposal_train_eval_v0"
DEFAULT_CONFIG = D3G_ROOT / "configs" / "mem" / "cnabu_node_proposal_1000.yaml"


@dataclass(frozen=True)
class Node:
    id: int
    class_id: int
    class_name: str
    mask: np.ndarray
    bbox_xyxy_abs: List[int]
    centroid_yx: List[float]
    area: int
    score: float
    was_split: bool = False


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_by_dotted_key(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = [part for part in key.split(".") if part]
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def parse_override_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    result = json.loads(json.dumps(config))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        set_by_dotted_key(result, key, parse_override_value(value))
    return result


def read_records(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"]]
    raise ValueError(f"records JSON must be a list or object with records list: {path}")


def split_records(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {str(record.get("sample_id")): dict(record) for record in records}
    result: Dict[str, List[Dict[str, Any]]] = {}
    limits = {"train": max_train, "val": max_val, "test": max_test}
    for split_name in ("train", "val", "test"):
        ids = [str(value) for value in split_manifest.get(f"{split_name}_sample_ids", [])]
        if limits[split_name] is not None:
            ids = ids[: int(limits[split_name])]
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"{split_name} ids missing from records JSON: {missing[:10]}")
        result[split_name] = [dict(by_id[sample_id]) for sample_id in ids]
    check_scene_disjoint(result)
    return result


def check_scene_disjoint(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    scenes = {
        split: sorted({str(record["sample_id"]).split("/", 1)[0] for record in records})
        for split, records in records_by_split.items()
    }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(scenes[left]) & set(scenes[right]))
        if overlap:
            overlaps[f"{left}_{right}"] = overlap
    if overlaps:
        raise ValueError(f"split is not scene-disjoint: {overlaps}")


def sample_dir_from_record(record: Mapping[str, Any], raw_root: Path) -> Path:
    sample_dir = record.get("sample_dir") or record.get("pre_action_dir")
    if sample_dir:
        path = Path(str(sample_dir))
        return path if path.is_absolute() else raw_root / path
    return raw_root / str(record["sample_id"]) / "pre_action"


def cnabu_path_from_record(record: Mapping[str, Any], raw_root: Path, cnabu_root: Path) -> Path:
    value = record.get("cnabu_hms_path")
    if value:
        path = Path(str(value))
        return path if path.is_absolute() else cnabu_root / path
    return cnabu_root / "samples" / str(record["sample_id"]) / "pre_action" / "cnabu_hms.npz"


def bbox_xyxy_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def centroid_yx_from_mask(mask: np.ndarray) -> List[float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return [0.0, 0.0]
    return [float(ys.mean()), float(xs.mean())]


def class_name(class_id: int) -> str:
    if 0 <= int(class_id) < len(DEFAULT_YCB_CLASS_NAMES):
        return str(DEFAULT_YCB_CLASS_NAMES[int(class_id)])
    return f"class_{int(class_id)}"


def pad_crop(array: np.ndarray, *, raw_shape_hw: Tuple[int, int], crop_rows: Tuple[int, int]) -> np.ndarray:
    result = np.zeros(raw_shape_hw, dtype=np.float32)
    result[crop_rows[0]:crop_rows[1], :] = np.asarray(array, dtype=np.float32)
    return result


def load_cnabu_features(cnabu_path: Path, *, raw_shape_hw: Tuple[int, int]) -> np.ndarray:
    with np.load(cnabu_path, allow_pickle=False) as data:
        occupancy = np.asarray(data["occupancy_mean"], dtype=np.float32)
        semantic = np.asarray(data["semantic_mean"], dtype=np.float32)
        occ_epi = np.asarray(data["occupancy_epistemic"], dtype=np.float32) if "occupancy_epistemic" in data.files else None
        sem_vac = np.asarray(data["semantic_vacuity"], dtype=np.float32) if "semantic_vacuity" in data.files else None
        crop_rows_arr = np.asarray(data["crop_rows"], dtype=np.int64)
    crop_rows = (int(crop_rows_arr[0]), int(crop_rows_arr[1]))
    occupied = occupancy >= 0.5
    any_occ = occupied.any(axis=0)
    z_indices = np.arange(occupancy.shape[0], dtype=np.float32)[:, None, None]
    top = np.where(occupied, z_indices, -1.0).max(axis=0) / max(1.0, occupancy.shape[0] - 1.0)
    bottom_raw = np.where(occupied, z_indices, occupancy.shape[0] + 1.0).min(axis=0)
    bottom = np.where(any_occ, bottom_raw, 0.0) / max(1.0, occupancy.shape[0] - 1.0)
    channels: List[np.ndarray] = []
    for channel in semantic[:14]:
        channels.append(pad_crop(channel, raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
    channels.extend(
        [
            pad_crop(occupancy.max(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            pad_crop(occupancy.mean(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            pad_crop(occupied.mean(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            pad_crop(np.where(any_occ, top, 0.0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
            pad_crop(bottom, raw_shape_hw=raw_shape_hw, crop_rows=crop_rows),
        ]
    )
    if occ_epi is not None:
        channels.append(pad_crop(occ_epi.mean(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
        channels.append(pad_crop(occ_epi.max(axis=0), raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
    else:
        channels.extend([np.zeros(raw_shape_hw, dtype=np.float32), np.zeros(raw_shape_hw, dtype=np.float32)])
    if sem_vac is not None:
        channels.append(pad_crop(sem_vac, raw_shape_hw=raw_shape_hw, crop_rows=crop_rows))
    else:
        channels.append(np.zeros(raw_shape_hw, dtype=np.float32))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def majority_label(values: np.ndarray) -> Optional[int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    rounded = np.round(values).astype(np.int64)
    labels, counts = np.unique(rounded, return_counts=True)
    order = np.lexsort((labels, -counts))
    return int(labels[order[0]])


def load_gt_nodes(gt_path: Path, *, num_classes: int = 14) -> List[Node]:
    with np.load(gt_path, allow_pickle=False) as data:
        instance_maps = np.asarray(data["instance_maps"])
        semantic_2d = np.asarray(data["semantic_2d"])
    stack = instance_maps[None] if instance_maps.ndim == 2 else instance_maps
    nodes: List[Node] = []
    for value in np.unique(stack[np.isfinite(stack)]):
        instance_id = int(round(float(value)))
        if instance_id <= 0 or not np.isclose(value, instance_id):
            continue
        mask = np.any(stack == instance_id, axis=0)
        if int(mask.sum()) <= 0:
            continue
        class_id = majority_label(semantic_2d[mask])
        if class_id is None or not (0 <= int(class_id) < int(num_classes)):
            continue
        nodes.append(
            Node(
                id=int(instance_id),
                class_id=int(class_id),
                class_name=class_name(int(class_id)),
                mask=mask.astype(bool, copy=False),
                bbox_xyxy_abs=bbox_xyxy_from_mask(mask),
                centroid_yx=centroid_yx_from_mask(mask),
                area=int(mask.sum()),
                score=1.0,
            )
        )
    return nodes


def draw_disk(target: np.ndarray, y: float, x: float, radius: int) -> None:
    height, width = target.shape
    cy, cx = int(round(y)), int(round(x))
    y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
    x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
    for yy in range(y1, y2):
        for xx in range(x1, x2):
            dist = math.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            if dist <= radius:
                target[yy, xx] = max(float(target[yy, xx]), 1.0 - dist / float(radius + 1))


def build_targets(
    nodes: Sequence[Node],
    *,
    num_classes: int,
    shape_hw: Tuple[int, int],
    center_radius: int,
    offset_normalizer: float,
) -> Dict[str, np.ndarray]:
    fg = np.zeros((num_classes, shape_hw[0], shape_hw[1]), dtype=np.float32)
    center = np.zeros_like(fg)
    offset = np.zeros((num_classes * 2, shape_hw[0], shape_hw[1]), dtype=np.float32)
    for node in nodes:
        class_id = int(node.class_id)
        if not (0 <= class_id < num_classes):
            continue
        fg[class_id, node.mask] = 1.0
        cy, cx = node.centroid_yx
        draw_disk(center[class_id], cy, cx, int(center_radius))
        ys, xs = np.nonzero(node.mask)
        offset[class_id * 2, ys, xs] = (float(cy) - ys.astype(np.float32)) / float(offset_normalizer)
        offset[class_id * 2 + 1, ys, xs] = (float(cx) - xs.astype(np.float32)) / float(offset_normalizer)
    return {"foreground": fg, "center": center, "offset": offset}


class CnabuNodeProposalDataset:
    def __init__(self, records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]):
        self.records = [dict(record) for record in records]
        self.raw_root = Path(config["DATA"]["RAW_ROOT"])
        self.cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
        self.num_classes = int(config["DATA"]["NUM_CLASSES"])
        self.shape_hw = (int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"]))
        self.center_radius = int(config["TRAIN"]["CENTER_RADIUS"])
        self.offset_normalizer = float(config["MODEL"]["OFFSET_NORMALIZER"])
        self.cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index not in self.cache:
            record = self.records[index]
            sample_dir = sample_dir_from_record(record, self.raw_root)
            cnabu_path = cnabu_path_from_record(record, self.raw_root, self.cnabu_root)
            gt_path = sample_dir / "gt_hms.npz"
            features = load_cnabu_features(cnabu_path, raw_shape_hw=self.shape_hw)
            gt_nodes = load_gt_nodes(gt_path, num_classes=self.num_classes)
            targets = build_targets(
                gt_nodes,
                num_classes=self.num_classes,
                shape_hw=self.shape_hw,
                center_radius=self.center_radius,
                offset_normalizer=self.offset_normalizer,
            )
            self.cache[index] = {
                "sample_id": str(record["sample_id"]),
                "features": features,
                "foreground": targets["foreground"],
                "center": targets["center"],
                "offset": targets["offset"],
                "gt_nodes": gt_nodes,
                "cnabu_path": cnabu_path,
            }
        return self.cache[index]


class TinyNodeProposalNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, semantic_prior_logit_scale: float = 1.0):
        super().__init__()
        h = int(hidden_channels)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, h, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(h, h * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(h * 2, h * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(h * 2, h * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(h * 4, h * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(h * 4 + h * 2, h * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(h * 2 + h, h, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(h, num_classes * 4, 1)
        self.num_classes = int(num_classes)
        self.semantic_prior_logit_scale = float(semantic_prior_logit_scale)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        up2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        logits = self.head(d1)
        foreground_logits = logits[:, : self.num_classes]
        if self.semantic_prior_logit_scale:
            semantic_prior = torch.clamp(x[:, : self.num_classes], min=1e-4, max=1.0 - 1e-4)
            prior_logits = torch.logit(semantic_prior) * self.semantic_prior_logit_scale
            foreground_logits = foreground_logits + prior_logits
        return {
            "foreground_logits": foreground_logits,
            "center_logits": logits[:, self.num_classes : self.num_classes * 2],
            "offset": logits[:, self.num_classes * 2 :],
        }


def stack_batch(items: Sequence[Mapping[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.tensor(np.stack([item["features"] for item in items]), dtype=torch.float32, device=device),
        "foreground": torch.tensor(np.stack([item["foreground"] for item in items]), dtype=torch.float32, device=device),
        "center": torch.tensor(np.stack([item["center"] for item in items]), dtype=torch.float32, device=device),
        "offset": torch.tensor(np.stack([item["offset"] for item in items]), dtype=torch.float32, device=device),
    }


def compute_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    fg_pos = torch.tensor(float(config["TRAIN"]["FOREGROUND_POS_WEIGHT"]), device=batch["features"].device)
    center_pos = torch.tensor(float(config["TRAIN"]["CENTER_POS_WEIGHT"]), device=batch["features"].device)
    fg_loss = F.binary_cross_entropy_with_logits(
        output["foreground_logits"],
        batch["foreground"],
        pos_weight=fg_pos,
    )
    center_loss = F.binary_cross_entropy_with_logits(
        output["center_logits"],
        batch["center"],
        pos_weight=center_pos,
    )
    offset_mask = batch["foreground"].repeat_interleave(2, dim=1) > 0.5
    if bool(offset_mask.any()):
        offset_loss = F.smooth_l1_loss(output["offset"][offset_mask], batch["offset"][offset_mask])
    else:
        offset_loss = torch.zeros((), dtype=torch.float32, device=batch["features"].device)
    total = (
        fg_loss
        + float(config["TRAIN"]["CENTER_LOSS_WEIGHT"]) * center_loss
        + float(config["TRAIN"]["OFFSET_LOSS_WEIGHT"]) * offset_loss
    )
    return {
        "total": total,
        "foreground": fg_loss.detach(),
        "center": center_loss.detach(),
        "offset": offset_loss.detach(),
    }


def train_model(
    model: nn.Module,
    dataset: CnabuNodeProposalDataset,
    *,
    config: Mapping[str, Any],
    iterations: int,
    device: torch.device,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["TRAIN"]["LEARNING_RATE"]),
        weight_decay=float(config["TRAIN"]["WEIGHT_DECAY"]),
    )
    batch_size = int(config["TRAIN"]["BATCH_SIZE"])
    history: List[Dict[str, Any]] = []
    for step in range(1, int(iterations) + 1):
        indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
        batch = stack_batch([dataset[index] for index in indices], device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features"])
        losses = compute_loss(output, batch, config)
        losses["total"].backward()
        optimizer.step()
        if step == 1 or step == int(iterations) or step % max(1, int(config["TRAIN"]["EVAL_PERIOD"])) == 0:
            history.append(
                {
                    "step": int(step),
                    "loss": float(losses["total"].detach().cpu().item()),
                    "foreground_loss": float(losses["foreground"].cpu().item()),
                    "center_loss": float(losses["center"].cpu().item()),
                    "offset_loss": float(losses["offset"].cpu().item()),
                }
            )
    return history


def infer_nodes(
    output: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any],
    sample_index: int = 0,
) -> List[Node]:
    fg_probs = torch.sigmoid(output["foreground_logits"][sample_index]).detach().cpu().numpy()
    center_probs = torch.sigmoid(output["center_logits"][sample_index]).detach().cpu().numpy()
    offset_pred = output.get("offset")
    offsets = offset_pred[sample_index].detach().cpu().numpy() if offset_pred is not None else None
    min_pixels = int(config["INFERENCE"]["MIN_NODE_PIXELS"])
    max_centers = int(config["INFERENCE"]["MAX_CENTERS_PER_CLASS"])
    center_nms_radius = int(config["INFERENCE"].get("CENTER_NMS_RADIUS", 8))
    local_max_kernel = int(config["INFERENCE"].get("CENTER_LOCAL_MAX_KERNEL", 9))
    offset_normalizer = float(config["MODEL"]["OFFSET_NORMALIZER"])
    nodes: List[Node] = []
    next_id = 1
    for class_id in range(int(config["DATA"]["NUM_CLASSES"])):
        fg_threshold = class_specific_float(
            config,
            "INFERENCE",
            "CLASS_FOREGROUND_THRESHOLDS",
            int(class_id),
            float(config["INFERENCE"]["FOREGROUND_THRESHOLD"]),
        )
        center_threshold = class_specific_float(
            config,
            "INFERENCE",
            "CLASS_CENTER_THRESHOLDS",
            int(class_id),
            float(config["INFERENCE"]["CENTER_THRESHOLD"]),
        )
        fg_mask = fg_probs[class_id] >= fg_threshold
        if int(fg_mask.sum()) < min_pixels:
            continue
        center_map = center_probs[class_id]
        local_max_kernel = max(3, local_max_kernel | 1)
        pooled = cv2.dilate(
            center_map.astype(np.float32),
            np.ones((local_max_kernel, local_max_kernel), dtype=np.uint8),
        )
        peak_mask = (center_map >= center_threshold) & (center_map >= pooled - 1e-6) & fg_mask
        ys, xs = np.nonzero(peak_mask)
        if ys.size == 0:
            center_candidate_mask = (center_map >= center_threshold) & fg_mask
            num_center_labels, center_labels = cv2.connectedComponents(
                center_candidate_mask.astype(np.uint8),
                connectivity=4,
            )
            for label in range(1, num_center_labels):
                component = center_labels == label
                comp_y, comp_x = np.nonzero(component)
                if comp_y.size == 0:
                    continue
                comp_scores = center_map[comp_y, comp_x]
                best = int(np.argmax(comp_scores))
                ys = np.append(ys, comp_y[best])
                xs = np.append(xs, comp_x[best])
        if ys.size == 0:
            continue
        centers: List[Tuple[int, int, float]] = []
        scores = center_map[ys, xs]
        for idx in np.argsort(scores)[::-1].tolist():
            centers.append((int(ys[idx]), int(xs[idx]), float(scores[idx])))
        centers.sort(key=lambda item: item[2], reverse=True)
        pruned_centers: List[Tuple[int, int, float]] = []
        min_dist_sq = max(1, center_nms_radius) ** 2
        for y, x, score in centers:
            if all((y - cy) ** 2 + (x - cx) ** 2 >= min_dist_sq for cy, cx, _ in pruned_centers):
                pruned_centers.append((y, x, score))
            if len(pruned_centers) >= max_centers:
                break
        centers = pruned_centers
        if not centers:
            continue
        if len(centers) == 1:
            region = keep_component_containing(fg_mask, centers[0][0], centers[0][1])
            area = int(region.sum())
            if area >= min_pixels:
                node_score = float(0.5 * centers[0][2] + 0.5 * float(fg_probs[class_id][region].mean()))
                nodes.append(
                    Node(
                        id=next_id,
                        class_id=int(class_id),
                        class_name=class_name(int(class_id)),
                        mask=region,
                        bbox_xyxy_abs=bbox_xyxy_from_mask(region),
                        centroid_yx=centroid_yx_from_mask(region),
                        area=area,
                        score=node_score,
                    )
                )
                next_id += 1
            continue
        pix_y, pix_x = np.nonzero(fg_mask)
        center_array = np.asarray([[cy, cx] for cy, cx, _ in centers], dtype=np.float32)
        if offsets is not None:
            est_y = pix_y.astype(np.float32) + offsets[class_id * 2, pix_y, pix_x] * offset_normalizer
            est_x = pix_x.astype(np.float32) + offsets[class_id * 2 + 1, pix_y, pix_x] * offset_normalizer
            assignment_points = np.stack([est_y, est_x], axis=1).astype(np.float32)
        else:
            assignment_points = np.stack([pix_y, pix_x], axis=1).astype(np.float32)
        distances = ((assignment_points[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        for center_index, (cy, cx, center_score) in enumerate(centers):
            region = np.zeros_like(fg_mask, dtype=bool)
            selected = assignments == center_index
            region[pix_y[selected], pix_x[selected]] = True
            region = keep_component_containing(region, cy, cx)
            area = int(region.sum())
            if area < min_pixels:
                continue
            node_score = float(0.5 * center_score + 0.5 * float(fg_probs[class_id][region].mean()))
            nodes.append(
                Node(
                    id=next_id,
                    class_id=int(class_id),
                    class_name=class_name(int(class_id)),
                    mask=region,
                    bbox_xyxy_abs=bbox_xyxy_from_mask(region),
                    centroid_yx=centroid_yx_from_mask(region),
                    area=area,
                    score=node_score,
                )
            )
            next_id += 1
    return postprocess_learned_nodes(nodes, config)


def keep_component_containing(mask: np.ndarray, y: int, x: int) -> np.ndarray:
    _num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return mask
    label = int(labels[y, x])
    if label <= 0:
        return mask
    return labels == label


def class_specific_float(
    config: Mapping[str, Any],
    section_name: str,
    key: str,
    class_id: int,
    default: float,
) -> float:
    value = config.get(section_name, {}).get(key, default)
    if isinstance(value, Mapping):
        if str(class_id) in value:
            return float(value[str(class_id)])
        name = class_name(class_id)
        if name in value:
            return float(value[name])
        return float(default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if 0 <= int(class_id) < len(value):
            return float(value[int(class_id)])
        return float(default)
    if value is None:
        return float(default)
    return float(value)


def class_prior_area(class_id: int) -> float:
    if 0 <= int(class_id) < len(DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS):
        return float(DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS[int(class_id)])
    return 250.0


def class_prior_width(class_id: int) -> float:
    if 0 <= int(class_id) < len(DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS):
        return float(DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS[int(class_id)])
    return 22.0


def class_prior_height(class_id: int) -> float:
    if 0 <= int(class_id) < len(DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS):
        return float(DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS[int(class_id)])
    return 22.0


def node_bbox_wh(node: Node) -> Tuple[int, int]:
    x1, y1, x2, y2 = node.bbox_xyxy_abs
    return max(0, int(x2) - int(x1)), max(0, int(y2) - int(y1))


def make_node(
    *,
    node_id: int,
    class_id: int,
    mask: np.ndarray,
    score: float,
    was_split: bool = False,
) -> Node:
    clean_mask = np.asarray(mask, dtype=bool)
    return Node(
        id=int(node_id),
        class_id=int(class_id),
        class_name=class_name(int(class_id)),
        mask=clean_mask,
        bbox_xyxy_abs=bbox_xyxy_from_mask(clean_mask),
        centroid_yx=centroid_yx_from_mask(clean_mask),
        area=int(clean_mask.sum()),
        score=float(score),
        was_split=bool(was_split),
    )


def renumber_nodes(nodes: Sequence[Node]) -> List[Node]:
    return [
        make_node(
            node_id=index + 1,
            class_id=node.class_id,
            mask=node.mask,
            score=node.score,
            was_split=node.was_split,
        )
        for index, node in enumerate(nodes)
    ]


def node_area_passes_prior(node: Node, config: Mapping[str, Any]) -> bool:
    cfg = config.get("CONSERVATIVE", {})
    if not bool(cfg.get("CLASS_SIZE_PRIOR_REJECTION", False)):
        return True
    prior = class_prior_area(int(node.class_id))
    min_mult = float(cfg.get("MIN_AREA_PRIOR_MULTIPLIER", 0.15))
    max_mult = float(cfg.get("MAX_AREA_PRIOR_MULTIPLIER", 2.75))
    return int(node.area) >= prior * min_mult and int(node.area) <= prior * max_mult


def suppress_overlapping_nodes(nodes: Sequence[Node], overlap_threshold: float) -> List[Node]:
    if float(overlap_threshold) <= 0.0:
        return list(nodes)
    kept: List[Node] = []
    ordered = sorted(nodes, key=lambda node: (float(node.score), int(node.area)), reverse=True)
    for node in ordered:
        suppress = False
        for kept_node in kept:
            if int(node.class_id) != int(kept_node.class_id):
                continue
            inter = int(np.logical_and(node.mask, kept_node.mask).sum())
            denom = max(1, min(int(node.area), int(kept_node.area)))
            if inter / denom >= float(overlap_threshold):
                suppress = True
                break
        if not suppress:
            kept.append(node)
    return kept


def postprocess_learned_nodes(nodes: Sequence[Node], config: Mapping[str, Any]) -> List[Node]:
    cfg = config.get("CONSERVATIVE", {})
    if not bool(cfg.get("POSTPROCESS_ENABLED", False)):
        return list(nodes)
    min_score_default = float(config.get("INFERENCE", {}).get("MIN_NODE_SCORE", 0.0))
    min_pixels_default = float(config.get("INFERENCE", {}).get("MIN_NODE_PIXELS", 1))
    filtered = []
    for node in nodes:
        min_score = class_specific_float(config, "INFERENCE", "CLASS_MIN_NODE_SCORES", int(node.class_id), min_score_default)
        min_pixels = class_specific_float(config, "INFERENCE", "CLASS_MIN_NODE_PIXELS", int(node.class_id), min_pixels_default)
        if float(node.score) < min_score:
            continue
        if int(node.area) < int(round(min_pixels)):
            continue
        if not node_area_passes_prior(node, config):
            continue
        filtered.append(node)
    filtered = suppress_overlapping_nodes(
        filtered,
        float(cfg.get("MASK_OVERLAP_SUPPRESSION_THRESHOLD", 0.0)),
    )
    return renumber_nodes(filtered)


def clipped_child_node(
    parent: Node,
    child: Node,
    node_id: int,
    config: Mapping[str, Any],
    *,
    enforce_prior: bool = True,
) -> Optional[Node]:
    cfg = config.get("CONSERVATIVE", {})
    inter_mask = np.logical_and(parent.mask, child.mask)
    inter = int(inter_mask.sum())
    if inter <= 0:
        return None
    child_inside = inter / max(1, int(child.area))
    parent_fraction = inter / max(1, int(parent.area))
    if child_inside < float(cfg.get("CHILD_INSIDE_PARENT_MIN_FRACTION", 0.70)):
        return None
    if parent_fraction < float(cfg.get("CHILD_PARENT_MIN_FRACTION", 0.025)):
        return None
    candidate = make_node(
        node_id=node_id,
        class_id=int(child.class_id),
        mask=inter_mask,
        score=float(child.score),
        was_split=True,
    )
    if enforce_prior and not node_area_passes_prior(candidate, config):
        return None
    return candidate


def parent_is_suspicious(parent: Node, children: Sequence[Node], config: Mapping[str, Any]) -> bool:
    cfg = config.get("CONSERVATIVE", {})
    if not bool(cfg.get("SUSPICIOUS_PARENT_ONLY", True)):
        return True
    class_id = int(parent.class_id)
    width, height = node_bbox_wh(parent)
    if int(parent.area) >= class_prior_area(class_id) * float(cfg.get("SUSPICIOUS_AREA_PRIOR_MULTIPLIER", 1.45)):
        return True
    if width >= class_prior_width(class_id) * float(cfg.get("SUSPICIOUS_BBOX_PRIOR_MULTIPLIER", 1.45)):
        return True
    if height >= class_prior_height(class_id) * float(cfg.get("SUSPICIOUS_BBOX_PRIOR_MULTIPLIER", 1.45)):
        return True
    if bool(cfg.get("ALLOW_MULTI_CHILD_SUSPICION", True)) and len(children) >= int(cfg.get("MIN_CHILDREN_FOR_SPLIT", 2)):
        return True
    return False


def children_inside_parent(
    parent: Node,
    source_nodes: Sequence[Node],
    config: Mapping[str, Any],
    *,
    enforce_prior: bool,
) -> List[Node]:
    children: List[Node] = []
    for child in source_nodes:
        if int(child.class_id) != int(parent.class_id):
            continue
        clipped = clipped_child_node(
            parent,
            child,
            len(children) + 1,
            config,
            enforce_prior=enforce_prior,
        )
        if clipped is not None:
            children.append(clipped)
    return children


def candidate_gated_nodes(
    parent_nodes: Sequence[Node],
    learned_nodes: Sequence[Node],
    config: Mapping[str, Any],
    *,
    guard_nodes: Optional[Sequence[Node]] = None,
) -> List[Node]:
    cfg = config.get("CONSERVATIVE", {})
    use_2d_guard = bool(cfg.get("USE_2D_SPLIT_GUARD", False))
    fallback_to_guard = bool(cfg.get("FALLBACK_TO_2D_GUARD", True))
    max_extra_children = int(cfg.get("MAX_EXTRA_CHILDREN_OVER_GUARD", 0))
    min_children = int(cfg.get("MIN_CHILDREN_FOR_SPLIT", 2))
    min_coverage = float(cfg.get("MIN_COMBINED_CHILD_PARENT_COVERAGE", 0.35))
    max_children = int(cfg.get("MAX_CHILDREN_PER_PARENT", 5))
    result: List[Node] = []
    next_id = 1
    for parent in parent_nodes:
        guard_children = children_inside_parent(
            parent,
            guard_nodes or (),
            config,
            enforce_prior=False,
        )
        guard_children = suppress_overlapping_nodes(
            guard_children,
            float(cfg.get("GUARD_CHILD_OVERLAP_SUPPRESSION_THRESHOLD", 0.80)),
        )
        guard_split = len(guard_children) >= min_children
        if use_2d_guard and not guard_split:
            result.append(
                make_node(
                    node_id=next_id,
                    class_id=int(parent.class_id),
                    mask=parent.mask,
                    score=float(parent.score),
                    was_split=bool(parent.was_split),
                )
            )
            next_id += 1
            continue
        children = children_inside_parent(parent, learned_nodes, config, enforce_prior=True)
        children = suppress_overlapping_nodes(
            children,
            float(cfg.get("CHILD_OVERLAP_SUPPRESSION_THRESHOLD", 0.60)),
        )
        children = sorted(children, key=lambda node: (node.score, node.area), reverse=True)[:max_children]
        combined_mask = np.zeros_like(parent.mask, dtype=bool)
        for child in children:
            combined_mask |= child.mask
        combined_coverage = int(combined_mask.sum()) / max(1, int(parent.area))
        accept_split = (
            len(children) >= min_children
            and combined_coverage >= min_coverage
            and parent_is_suspicious(parent, children, config)
        )
        if use_2d_guard:
            accept_split = accept_split and len(children) <= len(guard_children) + max_extra_children
        if accept_split:
            for child in children:
                result.append(
                    make_node(
                        node_id=next_id,
                        class_id=int(child.class_id),
                        mask=child.mask,
                        score=float(child.score),
                        was_split=True,
                    )
                )
                next_id += 1
        elif use_2d_guard and fallback_to_guard and guard_split:
            for guard_child in sorted(guard_children, key=lambda node: (node.score, node.area), reverse=True):
                result.append(
                    make_node(
                        node_id=next_id,
                        class_id=int(guard_child.class_id),
                        mask=guard_child.mask,
                        score=float(guard_child.score),
                        was_split=True,
                    )
                )
                next_id += 1
        else:
            result.append(
                make_node(
                    node_id=next_id,
                    class_id=int(parent.class_id),
                    mask=parent.mask,
                    score=float(parent.score),
                    was_split=bool(parent.was_split),
                )
            )
            next_id += 1
    return result


def pairwise_iou(preds: Sequence[Node], refs: Sequence[Node]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iou = np.zeros((len(preds), len(refs)), dtype=np.float32)
    pred_fraction = np.zeros_like(iou)
    ref_fraction = np.zeros_like(iou)
    for pred_index, pred in enumerate(preds):
        pred_area = max(int(pred.mask.sum()), 1)
        for ref_index, ref in enumerate(refs):
            if int(pred.class_id) != int(ref.class_id):
                continue
            ref_area = max(int(ref.mask.sum()), 1)
            inter = int(np.logical_and(pred.mask, ref.mask).sum())
            if inter <= 0:
                continue
            union = int(np.logical_or(pred.mask, ref.mask).sum())
            iou[pred_index, ref_index] = float(inter / union) if union else 0.0
            pred_fraction[pred_index, ref_index] = float(inter / pred_area)
            ref_fraction[pred_index, ref_index] = float(inter / ref_area)
    return iou, pred_fraction, ref_fraction


def match_nodes(iou: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    if iou.size == 0 or iou.shape[0] == 0 or iou.shape[1] == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    matches: List[Tuple[int, int, float]] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        value = float(iou[row, col])
        if value >= float(threshold):
            matches.append((int(row), int(col), value))
    return matches


def evaluate_nodes(
    preds: Sequence[Node],
    refs: Sequence[Node],
    *,
    thresholds: Sequence[float],
    overlap_fraction_threshold: float,
) -> Dict[str, Any]:
    iou, _pred_fraction, ref_fraction = pairwise_iou(preds, refs)
    merge_by_pred = (ref_fraction >= float(overlap_fraction_threshold)).sum(axis=1) if refs else np.zeros(len(preds))
    over_by_ref = (ref_fraction >= float(overlap_fraction_threshold)).sum(axis=0) if preds else np.zeros(len(refs))
    threshold_metrics: Dict[str, Any] = {}
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        matches = match_nodes(iou, float(threshold))
        matched = len(matches)
        precision = float(matched / len(preds)) if preds else 0.0
        recall = float(matched / len(refs)) if refs else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        threshold_metrics[key] = {
            "matched": int(matched),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": max(0, len(preds) - matched),
            "missed_gt": max(0, len(refs) - matched),
            "matches": [{"pred_index": row, "gt_index": col, "iou": value} for row, col, value in matches],
        }
    class_wise: Dict[str, Any] = {}
    class_ids = sorted({int(node.class_id) for node in preds} | {int(node.class_id) for node in refs})
    for class_id in class_ids:
        pred_indices = [idx for idx, node in enumerate(preds) if int(node.class_id) == class_id]
        ref_indices = [idx for idx, node in enumerate(refs) if int(node.class_id) == class_id]
        class_iou = iou[np.ix_(pred_indices, ref_indices)] if pred_indices and ref_indices else np.zeros((len(pred_indices), len(ref_indices)))
        class_thresholds = {}
        for threshold in thresholds:
            key = f"{float(threshold):.2f}"
            matched = len(match_nodes(class_iou, float(threshold)))
            precision = float(matched / len(pred_indices)) if pred_indices else 0.0
            recall = float(matched / len(ref_indices)) if ref_indices else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
            class_thresholds[key] = {"matched": matched, "precision": precision, "recall": recall, "f1": f1}
        class_wise[str(class_id)] = {
            "class_id": int(class_id),
            "class_name": class_name(class_id),
            "pred_count": int(len(pred_indices)),
            "gt_count": int(len(ref_indices)),
            "merge_indicators": int(np.sum(merge_by_pred[pred_indices] >= 2)) if pred_indices else 0,
            "over_split_indicators": int(np.sum(over_by_ref[ref_indices] >= 2)) if ref_indices else 0,
            "thresholds": class_thresholds,
        }
    return {
        "pred_count": int(len(preds)),
        "gt_count": int(len(refs)),
        "merge_indicators": int(np.sum(merge_by_pred >= 2)),
        "over_split_indicators": int(np.sum(over_by_ref >= 2)),
        "thresholds": threshold_metrics,
        "class_wise": class_wise,
    }


def aggregate_evaluations(items: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> Dict[str, Any]:
    pred_count = int(sum(int(item["pred_count"]) for item in items))
    gt_count = int(sum(int(item["gt_count"]) for item in items))
    result: Dict[str, Any] = {
        "samples": int(len(items)),
        "pred_count": pred_count,
        "gt_count": gt_count,
        "merge_indicators": int(sum(int(item["merge_indicators"]) for item in items)),
        "over_split_indicators": int(sum(int(item["over_split_indicators"]) for item in items)),
        "thresholds": {},
        "class_wise": {},
    }
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        matched = int(sum(int(item["thresholds"][key]["matched"]) for item in items))
        precision = float(matched / pred_count) if pred_count else 0.0
        recall = float(matched / gt_count) if gt_count else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        result["thresholds"][key] = {
            "matched": matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": int(sum(int(item["thresholds"][key]["false_positives"]) for item in items)),
            "missed_gt": int(sum(int(item["thresholds"][key]["missed_gt"]) for item in items)),
        }
    class_ids = sorted({class_id for item in items for class_id in item["class_wise"].keys()}, key=int)
    for class_id in class_ids:
        class_items = [item["class_wise"][class_id] for item in items if class_id in item["class_wise"]]
        pred = int(sum(int(item["pred_count"]) for item in class_items))
        gt = int(sum(int(item["gt_count"]) for item in class_items))
        class_summary = {
            "class_id": int(class_id),
            "class_name": class_name(int(class_id)),
            "pred_count": pred,
            "gt_count": gt,
            "merge_indicators": int(sum(int(item["merge_indicators"]) for item in class_items)),
            "over_split_indicators": int(sum(int(item["over_split_indicators"]) for item in class_items)),
            "thresholds": {},
        }
        for threshold in thresholds:
            key = f"{float(threshold):.2f}"
            matched = int(sum(int(item["thresholds"][key]["matched"]) for item in class_items))
            precision = float(matched / pred) if pred else 0.0
            recall = float(matched / gt) if gt else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
            class_summary["thresholds"][key] = {"matched": matched, "precision": precision, "recall": recall, "f1": f1}
        result["class_wise"][class_id] = class_summary
    return result


def nodes_from_graph(graph: Mapping[str, Any]) -> List[Node]:
    nodes: List[Node] = []
    for node in graph.get("nodes", []):
        encoded = node.get("mask")
        if not encoded:
            continue
        mask = decode_binary_mask_rle(encoded).astype(bool, copy=False)
        nodes.append(
            Node(
                id=int(node["id"]),
                class_id=int(node["class_id"]),
                class_name=str(node.get("class_name", class_name(int(node["class_id"])))),
                mask=mask,
                bbox_xyxy_abs=[int(value) for value in node["bbox_xyxy_abs"]],
                centroid_yx=[float(value) for value in node["centroid_yx"]],
                area=int(node.get("area_pixels", mask.sum())),
                score=float(node.get("score", 1.0)),
                was_split=bool(node.get("was_split", False)),
            )
        )
    return nodes


def rule_nodes_for_record(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    split_config: Mapping[str, Any],
) -> List[Node]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
    graph = predict_scene_graph_from_cnabu(
        cnabu_path=cnabu_path_from_record(record, raw_root, cnabu_root),
        component_split_config=split_config,
        edge_config={"opening_side": "low"},
        include_masks=True,
    )
    return nodes_from_graph(graph)


def evaluate_rule_baseline(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    split_config: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
    thresholds = [float(value) for value in config["EVAL"]["IOU_THRESHOLDS"]]
    overlap_threshold = float(config["EVAL"]["OVERLAP_FRACTION_THRESHOLD"])
    sample_metrics = []
    for record in records:
        sample_dir = sample_dir_from_record(record, raw_root)
        gt_nodes = load_gt_nodes(sample_dir / "gt_hms.npz", num_classes=int(config["DATA"]["NUM_CLASSES"]))
        pred_nodes = rule_nodes_for_record(
            record,
            config=config,
            split_config=split_config,
        )
        sample_metrics.append(
            evaluate_nodes(
                pred_nodes,
                gt_nodes,
                thresholds=thresholds,
                overlap_fraction_threshold=overlap_threshold,
            )
        )
    return aggregate_evaluations(sample_metrics, thresholds)


def evaluate_learned_model(
    model: nn.Module,
    dataset: CnabuNodeProposalDataset,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    variant_mode: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    thresholds = [float(value) for value in config["EVAL"]["IOU_THRESHOLDS"]]
    overlap_threshold = float(config["EVAL"]["OVERLAP_FRACTION_THRESHOLD"])
    sample_results: List[Dict[str, Any]] = []
    sample_metrics: List[Mapping[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            features = torch.tensor(item["features"][None], dtype=torch.float32, device=device)
            output = model(features)
            raw_pred_nodes = infer_nodes(output, config=config, sample_index=0)
            mode = str(variant_mode or config.get("CONSERVATIVE", {}).get("MODE", "global"))
            if mode == "candidate_gated":
                if records is None:
                    raise ValueError("records are required for candidate_gated learned evaluation")
                parent_nodes = rule_nodes_for_record(
                    records[index],
                    config=config,
                    split_config={"enabled": False},
                )
                guard_nodes = None
                if bool(config.get("CONSERVATIVE", {}).get("USE_2D_SPLIT_GUARD", False)):
                    guard_nodes = rule_nodes_for_record(
                        records[index],
                        config=config,
                        split_config={"enabled": True, "method": "candidate_gated_2d_footprint"},
                    )
                pred_nodes = candidate_gated_nodes(parent_nodes, raw_pred_nodes, config, guard_nodes=guard_nodes)
            elif mode == "global":
                pred_nodes = raw_pred_nodes
            else:
                raise ValueError(f"unknown learned evaluation mode: {mode}")
            gt_nodes = item["gt_nodes"]
            metrics = evaluate_nodes(
                pred_nodes,
                gt_nodes,
                thresholds=thresholds,
                overlap_fraction_threshold=overlap_threshold,
            )
            sample_metrics.append(metrics)
            sample_results.append(
                {
                    "sample_id": item["sample_id"],
                    "metrics": metrics,
                    "pred_nodes": pred_nodes,
                    "raw_pred_nodes": raw_pred_nodes,
                    "gt_nodes": gt_nodes,
                    "cnabu_path": str(item["cnabu_path"]),
                }
            )
    return aggregate_evaluations(sample_metrics, thresholds), sample_results


def tensor_to_nodes_from_target(target: np.ndarray) -> List[Node]:
    nodes: List[Node] = []
    node_id = 1
    for class_id in range(target.shape[0]):
        num, labels = cv2.connectedComponents((target[class_id] > 0.5).astype(np.uint8), connectivity=4)
        for label in range(1, num):
            mask = labels == label
            if int(mask.sum()) <= 0:
                continue
            nodes.append(
                Node(
                    id=node_id,
                    class_id=class_id,
                    class_name=class_name(class_id),
                    mask=mask,
                    bbox_xyxy_abs=bbox_xyxy_from_mask(mask),
                    centroid_yx=centroid_yx_from_mask(mask),
                    area=int(mask.sum()),
                    score=1.0,
                )
            )
            node_id += 1
    return nodes


def synthetic_merge_sample(config: Mapping[str, Any]) -> Dict[str, Any]:
    h, w = int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"])
    num_classes = int(config["DATA"]["NUM_CLASSES"])
    features = np.zeros((int(config["MODEL"]["IN_CHANNELS"]), h, w), dtype=np.float32)
    class_id = 9
    features[class_id, 45:80, 58:82] = 0.95
    features[class_id, 45:80, 85:109] = 0.95
    features[class_id, 57:68, 80:87] = 0.75
    features[14, 45:80, 58:109] = 0.95
    features[15, 45:80, 58:109] = 0.70
    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)
    mask_a[45:80, 58:82] = True
    mask_b[45:80, 85:109] = True
    nodes = [
        Node(1, class_id, class_name(class_id), mask_a, bbox_xyxy_from_mask(mask_a), centroid_yx_from_mask(mask_a), int(mask_a.sum()), 1.0),
        Node(2, class_id, class_name(class_id), mask_b, bbox_xyxy_from_mask(mask_b), centroid_yx_from_mask(mask_b), int(mask_b.sum()), 1.0),
    ]
    targets = build_targets(
        nodes,
        num_classes=num_classes,
        shape_hw=(h, w),
        center_radius=int(config["TRAIN"]["CENTER_RADIUS"]),
        offset_normalizer=float(config["MODEL"]["OFFSET_NORMALIZER"]),
    )
    return {
        "sample_id": "synthetic_merge",
        "features": features,
        "foreground": targets["foreground"],
        "center": targets["center"],
        "offset": targets["offset"],
        "gt_nodes": nodes,
        "cnabu_path": "",
    }


class ListDataset:
    def __init__(self, items: Sequence[Mapping[str, Any]]):
        self.items = [dict(item) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def run_synthetic_overfit(config: Mapping[str, Any], *, device: torch.device) -> Dict[str, Any]:
    sample = synthetic_merge_sample(config)
    dataset = ListDataset([sample])
    model = TinyNodeProposalNet(
        int(config["MODEL"]["IN_CHANNELS"]),
        int(config["MODEL"]["HIDDEN_CHANNELS"]),
        int(config["DATA"]["NUM_CLASSES"]),
        float(config["MODEL"].get("SEMANTIC_PRIOR_LOGIT_SCALE", 1.0)),
    ).to(device)
    history = train_model(
        model,
        dataset,
        config=config,
        iterations=int(config["TRAIN"]["SYNTHETIC_OVERFIT_ITERATIONS"]),
        device=device,
        seed=int(config["TRAIN"]["SEED"]),
    )
    metrics, samples = evaluate_learned_model(model, dataset, config=config, device=device)
    return {"history": history, "metrics": metrics, "num_pred_nodes": samples[0]["metrics"]["pred_count"]}


def render_panel(
    *,
    output_path: Path,
    sample_id: str,
    background_bgr: np.ndarray,
    columns: Sequence[Tuple[str, Sequence[Node]]],
) -> None:
    scale = 4
    panels = []
    for title, nodes in columns:
        image = cv2.resize(background_bgr, (background_bgr.shape[1] * scale, background_bgr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        for node in nodes:
            mask = cv2.resize(node.mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            color = np.asarray(DEFAULT_CLASS_PALETTE_BGR[min(node.class_id, len(DEFAULT_CLASS_PALETTE_BGR) - 1)], dtype=np.float32)
            overlay[mask] = np.clip(overlay[mask].astype(np.float32) * 0.60 + color * 0.40, 0, 255)
        image = cv2.addWeighted(overlay, 0.80, image, 0.20, 0.0)
        for node in nodes:
            x1, y1, x2, y2 = [int(round(value * scale)) for value in node.bbox_xyxy_abs]
            color = tuple(int(v) for v in DEFAULT_CLASS_PALETTE_BGR[min(node.class_id, len(DEFAULT_CLASS_PALETTE_BGR) - 1)])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{node.id} {node.class_name.replace('Ycb', '')[:10]}"
            cv2.putText(image, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(image, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        header = np.full((34, image.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header, f"{title} ({len(nodes)})", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 42, 48), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, image]))
    gap = 16
    header_h = 50
    total_w = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    total_h = header_h + max(panel.shape[0] for panel in panels)
    canvas = np.full((total_h, total_w, 3), 242, dtype=np.uint8)
    cv2.putText(canvas, sample_id, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 42, 48), 1, cv2.LINE_AA)
    x = 0
    for panel in panels:
        canvas[header_h:header_h + panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    cv2.imwrite(str(output_path), canvas)


def select_visual_examples(sample_results: Sequence[Mapping[str, Any]], max_examples: int) -> List[int]:
    scored = []
    for index, item in enumerate(sample_results):
        metrics = item["metrics"]
        score = int(metrics["thresholds"]["0.25"]["matched"]) * 10 - int(metrics["thresholds"]["0.25"]["false_positives"])
        merge = int(metrics["merge_indicators"])
        scored.append((merge, -score, index))
    chosen = [index for _merge, _score, index in sorted(scored, reverse=True)[: int(max_examples)]]
    return chosen


def build_context_background(cnabu_path: str) -> np.ndarray:
    with np.load(cnabu_path, allow_pickle=False) as data:
        context = build_cnabu_map_context(
            occupancy_mean=np.asarray(data["occupancy_mean"], dtype=np.float32),
            semantic_mean=np.asarray(data["semantic_mean"], dtype=np.float32),
            raw_shape_hw=(140, 200),
            crop_rows=np.asarray(data["crop_rows"], dtype=np.int64).tolist(),
        )
    return np.asarray(context["background_bgr"], dtype=np.uint8)


def write_markdown(summary: Mapping[str, Any]) -> str:
    config = summary["config"]
    lines = [
        "# MEM CNABU Node Proposal Prototype",
        "",
        f"Created: `{summary['created_at']}`",
        f"Host: `{summary['host']}`",
        f"Schema: `{summary['schema']}`",
        f"Output dir: `{summary.get('output_dir', '')}`",
        "",
        "## Training",
        "",
        f"- Device: `{summary['training']['device']}`",
        f"- Iterations: `{summary['training']['iterations']}`",
        f"- Checkpoints written: `{summary['safety']['checkpoint_write']}`",
        f"- Command: `{summary.get('command', {}).get('argv', '')}`",
        f"- Records: train `{summary['records']['train']}`, val `{summary['records']['val']}`, test `{summary['records']['test']}`",
        f"- Records JSON: `{config['DATA']['RECORDS_JSON']}`",
        f"- Split JSON: `{config['DATA']['SPLIT_JSON']}`",
        f"- CNABU root: `{config['DATA']['CNABU_ROOT']}`",
        f"- Raw GT root: `{config['DATA']['RAW_ROOT']}`",
        "",
        "## Metrics",
        "",
        "| Split | Variant | Pred | GT | Match@0.25 | P@0.25 | R@0.25 | F1@0.25 | FP@0.25 | Miss@0.25 | Match@0.50 | F1@0.50 | Merge | Over-split |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("val", "test"):
        for variant_name, metrics in summary["evaluation"][split_name].items():
            t25 = metrics["thresholds"]["0.25"]
            t50 = metrics["thresholds"]["0.50"]
            lines.append(
                "| {split} | {variant} | {pred} | {gt} | {m25} | {p25:.3f} | {r25:.3f} | {f25:.3f} | {fp25} | {miss25} | {m50} | {f50:.3f} | {merge} | {over} |".format(
                    split=split_name,
                    variant=variant_name,
                    pred=metrics["pred_count"],
                    gt=metrics["gt_count"],
                    m25=t25["matched"],
                    p25=t25["precision"],
                    r25=t25["recall"],
                    f25=t25["f1"],
                    fp25=t25["false_positives"],
                    miss25=t25["missed_gt"],
                    m50=t50["matched"],
                    f50=t50["f1"],
                    merge=metrics["merge_indicators"],
                    over=metrics["over_split_indicators"],
                )
            )
    lines.extend(
        [
            "",
            "## Merge Analysis",
            "",
            "| Split | F1@0.25 delta vs 2D | Recall@0.25 delta vs 2D | Merge delta vs 2D | Over-split delta vs 2D | FP delta vs 2D |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in ("val", "test"):
        learned = summary["evaluation"][split_name]["learned_node_proposal"]
        base = summary["evaluation"][split_name]["split_on_2d_candidate"]
        learned_t25 = learned["thresholds"]["0.25"]
        base_t25 = base["thresholds"]["0.25"]
        lines.append(
            "| {split} | {f1:+.3f} | {recall:+.3f} | {merge:+d} | {over:+d} | {fp:+d} |".format(
                split=split_name,
                f1=learned_t25["f1"] - base_t25["f1"],
                recall=learned_t25["recall"] - base_t25["recall"],
                merge=learned["merge_indicators"] - base["merge_indicators"],
                over=learned["over_split_indicators"] - base["over_split_indicators"],
                fp=learned_t25["false_positives"] - base_t25["false_positives"],
            )
        )
    lines.extend(["", "## Class Notes", ""])
    key_class_ids = ("9", "0", "4", "8", "13")
    lines.extend(
        [
            "| Split | Class | Variant | Pred | GT | R@0.25 | F1@0.25 | Merge | Over-split |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in ("val", "test"):
        for class_id in key_class_ids:
            for variant in ("split_on_2d_candidate", "learned_node_proposal"):
                item = summary["evaluation"][split_name][variant]["class_wise"].get(class_id)
                if not item:
                    continue
                t25 = item["thresholds"]["0.25"]
                lines.append(
                    "| {split} | {klass} | {variant} | {pred} | {gt} | {recall:.3f} | {f1:.3f} | {merge} | {over} |".format(
                        split=split_name,
                        klass=item["class_name"],
                        variant=variant,
                        pred=item["pred_count"],
                        gt=item["gt_count"],
                        recall=t25["recall"],
                        f1=t25["f1"],
                        merge=item["merge_indicators"],
                        over=item["over_split_indicators"],
                    )
                )
    lines.extend(["", "## Visual Diagnostics", ""])
    lines.append("Each panel shows `GT`, `split_off`, `2D_split`, and `learned` node masks on the same CNABU map background.")
    for path in summary.get("visual_examples", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Recommendation", ""])
    lines.extend(summary["recommendation"])
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Inputs at inference are CNABU belief maps only.",
            "- GT instance masks/classes are used only for offline training targets and evaluation.",
            "- No D3G relation inference, dataset generation, HDF5 export, staging, or commit was performed.",
        ]
    )
    return "\n".join(lines)


def recommendation(summary: Mapping[str, Any]) -> List[str]:
    val = summary["evaluation"]["val"]
    test = summary["evaluation"]["test"]
    base_val = val["split_on_2d_candidate"]["thresholds"]["0.25"]
    learned_val = val["learned_node_proposal"]["thresholds"]["0.25"]
    base_test = test["split_on_2d_candidate"]["thresholds"]["0.25"]
    learned_test = test["learned_node_proposal"]["thresholds"]["0.25"]
    val_f1_delta = learned_val["f1"] - base_val["f1"]
    test_f1_delta = learned_test["f1"] - base_test["f1"]
    val_recall_delta = learned_val["recall"] - base_val["recall"]
    test_recall_delta = learned_test["recall"] - base_test["recall"]
    val_merge_delta = val["learned_node_proposal"]["merge_indicators"] - val["split_on_2d_candidate"]["merge_indicators"]
    test_merge_delta = test["learned_node_proposal"]["merge_indicators"] - test["split_on_2d_candidate"]["merge_indicators"]
    passes = (
        val_f1_delta >= 0.01
        and test_f1_delta >= 0.01
        and val_recall_delta >= 0.02
        and test_recall_delta >= 0.02
        and val_merge_delta <= -0.2 * max(1, val["split_on_2d_candidate"]["merge_indicators"])
        and test_merge_delta <= -0.2 * max(1, test["split_on_2d_candidate"]["merge_indicators"])
        and learned_val["false_positives"] <= 2 * max(1, base_val["false_positives"])
        and learned_test["false_positives"] <= 2 * max(1, base_test["false_positives"])
    )
    return [
        f"Validation learned-vs-2D delta: F1@0.25 {val_f1_delta:+.3f}, recall {val_recall_delta:+.3f}, merge {val_merge_delta:+d}.",
        f"Test learned-vs-2D delta: F1@0.25 {test_f1_delta:+.3f}, recall {test_recall_delta:+.3f}, merge {test_merge_delta:+d}.",
        (
            "The learned node proposal prototype clears the suggested 5000-record scale-up threshold."
            if passes
            else "The learned node proposal prototype does not clear the suggested 5000-record scale-up threshold yet."
        ),
    ]


def run_train_eval(config: Mapping[str, Any], *, output_dir: Path, device: torch.device) -> Dict[str, Any]:
    set_seed(int(config["TRAIN"]["SEED"]))
    records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
    split_manifest = read_json(Path(config["DATA"]["SPLIT_JSON"]))
    records_by_split = split_records(records, split_manifest)
    train_dataset = CnabuNodeProposalDataset(records_by_split["train"], config=config)
    val_dataset = CnabuNodeProposalDataset(records_by_split["val"], config=config)
    test_dataset = CnabuNodeProposalDataset(records_by_split["test"], config=config)
    model = TinyNodeProposalNet(
        int(config["MODEL"]["IN_CHANNELS"]),
        int(config["MODEL"]["HIDDEN_CHANNELS"]),
        int(config["DATA"]["NUM_CLASSES"]),
        float(config["MODEL"].get("SEMANTIC_PRIOR_LOGIT_SCALE", 1.0)),
    ).to(device)
    history = train_model(
        model,
        train_dataset,
        config=config,
        iterations=int(config["TRAIN"]["ITERATIONS"]),
        device=device,
        seed=int(config["TRAIN"]["SEED"]),
    )
    eval_by_split: Dict[str, Dict[str, Any]] = {}
    learned_samples_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, dataset, split_records_list in (
        ("val", val_dataset, records_by_split["val"]),
        ("test", test_dataset, records_by_split["test"]),
    ):
        learned_metrics, sample_results = evaluate_learned_model(
            model,
            dataset,
            config=config,
            device=device,
            records=split_records_list,
        )
        learned_samples_by_split[split_name] = sample_results
        eval_by_split[split_name] = {
            "split_off": evaluate_rule_baseline(split_records_list, config=config, split_config={"enabled": False}),
            "split_on_2d_candidate": evaluate_rule_baseline(
                split_records_list,
                config=config,
                split_config={"enabled": True, "method": "candidate_gated_2d_footprint"},
            ),
            "learned_node_proposal": learned_metrics,
        }
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    visual_paths = []
    for idx in select_visual_examples(learned_samples_by_split["val"], int(config["EVAL"]["MAX_VISUAL_EXAMPLES"])):
        sample = learned_samples_by_split["val"][idx]
        record = records_by_split["val"][idx]
        background = build_context_background(str(sample["cnabu_path"]))
        image_path = examples_dir / f"val_{sample['sample_id'].replace('/', '_')}.png"
        split_off_nodes = rule_nodes_for_record(record, config=config, split_config={"enabled": False})
        split_2d_nodes = rule_nodes_for_record(
            record,
            config=config,
            split_config={"enabled": True, "method": "candidate_gated_2d_footprint"},
        )
        render_panel(
            output_path=image_path,
            sample_id=f"val {sample['sample_id']}",
            background_bgr=background,
            columns=(
                ("GT", sample["gt_nodes"]),
                ("split_off", split_off_nodes),
                ("2D_split", split_2d_nodes),
                ("learned", sample["pred_nodes"]),
            ),
        )
        visual_paths.append(str(image_path))
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "train_eval",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "output_dir": str(output_dir),
        "config": config,
        "records": {
            "train": len(records_by_split["train"]),
            "val": len(records_by_split["val"]),
            "test": len(records_by_split["test"]),
        },
        "training": {
            "device": str(device),
            "iterations": int(config["TRAIN"]["ITERATIONS"]),
            "history": history,
        },
        "evaluation": eval_by_split,
        "visual_examples": visual_paths,
        "safety": {
            "gt_used_for_runtime_inference_input": False,
            "gt_used_for_training_targets": True,
            "d3g_relation_inference": False,
            "checkpoint_write": False,
            "dataset_generation": False,
        },
    }
    summary["recommendation"] = recommendation(summary)
    return summary


def run_one_record_overfit(config: Mapping[str, Any], *, device: torch.device) -> Dict[str, Any]:
    records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
    split_manifest = read_json(Path(config["DATA"]["SPLIT_JSON"]))
    records_by_split = split_records(records, split_manifest, max_train=1, max_val=1, max_test=1)
    dataset = CnabuNodeProposalDataset(records_by_split["train"], config=config)
    model = TinyNodeProposalNet(
        int(config["MODEL"]["IN_CHANNELS"]),
        int(config["MODEL"]["HIDDEN_CHANNELS"]),
        int(config["DATA"]["NUM_CLASSES"]),
        float(config["MODEL"].get("SEMANTIC_PRIOR_LOGIT_SCALE", 1.0)),
    ).to(device)
    history = train_model(model, dataset, config=config, iterations=int(config["TRAIN"]["ONE_RECORD_OVERFIT_ITERATIONS"]), device=device, seed=int(config["TRAIN"]["SEED"]))
    metrics, _samples = evaluate_learned_model(model, dataset, config=config, device=device)
    return {"sample_id": dataset[0]["sample_id"], "history": history, "metrics": metrics}


def run_smoke20(config: Mapping[str, Any], *, device: torch.device) -> Dict[str, Any]:
    records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
    split_manifest = read_json(Path(config["DATA"]["SPLIT_JSON"]))
    records_by_split = split_records(records, split_manifest, max_train=16, max_val=4, max_test=4)
    train_dataset = CnabuNodeProposalDataset(records_by_split["train"], config=config)
    val_dataset = CnabuNodeProposalDataset(records_by_split["val"], config=config)
    model = TinyNodeProposalNet(
        int(config["MODEL"]["IN_CHANNELS"]),
        int(config["MODEL"]["HIDDEN_CHANNELS"]),
        int(config["DATA"]["NUM_CLASSES"]),
        float(config["MODEL"].get("SEMANTIC_PRIOR_LOGIT_SCALE", 1.0)),
    ).to(device)
    history = train_model(model, train_dataset, config=config, iterations=int(config["TRAIN"]["SMOKE20_ITERATIONS"]), device=device, seed=int(config["TRAIN"]["SEED"]))
    metrics, _samples = evaluate_learned_model(model, val_dataset, config=config, device=device)
    return {"records": {"train": len(train_dataset), "val": len(val_dataset)}, "history": history, "val_metrics": metrics}


def default_conservative_variants() -> List[Dict[str, Any]]:
    return [
        {
            "name": "learned_node_proposal_current",
            "description": "Global learned proposals with the current prototype thresholds.",
            "CONSERVATIVE": {"MODE": "global", "POSTPROCESS_ENABLED": False},
        },
        {
            "name": "learned_global_tight",
            "description": "Global learned proposals with score, area, and overlap filtering.",
            "INFERENCE": {
                "FOREGROUND_THRESHOLD": 0.50,
                "CENTER_THRESHOLD": 0.45,
                "MIN_NODE_SCORE": 0.60,
                "MIN_NODE_PIXELS": 28,
            },
            "CONSERVATIVE": {
                "MODE": "global",
                "POSTPROCESS_ENABLED": True,
                "CLASS_SIZE_PRIOR_REJECTION": True,
                "MIN_AREA_PRIOR_MULTIPLIER": 0.12,
                "MAX_AREA_PRIOR_MULTIPLIER": 2.50,
                "MASK_OVERLAP_SUPPRESSION_THRESHOLD": 0.55,
            },
        },
        {
            "name": "learned_candidate_gated_balanced",
            "description": "Use split_off CNABU components as parents and only replace suspicious parents with learned children.",
            "INFERENCE": {
                "FOREGROUND_THRESHOLD": 0.45,
                "CENTER_THRESHOLD": 0.35,
                "MIN_NODE_SCORE": 0.48,
                "MIN_NODE_PIXELS": 18,
            },
            "CONSERVATIVE": {
                "MODE": "candidate_gated",
                "POSTPROCESS_ENABLED": True,
                "CLASS_SIZE_PRIOR_REJECTION": True,
                "MIN_AREA_PRIOR_MULTIPLIER": 0.10,
                "MAX_AREA_PRIOR_MULTIPLIER": 2.75,
                "MASK_OVERLAP_SUPPRESSION_THRESHOLD": 0.60,
                "CHILD_OVERLAP_SUPPRESSION_THRESHOLD": 0.55,
                "SUSPICIOUS_PARENT_ONLY": True,
                "SUSPICIOUS_AREA_PRIOR_MULTIPLIER": 1.35,
                "SUSPICIOUS_BBOX_PRIOR_MULTIPLIER": 1.35,
                "ALLOW_MULTI_CHILD_SUSPICION": True,
                "MIN_CHILDREN_FOR_SPLIT": 2,
                "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.30,
                "CHILD_INSIDE_PARENT_MIN_FRACTION": 0.65,
                "CHILD_PARENT_MIN_FRACTION": 0.02,
                "MAX_CHILDREN_PER_PARENT": 5,
            },
        },
        {
            "name": "learned_candidate_gated_strict",
            "description": "Candidate-gated learned splitting with higher confidence and stronger child checks.",
            "INFERENCE": {
                "FOREGROUND_THRESHOLD": 0.50,
                "CENTER_THRESHOLD": 0.45,
                "MIN_NODE_SCORE": 0.58,
                "MIN_NODE_PIXELS": 24,
            },
            "CONSERVATIVE": {
                "MODE": "candidate_gated",
                "POSTPROCESS_ENABLED": True,
                "CLASS_SIZE_PRIOR_REJECTION": True,
                "MIN_AREA_PRIOR_MULTIPLIER": 0.12,
                "MAX_AREA_PRIOR_MULTIPLIER": 2.40,
                "MASK_OVERLAP_SUPPRESSION_THRESHOLD": 0.55,
                "CHILD_OVERLAP_SUPPRESSION_THRESHOLD": 0.50,
                "SUSPICIOUS_PARENT_ONLY": True,
                "SUSPICIOUS_AREA_PRIOR_MULTIPLIER": 1.45,
                "SUSPICIOUS_BBOX_PRIOR_MULTIPLIER": 1.45,
                "ALLOW_MULTI_CHILD_SUSPICION": True,
                "MIN_CHILDREN_FOR_SPLIT": 2,
                "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.40,
                "CHILD_INSIDE_PARENT_MIN_FRACTION": 0.75,
                "CHILD_PARENT_MIN_FRACTION": 0.03,
                "MAX_CHILDREN_PER_PARENT": 4,
            },
        },
        {
            "name": "learned_2d_guarded_balanced",
            "description": "Only apply learned splitting to split_off parents that the 2D candidate splitter also splits; otherwise keep/fallback to 2D-guarded nodes.",
            "INFERENCE": {
                "FOREGROUND_THRESHOLD": 0.45,
                "CENTER_THRESHOLD": 0.35,
                "MIN_NODE_SCORE": 0.50,
                "MIN_NODE_PIXELS": 18,
            },
            "CONSERVATIVE": {
                "MODE": "candidate_gated",
                "POSTPROCESS_ENABLED": True,
                "CLASS_SIZE_PRIOR_REJECTION": True,
                "MIN_AREA_PRIOR_MULTIPLIER": 0.10,
                "MAX_AREA_PRIOR_MULTIPLIER": 2.75,
                "MASK_OVERLAP_SUPPRESSION_THRESHOLD": 0.60,
                "CHILD_OVERLAP_SUPPRESSION_THRESHOLD": 0.55,
                "USE_2D_SPLIT_GUARD": True,
                "FALLBACK_TO_2D_GUARD": True,
                "MAX_EXTRA_CHILDREN_OVER_GUARD": 0,
                "SUSPICIOUS_PARENT_ONLY": True,
                "SUSPICIOUS_AREA_PRIOR_MULTIPLIER": 1.35,
                "SUSPICIOUS_BBOX_PRIOR_MULTIPLIER": 1.35,
                "ALLOW_MULTI_CHILD_SUSPICION": True,
                "MIN_CHILDREN_FOR_SPLIT": 2,
                "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.30,
                "CHILD_INSIDE_PARENT_MIN_FRACTION": 0.65,
                "CHILD_PARENT_MIN_FRACTION": 0.02,
                "MAX_CHILDREN_PER_PARENT": 4,
            },
        },
        {
            "name": "learned_2d_guarded_strict",
            "description": "2D-guarded learned splitting with stricter score, area, and child-count limits.",
            "INFERENCE": {
                "FOREGROUND_THRESHOLD": 0.52,
                "CENTER_THRESHOLD": 0.48,
                "MIN_NODE_SCORE": 0.62,
                "MIN_NODE_PIXELS": 24,
            },
            "CONSERVATIVE": {
                "MODE": "candidate_gated",
                "POSTPROCESS_ENABLED": True,
                "CLASS_SIZE_PRIOR_REJECTION": True,
                "MIN_AREA_PRIOR_MULTIPLIER": 0.12,
                "MAX_AREA_PRIOR_MULTIPLIER": 2.35,
                "MASK_OVERLAP_SUPPRESSION_THRESHOLD": 0.55,
                "CHILD_OVERLAP_SUPPRESSION_THRESHOLD": 0.50,
                "USE_2D_SPLIT_GUARD": True,
                "FALLBACK_TO_2D_GUARD": True,
                "MAX_EXTRA_CHILDREN_OVER_GUARD": 0,
                "SUSPICIOUS_PARENT_ONLY": True,
                "SUSPICIOUS_AREA_PRIOR_MULTIPLIER": 1.45,
                "SUSPICIOUS_BBOX_PRIOR_MULTIPLIER": 1.45,
                "ALLOW_MULTI_CHILD_SUSPICION": False,
                "MIN_CHILDREN_FOR_SPLIT": 2,
                "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.45,
                "CHILD_INSIDE_PARENT_MIN_FRACTION": 0.78,
                "CHILD_PARENT_MIN_FRACTION": 0.04,
                "MAX_CHILDREN_PER_PARENT": 4,
            },
        },
    ]


def config_for_variant(base_config: Mapping[str, Any], variant: Mapping[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(base_config))
    for section in ("MODEL", "TRAIN", "INFERENCE", "CONSERVATIVE", "EVAL"):
        if section in variant:
            result.setdefault(section, {})
            deep_update(result[section], variant[section])
    return result


def metric_delta_row(metrics: Mapping[str, Any], baseline: Mapping[str, Any], current: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    t25 = metrics["thresholds"]["0.25"]
    base_t25 = baseline["thresholds"]["0.25"]
    row = {
        "f1_delta_vs_2d": float(t25["f1"] - base_t25["f1"]),
        "recall_delta_vs_2d": float(t25["recall"] - base_t25["recall"]),
        "precision_delta_vs_2d": float(t25["precision"] - base_t25["precision"]),
        "fp_delta_vs_2d": int(t25["false_positives"] - base_t25["false_positives"]),
        "miss_delta_vs_2d": int(t25["missed_gt"] - base_t25["missed_gt"]),
        "merge_delta_vs_2d": int(metrics["merge_indicators"] - baseline["merge_indicators"]),
        "over_split_delta_vs_2d": int(metrics["over_split_indicators"] - baseline["over_split_indicators"]),
    }
    if current is not None:
        current_t25 = current["thresholds"]["0.25"]
        row.update(
            {
                "fp_delta_vs_current": int(t25["false_positives"] - current_t25["false_positives"]),
                "over_split_delta_vs_current": int(metrics["over_split_indicators"] - current["over_split_indicators"]),
                "merge_delta_vs_current": int(metrics["merge_indicators"] - current["merge_indicators"]),
                "f1_delta_vs_current": float(t25["f1"] - current_t25["f1"]),
            }
        )
    return row


def choose_conservative_variant(eval_items: Mapping[str, Any]) -> str:
    baseline = eval_items["split_on_2d_candidate"]
    current = eval_items["learned_node_proposal_current"]
    best_name = "learned_node_proposal_current"
    best_score = -1e9
    for name, metrics in eval_items.items():
        if not name.startswith("learned_") or name == "learned_node_proposal_current":
            continue
        delta = metric_delta_row(metrics, baseline, current)
        score = (
            4.0 * delta["f1_delta_vs_2d"]
            + 1.0 * delta["recall_delta_vs_2d"]
            - 0.003 * max(0, delta["fp_delta_vs_2d"])
            - 0.003 * max(0, delta["over_split_delta_vs_2d"])
            - 0.001 * max(0, delta["merge_delta_vs_2d"])
            - 0.002 * max(0, delta["fp_delta_vs_current"])
            - 0.002 * max(0, delta["over_split_delta_vs_current"])
        )
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def conservative_recommendation(summary: Mapping[str, Any]) -> List[str]:
    split_eval = summary["evaluation"]["val"]
    baseline = split_eval["split_on_2d_candidate"]
    current = split_eval["learned_node_proposal_current"]
    selected_name = str(summary["selected_variant"])
    selected = split_eval[selected_name]
    delta_vs_2d = metric_delta_row(selected, baseline, current)
    current_delta_vs_2d = metric_delta_row(current, baseline)
    previous = summary["previous_full_run_reference"]["val"]
    fp_target = math.floor(max(0, int(previous["learned_fp_delta_vs_2d"])) * 0.5)
    over_target = math.floor(max(0, int(previous["learned_over_split_delta_vs_2d"])) * 0.5)
    controls_fp = delta_vs_2d["fp_delta_vs_2d"] <= fp_target
    controls_over = delta_vs_2d["over_split_delta_vs_2d"] <= over_target
    preserves_recall = delta_vs_2d["recall_delta_vs_2d"] > 0.0
    preserves_merge = delta_vs_2d["merge_delta_vs_2d"] < 0
    improves_f1 = delta_vs_2d["f1_delta_vs_2d"] > 0.0
    medium_ready = controls_fp and controls_over and preserves_recall and preserves_merge
    full_ready = medium_ready and (delta_vs_2d["f1_delta_vs_2d"] >= 0.01 or delta_vs_2d["merge_delta_vs_2d"] <= -10)
    return [
        f"Selected `{selected_name}` for this subset.",
        (
            "Compared with split_on_2d_candidate: "
            f"F1@0.25 {delta_vs_2d['f1_delta_vs_2d']:+.3f}, "
            f"recall {delta_vs_2d['recall_delta_vs_2d']:+.3f}, "
            f"merge {delta_vs_2d['merge_delta_vs_2d']:+d}, "
            f"FP {delta_vs_2d['fp_delta_vs_2d']:+d}, "
            f"over-split {delta_vs_2d['over_split_delta_vs_2d']:+d}."
        ),
        (
            "Compared with current learned on the same subset: "
            f"F1@0.25 {delta_vs_2d['f1_delta_vs_current']:+.3f}, "
            f"merge {delta_vs_2d['merge_delta_vs_current']:+d}, "
            f"FP {delta_vs_2d['fp_delta_vs_current']:+d}, "
            f"over-split {delta_vs_2d['over_split_delta_vs_current']:+d}."
        ),
        (
            "The selected conservative setting controls FP/over-split while preserving recall/merge gains, so a medium subset run is justified."
            if medium_ready
            else "The selected conservative setting does not yet preserve recall/merge gains while controlling FP/over-split, so scale-up is not justified."
        ),
        (
            "A full 1000-record confirmation would still require Ehsan approval; this run does not launch it."
            if full_ready
            else "Do not run full 1000 confirmation yet; revise or validate on a medium subset first."
        ),
        (
            "Current learned subset reference vs 2D: "
            f"F1@0.25 {current_delta_vs_2d['f1_delta_vs_2d']:+.3f}, "
            f"FP {current_delta_vs_2d['fp_delta_vs_2d']:+d}, "
            f"over-split {current_delta_vs_2d['over_split_delta_vs_2d']:+d}."
        ),
    ]


def write_conservative_markdown(summary: Mapping[str, Any]) -> str:
    val_eval = summary["evaluation"]["val"]
    baseline = val_eval["split_on_2d_candidate"]
    current = val_eval["learned_node_proposal_current"]
    lines = [
        "# MEM CNABU Conservative Node Proposal",
        "",
        f"Created: `{summary['created_at']}`",
        f"Host: `{summary['host']}`",
        f"Schema: `{summary['schema']}`",
        f"Stage: `{summary['stage']}`",
        f"Output dir: `{summary['output_dir']}`",
        "",
        "## Run",
        "",
        f"- Command: `{summary['command']['argv']}`",
        f"- Device: `{summary['training']['device']}`",
        f"- Iterations: `{summary['training']['iterations']}`",
        f"- Records: train `{summary['records']['train']}`, val `{summary['records']['val']}`, test `{summary['records']['test']}`",
        f"- Checkpoint used: `{summary['training']['checkpoint_used']}`",
        f"- Checkpoint written: `{summary['safety']['checkpoint_write']}`",
        f"- CNABU root: `{summary['config']['DATA']['CNABU_ROOT']}`",
        "",
        "## Metrics",
        "",
        "| Variant | Pred | GT | P@0.25 | R@0.25 | F1@0.25 | FP@0.25 | Miss@0.25 | F1@0.50 | Merge | Over-split | FP vs 2D | Over vs 2D | Merge vs 2D |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_name, metrics in val_eval.items():
        t25 = metrics["thresholds"]["0.25"]
        t50 = metrics["thresholds"]["0.50"]
        delta = metric_delta_row(metrics, baseline, current if variant_name.startswith("learned_") else None)
        lines.append(
            "| {variant} | {pred} | {gt} | {p25:.3f} | {r25:.3f} | {f25:.3f} | {fp25} | {miss25} | {f50:.3f} | {merge} | {over} | {fp_delta:+d} | {over_delta:+d} | {merge_delta:+d} |".format(
                variant=variant_name,
                pred=metrics["pred_count"],
                gt=metrics["gt_count"],
                p25=t25["precision"],
                r25=t25["recall"],
                f25=t25["f1"],
                fp25=t25["false_positives"],
                miss25=t25["missed_gt"],
                f50=t50["f1"],
                merge=metrics["merge_indicators"],
                over=metrics["over_split_indicators"],
                fp_delta=delta["fp_delta_vs_2d"],
                over_delta=delta["over_split_delta_vs_2d"],
                merge_delta=delta["merge_delta_vs_2d"],
            )
        )
    lines.extend(
        [
            "",
            "## Previous Full-Run Reference",
            "",
            "| Split | Learned F1@0.25 delta vs 2D | Recall delta | Merge delta | FP delta | Over-split delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name, item in summary["previous_full_run_reference"].items():
        lines.append(
            f"| {split_name} | {item['learned_f1_delta_vs_2d']:+.3f} | {item['learned_recall_delta_vs_2d']:+.3f} | {item['learned_merge_delta_vs_2d']:+d} | {item['learned_fp_delta_vs_2d']:+d} | {item['learned_over_split_delta_vs_2d']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Class Breakdown",
            "",
            "| Class | Variant | Pred | GT | R@0.25 | F1@0.25 | Merge | Over-split |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_id in ("9", "0", "4", "8", "13"):
        for variant_name in ("split_on_2d_candidate", "learned_node_proposal_current", summary["selected_variant"]):
            item = val_eval[variant_name]["class_wise"].get(class_id)
            if not item:
                continue
            t25 = item["thresholds"]["0.25"]
            lines.append(
                f"| {item['class_name']} | {variant_name} | {item['pred_count']} | {item['gt_count']} | {t25['recall']:.3f} | {t25['f1']:.3f} | {item['merge_indicators']} | {item['over_split_indicators']} |"
            )
    lines.extend(["", "## Variant Settings", ""])
    for variant in summary["variants"]:
        lines.append(f"- `{variant['name']}`: {variant.get('description', '')}")
    lines.extend(["", "## Visual Diagnostics", ""])
    for path in summary.get("visual_examples", []):
        lines.append(f"- `{path}`")
    lines.extend(["", "## Recommendation", ""])
    lines.extend(summary["recommendation"])
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Inputs at inference are CNABU belief maps only.",
            "- GT instance masks/classes are used only for offline training targets and evaluation.",
            "- No full 1000-record confirmation, checkpoint write, dataset generation, HDF5 export, staging, or commit was performed.",
        ]
    )
    return "\n".join(lines)


def previous_full_run_reference() -> Dict[str, Any]:
    return {
        "val": {
            "learned_f1_delta_vs_2d": 0.004,
            "learned_recall_delta_vs_2d": 0.021,
            "learned_merge_delta_vs_2d": -35,
            "learned_fp_delta_vs_2d": 32,
            "learned_over_split_delta_vs_2d": 41,
        },
        "test": {
            "learned_f1_delta_vs_2d": 0.009,
            "learned_recall_delta_vs_2d": 0.026,
            "learned_merge_delta_vs_2d": -43,
            "learned_fp_delta_vs_2d": 25,
            "learned_over_split_delta_vs_2d": 43,
        },
    }


def run_conservative_subset(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    device: torch.device,
    stage: str,
) -> Dict[str, Any]:
    set_seed(int(config["TRAIN"]["SEED"]))
    cfg = config.get("CONSERVATIVE", {})
    if stage == "small":
        max_train = int(cfg.get("SMALL_MAX_TRAIN", 100))
        max_val = int(cfg.get("SMALL_MAX_VAL", 50))
        max_test = int(cfg.get("SMALL_MAX_TEST", 0))
        iterations = int(cfg.get("SMALL_ITERATIONS", 300))
    elif stage == "medium":
        max_train = int(cfg.get("MEDIUM_MAX_TRAIN", 400))
        max_val = int(cfg.get("MEDIUM_MAX_VAL", 50))
        max_test = int(cfg.get("MEDIUM_MAX_TEST", 0))
        iterations = int(cfg.get("MEDIUM_ITERATIONS", 400))
    else:
        raise ValueError(f"unknown conservative stage: {stage}")

    records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
    split_manifest = read_json(Path(config["DATA"]["SPLIT_JSON"]))
    records_by_split = split_records(
        records,
        split_manifest,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
    )
    train_dataset = CnabuNodeProposalDataset(records_by_split["train"], config=config)
    val_dataset = CnabuNodeProposalDataset(records_by_split["val"], config=config)
    model = TinyNodeProposalNet(
        int(config["MODEL"]["IN_CHANNELS"]),
        int(config["MODEL"]["HIDDEN_CHANNELS"]),
        int(config["DATA"]["NUM_CLASSES"]),
        float(config["MODEL"].get("SEMANTIC_PRIOR_LOGIT_SCALE", 1.0)),
    ).to(device)
    history = train_model(
        model,
        train_dataset,
        config=config,
        iterations=iterations,
        device=device,
        seed=int(config["TRAIN"]["SEED"]),
    )
    eval_items: Dict[str, Any] = {
        "split_off": evaluate_rule_baseline(records_by_split["val"], config=config, split_config={"enabled": False}),
        "split_on_2d_candidate": evaluate_rule_baseline(
            records_by_split["val"],
            config=config,
            split_config={"enabled": True, "method": "candidate_gated_2d_footprint"},
        ),
    }
    sample_results_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    variants = list(config.get("CONSERVATIVE", {}).get("SWEEP_VARIANTS", []) or default_conservative_variants())
    for variant in variants:
        variant_cfg = config_for_variant(config, variant)
        mode = str(variant_cfg.get("CONSERVATIVE", {}).get("MODE", "global"))
        metrics, samples = evaluate_learned_model(
            model,
            val_dataset,
            config=variant_cfg,
            device=device,
            records=records_by_split["val"],
            variant_mode=mode,
        )
        eval_items[str(variant["name"])] = metrics
        sample_results_by_variant[str(variant["name"])] = samples

    selected_variant = choose_conservative_variant(eval_items)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    visual_paths = []
    selected_samples = sample_results_by_variant[selected_variant]
    current_samples = sample_results_by_variant.get("learned_node_proposal_current", selected_samples)
    for idx in select_visual_examples(selected_samples, int(config["EVAL"]["MAX_VISUAL_EXAMPLES"])):
        sample = selected_samples[idx]
        current_sample = current_samples[idx]
        record = records_by_split["val"][idx]
        background = build_context_background(str(sample["cnabu_path"]))
        image_path = examples_dir / f"val_{sample['sample_id'].replace('/', '_')}.png"
        render_panel(
            output_path=image_path,
            sample_id=f"val {sample['sample_id']}",
            background_bgr=background,
            columns=(
                ("GT", sample["gt_nodes"]),
                ("split_off", rule_nodes_for_record(record, config=config, split_config={"enabled": False})),
                (
                    "2D_split",
                    rule_nodes_for_record(
                        record,
                        config=config,
                        split_config={"enabled": True, "method": "candidate_gated_2d_footprint"},
                    ),
                ),
                ("current", current_sample["pred_nodes"]),
                ("selected", sample["pred_nodes"]),
            ),
        )
        visual_paths.append(str(image_path))

    summary: Dict[str, Any] = {
        "schema": "mem_cnabu_node_proposal_conservative_eval_v0",
        "stage": stage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "output_dir": str(output_dir),
        "config": config,
        "records": {
            "train": len(records_by_split["train"]),
            "val": len(records_by_split["val"]),
            "test": len(records_by_split["test"]),
        },
        "training": {
            "device": str(device),
            "iterations": iterations,
            "history": history,
            "checkpoint_used": False,
        },
        "evaluation": {"val": eval_items},
        "selected_variant": selected_variant,
        "variants": variants,
        "previous_full_run_reference": previous_full_run_reference(),
        "visual_examples": visual_paths,
        "safety": {
            "gt_used_for_runtime_inference_input": False,
            "gt_used_for_training_targets": True,
            "d3g_relation_inference": False,
            "checkpoint_write": False,
            "dataset_generation": False,
            "full_1000_confirmation": False,
        },
    }
    summary["recommendation"] = conservative_recommendation(summary)
    return summary


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "import_smoke",
            "synthetic_overfit",
            "one_record_overfit",
            "smoke20",
            "train_eval",
            "staged",
            "conservative_small",
            "conservative_medium",
        ),
        default="staged",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config_file), args.override)
    if args.device is not None:
        config["TRAIN"]["DEVICE"] = str(args.device)
    device = torch.device(str(config["TRAIN"]["DEVICE"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = str(config["OUTPUT"]["RUN_PREFIX"])
        if str(args.mode).startswith("conservative_"):
            run_prefix = "mem_cnabu_node_proposal_conservative"
        output_dir = Path(config["OUTPUT"]["ROOT"]) / f"{run_prefix}_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    payload: Dict[str, Any] = {
        "schema": "mem_cnabu_node_proposal_command_v0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "mode": args.mode,
        "output_dir": str(output_dir),
        "device": str(device),
        "safety": {
            "checkpoint_write": False,
            "dataset_generation": False,
            "d3g_relation_inference": False,
        },
    }
    if args.mode == "import_smoke":
        records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
        payload["result"] = {"records_loaded": len(records), "torch_cuda_available": bool(torch.cuda.is_available())}
    elif args.mode == "synthetic_overfit":
        payload["result"] = run_synthetic_overfit(config, device=device)
    elif args.mode == "one_record_overfit":
        payload["result"] = run_one_record_overfit(config, device=device)
    elif args.mode == "smoke20":
        payload["result"] = run_smoke20(config, device=device)
    elif args.mode == "train_eval":
        payload = run_train_eval(config, output_dir=output_dir, device=device)
    elif args.mode == "conservative_small":
        payload = run_conservative_subset(config, output_dir=output_dir, device=device, stage="small")
    elif args.mode == "conservative_medium":
        payload = run_conservative_subset(config, output_dir=output_dir, device=device, stage="medium")
    elif args.mode == "staged":
        payload["stages"] = {
            "synthetic_overfit": run_synthetic_overfit(config, device=device),
            "one_record_overfit": run_one_record_overfit(config, device=device),
            "smoke20": run_smoke20(config, device=device),
        }
        payload["train_eval"] = run_train_eval(config, output_dir=output_dir, device=device)
    payload.setdefault("command", {})["argv"] = " ".join([str(Path(sys.executable)), *sys.argv])
    payload.setdefault("command", {})["cwd"] = str(Path.cwd())
    payload["timing_seconds"] = {"total": float(time.time() - started)}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    if "evaluation" in payload:
        if str(payload.get("schema", "")).startswith("mem_cnabu_node_proposal_conservative"):
            (output_dir / "summary.md").write_text(write_conservative_markdown(payload) + "\n", encoding="utf-8")
        else:
            (output_dir / "summary.md").write_text(write_markdown(payload) + "\n", encoding="utf-8")
    elif "train_eval" in payload:
        (output_dir / "summary.md").write_text(write_markdown(payload["train_eval"]) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "mode": args.mode}, sort_keys=True))
    return 0


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Node):
        return {
            "id": value.id,
            "class_id": value.class_id,
            "class_name": value.class_name,
            "bbox_xyxy_abs": value.bbox_xyxy_abs,
            "centroid_yx": value.centroid_yx,
            "area": value.area,
            "score": value.score,
        }
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
