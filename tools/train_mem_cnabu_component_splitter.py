#!/usr/bin/env python3
"""Train/evaluate a component-conditioned learned CNABU splitter.

This tool learns only inside existing CNABU split_off parent components. GT
instance masks/classes are used only as offline targets/evaluation references.
Inference input is CNABU-derived features plus the parent component mask.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from train_mem_cnabu_node_proposal import (  # noqa: E402
    DEFAULT_YCB_FOOTPRINT_AREA_PRIORS_PIXELS,
    DEFAULT_YCB_FOOTPRINT_HEIGHT_PRIORS_PIXELS,
    DEFAULT_YCB_FOOTPRINT_WIDTH_PRIORS_PIXELS,
    Node,
    aggregate_evaluations,
    apply_overrides,
    bbox_xyxy_from_mask,
    build_context_background,
    centroid_yx_from_mask,
    class_name,
    cnabu_path_from_record,
    evaluate_nodes,
    load_cnabu_features,
    load_gt_nodes,
    read_json,
    read_records,
    render_panel,
    rule_nodes_for_record,
    sample_dir_from_record,
    split_records,
)


THESIS_ROOT = Path("/home/user/ehsanullahm1/thesis")
D3G_ROOT = THESIS_ROOT / "D3G"
DEFAULT_CONFIG = D3G_ROOT / "configs" / "mem" / "cnabu_component_splitter_1000.yaml"
SCHEMA = "mem_cnabu_component_conditioned_splitter_v0"
CHECKPOINT_SCHEMA = "mem_cnabu_component_splitter_runtime_checkpoint_v0"
ALLOWED_CHECKPOINT_FILENAMES = {
    "model_best_validation.pth",
    "model_final.pth",
    "config_resolved.yaml",
    "checkpoint_metadata.json",
}


@dataclass
class ComponentSample:
    sample_id: str
    record_index: int
    parent: Node
    gt_children: List[Node]
    features: np.ndarray
    count_target: int
    center_target: np.ndarray
    crop_window_xyxy: Tuple[int, int, int, int]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def recursive_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def config_with_updates(config: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(config))
    recursive_update(result, updates)
    return result


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


def make_node(node_id: int, class_id: int, mask: np.ndarray, *, score: float = 1.0, was_split: bool = False) -> Node:
    clean = np.asarray(mask, dtype=bool)
    return Node(
        id=int(node_id),
        class_id=int(class_id),
        class_name=class_name(int(class_id)),
        mask=clean,
        bbox_xyxy_abs=bbox_xyxy_from_mask(clean),
        centroid_yx=centroid_yx_from_mask(clean),
        area=int(clean.sum()),
        score=float(score),
        was_split=bool(was_split),
    )


def renumber_nodes(nodes: Sequence[Node]) -> List[Node]:
    return [
        make_node(index + 1, int(node.class_id), node.mask, score=float(node.score), was_split=bool(node.was_split))
        for index, node in enumerate(nodes)
    ]


def crop_window(mask: np.ndarray, *, margin: int, shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy_from_mask(mask)
    x1 = max(0, int(x1) - int(margin))
    y1 = max(0, int(y1) - int(margin))
    x2 = min(int(shape_hw[1]), int(x2) + int(margin))
    y2 = min(int(shape_hw[0]), int(y2) + int(margin))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, int(shape_hw[1]), int(shape_hw[0]))
    return (x1, y1, x2, y2)


def resize_2d(array: np.ndarray, crop_xyxy: Tuple[int, int, int, int], crop_size: int, *, is_mask: bool = False) -> np.ndarray:
    x1, y1, x2, y2 = crop_xyxy
    crop = np.asarray(array)[y1:y2, x1:x2]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(crop.astype(np.float32), (int(crop_size), int(crop_size)), interpolation=interp)


def raw_to_crop_yx(y: float, x: float, crop_xyxy: Tuple[int, int, int, int], crop_size: int) -> Tuple[float, float]:
    x1, y1, x2, y2 = crop_xyxy
    h = max(1.0, float(y2 - y1))
    w = max(1.0, float(x2 - x1))
    return ((float(y) - y1) * float(crop_size) / h, (float(x) - x1) * float(crop_size) / w)


def draw_disk(target: np.ndarray, y: float, x: float, radius: int) -> None:
    cy, cx = int(round(y)), int(round(x))
    h, w = target.shape
    for yy in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for xx in range(max(0, cx - radius), min(w, cx + radius + 1)):
            dist = math.sqrt(float((yy - cy) ** 2 + (xx - cx) ** 2))
            if dist <= radius:
                target[yy, xx] = max(float(target[yy, xx]), 1.0 - dist / float(radius + 1))


def component_input_features(
    *,
    raw_features: np.ndarray,
    parent: Node,
    crop_xyxy: Tuple[int, int, int, int],
    config: Mapping[str, Any],
) -> np.ndarray:
    crop_size = int(config["MODEL"]["CROP_SIZE"])
    channels: List[np.ndarray] = [
        resize_2d(raw_features[channel], crop_xyxy, crop_size, is_mask=False)
        for channel in range(int(config["MODEL"]["BASE_FEATURE_CHANNELS"]))
    ]
    parent_crop = resize_2d(parent.mask.astype(np.float32), crop_xyxy, crop_size, is_mask=True)
    parent_crop = (parent_crop > 0.5).astype(np.float32)
    dist = cv2.distanceTransform(parent_crop.astype(np.uint8), cv2.DIST_L2, 3)
    if float(dist.max()) > 0:
        dist = dist / float(dist.max())
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, crop_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, crop_size, dtype=np.float32),
        indexing="ij",
    )
    class_norm = np.full((crop_size, crop_size), float(parent.class_id) / max(1, int(config["DATA"]["NUM_CLASSES"]) - 1), dtype=np.float32)
    area_norm = np.full(
        (crop_size, crop_size),
        min(3.0, float(parent.area) / max(1.0, class_prior_area(int(parent.class_id)))) / 3.0,
        dtype=np.float32,
    )
    channels.extend([parent_crop, dist.astype(np.float32), yy, xx, class_norm, area_norm])
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def clipped_gt_children(parent: Node, gt_nodes: Sequence[Node], config: Mapping[str, Any]) -> List[Node]:
    cfg = config["COMPONENTS"]
    children: List[Node] = []
    for gt in gt_nodes:
        if int(gt.class_id) != int(parent.class_id):
            continue
        inter = np.logical_and(parent.mask, gt.mask)
        inter_pixels = int(inter.sum())
        if inter_pixels < int(cfg["MIN_GT_INTERSECTION_PIXELS"]):
            continue
        if inter_pixels / max(1, int(gt.area)) < float(cfg["MIN_GT_FRACTION_IN_PARENT"]):
            continue
        if inter_pixels / max(1, int(parent.area)) < float(cfg["MIN_PARENT_FRACTION_FOR_CHILD"]):
            continue
        children.append(make_node(len(children) + 1, int(parent.class_id), inter, score=1.0, was_split=True))
    children.sort(key=lambda node: int(node.area), reverse=True)
    return children[: int(cfg["MAX_CHILDREN_TARGET"])]


def build_center_target(children: Sequence[Node], crop_xyxy: Tuple[int, int, int, int], config: Mapping[str, Any]) -> np.ndarray:
    crop_size = int(config["MODEL"]["CROP_SIZE"])
    target = np.zeros((crop_size, crop_size), dtype=np.float32)
    for child in children:
        cy, cx = child.centroid_yx
        yy, xx = raw_to_crop_yx(cy, cx, crop_xyxy, crop_size)
        draw_disk(target, yy, xx, int(config["TRAIN"]["CENTER_RADIUS"]))
    return target


def build_component_samples(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> List[ComponentSample]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
    shape_hw = (int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"]))
    samples: List[ComponentSample] = []
    for record_index, record in enumerate(records):
        sample_dir = sample_dir_from_record(record, raw_root)
        gt_nodes = load_gt_nodes(sample_dir / "gt_hms.npz", num_classes=int(config["DATA"]["NUM_CLASSES"]))
        parent_nodes = rule_nodes_for_record(record, config=config, split_config={"enabled": False})
        raw_features = load_cnabu_features(cnabu_path_from_record(record, raw_root, cnabu_root), raw_shape_hw=shape_hw)
        for parent in parent_nodes:
            children = clipped_gt_children(parent, gt_nodes, config)
            if not children:
                continue
            window = crop_window(parent.mask, margin=int(config["COMPONENTS"]["CROP_MARGIN_PIXELS"]), shape_hw=shape_hw)
            features = component_input_features(raw_features=raw_features, parent=parent, crop_xyxy=window, config=config)
            center_target = build_center_target(children, window, config)
            samples.append(
                ComponentSample(
                    sample_id=str(record["sample_id"]),
                    record_index=int(record_index),
                    parent=parent,
                    gt_children=children,
                    features=features,
                    count_target=min(2, len(children) - 1),
                    center_target=center_target[None],
                    crop_window_xyxy=window,
                )
            )
    return samples


class ComponentDataset:
    def __init__(self, samples: Sequence[ComponentSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ComponentSample:
        return self.samples[index]


class TinyComponentSplitterNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        h = int(hidden_channels)
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, h, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(h, h * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(h * 2, h * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(h * 2, h * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True), nn.Conv2d(h * 4, h * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(h * 4 + h * 2, h * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(h * 2 + h, h, 3, padding=1), nn.ReLU(inplace=True))
        self.center_head = nn.Conv2d(h, 1, 1)
        self.count_head = nn.Linear(h * 4, 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        pooled = F.adaptive_avg_pool2d(e3, (1, 1)).flatten(1)
        up2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        return {"count_logits": self.count_head(pooled), "center_logits": self.center_head(d1)}


def stack_batch(samples: Sequence[ComponentSample], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.tensor(np.stack([sample.features for sample in samples]), dtype=torch.float32, device=device),
        "count": torch.tensor([sample.count_target for sample in samples], dtype=torch.long, device=device),
        "center": torch.tensor(np.stack([sample.center_target for sample in samples]), dtype=torch.float32, device=device),
    }


def compute_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    count_loss = F.cross_entropy(output["count_logits"], batch["count"])
    center_pos = torch.tensor(float(config["TRAIN"]["CENTER_POS_WEIGHT"]), device=batch["features"].device)
    center_loss = F.binary_cross_entropy_with_logits(output["center_logits"], batch["center"], pos_weight=center_pos)
    total = float(config["TRAIN"]["COUNT_LOSS_WEIGHT"]) * count_loss + float(config["TRAIN"]["CENTER_LOSS_WEIGHT"]) * center_loss
    return {"total": total, "count": count_loss.detach(), "center": center_loss.detach()}


def train_model(model: nn.Module, dataset: ComponentDataset, *, config: Mapping[str, Any], iterations: int, device: torch.device) -> List[Dict[str, Any]]:
    if len(dataset) == 0:
        raise ValueError("component dataset is empty")
    rng = random.Random(int(config["TRAIN"]["SEED"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["TRAIN"]["LEARNING_RATE"]), weight_decay=float(config["TRAIN"]["WEIGHT_DECAY"]))
    history: List[Dict[str, Any]] = []
    model.train()
    for step in range(1, int(iterations) + 1):
        items = [dataset[rng.randrange(len(dataset))] for _ in range(int(config["TRAIN"]["BATCH_SIZE"]))]
        batch = stack_batch(items, device)
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
                    "count_loss": float(losses["count"].cpu().item()),
                    "center_loss": float(losses["center"].cpu().item()),
                }
            )
    return history


def extract_centers(center_prob: np.ndarray, parent_crop: np.ndarray, count: int, config: Mapping[str, Any]) -> List[Tuple[float, float, float]]:
    kernel = max(3, int(config["INFERENCE"]["CENTER_LOCAL_MAX_KERNEL"]) | 1)
    pooled = cv2.dilate(center_prob.astype(np.float32), np.ones((kernel, kernel), dtype=np.uint8))
    peak_mask = (
        (center_prob >= float(config["INFERENCE"]["CENTER_THRESHOLD"]))
        & (center_prob >= pooled - 1e-6)
        & (parent_crop > 0.5)
    )
    ys, xs = np.nonzero(peak_mask)
    if ys.size == 0:
        return []
    scores = center_prob[ys, xs]
    order = np.argsort(scores)[::-1]
    centers: List[Tuple[float, float, float]] = []
    min_dist_sq = max(1, int(config["INFERENCE"]["CENTER_NMS_RADIUS"])) ** 2
    for idx in order.tolist():
        y, x, score = float(ys[idx]), float(xs[idx]), float(scores[idx])
        if all((y - cy) ** 2 + (x - cx) ** 2 >= min_dist_sq for cy, cx, _ in centers):
            centers.append((y, x, score))
        if len(centers) >= int(count):
            break
    return centers


def area_passes_prior(mask: np.ndarray, class_id: int, config: Mapping[str, Any]) -> bool:
    if not bool(config["INFERENCE"]["CLASS_SIZE_PRIOR_REJECTION"]):
        return True
    area = int(mask.sum())
    prior = class_prior_area(class_id)
    return (
        area >= prior * float(config["INFERENCE"]["MIN_AREA_PRIOR_MULTIPLIER"])
        and area <= prior * float(config["INFERENCE"]["MAX_AREA_PRIOR_MULTIPLIER"])
    )


def node_bbox_wh(node: Node) -> Tuple[int, int]:
    x1, y1, x2, y2 = [int(value) for value in node.bbox_xyxy_abs]
    return max(0, x2 - x1), max(0, y2 - y1)


def parent_is_suspicious(parent: Node, config: Mapping[str, Any]) -> bool:
    if not bool(config["INFERENCE"].get("REQUIRE_SUSPICIOUS_PARENT", False)):
        return True
    class_id = int(parent.class_id)
    area_ratio = float(parent.area) / max(1.0, class_prior_area(class_id))
    width, height = node_bbox_wh(parent)
    width_ratio = float(width) / max(1.0, class_prior_width(class_id))
    height_ratio = float(height) / max(1.0, class_prior_height(class_id))
    area_ok = area_ratio >= float(config["INFERENCE"].get("SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER", 1.35))
    bbox_ok = (
        width_ratio >= float(config["INFERENCE"].get("SUSPICIOUS_PARENT_BBOX_PRIOR_MULTIPLIER", 1.35))
        or height_ratio >= float(config["INFERENCE"].get("SUSPICIOUS_PARENT_BBOX_PRIOR_MULTIPLIER", 1.35))
    )
    if bool(config["INFERENCE"].get("SUSPICIOUS_PARENT_REQUIRE_AREA", False)):
        return area_ok
    return area_ok or bbox_ok


def centers_are_separated(
    centers: Sequence[Tuple[float, float, float]],
    parent: Node,
    window: Tuple[int, int, int, int],
    config: Mapping[str, Any],
) -> bool:
    if len(centers) < 2:
        return False
    class_id = int(parent.class_id)
    min_prior = min(class_prior_width(class_id), class_prior_height(class_id))
    min_raw_dist = min_prior * float(config["INFERENCE"].get("MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER", 0.35))
    x1, y1, x2, y2 = window
    scale_y = max(1.0, float(config["MODEL"]["CROP_SIZE"])) / max(1.0, float(y2 - y1))
    scale_x = max(1.0, float(config["MODEL"]["CROP_SIZE"])) / max(1.0, float(x2 - x1))
    min_crop_dist = min_raw_dist * min(scale_y, scale_x)
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            dy = float(centers[left][0] - centers[right][0])
            dx = float(centers[left][1] - centers[right][1])
            if math.sqrt(dy * dy + dx * dx) < min_crop_dist:
                return False
    return True


def children_are_balanced(children: Sequence[Node], config: Mapping[str, Any]) -> bool:
    if len(children) < 2:
        return False
    areas = sorted([max(1, int(child.area)) for child in children])
    balance = float(areas[0]) / float(areas[-1])
    return balance >= float(config["INFERENCE"].get("MIN_CHILD_AREA_RATIO", 0.0))


def split_parent_with_model(
    *,
    model: nn.Module,
    raw_features: np.ndarray,
    parent: Node,
    config: Mapping[str, Any],
    device: torch.device,
) -> List[Node]:
    shape_hw = (int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"]))
    window = crop_window(parent.mask, margin=int(config["COMPONENTS"]["CROP_MARGIN_PIXELS"]), shape_hw=shape_hw)
    features = component_input_features(raw_features=raw_features, parent=parent, crop_xyxy=window, config=config)
    with torch.no_grad():
        output = model(torch.tensor(features[None], dtype=torch.float32, device=device))
        count_prob = torch.softmax(output["count_logits"][0], dim=0).detach().cpu().numpy()
        center_prob = torch.sigmoid(output["center_logits"][0, 0]).detach().cpu().numpy()
    best_count_index = int(np.argmax(count_prob))
    split_prob = float(count_prob[1] + count_prob[2])
    if not parent_is_suspicious(parent, config):
        return [parent]
    if best_count_index == 0 or split_prob < float(config["INFERENCE"]["SPLIT_PROB_THRESHOLD"]):
        return [parent]
    if float(count_prob[best_count_index] - count_prob[0]) < float(config["INFERENCE"]["COUNT_MARGIN_THRESHOLD"]):
        return [parent]

    predicted_count = min(int(config["INFERENCE"]["MAX_CHILDREN"]), best_count_index + 1)
    parent_crop = resize_2d(parent.mask.astype(np.float32), window, int(config["MODEL"]["CROP_SIZE"]), is_mask=True)
    centers = extract_centers(center_prob, parent_crop, predicted_count, config)
    if len(centers) < 2:
        return [parent]
    if not centers_are_separated(centers, parent, window, config):
        return [parent]

    x1, y1, x2, y2 = window
    ys, xs = np.nonzero(parent.mask)
    crop_h = max(1.0, float(y2 - y1))
    crop_w = max(1.0, float(x2 - x1))
    crop_size = float(config["MODEL"]["CROP_SIZE"])
    pix_crop_y = (ys.astype(np.float32) - float(y1)) * crop_size / crop_h
    pix_crop_x = (xs.astype(np.float32) - float(x1)) * crop_size / crop_w
    center_array = np.asarray([[cy, cx] for cy, cx, _ in centers], dtype=np.float32)
    pixel_array = np.stack([pix_crop_y, pix_crop_x], axis=1)
    assignments = ((pixel_array[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    children: List[Node] = []
    combined = np.zeros_like(parent.mask, dtype=bool)
    for center_index, (cy, cx, score) in enumerate(centers):
        child_mask = np.zeros_like(parent.mask, dtype=bool)
        selected = assignments == center_index
        child_mask[ys[selected], xs[selected]] = True
        area = int(child_mask.sum())
        if area < int(config["INFERENCE"]["MIN_CHILD_PIXELS"]):
            continue
        if area / max(1, int(parent.area)) < float(config["INFERENCE"]["MIN_CHILD_PARENT_FRACTION"]):
            continue
        if not area_passes_prior(child_mask, int(parent.class_id), config):
            continue
        combined |= child_mask
        children.append(make_node(len(children) + 1, int(parent.class_id), child_mask, score=float(score * split_prob), was_split=True))
    if len(children) < 2:
        return [parent]
    if not children_are_balanced(children, config):
        return [parent]
    if int(combined.sum()) / max(1, int(parent.area)) < float(config["INFERENCE"]["MIN_COMBINED_CHILD_PARENT_COVERAGE"]):
        return [parent]
    return children


def predict_record_nodes(model: nn.Module, record: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> List[Node]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
    raw_features = load_cnabu_features(
        cnabu_path_from_record(record, raw_root, cnabu_root),
        raw_shape_hw=(int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"])),
    )
    parent_nodes = rule_nodes_for_record(record, config=config, split_config={"enabled": False})
    result: List[Node] = []
    for parent in parent_nodes:
        result.extend(split_parent_with_model(model=model, raw_features=raw_features, parent=parent, config=config, device=device))
    return renumber_nodes(result)


def encode_binary_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    clean = np.asarray(mask, dtype=bool)
    flat = clean.astype(np.uint8, copy=False).reshape(-1)
    counts: List[int] = []
    previous = 0
    run_length = 0
    for value in flat.tolist():
        current = int(value)
        if current == previous:
            run_length += 1
        else:
            counts.append(int(run_length))
            run_length = 1
            previous = current
    counts.append(int(run_length))
    return {"shape": [int(clean.shape[0]), int(clean.shape[1])], "order": "C", "counts": counts}


def runtime_node_dict(
    *,
    node_id: int,
    node: Node,
    parent: Node,
    split_accepted: bool,
    child_index: int,
    num_children: int,
    include_mask: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": int(node_id),
        "class_id": int(node.class_id),
        "class_name": str(node.class_name),
        "bbox_xyxy_abs": [int(value) for value in node.bbox_xyxy_abs],
        "centroid_yx": [float(value) for value in node.centroid_yx],
        "area_pixels": int(node.area),
        "score": float(node.score),
        "parent_component_id": int(parent.id),
        "parent_component_bbox_xyxy_abs": [int(value) for value in parent.bbox_xyxy_abs],
        "parent_component_area_pixels": int(parent.area),
        "was_split": bool(split_accepted),
        "split_metadata": {
            "method": "component_conditioned_splitter",
            "accepted": bool(split_accepted),
            "child_index": int(child_index),
            "num_children": int(num_children),
        },
    }
    if include_mask:
        payload["mask"] = encode_binary_mask_rle(node.mask)
    return payload


def predict_record_runtime_nodes(
    model: nn.Module,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    *,
    include_masks: bool = True,
) -> List[Dict[str, Any]]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
    raw_features = load_cnabu_features(
        cnabu_path_from_record(record, raw_root, cnabu_root),
        raw_shape_hw=(int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"])),
    )
    parent_nodes = rule_nodes_for_record(record, config=config, split_config={"enabled": False})
    result: List[Dict[str, Any]] = []
    for parent in parent_nodes:
        children = split_parent_with_model(model=model, raw_features=raw_features, parent=parent, config=config, device=device)
        split_accepted = bool(len(children) > 1 or any(bool(child.was_split) for child in children))
        for child_index, node in enumerate(children, start=1):
            result.append(
                runtime_node_dict(
                    node_id=len(result) + 1,
                    node=node,
                    parent=parent,
                    split_accepted=split_accepted,
                    child_index=child_index if split_accepted else 0,
                    num_children=len(children) if split_accepted else 1,
                    include_mask=include_masks,
                )
            )
    return result


def checkpoint_payload(model: nn.Module, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {
            "class_name": "TinyComponentSplitterNet",
            "in_channels": int(config["MODEL"]["IN_CHANNELS"]),
            "hidden_channels": int(config["MODEL"]["HIDDEN_CHANNELS"]),
        },
        "config": json.loads(json.dumps(config)),
        "metadata": json.loads(json.dumps(metadata, default=json_default)),
    }


def validate_checkpoint_dir_contents(checkpoint_dir: Path) -> None:
    extra = sorted(path.name for path in checkpoint_dir.iterdir() if path.is_file() and path.name not in ALLOWED_CHECKPOINT_FILENAMES)
    if extra:
        raise ValueError(f"checkpoint dir contains files outside the allowed artifact set: {extra}")


def save_runtime_checkpoints(
    *,
    model: nn.Module,
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validate_checkpoint_dir_contents(checkpoint_dir)
    resolved_config = json.loads(json.dumps(config))
    (checkpoint_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    payload = checkpoint_payload(model, resolved_config, metadata)
    torch.save(payload, checkpoint_dir / "model_best_validation.pth")
    torch.save(payload, checkpoint_dir / "model_final.pth")
    metadata_payload = json.loads(json.dumps(metadata, default=json_default))
    metadata_payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_dir": str(checkpoint_dir),
            "allowed_files": sorted(ALLOWED_CHECKPOINT_FILENAMES),
            "optimizer_state_saved": False,
            "model_files": ["model_best_validation.pth", "model_final.pth"],
        }
    )
    (checkpoint_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_checkpoint_dir_contents(checkpoint_dir)
    files = sorted(path.name for path in checkpoint_dir.iterdir() if path.is_file())
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "files": files,
        "optimizer_state_saved": False,
        "model_best_validation": str(checkpoint_dir / "model_best_validation.pth"),
        "model_final": str(checkpoint_dir / "model_final.pth"),
        "config_resolved": str(checkpoint_dir / "config_resolved.yaml"),
        "metadata": str(checkpoint_dir / "checkpoint_metadata.json"),
    }


def load_runtime_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected checkpoint schema in {checkpoint_path}: {payload.get('schema')!r}")
    config = dict(payload["config"])
    model = TinyComponentSplitterNet(int(config["MODEL"]["IN_CHANNELS"]), int(config["MODEL"]["HIDDEN_CHANNELS"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, payload


def evaluate_component_splitter(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    thresholds = [float(value) for value in config["EVAL"]["IOU_THRESHOLDS"]]
    overlap_threshold = float(config["EVAL"]["OVERLAP_FRACTION_THRESHOLD"])
    metrics_by_sample = []
    samples: List[Dict[str, Any]] = []
    model.eval()
    for record in records:
        sample_dir = sample_dir_from_record(record, raw_root)
        gt_nodes = load_gt_nodes(sample_dir / "gt_hms.npz", num_classes=int(config["DATA"]["NUM_CLASSES"]))
        pred_nodes = predict_record_nodes(model, record, config, device)
        metrics = evaluate_nodes(pred_nodes, gt_nodes, thresholds=thresholds, overlap_fraction_threshold=overlap_threshold)
        metrics_by_sample.append(metrics)
        samples.append({"sample_id": str(record["sample_id"]), "record": dict(record), "gt_nodes": gt_nodes, "pred_nodes": pred_nodes, "metrics": metrics})
    return aggregate_evaluations(metrics_by_sample, thresholds), samples


def evaluate_rule_baseline_detailed(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split_config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    thresholds = [float(value) for value in config["EVAL"]["IOU_THRESHOLDS"]]
    overlap_threshold = float(config["EVAL"]["OVERLAP_FRACTION_THRESHOLD"])
    sample_metrics = []
    sample_results = []
    for record in records:
        sample_dir = sample_dir_from_record(record, raw_root)
        gt_nodes = load_gt_nodes(sample_dir / "gt_hms.npz", num_classes=int(config["DATA"]["NUM_CLASSES"]))
        pred_nodes = rule_nodes_for_record(record, config=config, split_config=split_config)
        metrics = evaluate_nodes(pred_nodes, gt_nodes, thresholds=thresholds, overlap_fraction_threshold=overlap_threshold)
        sample_metrics.append(metrics)
        sample_results.append({"sample_id": str(record["sample_id"]), "gt_nodes": gt_nodes, "pred_nodes": pred_nodes, "metrics": metrics})
    return aggregate_evaluations(sample_metrics, thresholds), sample_results


def evaluate_rule_baselines(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    result = {}
    samples = {}
    for name, split_config in (
        ("split_off", {"enabled": False}),
        ("split_on_2d_candidate", {"enabled": True, "method": "candidate_gated_2d_footprint"}),
    ):
        result[name], samples[name] = evaluate_rule_baseline_detailed(records, config, split_config)
    return result, samples


def same_class_merge_subset(
    *,
    sample_results_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Sequence[float],
    split_name: str = "validation",
) -> Dict[str, Any]:
    split_off = list(sample_results_by_variant["split_off"])
    indices = [idx for idx, item in enumerate(split_off) if int(item["metrics"]["merge_indicators"]) > 0]
    subset_eval = {}
    for name, samples in sample_results_by_variant.items():
        metrics = [samples[idx]["metrics"] for idx in indices]
        subset_eval[name] = aggregate_evaluations(metrics, thresholds) if metrics else {
            "samples": 0,
            "pred_count": 0,
            "gt_count": 0,
            "merge_indicators": 0,
            "over_split_indicators": 0,
            "thresholds": {f"{float(t):.2f}": {"matched": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "false_positives": 0, "missed_gt": 0} for t in thresholds},
            "class_wise": {},
        }
    return {"definition": f"{split_name} records where split_off has at least one merge indicator", "num_records": len(indices), "evaluation": subset_eval}


def synthetic_samples(config: Mapping[str, Any]) -> List[ComponentSample]:
    h, w = int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"])
    raw_features = np.zeros((int(config["MODEL"]["BASE_FEATURE_CHANNELS"]), h, w), dtype=np.float32)
    class_id = 9
    parent_mask = np.zeros((h, w), dtype=bool)
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    parent_mask[50:72, 60:108] = True
    left[50:72, 60:80] = True
    right[50:72, 88:108] = True
    raw_features[class_id, parent_mask] = 0.95
    raw_features[14, parent_mask] = 0.95
    raw_features[15, parent_mask] = 0.75
    parent = make_node(1, class_id, parent_mask)
    children = [make_node(1, class_id, left, was_split=True), make_node(2, class_id, right, was_split=True)]
    window = crop_window(parent_mask, margin=int(config["COMPONENTS"]["CROP_MARGIN_PIXELS"]), shape_hw=(h, w))
    return [
        ComponentSample(
            sample_id="synthetic_parent_split",
            record_index=0,
            parent=parent,
            gt_children=children,
            features=component_input_features(raw_features=raw_features, parent=parent, crop_xyxy=window, config=config),
            count_target=1,
            center_target=build_center_target(children, window, config)[None],
            crop_window_xyxy=window,
        )
    ]


def run_synthetic_overfit(config: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    samples = synthetic_samples(config)
    dataset = ComponentDataset(samples)
    model = TinyComponentSplitterNet(int(config["MODEL"]["IN_CHANNELS"]), int(config["MODEL"]["HIDDEN_CHANNELS"])).to(device)
    history = train_model(model, dataset, config=config, iterations=int(config["TRAIN"]["SYNTHETIC_OVERFIT_ITERATIONS"]), device=device)
    sample = samples[0]
    raw_features = np.zeros((int(config["MODEL"]["BASE_FEATURE_CHANNELS"]), int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"])), dtype=np.float32)
    raw_features[int(sample.parent.class_id), sample.parent.mask] = 0.95
    raw_features[14, sample.parent.mask] = 0.95
    raw_features[15, sample.parent.mask] = 0.75
    pred_nodes = split_parent_with_model(model=model, raw_features=raw_features, parent=sample.parent, config=config, device=device)
    metrics = evaluate_nodes(pred_nodes, sample.gt_children, thresholds=[0.25, 0.50], overlap_fraction_threshold=0.20)
    return {"history": history, "metrics": metrics, "pred_count": len(pred_nodes)}


def select_visual_examples(
    sample_results: Sequence[Mapping[str, Any]],
    max_examples: int,
    *,
    reference_results: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[int]:
    scored = []
    for index, item in enumerate(sample_results):
        metrics = item["metrics"]
        reference_bonus = 0
        if reference_results is not None and index < len(reference_results):
            pred_count = len(item["pred_nodes"])
            reference_count = len(reference_results[index]["pred_nodes"])
            reference_bonus = 1000 if pred_count != reference_count else 0
        score = int(metrics["merge_indicators"]) * 10 + int(metrics["thresholds"]["0.25"]["missed_gt"]) - int(metrics["thresholds"]["0.25"]["false_positives"])
        score += reference_bonus
        scored.append((score, index))
    return [index for _score, index in sorted(scored, reverse=True)[: int(max_examples)]]


def write_visuals(
    *,
    output_dir: Path,
    sample_results: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split_name: str = "val",
    reference_sample_results: Optional[Sequence[Mapping[str, Any]]] = None,
    reference_name: str = "previous_component",
) -> List[str]:
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(config["DATA"]["RAW_ROOT"])
    cnabu_root = Path(config["DATA"]["CNABU_ROOT"])
    paths: List[str] = []
    for index in select_visual_examples(
        sample_results,
        int(config["EVAL"]["MAX_VISUAL_EXAMPLES"]),
        reference_results=reference_sample_results,
    ):
        sample = sample_results[index]
        record = records[index]
        cnabu_path = cnabu_path_from_record(record, raw_root, cnabu_root)
        background = build_context_background(str(cnabu_path))
        image_path = examples_dir / f"{split_name}_{sample['sample_id'].replace('/', '_')}.png"
        columns = [
            ("GT", sample["gt_nodes"]),
            ("parent", rule_nodes_for_record(record, config=config, split_config={"enabled": False})),
            ("2D_split", rule_nodes_for_record(record, config=config, split_config={"enabled": True, "method": "candidate_gated_2d_footprint"})),
        ]
        if reference_sample_results is not None and index < len(reference_sample_results):
            columns.append((reference_name, reference_sample_results[index]["pred_nodes"]))
        columns.append(("selected_component", sample["pred_nodes"]))
        render_panel(
            output_path=image_path,
            sample_id=f"{split_name} {sample['sample_id']}",
            background_bgr=background,
            columns=columns,
        )
        paths.append(str(image_path))
    return paths


def previous_global_reference() -> Dict[str, Any]:
    return {
        "val": {"f1_delta_vs_2d": 0.004, "recall_delta_vs_2d": 0.021, "merge_delta_vs_2d": -35, "fp_delta_vs_2d": 32, "over_split_delta_vs_2d": 41},
        "test": {"f1_delta_vs_2d": 0.009, "recall_delta_vs_2d": 0.026, "merge_delta_vs_2d": -43, "fp_delta_vs_2d": 25, "over_split_delta_vs_2d": 43},
    }


def refined_variants() -> List[Dict[str, Any]]:
    return [
        {
            "name": "previous_component_conditioned_splitter",
            "description": "Previous component-conditioned behavior without the new split acceptance controls.",
            "updates": {
                "INFERENCE": {
                    "REQUIRE_SUSPICIOUS_PARENT": False,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": False,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.0,
                    "MIN_CHILD_AREA_RATIO": 0.0,
                }
            },
        },
        {
            "name": "refined_center_margin",
            "description": "Keep parent-size defaults but require more split/count/center confidence before accepting children.",
            "updates": {
                "INFERENCE": {
                    "SPLIT_PROB_THRESHOLD": 0.72,
                    "COUNT_MARGIN_THRESHOLD": 0.22,
                    "CENTER_THRESHOLD": 0.50,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.20,
                    "MIN_CHILD_AREA_RATIO": 0.10,
                }
            },
        },
        {
            "name": "refined_soft_parent_suspicious",
            "description": "Require a mildly suspicious parent footprint and soft child separation/balance checks.",
            "updates": {
                "INFERENCE": {
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": False,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.20,
                    "SUSPICIOUS_PARENT_BBOX_PRIOR_MULTIPLIER": 1.20,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.15,
                    "MIN_CHILD_AREA_RATIO": 0.10,
                }
            },
        },
        {
            "name": "refined_parent_suspicious",
            "description": "Require parent footprint/bbox to be suspicious before any learned split is accepted.",
            "updates": {
                "INFERENCE": {
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": False,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.30,
                    "SUSPICIOUS_PARENT_BBOX_PRIOR_MULTIPLIER": 1.30,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.20,
                    "MIN_CHILD_AREA_RATIO": 0.15,
                }
            },
        },
        {
            "name": "refined_balanced_children",
            "description": "Require suspicious parent, separated centers, and reasonably balanced child areas.",
            "updates": {
                "INFERENCE": {
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": False,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.35,
                    "SUSPICIOUS_PARENT_BBOX_PRIOR_MULTIPLIER": 1.35,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.30,
                    "MIN_CHILD_AREA_RATIO": 0.25,
                    "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.45,
                }
            },
        },
        {
            "name": "refined_strict_evidence",
            "description": "Stronger count/center evidence plus suspicious parent and balanced children.",
            "updates": {
                "INFERENCE": {
                    "SPLIT_PROB_THRESHOLD": 0.75,
                    "COUNT_MARGIN_THRESHOLD": 0.25,
                    "CENTER_THRESHOLD": 0.55,
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": False,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.40,
                    "SUSPICIOUS_PARENT_BBOX_PRIOR_MULTIPLIER": 1.40,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.35,
                    "MIN_CHILD_AREA_RATIO": 0.30,
                    "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.50,
                }
            },
        },
        {
            "name": "refined_area_evidence",
            "description": "Accept learned splits only when parent area itself is larger than the class prior.",
            "updates": {
                "INFERENCE": {
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": True,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.15,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.20,
                    "MIN_CHILD_AREA_RATIO": 0.15,
                }
            },
        },
        {
            "name": "refined_area_margin",
            "description": "Combine area-only parent evidence with stronger count/center confidence.",
            "updates": {
                "INFERENCE": {
                    "SPLIT_PROB_THRESHOLD": 0.72,
                    "COUNT_MARGIN_THRESHOLD": 0.22,
                    "CENTER_THRESHOLD": 0.50,
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": True,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.20,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.25,
                    "MIN_CHILD_AREA_RATIO": 0.20,
                }
            },
        },
        {
            "name": "refined_area_strict",
            "description": "Use stricter area-only parent evidence and larger accepted child footprints.",
            "updates": {
                "INFERENCE": {
                    "SPLIT_PROB_THRESHOLD": 0.75,
                    "COUNT_MARGIN_THRESHOLD": 0.25,
                    "CENTER_THRESHOLD": 0.55,
                    "MIN_CHILD_PARENT_FRACTION": 0.08,
                    "MIN_AREA_PRIOR_MULTIPLIER": 0.15,
                    "REQUIRE_SUSPICIOUS_PARENT": True,
                    "SUSPICIOUS_PARENT_REQUIRE_AREA": True,
                    "SUSPICIOUS_PARENT_AREA_PRIOR_MULTIPLIER": 1.30,
                    "MIN_CENTER_SEPARATION_PRIOR_MULTIPLIER": 0.30,
                    "MIN_CHILD_AREA_RATIO": 0.25,
                    "MIN_COMBINED_CHILD_PARENT_COVERAGE": 0.50,
                }
            },
        },
    ]


def metric_deltas(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    t25 = metrics["thresholds"]["0.25"]
    b25 = baseline["thresholds"]["0.25"]
    return {
        "f1_delta": float(t25["f1"] - b25["f1"]),
        "recall_delta": float(t25["recall"] - b25["recall"]),
        "fp_delta": int(t25["false_positives"] - b25["false_positives"]),
        "miss_delta": int(t25["missed_gt"] - b25["missed_gt"]),
        "merge_delta": int(metrics["merge_indicators"] - baseline["merge_indicators"]),
        "over_split_delta": int(metrics["over_split_indicators"] - baseline["over_split_indicators"]),
    }


def refined_success(metrics: Mapping[str, Any], baseline: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None) -> bool:
    t25 = metrics["thresholds"]["0.25"]
    base_t25 = baseline["thresholds"]["0.25"]
    if not (
        float(t25["f1"]) > float(base_t25["f1"])
        and float(t25["recall"]) > float(base_t25["recall"])
        and int(metrics["merge_indicators"]) < int(baseline["merge_indicators"])
        and int(t25["false_positives"]) <= 20
        and int(metrics["over_split_indicators"]) <= 6
    ):
        return False
    if previous is None:
        return True
    return (
        float(t25["f1"]) >= 0.870
        and int(metrics["merge_indicators"]) <= 20
        and int(metrics["over_split_indicators"]) <= 6
        and int(t25["false_positives"]) <= int(previous["thresholds"]["0.25"]["false_positives"]) + 7
    )


def medium_success(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    t25 = metrics["thresholds"]["0.25"]
    base_t25 = baseline["thresholds"]["0.25"]
    merge_ok = int(metrics["merge_indicators"]) <= 0.8 * max(1, int(baseline["merge_indicators"]))
    fp_ok = int(t25["false_positives"]) <= max(int(base_t25["false_positives"]) + 10, int(round(1.5 * int(base_t25["false_positives"]))))
    over_ok = int(metrics["over_split_indicators"]) <= max(20, 2 * max(1, int(baseline["over_split_indicators"])))
    return bool(
        float(t25["f1"]) > float(base_t25["f1"])
        and float(t25["recall"]) > float(base_t25["recall"])
        and merge_ok
        and fp_ok
        and over_ok
    )


def split_success_for_full1000(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    t25 = metrics["thresholds"]["0.25"]
    base_t25 = baseline["thresholds"]["0.25"]
    merge_delta = int(metrics["merge_indicators"] - baseline["merge_indicators"])
    over_delta = int(metrics["over_split_indicators"] - baseline["over_split_indicators"])
    merge_ok = int(metrics["merge_indicators"]) <= 0.8 * max(1, int(baseline["merge_indicators"]))
    fp_ok = int(t25["false_positives"]) <= max(int(base_t25["false_positives"]) + 10, int(round(1.5 * int(base_t25["false_positives"]))))
    over_ok = over_delta <= max(10, int(round(0.5 * abs(min(0, merge_delta)))))
    return bool(
        float(t25["f1"]) > float(base_t25["f1"])
        and float(t25["recall"]) > float(base_t25["recall"])
        and merge_ok
        and fp_ok
        and over_ok
    )


def split_learned_improves_recall_merge(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    t25 = metrics["thresholds"]["0.25"]
    base_t25 = baseline["thresholds"]["0.25"]
    return bool(
        float(t25["recall"]) > float(base_t25["recall"])
        and int(metrics["merge_indicators"]) < int(baseline["merge_indicators"])
    )


def choose_refined_variant(eval_items: Mapping[str, Any]) -> str:
    baseline = eval_items["split_on_2d_candidate"]
    previous = eval_items.get("previous_component_conditioned_splitter")
    candidates = [name for name in eval_items if name.startswith("refined_")]
    best_name = candidates[0] if candidates else "previous_component_conditioned_splitter"
    best_score = -1e9
    for name in candidates:
        metrics = eval_items[name]
        t25 = metrics["thresholds"]["0.25"]
        deltas = metric_deltas(metrics, baseline)
        success_bonus = 10.0 if refined_success(metrics, baseline, previous) else 0.0
        score = (
            success_bonus
            + 10.0 * deltas["f1_delta"]
            + 2.0 * deltas["recall_delta"]
            - 0.01 * max(0, deltas["fp_delta"])
            - 0.02 * max(0, deltas["over_split_delta"])
            - 0.01 * max(0, deltas["merge_delta"])
            - 0.002 * metrics["over_split_indicators"]
        )
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def recommendation(summary: Mapping[str, Any]) -> List[str]:
    mode = str(summary.get("mode", ""))
    if mode == "full1000":
        lines = ["Selected `component_conditioned_splitter`."]
        all_success = True
        recall_merge_success = True
        for split_name in ("val", "test"):
            learned = summary["evaluation"][split_name]["component_conditioned_splitter"]
            baseline = summary["evaluation"][split_name]["split_on_2d_candidate"]
            learned_t25 = learned["thresholds"]["0.25"]
            base_t25 = baseline["thresholds"]["0.25"]
            f1_delta = learned_t25["f1"] - base_t25["f1"]
            recall_delta = learned_t25["recall"] - base_t25["recall"]
            merge_delta = learned["merge_indicators"] - baseline["merge_indicators"]
            fp_delta = learned_t25["false_positives"] - base_t25["false_positives"]
            over_delta = learned["over_split_indicators"] - baseline["over_split_indicators"]
            lines.append(
                f"{split_name}: component vs 2D F1@0.25 {f1_delta:+.3f}, recall {recall_delta:+.3f}, "
                f"merge {merge_delta:+d}, FP {fp_delta:+d}, over-split {over_delta:+d}."
            )
            all_success = all_success and split_success_for_full1000(learned, baseline)
            recall_merge_success = recall_merge_success and split_learned_improves_recall_merge(learned, baseline)
        if all_success:
            lines.append(
                "Recommendation: proceed to checkpointed runtime-integration planning. "
                "The full 1000-record confirmation supports the original component-conditioned splitter as the learned node-extraction candidate."
            )
        elif recall_merge_success:
            lines.append(
                "Recommendation: revise before runtime default. The learned splitter improves recall/merge, but the full 1000-record tradeoff needs more over-split/FP control before default integration."
            )
        else:
            lines.append(
                "Recommendation: stop learned splitter scaling and keep split_on_2d_candidate as the safest runtime node extraction option."
            )
        if bool(summary.get("safety", {}).get("checkpoint_write", False)):
            lines.append("Runtime checkpoint artifacts were written under the approved checkpoint directory, and MEM runtime integration was not changed.")
        else:
            lines.append("No persistent checkpoints/model files were written, and MEM runtime integration was not changed.")
        return lines

    selected_name = str(summary.get("selected_variant", "component_conditioned_splitter"))
    learned = summary["evaluation"]["val"][selected_name]
    baseline = summary["evaluation"]["val"]["split_on_2d_candidate"]
    previous = summary["evaluation"]["val"].get("previous_component_conditioned_splitter")
    learned_t25 = learned["thresholds"]["0.25"]
    base_t25 = baseline["thresholds"]["0.25"]
    f1_delta = learned_t25["f1"] - base_t25["f1"]
    recall_delta = learned_t25["recall"] - base_t25["recall"]
    merge_delta = learned["merge_indicators"] - baseline["merge_indicators"]
    fp_delta = learned_t25["false_positives"] - base_t25["false_positives"]
    over_delta = learned["over_split_indicators"] - baseline["over_split_indicators"]
    promising = medium_success(learned, baseline) if mode.endswith("medium") else refined_success(learned, baseline, previous)
    lines = [
        f"Selected `{selected_name}`.",
        f"Refined splitter vs 2D: F1@0.25 {f1_delta:+.3f}, recall {recall_delta:+.3f}, merge {merge_delta:+d}, FP {fp_delta:+d}, over-split {over_delta:+d}.",
    ]
    if previous is not None and selected_name != "previous_component_conditioned_splitter":
        prev_t25 = previous["thresholds"]["0.25"]
        lines.append(
            "Refined vs previous component splitter: "
            f"F1@0.25 {learned_t25['f1'] - prev_t25['f1']:+.3f}, "
            f"recall {learned_t25['recall'] - prev_t25['recall']:+.3f}, "
            f"merge {learned['merge_indicators'] - previous['merge_indicators']:+d}, "
            f"FP {learned_t25['false_positives'] - prev_t25['false_positives']:+d}, "
            f"over-split {learned['over_split_indicators'] - previous['over_split_indicators']:+d}."
        )
    if mode.endswith("medium"):
        lines.append(
            "Recommendation: request approval for full 1000 confirmation. The medium subset meets the intended tradeoff."
            if promising
            else "Recommendation: stop/revise before full 1000 confirmation. The medium subset does not meet the intended tradeoff."
        )
    else:
        lines.append(
            "Recommendation: medium run. The small subset meets the intended tradeoff."
            if promising
            else "Recommendation: stop/revise before medium scale. The small subset does not meet the intended tradeoff."
        )
    lines.append("No full 1000-record confirmation was run; it would require approval after stronger small/medium evidence.")
    return lines


def write_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# MEM CNABU Component-Conditioned Splitter",
        "",
        f"Created: `{summary['created_at']}`",
        f"Host: `{summary['host']}`",
        f"Schema: `{summary['schema']}`",
        f"Mode: `{summary['mode']}`",
        f"Output dir: `{summary['output_dir']}`",
        "",
        "## Run",
        "",
        f"- Command: `{summary['command']['argv']}`",
        f"- Device: `{summary['training']['device']}`",
        f"- Iterations: `{summary['training']['iterations']}`",
        f"- Records: train `{summary['records']['train']}`, val `{summary['records']['val']}`, test `{summary['records']['test']}`",
        f"- Component samples: train `{summary['component_samples'].get('train', 0)}`, val `{summary['component_samples'].get('val', 0)}`, test `{summary['component_samples'].get('test', 0)}`",
        f"- Checkpoint used: `{summary['training']['checkpoint_used']}`",
        f"- Checkpoint written: `{summary['safety']['checkpoint_write']}`",
        f"- Selected variant: `{summary.get('selected_variant', 'component_conditioned_splitter')}`",
        "",
    ]
    if summary.get("execution"):
        execution = summary["execution"]
        lines.extend(
            [
                "## Execution Notes",
                "",
                f"- Execution mode: `{execution.get('mode', 'sequential')}`",
                f"- Parallel/subagent runs used: `{execution.get('parallel', False)}`",
                f"- Reason: {execution.get('reason', '')}",
                f"- Training happened: `{execution.get('training_happened', True)}`",
                f"- Checkpoints/model files written: `{summary['safety']['checkpoint_write']}`",
                f"- Dataset/HDF5 exports written: `{summary['safety']['dataset_generation']}`",
                f"- MEM runtime integration changed: `{summary['safety'].get('runtime_integration_changed', False)}`",
                "",
            ]
        )
    if summary.get("checkpoint"):
        checkpoint = summary["checkpoint"]
        lines.extend(
            [
                "## Checkpoint Artifacts",
                "",
                f"- Checkpoint directory: `{checkpoint['checkpoint_dir']}`",
                f"- Files: `{', '.join(checkpoint['files'])}`",
                f"- Optimizer state saved: `{checkpoint['optimizer_state_saved']}`",
                f"- Best validation model: `{checkpoint['model_best_validation']}`",
                f"- Final model: `{checkpoint['model_final']}`",
                "",
            ]
        )
    for split_name, split_eval in summary["evaluation"].items():
        title = "Validation" if split_name == "val" else split_name.capitalize()
        lines.extend(
            [
                f"## Metrics - {title}",
                "",
                "| Variant | Pred | GT | P@0.25 | R@0.25 | F1@0.25 | FP@0.25 | Miss@0.25 | P@0.50 | R@0.50 | F1@0.50 | FP@0.50 | Miss@0.50 | Merge | Over-split |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in split_eval.items():
            t25 = metrics["thresholds"]["0.25"]
            t50 = metrics["thresholds"]["0.50"]
            lines.append(
                "| {name} | {pred} | {gt} | {p:.3f} | {r:.3f} | {f:.3f} | {fp} | {miss} | {p50:.3f} | {r50:.3f} | {f50:.3f} | {fp50} | {miss50} | {merge} | {over} |".format(
                    name=name,
                    pred=metrics["pred_count"],
                    gt=metrics["gt_count"],
                    p=t25["precision"],
                    r=t25["recall"],
                    f=t25["f1"],
                    fp=t25["false_positives"],
                    miss=t25["missed_gt"],
                    p50=t50["precision"],
                    r50=t50["recall"],
                    f50=t50["f1"],
                    fp50=t50["false_positives"],
                    miss50=t50["missed_gt"],
                    merge=metrics["merge_indicators"],
                    over=metrics["over_split_indicators"],
                )
            )
    if summary.get("loaded_checkpoint_evaluation"):
        lines.extend(
            [
                "",
                "## Loaded Checkpoint Metrics",
                "",
                "| Split | P@0.25 | R@0.25 | F1@0.25 | FP@0.25 | Miss@0.25 | F1@0.50 | Merge | Over-split | F1 delta vs in-process | Recall delta | FP delta | Merge delta | Over delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for split_name, metrics in summary["loaded_checkpoint_evaluation"].items():
            t25 = metrics["thresholds"]["0.25"]
            t50 = metrics["thresholds"]["0.50"]
            deltas = (summary.get("loaded_checkpoint_deltas") or {}).get(split_name, {})
            lines.append(
                f"| {split_name} | {t25['precision']:.3f} | {t25['recall']:.3f} | {t25['f1']:.3f} | {t25['false_positives']} | {t25['missed_gt']} | {t50['f1']:.3f} | {metrics['merge_indicators']} | {metrics['over_split_indicators']} | {float(deltas.get('f1_25_delta', 0.0)):+.6f} | {float(deltas.get('recall_25_delta', 0.0)):+.6f} | {int(deltas.get('fp_25_delta', 0)):+d} | {int(deltas.get('merge_delta', 0)):+d} | {int(deltas.get('over_split_delta', 0)):+d} |"
            )
    if summary.get("variants"):
        lines.extend(["", "## Controls Tested", ""])
        for variant in summary["variants"]:
            lines.append(f"- `{variant['name']}`: {variant.get('description', '')}")
            lines.append(f"  - Settings: `{json.dumps(variant.get('updates', {}), sort_keys=True)}`")
    lines.extend(["", "## Previous Global Learned Reference", ""])
    lines.extend(["| Split | F1 delta vs 2D | Recall delta | Merge delta | FP delta | Over-split delta |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for split_name, item in summary["previous_global_reference"].items():
        lines.append(f"| {split_name} | {item['f1_delta_vs_2d']:+.3f} | {item['recall_delta_vs_2d']:+.3f} | {item['merge_delta_vs_2d']:+d} | {item['fp_delta_vs_2d']:+d} | {item['over_split_delta_vs_2d']:+d} |")
    lines.extend(["", "## Class Breakdown", ""])
    lines.extend(["| Class | Variant | Pred | GT | R@0.25 | F1@0.25 | Merge | Over-split |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    class_variant_names = ["split_off", "split_on_2d_candidate"]
    for optional_name in ("previous_component_conditioned_splitter", "refined_strict_evidence", "component_conditioned_splitter"):
        if any(optional_name in split_eval for split_eval in summary["evaluation"].values()):
            class_variant_names.append(optional_name)
    for split_name, split_eval in summary["evaluation"].items():
        title = "Validation" if split_name == "val" else split_name.capitalize()
        class_ids = sorted(
            {
                class_id
                for name in class_variant_names
                for class_id in split_eval.get(name, {}).get("class_wise", {}).keys()
            },
            key=lambda value: int(value),
        )
        for class_id in class_ids:
            for name in class_variant_names:
                item = split_eval.get(name, {}).get("class_wise", {}).get(class_id)
                if not item:
                    continue
                t25 = item["thresholds"]["0.25"]
                lines.append(f"| {title}: {item['class_name']} | {name} | {item['pred_count']} | {item['gt_count']} | {t25['recall']:.3f} | {t25['f1']:.3f} | {item['merge_indicators']} | {item['over_split_indicators']} |")
    if "same_class_merge_subset" in summary:
        lines.extend(["", "## Same-Class Merge Subset", ""])
        subset_payload = summary["same_class_merge_subset"]
        subset_by_split = subset_payload if all(isinstance(value, Mapping) and "evaluation" in value for value in subset_payload.values()) else {"val": subset_payload}
        for split_name, subset in subset_by_split.items():
            title = "Validation" if split_name == "val" else split_name.capitalize()
            lines.append(f"Subset definition: {subset['definition']}. Records: `{subset['num_records']}`.")
            lines.extend(
                [
                    "",
                    f"| {title} Variant | Pred | GT | R@0.25 | F1@0.25 | FP@0.25 | Miss@0.25 | Merge | Over-split |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, metrics in subset["evaluation"].items():
                t25 = metrics["thresholds"]["0.25"]
                lines.append(
                    f"| {name} | {metrics['pred_count']} | {metrics['gt_count']} | {t25['recall']:.3f} | {t25['f1']:.3f} | {t25['false_positives']} | {t25['missed_gt']} | {metrics['merge_indicators']} | {metrics['over_split_indicators']} |"
                )
            lines.append("")
    lines.extend(["", "## Visual Diagnostics", ""])
    if summary.get("visual_diagnostics_description"):
        lines.append(str(summary["visual_diagnostics_description"]))
        lines.append("")
    visual_examples = summary.get("visual_examples", [])
    if isinstance(visual_examples, Mapping):
        for split_name, paths in visual_examples.items():
            title = "Validation" if split_name == "val" else split_name.capitalize()
            lines.append(f"{title}:")
            for path in paths:
                lines.append(f"- `{path}`")
    else:
        for path in visual_examples:
            lines.append(f"- `{path}`")
    lines.extend(["", "## Recommendation", ""])
    lines.extend(summary["recommendation"])
    lines.extend(["", "## Safety", ""])
    lines.append("- The learned splitter only outputs child masks clipped inside the original CNABU parent component.")
    lines.append("- GT instance masks/classes are used only for offline targets/evaluation, not runtime input.")
    if bool(summary["safety"].get("checkpoint_write", False)):
        lines.append("- Checkpoint writing was explicitly enabled for this run; dataset generation, HDF5 export, staging, and commit were not performed.")
    else:
        lines.append("- No checkpoint write, dataset generation, HDF5 export, staging, or commit was performed.")
    lines.append(f"- Full 1000 confirmation run performed: `{summary['safety'].get('full_1000_confirmation', False)}`.")
    lines.append(f"- MEM runtime integration changed: `{summary['safety'].get('runtime_integration_changed', False)}`.")
    return "\n".join(lines)


def run_train_eval(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    device: torch.device,
    mode: str,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    set_seed(int(config["TRAIN"]["SEED"]))
    records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
    split_manifest = read_json(Path(config["DATA"]["SPLIT_JSON"]))
    refined_mode = mode.startswith("refined_")
    subset_mode = mode.replace("refined_", "")
    if subset_mode == "small":
        max_train = int(config["SUBSETS"]["SMALL_MAX_TRAIN"])
        max_val = int(config["SUBSETS"]["SMALL_MAX_VAL"])
        max_test = int(config["SUBSETS"]["SMALL_MAX_TEST"])
        iterations = int(config["TRAIN"]["SMALL_ITERATIONS"])
    elif subset_mode == "medium":
        max_train = int(config["SUBSETS"]["MEDIUM_MAX_TRAIN"])
        max_val = int(config["SUBSETS"]["MEDIUM_MAX_VAL"])
        max_test = int(config["SUBSETS"]["MEDIUM_MAX_TEST"])
        iterations = int(config["TRAIN"]["MEDIUM_ITERATIONS"])
    elif subset_mode == "full1000":
        max_train = int(config["SUBSETS"]["FULL1000_MAX_TRAIN"])
        max_val = int(config["SUBSETS"]["FULL1000_MAX_VAL"])
        max_test = int(config["SUBSETS"]["FULL1000_MAX_TEST"])
        iterations = int(config["TRAIN"]["FULL1000_ITERATIONS"])
    else:
        raise ValueError(f"unknown train/eval mode: {mode}")
    records_by_split = split_records(records, split_manifest, max_train=max_train, max_val=max_val, max_test=max_test)
    train_samples = build_component_samples(records_by_split["train"], config)
    component_sample_counts: Dict[str, int] = {"train": len(train_samples)}
    for split_name in ("val", "test"):
        component_sample_counts[split_name] = len(build_component_samples(records_by_split[split_name], config)) if records_by_split[split_name] else 0
    model = TinyComponentSplitterNet(int(config["MODEL"]["IN_CHANNELS"]), int(config["MODEL"]["HIDDEN_CHANNELS"])).to(device)
    history = train_model(model, ComponentDataset(train_samples), config=config, iterations=iterations, device=device)
    variants = (
        refined_variants()
        if refined_mode
        else [
        {"name": "component_conditioned_splitter", "description": "Current component-conditioned splitter defaults.", "updates": {}}
        ]
    )
    evaluation: Dict[str, Dict[str, Any]] = {}
    sample_results_by_split: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    sample_results_by_name_by_split: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    eval_split_names = [split_name for split_name in ("val", "test") if records_by_split[split_name]]
    selected_variant = "component_conditioned_splitter"
    for split_name in eval_split_names:
        baselines, baseline_samples = evaluate_rule_baselines(records_by_split[split_name], config)
        eval_items = dict(baselines)
        sample_results_by_variant = dict(baseline_samples)
        sample_results_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for variant in variants:
            variant_config = config_with_updates(config, variant.get("updates", {}))
            metrics, samples = evaluate_component_splitter(model, records_by_split[split_name], variant_config, device)
            eval_items[str(variant["name"])] = metrics
            sample_results_by_variant[str(variant["name"])] = samples
            sample_results_by_name[str(variant["name"])] = samples
        if refined_mode and split_name == "val":
            selected_variant = choose_refined_variant(eval_items)
        if refined_mode:
            eval_items["component_conditioned_splitter"] = eval_items[selected_variant]
            sample_results_by_variant["component_conditioned_splitter"] = sample_results_by_name[selected_variant]
        evaluation[split_name] = eval_items
        sample_results_by_split[split_name] = sample_results_by_variant
        sample_results_by_name_by_split[split_name] = sample_results_by_name
    selected_config = config_with_updates(
        config,
        next((variant.get("updates", {}) for variant in variants if variant["name"] == selected_variant), {}),
    )
    visual_paths_by_split: Dict[str, List[str]] = {}
    for split_name in eval_split_names:
        sample_results = sample_results_by_split[split_name]["component_conditioned_splitter"]
        reference_samples = sample_results_by_name_by_split[split_name].get("previous_component_conditioned_splitter") if refined_mode else None
        visual_paths_by_split[split_name] = write_visuals(
            output_dir=output_dir,
            sample_results=sample_results,
            records=records_by_split[split_name],
            config=selected_config,
            split_name=split_name,
            reference_sample_results=reference_samples,
            reference_name="previous_component",
        )
    checkpoint_info: Optional[Dict[str, Any]] = None
    loaded_checkpoint_evaluation: Optional[Dict[str, Any]] = None
    loaded_checkpoint_deltas: Optional[Dict[str, Any]] = None
    if checkpoint_dir is not None:
        checkpoint_metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "script": str(Path(__file__).resolve()),
            "mode": mode,
            "seed": int(config["TRAIN"]["SEED"]),
            "records": {"train": len(records_by_split["train"]), "val": len(records_by_split["val"]), "test": len(records_by_split["test"])},
            "component_samples": component_sample_counts,
            "training_iterations": int(iterations),
            "selected_variant": selected_variant,
            "class_names": [class_name(class_id) for class_id in range(int(config["DATA"]["NUM_CLASSES"]))],
            "input_channel_specification": {
                "base_feature_channels": int(config["MODEL"]["BASE_FEATURE_CHANNELS"]),
                "in_channels": int(config["MODEL"]["IN_CHANNELS"]),
                "crop_size": int(config["MODEL"]["CROP_SIZE"]),
                "extra_channels": ["parent_mask", "parent_distance_transform", "crop_y", "crop_x", "parent_class_norm", "parent_area_norm"],
            },
            "preprocessing": {
                "component_crop_margin_pixels": int(config["COMPONENTS"]["CROP_MARGIN_PIXELS"]),
                "raw_shape_hw": [int(config["DATA"]["HEIGHT"]), int(config["DATA"]["WIDTH"])],
            },
            "split_acceptance_thresholds": dict(config["INFERENCE"]),
            "in_process_component_metrics": {
                split_name: evaluation[split_name]["component_conditioned_splitter"]
                for split_name in eval_split_names
            },
        }
        checkpoint_info = save_runtime_checkpoints(
            model=model,
            config=selected_config,
            checkpoint_dir=checkpoint_dir,
            metadata=checkpoint_metadata,
        )
        loaded_model, loaded_config, _payload = load_runtime_checkpoint(Path(checkpoint_info["model_best_validation"]), device)
        loaded_checkpoint_evaluation = {}
        loaded_checkpoint_deltas = {}
        for split_name in eval_split_names:
            loaded_metrics, _loaded_samples = evaluate_component_splitter(loaded_model, records_by_split[split_name], loaded_config, device)
            loaded_checkpoint_evaluation[split_name] = loaded_metrics
            reference_metrics = evaluation[split_name]["component_conditioned_splitter"]
            loaded_t25 = loaded_metrics["thresholds"]["0.25"]
            ref_t25 = reference_metrics["thresholds"]["0.25"]
            loaded_checkpoint_deltas[split_name] = {
                "f1_25_delta": float(loaded_t25["f1"] - ref_t25["f1"]),
                "recall_25_delta": float(loaded_t25["recall"] - ref_t25["recall"]),
                "fp_25_delta": int(loaded_t25["false_positives"] - ref_t25["false_positives"]),
                "merge_delta": int(loaded_metrics["merge_indicators"] - reference_metrics["merge_indicators"]),
                "over_split_delta": int(loaded_metrics["over_split_indicators"] - reference_metrics["over_split_indicators"]),
            }
    visual_description = (
        "Panels show GT child masks, split_off parent components, 2D splitter output, previous component splitter output, "
        "and the selected refined output; previous-vs-selected differences are the refined accepted/rejected split decisions."
        if refined_mode
        else "Panels show GT child masks, split_off parent components, 2D splitter output, and component splitter output."
    )
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "output_dir": str(output_dir),
        "command": {"argv": " ".join([str(Path(sys.executable)), *sys.argv]), "cwd": str(Path.cwd())},
        "config": config,
        "records": {"train": len(records_by_split["train"]), "val": len(records_by_split["val"]), "test": len(records_by_split["test"])},
        "component_samples": component_sample_counts,
        "training": {"device": str(device), "iterations": iterations, "history": history, "checkpoint_used": False},
        "evaluation": evaluation,
        "loaded_checkpoint_evaluation": loaded_checkpoint_evaluation,
        "loaded_checkpoint_deltas": loaded_checkpoint_deltas,
        "selected_variant": selected_variant,
        "variants": variants,
        "previous_global_reference": previous_global_reference(),
        "same_class_merge_subset": {
            split_name: same_class_merge_subset(
                sample_results_by_variant=sample_results_by_split[split_name],
                thresholds=[float(value) for value in config["EVAL"]["IOU_THRESHOLDS"]],
                split_name=split_name,
            )
            for split_name in eval_split_names
        },
        "visual_examples": visual_paths_by_split if len(visual_paths_by_split) > 1 else next(iter(visual_paths_by_split.values()), []),
        "visual_diagnostics_description": visual_description,
        "checkpoint": checkpoint_info,
        "execution": {
            "mode": "sequential single job",
            "parallel": False,
            "reason": "The component variants share one trained in-memory model; separate parallel trainings would not be a controlled comparison.",
            "training_happened": True,
        },
        "safety": {
            "parent_component_constrained_output": True,
            "gt_used_for_runtime_inference_input": False,
            "gt_used_for_training_targets": True,
            "checkpoint_write": checkpoint_info is not None,
            "dataset_generation": False,
            "full_1000_confirmation": bool(subset_mode == "full1000"),
            "runtime_integration_changed": False,
        },
    }
    summary["recommendation"] = recommendation(summary)
    return summary


def run_one_record_overfit(config: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
    split_manifest = read_json(Path(config["DATA"]["SPLIT_JSON"]))
    records_by_split = split_records(records, split_manifest, max_train=1, max_val=1, max_test=0)
    samples = build_component_samples(records_by_split["train"], config)
    model = TinyComponentSplitterNet(int(config["MODEL"]["IN_CHANNELS"]), int(config["MODEL"]["HIDDEN_CHANNELS"])).to(device)
    history = train_model(model, ComponentDataset(samples), config=config, iterations=int(config["TRAIN"]["ONE_RECORD_OVERFIT_ITERATIONS"]), device=device)
    metrics, _ = evaluate_component_splitter(model, records_by_split["train"], config, device)
    return {"sample_id": records_by_split["train"][0]["sample_id"], "component_samples": len(samples), "history": history, "metrics": metrics}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("import_smoke", "synthetic_overfit", "one_record_overfit", "small", "medium", "refined_small", "refined_medium", "full1000"),
        default="small",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = apply_overrides(yaml.safe_load(args.config_file.read_text(encoding="utf-8")), args.override)
    if args.device is not None:
        config["TRAIN"]["DEVICE"] = str(args.device)
    device = torch.device(str(config["TRAIN"]["DEVICE"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = str(config["OUTPUT"]["RUN_PREFIX"])
    if str(args.mode).startswith("refined_") and not run_prefix.endswith("_refined"):
        run_prefix = f"{run_prefix}_refined"
    if str(args.mode) == "full1000" and not run_prefix.endswith("_full1000"):
        run_prefix = f"{run_prefix}_full1000"
    output_dir = args.output_dir or (Path(config["OUTPUT"]["ROOT"]) / f"{run_prefix}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    if args.mode == "import_smoke":
        records = read_records(Path(config["DATA"]["RECORDS_JSON"]))
        payload: Dict[str, Any] = {
            "schema": "mem_cnabu_component_conditioned_splitter_command_v0",
            "mode": args.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "output_dir": str(output_dir),
            "result": {"records_loaded": len(records), "torch_cuda_available": bool(torch.cuda.is_available())},
            "safety": {"checkpoint_write": False, "dataset_generation": False},
        }
    elif args.mode == "synthetic_overfit":
        payload = {
            "schema": "mem_cnabu_component_conditioned_splitter_command_v0",
            "mode": args.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "output_dir": str(output_dir),
            "result": run_synthetic_overfit(config, device),
            "safety": {"checkpoint_write": False, "dataset_generation": False},
        }
    elif args.mode == "one_record_overfit":
        payload = {
            "schema": "mem_cnabu_component_conditioned_splitter_command_v0",
            "mode": args.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "output_dir": str(output_dir),
            "result": run_one_record_overfit(config, device),
            "safety": {"checkpoint_write": False, "dataset_generation": False},
        }
    else:
        payload = run_train_eval(config, output_dir=output_dir, device=device, mode=args.mode, checkpoint_dir=args.checkpoint_dir)
    payload.setdefault("command", {})["argv"] = " ".join([str(Path(sys.executable)), *sys.argv])
    payload.setdefault("command", {})["cwd"] = str(Path.cwd())
    payload["timing_seconds"] = {"total": float(time.time() - started)}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    if "evaluation" in payload:
        (output_dir / "summary.md").write_text(write_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "output_dir": str(output_dir)}, sort_keys=True))
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
            "was_split": value.was_split,
        }
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
