"""MEM known-node dense graph model for Option 2A thesis experiments.

This module is intentionally small and independent from the RGB/DETR image path.
It consumes MEM map tensors from ``MemObservedGtMapper`` and predicts a direct
``blocks_access_to`` dense adjacency over mapper-provided known/GT nodes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Instances

from models.detr_gheads import DenseGraphTransformerHead


TensorOrLoss = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Union[int, float, List[int]]]]]

SUPPORTED_MEM_PAIR_GEOMETRY_FEATURES = (
    "source_in_front",
    "signed_dy",
    "signed_dx",
    "x_overlap_min",
    "x_overlap_union",
    "y_overlap_min",
    "y_gap_norm",
    "area_ratio_min_over_max",
    "area_ratio",
    "front_x_overlap_union",
    "front_x_overlap",
)


def _check_dense_graph_target(target: torch.Tensor, *, require_zero_diagonal: bool) -> None:
    if target.shape[-1] != target.shape[-2]:
        raise ValueError(f"graph target must be square, got shape {tuple(target.shape)}")
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("graph target must be binary with values 0/1")
    if require_zero_diagonal:
        diagonal = torch.diagonal(target, dim1=-2, dim2=-1)
        if not torch.all(diagonal == 0):
            raise ValueError("graph target diagonal/self-loops must be zero")


def masked_dense_graph_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mask_diagonal: bool = True,
    require_zero_diagonal: bool = True,
    pos_weight: Optional[float] = None,
    return_diagnostics: bool = False,
) -> TensorOrLoss:
    """Compute direct BCE for known-node dense graph logits.

    ``target[i, j] = 1`` is interpreted directly as node ``i`` blocking node
    ``j``. No transpose, sign flip, or Hungarian/query transport is applied.
    """

    original_shape = list(target.shape)
    if logits.shape != target.shape:
        raise ValueError(
            f"graph logits and target must have the same shape, got {tuple(logits.shape)} and {tuple(target.shape)}"
        )
    if logits.dim() not in (2, 3):
        raise ValueError(f"graph logits must have shape [N,N] or [B,N,N], got {tuple(logits.shape)}")

    target = target.to(device=logits.device, dtype=logits.dtype)
    _check_dense_graph_target(target, require_zero_diagonal=require_zero_diagonal)

    mask = torch.ones_like(target, dtype=torch.bool)
    if mask_diagonal:
        diagonal = torch.eye(target.shape[-1], dtype=torch.bool, device=target.device)
        if target.dim() == 3:
            diagonal = diagonal.unsqueeze(0).expand(target.shape[0], -1, -1)
        mask = mask & ~diagonal

    if not torch.any(mask):
        raise ValueError("graph loss mask selected zero node pairs")

    selected_logits = logits[mask]
    selected_target = target[mask]
    pos_weight_tensor = None
    if pos_weight is not None and float(pos_weight) > 0.0:
        pos_weight_tensor = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(selected_logits, selected_target, pos_weight=pos_weight_tensor)

    if not return_diagnostics:
        return loss

    num_pairs = int(mask.sum().item())
    num_positive_edges = int(selected_target.sum().item())
    diagnostics: Dict[str, Union[int, float, List[int]]] = {
        "target_shape": original_shape,
        "num_pairs": num_pairs,
        "num_positive_edges": num_positive_edges,
        "positive_edge_ratio": float(num_positive_edges / num_pairs),
    }
    return loss, diagnostics


def node_binary_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    if logits.dim() != 1:
        raise ValueError(f"node binary logits must have shape [N], got {tuple(logits.shape)}")
    target = target.to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        raise ValueError(
            f"node binary logits and target must have the same shape, got {tuple(logits.shape)} and {tuple(target.shape)}"
        )
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("node binary target must be binary with values 0/1")
    pos_weight_tensor = None
    if pos_weight is not None and float(pos_weight) > 0.0:
        pos_weight_tensor = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)


class MemInputNormalizer(nn.Module):
    """Normalize MEM map tensors while preserving the mapper channel contract."""

    def __init__(self, mode: str = "scaled_v0", semantic_max_value: float = 14.0) -> None:
        super().__init__()
        self.mode = mode
        self.semantic_max_value = float(semantic_max_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"MEM input must have shape [B,C,H,W], got {tuple(x.shape)}")
        x = x.float()
        if self.mode in ("none", "identity"):
            return x
        if self.mode != "scaled_v0":
            raise ValueError(f"unsupported MEM input normalization mode: {self.mode}")
        if self.semantic_max_value <= 0:
            raise ValueError("semantic_max_value must be positive")
        if x.shape[1] % 3 != 0:
            raise ValueError(
                f"scaled_v0 expects view-major MEM channels in groups of 3, got {x.shape[1]} channels"
            )
        y = x.clone()
        y[:, 2::3, :, :] = y[:, 2::3, :, :] / self.semantic_max_value
        return y


def _group_count(num_channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class MemMapEncoder(nn.Module):
    """Small CNN that directly consumes MEM map tensors."""

    def __init__(self, in_channels: int = 30, hidden_dim: int = 256) -> None:
        super().__init__()
        c1 = max(16, hidden_dim // 4)
        c2 = max(16, hidden_dim // 2)
        self.out_channels = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_count(c1), c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c2), c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"MEM map encoder expects [B,C,H,W], got {tuple(x.shape)}")
        return self.net(x)


class MemKnownNodeTokenExtractor(nn.Module):
    """Build ordered known-node tokens from MEM feature maps and GT instances.

    The v0 pooling path intentionally uses simple integer crop + adaptive average
    pooling. This keeps the first scaffold CPU-testable and avoids tying the
    thesis baseline to DETR/RGB backbone assumptions. It can be replaced with
    ROIAlign later if needed.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_object_classes: int = 14,
        pooler_resolution: int = 3,
    ) -> None:
        super().__init__()
        if pooler_resolution <= 0:
            raise ValueError("pooler_resolution must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_object_classes = int(num_object_classes)
        self.pooler_resolution = int(pooler_resolution)
        pooled_dim = self.feature_dim * self.pooler_resolution * self.pooler_resolution
        self.pool_proj = nn.Linear(pooled_dim, self.hidden_dim)
        self.class_embedding = nn.Embedding(self.num_object_classes, self.hidden_dim)
        self.box_geometry_mlp = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, features: torch.Tensor, instances_list: Sequence[Instances]) -> List[torch.Tensor]:
        return self.forward_with_options(features, instances_list, zero_map_node_features=False)

    def forward_with_options(
        self,
        features: torch.Tensor,
        instances_list: Sequence[Instances],
        *,
        zero_map_node_features: bool = False,
    ) -> List[torch.Tensor]:
        if features.dim() != 4:
            raise ValueError(f"features must have shape [B,C,Hf,Wf], got {tuple(features.shape)}")
        if features.shape[0] != len(instances_list):
            raise ValueError(
                f"feature batch size {features.shape[0]} does not match {len(instances_list)} instances objects"
            )
        return [
            self._tokens_for_one_image(
                features[index],
                instances,
                zero_map_node_features=zero_map_node_features,
            )
            for index, instances in enumerate(instances_list)
        ]

    def _tokens_for_one_image(
        self,
        feature: torch.Tensor,
        instances: Instances,
        *,
        zero_map_node_features: bool = False,
    ) -> torch.Tensor:
        if not instances.has("gt_boxes") or not instances.has("gt_classes"):
            raise ValueError("known-node MEM graph model requires instances.gt_boxes and instances.gt_classes")
        boxes = instances.gt_boxes.tensor.to(device=feature.device, dtype=feature.dtype)
        classes = instances.gt_classes.to(device=feature.device, dtype=torch.long)
        if boxes.shape[0] != classes.shape[0]:
            raise ValueError("instances.gt_boxes and instances.gt_classes must have the same length")
        if boxes.shape[0] == 0:
            return feature.new_zeros((0, self.hidden_dim))
        if torch.any(classes < 0) or torch.any(classes >= self.num_object_classes):
            raise ValueError(
                f"MEM gt_classes must be in [0, {self.num_object_classes - 1}], got {classes.detach().cpu().tolist()}"
            )

        class_tokens = self.class_embedding(classes)
        geometry_tokens = self.box_geometry_mlp(self._normalized_box_geometry(boxes, instances.image_size))
        if zero_map_node_features:
            pooled_tokens = torch.zeros_like(class_tokens)
        else:
            pooled = torch.stack([self._pool_box(feature, box, instances.image_size) for box in boxes], dim=0)
            pooled_tokens = self.pool_proj(pooled.flatten(1))
        return pooled_tokens + class_tokens + geometry_tokens

    def _pool_box(
        self,
        feature: torch.Tensor,
        box: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        channels, feature_height, feature_width = feature.shape
        image_height, image_width = int(image_size[0]), int(image_size[1])
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"invalid instances image_size: {image_size}")

        scale_x = feature_width / float(image_width)
        scale_y = feature_height / float(image_height)
        x1 = int(torch.floor(box[0] * scale_x).item())
        y1 = int(torch.floor(box[1] * scale_y).item())
        x2 = int(torch.ceil(box[2] * scale_x).item())
        y2 = int(torch.ceil(box[3] * scale_y).item())

        x1 = max(0, min(x1, feature_width - 1))
        y1 = max(0, min(y1, feature_height - 1))
        x2 = max(x1 + 1, min(x2, feature_width))
        y2 = max(y1 + 1, min(y2, feature_height))

        crop = feature[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            crop = feature.new_zeros((channels, 1, 1))
        pooled = F.adaptive_avg_pool2d(crop.unsqueeze(0), (self.pooler_resolution, self.pooler_resolution))
        return pooled.squeeze(0)

    def _normalized_box_geometry(self, boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        image_height, image_width = float(image_size[0]), float(image_size[1])
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        width = (x2 - x1).clamp(min=0.0)
        height = (y2 - y1).clamp(min=0.0)
        cx = x1 + 0.5 * width
        cy = y1 + 0.5 * height
        denom = boxes.new_tensor([image_width, image_height, image_width, image_height]).clamp(min=1.0)
        return torch.stack([cx, cy, width, height], dim=1) / denom


def _normalise_pair_geometry_feature_names(feature_names: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(feature_names, str):
        names = [name.strip() for name in feature_names.split(",") if name.strip()]
    else:
        names = [str(name).strip() for name in feature_names if str(name).strip()]
    if not names:
        raise ValueError("PAIR_GEOMETRY_FEATURES must contain at least one feature when pair geometry is enabled")
    unknown = sorted(set(names) - set(SUPPORTED_MEM_PAIR_GEOMETRY_FEATURES))
    if unknown:
        raise ValueError(
            "unsupported MEM pair geometry feature(s): {}; supported features: {}".format(
                unknown,
                list(SUPPORTED_MEM_PAIR_GEOMETRY_FEATURES),
            )
        )
    return tuple(names)


def build_mem_pair_geometry_features(instances: Instances, feature_names: Sequence[str]) -> torch.Tensor:
    """Build directed pair geometry features from mapper-provided node boxes.

    The returned tensor has shape ``[N, N, F]``. Entry ``[i, j]`` describes the
    directed pair source ``i`` -> target ``j`` using normalized box geometry in
    image coordinates, where lower y values are interpreted as closer to shelf
    access/front.
    """

    names = _normalise_pair_geometry_feature_names(feature_names)
    if not instances.has("gt_boxes"):
        raise ValueError("pair geometry requires instances.gt_boxes")
    boxes = instances.gt_boxes.tensor
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"pair geometry expects gt_boxes tensor [N,4], got {tuple(boxes.shape)}")
    image_height, image_width = float(instances.image_size[0]), float(instances.image_size[1])
    if image_height <= 0.0 or image_width <= 0.0:
        raise ValueError(f"invalid instances image_size for pair geometry: {instances.image_size}")

    boxes = boxes.to(dtype=torch.float32)
    num_nodes = int(boxes.shape[0])
    if num_nodes == 0:
        return boxes.new_zeros((0, 0, len(names)))

    eps = torch.finfo(boxes.dtype).eps
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    widths = (x2 - x1).clamp(min=eps)
    heights = (y2 - y1).clamp(min=eps)
    areas = widths * heights
    cx = x1 + 0.5 * widths
    cy = y1 + 0.5 * heights

    source_x1, target_x1 = x1[:, None], x1[None, :]
    source_y1, target_y1 = y1[:, None], y1[None, :]
    source_x2, target_x2 = x2[:, None], x2[None, :]
    source_y2, target_y2 = y2[:, None], y2[None, :]
    source_widths, target_widths = widths[:, None], widths[None, :]
    source_heights, target_heights = heights[:, None], heights[None, :]
    source_areas, target_areas = areas[:, None], areas[None, :]
    source_cx, target_cx = cx[:, None], cx[None, :]
    source_cy, target_cy = cy[:, None], cy[None, :]

    x_overlap = (torch.minimum(source_x2, target_x2) - torch.maximum(source_x1, target_x1)).clamp(min=0.0)
    x_overlap_min = x_overlap / torch.minimum(source_widths, target_widths).clamp(min=eps)
    x_span = (torch.maximum(source_x2, target_x2) - torch.minimum(source_x1, target_x1)).clamp(min=eps)
    x_overlap_union = x_overlap / x_span

    y_overlap = (torch.minimum(source_y2, target_y2) - torch.maximum(source_y1, target_y1)).clamp(min=0.0)
    y_overlap_min = y_overlap / torch.minimum(source_heights, target_heights).clamp(min=eps)
    y_gap_norm = (
        torch.maximum(source_y1, target_y1) - torch.minimum(source_y2, target_y2)
    ).clamp(min=0.0) / max(image_height, 1.0)

    area_ratio = torch.minimum(source_areas, target_areas) / torch.maximum(source_areas, target_areas).clamp(min=eps)
    source_in_front = (source_cy < target_cy).to(dtype=boxes.dtype)
    signed_dy = (source_cy - target_cy) / max(image_height, 1.0)
    signed_dx = (source_cx - target_cx) / max(image_width, 1.0)
    front_x_overlap_union = source_in_front * x_overlap_union

    feature_values = {
        "source_in_front": source_in_front,
        "signed_dy": signed_dy,
        "signed_dx": signed_dx,
        "x_overlap_min": x_overlap_min,
        "x_overlap_union": x_overlap_union,
        "y_overlap_min": y_overlap_min,
        "y_gap_norm": y_gap_norm,
        "area_ratio_min_over_max": area_ratio,
        "area_ratio": area_ratio,
        "front_x_overlap_union": front_x_overlap_union,
        "front_x_overlap": front_x_overlap_union,
    }
    return torch.stack([feature_values[name] for name in names], dim=-1)


@META_ARCH_REGISTRY.register()
class MemGraphDenseKnownNodes(nn.Module):
    """Known-node MEM dense graph predictor for Option 2A."""

    def __init__(self, cfg) -> None:
        super().__init__()
        mem_cfg = cfg.MODEL.MEM_GRAPH
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.mask_graph_diagonal = bool(mem_cfg.MASK_GRAPH_DIAGONAL)
        self.graph_loss_pos_weight = float(mem_cfg.GRAPH_LOSS_POS_WEIGHT)
        self.require_known_nodes = bool(mem_cfg.REQUIRE_KNOWN_NODES)
        self.zero_map_node_features = bool(mem_cfg.ZERO_MAP_NODE_FEATURES)
        self.pair_geometry_enabled = bool(mem_cfg.PAIR_GEOMETRY_ENABLED)
        self.visible_blocks_hidden_aux_enabled = bool(mem_cfg.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)
        self.visible_blocks_hidden_aux_loss_weight = float(mem_cfg.VISIBLE_BLOCKS_HIDDEN_AUX_LOSS_WEIGHT)
        self.visible_blocks_hidden_aux_pos_weight = float(mem_cfg.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT)
        self.pair_geometry_feature_names: Tuple[str, ...] = (
            _normalise_pair_geometry_feature_names(mem_cfg.PAIR_GEOMETRY_FEATURES)
            if self.pair_geometry_enabled
            else tuple()
        )
        self.normalizer = MemInputNormalizer(
            mode=mem_cfg.INPUT_NORMALIZATION,
            semantic_max_value=float(mem_cfg.SEMANTIC_MAX_VALUE),
        )
        self.encoder = MemMapEncoder(in_channels=int(mem_cfg.IN_CHANNELS), hidden_dim=int(mem_cfg.HIDDEN_DIM))
        self.node_token_extractor = MemKnownNodeTokenExtractor(
            feature_dim=int(mem_cfg.HIDDEN_DIM),
            hidden_dim=int(mem_cfg.HIDDEN_DIM),
            num_object_classes=int(mem_cfg.NUM_OBJECT_CLASSES),
            pooler_resolution=int(mem_cfg.POOLER_RESOLUTION),
        )
        graph_cfg = cfg.MODEL.GRAPH_HEAD
        if graph_cfg.NAME != "GraphTransformerDense":
            raise ValueError(
                "MemGraphDenseKnownNodes v0 expects MODEL.GRAPH_HEAD.NAME='GraphTransformerDense'"
            )
        self.graph_head = DenseGraphTransformerHead(
            num_layers=int(graph_cfg.NUM_LAYERS),
            in_dim=int(mem_cfg.HIDDEN_DIM),
            hidden_dim=int(graph_cfg.HIDDEN_DIM),
            num_heads=int(graph_cfg.NUM_HEADS),
            edge_features=graph_cfg.EDGE_FEATURES,
            extra_edge_feature_dim=len(self.pair_geometry_feature_names),
        )
        if self.visible_blocks_hidden_aux_enabled:
            self.visible_blocks_hidden_head = nn.Linear(int(mem_cfg.HIDDEN_DIM), 1)
        self.to(self.device)

    def forward(self, batched_inputs: Sequence[Dict[str, object]]):
        if len(batched_inputs) != 1:
            raise NotImplementedError("MemGraphDenseKnownNodes v0 supports batch size 1 before node padding is added")

        item = batched_inputs[0]
        image = item["image"].to(self.device).float()
        if image.dim() != 3:
            raise ValueError(f"item['image'] must have shape [C,H,W], got {tuple(image.shape)}")
        instances = item["instances"].to(self.device)
        if self.require_known_nodes and (not instances.has("gt_boxes") or not instances.has("gt_classes")):
            raise ValueError("MemGraphDenseKnownNodes requires mapper-provided known GT boxes/classes")

        if self.zero_map_node_features:
            features = image.new_zeros((1, int(self.encoder.out_channels), 1, 1))
        else:
            features = self.encoder(self.normalizer(image.unsqueeze(0)))
        node_tokens = self.node_token_extractor.forward_with_options(
            features,
            [instances],
            zero_map_node_features=self.zero_map_node_features,
        )[0]
        num_nodes = int(node_tokens.shape[0])
        if num_nodes == 0:
            raise ValueError("MemGraphDenseKnownNodes requires at least one known node in v0")

        pair_geometry_features = None
        if self.pair_geometry_enabled:
            pair_geometry_features = build_mem_pair_geometry_features(
                instances,
                self.pair_geometry_feature_names,
            ).to(device=self.device, dtype=node_tokens.dtype)

        graph_logits = self._predict_graph_logits(node_tokens, pair_geometry_features=pair_geometry_features)
        visible_blocks_hidden_logits: Optional[torch.Tensor] = None
        if self.visible_blocks_hidden_aux_enabled:
            visible_blocks_hidden_logits = self._predict_visible_blocks_hidden_logits(node_tokens)
        if self.training:
            target = item.get("graph_gt", item.get("dense_gt"))
            if target is None:
                raise ValueError("training inputs must contain graph_gt or dense_gt")
            target = target.to(self.device)
            loss = masked_dense_graph_bce_loss(
                graph_logits,
                target,
                mask_diagonal=self.mask_graph_diagonal,
                require_zero_diagonal=True,
                pos_weight=self.graph_loss_pos_weight,
                return_diagnostics=False,
            )
            losses = {"loss_mem_dense_graph": loss}
            if self.visible_blocks_hidden_aux_enabled and visible_blocks_hidden_logits is not None:
                aux_weight = float(self.visible_blocks_hidden_aux_loss_weight)
                if "visible_blocks_hidden_target" in item:
                    visible_target = item["visible_blocks_hidden_target"].to(self.device)
                    losses["loss_mem_visible_blocks_hidden_aux"] = aux_weight * node_binary_bce_loss(
                        visible_blocks_hidden_logits,
                        visible_target,
                        pos_weight=self.visible_blocks_hidden_aux_pos_weight,
                    )
            return losses

        metadata = item.get("mem_metadata", {}) or {}
        graph_probs = torch.sigmoid(graph_logits)
        output = {
            "image_id": item.get("image_id"),
            "num_nodes": num_nodes,
            "graph_logits": graph_logits,
            "graph_probs": graph_probs,
            "node_order_instance_ids": metadata.get("node_order_instance_ids", []),
            "pair_geometry_feature_names": list(self.pair_geometry_feature_names),
            "zero_map_node_features": self.zero_map_node_features,
        }
        if self.visible_blocks_hidden_aux_enabled and visible_blocks_hidden_logits is not None:
            output["visible_blocks_hidden_logits"] = visible_blocks_hidden_logits
            output["visible_blocks_hidden_probs"] = torch.sigmoid(visible_blocks_hidden_logits)
        return [output]

    def _predict_graph_logits(
        self,
        node_tokens: torch.Tensor,
        *,
        pair_geometry_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hs = node_tokens.unsqueeze(0).unsqueeze(0)  # [L=1, B=1, Q, C]
        graph_logits = self.graph_head(hs, edge_extra_features=pair_geometry_features)[-1, 0, :, :, 0]
        return graph_logits

    def _predict_visible_blocks_hidden_logits(self, node_tokens: torch.Tensor) -> torch.Tensor:
        return self.visible_blocks_hidden_head(node_tokens).squeeze(-1)
