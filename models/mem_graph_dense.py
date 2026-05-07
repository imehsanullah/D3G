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
        if features.dim() != 4:
            raise ValueError(f"features must have shape [B,C,Hf,Wf], got {tuple(features.shape)}")
        if features.shape[0] != len(instances_list):
            raise ValueError(
                f"feature batch size {features.shape[0]} does not match {len(instances_list)} instances objects"
            )
        return [self._tokens_for_one_image(features[index], instances) for index, instances in enumerate(instances_list)]

    def _tokens_for_one_image(self, feature: torch.Tensor, instances: Instances) -> torch.Tensor:
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

        pooled = torch.stack([self._pool_box(feature, box, instances.image_size) for box in boxes], dim=0)
        pooled_tokens = self.pool_proj(pooled.flatten(1))
        class_tokens = self.class_embedding(classes)
        geometry_tokens = self.box_geometry_mlp(self._normalized_box_geometry(boxes, instances.image_size))
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
        )
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

        features = self.encoder(self.normalizer(image.unsqueeze(0)))
        node_tokens = self.node_token_extractor(features, [instances])[0]
        num_nodes = int(node_tokens.shape[0])
        if num_nodes == 0:
            raise ValueError("MemGraphDenseKnownNodes requires at least one known node in v0")

        graph_logits = self._predict_graph_logits(node_tokens)
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
            return {"loss_mem_dense_graph": loss}

        metadata = item.get("mem_metadata", {}) or {}
        graph_probs = torch.sigmoid(graph_logits)
        return [
            {
                "image_id": item.get("image_id"),
                "num_nodes": num_nodes,
                "graph_logits": graph_logits,
                "graph_probs": graph_probs,
                "node_order_instance_ids": metadata.get("node_order_instance_ids", []),
            }
        ]

    def _predict_graph_logits(self, node_tokens: torch.Tensor) -> torch.Tensor:
        hs = node_tokens.unsqueeze(0).unsqueeze(0)  # [L=1, B=1, Q, C]
        graph_logits = self.graph_head(hs)[-1, 0, :, :, 0]
        return graph_logits
