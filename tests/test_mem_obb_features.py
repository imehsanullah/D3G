"""Tests for OBB-from-mask mapper helper and OBB-aware model/pair-geometry path.

Synthetic CPU-only; no real data, no training, no checkpoints.
"""

import math
import unittest

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances

from data.mem_observed_gt_mapper import (
    _bbox_xyxy_abs_from_mask,
    _obb_cxcywht_from_mask,
    _obb_cxcywht_from_xyxy,
    _obb_tensor,
)
from models.mem_graph_dense import (
    MemGraphDenseKnownNodes,
    MemKnownNodeTokenExtractor,
    build_mem_pair_geometry_features,
)
from utils.configs import add_dep_graph_config, add_detr_config


def _axis_aligned_mask(height: int, width: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((height, width), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def _diagonal_thick_mask(size: int = 60, half_thickness: int = 2) -> np.ndarray:
    """Approximately +45-degree thick stripe (long axis along y=x)."""
    m = np.zeros((size, size), dtype=np.uint8)
    for k in range(-half_thickness, half_thickness + 1):
        for i in range(size):
            j = i + k
            if 0 <= j < size:
                m[i, j] = 1
    return m


class MemObbHelperTest(unittest.TestCase):
    def test_obb_from_horizontal_aabb_recovers_zero_theta(self):
        mask = _axis_aligned_mask(20, 30, x1=5, y1=4, x2=25, y2=10)
        cx, cy, w, h, theta = _obb_cxcywht_from_mask(mask)
        self.assertAlmostEqual(w, 19.0, places=1)
        self.assertAlmostEqual(h, 5.0, places=1)
        self.assertAlmostEqual(theta, 0.0, places=3)
        self.assertAlmostEqual(cx, 14.5, places=1)
        self.assertAlmostEqual(cy, 6.5, places=1)

    def test_obb_from_vertical_aabb_recovers_quarter_pi_or_minus(self):
        mask = _axis_aligned_mask(30, 20, x1=8, y1=4, x2=12, y2=26)
        _, _, w, h, theta = _obb_cxcywht_from_mask(mask)
        # Long side is the vertical 21px run; short side is the 3px run.
        self.assertAlmostEqual(w, 21.0, places=1)
        self.assertAlmostEqual(h, 3.0, places=1)
        # Vertical -> theta is +/- pi/2; sin(2*theta) ~ 0 in either case.
        self.assertAlmostEqual(math.sin(2.0 * theta), 0.0, places=3)
        self.assertAlmostEqual(math.cos(2.0 * theta), -1.0, places=3)

    def test_obb_from_diagonal_thick_mask_has_45deg_long_axis(self):
        mask = _diagonal_thick_mask(size=60, half_thickness=2)
        _, _, w, h, theta = _obb_cxcywht_from_mask(mask)
        self.assertGreater(w, h)
        # +/- 45deg both give |sin(2 theta)| ~ 1 and cos(2 theta) ~ 0.
        self.assertAlmostEqual(abs(math.sin(2.0 * theta)), 1.0, delta=0.05)
        self.assertAlmostEqual(math.cos(2.0 * theta), 0.0, delta=0.05)

    def test_obb_lifted_from_xyxy_has_zero_theta_and_matches_centers(self):
        boxes = torch.tensor([[0.0, 0.0, 4.0, 6.0], [10.0, 8.0, 12.0, 12.0]])
        obb = _obb_cxcywht_from_xyxy(boxes)
        self.assertEqual(tuple(obb.shape), (2, 5))
        self.assertTrue(torch.allclose(obb[:, 4], torch.zeros(2)))
        self.assertAlmostEqual(float(obb[0, 0]), 2.0)
        self.assertAlmostEqual(float(obb[0, 1]), 3.0)
        self.assertAlmostEqual(float(obb[0, 2]), 4.0)
        self.assertAlmostEqual(float(obb[0, 3]), 6.0)

    def test_obb_tensor_shape_validation(self):
        good = [[1.0, 2.0, 5.0, 3.0, 0.0], [4.0, 4.0, 6.0, 2.0, 0.5]]
        tensor = _obb_tensor(good, num_nodes=2)
        self.assertEqual(tuple(tensor.shape), (2, 5))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            _obb_tensor([[1.0, 2.0]], num_nodes=1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _obb_tensor([[1.0, 2.0, -1.0, 3.0, 0.0]], num_nodes=1)


class MemObbModelTest(unittest.TestCase):
    def _instances_with_obb(self, *, height=40, width=60, with_obb=True):
        boxes = torch.tensor(
            [
                [2.0, 2.0, 12.0, 16.0],
                [14.0, 4.0, 30.0, 22.0],
                [32.0, 1.0, 56.0, 14.0],
            ],
            dtype=torch.float32,
        )
        inst = Instances((height, width))
        inst.gt_boxes = Boxes(boxes)
        inst.gt_classes = torch.tensor([0, 5, 11], dtype=torch.int64)
        if with_obb:
            obb = torch.tensor(
                [
                    [7.0, 9.0, 10.0, 14.0, 0.0],
                    [22.0, 13.0, 16.0, 18.0, 0.3],
                    [44.0, 7.5, 24.0, 13.0, -0.5],
                ],
                dtype=torch.float32,
            )
            inst.set("gt_obb_cxcywht", obb)
        return inst

    def test_node_token_extractor_in_obb_mode_consumes_sin_cos_theta(self):
        extractor = MemKnownNodeTokenExtractor(
            feature_dim=8,
            hidden_dim=8,
            num_object_classes=14,
            pooler_resolution=2,
            use_obb_features=True,
        )
        # Expect a 6-input geometry MLP (4 normalized box + sin2t + cos2t).
        first_linear = extractor.box_geometry_mlp[0]
        self.assertEqual(first_linear.in_features, 6)
        inst = self._instances_with_obb()
        features = torch.zeros(1, 8, 5, 6)
        tokens = extractor.forward(features, [inst])[0]
        self.assertEqual(tuple(tokens.shape), (3, 8))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_node_token_extractor_aabb_mode_does_not_require_obb_field(self):
        extractor = MemKnownNodeTokenExtractor(
            feature_dim=8,
            hidden_dim=8,
            num_object_classes=14,
            pooler_resolution=2,
            use_obb_features=False,
        )
        self.assertEqual(extractor.box_geometry_mlp[0].in_features, 4)
        inst = self._instances_with_obb(with_obb=False)
        features = torch.zeros(1, 8, 5, 6)
        tokens = extractor.forward(features, [inst])[0]
        self.assertEqual(tuple(tokens.shape), (3, 8))

    def test_node_token_extractor_obb_mode_raises_without_field(self):
        extractor = MemKnownNodeTokenExtractor(
            feature_dim=8,
            hidden_dim=8,
            num_object_classes=14,
            pooler_resolution=2,
            use_obb_features=True,
        )
        inst = self._instances_with_obb(with_obb=False)
        features = torch.zeros(1, 8, 5, 6)
        with self.assertRaisesRegex(ValueError, "gt_obb_cxcywht"):
            extractor.forward(features, [inst])

    def test_pair_geometry_obb_features_are_finite_and_correct_shape(self):
        inst = self._instances_with_obb()
        names = [
            "front_x_overlap_union",
            "relative_theta_sin",
            "relative_theta_cos",
            "obb_aspect_ratio_min_over_max",
        ]
        pair = build_mem_pair_geometry_features(inst, names)
        self.assertEqual(tuple(pair.shape), (3, 3, 4))
        self.assertTrue(torch.isfinite(pair).all())
        # Diagonal of relative_theta_sin should be zero (theta_i - theta_i = 0).
        diag_sin = torch.diagonal(pair[..., 1], dim1=0, dim2=1)
        self.assertTrue(torch.allclose(diag_sin, torch.zeros_like(diag_sin)))
        # Diagonal of relative_theta_cos should be one.
        diag_cos = torch.diagonal(pair[..., 2], dim1=0, dim2=1)
        self.assertTrue(torch.allclose(diag_cos, torch.ones_like(diag_cos)))

    def test_pair_geometry_obb_feature_requires_obb_field(self):
        inst = self._instances_with_obb(with_obb=False)
        with self.assertRaisesRegex(ValueError, "gt_obb_cxcywht"):
            build_mem_pair_geometry_features(inst, ["relative_theta_sin"])


class MemObbFullModelTest(unittest.TestCase):
    def _cfg(self, *, hidden_dim=16, use_obb_features=True):
        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.META_ARCHITECTURE = "MemGraphDenseKnownNodes"
        cfg.MODEL.GRAPH_HEAD.NAME = "GraphTransformerDense"
        cfg.MODEL.GRAPH_HEAD.EDGE_FEATURES = "concat"
        cfg.MODEL.GRAPH_HEAD.HIDDEN_DIM = hidden_dim
        cfg.MODEL.GRAPH_HEAD.NUM_HEADS = 2
        cfg.MODEL.GRAPH_HEAD.NUM_LAYERS = 1
        cfg.MODEL.MEM_GRAPH.IN_CHANNELS = 30
        cfg.MODEL.MEM_GRAPH.HIDDEN_DIM = hidden_dim
        cfg.MODEL.MEM_GRAPH.NUM_OBJECT_CLASSES = 14
        cfg.MODEL.MEM_GRAPH.INPUT_NORMALIZATION = "scaled_v0"
        cfg.MODEL.MEM_GRAPH.POOLER_RESOLUTION = 2
        cfg.MODEL.MEM_GRAPH.MASK_GRAPH_DIAGONAL = True
        cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED = True
        cfg.MODEL.MEM_GRAPH.USE_OBB_FEATURES = bool(use_obb_features)
        cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES = [
            "front_x_overlap_union",
            "relative_theta_sin",
            "relative_theta_cos",
            "obb_aspect_ratio_min_over_max",
        ]
        cfg.INPUT.MEM_EXPECTED_HEIGHT = 40
        cfg.INPUT.MEM_EXPECTED_WIDTH = 60
        cfg.INPUT.MEM_BOX_MODE = "obb_from_mask"
        return cfg

    def test_full_model_with_obb_features_runs_eval_forward(self):
        cfg = self._cfg()
        model = MemGraphDenseKnownNodes(cfg).eval()
        inst = Instances((40, 60))
        inst.gt_boxes = Boxes(
            torch.tensor(
                [
                    [2.0, 2.0, 12.0, 16.0],
                    [14.0, 4.0, 30.0, 22.0],
                    [32.0, 1.0, 56.0, 14.0],
                ],
                dtype=torch.float32,
            )
        )
        inst.gt_classes = torch.tensor([0, 5, 11], dtype=torch.int64)
        inst.set(
            "gt_obb_cxcywht",
            torch.tensor(
                [
                    [7.0, 9.0, 10.0, 14.0, 0.0],
                    [22.0, 13.0, 16.0, 18.0, 0.3],
                    [44.0, 7.5, 24.0, 13.0, -0.5],
                ],
                dtype=torch.float32,
            ),
        )
        item = {
            "image": torch.zeros(30, 40, 60),
            "instances": inst,
            "image_id": "synthetic",
            "mem_metadata": {"node_order_instance_ids": [1, 2, 3]},
        }
        with torch.no_grad():
            out = model([item])
        self.assertEqual(len(out), 1)
        graph_logits = out[0]["graph_logits"]
        self.assertEqual(tuple(graph_logits.shape), (3, 3))
        self.assertTrue(torch.isfinite(graph_logits).all())
        self.assertTrue(out[0]["use_obb_features"])

    def test_obb_config_yaml_parses_and_sets_expected_keys(self):
        from pathlib import Path

        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "mem" / "option2b_observed_visible_nodes_pair_geometry_obb.yaml"
        cfg.merge_from_file(str(cfg_path))
        self.assertEqual(cfg.INPUT.MEM_BOX_MODE, "obb_from_mask")
        self.assertTrue(cfg.MODEL.MEM_GRAPH.USE_OBB_FEATURES)
        self.assertTrue(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        for required in ("relative_theta_sin", "relative_theta_cos", "obb_aspect_ratio_min_over_max"):
            self.assertIn(required, cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES)


if __name__ == "__main__":
    unittest.main()
