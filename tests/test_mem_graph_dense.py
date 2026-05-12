import unittest
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances

from models.mem_graph_dense import (
    MemGraphDenseKnownNodes,
    MemKnownNodeTokenExtractor,
    MemMapEncoder,
    build_mem_pair_geometry_features,
    masked_dense_graph_bce_loss,
    node_binary_bce_loss,
)
from utils.configs import add_dep_graph_config, add_detr_config


class MemGraphDenseTest(unittest.TestCase):
    def _cfg(
        self,
        *,
        height=32,
        width=40,
        hidden_dim=16,
        pair_geometry=False,
        zero_map_node_features=False,
        visible_blocks_hidden_aux=False,
        visible_blocks_hidden_aux_pos_weight=2.0,
    ):
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
        cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED = pair_geometry
        cfg.MODEL.MEM_GRAPH.ZERO_MAP_NODE_FEATURES = zero_map_node_features
        cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED = visible_blocks_hidden_aux
        cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT = visible_blocks_hidden_aux_pos_weight
        cfg.INPUT.MEM_EXPECTED_HEIGHT = height
        cfg.INPUT.MEM_EXPECTED_WIDTH = width
        return cfg

    def _instances(self, *, height=32, width=40):
        instances = Instances((height, width))
        instances.gt_boxes = Boxes(
            torch.tensor(
                [
                    [1.0, 2.0, 12.0, 16.0],
                    [10.0, 6.0, 28.0, 24.0],
                    [20.0, 1.0, 38.0, 14.0],
                ],
                dtype=torch.float32,
            )
        )
        instances.gt_classes = torch.tensor([0, 5, 13], dtype=torch.int64)
        return instances

    def test_masked_dense_graph_loss_preserves_direct_edge_direction(self):
        target = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        direct_logits = torch.tensor([[-8.0, 8.0], [-8.0, -8.0]])
        transposed_logits = torch.tensor([[-8.0, -8.0], [8.0, -8.0]])

        direct_loss, direct_stats = masked_dense_graph_bce_loss(
            direct_logits,
            target,
            mask_diagonal=True,
            return_diagnostics=True,
        )
        transposed_loss, transposed_stats = masked_dense_graph_bce_loss(
            transposed_logits,
            target,
            mask_diagonal=True,
            return_diagnostics=True,
        )

        self.assertLess(float(direct_loss), 0.01)
        self.assertGreater(float(transposed_loss), float(direct_loss) + 7.0)
        self.assertEqual(direct_stats["num_pairs"], 2)
        self.assertEqual(direct_stats["num_positive_edges"], 1)
        self.assertEqual(transposed_stats["target_shape"], [2, 2])

    def test_masked_dense_graph_loss_rejects_non_binary_targets_and_self_loops(self):
        logits = torch.zeros(2, 2)

        with self.assertRaisesRegex(ValueError, "binary"):
            masked_dense_graph_bce_loss(logits, torch.tensor([[0.0, 0.5], [0.0, 0.0]]))

        with self.assertRaisesRegex(ValueError, "diagonal"):
            masked_dense_graph_bce_loss(logits, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))

    def test_mem_map_encoder_accepts_mem_tensor_and_returns_spatial_features(self):
        torch.manual_seed(0)
        encoder = MemMapEncoder(in_channels=30, hidden_dim=32)
        x = torch.randn(2, 30, 140, 200)

        features = encoder(x)

        self.assertEqual(features.shape, torch.Size([2, 32, 35, 50]))
        self.assertTrue(torch.isfinite(features).all())

    def test_known_node_token_extractor_uses_instance_order(self):
        torch.manual_seed(0)
        features = torch.arange(1 * 16 * 8 * 10, dtype=torch.float32).reshape(1, 16, 8, 10)
        instances = self._instances(height=32, width=40)
        first_only = Instances((32, 40))
        first_only.gt_boxes = Boxes(instances.gt_boxes.tensor[:1].clone())
        first_only.gt_classes = instances.gt_classes[:1].clone()
        extractor = MemKnownNodeTokenExtractor(
            feature_dim=16,
            hidden_dim=16,
            num_object_classes=14,
            pooler_resolution=2,
        )
        extractor.eval()

        tokens = extractor(features, [instances])[0]
        first_token_again = extractor(features, [first_only])[0][0]

        self.assertEqual(tokens.shape, torch.Size([3, 16]))
        self.assertTrue(torch.allclose(tokens[0], first_token_again))
        self.assertFalse(torch.allclose(tokens[0], tokens[1]))

    def test_mem_pair_geometry_features_are_directed_and_normalized(self):
        instances = Instances((100, 100))
        instances.gt_boxes = Boxes(
            torch.tensor(
                [
                    [10.0, 10.0, 30.0, 30.0],
                    [10.0, 40.0, 30.0, 60.0],
                    [50.0, 10.0, 70.0, 30.0],
                ],
                dtype=torch.float32,
            )
        )
        feature_names = [
            "source_in_front",
            "signed_dy",
            "signed_dx",
            "x_overlap_min",
            "x_overlap_union",
            "y_overlap_min",
            "y_gap_norm",
            "area_ratio_min_over_max",
            "front_x_overlap_union",
        ]

        features = build_mem_pair_geometry_features(instances, feature_names)
        index = {name: idx for idx, name in enumerate(feature_names)}

        self.assertEqual(features.shape, torch.Size([3, 3, len(feature_names)]))
        self.assertAlmostEqual(float(features[0, 1, index["source_in_front"]]), 1.0)
        self.assertAlmostEqual(float(features[1, 0, index["source_in_front"]]), 0.0)
        self.assertAlmostEqual(float(features[0, 1, index["signed_dy"]]), -0.3, places=6)
        self.assertAlmostEqual(float(features[1, 0, index["signed_dy"]]), 0.3, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["signed_dx"]]), 0.0, places=6)
        self.assertAlmostEqual(float(features[0, 2, index["signed_dx"]]), -0.4, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["x_overlap_min"]]), 1.0, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["x_overlap_union"]]), 1.0, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["y_overlap_min"]]), 0.0, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["y_gap_norm"]]), 0.1, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["area_ratio_min_over_max"]]), 1.0, places=6)
        self.assertAlmostEqual(float(features[0, 1, index["front_x_overlap_union"]]), 1.0, places=6)
        self.assertAlmostEqual(float(features[1, 0, index["front_x_overlap_union"]]), 0.0, places=6)

    def test_mem_graph_dense_known_nodes_returns_synthetic_training_loss(self):
        torch.manual_seed(0)
        cfg = self._cfg(hidden_dim=16)
        model = MemGraphDenseKnownNodes(cfg)
        model.train()
        item = {
            "image": torch.randn(30, 32, 40),
            "height": 32,
            "width": 40,
            "image_id": "synthetic/000000000",
            "instances": self._instances(height=32, width=40),
            "graph_gt": torch.tensor(
                [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                dtype=torch.long,
            ),
        }

        losses = model([item])

        self.assertEqual(set(losses.keys()), {"loss_mem_dense_graph"})
        self.assertEqual(losses["loss_mem_dense_graph"].ndim, 0)
        self.assertTrue(torch.isfinite(losses["loss_mem_dense_graph"]))

    def test_mem_graph_dense_known_nodes_eval_returns_direct_graph_logits(self):
        torch.manual_seed(0)
        cfg = self._cfg(hidden_dim=16)
        model = MemGraphDenseKnownNodes(cfg)
        model.eval()
        item = {
            "image": torch.randn(30, 32, 40),
            "height": 32,
            "width": 40,
            "image_id": "synthetic/000000001",
            "instances": self._instances(height=32, width=40),
            "graph_gt": torch.zeros(3, 3, dtype=torch.long),
            "mem_metadata": {"node_order_instance_ids": [101, 102, 103]},
        }

        with torch.no_grad():
            outputs = model([item])

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["image_id"], "synthetic/000000001")
        self.assertEqual(outputs[0]["num_nodes"], 3)
        self.assertEqual(outputs[0]["node_order_instance_ids"], [101, 102, 103])
        self.assertEqual(outputs[0]["graph_logits"].shape, torch.Size([3, 3]))
        self.assertEqual(outputs[0]["graph_probs"].shape, torch.Size([3, 3]))
        self.assertNotIn("visible_blocks_hidden_logits", outputs[0])
        self.assertTrue(torch.isfinite(outputs[0]["graph_logits"]).all())

    def test_visible_blocks_hidden_aux_forward_logits_are_config_gated(self):
        torch.manual_seed(0)
        cfg = self._cfg(hidden_dim=16, pair_geometry=True, visible_blocks_hidden_aux=True)
        model = MemGraphDenseKnownNodes(cfg)
        model.eval()
        item = {
            "image": torch.randn(30, 32, 40),
            "height": 32,
            "width": 40,
            "image_id": "synthetic/visible_hidden_aux_eval",
            "instances": self._instances(height=32, width=40),
            "graph_gt": torch.zeros(3, 3, dtype=torch.long),
            "visible_blocks_hidden_target": torch.tensor([0.0, 1.0, 1.0]),
        }

        with torch.no_grad():
            outputs = model([item])

        self.assertEqual(outputs[0]["visible_blocks_hidden_logits"].shape, torch.Size([3]))
        self.assertEqual(outputs[0]["visible_blocks_hidden_probs"].shape, torch.Size([3]))
        self.assertNotIn("hidden_blocks_visible_logits", outputs[0])
        self.assertTrue(torch.isfinite(outputs[0]["visible_blocks_hidden_logits"]).all())

    def test_visible_blocks_hidden_aux_loss_is_config_target_and_pos_weight_gated(self):
        torch.manual_seed(0)
        item = {
            "image": torch.randn(30, 32, 40),
            "height": 32,
            "width": 40,
            "image_id": "synthetic/visible_hidden_aux_train",
            "instances": self._instances(height=32, width=40),
            "graph_gt": torch.tensor(
                [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                dtype=torch.long,
            ),
            "visible_blocks_hidden_target": torch.tensor([1.0, 1.0, 1.0]),
        }
        disabled_model = MemGraphDenseKnownNodes(self._cfg(hidden_dim=16, visible_blocks_hidden_aux=False))
        disabled_model.train()

        disabled_losses = disabled_model([dict(item)])

        self.assertEqual(set(disabled_losses.keys()), {"loss_mem_dense_graph"})

        torch.manual_seed(0)
        pos1_model = MemGraphDenseKnownNodes(
            self._cfg(hidden_dim=16, visible_blocks_hidden_aux=True, visible_blocks_hidden_aux_pos_weight=1.0)
        )
        pos1_model.train()
        pos1_losses = pos1_model([dict(item)])

        self.assertEqual(
            set(pos1_losses.keys()),
            {
                "loss_mem_dense_graph",
                "loss_mem_visible_blocks_hidden_aux",
            },
        )
        self.assertTrue(torch.isfinite(pos1_losses["loss_mem_visible_blocks_hidden_aux"]))

        torch.manual_seed(0)
        pos5_model = MemGraphDenseKnownNodes(
            self._cfg(hidden_dim=16, visible_blocks_hidden_aux=True, visible_blocks_hidden_aux_pos_weight=5.0)
        )
        pos5_model.train()
        pos5_losses = pos5_model([dict(item)])
        self.assertAlmostEqual(
            float(pos5_losses["loss_mem_visible_blocks_hidden_aux"]),
            5.0 * float(pos1_losses["loss_mem_visible_blocks_hidden_aux"]),
            places=5,
        )
        self.assertAlmostEqual(
            float(
                node_binary_bce_loss(
                    torch.tensor([0.0]),
                    torch.tensor([1.0]),
                    pos_weight=5.0,
                )
            ),
            5.0 * float(node_binary_bce_loss(torch.tensor([0.0]), torch.tensor([1.0]), pos_weight=1.0)),
            places=6,
        )

        missing_target_losses = pos5_model(
            [
                {
                    key: value
                    for key, value in item.items()
                    if key != "visible_blocks_hidden_target"
                }
            ]
        )
        self.assertEqual(set(missing_target_losses.keys()), {"loss_mem_dense_graph"})

    def test_mem_graph_dense_known_nodes_pair_geometry_enabled_returns_logits(self):
        torch.manual_seed(0)
        cfg = self._cfg(hidden_dim=16, pair_geometry=True)
        model = MemGraphDenseKnownNodes(cfg)
        model.eval()
        item = {
            "image": torch.randn(30, 32, 40),
            "height": 32,
            "width": 40,
            "image_id": "synthetic/000000002",
            "instances": self._instances(height=32, width=40),
            "graph_gt": torch.zeros(3, 3, dtype=torch.long),
        }

        with torch.no_grad():
            outputs = model([item])

        self.assertEqual(model.graph_head.extra_edge_feature_dim, len(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES))
        self.assertEqual(outputs[0]["graph_logits"].shape, torch.Size([3, 3]))
        self.assertEqual(outputs[0]["pair_geometry_feature_names"], list(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES))
        self.assertTrue(torch.isfinite(outputs[0]["graph_logits"]).all())

    def test_zero_map_node_features_makes_logits_independent_of_mem_map_values(self):
        torch.manual_seed(0)
        cfg = self._cfg(hidden_dim=16, pair_geometry=True, zero_map_node_features=True)
        model = MemGraphDenseKnownNodes(cfg)
        model.eval()
        base_item = {
            "height": 32,
            "width": 40,
            "image_id": "synthetic/zero_map",
            "instances": self._instances(height=32, width=40),
            "graph_gt": torch.zeros(3, 3, dtype=torch.long),
        }
        item_a = dict(base_item, image=torch.randn(30, 32, 40))
        item_b = dict(base_item, image=torch.randn(30, 32, 40) * 100.0 + 50.0)

        with torch.no_grad():
            logits_a = model([item_a])[0]["graph_logits"]
            logits_b = model([item_b])[0]["graph_logits"]

        self.assertTrue(model.zero_map_node_features)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=0.0, rtol=0.0))

    def test_mem_graph_config_defaults_and_smoke_yaml_parse(self):
        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg.merge_from_file("configs/mem/option2a_known_nodes_graph_smoke.yaml")

        self.assertEqual(cfg.MODEL.META_ARCHITECTURE, "MemGraphDenseKnownNodes")
        self.assertEqual(cfg.MODEL.MEM_GRAPH.IN_CHANNELS, 30)
        self.assertEqual(cfg.MODEL.MEM_GRAPH.HIDDEN_DIM, 256)
        self.assertEqual(cfg.MODEL.MEM_GRAPH.NUM_OBJECT_CLASSES, 14)
        self.assertTrue(cfg.MODEL.MEM_GRAPH.REQUIRE_KNOWN_NODES)
        self.assertEqual(cfg.MODEL.GRAPH_HEAD.NAME, "GraphTransformerDense")
        self.assertEqual(cfg.MODEL.GRAPH_HEAD.EDGE_FEATURES, "concat")
        self.assertFalse(cfg.MODEL.MEM_GRAPH.ZERO_MAP_NODE_FEATURES)
        self.assertFalse(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        self.assertFalse(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)
        self.assertAlmostEqual(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_LOSS_WEIGHT, 0.1)
        self.assertAlmostEqual(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT, 2.0)
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, 1)
        self.assertEqual(cfg.DATALOADER.NUM_WORKERS, 0)

    def test_option2b_pair_geometry_config_enables_feature_path(self):
        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg.merge_from_file("configs/mem/option2b_observed_visible_nodes_pair_geometry.yaml")

        self.assertEqual(cfg.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertFalse(cfg.MODEL.MEM_GRAPH.ZERO_MAP_NODE_FEATURES)
        self.assertTrue(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        self.assertFalse(cfg.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)
        self.assertIn("front_x_overlap_union", list(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES))

    def test_option2b_pair_geometry_visible_hidden_aux_config_only_enables_aux_path(self):
        pair_geometry_path = Path("configs/mem/option2b_observed_visible_nodes_pair_geometry.yaml")
        pair_geometry_config_text = pair_geometry_path.read_text(encoding="utf-8")
        self.assertNotIn("VISIBLE_BLOCKS_HIDDEN_AUX", pair_geometry_config_text)
        self.assertNotIn("HIDDEN_DEPENDENCY_AUX", pair_geometry_config_text)

        cfg_pair = get_cfg()
        add_dep_graph_config(cfg_pair)
        add_detr_config(cfg_pair)
        cfg_pair.merge_from_file(str(pair_geometry_path))

        cfg_aux = get_cfg()
        add_dep_graph_config(cfg_aux)
        add_detr_config(cfg_aux)
        cfg_aux.merge_from_file("configs/mem/option2b_observed_visible_nodes_pair_geometry_visible_hidden_aux.yaml")

        self.assertEqual(cfg_aux.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg_aux.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertTrue(cfg_aux.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        self.assertTrue(cfg_aux.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)
        self.assertAlmostEqual(cfg_aux.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_LOSS_WEIGHT, 0.1)
        self.assertAlmostEqual(cfg_aux.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT, 2.0)
        self.assertFalse(cfg_pair.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)
        self.assertEqual(
            list(cfg_aux.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES),
            list(cfg_pair.MODEL.MEM_GRAPH.PAIR_GEOMETRY_FEATURES),
        )

    def test_option2b_pair_geometry_zero_map_config_keeps_geometry_but_zeros_map_node_features(self):
        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg.merge_from_file("configs/mem/option2b_observed_visible_nodes_pair_geometry_zero_map.yaml")

        self.assertEqual(cfg.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertTrue(cfg.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        self.assertTrue(cfg.MODEL.MEM_GRAPH.ZERO_MAP_NODE_FEATURES)

    def test_meta_architecture_is_registered(self):
        self.assertIs(META_ARCH_REGISTRY.get("MemGraphDenseKnownNodes"), MemGraphDenseKnownNodes)


if __name__ == "__main__":
    unittest.main()
