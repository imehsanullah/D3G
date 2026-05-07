import unittest

import torch
from detectron2.config import get_cfg
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances

from models.mem_graph_dense import (
    MemGraphDenseKnownNodes,
    MemKnownNodeTokenExtractor,
    MemMapEncoder,
    masked_dense_graph_bce_loss,
)
from utils.configs import add_dep_graph_config, add_detr_config


class MemGraphDenseTest(unittest.TestCase):
    def _cfg(self, *, height=32, width=40, hidden_dim=16):
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
        self.assertTrue(torch.isfinite(outputs[0]["graph_logits"]).all())

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
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, 1)
        self.assertEqual(cfg.DATALOADER.NUM_WORKERS, 0)

    def test_meta_architecture_is_registered(self):
        self.assertIs(META_ARCH_REGISTRY.get("MemGraphDenseKnownNodes"), MemGraphDenseKnownNodes)


if __name__ == "__main__":
    unittest.main()
