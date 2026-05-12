import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data import get_mapper
from detectron2.config import get_cfg

from data.mem_observed_gt_dataset import load_mem_observed_gt_records
from data.mem_observed_gt_mapper import MemObservedGtMapper, materialize_mem_observed_tensor
from utils.configs import add_dep_graph_config, add_detr_config


class MemObservedGtMapperTest(unittest.TestCase):
    def _write_hms_npz(self, pre_action_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        pre_action_dir.mkdir(parents=True, exist_ok=True)
        hms = np.zeros((3, 2, 4, 2), dtype=np.float32)
        semantic_hms = np.zeros((3, 2, 4), dtype=np.float32)
        for view_index in range(hms.shape[0]):
            hms[view_index, :, :, 0] = view_index * 100 + 1
            hms[view_index, :, :, 1] = view_index * 100 + 2
            semantic_hms[view_index, :, :] = view_index
        np.savez(
            pre_action_dir / "hms.npz",
            hms=hms,
            semantic_hms=semantic_hms,
            instance_maps=np.zeros((3, 2, 4), dtype=np.int32),
            depths=np.zeros((3, 8, 8), dtype=np.float32),
        )
        return hms, semantic_hms

    def _flat_record(self, pre_action_dir: Path) -> dict:
        return {
            "sample_id": "synthetic_group/000000000",
            "sample_dir": str(pre_action_dir),
            "selected_view_indices": [0, 2],
            "height": 2,
            "width": 4,
            "bbox_xyxy_abs": [[0, 0, 2, 1], [1, 0, 4, 2]],
            "bbox_categories": [3, 4],
            "node_order_instance_ids": [10, 11],
            "graph_gt": [[0, 1], [0, 0]],
        }

    def _write_option2b_hms_npz(self, pre_action_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pre_action_dir.mkdir(parents=True, exist_ok=True)
        hms = np.zeros((3, 4, 5, 2), dtype=np.float32)
        semantic_hms = np.full((3, 4, 5), 14, dtype=np.float32)
        instance_maps = np.full((3, 4, 5), -1, dtype=np.float32)

        # Instance 11 is visible in selected view 0.
        instance_maps[0, 0:2, 0:2] = 11
        semantic_hms[0, 0:2, 0:2] = 4

        # Instance 22 is visible in selected view 2.
        instance_maps[2, 2:4, 2:5] = 22
        semantic_hms[2, 2:4, 2:5] = 7

        # Instance 99 is visible but has no GT graph alignment; it must not enter
        # the induced target graph for Option 2B.
        instance_maps[0, 0:1, 4:5] = 99
        semantic_hms[0, 0:1, 4:5] = 5

        np.savez(pre_action_dir / "hms.npz", hms=hms, semantic_hms=semantic_hms, instance_maps=instance_maps)
        return hms, semantic_hms, instance_maps

    def _option2b_source_record(self, pre_action_dir: Path) -> dict:
        return {
            "sample_id": "synthetic_group/option2b_visible_nodes",
            "sample_dir": str(pre_action_dir),
            "selected_view_indices": [0, 2],
            "height": 4,
            "width": 5,
            "bbox_xyxy_abs": [[0, 0, 3, 3], [1, 1, 5, 4], [0, 2, 2, 4]],
            "bbox_categories": [4, 7, 8],
            "node_order_instance_ids": [11, 22, 33],
            "graph_gt": [
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 0],
            ],
        }

    def test_materialize_mem_observed_tensor_uses_view_major_channel_order(self):
        hms = np.zeros((3, 2, 2, 2), dtype=np.float32)
        semantic_hms = np.zeros((3, 2, 2), dtype=np.float32)
        for view_index in range(3):
            hms[view_index, :, :, 0] = view_index * 10 + 1
            hms[view_index, :, :, 1] = view_index * 10 + 2
            semantic_hms[view_index, :, :] = view_index * 10 + 3

        tensor, layout = materialize_mem_observed_tensor(hms, semantic_hms, [0, 2])

        self.assertEqual(tensor.shape, (6, 2, 2))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.array_equal(tensor[0], hms[0, :, :, 0]))
        self.assertTrue(np.array_equal(tensor[1], hms[0, :, :, 1]))
        self.assertTrue(np.array_equal(tensor[2], semantic_hms[0]))
        self.assertTrue(np.array_equal(tensor[3], hms[2, :, :, 0]))
        self.assertTrue(np.array_equal(tensor[4], hms[2, :, :, 1]))
        self.assertTrue(np.array_equal(tensor[5], semantic_hms[2]))
        self.assertEqual(
            layout,
            [
                {"channel_index": 0, "view_index": 0, "source_field": "hms", "source_channel": 0},
                {"channel_index": 1, "view_index": 0, "source_field": "hms", "source_channel": 1},
                {"channel_index": 2, "view_index": 0, "source_field": "semantic_hms", "source_channel": None},
                {"channel_index": 3, "view_index": 2, "source_field": "hms", "source_channel": 0},
                {"channel_index": 4, "view_index": 2, "source_field": "hms", "source_channel": 1},
                {"channel_index": 5, "view_index": 2, "source_field": "semantic_hms", "source_channel": None},
            ],
        )

    def test_mapper_outputs_d3g_item_with_direct_dense_mem_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pre_action_dir = Path(tmpdir) / "synthetic_group" / "000000000" / "pre_action"
            hms, semantic_hms = self._write_hms_npz(pre_action_dir)
            record = self._flat_record(pre_action_dir)

            mapper = MemObservedGtMapper(
                data_root=".",
                is_train=False,
                graph_gt_type="dense",
                expected_height=2,
                expected_width=4,
            )
            out = mapper(record)

            self.assertEqual(out["image"].shape, torch.Size([6, 2, 4]))
            self.assertEqual(out["image"].dtype, torch.float32)
            self.assertTrue(torch.equal(out["image"][0], torch.from_numpy(hms[0, :, :, 0])))
            self.assertTrue(torch.equal(out["image"][1], torch.from_numpy(hms[0, :, :, 1])))
            self.assertTrue(torch.equal(out["image"][2], torch.from_numpy(semantic_hms[0])))
            self.assertTrue(torch.equal(out["image"][3], torch.from_numpy(hms[2, :, :, 0])))
            self.assertTrue(torch.equal(out["image"][4], torch.from_numpy(hms[2, :, :, 1])))
            self.assertTrue(torch.equal(out["image"][5], torch.from_numpy(semantic_hms[2])))

            self.assertEqual(out["height"], 2)
            self.assertEqual(out["width"], 4)
            self.assertEqual(out["image_id"], "synthetic_group/000000000")
            self.assertEqual(out["instances"].image_size, (2, 4))
            self.assertTrue(torch.equal(out["instances"].gt_classes, torch.tensor([3, 4])))
            self.assertTrue(
                torch.equal(
                    out["instances"].gt_boxes.tensor,
                    torch.tensor([[0.0, 0.0, 2.0, 1.0], [1.0, 0.0, 4.0, 2.0]]),
                )
            )
            expected_graph = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
            self.assertTrue(torch.equal(out["graph_gt"], expected_graph))
            self.assertTrue(torch.equal(out["dense_gt"], expected_graph))
            self.assertEqual(out["mem_metadata"]["selected_view_indices"], [0, 2])
            self.assertEqual(out["mem_metadata"]["node_order_instance_ids"], [10, 11])

    def test_loader_normalizes_option2a_preview_records_and_dispatches_mapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pre_action_dir = Path(tmpdir) / "synthetic_group" / "000000000" / "pre_action"
            self._write_hms_npz(pre_action_dir)
            preview_record = {
                "schema": "mem_observed_input_gt_graph_preview_v0",
                "sample_id": "synthetic_group/000000000",
                "sample_dir": str(pre_action_dir),
                "image_height": 2,
                "image_width": 4,
                "observed_input": {
                    "source_file": "pre_action/hms.npz",
                    "selected_view_indices": [0, 2],
                    "fields": {
                        "hms": {"shape": [2, 2, 4, 2]},
                        "semantic_hms": {"shape": [2, 2, 4]},
                    },
                },
                "target_graph": {
                    "node_order_instance_ids": [10, 11],
                    "node_classes": [3, 4],
                    "bbox_xyxy_abs": [[0, 0, 2, 1], [1, 0, 4, 2]],
                    "graph_gt": [[0, 1], [0, 0]],
                    "dense_gt": [[0, 1], [0, 0]],
                    "num_nodes": 2,
                    "num_edges": 1,
                },
            }
            records_json = Path(tmpdir) / "records.json"
            records_json.write_text(json.dumps([preview_record]), encoding="utf-8")

            records = load_mem_observed_gt_records(records_json)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["sample_id"], "synthetic_group/000000000")
            self.assertEqual(records[0]["selected_view_indices"], [0, 2])
            self.assertEqual(records[0]["bbox_categories"], [3, 4])
            self.assertEqual(records[0]["graph_gt"], [[0, 1], [0, 0]])
            self.assertIs(get_mapper("mem_option2a_forward_smoke"), MemObservedGtMapper)

    def test_mapper_rejects_non_binary_or_self_loop_graph_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pre_action_dir = Path(tmpdir) / "synthetic_group" / "000000000" / "pre_action"
            self._write_hms_npz(pre_action_dir)
            record = self._flat_record(pre_action_dir)
            record["graph_gt"] = [[0, 2], [0, 1]]

            mapper = MemObservedGtMapper(
                data_root=".",
                is_train=False,
                graph_gt_type="dense",
                expected_height=2,
                expected_width=4,
            )

            with self.assertRaisesRegex(ValueError, "graph_gt must be binary"):
                mapper(record)

    def test_option2b_mapper_derives_visible_observed_nodes_and_induced_gt_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pre_action_dir = Path(tmpdir) / "synthetic_group" / "option2b_visible_nodes" / "pre_action"
            self._write_option2b_hms_npz(pre_action_dir)
            record = self._option2b_source_record(pre_action_dir)

            mapper = MemObservedGtMapper(
                data_root=".",
                is_train=False,
                graph_gt_type="dense",
                expected_height=4,
                expected_width=5,
                semantic_class_max=14,
                mem_node_source="observed_instance_maps",
                mem_graph_target_scope="observed_induced",
            )
            out = mapper(record)

            self.assertEqual(out["image"].shape, torch.Size([6, 4, 5]))
            self.assertEqual(out["instances"].image_size, (4, 5))
            self.assertTrue(torch.equal(out["instances"].gt_classes, torch.tensor([4, 7])))
            self.assertTrue(
                torch.equal(
                    out["instances"].gt_boxes.tensor,
                    torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 5.0, 4.0]]),
                )
            )
            expected_graph = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
            self.assertTrue(torch.equal(out["graph_gt"], expected_graph))
            self.assertTrue(torch.equal(out["dense_gt"], expected_graph))

            metadata = out["mem_metadata"]
            self.assertEqual(metadata["mem_node_source"], "observed_instance_maps")
            self.assertEqual(metadata["mem_graph_target_scope"], "observed_induced")
            self.assertFalse(metadata["is_oracle_node_conditioned"])
            self.assertEqual(metadata["node_order_instance_ids"], [11, 22])
            self.assertEqual(metadata["observed_visible_instance_ids"], [11, 22, 99])
            self.assertEqual(metadata["gt_aligned_instance_ids"], [11, 22])
            self.assertEqual(metadata["unmatched_observed_instance_ids"], [99])
            self.assertEqual(metadata["hidden_gt_instance_ids"], [33])
            self.assertEqual(metadata["observed_induced_source_gt_indices"], [0, 1])
            self.assertNotIn("hidden_blocks_visible", out)
            self.assertTrue(torch.equal(out["visible_blocks_hidden_target"], torch.tensor([1.0, 1.0])))
            self.assertEqual(metadata["visible_blocks_hidden_target"], [1, 1])
            self.assertEqual(metadata["num_visible_blocks_hidden_positive"], 2)

    def test_option2b_config_file_sets_node_source_and_target_scope(self):
        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs" / "mem" / "option2b_observed_visible_nodes.yaml"))

        self.assertEqual(cfg.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertEqual(cfg.MODEL.META_ARCHITECTURE, "MemGraphDenseKnownNodes")

    def test_option2a_config_file_keeps_gt_all_known_node_contract(self):
        cfg = get_cfg()
        add_dep_graph_config(cfg)
        add_detr_config(cfg)
        cfg.merge_from_file(str(Path(__file__).resolve().parents[1] / "configs" / "mem" / "option2a_gt_known_nodes.yaml"))

        self.assertEqual(cfg.INPUT.MEM_NODE_SOURCE, "gt")
        self.assertEqual(cfg.INPUT.MEM_GRAPH_TARGET_SCOPE, "gt_all")
        self.assertEqual(cfg.MODEL.META_ARCHITECTURE, "MemGraphDenseKnownNodes")


if __name__ == "__main__":
    unittest.main()
