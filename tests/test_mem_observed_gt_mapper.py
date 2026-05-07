import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data import get_mapper
from data.mem_observed_gt_dataset import load_mem_observed_gt_records
from data.mem_observed_gt_mapper import MemObservedGtMapper, materialize_mem_observed_tensor


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


if __name__ == "__main__":
    unittest.main()
