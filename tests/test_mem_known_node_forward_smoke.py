import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.smoke_mem_graph_dense_known_nodes_forward import (
    MAX_RECORDS,
    build_mem_graph_dense_known_nodes_forward_smoke_summary,
)


class MemKnownNodeForwardSmokeTest(unittest.TestCase):
    def _write_hms_npz(self, pre_action_dir: Path) -> None:
        pre_action_dir.mkdir(parents=True, exist_ok=True)
        hms = np.zeros((10, 8, 12, 2), dtype=np.float32)
        semantic_hms = np.zeros((10, 8, 12), dtype=np.float32)
        for view_index in range(10):
            hms[view_index, :, :, 0] = float(view_index) / 10.0
            hms[view_index, :, :, 1] = float(view_index) / 20.0
            semantic_hms[view_index, :, :] = float(view_index % 14)
        np.savez(
            pre_action_dir / "hms.npz",
            hms=hms,
            semantic_hms=semantic_hms,
            instance_maps=np.zeros((10, 8, 12), dtype=np.int32),
            depths=np.zeros((10, 16, 16), dtype=np.float32),
        )

    def _write_records_json(self, tmpdir: Path) -> Path:
        pre_action_dir = tmpdir / "synthetic_group" / "000000000" / "pre_action"
        self._write_hms_npz(pre_action_dir)
        record = {
            "sample_id": "synthetic_group/000000000",
            "sample_dir": str(pre_action_dir),
            "selected_view_indices": list(range(10)),
            "height": 8,
            "width": 12,
            "bbox_xyxy_abs": [[0, 0, 5, 5], [4, 2, 12, 8]],
            "bbox_categories": [0, 13],
            "node_order_instance_ids": [101, 102],
            "graph_gt": [[0, 1], [0, 0]],
            "dense_gt": [[0, 1], [0, 0]],
        }
        records_json = tmpdir / "records.json"
        records_json.write_text(json.dumps([record]), encoding="utf-8")
        return records_json

    def test_forward_smoke_runs_model_eval_forward_only_and_summarizes_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json = self._write_records_json(tmpdir)

            summary = build_mem_graph_dense_known_nodes_forward_smoke_summary(
                records_json=records_json,
                data_root=".",
                max_records=1,
                device="cpu",
                expected_height=8,
                expected_width=12,
                cfg_overrides=[
                    "MODEL.MEM_GRAPH.HIDDEN_DIM",
                    "16",
                    "MODEL.GRAPH_HEAD.HIDDEN_DIM",
                    "16",
                    "MODEL.GRAPH_HEAD.NUM_LAYERS",
                    "1",
                    "MODEL.GRAPH_HEAD.NUM_HEADS",
                    "2",
                ],
            )

        self.assertEqual(summary["schema"], "mem_d3g_known_node_forward_smoke_cli_summary_v0")
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["model_mode"], "eval")
        self.assertEqual(summary["num_records_loaded"], 1)
        self.assertEqual(summary["num_records_mapped"], 1)
        self.assertEqual(summary["num_records_forwarded"], 1)
        self.assertEqual(summary["validation_errors"], [])
        self.assertEqual(summary["total_nodes"], 2)
        self.assertEqual(summary["total_edges"], 1)
        record = summary["record_summaries"][0]
        self.assertEqual(record["image_id"], "synthetic_group/000000000")
        self.assertEqual(record["image_shape_chw"], [30, 8, 12])
        self.assertEqual(record["num_nodes"], 2)
        self.assertEqual(record["graph_gt_shape"], [2, 2])
        self.assertEqual(record["graph_logits_shape"], [2, 2])
        self.assertEqual(record["graph_probs_shape"], [2, 2])
        self.assertTrue(record["graph_logits_finite"])
        self.assertTrue(record["graph_probs_finite"])
        self.assertEqual(record["node_order_instance_ids"], [101, 102])
        self.assertEqual(record["selected_view_indices"], list(range(10)))
        self.assertTrue(summary["safety"]["runs_model_forward"])
        self.assertTrue(summary["safety"]["forward_only"])
        self.assertFalse(summary["safety"]["runs_backward"])
        self.assertFalse(summary["safety"]["runs_optimizer_step"])
        self.assertFalse(summary["safety"]["runs_training_loop"])
        self.assertFalse(summary["safety"]["writes_checkpoints_or_training_outputs"])
        self.assertFalse(summary["safety"]["loads_or_writes_model_weights"])

    def test_forward_smoke_refuses_more_than_five_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            records_json = self._write_records_json(Path(tmp))
            with self.assertRaisesRegex(ValueError, "max_records"):
                build_mem_graph_dense_known_nodes_forward_smoke_summary(
                    records_json=records_json,
                    data_root=".",
                    max_records=MAX_RECORDS + 1,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
