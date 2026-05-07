import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.debug_train_mem_graph_dense_known_nodes_tiny import (
    MAX_RECORDS,
    build_mem_graph_dense_known_nodes_tiny_train_summary,
    compute_average_precision,
    compute_binary_metrics_at_threshold,
    compute_score_distribution_diagnostics,
)


class MemKnownNodeTinyTrainDebugTest(unittest.TestCase):
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
        records = []
        for index in range(2):
            sample_id = f"synthetic_group/{index:09d}"
            pre_action_dir = tmpdir / "synthetic_group" / f"{index:09d}" / "pre_action"
            self._write_hms_npz(pre_action_dir)
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": str(pre_action_dir),
                    "selected_view_indices": list(range(10)),
                    "height": 8,
                    "width": 12,
                    "bbox_xyxy_abs": [[0, 0, 5, 5], [4, 2, 12, 8]],
                    "bbox_categories": [0, 13],
                    "node_order_instance_ids": [101 + index * 10, 102 + index * 10],
                    "graph_gt": [[0, 1], [0, 0]],
                    "dense_gt": [[0, 1], [0, 0]],
                }
            )
        records_json = tmpdir / "records.json"
        records_json.write_text(json.dumps(records), encoding="utf-8")
        return records_json

    def _write_imbalanced_records_json(self, tmpdir: Path) -> Path:
        records = []
        for index in range(2):
            sample_id = f"imbalanced_group/{index:09d}"
            pre_action_dir = tmpdir / "imbalanced_group" / f"{index:09d}" / "pre_action"
            self._write_hms_npz(pre_action_dir)
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": str(pre_action_dir),
                    "selected_view_indices": list(range(10)),
                    "height": 8,
                    "width": 12,
                    "bbox_xyxy_abs": [[0, 0, 4, 4], [4, 0, 8, 4], [2, 4, 10, 8]],
                    "bbox_categories": [0, 1, 13],
                    "node_order_instance_ids": [201 + index * 10, 202 + index * 10, 203 + index * 10],
                    "graph_gt": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                    "dense_gt": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                }
            )
        records_json = tmpdir / "imbalanced_records.json"
        records_json.write_text(json.dumps(records), encoding="utf-8")
        return records_json

    def _tiny_overrides(self):
        return [
            "MODEL.MEM_GRAPH.HIDDEN_DIM",
            "16",
            "MODEL.GRAPH_HEAD.HIDDEN_DIM",
            "16",
            "MODEL.GRAPH_HEAD.NUM_LAYERS",
            "1",
            "MODEL.GRAPH_HEAD.NUM_HEADS",
            "2",
        ]

    def test_synthetic_one_step_train_debug_runs_backward_optimizer_and_writes_only_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json = self._write_records_json(tmpdir)
            output_dir = tmpdir / "tiny_train_debug_summary"

            summary = build_mem_graph_dense_known_nodes_tiny_train_summary(
                records_json=records_json,
                data_root=".",
                train_sample_ids=["synthetic_group/000000000"],
                val_sample_ids=["synthetic_group/000000001"],
                max_records=2,
                device="cpu",
                max_iter=1,
                lr=1e-4,
                seed=0,
                output_dir=output_dir,
                no_checkpoint=True,
                acknowledge_backward=True,
                acknowledge_optimizer_step=True,
                expected_height=8,
                expected_width=12,
                cfg_overrides=self._tiny_overrides(),
                include_rich_diagnostics=True,
                diagnostic_thresholds=[0.1, 0.3, 0.5],
                histogram_bin_edges=[0.0, 0.5, 1.0],
            )

            written_files = sorted(path.name for path in output_dir.iterdir())
            checkpoint_like = [name for name in written_files if name.endswith((".pth", ".pt", ".ckpt"))]

        self.assertEqual(summary["schema"], "mem_d3g_known_node_tiny_train_debug_summary_v0")
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["max_iter"], 1)
        self.assertEqual(summary["loss_mode"], "bce")
        self.assertEqual(summary["graph_loss_pos_weight"], 0.0)
        self.assertEqual(summary["graph_loss_pos_weight_source"], "unweighted_bce")
        self.assertEqual(summary["safety"]["max_records_cap"], 100)
        self.assertEqual(summary["num_train_records"], 1)
        self.assertEqual(summary["num_val_records"], 1)
        self.assertEqual(summary["train_sample_ids"], ["synthetic_group/000000000"])
        self.assertEqual(summary["val_sample_ids"], ["synthetic_group/000000001"])
        self.assertEqual(len(summary["train_iterations"]), 1)
        iteration = summary["train_iterations"][0]
        self.assertEqual(iteration["iteration"], 0)
        self.assertTrue(iteration["loss_mem_dense_graph_finite"])
        self.assertGreaterEqual(iteration["loss_mem_dense_graph"], 0.0)
        self.assertTrue(iteration["grad_norm_total_finite"])
        self.assertTrue(iteration["parameter_norm_total_finite"])
        self.assertEqual(summary["train_target_summaries"][0]["num_positive_directed_edges"], 1)
        self.assertEqual(summary["train_target_summaries"][0]["num_non_diagonal_pairs"], 2)
        self.assertEqual(len(summary["validation_summaries"]), 1)
        val_summary = summary["validation_summaries"][0]
        self.assertIn("average_precision", val_summary["metrics_at_threshold_0_5"])
        self.assertEqual([item["threshold"] for item in val_summary["threshold_sweep"]], [0.1, 0.3, 0.5])
        self.assertIn("best_f1_threshold", val_summary)
        self.assertIn("probability_histogram", val_summary)
        self.assertEqual(val_summary["probability_histogram"]["bin_edges"], [0.0, 0.5, 1.0])
        self.assertIn("validation_aggregate_diagnostics", summary)
        self.assertEqual(summary["validation_aggregate_diagnostics"]["num_records"], 1)
        self.assertEqual(summary["validation_aggregate_diagnostics"]["num_pairs"], 2)
        self.assertEqual(written_files, ["summary.json"])
        self.assertEqual(checkpoint_like, [])
        self.assertTrue(summary["safety"]["runs_model_train_mode"])
        self.assertTrue(summary["safety"]["runs_backward"])
        self.assertTrue(summary["safety"]["runs_optimizer_step"])
        self.assertTrue(summary["safety"]["runs_training_loop"])
        self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])
        self.assertFalse(summary["safety"]["writes_hdf5_or_full_dataset"])
        self.assertFalse(summary["safety"]["loads_model_weights"])

    def test_train_pos_weighted_bce_uses_train_split_negative_positive_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json = self._write_imbalanced_records_json(tmpdir)

            summary = build_mem_graph_dense_known_nodes_tiny_train_summary(
                records_json=records_json,
                data_root=".",
                train_sample_ids=["imbalanced_group/000000000"],
                val_sample_ids=["imbalanced_group/000000001"],
                max_records=2,
                device="cpu",
                max_iter=1,
                lr=1e-4,
                seed=0,
                output_dir=tmpdir / "pos_weighted_summary",
                no_checkpoint=True,
                acknowledge_backward=True,
                acknowledge_optimizer_step=True,
                expected_height=8,
                expected_width=12,
                cfg_overrides=self._tiny_overrides(),
                loss_mode="train_pos_weighted_bce",
            )

        self.assertEqual(summary["loss_mode"], "train_pos_weighted_bce")
        self.assertAlmostEqual(summary["graph_loss_pos_weight"], 5.0)
        self.assertEqual(summary["graph_loss_pos_weight_source"], "train_split_negative_positive_ratio")
        self.assertEqual(summary["train_aggregate_target_summary"]["num_positive_directed_edges"], 1)
        self.assertEqual(summary["train_aggregate_target_summary"]["num_negative_directed_edges"], 5)
        self.assertAlmostEqual(summary["train_aggregate_target_summary"]["negative_positive_ratio"], 5.0)
        self.assertTrue(summary["train_iterations"][0]["loss_mem_dense_graph_finite"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is not available for GPU debug wrapper test")
    def test_cuda_one_step_train_debug_runs_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json = self._write_records_json(tmpdir)

            summary = build_mem_graph_dense_known_nodes_tiny_train_summary(
                records_json=records_json,
                data_root=".",
                train_sample_ids=["synthetic_group/000000000"],
                val_sample_ids=["synthetic_group/000000001"],
                max_records=2,
                device="cuda:0",
                max_iter=1,
                lr=1e-4,
                seed=0,
                output_dir=tmpdir / "cuda_summary",
                no_checkpoint=True,
                acknowledge_backward=True,
                acknowledge_optimizer_step=True,
                expected_height=8,
                expected_width=12,
                cfg_overrides=self._tiny_overrides(),
            )

        self.assertEqual(summary["device"], "cuda:0")
        self.assertTrue(summary["torch_cuda_available"])
        self.assertGreaterEqual(summary["torch_cuda_device_count"], 1)
        self.assertIn("cuda_device_name", summary)
        self.assertTrue(summary["train_iterations"][0]["loss_mem_dense_graph_finite"])
        self.assertTrue(summary["train_iterations"][0]["grad_norm_total_finite"])
        self.assertEqual(summary["safety"]["device"], "cuda:0")
        self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])

    def test_train_debug_refuses_unsafe_or_ambiguous_invocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json = self._write_records_json(tmpdir)
            existing_output_dir = tmpdir / "existing"
            existing_output_dir.mkdir()

            base_kwargs = dict(
                records_json=records_json,
                data_root=".",
                train_sample_ids=["synthetic_group/000000000"],
                max_records=1,
                device="cpu",
                max_iter=1,
                lr=1e-4,
                seed=0,
                output_dir=tmpdir / "new_output",
                no_checkpoint=True,
                acknowledge_backward=True,
                acknowledge_optimizer_step=True,
                expected_height=8,
                expected_width=12,
                cfg_overrides=self._tiny_overrides(),
            )

            with self.assertRaisesRegex(ValueError, "acknowledge"):
                build_mem_graph_dense_known_nodes_tiny_train_summary(
                    **{**base_kwargs, "acknowledge_backward": False, "output_dir": tmpdir / "ack_fail"}
                )
            with self.assertRaisesRegex(ValueError, "max_records"):
                build_mem_graph_dense_known_nodes_tiny_train_summary(
                    **{**base_kwargs, "max_records": MAX_RECORDS + 1, "output_dir": tmpdir / "max_fail"}
                )
            with self.assertRaisesRegex(ValueError, "CUDA device index"):
                build_mem_graph_dense_known_nodes_tiny_train_summary(
                    **{**base_kwargs, "device": "cuda:99", "output_dir": tmpdir / "cuda_fail"}
                )
            with self.assertRaisesRegex(FileExistsError, "refusing to write into existing output_dir"):
                build_mem_graph_dense_known_nodes_tiny_train_summary(
                    **{**base_kwargs, "output_dir": existing_output_dir}
                )
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                build_mem_graph_dense_known_nodes_tiny_train_summary(
                    **{**base_kwargs, "no_checkpoint": False, "output_dir": tmpdir / "ckpt_fail"}
                )

    def test_binary_metric_helpers_ignore_diagonal_and_compute_ap(self):
        scores = [0.1, 0.9, 0.8]
        labels = [0, 1, 1]
        self.assertAlmostEqual(compute_average_precision(scores, labels), 1.0)

        metrics = compute_binary_metrics_at_threshold(
            logits=[[-10.0, 3.0], [-4.0, -10.0]],
            target=[[0, 1], [0, 0]],
            threshold=0.5,
        )
        self.assertEqual(metrics["num_pairs"], 2)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
    def test_score_distribution_diagnostics_reports_sweep_histogram_and_best_threshold(self):
        diagnostics = compute_score_distribution_diagnostics(
            scores=[0.92, 0.60, 0.40, 0.20],
            labels=[1, 0, 1, 0],
            thresholds=[0.5, 0.3, 0.1],
            bin_edges=[0.0, 0.5, 1.0],
        )

        self.assertEqual(diagnostics["num_pairs"], 4)
        self.assertEqual(diagnostics["num_positive"], 2)
        self.assertAlmostEqual(diagnostics["positive_edge_ratio"], 0.5)
        self.assertAlmostEqual(diagnostics["average_precision"], 5.0 / 6.0)
        self.assertEqual([item["threshold"] for item in diagnostics["threshold_sweep"]], [0.5, 0.3, 0.1])
        best = diagnostics["best_f1_threshold"]
        self.assertEqual(best["threshold"], 0.3)
        self.assertEqual(best["true_positive"], 2)
        self.assertEqual(best["false_positive"], 1)
        self.assertEqual(best["false_negative"], 0)
        self.assertGreater(best["f1"], diagnostics["threshold_sweep"][0]["f1"])
        histogram = diagnostics["probability_histogram"]
        self.assertEqual(histogram["bin_edges"], [0.0, 0.5, 1.0])
        self.assertEqual(histogram["positive_counts"], [1, 1])
        self.assertEqual(histogram["negative_counts"], [1, 1])


if __name__ == "__main__":
    unittest.main()
