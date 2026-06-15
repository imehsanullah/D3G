import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from detectron2.data import DatasetCatalog
from detectron2.structures import Boxes, Instances

from tools.train_mem_graph import (
    build_mem_prediction_dump,
    compute_average_precision,
    compute_mem_graph_score_metrics,
    load_mem_graph_records_for_splits,
    load_mem_graph_split_manifest,
    register_mem_graph_datasets,
    run_mem_graph_training,
    setup_mem_graph_cfg,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTION2A_CONFIG = REPO_ROOT / "configs" / "mem" / "option2a_gt_known_nodes.yaml"
OPTION2B_CONFIG = REPO_ROOT / "configs" / "mem" / "option2b_observed_visible_nodes.yaml"
OPTION2B_PAIR_GEOMETRY_CONFIG = REPO_ROOT / "configs" / "mem" / "option2b_observed_visible_nodes_pair_geometry.yaml"
OPTION2B_PAIR_GEOMETRY_VISIBLE_HIDDEN_AUX_CONFIG = (
    REPO_ROOT / "configs" / "mem" / "option2b_observed_visible_nodes_pair_geometry_visible_hidden_aux.yaml"
)
OPTION2B_PAIR_GEOMETRY_OBB_CONFIG = (
    REPO_ROOT / "configs" / "mem" / "option2b_observed_visible_nodes_pair_geometry_obb.yaml"
)
OPTION2B_PAIR_GEOMETRY_CNABU_MEAN_CONFIG = (
    REPO_ROOT / "configs" / "mem" / "option2b_observed_visible_nodes_pair_geometry_cnabu_mean.yaml"
)
OPTION2B_PAIR_GEOMETRY_RAW_PLUS_CNABU_MEAN_CONFIG = (
    REPO_ROOT / "configs" / "mem" / "option2b_observed_visible_nodes_pair_geometry_raw_plus_cnabu_mean.yaml"
)
OPTION2C_CNABU_COMPONENTS_PAIR_GEOMETRY_CONFIG = (
    REPO_ROOT / "configs" / "mem" / "option2c_cnabu_components_pair_geometry.yaml"
)


class MemGraphTrainerTest(unittest.TestCase):
    def _write_hms_npz(self, pre_action_dir: Path, *, option2b_instances: bool = False) -> None:
        pre_action_dir.mkdir(parents=True, exist_ok=True)
        hms = np.zeros((10, 8, 12, 2), dtype=np.float32)
        semantic_hms = np.zeros((10, 8, 12), dtype=np.float32)
        instance_maps = np.zeros((10, 8, 12), dtype=np.int32)
        for view_index in range(10):
            hms[view_index, :, :, 0] = float(view_index) / 10.0
            hms[view_index, :, :, 1] = float(view_index) / 20.0
            semantic_hms[view_index, :, :] = 0.0
            if option2b_instances:
                instance_maps[view_index, 0:4, 0:5] = 1
                instance_maps[view_index, 4:8, 5:12] = 2
                instance_maps[view_index, 0:2, 9:12] = 3
                semantic_hms[view_index, 0:4, 0:5] = 0.0
                semantic_hms[view_index, 4:8, 5:12] = 1.0
                semantic_hms[view_index, 0:2, 9:12] = 2.0
        np.savez(
            pre_action_dir / "hms.npz",
            hms=hms,
            semantic_hms=semantic_hms,
            instance_maps=instance_maps,
            depths=np.zeros((10, 16, 16), dtype=np.float32),
        )

    def _write_option2c_cnabu_files(self, cnabu_root: Path, sample_rel: Path) -> None:
        cnabu_path = cnabu_root / "samples" / sample_rel / "cnabu_hms.npz"
        cnabu_path.parent.mkdir(parents=True, exist_ok=True)
        occupancy_mean = np.full((2, 8, 12), 0.8, dtype=np.float32)
        semantic_mean = np.full((15, 8, 12), 1.0 / 15.0, dtype=np.float32)
        semantic_mean[0, 0:4, 0:5] = 0.9
        semantic_mean[1, 4:8, 5:12] = 0.9
        semantic_mean[2, 0:2, 9:12] = 0.9
        semantic_mean = semantic_mean / np.maximum(semantic_mean.sum(axis=0, keepdims=True), 1e-8)
        np.savez(
            cnabu_path,
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            selected_view_indices=np.asarray(list(range(10)), dtype=np.int16),
            crop_rows=np.asarray([0, 8], dtype=np.int16),
            metadata_json=np.asarray(json.dumps({"source": "synthetic_option2c_cnabu"})),
        )

        masks = np.zeros((3, 8, 12), dtype=np.uint8)
        masks[0, 0:4, 0:5] = 1
        masks[1, 4:8, 5:12] = 1
        masks[2, 0:2, 9:12] = 1
        np.savez(
            cnabu_path.parent / "node_masks.npz",
            node_masks=masks,
            node_semantic_labels=np.asarray([0, 1, 2], dtype=np.int16),
            node_scores=np.asarray([0.95, 0.90, 0.70], dtype=np.float32),
            bbox_xyxy_abs=np.asarray([[0, 0, 5, 4], [5, 4, 12, 8], [9, 0, 12, 2]], dtype=np.int16),
            component_ids=np.asarray([101, 202, 303], dtype=np.int32),
            crop_rows=np.asarray([0, 8], dtype=np.int16),
            thresholds=np.asarray([0.5, 0.0], dtype=np.float32),
            node_source=np.asarray("cnabu_3d_components"),
            metadata_json=np.asarray(json.dumps({"source": "synthetic_option2c_nodes"})),
        )

    def _write_option2c_gt_hms(self, pre_action_dir: Path) -> None:
        gt_instance_maps = np.zeros((8, 12), dtype=np.int32)
        gt_instance_maps[0:4, 0:5] = 1
        gt_instance_maps[4:8, 5:12] = 2
        np.savez(pre_action_dir / "gt_hms.npz", instance_maps=gt_instance_maps)

    def _write_records_and_split(
        self,
        tmpdir: Path,
        *,
        option2b_instances: bool = False,
        hidden_gt_node: bool = False,
        option2c_components: bool = False,
        cnabu_root=None,
    ):
        records = []
        split_ids = {"train": [], "val": [], "test": []}
        split_scene = {"train": "scene0", "val": "scene1", "test": "scene2"}
        for split_name, scene in split_scene.items():
            sample_id = f"{scene}/000000000"
            pre_action_dir = tmpdir / scene / "000000000" / "pre_action"
            self._write_hms_npz(pre_action_dir, option2b_instances=option2b_instances)
            if option2c_components:
                if cnabu_root is None:
                    raise ValueError("cnabu_root is required for option2c_components")
                sample_rel = Path(scene) / "000000000" / "pre_action"
                self._write_option2c_gt_hms(pre_action_dir)
                self._write_option2c_cnabu_files(cnabu_root, sample_rel)
            if hidden_gt_node:
                bbox_xyxy_abs = [[0, 0, 5, 4], [5, 4, 12, 8], [0, 4, 5, 8]]
                bbox_categories = [0, 1, 2]
                node_order_instance_ids = [1, 2, 4]
                graph_gt = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
            else:
                bbox_xyxy_abs = [[0, 0, 5, 4], [5, 4, 12, 8]]
                bbox_categories = [0, 1]
                node_order_instance_ids = [1, 2]
                graph_gt = [[0, 1], [0, 0]]
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": str(pre_action_dir),
                    "selected_view_indices": list(range(10)),
                    "height": 8,
                    "width": 12,
                    "bbox_xyxy_abs": bbox_xyxy_abs,
                    "bbox_categories": bbox_categories,
                    "node_order_instance_ids": node_order_instance_ids,
                    "graph_gt": graph_gt,
                    "dense_gt": graph_gt,
                    "scene": scene,
                }
            )
            split_ids[split_name].append(sample_id)
        records_json = tmpdir / "records.json"
        records_json.write_text(json.dumps(records), encoding="utf-8")
        split_json = tmpdir / "split_manifest.json"
        split_json.write_text(
            json.dumps(
                {
                    "schema": "synthetic_mem_graph_split_v0",
                    "records_json": str(records_json),
                    "data_root": str(tmpdir),
                    "train_sample_ids": split_ids["train"],
                    "val_sample_ids": split_ids["val"],
                    "test_sample_ids": split_ids["test"],
                    "split_rule": "synthetic scene-disjoint split",
                }
            ),
            encoding="utf-8",
        )
        return records_json, split_json, split_ids

    def _tiny_overrides(self):
        return [
            "INPUT.MEM_EXPECTED_HEIGHT",
            "8",
            "INPUT.MEM_EXPECTED_WIDTH",
            "12",
            "MODEL.MEM_GRAPH.HIDDEN_DIM",
            "16",
            "MODEL.GRAPH_HEAD.HIDDEN_DIM",
            "16",
            "MODEL.GRAPH_HEAD.NUM_LAYERS",
            "1",
            "MODEL.GRAPH_HEAD.NUM_HEADS",
            "2",
        ]

    def test_config_parsing_for_option2a_and_option2b(self):
        cfg_2a = setup_mem_graph_cfg(OPTION2A_CONFIG, device="cpu", cfg_overrides=self._tiny_overrides())
        cfg_2b = setup_mem_graph_cfg(OPTION2B_CONFIG, device="cpu", cfg_overrides=self._tiny_overrides())

        self.assertEqual(cfg_2a.MODEL.META_ARCHITECTURE, "MemGraphDenseKnownNodes")
        self.assertEqual(cfg_2a.MODEL.GRAPH_HEAD.NAME, "GraphTransformerDense")
        self.assertEqual(cfg_2a.INPUT.MEM_NODE_SOURCE, "gt")
        self.assertEqual(cfg_2a.INPUT.MEM_GRAPH_TARGET_SCOPE, "gt_all")
        self.assertEqual(cfg_2b.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg_2b.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertFalse(cfg_2b.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)

        cfg_aux = setup_mem_graph_cfg(
            OPTION2B_PAIR_GEOMETRY_VISIBLE_HIDDEN_AUX_CONFIG,
            device="cpu",
            cfg_overrides=self._tiny_overrides(),
        )
        self.assertEqual(cfg_aux.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg_aux.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertTrue(cfg_aux.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        self.assertTrue(cfg_aux.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED)
        self.assertAlmostEqual(cfg_aux.MODEL.MEM_GRAPH.VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT, 2.0)

        cfg_cnabu = setup_mem_graph_cfg(
            OPTION2B_PAIR_GEOMETRY_CNABU_MEAN_CONFIG,
            device="cpu",
            cfg_overrides=self._tiny_overrides(),
        )
        self.assertEqual(cfg_cnabu.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg_cnabu.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertEqual(cfg_cnabu.INPUT.MEM_MAP_FEATURE_SOURCE, "cnabu_mean")
        self.assertEqual(cfg_cnabu.MODEL.MEM_GRAPH.IN_CHANNELS, 16)
        self.assertEqual(cfg_cnabu.MODEL.MEM_GRAPH.INPUT_NORMALIZATION, "none")

        cfg_concat = setup_mem_graph_cfg(
            OPTION2B_PAIR_GEOMETRY_RAW_PLUS_CNABU_MEAN_CONFIG,
            device="cpu",
            cfg_overrides=self._tiny_overrides(),
        )
        self.assertEqual(cfg_concat.INPUT.MEM_NODE_SOURCE, "observed_instance_maps")
        self.assertEqual(cfg_concat.INPUT.MEM_GRAPH_TARGET_SCOPE, "observed_induced")
        self.assertEqual(cfg_concat.INPUT.MEM_MAP_FEATURE_SOURCE, "raw_plus_cnabu_mean")
        self.assertEqual(cfg_concat.MODEL.MEM_GRAPH.IN_CHANNELS, 46)
        self.assertEqual(cfg_concat.MODEL.MEM_GRAPH.INPUT_NORMALIZATION, "raw_plus_cnabu_mean_v0")

        cfg_option2c = setup_mem_graph_cfg(
            OPTION2C_CNABU_COMPONENTS_PAIR_GEOMETRY_CONFIG,
            device="cpu",
            cfg_overrides=self._tiny_overrides(),
        )
        self.assertEqual(cfg_option2c.INPUT.MEM_NODE_SOURCE, "cnabu_components")
        self.assertEqual(cfg_option2c.INPUT.MEM_GRAPH_TARGET_SCOPE, "cnabu_induced")
        self.assertEqual(cfg_option2c.INPUT.MEM_MAP_FEATURE_SOURCE, "cnabu_mean")
        self.assertTrue(cfg_option2c.MODEL.MEM_GRAPH.PAIR_GEOMETRY_ENABLED)
        self.assertEqual(cfg_option2c.MODEL.MEM_GRAPH.IN_CHANNELS, 16)

    def test_manifest_loading_registers_scene_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, split_ids = self._write_records_and_split(tmpdir)
            split_manifest = load_mem_graph_split_manifest(split_json)
            records_by_split = load_mem_graph_records_for_splits(records_json, split_manifest, require_scene_disjoint=True)
            cfg = setup_mem_graph_cfg(OPTION2A_CONFIG, device="cpu", cfg_overrides=self._tiny_overrides())
            dataset_names = register_mem_graph_datasets(cfg, records_by_split, dataset_prefix="mem_unit_trainer")

            try:
                self.assertEqual(set(records_by_split.keys()), {"train", "val", "test"})
                self.assertEqual([record["sample_id"] for record in records_by_split["train"]], split_ids["train"])
                self.assertEqual([record["sample_id"] for record in DatasetCatalog.get(dataset_names["val"])], split_ids["val"])
                self.assertTrue(dataset_names["train"].startswith("mem_unit_trainer"))
            finally:
                for name in dataset_names.values():
                    if name in DatasetCatalog.list():
                        DatasetCatalog.remove(name)

    def test_score_metrics_are_consistent(self):
        metrics = compute_mem_graph_score_metrics(
            scores=[0.9, 0.8, 0.2, 0.1],
            labels=[1, 0, 1, 0],
            thresholds=[0.5, 0.3, 0.15],
        )

        self.assertEqual(metrics["num_pairs"], 4)
        self.assertEqual(metrics["num_positive"], 2)
        self.assertEqual(metrics["num_negative"], 2)
        self.assertAlmostEqual(metrics["positive_edge_base_rate"], 0.5)
        self.assertAlmostEqual(metrics["validation_ap"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["ap_over_base_rate"], (5.0 / 6.0) / 0.5)
        at_05 = metrics["metrics_at_threshold_0_5"]
        self.assertEqual(at_05["true_positive"], 1)
        self.assertEqual(at_05["false_positive"], 1)
        self.assertEqual(at_05["false_negative"], 1)
        self.assertAlmostEqual(at_05["precision"], 0.5)
        self.assertAlmostEqual(at_05["recall"], 0.5)
        self.assertAlmostEqual(at_05["f1"], 0.5)
        self.assertEqual(metrics["best_threshold"], 0.15)
        self.assertGreater(metrics["best_f1"], at_05["f1"])
        self.assertAlmostEqual(metrics["positive_negative_probability_gap"], 0.1)
        self.assertNotIn("score_histogram", metrics)

    def test_average_precision_groups_tied_scores(self):
        self.assertAlmostEqual(
            compute_average_precision(
                scores=[0.5, 0.5, 0.5, 0.5],
                labels=[1, 0, 1, 0],
            ),
            0.5,
        )
        self.assertAlmostEqual(
            compute_average_precision(
                scores=[0.5, 0.5, 0.5, 0.5],
                labels=[0, 1, 0, 1],
            ),
            0.5,
        )
        metrics = compute_mem_graph_score_metrics(
            scores=[0.5, 0.5, 0.5, 0.5],
            labels=[1, 0, 1, 0],
            thresholds=[0.5],
        )

        self.assertAlmostEqual(metrics["validation_ap"], 0.5)
        self.assertAlmostEqual(metrics["ap_over_base_rate"], 1.0)

    def test_score_histogram_is_opt_in(self):
        metrics = compute_mem_graph_score_metrics(
            scores=[0.9, 0.8, 0.2, 0.1],
            labels=[1, 0, 1, 0],
            thresholds=[0.5],
            score_histogram_bin_edges=[0.0, 0.5, 1.0],
        )

        histogram = metrics["score_histogram"]
        self.assertEqual(histogram["schema"], "mem_d3g_score_histogram_v0")
        self.assertEqual(histogram["bin_edges"], [0.0, 0.5, 1.0])
        self.assertEqual(histogram["positive_counts"], [1, 1])
        self.assertEqual(histogram["negative_counts"], [1, 1])
        self.assertEqual(histogram["total_counts"], [2, 2])
        self.assertEqual(histogram["positive_total"], 2)
        self.assertEqual(histogram["negative_total"], 2)

    def test_prediction_dump_includes_nodes_matrices_errors_and_option2b_metadata(self):
        instances = Instances((8, 12))
        instances.gt_boxes = Boxes(torch.tensor([[0, 0, 5, 4], [5, 0, 10, 4], [1, 4, 8, 8]], dtype=torch.float32))
        instances.gt_classes = torch.tensor([2, 3, 4], dtype=torch.long)
        mapped = {
            "image_id": "scene1/000000001",
            "instances": instances,
            "graph_gt": torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.long),
            "mem_metadata": {
                "node_order_instance_ids": [10, 20, 30],
                "observed_visible_instance_ids": [10, 20, 30, 40],
                "gt_aligned_instance_ids": [10, 20, 30],
                "hidden_gt_instance_ids": [50],
                "unmatched_observed_instance_ids": [40],
                "observed_node_records": [
                    {"node_index": 0, "observed_instance_id": 10, "semantic_class_id": 2},
                    {"node_index": 1, "observed_instance_id": 20, "semantic_class_id": 3},
                    {"node_index": 2, "observed_instance_id": 30, "semantic_class_id": 4},
                ],
            },
        }
        probability_matrix = torch.tensor(
            [
                [0.01, 0.20, 0.72],
                [0.95, 0.01, 0.80],
                [0.40, 0.62, 0.01],
            ],
            dtype=torch.float32,
        )
        logits = torch.logit(probability_matrix.clamp(1e-4, 1.0 - 1e-4))

        dump = build_mem_prediction_dump(
            mapped,
            logits,
            mapped["graph_gt"],
            split_name="val",
            threshold=0.5,
            top_k=2,
        )

        self.assertEqual(dump["schema"], "mem_d3g_prediction_dump_v0")
        self.assertEqual(dump["sample_id"], "scene1/000000001")
        self.assertEqual(dump["split"], "val")
        self.assertEqual(dump["node_ids"], [10, 20, 30])
        self.assertEqual(dump["node_classes"], [2, 3, 4])
        self.assertEqual(dump["node_boxes_xyxy_abs"][1], [5.0, 0.0, 10.0, 4.0])
        self.assertEqual(dump["gt_adjacency_matrix"], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.assertAlmostEqual(dump["predicted_probability_matrix"][1][0], 0.95, places=5)
        self.assertEqual(dump["threshold"], 0.5)
        self.assertEqual(dump["edge_confusion_at_threshold"]["false_positive"], 3)
        self.assertEqual(dump["edge_confusion_at_threshold"]["false_negative"], 2)
        top_fp = dump["top_false_positive_directed_edges"][0]
        self.assertEqual((top_fp["source_node_id"], top_fp["target_node_id"]), (20, 10))
        self.assertAlmostEqual(top_fp["probability"], 0.95, places=5)
        self.assertEqual(top_fp["reverse_gt_label"], 1)
        top_fn = dump["top_false_negative_directed_edges"][0]
        self.assertEqual((top_fn["source_node_id"], top_fn["target_node_id"]), (10, 20))
        self.assertAlmostEqual(top_fn["probability"], 0.20, places=5)
        self.assertEqual(top_fn["reverse_predicted_label"], 1)
        option2b = dump["option2b_metadata"]
        self.assertEqual(option2b["observed_visible_instance_ids"], [10, 20, 30, 40])
        self.assertEqual(option2b["hidden_gt_instance_ids"], [50])
        self.assertEqual(option2b["unmatched_observed_instance_ids"], [40])
        self.assertEqual(len(option2b["observed_node_records"]), 3)

    def test_prediction_dump_is_opt_in_and_selected_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, split_ids = self._write_records_and_split(tmpdir)
            output_dir = tmpdir / "option2a_dump_run"
            selected_val_id = split_ids["val"][0]

            summary = run_mem_graph_training(
                config_file=OPTION2A_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=0,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5],
                prediction_dump_dir=output_dir / "prediction_dumps",
                prediction_dump_sample_ids=[selected_val_id],
                prediction_dump_top_k=3,
            )

            dump_summary = summary["prediction_dump"]
            self.assertTrue(dump_summary["enabled"])
            self.assertEqual(dump_summary["requested_sample_ids"], [selected_val_id])
            self.assertEqual(dump_summary["written_dump_count"], 1)
            dump_files = sorted((output_dir / "prediction_dumps").glob("*.json"))
            self.assertEqual(len(dump_files), 1)
            dump = json.loads(dump_files[0].read_text(encoding="utf-8"))
            self.assertEqual(dump["sample_id"], selected_val_id)
            self.assertEqual(dump["split"], "val")
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])
            self.assertFalse(any(path.suffix in {".pth", ".pt", ".ckpt", ".h5", ".hdf5"} for path in output_dir.rglob("*")))

    def test_tiny_option2a_train_eval_writes_expected_artifacts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir)
            output_dir = tmpdir / "option2a_run"

            summary = run_mem_graph_training(
                config_file=OPTION2A_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5, 0.3, 0.15],
                score_histogram_bin_edges=[0.0, 0.5, 1.0],
            )

            self.assertEqual(summary["schema"], "mem_d3g_train_eval_summary_v0")
            self.assertEqual(summary["mem_node_source"], "gt")
            self.assertEqual(summary["mem_graph_target_scope"], "gt_all")
            self.assertEqual(summary["loss_mode"], "train_pos_weighted_bce")
            self.assertEqual(summary["num_train_records"], 1)
            self.assertEqual(summary["num_val_records"], 1)
            self.assertEqual(summary["checkpoint_policy"], "disabled")
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])
            self.assertIn("validation_ap", summary["validation_metrics"])
            saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary, summary)
            saved_config = (output_dir / "config.yaml").read_text(encoding="utf-8")
            self.assertIn(f"MEM_RECORDS_JSON: {records_json}", saved_config)
            self.assertIn(f"MEM_SPLIT_JSON: {split_json}", saved_config)
            self.assertIn(f"ROOT: {tmpdir}", saved_config)
            self.assertIn(f"OUTPUT_DIR: {output_dir}", saved_config)
            self.assertIn("MAX_ITER: 1", saved_config)
            self.assertIn("LOSS_MODE: train_pos_weighted_bce", saved_config)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                [
                    "artifact_inventory.json",
                    "command.json",
                    "config.yaml",
                    "metrics.json",
                    "split_manifest.json",
                    "summary.json",
                ],
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_mem_graph_training(
                    config_file=OPTION2A_CONFIG,
                    records_json=records_json,
                    split_json=split_json,
                    output_dir=output_dir,
                    data_root=str(tmpdir),
                    device="cpu",
                    max_iter=1,
                    seed=0,
                    cfg_overrides=self._tiny_overrides(),
                    enable_checkpoints=False,
                )

    def test_tiny_option2b_train_eval_reports_visible_alignment_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir, option2b_instances=True)
            output_dir = tmpdir / "option2b_run"

            summary = run_mem_graph_training(
                config_file=OPTION2B_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5, 0.3, 0.15],
                score_histogram_bin_edges=[0.0, 0.5, 1.0],
            )

            self.assertEqual(summary["mem_node_source"], "observed_instance_maps")
            self.assertEqual(summary["mem_graph_target_scope"], "observed_induced")
            counts = summary["validation_metrics"]["option2b_node_counts"]
            self.assertEqual(counts["num_observed_visible_instances"], 3)
            self.assertEqual(counts["num_gt_aligned_instances"], 2)
            self.assertEqual(counts["num_hidden_gt_instances"], 0)
            self.assertEqual(counts["num_unmatched_observed_instances"], 1)
            self.assertIn("score_histogram", summary["validation_metrics"])
            self.assertEqual(summary["score_histogram_bin_edges"], [0.0, 0.5, 1.0])
            self.assertIn("artifact_inventory", summary)

    def test_tiny_option2b_pair_geometry_train_eval_uses_config_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir, option2b_instances=True)
            output_dir = tmpdir / "option2b_pair_geometry_run"

            summary = run_mem_graph_training(
                config_file=OPTION2B_PAIR_GEOMETRY_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5, 0.3, 0.15],
            )

            self.assertEqual(summary["mem_node_source"], "observed_instance_maps")
            self.assertEqual(summary["mem_graph_target_scope"], "observed_induced")
            self.assertTrue(summary["mem_pair_geometry_enabled"])
            self.assertIn("front_x_overlap_union", summary["mem_pair_geometry_features"])
            self.assertIn("validation_ap", summary["validation_metrics"])
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])
            saved_config = (output_dir / "config.yaml").read_text(encoding="utf-8")
            self.assertIn("PAIR_GEOMETRY_ENABLED: true", saved_config)
            self.assertIn("VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED: false", saved_config)
            self.assertFalse(any(path.suffix in {".pth", ".pt", ".ckpt", ".h5", ".hdf5"} for path in output_dir.rglob("*")))

    def test_tiny_option2c_train_eval_reports_cnabu_component_counts_and_supervised_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cnabu_root = tmpdir / "cnabu"
            records_json, split_json, _ = self._write_records_and_split(
                tmpdir,
                option2c_components=True,
                cnabu_root=cnabu_root,
            )
            output_dir = tmpdir / "option2c_cnabu_components_run"

            summary = run_mem_graph_training(
                config_file=OPTION2C_CNABU_COMPONENTS_PAIR_GEOMETRY_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides()
                + [
                    "DATASETS.MEM_CNABU_DERIVED_ROOT",
                    str(cnabu_root),
                ],
                enable_checkpoints=False,
                thresholds=[0.5, 0.3, 0.15],
            )

            self.assertEqual(summary["mem_node_source"], "cnabu_components")
            self.assertEqual(summary["mem_graph_target_scope"], "cnabu_induced")
            self.assertTrue(summary["mem_pair_geometry_enabled"])
            self.assertEqual(summary["train_target_summary"]["num_nodes"], 3)
            self.assertEqual(summary["train_target_summary"]["num_non_diagonal_pairs"], 2)
            self.assertEqual(summary["train_target_summary"]["num_total_non_diagonal_pairs"], 6)
            self.assertTrue(summary["train_target_summary"]["uses_graph_loss_mask"])
            counts = summary["validation_metrics"]["cnabu_component_node_counts"]
            self.assertEqual(counts["num_cnabu_components"], 3)
            self.assertEqual(counts["num_cnabu_matched_components"], 2)
            self.assertEqual(counts["num_cnabu_unmatched_components"], 1)
            self.assertEqual(counts["pseudo_node_false_positive_count"], 1)
            self.assertEqual(counts["gt_node_false_negative_count"], 0)
            self.assertEqual(summary["validation_metrics"]["num_pairs"], 2)
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])

    def test_tiny_option2b_pair_geometry_obb_train_eval_uses_obb_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir, option2b_instances=True)
            output_dir = tmpdir / "option2b_pair_geometry_obb_run"

            summary = run_mem_graph_training(
                config_file=OPTION2B_PAIR_GEOMETRY_OBB_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5, 0.3, 0.15],
            )

            self.assertEqual(summary["mem_node_source"], "observed_instance_maps")
            self.assertEqual(summary["mem_graph_target_scope"], "observed_induced")
            self.assertTrue(summary["mem_pair_geometry_enabled"])
            for required in (
                "front_x_overlap_union",
                "relative_theta_sin",
                "relative_theta_cos",
                "obb_aspect_ratio_min_over_max",
            ):
                self.assertIn(required, summary["mem_pair_geometry_features"])
            self.assertIn("validation_ap", summary["validation_metrics"])
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])
            saved_config = (output_dir / "config.yaml").read_text(encoding="utf-8")
            self.assertIn("MEM_BOX_MODE: obb_from_mask", saved_config)
            self.assertIn("USE_OBB_FEATURES: true", saved_config)
            self.assertFalse(
                any(path.suffix in {".pth", ".pt", ".ckpt", ".h5", ".hdf5"} for path in output_dir.rglob("*"))
            )

    def test_tiny_option2b_pair_geometry_visible_hidden_aux_reports_separate_metrics_and_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(
                tmpdir,
                option2b_instances=True,
                hidden_gt_node=True,
            )
            output_dir = tmpdir / "option2b_pair_geometry_visible_hidden_aux_run"

            summary = run_mem_graph_training(
                config_file=OPTION2B_PAIR_GEOMETRY_VISIBLE_HIDDEN_AUX_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5, 0.3, 0.15],
            )

            self.assertTrue(summary["mem_pair_geometry_enabled"])
            self.assertTrue(summary["mem_visible_blocks_hidden_aux_enabled"])
            self.assertAlmostEqual(summary["mem_visible_blocks_hidden_aux_loss_weight"], 0.1)
            self.assertAlmostEqual(summary["mem_visible_blocks_hidden_aux_pos_weight"], 2.0)
            self.assertIn("loss_mem_visible_blocks_hidden_aux", summary["train_history"][0])
            self.assertNotIn("loss_mem_hidden_blocks_visible_aux", summary["train_history"][0])
            self.assertIn("validation_ap", summary["validation_metrics"])
            aux_metrics = summary["validation_metrics"]["visible_blocks_hidden_aux_metrics"]
            self.assertTrue(aux_metrics["enabled"])
            self.assertEqual(aux_metrics["available_records"], 1)
            self.assertEqual(aux_metrics["num_nodes"], 2)
            self.assertEqual(aux_metrics["num_positive"], 1)
            self.assertEqual(aux_metrics["num_negative"], 1)
            self.assertAlmostEqual(aux_metrics["base_rate"], 0.5)
            self.assertAlmostEqual(aux_metrics["positive_rate"], 0.5)
            self.assertIsNotNone(aux_metrics["ap"])
            self.assertIsNotNone(aux_metrics["ap_over_base_rate"])
            self.assertIn("f1", aux_metrics["metrics_at_threshold_0_5"])
            self.assertIsNotNone(aux_metrics["bce_loss"])
            self.assertIsNotNone(aux_metrics["scaled_bce_loss"])
            self.assertAlmostEqual(aux_metrics["pos_weight"], 2.0)
            saved_config = (output_dir / "config.yaml").read_text(encoding="utf-8")
            self.assertIn("VISIBLE_BLOCKS_HIDDEN_AUX_ENABLED: true", saved_config)
            self.assertIn("VISIBLE_BLOCKS_HIDDEN_AUX_POS_WEIGHT: 2.0", saved_config)
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])

    def test_controlled_checkpointing_writes_best_validation_and_final_only_to_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir, option2b_instances=True)
            output_dir = tmpdir / "option2b_checkpointed_run"
            checkpoint_dir = tmpdir / "controlled_checkpoints"

            summary = run_mem_graph_training(
                config_file=OPTION2B_PAIR_GEOMETRY_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=True,
                checkpoint_dir=checkpoint_dir,
                checkpoint_eval_period=1,
                thresholds=[0.5, 0.3, 0.15],
            )

            self.assertEqual(summary["checkpoint_policy"], "best_validation_and_final")
            self.assertTrue(summary["safety"]["writes_checkpoints_or_model_outputs"])
            self.assertEqual(summary["safety"]["checkpoint_count"], 2)
            self.assertFalse(summary["safety"]["writes_optimizer_state"])
            checkpoint_artifacts = summary["checkpoint_artifacts"]
            self.assertTrue(checkpoint_artifacts["enabled"])
            self.assertEqual(
                sorted(path.name for path in checkpoint_dir.iterdir()),
                ["model_best_validation.pth", "model_final.pth"],
            )
            self.assertFalse(any(path.suffix in {".pth", ".pt", ".ckpt"} for path in output_dir.iterdir()))
            best_payload = torch.load(checkpoint_dir / "model_best_validation.pth", map_location="cpu")
            final_payload = torch.load(checkpoint_dir / "model_final.pth", map_location="cpu")
            self.assertEqual(best_payload["schema"], "mem_d3g_checkpoint_v1")
            self.assertEqual(best_payload["role"], "best_validation")
            self.assertEqual(final_payload["role"], "final")
            self.assertFalse(best_payload["contains_optimizer_state"])
            self.assertIn("validation_ap", best_payload["validation_metrics"])
            self.assertFalse(any(path.suffix in {".h5", ".hdf5"} for path in tmpdir.rglob("*")))

    def test_fixed_pos_weighted_bce_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir, option2b_instances=True)
            output_dir = tmpdir / "option2b_fixed_weight_run"

            summary = run_mem_graph_training(
                config_file=OPTION2B_CONFIG,
                records_json=records_json,
                split_json=split_json,
                output_dir=output_dir,
                data_root=str(tmpdir),
                device="cpu",
                max_iter=1,
                seed=0,
                loss_mode="fixed_pos_weighted_bce",
                fixed_graph_loss_pos_weight=4.0,
                cfg_overrides=self._tiny_overrides(),
                enable_checkpoints=False,
                thresholds=[0.5],
            )

            self.assertEqual(summary["loss_mode"], "fixed_pos_weighted_bce")
            self.assertEqual(summary["fixed_graph_loss_pos_weight"], 4.0)
            self.assertEqual(summary["graph_loss_pos_weight"], 4.0)
            self.assertEqual(summary["graph_loss_pos_weight_source"], "fixed_graph_loss_pos_weight")
            self.assertFalse(summary["safety"]["writes_checkpoints_or_model_outputs"])
            saved_config = (output_dir / "config.yaml").read_text(encoding="utf-8")
            self.assertIn("LOSS_MODE: fixed_pos_weighted_bce", saved_config)
            self.assertIn("GRAPH_LOSS_POS_WEIGHT: 4.0", saved_config)

    def test_fixed_pos_weighted_bce_requires_positive_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records_json, split_json, _ = self._write_records_and_split(tmpdir, option2b_instances=True)
            with self.assertRaisesRegex(ValueError, "fixed_graph_loss_pos_weight"):
                run_mem_graph_training(
                    config_file=OPTION2B_CONFIG,
                    records_json=records_json,
                    split_json=split_json,
                    output_dir=tmpdir / "option2b_missing_fixed_weight_run",
                    data_root=str(tmpdir),
                    device="cpu",
                    max_iter=0,
                    seed=0,
                    loss_mode="fixed_pos_weighted_bce",
                    cfg_overrides=self._tiny_overrides(),
                    enable_checkpoints=False,
                )


if __name__ == "__main__":
    unittest.main()
