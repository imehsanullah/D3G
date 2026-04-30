# Running Training

This project uses a two-stage training flow:

1. Pretrain the detector on MetaGraspNetV2 detection.
2. Train/fine-tune the full graph/relation model using the detector pretrain checkpoint from step 1.

The full dataset is already available through the repository symlink:

```bash
cd /home/user/ehsanullahm1/thesis/D3G
ls -l datasets
```

Expected:

```text
datasets -> /tmp/d3g_datasets
```

## Current Status

The environment and dataset are ready for training.

Verified environment:

- Conda environment: `d3g`
- CUDA PyTorch is installed.
- PyTorch sees the local multi-GPU machine when run outside the sandbox.
- Deformable DETR CUDA ops were built successfully.
- A 1-iteration DETR pretraining smoke test completed successfully on the full dataset.

Downloaded author-provided starter checkpoints:

```text
checkpoints/detr-r50-dc5.pth
checkpoints/r50_deformable_detr-checkpoint.pth
```

These are fixed-key COCO starter checkpoints. They are not the final D3G/MetaGraspNetV2 trained model weights.

The final graph/relation configs expect detector-pretrained checkpoints that still need to be produced locally:

```text
checkpoints/pretrain_metagraspnetv2_detr.pth
checkpoints/pretrain_metagraspnetv2_dfedetr.pth
checkpoints/pretrain_metagraspnetv2_maskrcnn.pth
```

## Activate Environment

```bash
conda activate d3g
cd /home/user/ehsanullahm1/thesis/D3G
```

## DETR Pipeline

Run DETR detector pretraining first:

```bash
python main.py \
  --num-gpus 4 \
  --config-file ./configs/pretrains/detr_pretrain.yaml \
  --data-path ./datasets
```

The config uses:

```text
MODEL.WEIGHTS: ./checkpoints/detr-r50-dc5.pth
SOLVER.IMS_PER_BATCH: 32
SOLVER.MAX_ITER: 60000
```

After pretraining finishes, place or rename the selected checkpoint to:

```text
checkpoints/pretrain_metagraspnetv2_detr.pth
```

Then run the DETR graph/relation training:

```bash
python main.py \
  --num-gpus 4 \
  --config-file ./configs/detr_graphdense.yaml \
  --data-path ./datasets
```

The graph config expects:

```text
MODEL.WEIGHTS: ./checkpoints/pretrain_metagraspnetv2_detr.pth
SOLVER.IMS_PER_BATCH: 32
SOLVER.MAX_ITER: 30000
```

## Deformable DETR Pipeline

Run Deformable DETR detector pretraining:

```bash
python main.py \
  --num-gpus 4 \
  --config-file ./configs/pretrains/defdetr_pretrain.yaml \
  --data-path ./datasets
```

The config uses:

```text
MODEL.WEIGHTS: ./checkpoints/r50_deformable_detr-checkpoint.pth
SOLVER.IMS_PER_BATCH: 32
SOLVER.MAX_ITER: 60000
```

After pretraining finishes, place or rename the selected checkpoint to:

```text
checkpoints/pretrain_metagraspnetv2_dfedetr.pth
```

Then run Deformable DETR graph/relation training:

```bash
python main.py \
  --num-gpus 4 \
  --config-file ./configs/def_detr_graphdense.yaml \
  --data-path ./datasets
```

The graph config expects:

```text
MODEL.WEIGHTS: ./checkpoints/pretrain_metagraspnetv2_dfedetr.pth
SOLVER.IMS_PER_BATCH: 32
SOLVER.MAX_ITER: 30000
```

## Mask R-CNN Pipeline

Run Mask R-CNN detector pretraining:

```bash
python main.py \
  --num-gpus 4 \
  --config-file ./configs/pretrains/rcnn_pretrain.yaml \
  --data-path ./datasets
```

The base Mask R-CNN config starts from the public Detectron2 COCO checkpoint:

```text
MODEL.WEIGHTS: https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl
```

After pretraining finishes, place or rename the selected checkpoint to:

```text
checkpoints/pretrain_metagraspnetv2_maskrcnn.pth
```

Then run one of the Mask R-CNN graph/relation configs:

```bash
python main.py \
  --num-gpus 4 \
  --config-file ./configs/gru_gnn.yaml \
  --data-path ./datasets
```

Alternative Mask R-CNN graph/relation configs:

```text
configs/pair_gnn.yaml
configs/vrmn.yaml
```

These configs expect:

```text
MODEL.WEIGHTS: ./checkpoints/pretrain_metagraspnetv2_maskrcnn.pth
SOLVER.IMS_PER_BATCH: 32
SOLVER.MAX_ITER: 30000
```

## Smoke Test Command

This is the short command used to confirm DETR pretraining can start on the current machine and dataset:

```bash
python main.py \
  --num-gpus 1 \
  --config-file ./configs/pretrains/detr_pretrain.yaml \
  --data-path ./datasets \
  SOLVER.MAX_ITER=1 \
  SOLVER.IMS_PER_BATCH=1 \
  DATALOADER.NUM_WORKERS=0 \
  TEST.EVAL_PERIOD=0 \
  SOLVER.CHECKPOINT_PERIOD=100000 \
  'DATASETS.TEST=()' \
  'DATASETS.EVAL=()' \
  OUTPUT_DIR=/tmp/d3g_smoke_detr_pretrain
```

Important shell detail: quote tuple-style overrides like `'DATASETS.TEST=()'`; otherwise Bash treats `()` as syntax.

Observed result:

- `./checkpoints/detr-r50-dc5.pth` loaded.
- Classifier head mismatch was skipped as expected because COCO has 81 classes and this project uses 99 classes.
- Full training dataset was built and serialized.
- Iteration 0 completed without crashing.

## Practical Notes

Use `--num-gpus 4` for full runs on this machine. The default pretraining batch sizes are intended for multi-GPU training.

If a run hits CUDA OOM, reduce the global batch size with a config override, for example:

```bash
SOLVER.IMS_PER_BATCH=16
```

If `/tmp` is cleaned by the system, the `datasets` symlink will break and the dataset must be restored using `AGENTS/downloading_datasets.md`.

Full graph/relation training should not be started until the relevant `pretrain_metagraspnetv2_*.pth` checkpoint exists, unless intentionally training from scratch by overriding `MODEL.WEIGHTS`.
