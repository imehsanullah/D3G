# Running DETR Base Detection

This note documents the DETR Base detector pretraining run and the fix needed when the default run crashed during matching.

## Config

Use the default DETR detector pretraining config:

```bash
./configs/pretrains/detr_pretrain.yaml
```

Important defaults:

```text
MODEL.META_ARCHITECTURE: "Detr"
MODEL.WEIGHTS: "./checkpoints/detr-r50-dc5.pth"
MODEL.MASK_ON: False
MODEL.DETR.NUM_CLASSES: 98
SOLVER.IMS_PER_BATCH: 32
SOLVER.MAX_ITER: 60000
```

This is standard DETR detection pretraining. It does not use instance masks.

## GPU 0 Run

The GPU 0 run was started with:

```bash
tmux new-session -d -s d3g_detr_pretrain_base_gpu0 \
  "cd /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/D3G && \
   CUDA_VISIBLE_DEVICES=0 \
   /home/user/ehsanullahm1/miniconda3/envs/d3g/bin/python main.py \
     --num-gpus 1 \
     --config-file ./configs/pretrains/detr_pretrain.yaml \
     --data-path ./datasets \
     OUTPUT_DIR ./output/detr_pretrain_base_default_gpu0 \
     2>&1 | tee -a ./output/detr_pretrain_base_default_gpu0/train.log"
```

Monitor it with:

```bash
tail -f ./output/detr_pretrain_base_default_gpu0/train.log
```

## Crash Observed

The first run crashed around iteration `13148` with:

```text
AssertionError
utils/box_ops.py:51 in generalized_box_iou
assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
```

The stack trace came from:

```text
models/detr_modules/matcher.py
cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
```

The run had already saved:

```text
output/detr_pretrain_base_default_gpu0/model_0004999.pth
output/detr_pretrain_base_default_gpu0/model_0009999.pth
```

`last_checkpoint` pointed to:

```text
model_0009999.pth
```

## Required Fix

Patch DETR box predictions in:

```text
models/detr_modules/detr.py
```

Change:

```python
outputs_coord = self.bbox_embed(hs).sigmoid()
```

to:

```python
outputs_coord = self.bbox_embed(hs).sigmoid()
outputs_coord = torch.nan_to_num(outputs_coord, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
```

Reason: DETR box predictions pass through `sigmoid()`, so valid finite values should already be in `[0, 1]`. The observed GIoU assertion is consistent with non-finite predicted box values reaching `box_cxcywh_to_xyxy()`. The guard replaces `NaN`/infinities and clamps the result before Hungarian matching and GIoU.

Validate syntax after patching:

```bash
/home/user/ehsanullahm1/miniconda3/envs/d3g/bin/python -m py_compile models/detr_modules/detr.py
```

## Resume Command

Resume the same run from `model_0009999.pth`:

```bash
tmux new-session -d -s d3g_detr_pretrain_base_gpu0_resume \
  "cd /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/D3G && \
   CUDA_VISIBLE_DEVICES=0 \
   /home/user/ehsanullahm1/miniconda3/envs/d3g/bin/python main.py \
     --resume \
     --num-gpus 1 \
     --config-file ./configs/pretrains/detr_pretrain.yaml \
     --data-path ./datasets \
     OUTPUT_DIR ./output/detr_pretrain_base_default_gpu0 \
     2>&1 | tee -a ./output/detr_pretrain_base_default_gpu0/train.log"
```

Expected confirmation in the log:

```text
Loading from ./output/detr_pretrain_base_default_gpu0/model_0009999.pth
Starting training from iteration 10000
```

Monitor resumed training:

```bash
tail -f ./output/detr_pretrain_base_default_gpu0/train.log
```

