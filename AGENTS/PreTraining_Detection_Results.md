# Pretraining Detection Results

This note summarizes the detector pretraining runs completed for D3G / MetaGraspNetV2.

## Run Status

| Run | Output directory | Status |
|---|---|---|
| Mask R-CNN Base | `output/rcnn_pretrain_maskrcnn_base_gpu1_2` | Completed |
| Deformable DETR Base | `output/defdetr_pretrain_base_gpu3` | Completed |
| Deformable DETR Base + instance masks | `output/defdetr_pretrain_instancemasks_on_gpu1_2_v2` | Completed |
| DETR Base | `output/detr_pretrain_base_default_gpu0` | Completed after fix/resume |

All completed runs reached:

```text
iteration 60000
model_final.pth
```

## Important Caveats

- `output/defdetr_pretrain_instancemasks_on_gpu1_2` was the first failed attempt for Deformable DETR with masks. Ignore that directory.
- The valid Deformable DETR with instance masks run is:

```text
output/defdetr_pretrain_instancemasks_on_gpu1_2_v2
```

- DETR Base initially crashed around iteration `13148` with a GIoU invalid-box assertion.
- The DETR Base run was fixed by guarding predicted boxes in `models/detr_modules/detr.py`:

```python
outputs_coord = self.bbox_embed(hs).sigmoid()
outputs_coord = torch.nan_to_num(outputs_coord, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
```

- DETR Base was then resumed from:

```text
output/detr_pretrain_base_default_gpu0/model_0009999.pth
```

and completed successfully.

## Final Detection mAP

| Model | Synth Easy | Synth Medium | Synth Hard | Real Test |
|---|---:|---:|---:|---:|
| Mask R-CNN Base | 0.837 | 0.742 | 0.614 | 0.293 |
| Deformable DETR Base | 0.893 | 0.783 | 0.702 | 0.471 |
| Deformable DETR + masks | 0.892 | 0.814 | 0.697 | 0.430 |
| DETR Base | 0.853 | 0.741 | 0.641 | 0.475 |

## Final Checkpoints

Use these checkpoints for downstream graph/relation training:

```text
output/rcnn_pretrain_maskrcnn_base_gpu1_2/model_final.pth
output/defdetr_pretrain_base_gpu3/model_final.pth
output/defdetr_pretrain_instancemasks_on_gpu1_2_v2/model_final.pth
output/detr_pretrain_base_default_gpu0/model_final.pth
```

## Interpretation

Best synthetic detector overall:

```text
Deformable DETR Base
```

Best real-test detector:

```text
DETR Base
```

Real-test comparison:

```text
DETR Base real mAP:            0.475
Deformable DETR Base real mAP: 0.471
Deformable DETR + masks mAP:   0.430
Mask R-CNN Base real mAP:      0.293
```

The strongest detector-pretrain candidates for the next graph/relation stage are:

```text
output/detr_pretrain_base_default_gpu0/model_final.pth
output/defdetr_pretrain_base_gpu3/model_final.pth
```

