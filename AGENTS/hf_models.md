# Hugging Face Models

This note records the Hugging Face archive for the completed D3G / MetaGraspNetV2 detector pretraining runs.

## Repository

```text
https://huggingface.co/iamehsanullah/d3g-metagraspnetv2-detector-pretrains
```

The repository is a checkpoint archive for D3G detector pretraining. These are PyTorch research checkpoints from this codebase, not Hugging Face `AutoModel` checkpoints.

## Uploaded Files

```text
checkpoints/mask_rcnn_base/model_final.pth
checkpoints/deformable_detr_base/model_final.pth
checkpoints/deformable_detr_instancemasks/model_final.pth
checkpoints/detr_base/model_final.pth

metrics/mask_rcnn_base_metrics.json
metrics/deformable_detr_base_metrics.json
metrics/deformable_detr_instancemasks_metrics.json
metrics/detr_base_metrics.json

logs/mask_rcnn_base_train.log
logs/deformable_detr_base_train.log
logs/deformable_detr_instancemasks_train.log
logs/detr_base_train.log
```

## Local-To-Hub Mapping

| Model | Local checkpoint | Hugging Face checkpoint |
|---|---|---|
| Mask R-CNN Base | `output/rcnn_pretrain_maskrcnn_base_gpu1_2/model_final.pth` | `checkpoints/mask_rcnn_base/model_final.pth` |
| Deformable DETR Base | `output/defdetr_pretrain_base_gpu3/model_final.pth` | `checkpoints/deformable_detr_base/model_final.pth` |
| Deformable DETR + instance masks | `output/defdetr_pretrain_instancemasks_on_gpu1_2_v2/model_final.pth` | `checkpoints/deformable_detr_instancemasks/model_final.pth` |
| DETR Base | `output/detr_pretrain_base_default_gpu0/model_final.pth` | `checkpoints/detr_base/model_final.pth` |

## Results

| Model | Synth Easy | Synth Medium | Synth Hard | Real Test |
|---|---:|---:|---:|---:|
| Mask R-CNN Base | 0.837 | 0.742 | 0.614 | 0.293 |
| Deformable DETR Base | 0.893 | 0.783 | 0.702 | 0.471 |
| Deformable DETR + masks | 0.892 | 0.814 | 0.697 | 0.430 |
| DETR Base | 0.853 | 0.741 | 0.641 | 0.475 |

## Caveats

- `output/defdetr_pretrain_instancemasks_on_gpu1_2` was a failed first attempt and was not uploaded.
- The valid Deformable DETR + instance masks run is `output/defdetr_pretrain_instancemasks_on_gpu1_2_v2`.
- DETR Base initially crashed around iteration `13148`, was fixed, resumed from `output/detr_pretrain_base_default_gpu0/model_0009999.pth`, and completed successfully.
- The uploaded DETR Base checkpoint is the completed final result: `output/detr_pretrain_base_default_gpu0/model_final.pth`.
- Intermediate checkpoints were not uploaded because the final checkpoints are the intended downstream graph/relation training inputs.

