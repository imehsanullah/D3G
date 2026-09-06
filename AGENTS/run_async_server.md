# Running Long Jobs With tmux

Use `tmux` for long-running jobs such as training so the process keeps running after the terminal disconnects.

## Pattern

Start a detached session:

```bash
tmux new-session -d -s SESSION_NAME "COMMAND"
```

Attach to watch it interactively:

```bash
tmux attach -t SESSION_NAME
```

Detach without stopping the job:

```text
Ctrl-b then d
```

List running sessions:

```bash
tmux ls
```

Stop a session:

```bash
tmux kill-session -t SESSION_NAME
```

## Training Example

This is the pattern used to start Mask R-CNN pretraining on physical GPUs `1` and `2`:

```bash
tmux new-session -d -s d3g_rcnn_pretrain_gpu12 \
  "cd /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/D3G && \
   CUDA_VISIBLE_DEVICES=1,2 \
   /home/user/ehsanullahm1/miniconda3/envs/d3g/bin/python main.py \
     --num-gpus 2 \
     --config-file ./configs/pretrains/rcnn_pretrain.yaml \
     --data-path ./datasets \
     OUTPUT_DIR ./output/rcnn_pretrain_maskrcnn_base_gpu1_2 \
     2>&1 | tee -a ./output/rcnn_pretrain_maskrcnn_base_gpu1_2/train.log"
```

Important details:

- `CUDA_VISIBLE_DEVICES=1,2` restricts the job to physical GPUs `1` and `2`.
- `--num-gpus 2` tells Detectron2 to launch two workers.
- `OUTPUT_DIR ...` keeps checkpoints and event files separate from other runs.
- `2>&1 | tee -a .../train.log` writes stdout and stderr to both the tmux window and a log file.
- Use the absolute Python path from the `d3g` conda environment so the job does not depend on interactive shell activation.

## Monitoring

Watch the log without attaching to tmux:

```bash
tail -f /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/D3G/output/rcnn_pretrain_maskrcnn_base_gpu1_2/train.log
```

Check GPU usage:

```bash
nvidia-smi
```

Check which processes are running:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'main.py|torchrun|launch' | grep -v grep
```

## Notes

If `tmux` cannot be accessed from a sandboxed command, rerun the `tmux` command outside the sandbox.

For reproducibility, always put each long run in its own named session and its own `OUTPUT_DIR`.
