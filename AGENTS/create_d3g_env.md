# Create the `d3g` Conda Environment

This documents the environment setup performed for this project from `D3G/README.md`, plus the extra dependencies required by the source imports.

## Starting Point

Work from the project root:

```bash
cd /home/user/ehsanullahm1/thesis/D3G
```

The README says to create a fresh Python environment and install:

```bash
pip install -r requirements.txt
```

The repository also imports packages that are not listed in `requirements.txt`, including `detectron2`, `cv2`, `imageio`, and `scipy`.

## Create the Environment

The default Anaconda channels required Terms of Service acceptance, so the environment was created using `conda-forge` only:

```bash
conda create --name d3g --override-channels -c conda-forge python=3.10 pip -y
```

Expected environment path:

```text
/home/user/ehsanullahm1/miniconda3/envs/d3g
```

## GPU Visibility Note

In the non-escalated sandbox, `nvidia-smi` initially failed:

```bash
nvidia-smi
```

Result:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

That was a sandbox visibility issue, not the actual host state. With escalated execution, the machine exposes four RTX A6000 GPUs:

```bash
nvidia-smi
```

Observed GPU summary:

```text
Driver Version: 590.48.01
CUDA Version: 13.1
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000
GPU 2: NVIDIA RTX A6000
GPU 3: NVIDIA RTX A6000
```

## Install PyTorch CUDA Stack

Install PyTorch CUDA 12.4 wheels. The NVIDIA 590 driver supports this runtime:

```bash
conda run -n d3g python -m pip install --upgrade --force-reinstall \
  torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

## Install README Requirements

Install the project requirements while keeping the CUDA PyTorch build pinned:

```bash
conda run -n d3g python -m pip install \
  -r requirements.txt \
  torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

## Install Missing Runtime Dependencies

These packages are imported by the project but are not declared in `requirements.txt`:

```bash
conda run -n d3g python -m pip install opencv-python-headless imageio scipy
```

## Install Detectron2

`main.py` and most model/data modules require Detectron2. Installing directly with pip build isolation failed because Detectron2's setup could not see the already-installed `torch`, so it was installed with build isolation disabled:

```bash
conda run -n d3g python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'
```

After installing CUDA PyTorch `2.4.1+cu124`, Detectron2 was rebuilt against that PyTorch installation:

```bash
conda run -n d3g python -m pip install \
  --force-reinstall --no-deps --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'
```

Installed Detectron2 source revision:

```text
facebookresearch/detectron2@b599f139756bd3646a26a909caf86a1a159e53a7
```

## Matplotlib Cache Setting

The sandbox could not write to the default Matplotlib config path under the user home directory. A writable temporary cache path was configured for the conda environment:

```bash
mkdir -p /tmp/d3g-matplotlib
conda env config vars set -n d3g MPLCONFIGDIR=/tmp/d3g-matplotlib
```

Reactivate the environment after setting conda env vars:

```bash
conda deactivate
conda activate d3g
```

## Verification

Check the environment exists:

```bash
conda env list
```

Expected entry:

```text
d3g    /home/user/ehsanullahm1/miniconda3/envs/d3g
```

Check package consistency:

```bash
conda run -n d3g python -m pip check
```

Expected result:

```text
No broken requirements found.
```

Check core imports and versions:

```bash
conda run -n d3g python -c "import torch, torchvision, detectron2, transformers, torch_geometric, cv2, imageio, scipy; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('detectron2', detectron2.__version__); print('transformers', transformers.__version__)"
```

Observed result:

```text
torch 2.4.1+cu124 cuda_available True
torchvision 0.19.1+cu124
detectron2 0.6
transformers 5.6.2
```

Check GPU count and names:

```bash
conda run -n d3g python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Observed result:

```text
2.4.1+cu124 12.4 True 4
['NVIDIA RTX A6000', 'NVIDIA RTX A6000', 'NVIDIA RTX A6000', 'NVIDIA RTX A6000']
```

Check the project entrypoint:

```bash
conda run -n d3g python main.py --help
```

Expected result: the Detectron2/default trainer CLI help is displayed.

## Running the Project

Example from the README, adjusted to an existing config file:

```bash
conda activate d3g
cd /home/user/ehsanullahm1/thesis/D3G
python main.py --config-file configs/detr_base.yaml --data-path /your/data/path
```

The README example references `configs/config.yaml`, but this repository currently does not contain that file.

## Optional Deformable DETR CUDA Op

The README says to build the optional Deformable DETR op with:

```bash
cd /home/user/ehsanullahm1/thesis/D3G/models/deformable_detr_modules/ops
conda run -n d3g sh ./make.sh
conda run -n d3g python ./test.py
```

The op was built successfully after CUDA PyTorch was installed and GPU visibility was verified:

```bash
cd /home/user/ehsanullahm1/thesis/D3G/models/deformable_detr_modules/ops
conda run -n d3g sh ./make.sh
```

The build used `/usr/local/cuda/bin/nvcc` and installed:

```text
MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so
```

The build emitted a warning because local CUDA was detected as 12.9 while PyTorch was compiled with CUDA 12.4:

```text
The detected CUDA version (12.9) has a minor version mismatch with the version that was used to compile PyTorch (12.4). Most likely this shouldn't be a problem.
```

The provided `test.py` is a very large gradient-check test. It passed the forward checks and gradient checks through `D=1025`, then failed at `D=2048` due to GPU memory pressure:

```text
* True check_forward_equal_with_pytorch_double
* True check_forward_equal_with_pytorch_float
* True check_gradient_numerical(D=30)
* True check_gradient_numerical(D=32)
* True check_gradient_numerical(D=64)
* True check_gradient_numerical(D=71)
* True check_gradient_numerical(D=1025)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.50 GiB.
```

Result: the D3G environment is installed with CUDA-enabled PyTorch, Detectron2 is rebuilt against that CUDA PyTorch, the project entrypoint imports, and the optional Deformable DETR CUDA extension builds successfully. The full upstream op test can exceed the available memory on a 48 GB A6000 because of its largest gradient-check cases.
