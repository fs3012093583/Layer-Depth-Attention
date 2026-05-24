# FlashAttention Environment Setup For The Server

This project does not currently integrate `flash-attn`, and the default Windows
environment on the server is not suitable for it as-is.

## Current server status

Checked on April 11, 2026:

- Windows host with `NVIDIA GeForce RTX 4060 Ti`
- Windows conda env `pt-3.9`
- Python `3.9.20`
- PyTorch `1.13.1+cu116`
- WSL2 Ubuntu `24.04.2 LTS` is already installed and running
- GPU is visible inside WSL2
- `nvcc` is not currently installed inside WSL2

## Why the current Windows env is the wrong target

According to the official `flash-attn` repository, current installation
requirements include:

- `PyTorch 2.2 and above`
- `CUDA toolkit 12.0 and above`
- `Linux`

The README also notes that Windows "might work" for some versions, but Windows
compilation still requires more testing. In practice, the most reliable path on
this server is WSL2 Ubuntu, not native Windows Python.

Official source:

- <https://github.com/Dao-AILab/flash-attention>

## Recommended layout

Use WSL2 Ubuntu as the runtime for training and keep a dedicated environment for
FlashAttention.

Recommended stack:

- OS: WSL2 Ubuntu 24.04
- Python: 3.10
- PyTorch: 2.2+ with CUDA 12.x wheels
- CUDA toolkit: 12.x installed inside WSL2 so `nvcc` is available
- Package: `flash-attn`

## One-time setup in WSL2

Open the server and enter WSL:

```powershell
wsl
```

Install a CUDA toolkit inside WSL if `nvcc --version` is missing. The exact
package name can vary by your NVIDIA apt repo, but the target is a CUDA 12.x
toolkit that exposes `nvcc`.

After that, create the Python environment and install the project:

```bash
cd ~/projects
git clone /mnt/d/Projects/Layer-Depth-Attention Layer-Depth-Attention
cd Layer-Depth-Attention
bash scripts/setup_flash_attn_wsl.sh
```

If you prefer to work directly on the Windows-mounted repo, replace the project
path with:

```bash
cd /mnt/d/Projects/Layer-Depth-Attention
bash scripts/setup_flash_attn_wsl.sh
```

Running from the Linux filesystem under `~/projects` is usually faster and more
stable than `/mnt/d/...` for builds.

## What the setup script does

The setup script:

1. Checks that it is running on Linux with GPU visibility.
2. Refuses to continue if `nvcc` is missing, because `flash-attn` needs a real
   CUDA toolkit during build.
3. Creates a conda environment named `flashattn-py310`.
4. Installs PyTorch CUDA wheels.
5. Installs build helpers: `packaging`, `psutil`, `ninja`, `wheel`,
   `setuptools`.
6. Installs `flash-attn`.
7. Installs this repo in editable mode.
8. Runs a verification script.

## Verification

After setup:

```bash
conda activate flashattn-py310
python scripts/verify_flash_attn.py
```

Expected result:

- `torch.cuda.is_available()` is `True`
- `flash_attn` imports successfully
- a small causal attention forward pass runs on the GPU

## Notes for this project

This repository still uses handwritten attention kernels in
`src/layer_depth_attention/model.py` and `src/layer_depth_attention/ablation_models.py`.
So this environment setup only makes FlashAttention available. A separate code
change is still needed if you want the model implementation itself to call
FlashAttention kernels.
