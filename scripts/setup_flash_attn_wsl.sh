#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-flashattn-py310}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
MAX_JOBS="${MAX_JOBS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[info] project root: ${PROJECT_ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[error] this script is intended for WSL2/Linux."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found. GPU passthrough is not ready inside WSL2."
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat <<'EOF'
[error] nvcc was not found.

flash-attn needs a real CUDA toolkit during build. Install a CUDA 12.x toolkit
inside WSL2 first, then rerun this script.

Quick check:
  nvcc --version
EOF
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  cat <<'EOF'
[error] conda was not found in PATH.

Please install Miniforge or Miniconda inside WSL2, then rerun this script.
EOF
  exit 3
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[info] reusing existing conda env: ${ENV_NAME}"
else
  echo "[info] creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
  torch \
  torchvision \
  torchaudio \
  --index-url "${TORCH_INDEX_URL}"

python -m pip install packaging psutil ninja wheel setuptools

echo "[info] torch build:"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY

echo "[info] installing flash-attn"
MAX_JOBS="${MAX_JOBS}" python -m pip install flash-attn --no-build-isolation

if ! grep -q "layer_depth_attention" "${HOME}/.bashrc" 2>/dev/null; then
  echo "" >> "${HOME}/.bashrc"
  echo "# Layer-Depth-Attention" >> "${HOME}/.bashrc"
  echo "export PYTHONPATH=\"${PROJECT_ROOT}/src:\${PYTHONPATH}\"" >> "${HOME}/.bashrc"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "[info] PYTHONPATH configured with ${PROJECT_ROOT}/src"

echo "[info] verifying flash-attn"
python "${PROJECT_ROOT}/scripts/verify_flash_attn.py"

echo "[done] environment ${ENV_NAME} is ready"
