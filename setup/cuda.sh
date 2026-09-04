#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-cuda}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is unavailable; install/repair the NVIDIA driver first." >&2
  exit 1
}
nvidia-smi

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
CC="${CC:-gcc}" CXX="${CXX:-g++}" \
  "${VENV_PATH}/bin/python" -m pip install -e \
  "${PROJECT_ROOT}[dev,data,exact-jpeg,cuda,download]"

"${VENV_PATH}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch installed, but torch.cuda.is_available() is false")
properties = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(properties.total_memory / 1024**3, 2))
PY

echo "CUDA environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
