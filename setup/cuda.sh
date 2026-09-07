#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-cuda}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SETUP_LOG_DIR="${PROJECT_ROOT}/output/logs/setup"
SETUP_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SETUP_LOG="${SETUP_LOG_DIR}/cuda-${SETUP_STAMP}-$$.log"
PROFILE_MARKER="${VENV_PATH}/.deepdocforgery-profile"

mkdir -p "${SETUP_LOG_DIR}"
exec > >(tee -a "${SETUP_LOG}") 2>&1
echo "STARTED | CUDA setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: ${SETUP_LOG}"

report_setup_exit() {
  local exit_code=$?
  if [[ "${exit_code}" != "0" ]]; then
    echo "FAILED | CUDA setup | exit=${exit_code} | $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  fi
}
trap report_setup_exit EXIT

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is unavailable; install/repair the NVIDIA driver first." >&2
  exit 1
}
nvidia-smi
GPU_COMPUTE_PROCESSES="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
GPU_BUSY=0
if [[ -n "${GPU_COMPUTE_PROCESSES//[[:space:]]/}" ]]; then
  GPU_BUSY=1
  echo "ATTENTION REQUIRED: another CUDA compute process is active:"
  echo "${GPU_COMPUTE_PROCESSES}"
  echo "Installation will continue, but doctor will block full training until the GPU is free."
fi

if [[ -d "${VENV_PATH}" && ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
  echo "Refusing to clear ${VENV_PATH}: it is not a Python virtual environment." >&2
  exit 2
fi
EXISTING_PROFILE=""
if [[ -f "${PROFILE_MARKER}" ]]; then
  EXISTING_PROFILE="$(<"${PROFILE_MARKER}")"
fi
NEEDS_RESET=0
if [[ "${RECREATE_VENV:-0}" == "1" ]] || \
   [[ -d "${VENV_PATH}" && "${EXISTING_PROFILE}" != "cuda" ]]; then
  NEEDS_RESET=1
fi
if [[ "${NEEDS_RESET}" == "1" && "${VIRTUAL_ENV:-}" == "${VENV_PATH}" ]]; then
  echo "Cannot clear the active environment ${VENV_PATH}; run 'deactivate' first." >&2
  exit 2
fi
if [[ "${NEEDS_RESET}" == "1" ]]; then
  echo "Clearing legacy or mismatched virtual environment: ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv --clear "${VENV_PATH}"
else
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi
if grep -Eq '^include-system-site-packages = true' "${VENV_PATH}/pyvenv.cfg"; then
  echo "Virtual environment unexpectedly exposes system packages: ${VENV_PATH}" >&2
  exit 2
fi

"${VENV_PATH}/bin/python" -m pip install --upgrade pip wheel "setuptools<82"
"${VENV_PATH}/bin/python" -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
CC="${CC:-gcc}" CXX="${CXX:-g++}" \
  "${VENV_PATH}/bin/python" -m pip install -e \
  "${PROJECT_ROOT}[dev,data,exact-jpeg,cuda,download]"
"${VENV_PATH}/bin/python" -m pip check

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  if [[ "${GPU_BUSY}" == "1" ]]; then
    echo "Running the complete CPU-safe suite; CUDA-only tests are skipped while the GPU is busy."
    CUDA_VISIBLE_DEVICES="" "${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"
  else
    "${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"
  fi
fi

"${VENV_PATH}/bin/python" - <<'PY'
import json
import torch

from deepdocforgery.telemetry import configure_compute, resource_snapshot

if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch installed, but torch.cuda.is_available() is false")
properties = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(properties.total_memory / 1024**3, 2))
compute = configure_compute(torch.device("cuda"), cpu_threads="auto")
print(json.dumps({"compute": compute, "resources": resource_snapshot(torch.device("cuda"))}, indent=2))
PY

printf 'cuda\n' > "${PROFILE_MARKER}"
echo "SUCCEEDED | CUDA setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CUDA environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
