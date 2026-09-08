#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-cuda}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SETUP_LOG_DIR="${PROJECT_ROOT}/output/logs/setup"
SETUP_STATE_DIR="${PROJECT_ROOT}/output/state/setup"
SETUP_STATE="${SETUP_STATE_DIR}/cuda.json"
SETUP_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SETUP_LOG="${SETUP_LOG_DIR}/cuda-${SETUP_STAMP}-$$.log"
PROFILE_MARKER="${VENV_PATH}/.deepdocforgery-profile"

# Prevent a sourced ROS installation or user-site package from auto-loading
# unrelated pytest plugins into this isolated training environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

mkdir -p "${SETUP_LOG_DIR}" "${SETUP_STATE_DIR}"
exec > >(tee -a "${SETUP_LOG}") 2>&1
echo "STARTED | CUDA setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: ${SETUP_LOG}"

COMPLETED_STEP="started"

write_setup_state() {
  local status="$1"
  local step="$2"
  local error="${3:-}"
  "${PYTHON_BIN}" - "${SETUP_STATE}" "cuda" "${status}" "${step}" "${SETUP_LOG}" "${VENV_PATH}" "${error}" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

path, profile, status, step, log, venv, error = sys.argv[1:]
target = pathlib.Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "format_version": 1,
    "stage": "setup/cuda",
    "contract": profile,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "completed_step": step,
    "environment": str(pathlib.Path(venv).resolve()),
    "text_log": str(pathlib.Path(log).resolve()),
    "resume_command": "bash setup/cuda.sh",
}
if error:
    payload["error"] = error
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

write_setup_state "running" "${COMPLETED_STEP}"

report_setup_exit() {
  local exit_code=$?
  if [[ "${exit_code}" != "0" ]]; then
    echo "FAILED | CUDA setup | exit=${exit_code} | $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    set +e
    write_setup_state "failed" "${COMPLETED_STEP}" "setup exited with code ${exit_code}"
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

# Mark the environment before expensive CUDA wheel downloads.  If a later
# validation step fails, rerunning this script reuses the partial environment.
printf 'cuda\n' > "${PROFILE_MARKER}"
COMPLETED_STEP="environment_created"
write_setup_state "running" "${COMPLETED_STEP}"

"${VENV_PATH}/bin/python" -m pip install --upgrade pip wheel "setuptools<82"
"${VENV_PATH}/bin/python" -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
CC="${CC:-gcc}" CXX="${CXX:-g++}" \
  "${VENV_PATH}/bin/python" -m pip install -e \
  "${PROJECT_ROOT}[dev,data,exact-jpeg,cuda,download]"
COMPLETED_STEP="dependencies_installed"
write_setup_state "running" "${COMPLETED_STEP}"
"${VENV_PATH}/bin/python" -m pip check

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  if [[ "${GPU_BUSY}" == "1" ]]; then
    echo "Running the complete CPU-safe suite; CUDA-only tests are skipped while the GPU is busy."
    CUDA_VISIBLE_DEVICES="" "${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"
  else
    "${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"
  fi
  COMPLETED_STEP="tests_passed"
else
  COMPLETED_STEP="tests_skipped"
fi
write_setup_state "running" "${COMPLETED_STEP}"

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

COMPLETED_STEP="runtime_verified"
write_setup_state "running" "${COMPLETED_STEP}"
COMPLETED_STEP="completed"
write_setup_state "completed" "${COMPLETED_STEP}"
echo "SUCCEEDED | CUDA setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CUDA environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
