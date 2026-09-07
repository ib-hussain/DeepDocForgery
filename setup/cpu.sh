#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-cpu}"
PYTORCH_CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
SETUP_LOG_DIR="${PROJECT_ROOT}/output/logs/setup"
SETUP_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SETUP_LOG="${SETUP_LOG_DIR}/cpu-${SETUP_STAMP}-$$.log"
PROFILE_MARKER="${VENV_PATH}/.deepdocforgery-profile"

mkdir -p "${SETUP_LOG_DIR}"
exec > >(tee -a "${SETUP_LOG}") 2>&1
echo "STARTED | CPU setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: ${SETUP_LOG}"

report_setup_exit() {
  local exit_code=$?
  if [[ "${exit_code}" != "0" ]]; then
    echo "FAILED | CPU setup | exit=${exit_code} | $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  fi
}
trap report_setup_exit EXIT

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
   [[ -d "${VENV_PATH}" && "${EXISTING_PROFILE}" != "cpu" ]]; then
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
"${VENV_PATH}/bin/python" -m pip install torch --index-url "${PYTORCH_CPU_INDEX_URL}"
"${VENV_PATH}/bin/python" -m pip install -e "${PROJECT_ROOT}[dev,data]"
"${VENV_PATH}/bin/python" -m pip check
"${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"

"${VENV_PATH}/bin/python" - <<'PY'
import json
import torch

from deepdocforgery.telemetry import configure_compute, resource_snapshot

compute = configure_compute(torch.device("cpu"), cpu_threads="auto")
print(json.dumps({"compute": compute, "resources": resource_snapshot()}, indent=2))
PY

printf 'cpu\n' > "${PROFILE_MARKER}"
echo "SUCCEEDED | CPU setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CPU environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
