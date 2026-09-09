#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-cpu}"
PYTORCH_CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
SETUP_LOG_DIR="${PROJECT_ROOT}/output/logs/setup"
SETUP_STATE_DIR="${PROJECT_ROOT}/output/state/setup"
SETUP_STATE="${SETUP_STATE_DIR}/cpu.json"
SETUP_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SETUP_LOG="${SETUP_LOG_DIR}/cpu-${SETUP_STAMP}-$$.log"
PROFILE_MARKER="${VENV_PATH}/.deepdocforgery-profile"

# ROS and user-site paths can inject unrelated pytest plugins even inside a
# normal venv.  This project setup is deliberately isolated from that ambient
# Python state.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

mkdir -p "${SETUP_LOG_DIR}" "${SETUP_STATE_DIR}"
exec > >(tee -a "${SETUP_LOG}") 2>&1
echo "STARTED | CPU setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Log: ${SETUP_LOG}"

COMPLETED_STEP="started"

write_setup_state() {
  local status="$1"
  local step="$2"
  local error="${3:-}"
  "${PYTHON_BIN}" - "${SETUP_STATE}" "cpu" "${status}" "${step}" "${SETUP_LOG}" "${VENV_PATH}" "${error}" <<'PY'
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
    "stage": "setup/cpu",
    "contract": profile,
    "status": status,
    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "completed_step": step,
    "environment": str(pathlib.Path(venv).resolve()),
    "text_log": str(pathlib.Path(log).resolve()),
    "resume_command": "bash setup/cpu.sh",
}
if error:
    payload["error"] = error
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

write_setup_state "running" "${COMPLETED_STEP}"

pip_retry() {
  local attempts="${PIP_INSTALL_ATTEMPTS:-3}"
  local delay_seconds="${PIP_RETRY_DELAY_SECONDS:-2}"
  local attempt
  if ! [[ "${attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PIP_INSTALL_ATTEMPTS must be a positive integer, got: ${attempts}" >&2
    return 2
  fi
  for ((attempt=1; attempt<=attempts; attempt++)); do
    if "$@"; then
      return 0
    fi
    if (( attempt == attempts )); then
      echo "pip command failed after ${attempts} attempt(s): $*" >&2
      return 1
    fi
    echo "pip command failed (attempt ${attempt}/${attempts}); retrying..." >&2
    sleep "${delay_seconds}"
  done
}

report_setup_exit() {
  local exit_code=$?
  if [[ "${exit_code}" != "0" ]]; then
    echo "FAILED | CPU setup | exit=${exit_code} | $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    set +e
    write_setup_state "failed" "${COMPLETED_STEP}" "setup exited with code ${exit_code}"
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

# Write the marker before downloads/tests.  A failed setup can then be resumed
# without clearing and redownloading an otherwise valid partial environment.
printf 'cpu\n' > "${PROFILE_MARKER}"
COMPLETED_STEP="environment_created"
write_setup_state "running" "${COMPLETED_STEP}"

pip_retry "${VENV_PATH}/bin/python" -m pip install --upgrade pip wheel "setuptools<82"
pip_retry "${VENV_PATH}/bin/python" -m pip install torch --index-url "${PYTORCH_CPU_INDEX_URL}"
pip_retry "${VENV_PATH}/bin/python" -m pip install -e "${PROJECT_ROOT}[dev,data]"
COMPLETED_STEP="dependencies_installed"
write_setup_state "running" "${COMPLETED_STEP}"
"${VENV_PATH}/bin/python" -m pip check
"${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"
COMPLETED_STEP="tests_passed"
write_setup_state "running" "${COMPLETED_STEP}"

"${VENV_PATH}/bin/python" - <<'PY'
import json
import torch

from deepdocforgery.telemetry import configure_compute, resource_snapshot

compute = configure_compute(torch.device("cpu"), cpu_threads="auto")
print(json.dumps({"compute": compute, "resources": resource_snapshot()}, indent=2))
PY

COMPLETED_STEP="runtime_verified"
write_setup_state "running" "${COMPLETED_STEP}"
COMPLETED_STEP="completed"
write_setup_state "completed" "${COMPLETED_STEP}"
echo "SUCCEEDED | CPU setup | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CPU environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
