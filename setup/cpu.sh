#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-cpu}"
PYTORCH_CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install torch --index-url "${PYTORCH_CPU_INDEX_URL}"
"${VENV_PATH}/bin/python" -m pip install -e "${PROJECT_ROOT}[dev,data]"
"${VENV_PATH}/bin/python" -m pytest -q "${PROJECT_ROOT}/tests"

echo "CPU environment ready: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
