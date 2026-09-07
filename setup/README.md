# Setup

Use exactly one installer:

- `bash setup/cpu.sh` creates `.venv-cpu` without CUDA-only dependencies.
- `bash setup/cuda.sh` creates `.venv-cuda`, installs CUDA PyTorch, timm,
  LMDB, and exact-JPEG support, and verifies the GPU.

The CUDA installer defaults to the PyTorch `cu128` wheel index. Override
`PYTORCH_INDEX_URL` if your driver requires another supported wheel channel.
`jpegio` may compile locally, so install `gcc`, `g++`, and Python development
headers if a wheel is unavailable.

Both environments install `tqdm` and `psutil`, which provide terminal progress
and portable CPU/RAM telemetry. CUDA telemetry comes from PyTorch and
`nvidia-smi`, so no separate NVIDIA monitoring Python package is required.

Each installer records its complete terminal output in `output/logs/setup/`,
runs `pip check`, and runs the test suite. A legacy environment without the
DeepDocForgery profile marker is cleared once; subsequent runs reuse it. Set
`RECREATE_VENV=1` to deliberately rebuild a marked environment from scratch.
If that environment is currently active, run `deactivate` before invoking its
installer; an active environment is never cleared underneath the shell.

The CUDA installer reports active compute processes. If the GPU is busy it
runs the CPU-safe tests so setup can finish, but `deepdocforgery doctor
--profile cuda` returns `attention_required` until at least 16 GiB VRAM is free
and the competing process has stopped.
