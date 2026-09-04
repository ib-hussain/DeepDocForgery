# Setup

Use exactly one installer:

- `bash setup/cpu.sh` creates `.venv-cpu` without CUDA-only dependencies.
- `bash setup/cuda.sh` creates `.venv-cuda`, installs CUDA PyTorch, timm,
  LMDB, and exact-JPEG support, and verifies the GPU.

The CUDA installer defaults to the PyTorch `cu128` wheel index. Override
`PYTORCH_INDEX_URL` if your driver requires another supported wheel channel.
`jpegio` may compile locally, so install `gcc`, `g++`, and Python development
headers if a wheel is unavailable.
