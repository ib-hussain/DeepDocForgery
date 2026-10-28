#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepFakeDetection/DeepDocForgery"
source .venv-cuda/bin/activate
if ! test -f output/recovery/EPOCH33_READY.txt; then
  echo 'STOP: prepare.sh did not complete successfully.' >&2
  exit 1
fi
if ! mountpoint -q output/processed/cuda; then
  echo 'STOP: dataset mount output/processed/cuda is missing.' >&2
  exit 1
fi
if ! test -s output/model/cuda-full/epoch33_RESUMABLE.pt; then
  echo 'STOP: converted checkpoint missing.' >&2
  exit 1
fi
mkdir -p output/recovery
printf '\n=== TRAINING FROM COMPLETED EPOCH 33; NEXT EPOCH MUST BE 34/50 ===\n'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m deepdocforgery train \
  --config configs/cuda/HPO_RESULTS.yaml \
  --device cuda \
  --output output/model/cuda-full \
  --resume output/model/cuda-full/epoch33_RESUMABLE.pt \
  2>&1 | tee -a output/recovery/epoch34-launch.log
