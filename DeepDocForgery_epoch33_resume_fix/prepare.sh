#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/deepFakeDetection/DeepDocForgery"
source .venv-cuda/bin/activate

if ! mountpoint -q output/processed/cuda; then
    echo 'STOP: output/processed/cuda is not mounted. Re-mount /dev/nvme0n1p5 before continuing.' >&2
    exit 1
fi

mkdir -p output/recovery/epoch33-backup
cp --update=none output/model/cuda-full/epoch33_EMERGENCY.pt output/recovery/epoch33-backup/
cp --update=none output/model/cuda-full/last.pt output/recovery/epoch33-backup/last-epoch32.pt
cp --update=none output/model/cuda-full/best.pt output/recovery/epoch33-backup/best-before-resume.pt

SCRIPT="DeepDocForgery_epoch33_resume_fix"
echo '=== 1. Convert epoch33 rescue to native checkpoint ==='
python "$SCRIPT/convert_epoch33.py"
echo '=== 2. Patch AUROC and pre-validation checkpoint saving ==='
python "$SCRIPT/patch_training.py"
echo '=== 3. Syntax and AUROC regression tests ==='
python -m py_compile deepdocforgery/metrics.py deepdocforgery/train.py
python "$SCRIPT/test_fix.py" -v
echo '=== 4. Native GPU checkpoint preflight for 33 completed epochs ==='
python -m deepdocforgery.epoch33_preflight \
  --config configs/cuda/HPO_RESULTS.yaml \
  --checkpoint output/model/cuda-full/epoch33_RESUMABLE.pt \
  --device cuda \
  --expected-completed-epochs 33
printf 'ready\n' > output/recovery/EPOCH33_READY.txt
echo 'READY: epoch 33 native checkpoint passed preflight. Run launch.sh next.'
