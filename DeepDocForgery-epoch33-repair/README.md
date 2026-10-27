# DeepDocForgery epoch-33 repair — v1

Full replacement files, a checked installer, regression tests, a checkpoint/GPU preflight, a reviewable diff and a complete run-history report are included. The installer changes three existing modules and adds two files. It does not install configurations, reset optimiser state, edit manifests or touch checkpoint tensors.

Read docs/DIAGNOSIS.md for the reasoning and limits. No GPU completion is claimed. Run these commands in the existing university checkout, one line at a time, and continue only when each command succeeds.

## 1. Prepare the existing checkout

Download or copy `DeepDocForgery-epoch33-repair-v1.zip` to the university Linux user's Downloads directory (not the Windows Downloads directory). Stop any currently running training command normally before installation; do not terminate desktop GPU processes. Use the existing environment:

```bash
cd /home/fastlab/deepFakeDetection/DeepDocForgery
source .venv-cuda/bin/activate
python -c 'import sys,torch; print(sys.executable); print(torch.__version__); print("CUDA:",torch.cuda.is_available()); assert torch.cuda.is_available()'
nvidia-smi
```

The normal Ubuntu desktop allocation can remain. Another large training process should not be using the GPU. Do not run the old performance script or reinstall PyTorch for this fix.

Check that the processed volume is available in its original location. Mount it using your established correct device/mount procedure if necessary; a device name is not guessed here. A missing mount caused three earlier starts to fail.

```bash
findmnt -T output/processed/cuda
test -r output/processed/cuda/doctamper/DocTamperV1-TrainingSet/images/000000000.jpg && test -r output/processed/cuda/doctamper/DocTamperV1-TrainingSet/masks/000000000.png && echo 'Processed sample is available'
test -s output/model/cuda-full/last.pt && test -s output/manifests/cuda.jsonl && echo 'Checkpoint and manifest are available'
```

The sample test is only an early mount check; the doctor below audits all referenced file paths. Do not regenerate preparation output to work around a missing volume.

## 2. Back up and install

```bash
cp --reflink=auto --no-clobber output/model/cuda-full/last.pt output/model/cuda-full/last-before-epoch33-repair.pt
unzip -n "$HOME/Downloads/DeepDocForgery-epoch33-repair-v1.zip" -d "$HOME/Downloads"
python "$HOME/Downloads/DeepDocForgery-epoch33-repair/apply_repair.py" --repo . --check
python "$HOME/Downloads/DeepDocForgery-epoch33-repair/apply_repair.py" --repo .
```

Expected installer result: `status: installed` and `config_unchanged: true`. It saves the original source under `output/recovery/epoch33-code-<UTC timestamp>/` and verifies every replacement against its expected original hash. A second invocation reports `already_installed`. If it reports local edits, it changes nothing: use repair.diff to merge those edits rather than forcing replacement. The checkpoint backup command never overwrites a prior backup; verify its date/content if one already existed.

## 3. Check the actual saved state and data

```bash
python -m deepdocforgery.epoch33_preflight --config configs/cuda/HPO_RESULTS.yaml --checkpoint output/model/cuda-full/last.pt --device cuda --expected-completed-epochs 32
python -m deepdocforgery doctor --profile cuda --config configs/cuda/HPO_RESULTS.yaml --no-progress
```

The first result must show `status: ok`, `completed_epochs: 32`, `next_epoch: 33`, `global_step: 78848`, and `config_and_checkpoint_unchanged: true`. It scans all model/optimiser tensors for non-finite values on CPU and checks the saved schedule/contract. The saved group learning rates should be approximately 2.9423925067528905e-05 and 7.890648501219129e-06. Those decayed rates are restored; the run does not restart the 0.0001 learning rate.

The doctor must complete with no blocking issue. It checks file existence and protocol, not a full decode of every image. A corrupted PNG may still require restoring that exact file from its original dataset if it recurs. The preflight deliberately rejects a different completed epoch: if training has since advanced, inspect that checkpoint and explicitly supply its actual completed-epoch count instead of rolling back silently.

If pytest is already installed, also run the focused regression suite in your environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_epoch33_recovery.py tests/test_resume_v051.py
```

## 4. Probe the original GPU workload and rejected-forward path

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m deepdocforgery.epoch33_preflight --config configs/cuda/HPO_RESULTS.yaml --checkpoint output/model/cuda-full/last.pt --device cuda --expected-completed-epochs 32 --probe-batches 8 --inject-rejected-forward
```

This loads the real saved model/Adam/scaler/RNG state, uses the configured batch 11, 512×512 images and accumulation 4, deliberately rejects one forward, and attempts the same batch in BF16. It then processes eight real training batches. The final result must be `status: ok`, `probe_batches: 8`, `in_memory_updates_discarded: 2`, `precision_retries` at least 1 and `config_and_checkpoint_unchanged: true`. It checks both original file hashes after the probe. These two optimiser updates exist only in this temporary process and are discarded; the saved epoch-32 run is untouched. Do not run the probe and training simultaneously.

The RTX 4000 Ada is expected to support BF16; the actual runtime capability check decides. If this probe fails, retain its complete traceback and stop before the long run. It tests startup/recovery memory at the real dimensions; it cannot reproduce the unknown bad batch thousands of iterations later or guarantee a whole epoch.

## 5. Resume the main run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m deepdocforgery train --config configs/cuda/HPO_RESULTS.yaml --device cuda --output output/model/cuda-full --resume output/model/cuda-full/last.pt
```

The explicit checkpoint path makes a missing last.pt an error, rather than allowing an accidental fresh run. Do not add `--fresh`, `--epochs`, `--maximum-train-batches` or `--maximum-val-batches`. Keep HPO_RESULTS.yaml, the existing manifest and the repository path unchanged. Do not use best.pt for continuation.

Confirm the startup messages say `batch=11`, `accumulation=4`, `effective batch=44`, `AMP=True`, `Resumed after epoch 32/50`, and `Epoch 33/50 started`. A precision-retry warning is expected only if an actual FP16 loss becomes non-finite. A recovered batch continues; a failed retry records component losses and sample identities. Consecutive-failure guards stop persistent instability rather than saving a corrupt update.

Leave the run through all 9,853 training batches and validation. Success is `Epoch 33/50 complete` plus `Last checkpoint saved`, followed by an epoch-33 metrics row and checkpoint epoch index 32. A progress line alone does not establish a completed epoch. The schedule then continues towards 50 epochs. Training remains checkpointed at epoch boundaries; interrupting mid-epoch restarts that incomplete epoch from the last saved one.

## If an error remains

Do not change YAML or delete last.pt. Keep the text and JSONL files named by the new run. They now contain `original_non_finite_components`, `original_losses`, `non_finite_components`, `losses`, sample IDs, scale changes and recovery counters. The exact error plus these fields distinguishes unrecovered numerical divergence, data decode failure and genuine VRAM pressure. The old logs did not provide this distinction.

If you need to undo this code repair, stop training and copy the three original modules from the installer-reported backup back into deepdocforgery. Remove the two newly added files only if desired. Do not restore an old checkpoint over new completed training unless you intentionally choose that rollback. Original YAML and data remain unchanged throughout.

## Contents

- `payload/deepdocforgery/train.py`: graph lifetime, bounded BF16 retry, safe gradient update, accumulation tail, CPU checkpoint loading and diagnostics.
- `payload/deepdocforgery/spatial.py` and `decoder.py`: small FP32 reduction guards with unchanged parameter layout.
- `payload/deepdocforgery/epoch33_preflight.py`: saved-state validation and optional GPU probe.
- `payload/tests/test_epoch33_recovery.py`: focused regression tests.
- `repair.diff`, `repair-manifest.json`: review and source-integrity checks.
- `docs/DIAGNOSIS.md`, `ALL_TRAINING_RUNS.md`, `COMMIT_HISTORY.md`, `FILE_INVENTORY.md`, `VALIDATION.md`: evidence, coverage and executed-test limits.

The bundle excludes datasets, model checkpoints, original logs, virtual environments, caches and replacement YAML files.
