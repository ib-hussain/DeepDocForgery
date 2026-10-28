# DeepDocForgery epoch33 continuation kit (2026-10-28)

The kit expects the exact checksums of the native epoch32 `last.pt` and 590MiB epoch33 emergency rescue shown in the Oct 28 audit. **It fails rather than guessing if either file differs.**

1. Copy the ZIP to `~/deepFakeDetection/DeepDocForgery` and unzip it.
2. Mount the existing data partition, if necessary: `sudo mount /dev/nvme0n1p5 output/processed/cuda`.
3. `bash DeepDocForgery_epoch33_resume_fix/prepare.sh`.
4. Only if READY appears: `bash DeepDocForgery_epoch33_resume_fix/launch.sh`.

`prepare.sh` verifies the original checkpoint bytes, checks the recovered model and optimizer states for non-finite tensors, validates the scheduler, constructs a native zero-based epoch=32 checkpoint with inferred global step 81312, patches the actual AUROC source, adds a pre-validation checkpoint after each epoch's optimizer updates, runs 8 AUROC tests and invokes the existing GPU preflight. It does NOT overwrite `last.pt` or the original rescue file.

The converter reconstructs the post-epoch33 global step based on 9,853 training batches and 4 microbatches per accumulation, resetting random-number generators reproducibly and inheriting the epoch32 AMP scaler (neither was present in the emergency file). This is a recovery, not a bit-exact resumption. Accuracy and training stability still need attention. The source patch backs up `metrics.py` and `train.py` before editing. The repo's original `atomic_torch_save` function is reused: metrics failure will not prevent an epoch's *pre-validation* state from being saved.

If `prepare.sh` stops or preflight fails, do **not** run `launch.sh`. Read the terminal error and report it. If the training actually launches, confirm the log says `Resumed after epoch 33` and `Epoch 34/50 started`. If it instead starts epoch33, interrupt immediately and retain the backups.

**Review caveat:** Source changes are applied automatically only if the user's training/metric functions still match the source fragments seen in the Oct 28 logs. There is no live access to the GPU machine to test its exact full code, so the preflight and tests are mandatory.

Rollback source changes if needed:
```
cp output/recovery/epoch33-source-before-resume-fix/metrics.py deepdocforgery/metrics.py
cp output/recovery/epoch33-source-before-resume-fix/train.py deepdocforgery/train.py
```

Do not run training with a patched metric function unless its unit tests pass and the native checkpoint preflight succeeds.
