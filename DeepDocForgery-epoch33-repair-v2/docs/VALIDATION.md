# Executed validation

Environment: Python 3.12, torch 2.11.0+cpu, torchvision 0.26.0+cpu, timm 1.0.30. No CUDA device was available.

- Ruff on all five changed/added Python files: passed.
- Complete repository suite: **104 passed, 3 CUDA tests skipped, 1 environment-related failure**. The failure is the existing subprocess CLI-help test. Its child process reports PID 5 while that PID is absent from the hosted `/proc`; `psutil.Process()` raises NoSuchProcess at telemetry import. Running the exact CLI command directly succeeded. The training code is not modified to conceal this host limitation.
- Focused recovery and existing resume tests: passed on CPU; the CUDA recovery test was skipped. Covered graph release after rejected/successful forwards, BF16 replay of RNG with and without enabled scaling, preservation of accumulated gradients, non-finite-gradient/norm-overflow rejection without Adam/weight mutation, tail flushing after a skipped final batch, strict checkpoint continuation and unchanged YAML/contract.
- The preflight was exercised against a synthetic complete-model checkpoint: four real fixture batches, one in-memory update, and byte-for-byte unchanged checkpoint/config afterwards. Non-finite checkpoint tensors are rejected.
- CPU smoke command: two full model forward/backward/optimiser steps, finite final loss 3.54038143157959, all tracked model/spatial/fusion/decoder parameters updated.
- Installer: dry-run, baseline installation, source-conflict refusal without partial edits, backup correctness, unchanged configuration hash and repeat-install idempotence passed.
- Version-1 to version-2 installer upgrade: passed. Existing repaired files are retained; `metrics.py`, `runtime.py`, and their regression test are added atomically.
- NaN AUROC regression: non-finite scores return `None` and record their count instead of entering the equality-group loop. The live-rescue frame mutation was exercised against a synthetic `binary_auroc` stack and changed only non-finite list entries.
- Full configured Swin model construction with pretrained download disabled: original and repaired models both have 51,474,589 parameters, 668 state-dictionary entries and schema SHA-256 d5c36f909a55a4b948c0337fc014315412162a612d74fed1c81047cf8d1b3491. There are no BatchNorm modules with running statistics. This is a parameter-layout/runtime import check, not a full-data training result.

The exact checkpoint tensors, actual training images, target GPU memory behaviour and full epoch 33 have NOT been tested here. The provided target-machine preflight performs the missing state and short GPU checks. It cannot prove a complete epoch will succeed or identify the historical first NaN from logs that did not record its components.
