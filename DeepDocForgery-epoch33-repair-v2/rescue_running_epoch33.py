"""One-time live rescue for PID 9090's NaN AUROC loop; execute only through GDB."""

from __future__ import annotations

import math
import sys


frame = sys._getframe()
while frame is not None and frame.f_code.co_name != "binary_auroc":
    frame = frame.f_back
if frame is None:
    raise RuntimeError("binary_auroc frame was not found; do not modify this process")

scores = frame.f_locals.get("scores")
if not isinstance(scores, list):
    raise RuntimeError("binary_auroc scores list was not found; do not modify this process")
non_finite = sum(not math.isfinite(float(score)) for score in scores)
if non_finite < 1:
    raise RuntimeError("No non-finite AUROC score was found; this is not the expected loop")

# The historical validation already consumed these predictions. A neutral
# finite placeholder only lets the old AUROC loop terminate so the trained
# weights can be checkpointed. Epoch 33's image AUROC is invalid and must not
# be used scientifically; v2 returns None instead and retries future validation
# batches in BF16 before metrics consume them.
scores[:] = [float(score) if math.isfinite(float(score)) else 0.5 for score in scores]
print(f"EPOCH33_RESCUED_NONFINITE_AUROC_SCORES={non_finite}", flush=True)
