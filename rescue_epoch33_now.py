import sys
import gc
import os
import torch
from pathlib import Path

try:
    frames = sys._current_frames()
    target = None

    for frame in frames.values():
        while frame:
            if frame.f_code.co_name == "_train" and \
               frame.f_code.co_filename.endswith("train.py"):
                target = frame
                break
            frame = frame.f_back
        if target:
            break

    if target is None:
        raise RuntimeError("Training frame not found")

    v = target.f_locals
    model = v["model"] if "model" in v else None

    if model is None:
        for obj in gc.get_objects():
            if isinstance(obj, torch.nn.Module) and \
               obj.__class__.__name__ == "DeepDocForgeryModel":
                model = obj
                break

    if model is None:
        raise RuntimeError("Model not found")

    optimizer = v.get("optimizer")
    scheduler = v.get("scheduler")
    epoch = v.get("epoch", 32)

    out = Path("output/model/cuda-full")
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler else None
        ),
        "best_metric": v.get("best_metric"),
        "contract": v.get("contract"),
        "rescue": True,
    }

    path = out / "epoch33_EMERGENCY.pt"
    tmp = out / "epoch33_EMERGENCY.pt.tmp"

    torch.save(checkpoint, tmp)
    os.replace(tmp, path)

    with open(out / "epoch33_RESCUE_SUCCESS.txt", "w") as f:
        f.write(f"Saved epoch {epoch + 1}\n")
        f.write(f"Checkpoint: {path}\n")

except Exception:
    import traceback
    with open(
        "output/model/cuda-full/epoch33_RESCUE_ERROR.txt", "w"
    ) as f:
        traceback.print_exc(file=f)

