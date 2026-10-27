"""Check a saved run and optionally probe its batches without saving updates."""

from __future__ import annotations

import argparse
import math
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from deepdocforgery.io import load_yaml
from deepdocforgery.model import DeepDocForgeryModel
from deepdocforgery.objectives import DeepDocForgeryCriterion
from deepdocforgery.runtime import (
    CosineEpochScheduler,
    _as_byte_tensor,
    create_dataloader,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from deepdocforgery.state import StageLock, sha256_file
from deepdocforgery.telemetry import configure_compute, print_result, resource_snapshot
from deepdocforgery.train import (
    _batch_with_precision_recovery,
    _optimizer_step,
    _training_contract,
)


def check_tensors(value: Any, path: str) -> int:
    """Check one tensor at a time on CPU, without making another state copy."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"Non-finite checkpoint tensor: {path}")
        return 1
    if isinstance(value, dict):
        return sum(check_tensors(item, f"{path}.{key}") for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(check_tensors(item, f"{path}[{i}]") for i, item in enumerate(value))
    return 0


def verify(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    config_hash = sha256_file(args.config)
    checkpoint_hash = sha256_file(args.checkpoint)
    # Only load a trusted checkpoint produced by this repository.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = ("model", "optimizer", "scheduler", "scaler", "rng_state",
                "data_generator_state", "epoch", "global_step", "contract")
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint is missing full resume state; do not start a fresh run")
    completed = int(checkpoint["epoch"]) + 1
    if completed != args.expected_completed_epochs:
        raise ValueError(f"Expected {args.expected_completed_epochs} completed epochs; found {completed}")
    if checkpoint["contract"] != _training_contract(config, device=args.device):
        raise ValueError("Checkpoint/config/manifest contract differs; keep the original YAML, manifest and path")
    training = config["training"]
    if int(checkpoint["scheduler"]["total_epochs"]) != int(training["epochs"]):
        raise ValueError("Saved scheduler total does not match the unchanged YAML")
    if int(checkpoint["scheduler"]["last_epoch"]) != completed:
        raise ValueError("Saved scheduler epoch does not match the checkpoint epoch")
    if checkpoint["rng_state"] is None or checkpoint["data_generator_state"] is None:
        raise ValueError("The original RNG or data generator state is missing")
    tensors = check_tensors(checkpoint["model"], "model")
    tensors += check_tensors(checkpoint["optimizer"], "optimizer")
    report: dict[str, Any] = {
        "status": "ok", "completed_epochs": completed, "next_epoch": completed + 1,
        "global_step": checkpoint["global_step"], "finite_state_tensors": tensors,
        "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
        "physical_batch_size": config["data"]["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "saved_learning_rates": [group["lr"] for group in checkpoint["optimizer"]["param_groups"]],
        "probe_batches": 0,
    }
    if args.probe_batches:
        report.update(probe(config, checkpoint, args))
    if sha256_file(args.config) != config_hash or sha256_file(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Config/checkpoint changed during the check; stop before resuming")
    report["config_and_checkpoint_unchanged"] = True
    return report


def probe(config: dict, checkpoint: dict, args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    training = config["training"]
    configure_compute(device, cpu_threads=training.get("cpu_threads", "auto"))
    seed_everything(int(training.get("seed", 7)), deterministic=bool(training.get("deterministic", False)))
    loader = create_dataloader(config, split="train", shuffle=True)
    model = DeepDocForgeryModel.from_config(config["model"], load_pretrained=False).to(device).train()
    criterion = DeepDocForgeryCriterion.from_config(config.get("loss", {})).to(device)
    backbone = list(model.front_end.spatial.vit.parameters())
    backbone_ids = {id(p) for p in backbone}
    lr = float(training.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": lr, "name": "main"},
        {"params": backbone, "lr": lr * float(training.get("backbone_learning_rate_multiplier", 1.0)), "name": "spatial_backbone"},
    ], weight_decay=float(training.get("weight_decay", 1e-4)))
    scheduler = CosineEpochScheduler(optimizer, total_epochs=int(training["epochs"]),
                                    minimum_learning_rate=float(training.get("minimum_learning_rate", 1e-6)))
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    loader.generator.set_state(_as_byte_tensor(checkpoint["data_generator_state"]))
    # verify() no longer needs tensors; release the CPU copy before the probe.
    checkpoint.clear()
    allow_retry = False
    if use_amp:
        with torch.cuda.device(device):
            allow_retry = torch.cuda.is_bf16_supported()
            torch.cuda.reset_peak_memory_stats(device)
    accumulation = int(training["gradient_accumulation_steps"])
    if args.probe_batches < accumulation or args.probe_batches % accumulation:
        raise ValueError("--probe-batches must be a positive multiple of the configured accumulation")
    if args.probe_batches > len(loader):
        raise ValueError("The manifest does not contain that many probe batches")
    if args.inject_rejected_forward and not allow_retry:
        raise ValueError("The injected recovery probe requires CUDA BF16 support")

    class RejectOnce:
        def __init__(self):
            self.pending = args.inject_rejected_forward

        def __call__(self, output, supervision):
            losses = criterion(output, supervision)
            if self.pending:
                self.pending = False
                losses["total"] = losses["total"] * float("nan")
            return losses

    checked_criterion = RejectOnce()
    updates = 0
    retries = 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(islice(loader, args.probe_batches), start=1):
        batch = batch.to(device, non_blocking=True)
        values, _, bad, first = _batch_with_precision_recovery(
            model, checked_criterion, batch, device=device, use_amp=use_amp,
            scaler=scaler, accumulation_steps=accumulation, allow_bf16_retry=allow_retry,
        )
        retries += int(first is not None)
        if bad:
            raise RuntimeError(f"Probe batch {i}: non-finite components {bad}; losses={values}")
        del batch
        if i % accumulation == 0:
            norm, updated = _optimizer_step(model, optimizer, scaler, float(training.get("gradient_clip_norm", 5.0)))
            if not updated or not math.isfinite(norm):
                raise RuntimeError(f"Probe batch {i}: non-finite gradient norm {norm}")
            updates += 1
        print_result({"probe_batch": i, "total_loss": values["total"], "in_memory_updates": updates})
    if args.inject_rejected_forward and retries < 1:
        raise RuntimeError("The requested rejected-forward recovery was not exercised")
    return {"probe_batches": args.probe_batches, "in_memory_updates_discarded": updates,
            "precision_retries": retries, "injected_rejected_forward": args.inject_rejected_forward,
            "resources": resource_snapshot(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cuda/HPO_RESULTS.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("output/model/cuda-full/last.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-completed-epochs", type=int, default=32)
    parser.add_argument("--probe-batches", type=int, default=0)
    parser.add_argument("--inject-rejected-forward", action="store_true")
    args = parser.parse_args()
    if args.probe_batches < 0 or (args.inject_rejected_forward and not args.probe_batches):
        parser.error("Choose positive --probe-batches when injecting a rejected forward")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}; do not start a fresh run")
    with StageLock(args.checkpoint.parent / ".train.lock", stage="train"):
        print_result(verify(args))


if __name__ == "__main__":
    main()
