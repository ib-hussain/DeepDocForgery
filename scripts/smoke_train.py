"""Run a tiny forward/backward/optimizer cycle on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inputLayer.dataModelling import load_yaml, make_synthetic_batch
from src.inputLayer.degradationEstimator import (
    InputForensicsFrontEnd,
    MultiScaleDegradationLoss,
)


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke_train.yaml"),
        help="YAML configuration path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit CUDA device such as cuda:0",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override configured step count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training = config.get("training", {})
    device = _device_from_argument(args.device)
    if device.type == "cpu":
        torch.set_num_threads(int(training.get("cpu_threads", 2)))

    seed = int(training.get("seed", 7))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = InputForensicsFrontEnd.from_config(config.get("model", {})).to(device)
    criterion = MultiScaleDegradationLoss.from_config(config.get("loss", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    steps = int(args.steps if args.steps is not None else training.get("steps", 2))
    if steps < 1:
        raise ValueError("Smoke training requires at least one step")

    first_parameter = next(model.parameters())
    initial_parameter = first_parameter.detach().clone()
    final_record: dict[str, float | int | str] = {}
    for step in range(steps):
        batch = make_synthetic_batch(
            batch_size=int(training.get("batch_size", 2)),
            height=int(training.get("height", 64)),
            width=int(training.get("width", 80)),
            number_of_noise_types=len(model.degradation_estimator.noise_types),
            maximum_noise_strength=model.degradation_estimator.maximum_noise_strength,
            device=device,
            seed=seed + step,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.rgb, metadata=batch.metadata)
        losses = criterion(output.degradation, batch.targets)
        total = losses["total"] + output.dct.consistency_loss
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss encountered at step {step}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient encountered at step {step}")
        optimizer.step()
        final_record = {
            "step": step + 1,
            "total_loss": float(total.detach().cpu()),
            "quality_loss": float(losses["quality"].detach().cpu()),
            "double_compression_loss": float(losses["double_compression"].detach().cpu()),
            "noise_type_loss": float(losses["noise_type"].detach().cpu()),
            "noise_strength_loss": float(losses["noise_strength"].detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

    parameter_delta = (first_parameter.detach() - initial_parameter).abs().max()
    if not bool(parameter_delta > 0):
        raise RuntimeError("Optimizer step did not update the first model parameter")
    final_record.update(
        {
            "device": str(device),
            "steps": steps,
            "maximum_parameter_delta": float(parameter_delta.cpu()),
            "status": "ok",
        }
    )
    print(json.dumps(final_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
