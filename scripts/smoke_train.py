"""Run a tiny complete forward/backward/optimizer cycle on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.inputLayer.dataModelling import load_yaml, make_synthetic_batch
from src.model import DeepDocForgeryModel
from src.training.objectives import DeepDocForgeryCriterion, DeepDocForgerySupervision
from src.training.runtime import resolve_device


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
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training = config.get("training", {})
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(int(training.get("cpu_threads", 2)))
    seed = int(training.get("seed", 7))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DeepDocForgeryModel.from_config(config.get("model", {})).to(device).train()
    criterion = DeepDocForgeryCriterion.from_config(config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    steps = int(args.steps if args.steps is not None else training.get("steps", 2))
    if steps < 1:
        raise ValueError("Smoke training requires at least one step")

    tracked = {
        "model": next(model.parameters()),
        "fusion": model.front_end.fusion.attention_heads[
            model.front_end.fusion.level_names[0]
        ].local[-1].weight,
        "spatial": model.front_end.spatial.vit.patch_stem.projection.weight,
        "decoder": model.decoder.segmentation_head[-1].weight,
    }
    initial = {name: value.detach().clone() for name, value in tracked.items()}
    final_record: dict[str, float | int | str] = {}
    for step in range(steps):
        batch = make_synthetic_batch(
            batch_size=int(training.get("batch_size", 2)),
            height=int(training.get("height", 64)),
            width=int(training.get("width", 80)),
            number_of_noise_types=len(
                model.front_end.forensics.degradation_estimator.noise_types
            ),
            maximum_noise_strength=(
                model.front_end.forensics.degradation_estimator.maximum_noise_strength
            ),
            device=device,
            seed=seed + step,
        )
        labels = (batch.tamper_mask.flatten(1).amax(dim=1, keepdim=True) > 0).float()
        supervision = DeepDocForgerySupervision(
            tamper_mask=batch.tamper_mask,
            image_label=labels,
            valid_mask=torch.ones_like(batch.tamper_mask),
            degradation=batch.targets,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.rgb, metadata=batch.metadata)
        losses = criterion(output, supervision)
        total = losses["total"]
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"Non-finite loss encountered at step {step}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"Non-finite gradient encountered at step {step}")
        optimizer.step()
        final_record = {
            "step": step + 1,
            "total_loss": float(total.detach().cpu()),
            "mask_bce_loss": float(losses["main/mask_bce"].detach().cpu()),
            "mask_dice_loss": float(losses["main/mask_dice"].detach().cpu()),
            "image_loss": float(losses["main/image"].detach().cpu()),
            "boundary_loss": float(losses["main/boundary"].detach().cpu()),
            "agreement_loss": float(losses["main/agreement"].detach().cpu()),
            "degradation_loss": float(losses["degradation/total"].detach().cpu()),
            "adn_loss": float(losses["adn/total"].detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "classification_to_localization_strength": float(
                output.decoder.classification_to_localization_strength.detach().cpu()
            ),
            "denoising_strength": float(output.decoder.denoising_strength.detach().cpu()),
        }
        first_level = model.front_end.fusion.level_names[0]
        for branch, weight in output.front_end.fusion.mean_attention()[first_level].items():
            final_record[f"{branch}_attention"] = float(weight.detach().cpu())

    for name, parameter in tracked.items():
        delta = (parameter.detach() - initial[name]).abs().max()
        if not bool(delta > 0):
            raise RuntimeError(f"Optimizer step did not update the {name} parameters")
        final_record[f"maximum_{name}_parameter_delta"] = float(delta.cpu())
    final_record.update(
        {
            "device": str(device),
            "steps": steps,
            "status": "ok",
        }
    )
    print(json.dumps(final_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
