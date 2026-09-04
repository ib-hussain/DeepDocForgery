"""Train DeepDocForgery end to end from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from deepdocforgery.io import load_yaml
from deepdocforgery.model import DeepDocForgeryModel
from deepdocforgery.objectives import DeepDocForgeryCriterion
from deepdocforgery.runtime import (
    CosineEpochScheduler,
    atomic_torch_save,
    create_dataloader,
    evaluate_model,
    load_checkpoint,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cpu/sample.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override data.manifest without editing the YAML file",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--maximum-train-batches",
        type=int,
        default=None,
        help="Debug override; omit for complete epochs",
    )
    parser.add_argument(
        "--maximum-val-batches",
        type=int,
        default=None,
        help="Debug override; omit for complete validation",
    )
    return parser.parse_args()


def _checkpoint_payload(
    *,
    model: DeepDocForgeryModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    validation: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = str(args.manifest.resolve())
    training = config.setdefault("training", {})
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        training["epochs"] = args.epochs
    if args.output is not None:
        training["output_dir"] = str(args.output.resolve())
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(int(training.get("cpu_threads", 2)))
    seed = int(training.get("seed", 7))
    seed_everything(seed, deterministic=bool(training.get("deterministic", False)))

    output_dir = (
        args.output
        if args.output is not None
        else Path(training.get("output_dir", "output/model/deepdocforgery"))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    train_loader = create_dataloader(config, split="train", shuffle=True)
    val_loader = create_dataloader(config, split="val", shuffle=False)
    train_labels = {
        record.label for record in train_loader.dataset.records if record.classification_supervised
    }
    if (
        train_labels
        and len(train_labels) < 2
        and not bool(training.get("allow_single_class", False))
    ):
        raise ValueError(
            "Training split contains only one image class. Add authentic and forged samples "
            "or set training.allow_single_class=true for a deliberate diagnostic run."
        )

    model = DeepDocForgeryModel.from_config(
        config.get("model", {}), load_pretrained=args.resume is None
    ).to(device)
    criterion = DeepDocForgeryCriterion.from_config(config.get("loss", {})).to(device)
    learning_rate = float(training.get("learning_rate", 3e-4))
    backbone_multiplier = float(training.get("backbone_learning_rate_multiplier", 1.0))
    if backbone_multiplier <= 0:
        raise ValueError("backbone_learning_rate_multiplier must be positive")
    backbone_parameters = list(model.front_end.spatial.vit.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [p for p in model.parameters() if id(p) not in backbone_ids]
    parameter_groups = [
        {"params": other_parameters, "lr": learning_rate, "name": "main"},
        {
            "params": backbone_parameters,
            "lr": learning_rate * backbone_multiplier,
            "name": "spatial_backbone",
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(training.get("weight_decay", 1e-4))
    )
    epochs = int(training.get("epochs", 20))
    scheduler = CosineEpochScheduler(
        optimizer,
        total_epochs=max(1, epochs),
        minimum_learning_rate=float(training.get("minimum_learning_rate", 1e-6)),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))

    metric_name = str(training.get("selection_metric", "pixel/f1"))
    log_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sums: dict[str, float] = defaultdict(float)
        example_count = 0
        final_gradient_norm = 0.0
        final_attention: dict[str, float] = {}
        processed_batches = 0
        dataset_draws: Counter[str] = Counter()
        exact_dct_samples = 0
        supervision_draws: Counter[str] = Counter()
        for batch_index, batch in enumerate(train_loader):
            if args.maximum_train_batches is not None and batch_index >= args.maximum_train_batches:
                break
            batch = batch.to(device, non_blocking=True)
            dataset_draws.update(record.dataset for record in batch.records)
            if batch.exact_dct is not None and batch.exact_dct.valid is not None:
                exact_dct_samples += int(batch.exact_dct.valid.sum().item())
            supervision_draws["classification"] += int(batch.supervision.image_valid.sum().item())
            supervision_draws["localization"] += int(
                (batch.supervision.valid_mask.flatten(1).sum(1) > 0).sum().item()
            )
            if batch.supervision.adn is not None:
                supervision_draws["adn"] += int(
                    (batch.supervision.adn.valid_mask.flatten(1).sum(1) > 0).sum().item()
                )
            if batch.supervision.degradation is not None:
                supervision_draws["jpeg_degradation"] += int(
                    (batch.supervision.degradation.jpeg_valid.flatten(1).sum(1) > 0).sum().item()
                )
                supervision_draws["noise_degradation"] += int(
                    (batch.supervision.degradation.noise_valid.flatten(1).sum(1) > 0).sum().item()
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(
                    batch.rgb,
                    metadata=batch.metadata,
                    exact_dct=batch.exact_dct,
                    classification_valid=batch.supervision.image_valid,
                )
                losses = criterion(output, batch.supervision)
                scaled_loss = losses["total"] / accumulation_steps
            if not bool(torch.isfinite(losses["total"])):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(scaled_loss).backward()
            processed_batches += 1
            should_step = processed_batches % accumulation_steps == 0
            is_last_loader_batch = batch_index + 1 == len(train_loader)
            is_last_debug_batch = (
                args.maximum_train_batches is not None
                and batch_index + 1 >= args.maximum_train_batches
            )
            if should_step or is_last_loader_batch or is_last_debug_batch:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("gradient_clip_norm", 5.0))
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError(f"Non-finite gradient at epoch {epoch}, batch {batch_index}")
                final_gradient_norm = float(gradient_norm.detach().cpu())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            batch_size = batch.rgb.shape[0]
            example_count += batch_size
            for name, value in losses.items():
                loss_sums[name] += float(value.detach().cpu()) * batch_size
            first_level = model.front_end.fusion.level_names[0]
            final_attention = {
                name: float(value.detach().cpu())
                for name, value in output.front_end.fusion.mean_attention()[first_level].items()
            }
        if processed_batches == 0:
            raise RuntimeError("No training batches were processed")
        scheduler.step(epoch + 1)

        validation = evaluate_model(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            maximum_batches=args.maximum_val_batches,
            metric_config=config.get("metrics", {}),
        )
        selected = validation.get(metric_name)
        selected_value = float(selected) if selected is not None else float("-inf")
        is_best = selected_value > best_metric
        if is_best:
            best_metric = selected_value
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "epochs": epochs,
            "global_step": global_step,
            "learning_rates": {
                str(group.get("name", index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "gradient_norm": final_gradient_norm,
            "best_metric": best_metric,
            "selection_metric": metric_name,
            "attention": final_attention,
            "dataset_draws": dict(dataset_draws),
            "exact_dct_samples": exact_dct_samples,
            "supervision_draws": dict(supervision_draws),
            "train": {name: value / max(1, example_count) for name, value in loss_sums.items()},
            "validation": validation,
        }
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            validation=validation,
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if is_best:
            atomic_torch_save(payload, output_dir / "best.pt")
        save_every = int(training.get("save_every_epochs", 0))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            atomic_torch_save(payload, output_dir / f"epoch-{epoch + 1:04d}.pt")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "best_checkpoint": str(output_dir / "best.pt"),
                "last_checkpoint": str(output_dir / "last.pt"),
                "best_metric": best_metric,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
