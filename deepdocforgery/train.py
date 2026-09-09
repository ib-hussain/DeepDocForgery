"""Train DeepDocForgery end to end from a JSONL manifest."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter, defaultdict
from itertools import islice
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
    capture_rng_state,
    create_dataloader,
    evaluate_model,
    load_checkpoint,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from deepdocforgery.state import StageLock, command_text, file_identity, fingerprint
from deepdocforgery.telemetry import (
    RunLogger,
    compact_resources,
    configure_compute,
    print_result,
    reset_peak_memory,
    resource_snapshot,
    utc_now,
    write_json_atomic,
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
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default="auto",
        metavar="CHECKPOINT",
        help="Resume a checkpoint; without a path, auto-use <output>/last.pt (default)",
    )
    resume_group.add_argument(
        "--fresh",
        action="store_true",
        help="Disable automatic resume; use a new output directory when artefacts already exist",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument("--no-progress", action="store_true")
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


def _metric_text(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _training_contract(
    config: dict[str, Any],
    *,
    maximum_train_batches: int | None = None,
    maximum_val_batches: int | None = None,
    device: str | None = None,
) -> str:
    """Fingerprint every setting that changes optimisation semantics."""

    contract_config = copy.deepcopy(config)
    training = contract_config.get("training")
    if isinstance(training, dict):
        training.pop("output_dir", None)
    contract_config.pop("logging", None)
    manifest_value = contract_config.get("data", {}).get("manifest")
    if manifest_value is None:
        raise ValueError("Configuration data.manifest is required")
    manifest = Path(str(manifest_value)).resolve()
    manifest_identity = file_identity(manifest, content=True)
    manifest_contract = {
        "path": manifest_identity["path"],
        "sha256": manifest_identity["sha256"],
    }
    contract_config["data"]["manifest"] = manifest_contract
    return fingerprint(
        {
            "stage": "train",
            "config": contract_config,
            "maximum_train_batches": maximum_train_batches,
            "maximum_val_batches": maximum_val_batches,
            "device": device,
        }
    )


def _train_resume_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    fallback_checkpoint: Path | None = None,
) -> str:
    parts: list[object] = [
        "python",
        "-m",
        "deepdocforgery",
        "train",
        "--config",
        args.config,
        "--device",
        args.device,
        "--output",
        output_dir,
        "--log-dir",
        args.log_dir,
    ]
    if args.manifest is not None:
        parts.extend(("--manifest", args.manifest))
    if args.epochs is not None:
        parts.extend(("--epochs", args.epochs))
    if args.maximum_train_batches is not None:
        parts.extend(("--maximum-train-batches", args.maximum_train_batches))
    if args.maximum_val_batches is not None:
        parts.extend(("--maximum-val-batches", args.maximum_val_batches))
    if not (output_dir / "last.pt").is_file() and fallback_checkpoint is not None:
        parts.extend(("--resume", fallback_checkpoint))
    if args.no_progress:
        parts.append("--no-progress")
    return command_text(parts)


def _resolve_resume_path(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.fresh:
        return None
    if args.resume in (None, "auto"):
        candidate = output_dir / "last.pt"
        return candidate if candidate.is_file() else None
    candidate = Path(str(args.resume)).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")
    return candidate


def _reconcile_metrics(path: Path, completed_epochs: int) -> None:
    """Remove rows newer than an explicitly selected checkpoint."""

    if not path.is_file():
        return
    kept: list[str] = []
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "epoch" not in value:
            raise ValueError(f"Invalid training metrics row in {path}")
        if int(value["epoch"]) <= completed_epochs:
            kept.append(json.dumps(value, sort_keys=True))
        else:
            changed = True
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        temporary.replace(path)


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
    contract: str,
    data_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "validation": validation,
        "contract": contract,
        "rng_state": capture_rng_state(),
        "data_generator_state": data_generator_state,
    }


def _train(args: argparse.Namespace, run: RunLogger) -> dict[str, Any]:
    for name in ("maximum_train_batches", "maximum_val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
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
    compute = configure_compute(
        device,
        cpu_threads=training.get("cpu_threads", "auto"),
    )
    seed = int(training.get("seed", 7))
    seed_everything(seed, deterministic=bool(training.get("deterministic", False)))

    output_dir = (
        args.output
        if args.output is not None
        else Path(training.get("output_dir", "output/model/deepdocforgery"))
    ).resolve()
    resume_path = _resolve_resume_path(args, output_dir)
    contract = _training_contract(
        config,
        maximum_train_batches=args.maximum_train_batches,
        maximum_val_batches=args.maximum_val_batches,
        device=str(device),
    )
    if resume_path is None:
        existing_run_files = [
            path
            for path in (
                output_dir / "status.json",
                output_dir / "metrics.jsonl",
                output_dir / "last.pt",
                output_dir / "best.pt",
            )
            if path.exists() and path.name != "status.json"
        ]
        if existing_run_files:
            raise FileExistsError(
                f"Training output already contains a run: {output_dir}. "
                "Automatic resume could not find last.pt; choose a new --output directory "
                "or recover an explicit checkpoint with --resume CHECKPOINT."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    run.info(
        f"Device: {device} | CPU threads: {compute['torch_threads']} | "
        f"physical/logical CPUs: {compute['physical_cpus']}/{compute['logical_cpus']}",
        event="compute_configured",
        device=str(device),
        compute=compute,
    )
    initial_resources = run.resource(device, event="resources_initial")
    write_json_atomic(
        status_path,
        {
            "status": "running",
            "command": "train",
            "run_id": run.run_id,
            "started_at": run.started_at,
            "updated_at": utc_now(),
            "output_dir": str(output_dir),
            "compute": compute,
            "resources": initial_resources,
            "text_log": str(run.text_path),
            "events_log": str(run.events_path),
            "contract": contract,
            "resume_from": None if resume_path is None else str(resume_path.resolve()),
            "resume_command": _train_resume_command(
                args, output_dir, fallback_checkpoint=resume_path
            ),
        },
    )
    reset_peak_memory(device)

    train_loader = create_dataloader(config, split="train", shuffle=True)
    val_loader = create_dataloader(config, split="val", shuffle=False)
    run.info(
        f"Data ready: train={len(train_loader.dataset)} samples, "
        f"val={len(val_loader.dataset)} samples, workers={train_loader.num_workers}",
        event="data_ready",
        train_samples=len(train_loader.dataset),
        validation_samples=len(val_loader.dataset),
        train_batches=len(train_loader),
        validation_batches=len(val_loader),
        data_workers=train_loader.num_workers,
        manifest=str(Path(config["data"]["manifest"]).resolve()),
    )
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
    supervised_counts = Counter(
        record.label for record in train_loader.dataset.records if record.classification_supervised
    )
    if supervised_counts:
        smallest = min(supervised_counts.values())
        largest = max(supervised_counts.values())
        if smallest > 0 and largest / smallest >= 3.0:
            run.warning(
                "Image-level supervision is imbalanced; monitor specificity, false-positive "
                "rate, and AUROC for classifier collapse.",
                event="classification_imbalance",
                class_counts=dict(supervised_counts),
                imbalance_ratio=largest / smallest,
            )

    model = DeepDocForgeryModel.from_config(
        config.get("model", {}), load_pretrained=resume_path is None
    ).to(device)
    criterion = DeepDocForgeryCriterion.from_config(config.get("loss", {})).to(device)
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    run.info(
        f"Model ready: {parameters['trainable']:,} trainable parameters",
        event="model_ready",
        parameters=parameters,
    )
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
    physical_batch = int(train_loader.batch_size or 1)
    run.info(
        f"Training plan: epochs={epochs} | batch={physical_batch} | "
        f"accumulation={accumulation_steps} | effective batch="
        f"{physical_batch * accumulation_steps} | AMP={use_amp}",
        event="training_plan",
        epochs=epochs,
        physical_batch_size=physical_batch,
        gradient_accumulation_steps=accumulation_steps,
        effective_batch_size=physical_batch * accumulation_steps,
        amp=use_amp,
        learning_rate=learning_rate,
        minimum_learning_rate=float(training.get("minimum_learning_rate", 1e-6)),
        selection_metric=str(training.get("selection_metric", "pixel/f1")),
    )

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    consecutive_non_finite_steps = 0
    max_consecutive_non_finite_steps = int(
        training.get("max_consecutive_non_finite_grad_steps", 20)
    )
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        saved_contract = checkpoint.get("contract")
        if saved_contract is None:
            saved_config = checkpoint.get("config")
            if not isinstance(saved_config, dict):
                raise ValueError("Legacy checkpoint has no configuration for compatibility checks")
            saved_contract = _training_contract(
                saved_config,
                maximum_train_batches=args.maximum_train_batches,
                maximum_val_batches=args.maximum_val_batches,
                device=str(device),
            )
        if saved_contract != contract:
            raise ValueError(
                "Checkpoint configuration/data do not match this training run. "
                "Use the original config and manifest, or start a documented fresh run."
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))
        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint has completed {start_epoch} epochs but this run requests {epochs}"
            )
        restored_rng = restore_rng_state(checkpoint.get("rng_state"))
        loader_generator = getattr(train_loader, "generator", None)
        saved_generator_state = checkpoint.get("data_generator_state")
        if loader_generator is not None and saved_generator_state is not None:
            loader_generator.set_state(saved_generator_state)
        _reconcile_metrics(output_dir / "metrics.jsonl", start_epoch)
        run.info(
            f"Resumed after epoch {start_epoch}/{epochs} from {resume_path.resolve()}",
            event="checkpoint_resumed",
            checkpoint=str(resume_path.resolve()),
            next_epoch=None if start_epoch == epochs else start_epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
            rng_restored=restored_rng,
        )
        if not restored_rng:
            run.warning(
                "Legacy checkpoint has no RNG state; weights and optimiser are restored, but "
                "the next epoch will not be bit-for-bit reproducible.",
                event="legacy_checkpoint_resumed",
                checkpoint=str(resume_path.resolve()),
            )
        if start_epoch == epochs and not (output_dir / "last.pt").is_file():
            # An explicitly supplied checkpoint may already have finished the
            # requested schedule but live outside this output directory. Keep
            # the local run self-contained instead of reporting a missing
            # latest checkpoint.
            atomic_torch_save(checkpoint, output_dir / "last.pt")
            if not (output_dir / "best.pt").is_file():
                atomic_torch_save(checkpoint, output_dir / "best.pt")
            run.info(
                "Completed external checkpoint adopted into this run directory",
                event="checkpoint_adopted",
                checkpoint=str(resume_path.resolve()),
                last_checkpoint=str((output_dir / "last.pt").resolve()),
            )

    metric_name = str(training.get("selection_metric", "pixel/f1"))
    log_path = output_dir / "metrics.jsonl"
    logging_config = config.get("logging", {})
    resource_interval = int(logging_config.get("resource_interval_batches", 25))
    if resource_interval < 1:
        raise ValueError("logging.resource_interval_batches must be positive")
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        run.info(
            f"Epoch {epoch + 1}/{epochs} started",
            event="epoch_started",
            epoch=epoch + 1,
            epochs=epochs,
        )
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
        total_train_batches = len(train_loader)
        if args.maximum_train_batches is not None:
            total_train_batches = min(total_train_batches, args.maximum_train_batches)
        batches = islice(train_loader, total_train_batches)
        progress = run.progress(
            enumerate(batches),
            total=total_train_batches,
            description=f"Train {epoch + 1}/{epochs}",
            unit="batch",
        )
        training_started = time.perf_counter()
        for batch_index, batch in progress:
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
            is_last_batch = batch_index + 1 == total_train_batches
            if should_step or is_last_batch:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("gradient_clip_norm", 5.0))
                )
                # A non-finite gradient here is expected, normal behaviour while
                # GradScaler calibrates its loss-scale factor early in training (or
                # occasionally afterwards) -- that is exactly what scaler.step()/
                # scaler.update() below are designed to absorb: step() skips the
                # optimizer update for this batch and update() backs off the scale.
                # Only a *persistent* run of non-finite gradients indicates a real
                # divergence problem worth stopping for.
                if not bool(torch.isfinite(gradient_norm)):
                    consecutive_non_finite_steps += 1
                    run.warning(
                        f"Non-finite gradient at epoch {epoch}, batch {batch_index} "
                        f"(scaler backing off scale; {consecutive_non_finite_steps}/"
                        f"{max_consecutive_non_finite_steps} consecutive)",
                        event="non_finite_gradient_skipped",
                        epoch=epoch,
                        batch=batch_index,
                        consecutive_non_finite_steps=consecutive_non_finite_steps,
                        scale=float(scaler.get_scale()),
                    )
                    if consecutive_non_finite_steps >= max_consecutive_non_finite_steps:
                        raise RuntimeError(
                            f"{consecutive_non_finite_steps} consecutive non-finite gradients "
                            f"through epoch {epoch}, batch {batch_index}; this is beyond normal "
                            "AMP loss-scale calibration and likely indicates a real divergence "
                            "(check learning rate, loss weights, or data)."
                        )
                else:
                    consecutive_non_finite_steps = 0
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
            completed_batches = batch_index + 1
            should_report = completed_batches == total_train_batches or (
                resource_interval > 0 and completed_batches % resource_interval == 0
            )
            if should_report:
                snapshot = resource_snapshot(device)
                mean_loss = loss_sums["total"] / max(1, example_count)
                progress.set_postfix_str(
                    f"loss={mean_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"{compact_resources(snapshot)}",
                    refresh=False,
                )
                run.event(
                    "training_progress",
                    f"Epoch {epoch + 1}: {completed_batches}/{total_train_batches} batches | "
                    f"loss={mean_loss:.4f} | {compact_resources(snapshot)}",
                    epoch=epoch + 1,
                    completed_batches=completed_batches,
                    total_batches=total_train_batches,
                    examples=example_count,
                    mean_total_loss=mean_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    resources=snapshot,
                )
        progress.close()
        training_duration = max(time.perf_counter() - training_started, 1e-9)
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
            run=run,
            progress_description=f"Validate {epoch + 1}/{epochs}",
            resource_interval_batches=resource_interval,
        )
        selected = validation.get(metric_name)
        if selected is None:
            raise RuntimeError(
                f"Selection metric {metric_name!r} is unavailable on validation data"
            )
        selected_value = float(selected)
        is_best = selected_value > best_metric
        if is_best:
            best_metric = selected_value
        record: dict[str, Any] = {
            "run_id": run.run_id,
            "timestamp": utc_now(),
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
            "resources": resource_snapshot(device),
        }
        epoch_duration = max(time.perf_counter() - epoch_started, 1e-9)
        record["performance"] = {
            "epoch_duration_seconds": epoch_duration,
            "training_duration_seconds": training_duration,
            "train_samples_per_second": example_count / training_duration,
            "train_batches_per_second": processed_batches / training_duration,
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
            contract=contract,
            data_generator_state=(
                None
                if getattr(train_loader, "generator", None) is None
                else train_loader.generator.get_state()
            ),
        )
        # Write metrics before the checkpoint. If interrupted between these
        # two atomic boundaries, resume truncates the uncommitted metric row;
        # the inverse order could permanently lose a completed epoch's metrics.
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_torch_save(payload, output_dir / "last.pt")
        if is_best:
            atomic_torch_save(payload, output_dir / "best.pt")
            run.info(
                f"New best checkpoint: {metric_name}={selected_value:.6f}",
                event="best_checkpoint_saved",
                epoch=epoch + 1,
                metric=metric_name,
                value=selected_value,
                checkpoint=str((output_dir / "best.pt").resolve()),
            )
        run.info(
            "Last checkpoint saved",
            event="last_checkpoint_saved",
            epoch=epoch + 1,
            checkpoint=str((output_dir / "last.pt").resolve()),
        )
        save_every = int(training.get("save_every_epochs", 0))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            atomic_torch_save(payload, output_dir / f"epoch-{epoch + 1:04d}.pt")
        epoch_loss = record["train"].get("total", float("nan"))
        validation_precision = validation.get("pixel/precision")
        validation_recall = validation.get("pixel/recall")
        catastrophic_rate = validation.get("failure/catastrophic_miss_rate")
        run.info(
            f"Epoch {epoch + 1}/{epochs} complete | train loss={epoch_loss:.4f} | "
            f"{metric_name}={selected_value:.6f} | "
            f"P/R={_metric_text(validation_precision)}/{_metric_text(validation_recall)} | "
            f"catastrophic={_metric_text(catastrophic_rate)} | best={best_metric:.6f}",
            event="epoch_completed",
            epoch=epoch + 1,
            epochs=epochs,
            train_total_loss=epoch_loss,
            selection_metric=metric_name,
            selected_value=selected_value,
            best_metric=best_metric,
            validation_precision=validation_precision,
            validation_recall=validation_recall,
            catastrophic_miss_rate=catastrophic_rate,
            performance=record["performance"],
            resources=record["resources"],
        )
        write_json_atomic(
            status_path,
            {
                "status": "running" if epoch + 1 < epochs else "succeeded",
                "command": "train",
                "run_id": run.run_id,
                "started_at": run.started_at,
                "updated_at": utc_now(),
                "epoch": epoch + 1,
                "epochs": epochs,
                "global_step": global_step,
                "selection_metric": metric_name,
                "best_metric": best_metric,
                "last_checkpoint": str((output_dir / "last.pt").resolve()),
                "best_checkpoint": str((output_dir / "best.pt").resolve()),
                "resources": record["resources"],
                "performance": record["performance"],
                "text_log": str(run.text_path),
                "events_log": str(run.events_path),
                "contract": contract,
                "resume_command": _train_resume_command(args, output_dir),
            },
        )

    final_resources = resource_snapshot(device)
    write_json_atomic(
        status_path,
        {
            "status": "succeeded",
            "command": "train",
            "run_id": run.run_id,
            "started_at": run.started_at,
            "updated_at": utc_now(),
            "epoch": epochs,
            "epochs": epochs,
            "global_step": global_step,
            "selection_metric": metric_name,
            "best_metric": best_metric,
            "last_checkpoint": str((output_dir / "last.pt").resolve()),
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
            "resources": final_resources,
            "text_log": str(run.text_path),
            "events_log": str(run.events_path),
            "contract": contract,
            "resume_command": _train_resume_command(args, output_dir),
        },
    )
    result = {
        "status": "ok",
        "resumed": resume_path is not None,
        "resume_from": None if resume_path is None else str(resume_path.resolve()),
        "contract": contract,
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "metrics_log": str(log_path),
        "status_report": str(status_path),
        "best_metric": best_metric,
        "selection_metric": metric_name,
        "resources": final_resources,
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
    }
    run.finish(
        "succeeded",
        output_dir=str(output_dir),
        best_metric=best_metric,
        selection_metric=metric_name,
        resources=result["resources"],
    )
    return result


def main() -> None:
    args = parse_args()
    run = RunLogger(
        "train",
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    )
    try:
        with run:
            config = load_yaml(args.config)
            training = config.get("training", {})
            output_dir = (
                args.output
                if args.output is not None
                else Path(training.get("output_dir", "output/model/deepdocforgery"))
            ).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            with StageLock(output_dir / ".train.lock", stage="train"):
                result = _train(args, run)
    except BaseException as error:
        try:
            config = load_yaml(args.config)
            training = config.get("training", {})
            output_dir = (
                args.output
                if args.output is not None
                else Path(training.get("output_dir", "output/model/deepdocforgery"))
            ).resolve()
            status_path = output_dir / "status.json"
            status_owner = None
            if status_path.is_file():
                existing_status = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(existing_status, dict):
                    status_owner = existing_status.get("run_id")
            prior_artifacts = any(
                path.exists()
                for path in (
                    output_dir / "metrics.jsonl",
                    output_dir / "last.pt",
                    output_dir / "best.pt",
                )
            )
            if status_owner == run.run_id or (status_owner is None and not prior_artifacts):
                write_json_atomic(
                    status_path,
                    {
                        "status": (
                            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                        ),
                        "command": "train",
                        "run_id": run.run_id,
                        "started_at": run.started_at,
                        "updated_at": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "text_log": str(run.text_path),
                        "events_log": str(run.events_path),
                        "resources": run.last_resources,
                        "last_checkpoint": (
                            str((output_dir / "last.pt").resolve())
                            if (output_dir / "last.pt").is_file()
                            else None
                        ),
                        "resume_command": _train_resume_command(
                            args,
                            output_dir,
                            fallback_checkpoint=(
                                None
                                if args.resume in (None, "auto")
                                else Path(str(args.resume)).resolve()
                            ),
                        ),
                    },
                )
        except Exception:
            pass
        raise
    print_result(result)


if __name__ == "__main__":
    main()
