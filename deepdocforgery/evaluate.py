"""Evaluate a trained checkpoint on a manifest split."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from deepdocforgery.io import load_yaml
from deepdocforgery.model import DeepDocForgeryModel
from deepdocforgery.objectives import DeepDocForgeryCriterion
from deepdocforgery.runtime import (
    create_dataloader,
    evaluate_model,
    load_checkpoint,
    resolve_device,
)
from deepdocforgery.state import (
    StageLock,
    command_text,
    file_identity,
    fingerprint,
    load_json_object,
    save_stage_state,
    validate_state,
)
from deepdocforgery.telemetry import (
    RunLogger,
    configure_compute,
    print_result,
    reset_peak_memory,
    resource_snapshot,
    utc_now,
    write_json_atomic,
)


def _metric_text(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override data.manifest so official test subsets can be evaluated separately",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--maximum-batches", type=int, default=None)
    parser.add_argument(
        "--state-interval-batches",
        type=int,
        default=100,
        help="Checkpoint metric accumulators after this many batches",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore matching evaluation state")
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _report_path(args: argparse.Namespace) -> Path:
    default_name = "test-metrics.json" if args.split == "test" else f"evaluation-{args.split}.json"
    return (args.output or args.checkpoint.resolve().parent / default_name).resolve()


def _evaluation_resume_command(args: argparse.Namespace, report: Path) -> str:
    parts: list[object] = [
        "python",
        "-m",
        "deepdocforgery",
        "evaluate",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
        "--device",
        args.device,
        "--output",
        report,
        "--state-interval-batches",
        args.state_interval_batches,
        "--log-dir",
        args.log_dir,
    ]
    if args.manifest is not None:
        parts.extend(("--manifest", args.manifest))
    if args.maximum_batches is not None:
        parts.extend(("--maximum-batches", args.maximum_batches))
    if args.no_progress:
        parts.append("--no-progress")
    return command_text(parts)


def _evaluation_contract(config: dict[str, object], args: argparse.Namespace, report: Path) -> str:
    normalised = copy.deepcopy(config)
    manifest = Path(str(normalised.get("data", {}).get("manifest"))).resolve()
    manifest_info = file_identity(manifest, content=True)
    normalised["data"]["manifest"] = {
        "path": manifest_info["path"],
        "sha256": manifest_info["sha256"],
    }
    checkpoint = file_identity(args.checkpoint)
    return fingerprint(
        {
            "stage": "evaluate",
            "config": normalised,
            "checkpoint": checkpoint,
            "split": args.split,
            "maximum_batches": args.maximum_batches,
            "report": str(report),
        }
    )


def _evaluate(args: argparse.Namespace, run: RunLogger) -> dict[str, object]:
    if args.maximum_batches is not None and args.maximum_batches < 1:
        raise ValueError("--maximum-batches must be positive")
    if args.state_interval_batches < 1:
        raise ValueError("--state-interval-batches must be positive")
    config = load_yaml(args.config)
    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = str(args.manifest.resolve())
    report = _report_path(args)
    state_path = report.with_suffix(report.suffix + ".state.json")
    contract = _evaluation_contract(config, args, report)
    resume_command = _evaluation_resume_command(args, report)
    saved_state = load_json_object(state_path) if not args.fresh else None
    if saved_state is not None:
        validate_state(saved_state, stage="evaluate", contract=contract, path=state_path)
        if saved_state.get("status") == "completed" and report.is_file():
            cached = load_json_object(report)
            if cached is None:
                raise ValueError(f"Completed evaluation state has no valid report: {report}")
            cached.update(
                {
                    "cache_hit": True,
                    "resume_state": str(state_path),
                    "text_log": str(run.text_path),
                    "events_log": str(run.events_path),
                }
            )
            run.info(
                f"Evaluation already complete; reused {report}",
                event="evaluation_reused",
                report=str(report),
                state=str(state_path),
            )
            run.finish("succeeded", cache_hit=True, report=str(report))
            return cached
    initial_accumulator = None
    if saved_state is not None:
        candidate = saved_state.get("accumulator")
        if isinstance(candidate, dict):
            initial_accumulator = candidate
    save_stage_state(
        state_path,
        stage="evaluate",
        contract=contract,
        status="running",
        run_id=run.run_id,
        report=str(report),
        accumulator=initial_accumulator,
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
    )
    device = resolve_device(args.device)
    compute = configure_compute(
        device,
        cpu_threads=config.get("training", {}).get("cpu_threads", "auto"),
    )
    run.info(
        f"Evaluating split={args.split} on {device} | CPU threads={compute['torch_threads']}",
        event="evaluation_started",
        split=args.split,
        device=str(device),
        compute=compute,
    )
    run.resource(device, event="resources_initial")
    reset_peak_memory(device)
    model = DeepDocForgeryModel.from_config(config.get("model", {}), load_pretrained=False).to(
        device
    )
    criterion = DeepDocForgeryCriterion.from_config(config.get("loss", {})).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    processed_samples = 0
    processed_batches = 0
    if initial_accumulator is not None:
        processed_samples = int(initial_accumulator.get("processed_samples", 0))
        processed_batches = int(initial_accumulator.get("processed_batches", 0))
    full_loader = create_dataloader(config, split=args.split, shuffle=False)
    full_samples = len(full_loader.dataset)
    full_batches = len(full_loader)
    target_batches = (
        full_batches if args.maximum_batches is None else min(full_batches, args.maximum_batches)
    )
    if processed_batches > target_batches or processed_samples > full_samples:
        raise ValueError("Evaluation resume cursor is beyond the requested dataset")
    loader = (
        full_loader
        if processed_samples == 0
        else create_dataloader(
            config,
            split=args.split,
            shuffle=False,
            record_offset=processed_samples,
        )
    )
    run.info(
        f"Checkpoint loaded | samples={full_samples} | batches={target_batches} | "
        f"workers={loader.num_workers}",
        event="evaluation_data_ready",
        checkpoint=str(args.checkpoint.resolve()),
        samples=full_samples,
        batches=target_batches,
        resumed_batches=processed_batches,
        resumed_samples=processed_samples,
        data_workers=loader.num_workers,
        manifest=str(Path(config["data"]["manifest"]).resolve()),
    )
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    resource_interval = int(config.get("logging", {}).get("resource_interval_batches", 25))
    if resource_interval < 1:
        raise ValueError("logging.resource_interval_batches must be positive")
    latest_accumulator = initial_accumulator

    def checkpoint_accumulator(value: dict[str, object]) -> None:
        nonlocal latest_accumulator
        latest_accumulator = value
        save_stage_state(
            state_path,
            stage="evaluate",
            contract=contract,
            status="running",
            run_id=run.run_id,
            report=str(report),
            accumulator=value,
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
            resources=run.last_resources,
        )

    save_stage_state(
        state_path,
        stage="evaluate",
        contract=contract,
        status="running",
        run_id=run.run_id,
        report=str(report),
        accumulator=initial_accumulator,
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
        resources=run.last_resources,
    )
    try:
        metrics = evaluate_model(
            model,
            loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            maximum_batches=args.maximum_batches,
            metric_config=config.get("metrics", {}),
            run=run,
            progress_description=(
                f"Test {args.split}" if args.split == "test" else f"Evaluate {args.split}"
            ),
            resource_interval_batches=resource_interval,
            initial_state=initial_accumulator,
            state_callback=checkpoint_accumulator,
            total_batches_override=target_batches,
            state_interval_batches=args.state_interval_batches,
        )
    except BaseException as error:
        save_stage_state(
            state_path,
            stage="evaluate",
            contract=contract,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            run_id=run.run_id,
            report=str(report),
            accumulator=latest_accumulator,
            error_type=type(error).__name__,
            error=str(error),
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
            resources=run.last_resources,
        )
        raise
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "samples": int(metrics.get("evidence/total_samples", full_samples)),
        "metrics": metrics,
        "status": "ok",
        "report": str(report),
        "resources": resource_snapshot(device),
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
        "resume_state": str(state_path),
        "cache_hit": False,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, result)
    save_stage_state(
        state_path,
        stage="evaluate",
        contract=contract,
        status="completed",
        run_id=run.run_id,
        report=str(report),
        accumulator=latest_accumulator,
        result=result,
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
        resources=result["resources"],
    )
    key_metric = metrics.get("pixel_macro/f1_forged")
    precision = metrics.get("pixel/precision")
    recall = metrics.get("pixel/recall")
    catastrophic_rate = metrics.get("failure/catastrophic_miss_rate")
    image_auroc = metrics.get("image/auroc")
    run.info(
        f"Evaluation complete | macro F1={_metric_text(key_metric)} | "
        f"P/R={_metric_text(precision)}/{_metric_text(recall)} | "
        f"catastrophic={_metric_text(catastrophic_rate)} | "
        f"image AUROC={_metric_text(image_auroc)}",
        event="evaluation_completed",
        split=args.split,
        samples=result["samples"],
        key_metric=key_metric,
        precision=precision,
        recall=recall,
        catastrophic_miss_rate=catastrophic_rate,
        image_auroc=image_auroc,
        report=str(report),
        resources=result["resources"],
    )
    run.finish(
        "succeeded",
        split=args.split,
        samples=result["samples"],
        key_metric=key_metric,
        report=str(report),
        resources=result["resources"],
    )
    return result


def main() -> None:
    args = parse_args()
    command_name = "test" if args.split == "test" else "evaluate"
    run = RunLogger(
        command_name,
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    )
    report = _report_path(args)
    state_path = report.with_suffix(report.suffix + ".state.json")
    try:
        with run:
            with StageLock(report.with_suffix(report.suffix + ".lock"), stage="evaluate"):
                result = _evaluate(args, run)
    except BaseException as error:
        try:
            state = load_json_object(state_path)
            if state is not None and state.get("run_id") == run.run_id:
                state.update(
                    {
                        "status": (
                            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                        ),
                        "updated_at": utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "text_log": str(run.text_path),
                        "events_log": str(run.events_path),
                        "resources": run.last_resources,
                    }
                )
                write_json_atomic(state_path, state)
        except Exception:
            pass
        raise
    print_result(result)


if __name__ == "__main__":
    main()
