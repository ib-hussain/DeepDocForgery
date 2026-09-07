"""Evaluate a trained checkpoint on a manifest split."""

from __future__ import annotations

import argparse
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
from deepdocforgery.telemetry import (
    RunLogger,
    configure_compute,
    print_result,
    reset_peak_memory,
    resource_snapshot,
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
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _evaluate(args: argparse.Namespace, run: RunLogger) -> dict[str, object]:
    if args.maximum_batches is not None and args.maximum_batches < 1:
        raise ValueError("--maximum-batches must be positive")
    config = load_yaml(args.config)
    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = str(args.manifest.resolve())
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
    loader = create_dataloader(config, split=args.split, shuffle=False)
    run.info(
        f"Checkpoint loaded | samples={len(loader.dataset)} | batches={len(loader)} | "
        f"workers={loader.num_workers}",
        event="evaluation_data_ready",
        checkpoint=str(args.checkpoint.resolve()),
        samples=len(loader.dataset),
        batches=len(loader),
        data_workers=loader.num_workers,
        manifest=str(Path(config["data"]["manifest"]).resolve()),
    )
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    resource_interval = int(config.get("logging", {}).get("resource_interval_batches", 25))
    if resource_interval < 1:
        raise ValueError("logging.resource_interval_batches must be positive")
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
    )
    default_name = "test-metrics.json" if args.split == "test" else f"evaluation-{args.split}.json"
    report = (args.output or Path("output/logs") / default_name).resolve()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "samples": len(loader.dataset),
        "metrics": metrics,
        "status": "ok",
        "report": str(report),
        "resources": resource_snapshot(device),
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, result)
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
        samples=len(loader.dataset),
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
        samples=len(loader.dataset),
        key_metric=key_metric,
        report=str(report),
        resources=result["resources"],
    )
    return result


def main() -> None:
    args = parse_args()
    command_name = "test" if args.split == "test" else "evaluate"
    with RunLogger(
        command_name,
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    ) as run:
        result = _evaluate(args, run)
    print_result(result)


if __name__ == "__main__":
    main()
