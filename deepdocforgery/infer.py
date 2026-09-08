"""Run checkpoint inference and save masks, overlays, instances, and JSON evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from deepdocforgery.data import _letterbox, upright_image_copy
from deepdocforgery.frequency import ExactJPEGDCTReader
from deepdocforgery.io import load_yaml, read_jpeg_metadata
from deepdocforgery.model import DeepDocForgeryModel
from deepdocforgery.postprocess import extract_instances
from deepdocforgery.runtime import load_checkpoint, resolve_device
from deepdocforgery.state import (
    JsonlJournal,
    StageLock,
    command_text,
    file_identity,
    fingerprint,
    load_json_object,
)
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override; otherwise use the config embedded in the checkpoint",
    )
    parser.add_argument("--output", type=Path, default=Path("output/inference"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--image-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-instance-area", type=int, default=16)
    parser.add_argument(
        "--exact-jpeg",
        action="store_true",
        help="Use exact coefficients for JPEG inputs at native resolution",
    )
    parser.add_argument(
        "--native-size",
        action="store_true",
        help="Run all images at native resolution instead of configured letterboxing",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Persist inference results after this many images",
    )
    parser.add_argument("--fresh", action="store_true", help="Discard matching inference state")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _paths(value: Path) -> list[Path]:
    if value.is_file():
        return [value.resolve()]
    if value.is_dir():
        paths = sorted(
            path.resolve()
            for path in value.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if paths:
            return paths
    raise FileNotFoundError(f"No supported images found at {value}")


def _inference_resume_command(args: argparse.Namespace) -> str:
    parts: list[object] = [
        "python",
        "-m",
        "deepdocforgery",
        "infer",
        "--input",
        args.input,
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--device",
        args.device,
        "--mask-threshold",
        args.mask_threshold,
        "--image-threshold",
        args.image_threshold,
        "--minimum-instance-area",
        args.minimum_instance_area,
        "--checkpoint-interval",
        args.checkpoint_interval,
        "--log-dir",
        args.log_dir,
    ]
    if args.config is not None:
        parts.extend(("--config", args.config))
    if args.exact_jpeg:
        parts.append("--exact-jpeg")
    if args.native_size:
        parts.append("--native-size")
    if args.no_progress:
        parts.append("--no-progress")
    return command_text(parts)


def _native_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _restore_letterbox(probability: torch.Tensor, transform: object) -> torch.Tensor:
    crop = probability[
        :,
        :,
        transform.top : transform.top + transform.resized_height,
        transform.left : transform.left + transform.resized_width,
    ]
    return F.interpolate(
        crop,
        size=(transform.original_height, transform.original_width),
        mode="bilinear",
        align_corners=False,
    )


def _save_visuals(
    image: Image.Image,
    probability: torch.Tensor,
    output_prefix: Path,
    threshold: float,
) -> tuple[Path, Path, Path]:
    values = probability.squeeze().detach().float().cpu().clamp(0.0, 1.0).numpy()
    probability_path = output_prefix.with_name(output_prefix.name + "-probability.png")
    binary_path = output_prefix.with_name(output_prefix.name + "-mask.png")
    overlay_path = output_prefix.with_name(output_prefix.name + "-overlay.png")
    Image.fromarray(np.round(values * 255.0).astype(np.uint8)).save(probability_path)
    Image.fromarray((values >= threshold).astype(np.uint8) * 255).save(binary_path)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    alpha = (0.65 * values[..., None]).astype(np.float32)
    overlay = rgb * (1.0 - alpha) + red * alpha
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)
    return probability_path, binary_path, overlay_path


def _infer(args: argparse.Namespace, run: RunLogger) -> dict[str, object]:
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be positive")
    if args.config is None:
        raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = raw_checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("Checkpoint has no embedded config; pass --config")
    else:
        config = load_yaml(args.config)
    device = resolve_device(args.device)
    compute = configure_compute(
        device,
        cpu_threads=config.get("training", {}).get("cpu_threads", "auto"),
    )
    reset_peak_memory(device)
    paths = _paths(args.input)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = fingerprint(
        {
            "stage": "infer",
            "checkpoint": file_identity(args.checkpoint),
            "config": config,
            "inputs": [file_identity(path) for path in paths],
            "mask_threshold": args.mask_threshold,
            "image_threshold": args.image_threshold,
            "minimum_instance_area": args.minimum_instance_area,
            "exact_jpeg": args.exact_jpeg,
            "native_size": args.native_size,
        }
    )
    journal = JsonlJournal(
        records_path=output_dir / "predictions.records.jsonl",
        state_path=output_dir / "state.json",
        stage="infer",
        contract=contract,
        total_items=len(paths),
        resume=not args.fresh,
    )
    records: list[dict[str, object]] = [dict(value) for value in journal.records]
    resumed_images = len(records)
    for index, record in enumerate(records):
        if record.get("input") != str(paths[index]):
            journal.close()
            raise ValueError("Inference resume journal no longer matches the ordered inputs")
        files = record.get("files")
        if not isinstance(files, dict) or any(
            not Path(str(files.get(name, ""))).is_file()
            for name in ("probability", "mask", "overlay")
        ):
            journal.close()
            raise ValueError(
                "Inference resume artefacts are incomplete; use --fresh to regenerate them"
            )
    resume_command = _inference_resume_command(args)
    journal.checkpoint(
        "completed" if journal.complete else "running",
        run_id=run.run_id,
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
    )
    run.info(
        f"Inference started: {len(paths)} images on {device} | "
        f"CPU threads={compute['torch_threads']}",
        event="inference_started",
        images=len(paths),
        device=str(device),
        checkpoint=str(args.checkpoint.resolve()),
        compute=compute,
        resumed_images=len(records),
        resume_state=str(journal.state_path),
    )
    run.resource(device, event="resources_initial")
    report = output_dir / "predictions.json"
    if journal.complete:
        cached = load_json_object(report)
        if cached is None:
            journal.close()
            raise ValueError(f"Completed inference state has no valid report: {report}")
        resources = resource_snapshot(device)
        journal.finish(
            run_id=run.run_id,
            report=str(report),
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
        )
        journal.close()
        run.info(
            f"Inference already complete; reused {report}",
            event="inference_reused",
            images=len(records),
            report=str(report),
            state=str(output_dir / "state.json"),
            resources=resources,
        )
        run.finish(
            "succeeded",
            cache_hit=True,
            images=len(records),
            report=str(report),
            resources=resources,
        )
        return {
            "status": "ok",
            "cache_hit": True,
            "images": len(records),
            "report": str(report),
            "resources": resources,
            "text_log": str(run.text_path),
            "events_log": str(run.events_path),
        }
    model = (
        DeepDocForgeryModel.from_config(config.get("model", {}), load_pretrained=False)
        .to(device)
        .eval()
    )
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    configured_size = config.get("data", {}).get("image_size", [512, 512])
    exact_reader = ExactJPEGDCTReader() if args.exact_jpeg else None
    progress = run.progress(
        enumerate(paths[len(records) :], start=len(records)),
        total=len(paths),
        initial=len(records),
        description="Inference",
        unit="image",
    )
    try:
        for index, image_path in progress:
            with Image.open(image_path) as image_file:
                original, orientation = upright_image_copy(image_file)
                original = original.convert("RGB")
            exact = None
            transform = None
            is_jpeg = image_path.suffix.lower() in {".jpg", ".jpeg"}
            exact_eligible = args.exact_jpeg and is_jpeg and orientation == 1
            use_native = args.native_size or exact_eligible
            if use_native:
                rgb = _native_tensor(original)
            else:
                rgb, _, _, transform = _letterbox(
                    original,
                    None,
                    (int(configured_size[0]), int(configured_size[1])),
                )
                rgb = rgb.unsqueeze(0)
            if exact_eligible:
                exact = exact_reader.read(image_path)
                metadata = exact.metadata
            else:
                metadata = read_jpeg_metadata(image_path)
            if args.exact_jpeg and is_jpeg and orientation != 1:
                run.warning(
                    f"Exact DCT disabled for EXIF-oriented JPEG: {image_path}",
                    event="exact_dct_orientation_fallback",
                    input=str(image_path),
                    exif_orientation=orientation,
                )
            with torch.no_grad():
                output = model(
                    rgb.to(device),
                    metadata=metadata.to(device),
                    exact_dct=None if exact is None else exact.to(device),
                    # Consistency is a training objective, not required for predictions.
                    compute_consistency=False,
                )
            probability = output.decoder.mask_probability.cpu()
            if transform is not None:
                probability = _restore_letterbox(probability, transform)
            instances = extract_instances(
                probability,
                threshold=args.mask_threshold,
                minimum_area=args.minimum_instance_area,
            )[0]
            output_prefix = output_dir / f"{index:05d}-{image_path.stem}"
            probability_path, binary_path, overlay_path = _save_visuals(
                original, probability, output_prefix, args.mask_threshold
            )
            image_probability = float(output.decoder.image_probability.squeeze().cpu())
            attention = {
                level: {branch: float(value.detach().cpu()) for branch, value in branches.items()}
                for level, branches in output.front_end.fusion.mean_attention().items()
            }
            record = {
                "input": str(image_path),
                "image_probability": image_probability,
                "decision": "forged" if image_probability >= args.image_threshold else "authentic",
                "mask_threshold": args.mask_threshold,
                "exact_jpeg_used": exact is not None,
                "exif_orientation": orientation,
                "instances": [instance.to_dict() for instance in instances],
                "attention": attention,
                "classification_to_localization_strength": float(
                    output.decoder.classification_to_localization_strength.cpu()
                ),
                "denoising_strength": float(output.decoder.denoising_strength.cpu()),
                "files": {
                    "probability": str(probability_path),
                    "mask": str(binary_path),
                    "overlay": str(overlay_path),
                },
            }
            records.append(record)
            journal.append(record)
            completed = index + 1
            if completed == len(paths) or completed % args.checkpoint_interval == 0:
                journal.checkpoint(
                    "running",
                    run_id=run.run_id,
                    resume_command=resume_command,
                    text_log=str(run.text_path),
                    events_log=str(run.events_path),
                )
                snapshot = resource_snapshot(device)
                progress.set_postfix_str(compact_resources(snapshot), refresh=False)
                run.event(
                    "inference_progress",
                    f"Inference: {completed}/{len(paths)} images | {compact_resources(snapshot)}",
                    completed=completed,
                    total=len(paths),
                    resources=snapshot,
                )
        progress.close()
    except BaseException as error:
        progress.close()
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        journal.checkpoint(
            status,
            run_id=run.run_id,
            error=str(error),
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
        )
        journal.close()
        raise
    result = {
        "status": "ok",
        "checkpoint": str(args.checkpoint.resolve()),
        "results": records,
        "resources": resource_snapshot(device),
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
        "resume_state": str(journal.state_path),
        "resumed_images": resumed_images,
    }
    write_json_atomic(report, result)
    journal.finish(
        run_id=run.run_id,
        report=str(report),
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
    )
    journal.close()
    summary = {
        "status": "ok",
        "cache_hit": False,
        "images": len(records),
        "report": str(report),
        "resources": result["resources"],
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
    }
    run.finish(
        "succeeded",
        images=len(records),
        report=str(report),
        resources=result["resources"],
    )
    return summary


def main() -> None:
    args = parse_args()
    run = RunLogger(
        "infer",
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    )
    output = args.output.resolve()
    try:
        with run:
            output.mkdir(parents=True, exist_ok=True)
            with StageLock(output / ".infer.lock", stage="infer"):
                result = _infer(args, run)
    except BaseException as error:
        try:
            state_path = output / "state.json"
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
