"""Check a CPU or CUDA profile before expensive work starts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from deepdocforgery.data import (
    load_manifest,
    summarize_manifest,
    validate_manifest_protocol,
)
from deepdocforgery.io import load_yaml
from deepdocforgery.telemetry import (
    RunLogger,
    configure_compute,
    print_result,
    resolve_worker_count,
    resource_snapshot,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _doctor(args: argparse.Namespace, run: RunLogger) -> dict[str, object]:
    config_path = args.config or Path(
        f"configs/{args.profile}/{'sample' if args.profile == 'cpu' else 'full'}.yaml"
    )
    config = load_yaml(config_path)
    manifest_value = config.get("data", {}).get("manifest")
    manifest = None if not manifest_value else Path(str(manifest_value))
    issues: list[str] = []
    device = torch.device("cuda" if args.profile == "cuda" and torch.cuda.is_available() else "cpu")
    compute = configure_compute(
        device,
        cpu_threads=config.get("training", {}).get("cpu_threads", "auto"),
    )
    resources = resource_snapshot(device)
    checks: dict[str, object] = {
        "python_torch": torch.__version__,
        "config": str(config_path.resolve()),
        "manifest": None if manifest is None else str(manifest.resolve()),
        "cuda_available": torch.cuda.is_available(),
        "compute": compute,
        "data_workers": resolve_worker_count(config.get("data", {}).get("num_workers", "auto")),
        "resources": resources,
    }
    run.info(
        f"Doctor profile={args.profile} | config={config_path} | "
        f"CPU threads={compute['torch_threads']}",
        event="doctor_started",
        profile=args.profile,
        config=str(config_path.resolve()),
        compute=compute,
    )
    run.resource(device, event="resources_initial")
    progress = run.progress(
        ("runtime", "dependencies", "manifest", "protocol"),
        total=4,
        description=f"Doctor {args.profile}",
        unit="check",
    )
    stages: dict[str, str] = {}
    checks["stages"] = stages

    runtime_issues: list[str] = []
    if args.profile == "cuda":
        if not torch.cuda.is_available():
            runtime_issues.append("CUDA profile requested but torch.cuda.is_available() is false")
        else:
            properties = torch.cuda.get_device_properties(0)
            checks["gpu"] = torch.cuda.get_device_name(0)
            checks["vram_gib"] = round(properties.total_memory / 1024**3, 2)
            runtime_config = config.get("runtime", {})
            minimum_total = float(runtime_config.get("minimum_total_vram_gib", 18.0))
            minimum_free = float(runtime_config.get("minimum_free_vram_gib", 16.0))
            maximum_utilisation = float(
                runtime_config.get("maximum_preflight_gpu_utilization_percent", 50.0)
            )
            cuda_resources = resources.get("cuda", {})
            free_vram = float(cuda_resources.get("device_free_gib", 0.0))
            if properties.total_memory < minimum_total * 1024**3:
                runtime_issues.append(
                    f"CUDA profile requires at least {minimum_total:.1f} GiB total VRAM"
                )
            if free_vram < minimum_free:
                runtime_issues.append(
                    f"Only {free_vram:.2f} GiB VRAM is free; the full profile requires "
                    f"at least {minimum_free:.1f} GiB before training"
                )
            nvidia_smi = cuda_resources.get("nvidia_smi", {})
            if isinstance(nvidia_smi, dict):
                utilisation = nvidia_smi.get("utilization_percent")
                if utilisation is not None and float(utilisation) > maximum_utilisation:
                    runtime_issues.append(
                        f"GPU utilisation is already {float(utilisation):.0f}%; stop other GPU "
                        "workloads before training"
                    )
                competing = [
                    process
                    for process in nvidia_smi.get("compute_processes", [])
                    if process.get("pid") != os.getpid()
                    and float(process.get("used_vram_gib") or 0.0) >= 0.25
                ]
                if competing:
                    preview = ", ".join(
                        f"{Path(str(process['name'])).name} "
                        f"(pid {process['pid']}, {float(process['used_vram_gib']):.2f} GiB)"
                        for process in competing[:5]
                    )
                    runtime_issues.append(f"Other CUDA compute processes are active: {preview}")
    issues.extend(runtime_issues)
    stages["runtime"] = "passed" if not runtime_issues else "attention_required"
    next(progress)
    run.info(
        f"Runtime check: {stages['runtime']}",
        event="doctor_check",
        check="runtime",
        status=stages["runtime"],
        issues=runtime_issues,
    )

    dependency_issues: list[str] = []
    if args.profile == "cuda":
        for module in ("timm", "jpegio", "lmdb"):
            available = importlib.util.find_spec(module) is not None
            checks[f"dependency/{module}"] = available
            if not available:
                dependency_issues.append(f"CUDA dependency is missing: {module}")
    else:
        for module in ("lmdb", "psutil", "tqdm"):
            available = importlib.util.find_spec(module) is not None
            checks[f"dependency/{module}"] = available
            if not available:
                dependency_issues.append(f"CPU dependency is missing: {module}")
    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        pip_message = (pip_check.stdout or pip_check.stderr).strip()
        checks["dependency/pip_check"] = pip_message or "No broken requirements found"
        if pip_check.returncode != 0:
            dependency_issues.append(f"Python environment has broken requirements: {pip_message}")
    except (OSError, subprocess.SubprocessError) as error:
        checks["dependency/pip_check"] = f"unavailable: {error}"
        dependency_issues.append(f"Could not run pip check: {error}")
    issues.extend(dependency_issues)
    stages["dependencies"] = "passed" if not dependency_issues else "attention_required"
    next(progress)
    run.info(
        f"Dependency check: {stages['dependencies']}",
        event="doctor_check",
        check="dependencies",
        status=stages["dependencies"],
        issues=dependency_issues,
    )

    manifest_issues: list[str] = []
    records = None
    if manifest is None or not manifest.is_file():
        manifest_issues.append(
            f"Manifest is missing; run: python -m deepdocforgery prepare --profile {args.profile}"
        )
    else:
        try:
            records = load_manifest(manifest)
            checks["manifest_summary"] = summarize_manifest(records)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            manifest_issues.append(f"Manifest cannot be loaded: {error}")
    issues.extend(manifest_issues)
    stages["manifest"] = "passed" if not manifest_issues else "attention_required"
    next(progress)
    run.info(
        f"Manifest check: {stages['manifest']}",
        event="doctor_check",
        check="manifest",
        status=stages["manifest"],
        issues=manifest_issues,
    )

    protocol_issues: list[str] = []
    if records is None:
        stages["protocol"] = "blocked"
    else:
        try:
            validate_manifest_protocol(records)
        except ValueError as error:
            protocol_issues.append(f"Manifest protocol is invalid: {error}")
        split_counts = {split: 0 for split in ("train", "val", "test")}
        for record in records:
            split_counts[record.split] += 1
        missing_splits = [split for split, count in split_counts.items() if count == 0]
        if missing_splits:
            protocol_issues.append(
                f"Combined manifest has no records in required splits: {missing_splits}"
            )
        training_datasets = {record.dataset for record in records if record.split == "train"}
        missing_datasets = {"doctamper", "midv"}.difference(training_datasets)
        if missing_datasets:
            protocol_issues.append(
                f"Combined training split is missing datasets: {sorted(missing_datasets)}"
            )
        training_localization = sum(
            record.localization_supervised for record in records if record.split == "train"
        )
        if training_localization == 0:
            protocol_issues.append("Training split has no localization supervision")
        train_classes = {
            record.label
            for record in records
            if record.split == "train" and record.classification_supervised
        }
        if len(train_classes) < 2:
            protocol_issues.append(
                "Training data does not contain both supervised image classes; "
                "classification is diagnostic only"
            )
        checks["protocol/split_counts"] = split_counts
        checks["protocol/training_datasets"] = sorted(training_datasets)
        checks["protocol/training_localization_supervised"] = training_localization
        stages["protocol"] = "passed" if not protocol_issues else "attention_required"
    issues.extend(protocol_issues)
    next(progress)
    run.info(
        f"Protocol check: {stages['protocol']}",
        event="doctor_check",
        check="protocol",
        status=stages["protocol"],
        issues=protocol_issues,
    )
    progress.close()
    for issue in issues:
        run.warning(issue, event="doctor_issue", profile=args.profile)
    final_resources = resource_snapshot(device)
    result = {
        "status": "ok" if not issues else "attention_required",
        "profile": args.profile,
        "checks": checks,
        "issues": issues,
        "resources": final_resources,
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
    }
    report = (args.output or Path(f"output/logs/doctor-{args.profile}.json")).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    result["report"] = str(report)
    write_json_atomic(report, result)
    run.finish(
        "succeeded" if not issues else "attention_required",
        profile=args.profile,
        issues=len(issues),
        report=str(report),
        resources=final_resources,
    )
    return result


def main() -> None:
    args = parse_args()
    with RunLogger(
        "doctor",
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    ) as run:
        result = _doctor(args, run)
    print_result(result)
    if result["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
