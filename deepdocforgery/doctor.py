"""Check a CPU or CUDA profile before expensive work starts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import torch

from deepdocforgery.data import (
    ManifestRecord,
    load_manifest,
    summarize_manifest,
    validate_manifest_protocol,
)
from deepdocforgery.io import load_yaml
from deepdocforgery.state import (
    StageLock,
    command_text,
    file_identity,
    fingerprint,
    load_json_object,
    save_stage_state,
)
from deepdocforgery.telemetry import (
    RunLogger,
    configure_compute,
    print_result,
    resolve_worker_count,
    resource_snapshot,
    utc_now,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--state",
        type=Path,
        help="Progress state (default: output/state/doctor/<profile>.json)",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _doctor_resume_command(args: argparse.Namespace) -> str:
    parts: list[object] = [
        "python",
        "-m",
        "deepdocforgery",
        "doctor",
        "--profile",
        args.profile,
        "--log-dir",
        args.log_dir,
    ]
    for option, value in (
        ("--config", args.config),
        ("--output", args.output),
        ("--state", args.state),
    ):
        if value is not None:
            parts.extend((option, value))
    if args.no_progress:
        parts.append("--no-progress")
    return command_text(parts)


def _audit_record_files(
    record: ManifestRecord,
    *,
    manifest_root: Path,
) -> tuple[int, list[dict[str, str]]]:
    """Check one manifest record without decoding its potentially large images."""

    references = [("image", record.image)]
    if record.mask is not None:
        references.append(("mask", record.mask))
    if record.adn_text_mask is not None:
        references.append(("adn_text_mask", record.adn_text_mask))
    missing: list[dict[str, str]] = []
    for kind, value in references:
        path = Path(value)
        resolved = path if path.is_absolute() else manifest_root / path
        if not resolved.is_file():
            missing.append(
                {
                    "sample_id": record.sample_id,
                    "kind": kind,
                    "path": str(resolved),
                }
            )
    return len(references), missing


def _iter_file_audits(
    records: list[ManifestRecord],
    *,
    manifest_root: Path,
    workers: int,
) -> Iterable[tuple[int, list[dict[str, str]]]]:
    check = partial(_audit_record_files, manifest_root=manifest_root)
    if workers <= 1:
        for record in records:
            yield check(record)
        return
    # Limit queued work: a full combined manifest may contain hundreds of
    # thousands of records, while each individual stat call is very small.
    batch_size = max(32, workers * 16)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="doctor-files") as executor:
        for start in range(0, len(records), batch_size):
            yield from executor.map(check, records[start : start + batch_size])


def audit_manifest_files(
    records: list[ManifestRecord],
    manifest: Path,
    *,
    workers: int,
    run: RunLogger | None = None,
    missing_preview_limit: int = 10,
) -> dict[str, Any]:
    """Verify every image/mask reference and retain only a bounded error preview."""

    checked = 0
    missing_count = 0
    preview: list[dict[str, str]] = []
    audits = _iter_file_audits(
        records,
        manifest_root=manifest.resolve().parent,
        workers=workers,
    )
    tracked = (
        run.progress(
            audits,
            total=len(records),
            description="Doctor files",
            unit="record",
        )
        if run is not None
        else audits
    )
    for reference_count, missing in tracked:
        checked += reference_count
        missing_count += len(missing)
        remaining = max(0, missing_preview_limit - len(preview))
        preview.extend(missing[:remaining])
    return {
        "records_checked": len(records),
        "references_checked": checked,
        "missing_references": missing_count,
        "missing_preview": preview,
        "workers": workers,
    }


def _doctor(args: argparse.Namespace, run: RunLogger) -> dict[str, object]:
    config_path = args.config or Path(
        f"configs/{args.profile}/{'sample' if args.profile == 'cpu' else 'full'}.yaml"
    )
    config = load_yaml(config_path)
    manifest_value = config.get("data", {}).get("manifest")
    manifest = None if not manifest_value else Path(str(manifest_value))
    report = (args.output or Path(f"output/logs/doctor-{args.profile}.json")).resolve()
    state_path = (args.state or Path(f"output/state/doctor/{args.profile}.json")).resolve()
    contract = fingerprint(
        {
            "stage": "doctor",
            "profile": args.profile,
            "config": file_identity(config_path, content=True),
            "manifest": (
                None if manifest is None or not manifest.is_file() else file_identity(manifest)
            ),
            "python": str(Path(sys.executable).resolve()),
            "torch": torch.__version__,
        }
    )
    previous = load_json_object(state_path)
    issues: list[str] = []
    device = torch.device("cuda" if args.profile == "cuda" and torch.cuda.is_available() else "cpu")
    compute = configure_compute(
        device,
        cpu_threads=config.get("training", {}).get("cpu_threads", "auto"),
    )
    resources = resource_snapshot(device)
    data_workers = resolve_worker_count(config.get("data", {}).get("num_workers", "auto"))
    checks: dict[str, object] = {
        "python_torch": torch.__version__,
        "config": str(config_path.resolve()),
        "manifest": None if manifest is None else str(manifest.resolve()),
        "cuda_available": torch.cuda.is_available(),
        "compute": compute,
        "data_workers": data_workers,
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
    completed_stages: list[str] = []
    resume_command = _doctor_resume_command(args)

    if previous is not None:
        run.info(
            "Previous doctor state found; volatile checks will be refreshed",
            event="doctor_state_loaded",
            previous_status=previous.get("status"),
            previous_updated_at=previous.get("updated_at"),
            compatible=previous.get("contract") == contract,
            state=str(state_path),
        )

    def checkpoint(status: str = "running") -> None:
        save_stage_state(
            state_path,
            stage="doctor",
            contract=contract,
            status=status,
            run_id=run.run_id,
            profile=args.profile,
            completed_stages=completed_stages,
            checks=checks,
            issues=issues,
            report=str(report),
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
            resources=resource_snapshot(device),
        )

    checkpoint()

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
    completed_stages.append("runtime")
    checkpoint()

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
    completed_stages.append("dependencies")
    checkpoint()

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
            file_audit = audit_manifest_files(
                records,
                manifest,
                workers=data_workers,
                run=run,
            )
            checks["manifest_files"] = file_audit
            if file_audit["missing_references"]:
                manifest_issues.append(
                    f"Manifest references {file_audit['missing_references']} missing files; "
                    f"examples: {file_audit['missing_preview']}"
                )
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
    completed_stages.append("manifest")
    checkpoint()

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
    completed_stages.append("protocol")
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
    report.parent.mkdir(parents=True, exist_ok=True)
    result["report"] = str(report)
    write_json_atomic(report, result)
    checkpoint("completed" if not issues else "attention_required")
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
    run = RunLogger(
        "doctor",
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    )
    state_path = (args.state or Path(f"output/state/doctor/{args.profile}.json")).resolve()
    try:
        with run:
            with StageLock(state_path.with_suffix(".lock"), stage=f"doctor/{args.profile}"):
                result = _doctor(args, run)
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
    if result["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
