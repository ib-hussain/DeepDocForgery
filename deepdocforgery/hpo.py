"""Small deterministic random search for the CUDA profile."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from deepdocforgery.io import load_yaml
from deepdocforgery.runtime import resolve_device
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
    compact_resources,
    configure_compute,
    print_result,
    resource_snapshot,
    utc_now,
    write_json_atomic,
)


def _assign(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"HPO path collides with a scalar: {dotted_key}")
        current = child
    current[parts[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cuda/full.yaml"))
    parser.add_argument("--search", type=Path, default=Path("configs/cuda/hpo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output/hpo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a separate study instead of resuming its deterministic study directory",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _hpo_resume_command(args: argparse.Namespace) -> str:
    parts: list[object] = [
        "python",
        "-m",
        "deepdocforgery",
        "hpo",
        "--config",
        args.config,
        "--search",
        args.search,
        "--output",
        args.output,
        "--device",
        args.device,
        "--log-dir",
        args.log_dir,
    ]
    if args.trials is not None:
        parts.extend(("--trials", args.trials))
    if args.no_progress:
        parts.append("--no-progress")
    return command_text(parts)


def _hpo(args: argparse.Namespace, run: RunLogger) -> dict[str, Any]:
    base = load_yaml(args.config)
    search = load_yaml(args.search)
    trials = int(search.get("trials", 12) if args.trials is None else args.trials)
    if trials < 1:
        raise ValueError("At least one HPO trial is required")
    seed = int(search.get("seed", 7))
    randomizer = random.Random(seed)
    space = search.get("parameters", {})
    if not isinstance(space, dict) or not space:
        raise ValueError("HPO search must define a non-empty parameters mapping")
    plans: list[dict[str, Any]] = []
    for _ in range(trials):
        selected: dict[str, Any] = {}
        for key, candidates in sorted(space.items()):
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"HPO parameter {key!r} must be a non-empty list")
            selected[key] = randomizer.choice(candidates)
        plans.append(selected)
    base_contract = copy.deepcopy(base)
    manifest_value = base_contract.get("data", {}).get("manifest")
    if manifest_value is None:
        raise ValueError("HPO base configuration data.manifest is required")
    manifest = file_identity(Path(str(manifest_value)).resolve(), content=True)
    base_contract["data"]["manifest"] = {
        "path": manifest["path"],
        "sha256": manifest["sha256"],
    }
    contract = fingerprint(
        {
            "stage": "hpo",
            "base": base_contract,
            "search": search,
            "requested_trials": trials,
            "device": args.device,
            "plans": plans,
        }
    )
    output_root = args.output.resolve()
    study_name = f"study-{contract[:12]}"
    if args.fresh:
        study_name = f"{study_name}-{run.run_id}"
    output = output_root / study_name
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    resume_command = _hpo_resume_command(args)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("HPO is CUDA-only; pass --device cuda")
    compute = configure_compute(
        device,
        cpu_threads=base.get("training", {}).get("cpu_threads", "auto"),
    )
    run.info(
        f"HPO started: {trials} trials on {device}",
        event="hpo_started",
        trials=trials,
        device=str(device),
        output=str(output),
        output_root=str(output_root),
        compute=compute,
        search_space=space,
        contract=contract,
        resume=not args.fresh,
    )
    run.resource(device, event="resources_initial")
    saved_state = load_json_object(state_path) if not args.fresh else None
    results_by_trial: dict[int, dict[str, Any]] = {}
    if saved_state is not None:
        validate_state(saved_state, stage="hpo", contract=contract, path=state_path)
        for value in saved_state.get("trials", []):
            if isinstance(value, dict) and "trial" in value:
                results_by_trial[int(value["trial"])] = value
        run.info(
            f"HPO resume state loaded: {len(results_by_trial)}/{trials} recorded trials",
            event="hpo_resumed",
            state=str(state_path),
            recorded_trials=len(results_by_trial),
            trials=trials,
        )
    save_stage_state(
        state_path,
        stage="hpo",
        contract=contract,
        status="running",
        run_id=run.run_id,
        output_dir=str(output),
        plans=plans,
        trials=[results_by_trial[index] for index in sorted(results_by_trial)],
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
    )
    results: list[dict[str, Any]] = []
    selection = str(base.get("training", {}).get("selection_metric", "pixel_macro/f1_forged"))

    def record_trial(value: dict[str, Any]) -> None:
        trial_number = int(value["trial"])
        results_by_trial[trial_number] = value
        write_json_atomic(output / f"trial-{trial_number:03d}" / "result.json", value)
        save_stage_state(
            state_path,
            stage="hpo",
            contract=contract,
            status="running",
            run_id=run.run_id,
            output_dir=str(output),
            plans=plans,
            trials=[results_by_trial[index] for index in sorted(results_by_trial)],
            current_trial=trial_number,
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
            resources=resource_snapshot(device),
        )

    progress = run.progress(range(trials), total=trials, description="HPO", unit="trial")
    for trial_index in progress:
        trial_started = time.perf_counter()
        config = copy.deepcopy(base)
        selected = plans[trial_index]
        for key, value in selected.items():
            _assign(config, key, value)
        previous = results_by_trial.get(trial_index)
        if previous is not None and previous.get("status") == "ok":
            results.append(previous)
            run.info(
                f"Trial {trial_index + 1}/{trials} reused | {selection}={previous.get('score')}",
                event="hpo_trial_reused",
                trial=trial_index,
                parameters=selected,
                score=previous.get("score"),
            )
            continue
        trial_dir = output / f"trial-{trial_index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run.info(
            f"Trial {trial_index + 1}/{trials} started: {selected}",
            event="hpo_trial_started",
            trial=trial_index,
            trials=trials,
            parameters=selected,
            resources=resource_snapshot(device),
        )
        command = [
            sys.executable,
            "-m",
            "deepdocforgery",
            "train",
            "--config",
            str(config_path),
            "--device",
            args.device,
            "--output",
            str(trial_dir / "run"),
            "--epochs",
            str(int(search.get("epochs_per_trial", 5))),
            "--maximum-train-batches",
            str(int(search.get("maximum_train_batches", 400))),
            "--maximum-val-batches",
            str(int(search.get("maximum_val_batches", 100))),
            "--log-dir",
            str(args.log_dir.resolve()),
        ]
        if args.no_progress:
            command.append("--no-progress")
        # Preserve HPO stdout as one valid final JSON document. Child training
        # progress remains live on stderr, while its machine-readable stdout
        # is retained with the trial.
        child_result_path = trial_dir / "train-result.json"
        with child_result_path.open("w", encoding="utf-8") as child_result:
            completed = subprocess.run(command, check=False, stdout=child_result)
        if completed.returncode != 0:
            result = {
                "trial": trial_index,
                "status": "failed",
                "return_code": completed.returncode,
                "duration_seconds": time.perf_counter() - trial_started,
                "parameters": selected,
                "train_result": str(child_result_path),
            }
            results.append(result)
            record_trial(result)
            run.warning(
                f"Trial {trial_index + 1}/{trials} failed with exit code {completed.returncode}",
                event="hpo_trial_failed",
                trial=trial_index,
                return_code=completed.returncode,
                parameters=selected,
                train_result=str(child_result_path),
                resources=resource_snapshot(device),
            )
            continue
        trial_status_path = trial_dir / "run" / "status.json"
        metrics_path = trial_dir / "run" / "metrics.jsonl"
        try:
            trial_status = json.loads(trial_status_path.read_text(encoding="utf-8"))
            if not isinstance(trial_status, dict):
                raise ValueError("trial status is not a JSON object")
            if trial_status.get("status") != "succeeded":
                raise ValueError(
                    f"trial status is {trial_status.get('status', 'missing')!r}, not 'succeeded'"
                )
            trial_resources = trial_status.get("resources")
            log_lines = metrics_path.read_text(encoding="utf-8").splitlines()
            epochs = [json.loads(line) for line in log_lines if line.strip()]
            if not all(isinstance(epoch, dict) for epoch in epochs):
                raise ValueError("trial metrics contain a non-object JSON value")
        except (OSError, ValueError) as error:
            result = {
                "trial": trial_index,
                "status": "failed",
                "reason": f"invalid training artefacts: {error}",
                "duration_seconds": time.perf_counter() - trial_started,
                "parameters": selected,
                "train_result": str(child_result_path),
            }
            results.append(result)
            record_trial(result)
            run.warning(
                f"Trial {trial_index + 1}/{trials} produced invalid artefacts: {error}",
                event="hpo_trial_failed",
                trial=trial_index,
                reason="invalid_training_artefacts",
                error=str(error),
                parameters=selected,
                train_result=str(child_result_path),
                resources=resource_snapshot(device),
            )
            continue
        eligible = [
            item
            for item in epochs
            if isinstance(item.get("validation"), dict)
            and item["validation"].get(selection) is not None
        ]
        best_epoch = (
            max(eligible, key=lambda item: float(item["validation"][selection]))
            if eligible
            else None
        )
        score = None if best_epoch is None else best_epoch["validation"][selection]
        if score is None:
            result = {
                "trial": trial_index,
                "status": "failed",
                "reason": f"selection metric unavailable: {selection}",
                "duration_seconds": time.perf_counter() - trial_started,
                "parameters": selected,
                "train_result": str(child_result_path),
                "resources": trial_resources,
            }
            results.append(result)
            record_trial(result)
            run.warning(
                f"Trial {trial_index + 1}/{trials} has no {selection} value",
                event="hpo_trial_failed",
                trial=trial_index,
                reason="selection_metric_unavailable",
                selection_metric=selection,
                parameters=selected,
                resources=(
                    trial_resources
                    if isinstance(trial_resources, dict)
                    else resource_snapshot(device)
                ),
            )
            continue
        result = {
            "trial": trial_index,
            "status": "ok",
            "score": score,
            "selection_metric": selection,
            "best_epoch": None if best_epoch is None else best_epoch["epoch"],
            "duration_seconds": time.perf_counter() - trial_started,
            "parameters": selected,
            "train_result": str(child_result_path),
            "resources": trial_resources,
        }
        results.append(result)
        record_trial(result)
        snapshot = (
            trial_resources if isinstance(trial_resources, dict) else resource_snapshot(device)
        )
        progress.set_postfix_str(
            f"score={score if score is not None else 'n/a'} | {compact_resources(snapshot)}",
            refresh=False,
        )
        run.info(
            f"Trial {trial_index + 1}/{trials} complete | {selection}={score}",
            event="hpo_trial_completed",
            trial=trial_index,
            score=score,
            selection_metric=selection,
            parameters=selected,
            resources=snapshot,
        )
    progress.close()
    successful = [item for item in results if item.get("score") is not None]
    failed_trials = sum(item["status"] == "failed" for item in results)
    best = max(successful, key=lambda item: float(item["score"])) if successful else None
    summary_path = output / "summary.json"
    if best is None:
        report_status = "failed"
    elif failed_trials:
        report_status = "ok_with_warnings"
    else:
        report_status = "ok"
    report = {
        "status": report_status,
        "contract": contract,
        "resumed": saved_state is not None,
        "best": best,
        "successful_trials": len(successful),
        "failed_trials": failed_trials,
        "output_dir": str(output),
        "trials": results,
        "resources": resource_snapshot(device),
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
        "summary": str(summary_path),
    }
    write_json_atomic(summary_path, report)
    if best is None:
        save_stage_state(
            state_path,
            stage="hpo",
            contract=contract,
            status="failed",
            run_id=run.run_id,
            output_dir=str(output),
            plans=plans,
            trials=results,
            summary=str(summary_path),
            error="All HPO trials failed",
            resume_command=resume_command,
            text_log=str(run.text_path),
            events_log=str(run.events_path),
            resources=report["resources"],
        )
        raise RuntimeError("All HPO trials failed")
    save_stage_state(
        state_path,
        stage="hpo",
        contract=contract,
        status="completed" if not failed_trials else "completed_with_warnings",
        run_id=run.run_id,
        output_dir=str(output),
        plans=plans,
        trials=results,
        best=best,
        summary=str(summary_path),
        resume_command=resume_command,
        text_log=str(run.text_path),
        events_log=str(run.events_path),
        resources=report["resources"],
    )
    run.info(
        f"HPO complete | best {selection}={best['score']} (trial {best['trial'] + 1})",
        event="hpo_completed",
        best=best,
        summary=str(summary_path),
        resources=report["resources"],
    )
    run.finish(
        "attention_required" if failed_trials else "succeeded",
        trials=trials,
        successful_trials=len(successful),
        failed_trials=failed_trials,
        best=best,
        summary=str(summary_path),
        resources=report["resources"],
    )
    return report


def main() -> None:
    args = parse_args()
    run = RunLogger(
        "hpo",
        log_root=args.log_dir,
        progress_enabled=not args.no_progress,
    )
    try:
        with run:
            with StageLock(args.output.resolve() / ".hpo.lock", stage="hpo"):
                result = _hpo(args, run)
    except BaseException as error:
        try:
            for state_path in args.output.resolve().glob("study-*/state.json"):
                state = load_json_object(state_path)
                if state is None or state.get("run_id") != run.run_id:
                    continue
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
