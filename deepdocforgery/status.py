"""Summarise pipeline state and print exact recovery commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deepdocforgery.telemetry import RunLogger, print_result, resource_snapshot

HEALTHY = {"ok", "succeeded", "completed"}
ACTIONABLE = {
    "attention_required",
    "completed_with_warnings",
    "failed",
    "interrupted",
    "running",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--log-dir", type=Path, default=Path("output/logs"))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _state_paths(output: Path) -> list[Path]:
    candidates: set[Path] = set()
    state_root = output / "state"
    if state_root.is_dir():
        candidates.update(path for path in state_root.rglob("*.json") if path.is_file())
    candidates.update(path for path in output.rglob("state.json") if path.is_file())
    candidates.update(path for path in output.rglob("*.state.json") if path.is_file())
    return sorted(candidates)


def _compact_state(path: Path, value: dict[str, Any], output: Path) -> dict[str, Any]:
    completed = value.get("completed_items")
    total = value.get("total_items")
    progress: dict[str, Any] | None = None
    if completed is not None and total is not None:
        progress = {
            "completed": int(completed),
            "total": int(total),
            "percent": round(100.0 * int(completed) / max(1, int(total)), 2),
        }
    elif isinstance(value.get("completed_stages"), list):
        progress = {"completed_stages": value["completed_stages"]}
    elif value.get("completed_step") is not None:
        progress = {"completed_step": value["completed_step"]}
    return {
        "path": str(path),
        "stage": value.get("stage", path.parent.name),
        "status": value.get("status", "unknown"),
        "updated_at": value.get("updated_at"),
        "progress": progress,
        "resume_command": value.get("resume_command"),
        "report": value.get("report") or value.get("summary") or value.get("manifest"),
        "error": value.get("error"),
        "relative_path": str(path.relative_to(output)),
    }


def _status(args: argparse.Namespace, run: RunLogger) -> dict[str, Any]:
    output = args.output_root.resolve()
    operations: dict[str, dict[str, Any]] = {}
    log_root = output / "logs"
    if log_root.is_dir():
        for latest in sorted(log_root.glob("*/latest.json")):
            # This command writes its own latest.json while the scan runs.
            if latest.parent.name == "status":
                continue
            value = _read_json(latest)
            if value is not None:
                operations[latest.parent.name] = value

    training_runs: dict[str, dict[str, Any]] = {}
    model_root = output / "model"
    if model_root.is_dir():
        for status_path in sorted(model_root.glob("*/status.json")):
            value = _read_json(status_path)
            if value is not None:
                training_runs[status_path.parent.name] = value

    stage_states: dict[str, dict[str, Any]] = {}
    for path in _state_paths(output):
        value = _read_json(path)
        if value is None or "status" not in value:
            continue
        compact = _compact_state(path, value, output)
        stage_states[compact["relative_path"]] = compact

    recoverable: list[dict[str, Any]] = []
    for state in stage_states.values():
        if state["status"] in ACTIONABLE and state.get("resume_command"):
            recoverable.append(
                {
                    "stage": state["stage"],
                    "status": state["status"],
                    "progress": state["progress"],
                    "command": state["resume_command"],
                    "state": state["path"],
                }
            )
    for name, state in training_runs.items():
        if state.get("status") in ACTIONABLE and state.get("resume_command"):
            recoverable.append(
                {
                    "stage": f"train/{name}",
                    "status": state.get("status"),
                    "progress": {
                        "epoch": state.get("epoch"),
                        "epochs": state.get("epochs"),
                        "global_step": state.get("global_step"),
                    },
                    "command": state.get("resume_command"),
                    "state": str(model_root / name / "status.json"),
                }
            )

    statuses = [str(value.get("status", "unknown")) for value in operations.values()]
    statuses.extend(str(value.get("status", "unknown")) for value in training_runs.values())
    statuses.extend(str(value.get("status", "unknown")) for value in stage_states.values())
    unhealthy = [status for status in statuses if status not in HEALTHY]
    resources = resource_snapshot()
    result = {
        "status": "attention_required" if not statuses or unhealthy else "ok",
        "message": "No recorded runs found" if not statuses else "Pipeline states loaded",
        "output_root": str(output),
        "operations": operations,
        "training_runs": training_runs,
        "stage_states": stage_states,
        "recovery_commands": recoverable,
        "resources": resources,
        "text_log": str(run.text_path),
        "events_log": str(run.events_path),
    }
    run.info(
        f"Status: {len(stage_states)} states, {len(training_runs)} training runs, "
        f"{len(recoverable)} recovery commands",
        event="status_loaded",
        stage_states=len(stage_states),
        training_runs=len(training_runs),
        recovery_commands=len(recoverable),
        resources=resources,
    )
    run.finish(
        "attention_required" if result["status"] != "ok" else "succeeded",
        stage_states=len(stage_states),
        training_runs=len(training_runs),
        recovery_commands=len(recoverable),
        resources=resources,
    )
    return result


def main() -> None:
    args = parse_args()
    with RunLogger("status", log_root=args.log_dir, progress_enabled=False) as run:
        result = _status(args, run)
    print_result(result)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
