"""Summarise the latest pipeline operations and model-training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deepdocforgery.telemetry import print_result, resource_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    operations: dict[str, dict[str, Any]] = {}
    log_root = output / "logs"
    if log_root.is_dir():
        for latest in sorted(log_root.glob("*/latest.json")):
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

    states = [str(value.get("status", "unknown")) for value in operations.values()]
    states.extend(str(value.get("status", "unknown")) for value in training_runs.values())
    unhealthy = {"failed", "interrupted", "attention_required", "running", "unknown"}
    result = {
        "status": (
            "attention_required"
            if not states or any(state in unhealthy for state in states)
            else "ok"
        ),
        "message": "No recorded runs found" if not states else "Latest run states loaded",
        "output_root": str(output),
        "operations": operations,
        "training_runs": training_runs,
        "resources": resource_snapshot(),
    }
    print_result(result)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
