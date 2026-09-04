"""Small deterministic random search for the CUDA profile."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from deepdocforgery.io import load_yaml


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    selection = str(base.get("training", {}).get("selection_metric", "pixel_macro/f1_forged"))
    for trial_index in range(trials):
        config = copy.deepcopy(base)
        selected: dict[str, Any] = {}
        for key, candidates in sorted(space.items()):
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"HPO parameter {key!r} must be a non-empty list")
            value = randomizer.choice(candidates)
            _assign(config, key, value)
            selected[key] = value
        trial_dir = output / f"trial-{trial_index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
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
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            results.append({"trial": trial_index, "status": "failed", "parameters": selected})
            continue
        log_lines = (trial_dir / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        epochs = [json.loads(line) for line in log_lines]
        eligible = [item for item in epochs if item["validation"].get(selection) is not None]
        best_epoch = (
            max(eligible, key=lambda item: float(item["validation"][selection]))
            if eligible
            else None
        )
        score = None if best_epoch is None else best_epoch["validation"][selection]
        results.append(
            {
                "trial": trial_index,
                "status": "ok",
                "score": score,
                "selection_metric": selection,
                "best_epoch": None if best_epoch is None else best_epoch["epoch"],
                "parameters": selected,
            }
        )
    successful = [item for item in results if item.get("score") is not None]
    best = max(successful, key=lambda item: float(item["score"])) if successful else None
    report = {"status": "ok" if best else "failed", "best": best, "trials": results}
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if best is None:
        raise RuntimeError("All HPO trials failed")


if __name__ == "__main__":
    main()
