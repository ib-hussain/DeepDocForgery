"""Check a CPU or CUDA profile before expensive work starts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from deepdocforgery.data import load_manifest, summarize_manifest
from deepdocforgery.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or Path(
        f"configs/{args.profile}/{'sample' if args.profile == 'cpu' else 'full'}.yaml"
    )
    config = load_yaml(config_path)
    manifest = Path(config.get("data", {}).get("manifest", ""))
    issues: list[str] = []
    checks: dict[str, object] = {
        "python_torch": torch.__version__,
        "config": str(config_path.resolve()),
        "manifest": str(manifest.resolve()) if str(manifest) else None,
        "cuda_available": torch.cuda.is_available(),
    }
    if args.profile == "cuda":
        if not torch.cuda.is_available():
            issues.append("CUDA profile requested but torch.cuda.is_available() is false")
        else:
            properties = torch.cuda.get_device_properties(0)
            checks["gpu"] = torch.cuda.get_device_name(0)
            checks["vram_gib"] = round(properties.total_memory / 1024**3, 2)
            if properties.total_memory < 18 * 1024**3:
                issues.append("CUDA full profile is tuned for approximately 20 GiB VRAM")
        for module in ("timm", "jpegio", "lmdb"):
            available = importlib.util.find_spec(module) is not None
            checks[f"dependency/{module}"] = available
            if not available:
                issues.append(f"Optional CUDA dependency is missing: {module}")
    if not str(manifest) or not manifest.is_file():
        issues.append(
            f"Manifest is missing; run: python -m deepdocforgery prepare --profile {args.profile}"
        )
    else:
        records = load_manifest(manifest)
        checks["manifest_summary"] = summarize_manifest(records)
        train_classes = {
            record.label
            for record in records
            if record.split == "train" and record.classification_supervised
        }
        if len(train_classes) < 2:
            issues.append(
                "Training data does not contain both supervised image classes; "
                "classification is diagnostic only"
            )
    result = {
        "status": "ok" if not issues else "attention_required",
        "profile": args.profile,
        "checks": checks,
        "issues": issues,
    }
    report = (args.output or Path(f"output/logs/doctor-{args.profile}.json")).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    result["report"] = str(report)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
