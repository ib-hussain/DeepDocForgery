"""Evaluate a trained checkpoint on a manifest split."""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = str(args.manifest.resolve())
    device = resolve_device(args.device)
    model = DeepDocForgeryModel.from_config(config.get("model", {}), load_pretrained=False).to(
        device
    )
    criterion = DeepDocForgeryCriterion.from_config(config.get("loss", {})).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    loader = create_dataloader(config, split=args.split, shuffle=False)
    use_amp = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        maximum_batches=args.maximum_batches,
        metric_config=config.get("metrics", {}),
    )
    report = (args.output or Path("output/logs") / f"evaluation-{args.split}.json").resolve()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "samples": len(loader.dataset),
        "metrics": metrics,
        "status": "ok",
        "report": str(report),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
