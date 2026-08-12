"""Build a group-safe JSONL manifest from ordinary image and mask folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from PIL import Image

from src.data.manifest import ManifestRecord, summarize_manifest, validate_group_disjointness

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="generic")
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument(
        "--group-regex",
        default=None,
        help="Regex whose first capture group identifies derivatives of one source",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--allow-unmasked-authentic",
        action="store_true",
        help="Treat images without a matching mask as classification-only authentic samples",
    )
    return parser.parse_args()


def _split(group: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _relative(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest.parent.resolve())).as_posix()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0 or not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("Ratios must be within [0,1]")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave a non-zero test split")
    image_root = args.images.resolve()
    mask_root = None if args.masks is None else args.masks.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found under {image_root}")
    masks: dict[str, Path] = {}
    if mask_root is not None:
        for path in mask_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                key = path.stem
                if key in masks:
                    raise ValueError(f"Duplicate mask stem {key!r}: {masks[key]} and {path}")
                masks[key] = path
    group_pattern = None if args.group_regex is None else re.compile(args.group_regex)
    records: list[ManifestRecord] = []
    for image_path in images:
        mask_path = masks.get(image_path.stem + args.mask_suffix)
        if mask_path is None and not args.allow_unmasked_authentic:
            raise FileNotFoundError(f"No mask found for {image_path.name}")
        if mask_path is None:
            label = 0
        else:
            with Image.open(mask_path) as mask_image:
                extrema = mask_image.convert("L").getextrema()
            label = int(extrema is not None and extrema[1] > 0)
        group = image_path.stem
        if group_pattern is not None:
            match = group_pattern.search(image_path.stem)
            if match is None or not match.groups():
                raise ValueError(f"group-regex did not capture a group from {image_path.stem!r}")
            group = match.group(1)
        records.append(
            ManifestRecord(
                sample_id=image_path.relative_to(image_root).with_suffix("").as_posix(),
                image=_relative(image_path, output),
                mask=None if mask_path is None else _relative(mask_path, output),
                split=_split(group, args.seed, args.train_ratio, args.val_ratio),
                label=label,
                source_group=group,
                dataset=args.dataset,
            )
        )
    validate_group_disjointness(records)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    print(json.dumps(summarize_manifest(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
