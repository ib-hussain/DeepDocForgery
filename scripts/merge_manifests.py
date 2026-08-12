"""Merge dataset manifests while preserving paths, splits, and source groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.manifest import (
    ManifestRecord,
    load_manifest,
    summarize_manifest,
    validate_group_disjointness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reroot(value: str | None, source: Path, output: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    absolute = path if path.is_absolute() else source.parent / path
    return Path(os.path.relpath(absolute.resolve(), output.parent.resolve())).as_posix()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged: list[ManifestRecord] = []
    for source_argument in args.input:
        source = source_argument.resolve()
        records = load_manifest(source)
        validate_group_disjointness(records)
        for record in records:
            namespace = record.dataset.replace("/", "-")
            merged.append(
                ManifestRecord(
                    sample_id=f"{namespace}:{record.sample_id}",
                    image=_reroot(record.image, source, output) or "",
                    mask=_reroot(record.mask, source, output),
                    split=record.split,
                    label=record.label,
                    source_group=f"{namespace}:{record.source_group}",
                    dataset=record.dataset,
                    tamper_type=record.tamper_type,
                    jpeg_quality=record.jpeg_quality,
                    double_compression=record.double_compression,
                    noise_type=record.noise_type,
                    noise_strength=record.noise_strength,
                )
            )
    validate_group_disjointness(merged)
    with output.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    result = summarize_manifest(merged)
    result["manifest"] = str(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
