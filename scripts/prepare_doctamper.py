"""Export an authorized DocTamper LMDB into images, masks, and a JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from src.data.manifest import ManifestRecord, summarize_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing data.mdb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _split(group: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big")
    fraction = value / float(2**64)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    try:
        import lmdb
    except ImportError as error:
        raise RuntimeError('Install the data extra first: python -m pip install -e ".[data]"') from error
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave a test split")
    source = args.input.resolve()
    output = args.output.resolve()
    image_dir = output / "images"
    mask_dir = output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(
        str(source),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
    )
    records: list[ManifestRecord] = []
    try:
        with environment.begin(write=False) as transaction:
            count_value = transaction.get(b"num-samples")
            if count_value is None:
                raise ValueError("LMDB has no num-samples key")
            count = int(count_value)
            if args.limit is not None:
                count = min(count, args.limit)
            zero_based = transaction.get(b"image-000000000") is not None
            offset = 0 if zero_based else 1
            for position in range(count):
                source_index = position + offset
                image_bytes = transaction.get(f"image-{source_index:09d}".encode())
                mask_bytes = transaction.get(f"label-{source_index:09d}".encode())
                if image_bytes is None or mask_bytes is None:
                    raise ValueError(f"Missing image/label keys for LMDB index {source_index}")
                sample_id = f"doctamper-{source_index:09d}"
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image_format = (image.format or "JPEG").upper()
                    extension = ".png" if image_format == "PNG" else ".jpg"
                image_path = image_dir / f"{sample_id}{extension}"
                if extension == ".jpg":
                    image_path.write_bytes(image_bytes)
                else:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.convert("RGB").save(image_path)
                with Image.open(io.BytesIO(mask_bytes)) as mask_image:
                    mask = mask_image.convert("L").point(lambda value: 255 if value else 0)
                    label = int(mask.getbbox() is not None)
                    mask_path = mask_dir / f"{sample_id}.png"
                    mask.save(mask_path)
                # DocTamper's public LMDB keys do not expose the original clean
                # document identifier. The sample id is therefore the safest
                # available group, and this limitation is reported in DATASETS.md.
                source_group = sample_id
                records.append(
                    ManifestRecord(
                        sample_id=sample_id,
                        image=image_path.relative_to(output).as_posix(),
                        mask=mask_path.relative_to(output).as_posix(),
                        split=_split(
                            source_group, args.seed, args.train_ratio, args.val_ratio
                        ),
                        label=label,
                        source_group=source_group,
                        dataset="doctamper",
                        tamper_type="text_tampering" if label else "none",
                    )
                )
    finally:
        environment.close()
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    result = summarize_manifest(records)
    result["manifest"] = str(manifest)
    result["warning"] = "LMDB source-document group ids were unavailable; audit split provenance."
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
