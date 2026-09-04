"""Build one leakage-checked manifest from DocTamper and MIDV-DM."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from deepdocforgery.data import (
    IMAGE_EXTENSIONS,
    ManifestRecord,
    summarize_manifest,
    validate_manifest_protocol,
    write_manifest,
)

DOCTAMPER_SUBSETS = {
    "DocTamperV1-TrainingSet": "doctamper-training",
    "DocTamperV1-TestingSet": "doctamper-testing",
    "DocTamperV1-FCD": "doctamper-fcd",
    "DocTamperV1-SCD": "doctamper-scd",
}

PROFILE_DEFAULTS = {
    "cpu": (
        Path("data/sample-doctamper"),
        Path("data/sample-midv"),
        Path("output/manifests/cpu.jsonl"),
    ),
    "cuda": (
        Path("data/dataset-doctamper"),
        Path("data/dataset-midv"),
        Path("output/manifests/cuda.jsonl"),
    ),
}
DEFAULT_PROCESSED_ROOT = Path("output/processed")


def infer_subset(path: Path, requested: str = "auto") -> str:
    """Return the canonical DocTamper subset name."""

    if requested != "auto":
        if requested not in {"training", "testing", "fcd", "scd"}:
            raise ValueError("subset must be auto, training, testing, fcd, or scd")
        return requested
    name = path.name.lower()
    for token in ("training", "testing", "fcd", "scd"):
        if token in name:
            return token
    raise ValueError(f"Cannot infer DocTamper subset from {path}")


def assigned_split(sample_id: str, subset: str, seed: int, val_ratio: float) -> str:
    """Compatibility helper implementing the official split rule."""

    if subset in {"testing", "fcd", "scd"}:
        return "test"
    if subset != "training":
        raise ValueError(f"Unknown DocTamper subset {subset!r}")
    return "val" if _hash_fraction(sample_id, seed) < val_ratio else "train"


def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _split_groups(
    records: list[ManifestRecord],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> list[ManifestRecord]:
    groups = sorted({record.source_group for record in records})
    assignments: dict[str, str] = {}
    for group in groups:
        value = _hash_fraction(group, seed)
        if value < test_fraction:
            split = "test"
        elif value < test_fraction + validation_fraction:
            split = "val"
        else:
            split = "train"
        assignments[group] = split
    # Tiny samples should still exercise validation where mathematically possible.
    if len(groups) >= 2 and "val" not in assignments.values():
        assignments[groups[0]] = "val"
    if len(groups) >= 3 and test_fraction > 0 and "test" not in assignments.values():
        assignments[groups[1]] = "test"
    if groups and "train" not in assignments.values():
        assignments[groups[-1]] = "train"
    return [replace(record, split=assignments[record.source_group]) for record in records]


def _mask_array(image: Image.Image) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 2:
        return values > 0
    # Alpha is deliberately ignored: the supplied authentic MIDV mask has an
    # opaque alpha plane but zero RGB, which correctly means no tamper pixels.
    return values[..., :3].max(axis=2) > 0


def _write_binary_mask(image: Image.Image, path: Path) -> int:
    binary = _mask_array(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(binary.astype(np.uint8) * 255).save(temporary)
    os.replace(temporary, path)
    return int(binary.any())


def _doctamper_source_group(image: Image.Image, mask: Image.Image) -> str:
    """Conservative perceptual proxy after hiding the labelled edit region."""

    gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    edited = _mask_array(mask)
    if edited.shape != gray.shape:
        raise ValueError("DocTamper grouping image/mask shapes do not match")
    background = gray[~edited]
    gray[edited] = int(np.median(background)) if background.size else 255
    thumbnail = np.asarray(
        Image.fromarray(gray).resize((8, 8), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    bits = thumbnail >= np.median(thumbnail)
    return f"doctamper:masked-ahash-{np.packbits(bits.reshape(-1)).tobytes().hex()}"


def _lmdb_items(source: Path, limit: int | None) -> Iterable[tuple[int, bytes, bytes]]:
    try:
        import lmdb
    except ImportError as error:
        raise RuntimeError("LMDB input requires: python -m pip install -e '.[data]'") from error
    environment = lmdb.open(
        str(source), readonly=True, lock=False, readahead=False, meminit=False, max_readers=64
    )
    try:
        with environment.begin(write=False) as transaction:
            raw_count = transaction.get(b"num-samples")
            if raw_count is None:
                raise ValueError(f"{source} has no num-samples key")
            count = int(raw_count)
            if limit is not None:
                count = min(count, limit)
            zero_based = transaction.get(b"image-000000000") is not None
            offset = 0 if zero_based else 1
            for position in range(count):
                index = position + offset
                image = transaction.get(f"image-{index:09d}".encode())
                label = transaction.get(f"label-{index:09d}".encode())
                if image is None or label is None:
                    raise ValueError(f"Missing image/label keys for LMDB index {index}")
                yield index, image, label
    finally:
        environment.close()


def _doctamper_lmdb_records(
    source: Path,
    prepared: Path,
    *,
    benchmark: str,
    limit: int | None,
) -> list[ManifestRecord]:
    image_dir = prepared / "images"
    mask_dir = prepared / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[ManifestRecord] = []
    for index, image_bytes, mask_bytes in _lmdb_items(source, limit):
        sample_id = f"{benchmark}-{index:09d}"
        with Image.open(io.BytesIO(image_bytes)) as image:
            extension = ".jpg" if image.format == "JPEG" else ".png"
            image_size = image.size
            grouping_image = image.convert("RGB").copy()
        with Image.open(io.BytesIO(mask_bytes)) as raw_mask:
            grouping_mask = raw_mask.copy()
        image_path = image_dir / f"{index:09d}{extension}"
        if not image_path.exists():
            temporary = image_path.with_suffix(image_path.suffix + ".tmp")
            if extension == ".jpg":
                temporary.write_bytes(image_bytes)
            else:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.convert("RGB").save(temporary, format="PNG")
            os.replace(temporary, image_path)
        mask_path = mask_dir / f"{index:09d}.png"
        if mask_path.exists():
            with Image.open(mask_path) as mask:
                label = int(_mask_array(mask).any())
                if mask.size != image_size:
                    raise ValueError(f"Image/mask size mismatch for {sample_id}")
        else:
            with Image.open(io.BytesIO(mask_bytes)) as mask:
                if mask.size != image_size:
                    raise ValueError(f"Image/mask size mismatch for {sample_id}")
                label = _write_binary_mask(mask, mask_path)
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                image=str(image_path.resolve()),
                mask=str(mask_path.resolve()),
                split="test",
                label=label,
                source_group=_doctamper_source_group(grouping_image, grouping_mask),
                dataset="doctamper",
                benchmark=benchmark,
                classification_supervised=False,
                localization_supervised=True,
                adn_supervision="proxy",
                tamper_type="text_tampering" if label else "none",
            )
        )
    return records


def _paired_file(directory: Path, stem: Path) -> Path:
    candidates = [directory / stem.with_suffix(extension) for extension in sorted(IMAGE_EXTENSIONS)]
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if not matches:
        raise FileNotFoundError(f"No paired mask found for {stem.as_posix()} under {directory}")
    if len(matches) > 1:
        raise ValueError(f"Multiple paired masks found for {stem.as_posix()}: {matches}")
    return matches[0]


def _doctamper_extracted_records(
    source: Path,
    *,
    benchmark: str,
    limit: int | None,
) -> list[ManifestRecord]:
    image_dir = source / "images"
    mask_dir = source / ("labels" if (source / "labels").is_dir() else "masks")
    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    records: list[ManifestRecord] = []
    for image_path in paths:
        relative = image_path.relative_to(image_dir)
        mask_path = _paired_file(mask_dir, relative.with_suffix(""))
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Image/mask size mismatch: {image_path} vs {mask_path}")
            label = int(_mask_array(mask).any())
            source_group = _doctamper_source_group(image, mask)
        token = relative.with_suffix("").as_posix().replace("/", "-")
        sample_id = f"{benchmark}-{token}"
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                image=str(image_path.resolve()),
                mask=str(mask_path.resolve()),
                split="test",
                label=label,
                source_group=source_group,
                dataset="doctamper",
                benchmark=benchmark,
                classification_supervised=False,
                localization_supervised=True,
                adn_supervision="proxy",
                tamper_type="text_tampering" if label else "none",
            )
        )
    if not records:
        raise FileNotFoundError(f"No extracted DocTamper images found under {image_dir}")
    return records


def prepare_doctamper(
    root: Path,
    prepared_root: Path,
    *,
    seed: int,
    validation_fraction: float,
    limit: int | None,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for directory_name, benchmark in DOCTAMPER_SUBSETS.items():
        source = root / directory_name
        if not source.is_dir():
            raise FileNotFoundError(f"Required DocTamper subset is missing: {source}")
        if (source / "data.mdb").is_file():
            subset = _doctamper_lmdb_records(
                source, prepared_root / directory_name, benchmark=benchmark, limit=limit
            )
        else:
            subset = _doctamper_extracted_records(source, benchmark=benchmark, limit=limit)
        if benchmark == "doctamper-training":
            subset = _split_groups(
                subset,
                seed=seed,
                validation_fraction=validation_fraction,
                test_fraction=0.0,
            )
        records.extend(subset)
    return records


def _load_annotation(root: Path, relative: Path, image_path: Path) -> dict[str, Any]:
    candidates = [
        root / "annotations" / relative.with_suffix(".json"),
        image_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"MIDV annotation must be an object: {candidate}")
            return value
    return {}


def _midv_source_group(relative: Path, annotation: dict[str, Any]) -> str:
    base_image = annotation.get("base_image")
    if isinstance(base_image, str) and base_image.strip():
        path = PurePosixPath(base_image.replace("\\", "/"))
        parts = list(path.parts)
        if "images" in parts:
            parts = parts[parts.index("images") + 1 :]
        # Remove manipulation category when present; document type + stem stay.
        if parts and parts[0] in {
            "authentic",
            "copy_move",
            "foreign_object",
            "gluing",
            "information_masking",
            "splicing_document",
            "splicing_photo",
            "splicing_symbol",
            "text_overlay",
        }:
            parts = parts[1:]
        return "midv:" + PurePosixPath(*parts).with_suffix("").as_posix()
    parts = relative.parts
    without_category = Path(*parts[1:]) if len(parts) > 1 else relative
    return "midv:" + without_category.with_suffix("").as_posix()


def prepare_midv(
    root: Path,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    limit: int | None,
) -> list[ManifestRecord]:
    image_root = root / "images"
    mask_root = root / "masks"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"MIDV root must contain images/ and masks/ directories: {root}")
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    records: list[ManifestRecord] = []
    for image_path in image_paths:
        relative = image_path.relative_to(image_root)
        if len(relative.parts) < 2:
            raise ValueError(f"MIDV image lacks category/document structure: {image_path}")
        category = relative.parts[0].lower()
        mask_path = _paired_file(mask_root, relative.with_suffix(""))
        annotation = _load_annotation(root, relative, image_path)
        declared = str(annotation.get("forgery_type", category)).lower()
        if "forgery_type" in annotation and (
            (category == "authentic") != (declared == "authentic")
        ):
            raise ValueError(f"MIDV category and annotation disagree: {image_path}")
        authentic = category == "authentic" or declared == "authentic"
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Image/mask size mismatch: {image_path} vs {mask_path}")
            has_mask = bool(_mask_array(mask).any())
        if authentic and has_mask:
            raise ValueError(f"Authentic MIDV mask is non-empty: {mask_path}")
        if not authentic and not has_mask:
            raise ValueError(f"Forged MIDV mask is empty: {mask_path}")
        sample_token = relative.with_suffix("").as_posix().replace("/", "-")
        records.append(
            ManifestRecord(
                sample_id=f"midv-{sample_token}",
                image=str(image_path.resolve()),
                mask=str(mask_path.resolve()),
                split="train",
                label=0 if authentic else 1,
                source_group=_midv_source_group(relative, annotation),
                dataset="midv",
                benchmark="midv-dm",
                classification_supervised=True,
                localization_supervised=True,
                adn_supervision="proxy",
                tamper_type="none" if authentic else declared,
            )
        )
    if not records:
        raise FileNotFoundError(f"No MIDV images found under {image_root}")
    return _split_groups(
        records,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--doctamper-root", type=Path)
    parser.add_argument("--midv-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--processed-root",
        type=Path,
        help="DocTamper LMDB export directory (default: output/processed/<profile>/doctamper)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--midv-test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--limit-per-subset",
        type=int,
        default=None,
        help="Debug-only cap; omitted by the CUDA full profile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in [0,1)")
    if not 0.0 <= args.midv_test_fraction < 1.0:
        raise ValueError("midv-test-fraction must be in [0,1)")
    if args.validation_fraction + args.midv_test_fraction >= 1.0:
        raise ValueError("MIDV validation and test fractions must leave training data")
    default_doc, default_midv, default_output = PROFILE_DEFAULTS[args.profile]
    doctamper_root = (args.doctamper_root or default_doc).resolve()
    midv_root = (args.midv_root or default_midv).resolve()
    output = (args.output or default_output).resolve()
    prepared = (
        args.processed_root or DEFAULT_PROCESSED_ROOT / args.profile / "doctamper"
    ).resolve()
    records = prepare_doctamper(
        doctamper_root,
        prepared,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        limit=args.limit_per_subset,
    )
    records.extend(
        prepare_midv(
            midv_root,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.midv_test_fraction,
            limit=args.limit_per_subset,
        )
    )
    records.sort(key=lambda record: (record.split, record.dataset, record.sample_id))
    validate_manifest_protocol(records)
    write_manifest(records, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = summarize_manifest(records)
    summary.update(
        {
            "status": "ok",
            "profile": args.profile,
            "manifest": str(output),
            "sha256": digest,
            "protocol": {
                "doctamper_training": "train/validation only",
                "doctamper_testing_fcd_scd": "test only",
                "doctamper_classification": "disabled (positive-only release)",
                "doctamper_grouping": "masked perceptual proxy; audit provenance",
                "midv": "source-group-disjoint train/validation/test",
            },
        }
    )
    summary_path = output.with_name(f"{output.stem}.summary.json")
    summary["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
