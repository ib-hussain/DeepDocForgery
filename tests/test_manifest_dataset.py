from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.manifest import (
    ForgeryManifestDataset,
    collate_manifest_samples,
    load_manifest,
    summarize_manifest,
    validate_group_disjointness,
)


def _write_sample(root: Path, name: str, forged: bool) -> tuple[Path, Path]:
    image = Image.new("RGB", (53, 37), (240, 242, 245))
    mask = Image.new("L", image.size, 0)
    if forged:
        for x in range(12, 30):
            for y in range(10, 20):
                image.putpixel((x, y), (80, 80, 82))
                mask.putpixel((x, y), 255)
    image_path = root / f"{name}.jpg"
    mask_path = root / f"{name}.png"
    image.save(image_path, quality=84)
    mask.save(mask_path)
    return image_path, mask_path


def _manifest(tmp_path: Path) -> Path:
    records = []
    for index, (split, forged) in enumerate(
        (("train", False), ("train", True), ("val", False), ("test", True))
    ):
        image, mask = _write_sample(tmp_path, f"sample-{index}", forged)
        records.append(
            {
                "sample_id": f"sample-{index}",
                "image": image.name,
                "mask": mask.name,
                "split": split,
                "label": int(forged),
                "source_group": f"group-{index}",
                "dataset": "unit",
                "jpeg_quality": 84,
                "double_compression": False,
                "noise_type": "none",
                "noise_strength": 0.0,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return manifest


def test_manifest_dataset_letterboxes_and_collates_metadata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    dataset = ForgeryManifestDataset(manifest, split="train", image_size=(64, 80))
    batch = collate_manifest_samples([dataset[0], dataset[1]])
    assert batch.rgb.shape == (2, 3, 64, 80)
    assert batch.supervision.tamper_mask.shape == (2, 1, 64, 80)
    assert batch.supervision.image_label.shape == (2, 1)
    assert batch.metadata.qtables.shape == (2, 3, 8, 8)
    assert batch.supervision.degradation is not None
    assert float(batch.supervision.degradation.jpeg_valid.sum()) > 0.0
    moved = batch.to(torch.device("cpu"))
    assert moved.rgb.device.type == "cpu"


def test_manifest_summary_and_group_leakage_detection(tmp_path: Path) -> None:
    records = load_manifest(_manifest(tmp_path))
    summary = summarize_manifest(records)
    assert summary["samples"] == 4
    assert summary["splits"] == {"train": 2, "val": 1, "test": 1}
    altered = list(records)
    altered[1] = type(altered[1])(
        **{**altered[1].__dict__, "source_group": altered[0].source_group, "split": "val"}
    )
    with pytest.raises(ValueError, match="Source groups"):
        validate_group_disjointness(altered)


def test_classification_only_sample_has_no_fake_negative_mask_supervision(
    tmp_path: Path,
) -> None:
    image, _ = _write_sample(tmp_path, "classification-only", False)
    manifest = tmp_path / "classification.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "only",
                "image": image.name,
                "split": "train",
                "label": 0,
                "source_group": "only",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ForgeryManifestDataset(manifest, split="train", image_size=(64, 64))
    sample = dataset[0]
    assert float(sample["valid_mask"].sum()) == 0.0
    assert float(sample["tamper_mask"].sum()) == 0.0
