from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from deepdocforgery.data import (
    ManifestRecord,
    letterbox_pair,
    validate_manifest_protocol,
)
from deepdocforgery.prepare import (
    DEFAULT_PROCESSED_ROOT,
    DOCTAMPER_SUBSETS,
    PROFILE_DEFAULTS,
    _doctamper_source_group,
    prepare_doctamper,
    prepare_midv,
)


def test_generated_manifest_defaults_live_under_output() -> None:
    assert PROFILE_DEFAULTS["cpu"][2] == Path("output/manifests/cpu.jsonl")
    assert PROFILE_DEFAULTS["cuda"][2] == Path("output/manifests/cuda.jsonl")
    assert DEFAULT_PROCESSED_ROOT == Path("output/processed")


def _write_midv_authentic_fixture(root: Path) -> None:
    image_dir = root / "images" / "authentic" / "alb_id"
    mask_dir = root / "masks" / "authentic" / "alb_id"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (2268, 4032), "white").save(image_dir / "00.jpg", quality=80)
    # MIDV authentic masks can be opaque RGBA files whose RGB channels are all zero.
    Image.new("RGBA", (2268, 4032), (0, 0, 0, 255)).save(mask_dir / "00.png")
    (image_dir / "00.json").write_text(
        json.dumps(
            {
                "base_image": "/photo/images/alb_id/00.jpg",
                "forgery_type": "authentic",
            }
        ),
        encoding="utf-8",
    )


def test_midv_2268x4032_authentic_contract_is_understood(tmp_path: Path) -> None:
    root = tmp_path / "midv"
    _write_midv_authentic_fixture(root)
    records = prepare_midv(
        root,
        seed=7,
        validation_fraction=0.15,
        test_fraction=0.15,
        limit=None,
    )
    assert len(records) == 1
    record = records[0]
    assert record.label == 0
    assert record.classification_supervised
    assert record.source_group == "midv:alb_id/00"
    with Image.open(record.image) as image, Image.open(record.mask) as mask:
        assert image.size == mask.size == (2268, 4032)
        assert np.asarray(mask)[..., :3].max() == 0


def test_midv_portrait_geometry_is_letterboxed_without_distortion() -> None:
    image = Image.new("RGB", (2268, 4032), "white")
    mask = Image.new("RGBA", image.size, (0, 0, 0, 255))
    rgb, target, valid, transform = letterbox_pair(image, mask, (512, 512), localization_valid=True)
    assert rgb.shape == (3, 512, 512)
    assert transform.resized_height == 512
    assert transform.resized_width == 288
    assert transform.left == 112
    assert float(valid.sum()) == 288 * 512
    assert float(target.sum()) == 0.0


def test_doctamper_group_proxy_ignores_masked_edits() -> None:
    first = Image.new("RGB", (64, 64), "white")
    second = first.copy()
    first_mask = Image.new("L", first.size, 0)
    second_mask = Image.new("L", second.size, 0)
    ImageDraw.Draw(first).rectangle((4, 4, 15, 15), fill="black")
    ImageDraw.Draw(first_mask).rectangle((4, 4, 15, 15), fill=255)
    ImageDraw.Draw(second).rectangle((40, 40, 55, 55), fill="black")
    ImageDraw.Draw(second_mask).rectangle((40, 40, 55, 55), fill=255)
    assert _doctamper_source_group(first, first_mask) == _doctamper_source_group(
        second, second_mask
    )


def test_midv_base_image_keeps_derivatives_in_one_split(tmp_path: Path) -> None:
    for category, forged in (("authentic", False), ("text_overlay", True)):
        for directory in ("images", "masks", "annotations"):
            (tmp_path / directory / category / "alb_id").mkdir(parents=True)
        image = Image.new("RGB", (80, 120), "white")
        mask = Image.new("L", image.size, 0)
        if forged:
            for x in range(20, 40):
                for y in range(30, 45):
                    mask.putpixel((x, y), 255)
        image.save(tmp_path / "images" / category / "alb_id" / "00.jpg")
        mask.save(tmp_path / "masks" / category / "alb_id" / "00.png")
        annotation = {
            "base_image": "/photo/images/alb_id/00.jpg",
            "forgery_type": category,
        }
        (tmp_path / "annotations" / category / "alb_id" / "00.json").write_text(
            json.dumps(annotation), encoding="utf-8"
        )
    records = prepare_midv(
        tmp_path,
        seed=7,
        validation_fraction=0.2,
        test_fraction=0.2,
        limit=None,
    )
    assert {record.label for record in records} == {0, 1}
    assert len({record.source_group for record in records}) == 1
    assert len({record.split for record in records}) == 1


def _write_extracted_doctamper_fixture(root: Path) -> None:
    for index, directory_name in enumerate(DOCTAMPER_SUBSETS):
        image_dir = root / directory_name / "images"
        label_dir = root / directory_name / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image = Image.new("RGB", (64, 64), "white")
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(image).rectangle((8 + index, 20, 24 + index, 31), fill="black")
        ImageDraw.Draw(mask).rectangle((8 + index, 20, 24 + index, 31), fill=255)
        image.save(image_dir / "000000000.jpg", quality=80)
        mask.save(label_dir / "000000000.png")


def test_doctamper_fixture_preserves_official_protocol(tmp_path: Path) -> None:
    source = tmp_path / "doctamper"
    _write_extracted_doctamper_fixture(source)
    records = prepare_doctamper(
        source,
        tmp_path / "prepared",
        seed=7,
        validation_fraction=0.34,
        limit=None,
    )
    for record in records:
        assert not record.classification_supervised
        if record.benchmark == "doctamper-training":
            assert record.split in {"train", "val"}
        else:
            assert record.split == "test"


def test_protocol_rejects_leaky_fcd_and_fake_classification() -> None:
    record = ManifestRecord(
        sample_id="bad",
        image="image.jpg",
        mask="mask.png",
        split="train",
        label=1,
        source_group="bad",
        dataset="doctamper",
        benchmark="doctamper-fcd",
        classification_supervised=True,
        localization_supervised=True,
    )
    with pytest.raises(ValueError, match="classification"):
        validate_manifest_protocol([record])


def test_non_authentic_midv_empty_mask_fails(tmp_path: Path) -> None:
    for directory in ("images", "masks"):
        (tmp_path / directory / "copy_move" / "alb_id").mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(
        tmp_path / "images" / "copy_move" / "alb_id" / "00.jpg"
    )
    Image.new("L", (32, 32), 0).save(tmp_path / "masks" / "copy_move" / "alb_id" / "00.png")
    with pytest.raises(ValueError, match="mask is empty"):
        prepare_midv(
            tmp_path,
            seed=7,
            validation_fraction=0.1,
            test_fraction=0.1,
            limit=None,
        )
