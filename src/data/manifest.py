"""Portable JSONL manifests and paired image/mask loading.

Every derivative of one source document should share ``source_group``. Splits
are checked at this group boundary, which prevents an authentic document from
appearing in training while a forged derivative of it appears in evaluation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.inputLayer.dataModelling import read_jpeg_metadata, stack_metadata
from src.inputLayer.degradationEstimator import DegradationTargets
from src.inputLayer.freqFeatures import JPEGMetadata
from src.training.objectives import DeepDocForgerySupervision

VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    image: str
    split: str
    label: int
    source_group: str
    dataset: str = "unknown"
    mask: str | None = None
    tamper_type: str | None = None
    jpeg_quality: float | None = None
    double_compression: bool | None = None
    noise_type: str | int | None = None
    noise_strength: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, line_number: int) -> ManifestRecord:
        required = ("sample_id", "image", "split", "label")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"Manifest line {line_number} is missing {missing}")
        split = str(value["split"])
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Manifest line {line_number} has split {split!r}; expected {VALID_SPLITS}"
            )
        label = int(value["label"])
        if label not in (0, 1):
            raise ValueError(f"Manifest line {line_number} label must be 0 or 1")
        sample_id = str(value["sample_id"])
        return cls(
            sample_id=sample_id,
            image=str(value["image"]),
            split=split,
            label=label,
            source_group=str(value.get("source_group", sample_id)),
            dataset=str(value.get("dataset", "unknown")),
            mask=None if value.get("mask") is None else str(value["mask"]),
            tamper_type=(
                None if value.get("tamper_type") is None else str(value["tamper_type"])
            ),
            jpeg_quality=(
                None if value.get("jpeg_quality") is None else float(value["jpeg_quality"])
            ),
            double_compression=(
                None
                if value.get("double_compression") is None
                else bool(value["double_compression"])
            ),
            noise_type=value.get("noise_type"),
            noise_strength=(
                None if value.get("noise_strength") is None else float(value["noise_strength"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "sample_id": self.sample_id,
                "image": self.image,
                "mask": self.mask,
                "split": self.split,
                "label": self.label,
                "source_group": self.source_group,
                "dataset": self.dataset,
                "tamper_type": self.tamper_type,
                "jpeg_quality": self.jpeg_quality,
                "double_compression": self.double_compression,
                "noise_type": self.noise_type,
                "noise_strength": self.noise_strength,
            }.items()
            if value is not None
        }


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on manifest line {line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            record = ManifestRecord.from_dict(raw, line_number=line_number)
            if record.sample_id in seen:
                raise ValueError(f"Duplicate sample_id {record.sample_id!r}")
            seen.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"Manifest {manifest_path} contains no samples")
    return records


def validate_group_disjointness(records: list[ManifestRecord]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record.source_group].add(record.split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        preview = list(sorted(leaked.items()))[:5]
        raise ValueError(f"Source groups cross dataset splits: {preview}")


def summarize_manifest(records: list[ManifestRecord]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "groups": len({record.source_group for record in records}),
        "splits": dict(Counter(record.split for record in records)),
        "labels": {
            split: dict(Counter(record.label for record in records if record.split == split))
            for split in VALID_SPLITS
        },
        "datasets": dict(Counter(record.dataset for record in records)),
        "mask_supervised": sum(record.mask is not None for record in records),
        "degradation_supervised": sum(
            record.jpeg_quality is not None
            or record.noise_type is not None
            or record.noise_strength is not None
            for record in records
        ),
    }


@dataclass(frozen=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int
    target_height: int
    target_width: int


def _letterbox(
    image: Image.Image,
    mask: Image.Image | None,
    target_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, LetterboxTransform]:
    target_height, target_width = target_size
    original_width, original_height = image.size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2

    image_resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    image_canvas = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
    image_canvas.paste(image_resized, (left, top))
    image_array = np.asarray(image_canvas, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

    mask_tensor = torch.zeros(1, target_height, target_width, dtype=torch.float32)
    valid_tensor = torch.zeros_like(mask_tensor)
    valid_tensor[:, top : top + resized_height, left : left + resized_width] = 1.0
    if mask is not None:
        mask_resized = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
        mask_array = np.asarray(mask_resized, dtype=np.uint8)
        mask_tensor[:, top : top + resized_height, left : left + resized_width] = torch.from_numpy(
            (mask_array > 0).astype(np.float32)
        )
    else:
        # Classification-only samples do not silently become negative mask
        # examples. A zero validity map excludes them from localization loss.
        valid_tensor.zero_()

    transform = LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        top=top,
        left=left,
        target_height=target_height,
        target_width=target_width,
    )
    return image_tensor, mask_tensor, valid_tensor, transform


class ForgeryManifestDataset(Dataset[dict[str, Any]]):
    """Load paired images/masks while retaining JPEG container metadata."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        image_size: tuple[int, int] = (512, 512),
        noise_types: tuple[str, ...] = ("none", "gaussian", "poisson", "speckle"),
        validate_paths: bool = True,
    ) -> None:
        super().__init__()
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}")
        if len(image_size) != 2 or min(image_size) < 32:
            raise ValueError("image_size must contain two dimensions of at least 32 pixels")
        self.manifest_path = Path(manifest).resolve()
        all_records = load_manifest(self.manifest_path)
        validate_group_disjointness(all_records)
        self.records = [record for record in all_records if record.split == split]
        if not self.records:
            raise ValueError(f"Manifest has no samples in split {split!r}")
        self.root = self.manifest_path.parent
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.noise_types = tuple(noise_types)
        if validate_paths:
            missing: list[Path] = []
            for record in self.records:
                for value in (record.image, record.mask):
                    if value is not None and not self._resolve(value).is_file():
                        missing.append(self._resolve(value))
            if missing:
                raise FileNotFoundError(f"Missing dataset files: {missing[:5]}")

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __len__(self) -> int:
        return len(self.records)

    def _noise_index(self, value: str | int | None) -> int:
        if value is None:
            return 0
        if isinstance(value, int):
            if not 0 <= value < len(self.noise_types):
                raise ValueError(f"Noise class index {value} is outside the configured taxonomy")
            return value
        try:
            return self.noise_types.index(str(value))
        except ValueError as error:
            raise ValueError(
                f"Noise class {value!r} is not in configured noise_types {self.noise_types}"
            ) from error

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self._resolve(record.image)
        mask_path = None if record.mask is None else self._resolve(record.mask)
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            if mask_path is None:
                mask = None
            else:
                with Image.open(mask_path) as mask_file:
                    mask = mask_file.convert("L")
            rgb, tamper_mask, valid_mask, transform = _letterbox(
                image, mask, self.image_size
            )
        metadata = read_jpeg_metadata(image_path)
        return {
            "rgb": rgb,
            "tamper_mask": tamper_mask,
            "valid_mask": valid_mask,
            "image_label": torch.tensor([float(record.label)], dtype=torch.float32),
            "metadata": metadata,
            "record": record,
            "transform": transform,
            "noise_index": self._noise_index(record.noise_type),
        }


@dataclass
class ManifestBatch:
    rgb: Tensor
    metadata: JPEGMetadata
    supervision: DeepDocForgerySupervision
    records: tuple[ManifestRecord, ...]
    transforms: tuple[LetterboxTransform, ...]

    def to(self, *args: Any, **kwargs: Any) -> ManifestBatch:
        return ManifestBatch(
            rgb=self.rgb.to(*args, **kwargs),
            metadata=self.metadata.to(*args, **kwargs),
            supervision=self.supervision.to(*args, **kwargs),
            records=self.records,
            transforms=self.transforms,
        )


def collate_manifest_samples(samples: list[dict[str, Any]]) -> ManifestBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    rgb = torch.stack([sample["rgb"] for sample in samples])
    masks = torch.stack([sample["tamper_mask"] for sample in samples])
    valid = torch.stack([sample["valid_mask"] for sample in samples])
    labels = torch.stack([sample["image_label"] for sample in samples])
    metadata = stack_metadata(sample["metadata"] for sample in samples)
    height, width = masks.shape[-2:]

    quality = torch.full_like(masks, 100.0)
    double_compression = torch.zeros_like(masks)
    noise_type = torch.zeros(masks.shape[0], height, width, dtype=torch.long)
    noise_strength = torch.zeros_like(masks)
    jpeg_valid = torch.zeros_like(masks)
    noise_valid = torch.zeros_like(masks)
    for index, sample in enumerate(samples):
        record: ManifestRecord = sample["record"]
        if record.jpeg_quality is not None and record.double_compression is not None:
            quality[index].fill_(record.jpeg_quality)
            double_compression[index].fill_(float(record.double_compression))
            jpeg_valid[index] = valid[index]
        if record.noise_type is not None and record.noise_strength is not None:
            noise_type[index].fill_(int(sample["noise_index"]))
            noise_strength[index].fill_(record.noise_strength)
            noise_valid[index] = valid[index]

    degradation = DegradationTargets(
        jpeg_quality=quality,
        double_compression=double_compression,
        noise_type=noise_type,
        noise_strength=noise_strength,
        jpeg_valid=jpeg_valid,
        noise_valid=noise_valid,
    )
    supervision = DeepDocForgerySupervision(
        tamper_mask=masks,
        image_label=labels,
        valid_mask=valid,
        degradation=degradation,
    )
    return ManifestBatch(
        rgb=rgb,
        metadata=metadata,
        supervision=supervision,
        records=tuple(sample["record"] for sample in samples),
        transforms=tuple(sample["transform"] for sample in samples),
    )
