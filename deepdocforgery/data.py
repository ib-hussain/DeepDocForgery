"""Manifest contract, paired transforms, and mixed-evidence batching.

The manifest is the boundary between dataset-specific preparation and the
model.  DocTamper and MIDV-DM are intentionally normalized here without
pretending that their official evaluation protocols are interchangeable.
"""

from __future__ import annotations

import io
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from deepdocforgery.degradation import DegradationTargets
from deepdocforgery.frequency import ExactDCTBatch, ExactJPEGDCTReader, JPEGMetadata
from deepdocforgery.io import jpeg_quantization_tables, read_jpeg_metadata, stack_metadata
from deepdocforgery.objectives import DeepDocForgerySupervision
from deepdocforgery.spatial import ADNSupervision

VALID_SPLITS = ("train", "val", "test")
VALID_ADN_SUPERVISION = ("none", "proxy", "ground_truth")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ASPECT_RATIO_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    image: str
    split: str
    label: int
    source_group: str
    dataset: str = "unknown"
    benchmark: str = "default"
    mask: str | None = None
    classification_supervised: bool = True
    localization_supervised: bool = True
    adn_supervision: str = "none"
    adn_text_mask: str | None = None
    tamper_type: str | None = None
    jpeg_quality: float | None = None
    double_compression: bool | None = None
    noise_type: str | int | None = None
    noise_strength: float | None = None
    mask_scale_aligned: bool = False

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
        mask = None if value.get("mask") is None else str(value["mask"])
        classification = value.get("classification_supervised", True)
        localization = value.get("localization_supervised", mask is not None)
        if not isinstance(classification, bool) or not isinstance(localization, bool):
            raise ValueError(f"Manifest line {line_number} supervision flags must be boolean")
        adn_mode = str(value.get("adn_supervision", "none"))
        if adn_mode not in VALID_ADN_SUPERVISION:
            raise ValueError(
                f"Manifest line {line_number} adn_supervision must be one of "
                f"{VALID_ADN_SUPERVISION}"
            )
        adn_text_mask = None if value.get("adn_text_mask") is None else str(value["adn_text_mask"])
        if localization and mask is None:
            raise ValueError(f"Manifest line {line_number} enables localization without a mask")
        if adn_mode == "ground_truth" and adn_text_mask is None:
            raise ValueError(
                f"Manifest line {line_number} requires adn_text_mask for ground_truth ADN"
            )
        sample_id = str(value["sample_id"])
        source_group = str(value.get("source_group", sample_id))
        dataset = str(value.get("dataset", "unknown"))
        benchmark = str(value.get("benchmark", "default"))
        if not all(item.strip() for item in (sample_id, source_group, dataset, benchmark)):
            raise ValueError(f"Manifest line {line_number} contains an empty identifier")
        double_compression = value.get("double_compression")
        if double_compression is not None and not isinstance(double_compression, bool):
            raise ValueError(f"Manifest line {line_number} double_compression must be boolean")
        jpeg_quality = value.get("jpeg_quality")
        if jpeg_quality is not None and not 1 <= float(jpeg_quality) <= 100:
            raise ValueError(f"Manifest line {line_number} jpeg_quality must be in [1,100]")
        noise_strength = value.get("noise_strength")
        if noise_strength is not None and float(noise_strength) < 0:
            raise ValueError(f"Manifest line {line_number} noise_strength must be non-negative")
        mask_scale_aligned = value.get("mask_scale_aligned", False)
        if not isinstance(mask_scale_aligned, bool):
            raise ValueError(f"Manifest line {line_number} mask_scale_aligned must be boolean")
        return cls(
            sample_id=sample_id,
            image=str(value["image"]),
            split=split,
            label=label,
            source_group=source_group,
            dataset=dataset,
            benchmark=benchmark,
            mask=mask,
            classification_supervised=classification,
            localization_supervised=localization,
            adn_supervision=adn_mode,
            adn_text_mask=adn_text_mask,
            tamper_type=(None if value.get("tamper_type") is None else str(value["tamper_type"])),
            jpeg_quality=(None if jpeg_quality is None else float(jpeg_quality)),
            double_compression=(None if double_compression is None else double_compression),
            noise_type=value.get("noise_type"),
            noise_strength=(None if noise_strength is None else float(noise_strength)),
            mask_scale_aligned=mask_scale_aligned,
        )

    def to_dict(self) -> dict[str, Any]:
        values = {
            "sample_id": self.sample_id,
            "image": self.image,
            "mask": self.mask,
            "split": self.split,
            "label": self.label,
            "source_group": self.source_group,
            "dataset": self.dataset,
            "benchmark": self.benchmark,
            "classification_supervised": self.classification_supervised,
            "localization_supervised": self.localization_supervised,
            "adn_supervision": self.adn_supervision,
            "adn_text_mask": self.adn_text_mask,
            "tamper_type": self.tamper_type,
            "jpeg_quality": self.jpeg_quality,
            "double_compression": self.double_compression,
            "noise_type": self.noise_type,
            "noise_strength": self.noise_strength,
            "mask_scale_aligned": self.mask_scale_aligned,
        }
        return {key: value for key, value in values.items() if value is not None}


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
    validate_manifest_protocol(records)
    return records


def write_manifest(records: list[ManifestRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def validate_group_disjointness(records: list[ManifestRecord]) -> None:
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        group_splits[(record.dataset, record.source_group)].add(record.split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        preview = list(sorted(leaked.items()))[:5]
        raise ValueError(f"Source groups cross dataset splits: {preview}")


def validate_manifest_protocol(records: list[ManifestRecord]) -> None:
    """Reject the protocol mistakes that made earlier FCD results misleading."""

    validate_group_disjointness(records)
    official_tests = {"doctamper-testing", "doctamper-fcd", "doctamper-scd"}
    for record in records:
        if record.dataset == "doctamper" and record.classification_supervised:
            raise ValueError(
                f"DocTamper record {record.sample_id!r} cannot supervise image "
                "classification because the release is positive-only"
            )
        if record.benchmark in official_tests and record.split != "test":
            raise ValueError(f"Official benchmark {record.benchmark!r} must remain test-only")
        if record.benchmark == "doctamper-training" and record.split == "test":
            raise ValueError("DocTamper TrainingSet may only supply train/validation records")


def summarize_manifest(records: list[ManifestRecord]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "groups": len({(record.dataset, record.source_group) for record in records}),
        "splits": dict(Counter(record.split for record in records)),
        "labels": {
            split: dict(Counter(record.label for record in records if record.split == split))
            for split in VALID_SPLITS
        },
        "datasets": dict(Counter(record.dataset for record in records)),
        "benchmarks": dict(Counter(record.benchmark for record in records)),
        "localization_supervised": sum(record.localization_supervised for record in records),
        "classification_supervised": sum(record.classification_supervised for record in records),
        "adn_supervision": dict(Counter(record.adn_supervision for record in records)),
        "scale_aligned_masks": sum(record.mask_scale_aligned for record in records),
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


def same_aspect_ratio(
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    tolerance: float = ASPECT_RATIO_TOLERANCE,
) -> bool:
    """Return whether two ``(width, height)`` geometries differ only by scale."""

    first_width, first_height = first
    second_width, second_height = second
    if min(first_width, first_height, second_width, second_height) < 1:
        return False
    first_cross = first_width * second_height
    second_cross = second_width * first_height
    return abs(first_cross - second_cross) / max(first_cross, second_cross) <= tolerance


def align_mask_to_image(mask: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    """Scale a spatial target onto its image grid without changing geometry."""

    if mask.size == image_size:
        return mask
    if not same_aspect_ratio(mask.size, image_size):
        raise ValueError(f"Image/mask aspect-ratio mismatch: image={image_size}, mask={mask.size}")
    return mask.resize(image_size, Image.Resampling.NEAREST)


def letterbox_pair(
    image: Image.Image,
    mask: Image.Image | None,
    target_size: tuple[int, int],
    *,
    localization_valid: bool = True,
) -> tuple[Tensor, Tensor, Tensor, LetterboxTransform]:
    """Resize without distortion and mark padded pixels invalid."""

    target_height, target_width = target_size
    if min(target_height, target_width) < 1:
        raise ValueError("target_size dimensions must be positive")
    original_width, original_height = image.size
    if mask is not None and not same_aspect_ratio(image.size, mask.size):
        raise ValueError(f"Image/mask aspect-ratio mismatch: image={image.size}, mask={mask.size}")
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2

    resized = image.convert("RGB").resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    target = torch.zeros(1, target_height, target_width, dtype=torch.float32)
    valid = torch.zeros_like(target)
    if localization_valid:
        valid[:, top : top + resized_height, left : left + resized_width] = 1.0
    if mask is not None:
        resized_mask = mask.convert("L").resize(
            (resized_width, resized_height), Image.Resampling.NEAREST
        )
        mask_array = (np.asarray(resized_mask, dtype=np.uint8) > 0).astype(np.float32)
        target[:, top : top + resized_height, left : left + resized_width] = torch.from_numpy(
            mask_array
        )

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
    return rgb, target, valid, transform


# Backward-compatible internal name used by the inference module.
_letterbox = letterbox_pair


def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def _text_proxy(rgb: Tensor, valid: Tensor) -> Tensor:
    """Weak dark-stroke proxy; deliberately independent of the tamper mask."""

    gray = rgb.mean(dim=0, keepdim=True)
    local = torch.nn.functional.avg_pool2d(
        gray.unsqueeze(0), kernel_size=9, stride=1, padding=4
    ).squeeze(0)
    return ((local - gray) > 0.08).float() * valid


def _aligned_random_crop(
    image: Image.Image,
    mask: Image.Image | None,
    adn_mask: Image.Image | None,
    scale_range: tuple[float, float],
) -> tuple[Image.Image, Image.Image | None, Image.Image | None]:
    """Crop all targets together, retaining a tampered pixel when one exists."""

    width, height = image.size
    scale = random.uniform(*scale_range)
    crop_width, crop_height = max(1, round(width * scale)), max(1, round(height * scale))
    positives = None if mask is None else np.flatnonzero(np.asarray(mask) > 0)
    if positives is not None and positives.size:
        flat = int(random.choice(positives))
        y, x = divmod(flat, width)
        left = random.randint(max(0, x - crop_width + 1), min(x, width - crop_width))
        top = random.randint(max(0, y - crop_height + 1), min(y, height - crop_height))
    else:
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
    box = (left, top, left + crop_width, top + crop_height)
    return (
        image.crop(box),
        None if mask is None else mask.crop(box),
        None if adn_mask is None else adn_mask.crop(box),
    )


class ForgeryManifestDataset(Dataset[dict[str, Any]]):
    """Load either dataset through one explicit supervision contract."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        image_size: tuple[int, int] = (512, 512),
        noise_types: tuple[str, ...] = ("none", "gaussian", "poisson", "speckle"),
        augment: bool = False,
        horizontal_flip_probability: float = 0.0,
        jpeg_probability: float = 0.0,
        jpeg_quality_range: tuple[int, int] = (55, 95),
        gaussian_noise_probability: float = 0.0,
        gaussian_noise_maximum: float = 0.03,
        exact_dct_probability: float = 0.0,
        crop_probability: float = 0.0,
        crop_scale_range: tuple[float, float] = (0.6, 1.0),
        validate_paths: bool = True,
    ) -> None:
        super().__init__()
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}")
        if len(image_size) != 2 or min(image_size) < 32:
            raise ValueError("image_size must contain two dimensions of at least 32 pixels")
        for name, probability in {
            "horizontal_flip_probability": horizontal_flip_probability,
            "jpeg_probability": jpeg_probability,
            "gaussian_noise_probability": gaussian_noise_probability,
            "exact_dct_probability": exact_dct_probability,
            "crop_probability": crop_probability,
        }.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not noise_types:
            raise ValueError("noise_types must not be empty")
        if (
            len(jpeg_quality_range) != 2
            or not 1 <= jpeg_quality_range[0] <= jpeg_quality_range[1] <= 100
        ):
            raise ValueError("jpeg_quality_range must be ordered within [1,100]")
        if gaussian_noise_maximum < 0:
            raise ValueError("gaussian_noise_maximum must be non-negative")
        if len(crop_scale_range) != 2 or not 0 < crop_scale_range[0] <= crop_scale_range[1] <= 1:
            raise ValueError("crop_scale_range must be ordered within (0,1]")
        self.manifest_path = Path(manifest).resolve()
        all_records = load_manifest(self.manifest_path)
        self.records = [record for record in all_records if record.split == split]
        if not self.records:
            raise ValueError(f"Manifest has no samples in split {split!r}")
        self.root = self.manifest_path.parent
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.noise_types = tuple(noise_types)
        self.augment = bool(augment and split == "train")
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.jpeg_probability = float(jpeg_probability)
        self.jpeg_quality_range = (int(jpeg_quality_range[0]), int(jpeg_quality_range[1]))
        self.gaussian_noise_probability = float(gaussian_noise_probability)
        self.gaussian_noise_maximum = float(gaussian_noise_maximum)
        self.exact_dct_probability = float(exact_dct_probability)
        self.crop_probability = float(crop_probability)
        self.crop_scale_range = (float(crop_scale_range[0]), float(crop_scale_range[1]))
        self._exact_reader: ExactJPEGDCTReader | None = None
        if validate_paths:
            missing: list[Path] = []
            for record in self.records:
                for value in (record.image, record.mask, record.adn_text_mask):
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
        adn_path = None if record.adn_text_mask is None else self._resolve(record.adn_text_mask)
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        mask = None
        if mask_path is not None:
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L")
        adn_mask = None
        if adn_path is not None:
            with Image.open(adn_path) as adn_file:
                adn_mask = adn_file.convert("L")
        # MIDV-DM distributes some masks on a lower-resolution grid than its
        # 2268x4032 photographs. Their aspect ratios are identical, so nearest
        # neighbour scaling preserves the labelled coordinates before any
        # shared crop/flip augmentation is applied.
        if mask is not None:
            mask = align_mask_to_image(mask, image.size)
        if adn_mask is not None:
            adn_mask = align_mask_to_image(adn_mask, image.size)

        augmented = False
        jpeg_quality: float | None = record.jpeg_quality
        double_compression = record.double_compression
        noise_type = record.noise_type
        noise_strength = record.noise_strength
        if self.augment and random.random() < self.crop_probability:
            image, mask, adn_mask = _aligned_random_crop(
                image, mask, adn_mask, self.crop_scale_range
            )
            augmented = True
        if self.augment and random.random() < self.horizontal_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = None if mask is None else mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            adn_mask = (
                None if adn_mask is None else adn_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            )
            augmented = True
        if self.augment and random.random() < self.jpeg_probability:
            quality = random.randint(*self.jpeg_quality_range)
            image = _jpeg_recompress(image, quality)
            jpeg_quality = float(quality)
            double_compression = image_path.suffix.lower() in {".jpg", ".jpeg"}
            augmented = True
        if self.augment and random.random() < self.gaussian_noise_probability:
            sigma = random.random() * self.gaussian_noise_maximum
            values = np.asarray(image, dtype=np.float32) / 255.0
            values += np.random.normal(0.0, sigma, values.shape).astype(np.float32)
            image = Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8))
            noise_type = "gaussian"
            noise_strength = float(sigma)
            augmented = True

        native_geometry = image.size == (self.image_size[1], self.image_size[0])
        can_use_exact = (
            not augmented
            and native_geometry
            and image_path.suffix.lower() in {".jpg", ".jpeg"}
            and image.size[0] % 8 == 0
            and image.size[1] % 8 == 0
            and random.random() < self.exact_dct_probability
        )
        exact: ExactDCTBatch | None = None
        if can_use_exact:
            if self._exact_reader is None:
                self._exact_reader = ExactJPEGDCTReader()
            exact = self._exact_reader.read(image_path)

        rgb, tamper_mask, valid_mask, transform = letterbox_pair(
            image,
            mask,
            self.image_size,
            localization_valid=record.localization_supervised,
        )
        if record.adn_supervision == "ground_truth":
            _, adn_target, adn_valid, _ = letterbox_pair(
                image, adn_mask, self.image_size, localization_valid=True
            )
        elif record.adn_supervision == "proxy":
            adn_target = _text_proxy(rgb, valid_mask)
            adn_valid = valid_mask.clone()
        else:
            adn_target = torch.zeros_like(tamper_mask)
            adn_valid = torch.zeros_like(valid_mask)

        if jpeg_quality is not None:
            tables = jpeg_quantization_tables(float(jpeg_quality))
            metadata = JPEGMetadata(
                qtables=tables,
                qtable_valid=torch.ones(1, 3),
                is_jpeg=torch.ones(1, 1),
                subsampling=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            )
        else:
            metadata = read_jpeg_metadata(image_path)
        return {
            "rgb": rgb,
            "tamper_mask": tamper_mask,
            "valid_mask": valid_mask,
            "image_label": torch.tensor([float(record.label)], dtype=torch.float32),
            "image_valid": torch.tensor(
                [float(record.classification_supervised)], dtype=torch.float32
            ),
            "adn_target": adn_target,
            "adn_valid": adn_valid,
            "metadata": metadata,
            "exact_dct": exact,
            "record": record,
            "transform": transform,
            "noise_index": self._noise_index(noise_type),
            "jpeg_quality": jpeg_quality,
            "double_compression": double_compression,
            "noise_type": noise_type,
            "noise_strength": noise_strength,
        }


@dataclass
class ManifestBatch:
    rgb: Tensor
    metadata: JPEGMetadata
    exact_dct: ExactDCTBatch | None
    supervision: DeepDocForgerySupervision
    records: tuple[ManifestRecord, ...]
    transforms: tuple[LetterboxTransform, ...]

    def to(self, *args: Any, **kwargs: Any) -> ManifestBatch:
        return ManifestBatch(
            rgb=self.rgb.to(*args, **kwargs),
            metadata=self.metadata.to(*args, **kwargs),
            exact_dct=(None if self.exact_dct is None else self.exact_dct.to(*args, **kwargs)),
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
    image_valid = torch.stack([sample["image_valid"] for sample in samples])
    metadata = stack_metadata(sample["metadata"] for sample in samples)
    height, width = masks.shape[-2:]

    quality = torch.full_like(masks, 100.0)
    double = torch.zeros_like(masks)
    noise_class = torch.zeros(masks.shape[0], height, width, dtype=torch.long)
    noise_strength = torch.zeros_like(masks)
    jpeg_valid = torch.zeros_like(masks)
    noise_valid = torch.zeros_like(masks)
    for index, sample in enumerate(samples):
        if sample["jpeg_quality"] is not None and sample["double_compression"] is not None:
            quality[index].fill_(float(sample["jpeg_quality"]))
            double[index].fill_(float(sample["double_compression"]))
            jpeg_valid[index] = valid[index]
        if sample["noise_type"] is not None and sample["noise_strength"] is not None:
            noise_class[index].fill_(int(sample["noise_index"]))
            noise_strength[index].fill_(float(sample["noise_strength"]))
            noise_valid[index] = valid[index]
    degradation = DegradationTargets(
        jpeg_quality=quality,
        double_compression=double,
        noise_type=noise_class,
        noise_strength=noise_strength,
        jpeg_valid=jpeg_valid,
        noise_valid=noise_valid,
    )

    adn_target = torch.stack([sample["adn_target"] for sample in samples])
    adn_valid = torch.stack([sample["adn_valid"] for sample in samples])
    adn = (
        ADNSupervision(artifact_mask=adn_target, valid_mask=adn_valid)
        if bool(adn_valid.sum() > 0)
        else None
    )
    supervision = DeepDocForgerySupervision(
        tamper_mask=masks,
        image_label=labels,
        image_valid=image_valid,
        valid_mask=valid,
        degradation=degradation,
        adn=adn,
    )

    exact_items = [sample["exact_dct"] for sample in samples]
    exact_batch: ExactDCTBatch | None = None
    available = [item for item in exact_items if item is not None]
    if available:
        grid = available[0].coefficients.shape[-2:]
        if any(item.coefficients.shape[-2:] != grid for item in available):
            raise ValueError("Exact DCT samples in one batch must share a block grid")
        coefficients: list[Tensor] = []
        exact_valid: list[float] = []
        for item in exact_items:
            if item is None:
                coefficients.append(torch.zeros(3, 64, *grid))
                exact_valid.append(0.0)
            else:
                coefficients.append(item.coefficients.squeeze(0))
                exact_valid.append(1.0)
        exact_batch = ExactDCTBatch(
            coefficients=torch.stack(coefficients),
            metadata=metadata,
            valid=torch.tensor(exact_valid, dtype=torch.float32),
        )
    return ManifestBatch(
        rgb=rgb,
        metadata=metadata,
        exact_dct=exact_batch,
        supervision=supervision,
        records=tuple(sample["record"] for sample in samples),
        transforms=tuple(sample["transform"] for sample in samples),
    )
