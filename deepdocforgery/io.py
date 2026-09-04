"""Data helpers for the DCT/degradation-estimator milestone.

This file keeps JPEG container parsing out of model forward passes and supplies
a deterministic synthetic batch for shape/backpropagation smoke tests. The
synthetic generator validates plumbing only; it is not a scientific training
dataset or a replacement for Real-ESRGAN-style compound degradation synthesis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from .degradation import DegradationTargets
from .frequency import JPEGMetadata

# ITU-T T.81 / libjpeg baseline tables in natural row-major order.
_LUMA_QUANTIZATION_TABLE = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
_CHROMA_QUANTIZATION_TABLE = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous non-mapping roots."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    return value


def jpeg_quantization_tables(
    quality: Tensor | Sequence[float] | float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Approximate libjpeg Y/Cb/Cr tables for quality values in ``[1,100]``."""

    values = torch.as_tensor(quality, device=device, dtype=dtype).flatten().clamp(1.0, 100.0)
    scale = torch.where(values < 50.0, 5000.0 / values, 200.0 - 2.0 * values)
    base = torch.stack(
        (_LUMA_QUANTIZATION_TABLE, _CHROMA_QUANTIZATION_TABLE, _CHROMA_QUANTIZATION_TABLE)
    ).to(device=values.device, dtype=dtype)
    tables = torch.floor((base.unsqueeze(0) * scale[:, None, None, None] + 50.0) / 100.0)
    return tables.clamp(1.0, 255.0)


def estimate_quality_from_qtables(qtables: Tensor) -> Tensor:
    """Estimate the closest libjpeg quality by exhaustive table matching."""

    if qtables.ndim != 4 or qtables.shape[1:] != (3, 8, 8):
        raise ValueError("qtables must have shape [B, 3, 8, 8]")
    candidates = jpeg_quantization_tables(
        torch.arange(1, 101, device=qtables.device, dtype=qtables.dtype),
        device=qtables.device,
        dtype=qtables.dtype,
    )
    distance = (qtables[:, None] - candidates[None]).abs().mean(dim=(2, 3, 4))
    return distance.argmin(dim=1).to(qtables.dtype) + 1.0


def _subsampling_from_pillow_layers(layers: Sequence[Any]) -> Tensor:
    one_hot = torch.zeros(4, dtype=torch.float32)
    if len(layers) < 3:
        one_hot[3] = 1.0
        return one_hot
    y_h, y_v = int(layers[0][1]), int(layers[0][2])
    c_h, c_v = max(1, int(layers[1][1])), max(1, int(layers[1][2]))
    ratio = (y_h // c_h, y_v // c_v)
    one_hot[{(1, 1): 0, (2, 1): 1, (2, 2): 2}.get(ratio, 3)] = 1.0
    return one_hot


def read_jpeg_metadata(path: str | Path) -> JPEGMetadata:
    """Read three-component quantization and subsampling metadata with Pillow.

    Pillow already converts DQT marker values from JPEG zigzag serialization to
    natural row-major order. Reordering ``Image.quantization`` a second time is
    therefore incorrect. This is a verified extension of the table extraction
    pattern in ADCD-Net's ``get_qt.py``.
    """

    with Image.open(path) as image:
        if image.format != "JPEG" or not getattr(image, "quantization", None):
            return JPEGMetadata.for_non_jpeg(1)

        raw_tables: Mapping[int, Sequence[int]] = image.quantization
        layers = list(getattr(image, "layer", []))
        tables = torch.ones(3, 8, 8, dtype=torch.float32)
        valid = torch.zeros(3, dtype=torch.float32)
        for component_index in range(min(3, max(1, len(layers)))):
            default_id = 0 if component_index == 0 else min(raw_tables)
            if component_index > 0 and 1 in raw_tables:
                default_id = 1
            table_id = (
                int(layers[component_index][3]) if component_index < len(layers) else default_id
            )
            values = raw_tables.get(table_id)
            if values is not None and len(values) == 64:
                tables[component_index] = torch.as_tensor(values, dtype=torch.float32).reshape(8, 8)
                valid[component_index] = 1.0

        return JPEGMetadata(
            qtables=tables.unsqueeze(0),
            qtable_valid=valid.unsqueeze(0),
            is_jpeg=torch.ones(1, 1),
            subsampling=_subsampling_from_pillow_layers(layers).unsqueeze(0),
        )


def stack_metadata(items: Iterable[JPEGMetadata]) -> JPEGMetadata:
    """Stack single- or multi-item metadata objects into one batch."""

    values = list(items)
    if not values:
        raise ValueError("Cannot stack an empty metadata sequence")
    for value in values:
        value.validate()
    return JPEGMetadata(
        qtables=torch.cat([value.qtables for value in values], dim=0),
        qtable_valid=torch.cat([value.qtable_valid for value in values], dim=0),
        is_jpeg=torch.cat([value.is_jpeg for value in values], dim=0),
        subsampling=torch.cat([value.subsampling for value in values], dim=0),
    )


def load_rgb(path: str | Path) -> Tensor:
    """Load one image as normalized ``[1,3,H,W]`` RGB without torchvision."""

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


@dataclass
class ShuffledPatchBatch:
    """Image/mask pair after aligned ADN patch shuffling."""

    rgb: Tensor
    artifact_mask: Tensor
    mode: str
    patch_size: int


def shuffle_artifact_patches(
    rgb: Tensor,
    artifact_mask: Tensor,
    *,
    patch_size: int = 32,
    mode: str = "internal",
    generator: torch.Generator | None = None,
) -> ShuffledPatchBatch:
    """Shuffle aligned image/mask patches for ADN non-text supervision.

    ``internal`` permutes patches independently within every image, disrupting
    global semantics. ``external`` permutes the complete batch-wide patch pool,
    making every patch an independently routed training example.  The same
    permutation is applied to the artifact mask so labels never drift.
    """

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B, 3, H, W]")
    if artifact_mask.ndim == 3:
        artifact_mask = artifact_mask.unsqueeze(1)
    if artifact_mask.ndim != 4 or artifact_mask.shape[1] != 1:
        raise ValueError("artifact_mask must have shape [B,1,H,W] or [B,H,W]")
    if rgb.shape[0] != artifact_mask.shape[0] or rgb.shape[-2:] != artifact_mask.shape[-2:]:
        raise ValueError("rgb and artifact_mask batch/spatial dimensions must match")
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if mode not in {"internal", "external"}:
        raise ValueError("mode must be 'internal' or 'external'")

    height, width = rgb.shape[-2:]
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    padded_rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="replicate")
    padded_mask = F.pad(artifact_mask.float(), (0, pad_w, 0, pad_h), mode="replicate")
    grid_h = padded_rgb.shape[-2] // patch_size
    grid_w = padded_rgb.shape[-1] // patch_size

    def patchify(values: Tensor) -> Tensor:
        return (
            values.unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(values.shape[0], grid_h * grid_w, values.shape[1], patch_size, patch_size)
        )

    def unpatchify(patches: Tensor) -> Tensor:
        batch, _, channels, _, _ = patches.shape
        return (
            patches.reshape(batch, grid_h, grid_w, channels, patch_size, patch_size)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, grid_h * patch_size, grid_w * patch_size)
        )

    rgb_patches = patchify(padded_rgb)
    mask_patches = patchify(padded_mask)
    batch, number_of_patches = rgb_patches.shape[:2]
    if mode == "internal":
        permutations = [
            torch.randperm(number_of_patches, device=rgb.device, generator=generator)
            for _ in range(batch)
        ]
        rgb_patches = torch.stack(
            [rgb_patches[index, order] for index, order in enumerate(permutations)]
        )
        mask_patches = torch.stack(
            [mask_patches[index, order] for index, order in enumerate(permutations)]
        )
    else:
        order = torch.randperm(
            batch * number_of_patches,
            device=rgb.device,
            generator=generator,
        )
        rgb_patches = rgb_patches.flatten(0, 1)[order].reshape_as(rgb_patches)
        mask_patches = mask_patches.flatten(0, 1)[order].reshape_as(mask_patches)

    return ShuffledPatchBatch(
        rgb=unpatchify(rgb_patches)[:, :, :height, :width],
        artifact_mask=unpatchify(mask_patches)[:, :, :height, :width],
        mode=mode,
        patch_size=patch_size,
    )


@dataclass
class SyntheticBatch:
    rgb: Tensor
    metadata: JPEGMetadata
    targets: DegradationTargets
    tamper_mask: Tensor


def _document_canvas(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    canvas = 0.88 + 0.1 * torch.rand(
        batch_size, 3, height, width, device=device, generator=generator
    )
    for batch_index in range(batch_size):
        for line_index in range(max(4, height // 20)):
            y = int(torch.randint(3, max(4, height - 3), (1,), device=device, generator=generator))
            x = int(torch.randint(2, max(3, width // 4), (1,), device=device, generator=generator))
            line_width = int(
                torch.randint(
                    max(4, width // 4), max(5, width - x), (1,), device=device, generator=generator
                )
            )
            thickness = 1 + line_index % 2
            shade = float(torch.rand((), device=device, generator=generator).mul(0.2).add(0.05))
            canvas[batch_index, :, y : min(height, y + thickness), x : x + line_width] = shade
    return canvas.clamp(0.0, 1.0)


def make_synthetic_batch(
    *,
    batch_size: int = 2,
    height: int = 128,
    width: int = 128,
    number_of_noise_types: int = 4,
    maximum_noise_strength: float = 0.2,
    device: torch.device | str = "cpu",
    seed: int = 7,
) -> SyntheticBatch:
    """Create regional labels and visibly correlated toy degradations.

    This function is only for unit/smoke testing. The JPEG proxy is block-wise
    smoothing, not an actual JPEG codec, and benchmark conclusions must never be
    drawn from it.
    """

    if batch_size < 1 or min(height, width) < 32:
        raise ValueError("Use batch_size >= 1 and spatial dimensions >= 32")
    if number_of_noise_types < 1:
        raise ValueError("number_of_noise_types must be positive")
    if maximum_noise_strength <= 0:
        raise ValueError("maximum_noise_strength must be positive")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    clean = _document_canvas(batch_size, height, width, device=resolved_device, generator=generator)

    patch_mask = torch.zeros(batch_size, 1, height, width, device=resolved_device)
    background_quality = torch.empty(batch_size, device=resolved_device).uniform_(
        75.0, 96.0, generator=generator
    )
    patch_quality = torch.empty(batch_size, device=resolved_device).uniform_(
        25.0, 70.0, generator=generator
    )
    double_flags = torch.randint(
        0, 2, (batch_size,), device=resolved_device, generator=generator
    ).float()
    noise_classes = torch.randint(
        0,
        number_of_noise_types,
        (batch_size,),
        device=resolved_device,
        generator=generator,
    )
    strength_upper = min(maximum_noise_strength * 0.8, 0.15)
    strength_lower = min(0.02, strength_upper * 0.25)
    strengths = torch.empty(batch_size, device=resolved_device).uniform_(
        strength_lower,
        strength_upper,
        generator=generator,
    )
    strengths = strengths * (noise_classes != 0).to(strengths.dtype)

    for batch_index in range(batch_size):
        patch_h = int(
            torch.randint(
                height // 4, height // 2 + 1, (1,), device=resolved_device, generator=generator
            )
        )
        patch_w = int(
            torch.randint(
                width // 4, width // 2 + 1, (1,), device=resolved_device, generator=generator
            )
        )
        top = int(
            torch.randint(
                0, height - patch_h + 1, (1,), device=resolved_device, generator=generator
            )
        )
        left = int(
            torch.randint(0, width - patch_w + 1, (1,), device=resolved_device, generator=generator)
        )
        patch_mask[batch_index, :, top : top + patch_h, left : left + patch_w] = 1.0

    quality = (
        background_quality[:, None, None, None] * (1.0 - patch_mask)
        + patch_quality[:, None, None, None] * patch_mask
    )
    double_compression = double_flags[:, None, None, None] * patch_mask
    noise_type = noise_classes[:, None, None] * patch_mask.squeeze(1).long()
    noise_strength = strengths[:, None, None, None] * patch_mask

    block_proxy = F.interpolate(
        F.avg_pool2d(clean, kernel_size=8, stride=8),
        size=(height, width),
        mode="nearest",
    )
    block_weight = ((100.0 - quality) / 100.0 * 0.65).clamp(0.0, 0.65)
    degraded = clean * (1.0 - block_weight) + block_proxy * block_weight
    second_proxy = F.interpolate(
        F.avg_pool2d(degraded, kernel_size=4, stride=4),
        size=(height, width),
        mode="nearest",
    )
    degraded = degraded * (1.0 - 0.25 * double_compression) + second_proxy * (
        0.25 * double_compression
    )

    random_noise = torch.randn(
        degraded.shape, device=resolved_device, generator=generator, dtype=degraded.dtype
    )
    for batch_index in range(batch_size):
        class_index = int(noise_classes[batch_index])
        strength = strengths[batch_index]
        mask = patch_mask[batch_index]
        if class_index == 0:
            continue
        if class_index == 2:
            # Signal-dependent approximation used only for the smoke path.
            noise = (
                random_noise[batch_index] * degraded[batch_index].clamp_min(0.05).sqrt() * strength
            )
        elif class_index == 3:
            noise = random_noise[batch_index] * degraded[batch_index] * strength
        else:
            noise = random_noise[batch_index] * strength
        degraded[batch_index] = degraded[batch_index] + noise * mask
    degraded = degraded.clamp(0.0, 1.0)

    metadata = JPEGMetadata(
        qtables=jpeg_quantization_tables(
            background_quality, device=resolved_device, dtype=degraded.dtype
        ),
        qtable_valid=torch.ones(batch_size, 3, device=resolved_device),
        is_jpeg=torch.ones(batch_size, 1, device=resolved_device),
        subsampling=F.one_hot(
            torch.full((batch_size,), 2, device=resolved_device, dtype=torch.long),
            num_classes=4,
        ).to(degraded.dtype),
    )
    valid = torch.ones(batch_size, 1, height, width, device=resolved_device)
    targets = DegradationTargets(
        jpeg_quality=quality,
        double_compression=double_compression,
        noise_type=noise_type,
        noise_strength=noise_strength,
        jpeg_valid=valid,
        noise_valid=valid,
    )
    return SyntheticBatch(
        rgb=degraded,
        metadata=metadata,
        targets=targets,
        tamper_mask=patch_mask,
    )
