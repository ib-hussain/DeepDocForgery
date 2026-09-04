"""DCT feature extraction for the DeepDocForgery input layer.

The downstream interface intentionally exposes learned dense feature maps only.
Internally, JPEG files can use exact quantized coefficients supplied by
``jpegio``; decoded RGB tensors always provide a differentiable pixel-domain
fallback. When both representations are available, a feature-space consistency
loss teaches the fallback to agree with the exact-coefficient branch.

The quantization-table handling and frequency-pyramid design are adaptations of
ideas used by ADCD-Net (MIT; notice in ``third_party/ADCD-Net-LICENSE``). The
implementation here is reorganized for three YCbCr channels, exact/fallback
agreement training, arbitrary image sizes, and later feature fusion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm divisor no greater than ``maximum``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Convolution followed by batch-size-independent normalization and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Compact residual refinement block used at every pyramid scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


@dataclass
class JPEGMetadata:
    """Batchable JPEG facts that are safe to provide to the estimator.

    ``qtables`` has shape ``[B, 3, 8, 8]`` in Y, Cb, Cr order.
    ``qtable_valid`` has shape ``[B, 3]``. ``subsampling`` is a four-way one-hot
    vector for 4:4:4, 4:2:2, 4:2:0, and other/unknown. Target labels such as
    double-compression status are deliberately not stored here, preventing
    supervision leakage into the estimator.
    """

    qtables: Tensor
    qtable_valid: Tensor
    is_jpeg: Tensor
    subsampling: Tensor

    def validate(self) -> None:
        if self.qtables.ndim != 4 or self.qtables.shape[1:] != (3, 8, 8):
            raise ValueError("qtables must have shape [B, 3, 8, 8]")
        batch = self.qtables.shape[0]
        if self.qtable_valid.shape != (batch, 3):
            raise ValueError("qtable_valid must have shape [B, 3]")
        if self.is_jpeg.shape != (batch, 1):
            raise ValueError("is_jpeg must have shape [B, 1]")
        if self.subsampling.shape != (batch, 4):
            raise ValueError("subsampling must have shape [B, 4]")

    @property
    def batch_size(self) -> int:
        return int(self.qtables.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> JPEGMetadata:
        """Return a metadata batch with every tensor moved together."""

        return JPEGMetadata(
            qtables=self.qtables.to(*args, **kwargs),
            qtable_valid=self.qtable_valid.to(*args, **kwargs),
            is_jpeg=self.is_jpeg.to(*args, **kwargs),
            subsampling=self.subsampling.to(*args, **kwargs),
        )

    @classmethod
    def for_non_jpeg(
        cls,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> JPEGMetadata:
        """Construct neutral metadata for PNG, TIFF, screenshots, or raw tensors."""

        qtables = torch.ones(batch_size, 3, 8, 8, device=device, dtype=dtype)
        valid = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        is_jpeg = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        subsampling = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        subsampling[:, 3] = 1.0
        return cls(qtables, valid, is_jpeg, subsampling)


@dataclass
class ExactDCTBatch:
    """Exact quantized JPEG coefficients aligned to the luma block grid."""

    coefficients: Tensor
    metadata: JPEGMetadata
    valid: Tensor | None = None

    def validate(self) -> None:
        if self.coefficients.ndim != 5 or self.coefficients.shape[1:3] != (3, 64):
            raise ValueError("coefficients must have shape [B, 3, 64, Hb, Wb]")
        self.metadata.validate()
        if self.coefficients.shape[0] != self.metadata.batch_size:
            raise ValueError("coefficient and metadata batch sizes do not match")
        if self.valid is not None and self.valid.shape != (self.metadata.batch_size,):
            raise ValueError("valid must have shape [B]")

    def to(self, *args: Any, **kwargs: Any) -> ExactDCTBatch:
        return ExactDCTBatch(
            coefficients=self.coefficients.to(*args, **kwargs),
            metadata=self.metadata.to(*args, **kwargs),
            valid=None if self.valid is None else self.valid.to(*args, **kwargs),
        )


class ExactJPEGDCTReader:
    """Read exact JPEG coefficients with the optional ``jpegio`` package.

    This belongs in the data-loading process, not inside ``nn.Module.forward``.
    It returns one sample per call because exact coefficients must receive the
    same block-aligned crop as the decoded RGB image. Arbitrary resizing would
    destroy their exact correspondence.
    """

    @staticmethod
    def _import_jpegio() -> Any:
        try:
            import jpegio  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Exact JPEG coefficients require the optional 'jpegio' package. "
                "Install the project with: pip install -e '.[exact-jpeg]'"
            ) from exc
        return jpegio

    @staticmethod
    def _blockify(array: Any) -> Tensor:
        values = torch.as_tensor(array, dtype=torch.float32)
        height, width = values.shape
        if height % 8 or width % 8:
            raise ValueError("JPEG coefficient arrays must be divisible into 8x8 blocks")
        return (
            values.reshape(height // 8, 8, width // 8, 8)
            .permute(1, 3, 0, 2)
            .reshape(64, height // 8, width // 8)
            .contiguous()
        )

    @staticmethod
    def _subsampling_one_hot(comp_info: Sequence[Any]) -> Tensor:
        one_hot = torch.zeros(4, dtype=torch.float32)
        if len(comp_info) < 3:
            one_hot[3] = 1.0
            return one_hot

        y_h = int(getattr(comp_info[0], "h_samp_factor", 1))
        y_v = int(getattr(comp_info[0], "v_samp_factor", 1))
        c_h = max(1, int(getattr(comp_info[1], "h_samp_factor", 1)))
        c_v = max(1, int(getattr(comp_info[1], "v_samp_factor", 1)))
        ratio = (y_h // c_h, y_v // c_v)
        index = {(1, 1): 0, (2, 1): 1, (2, 2): 2}.get(ratio, 3)
        one_hot[index] = 1.0
        return one_hot

    def read(self, path: str | Path) -> ExactDCTBatch:
        jpegio = self._import_jpegio()
        jpeg = jpegio.read(str(Path(path)))
        arrays = list(jpeg.coef_arrays)
        if not arrays:
            raise ValueError(f"No JPEG coefficient arrays found in {path}")

        component_features = [self._blockify(array) for array in arrays[:3]]
        target_hw = component_features[0].shape[-2:]
        aligned: list[Tensor] = []
        for component in component_features:
            if component.shape[-2:] != target_hw:
                component = F.interpolate(
                    component.unsqueeze(0), size=target_hw, mode="nearest"
                ).squeeze(0)
            aligned.append(component)

        valid = torch.zeros(3, dtype=torch.float32)
        valid[: len(aligned)] = 1.0
        while len(aligned) < 3:
            aligned.append(torch.zeros_like(aligned[0]))
        coefficients = torch.stack(aligned[:3], dim=0).unsqueeze(0)

        qtables = torch.ones(3, 8, 8, dtype=torch.float32)
        raw_qtables = list(jpeg.quant_tables)
        comp_info = list(getattr(jpeg, "comp_info", []))
        for component_index in range(min(3, len(arrays))):
            default_table = 0 if component_index == 0 else min(1, len(raw_qtables) - 1)
            table_index = default_table
            if component_index < len(comp_info):
                table_index = int(
                    getattr(comp_info[component_index], "quant_tbl_no", default_table)
                )
            if 0 <= table_index < len(raw_qtables):
                qtables[component_index] = torch.as_tensor(
                    raw_qtables[table_index], dtype=torch.float32
                ).reshape(8, 8)

        metadata = JPEGMetadata(
            qtables=qtables.unsqueeze(0),
            qtable_valid=valid.unsqueeze(0),
            is_jpeg=torch.ones(1, 1),
            subsampling=self._subsampling_one_hot(comp_info).unsqueeze(0),
        )
        result = ExactDCTBatch(
            coefficients=coefficients,
            metadata=metadata,
            valid=torch.ones(1, dtype=torch.float32),
        )
        result.validate()
        return result


class PixelDomainDCT(nn.Module):
    """Differentiable 8x8 orthonormal DCT over all YCbCr channels."""

    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        if block_size != 8:
            raise ValueError("JPEG-compatible processing currently requires block_size=8")
        self.block_size = block_size
        self.register_buffer("dct_matrix", self._make_dct_matrix(block_size))

    @staticmethod
    def _make_dct_matrix(size: int) -> Tensor:
        matrix = torch.empty(size, size, dtype=torch.float32)
        for frequency in range(size):
            alpha = math.sqrt(1.0 / size) if frequency == 0 else math.sqrt(2.0 / size)
            for sample in range(size):
                matrix[frequency, sample] = alpha * math.cos(
                    math.pi * (2 * sample + 1) * frequency / (2 * size)
                )
        return matrix

    @staticmethod
    def rgb_to_ycbcr(rgb: Tensor) -> Tensor:
        """Convert normalized RGB to full-range normalized JPEG YCbCr."""

        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        red, green, blue = rgb.unbind(dim=1)
        y = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
        cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
        return torch.stack((y, cb, cr), dim=1)

    def _channel_dct(self, channel: Tensor) -> Tensor:
        """Transform ``[1,1,H,W]`` into ``[1,64,Hb,Wb]`` blocks."""

        height, width = channel.shape[-2:]
        pad_h = (-height) % self.block_size
        pad_w = (-width) % self.block_size
        if pad_h or pad_w:
            channel = F.pad(channel, (0, pad_w, 0, pad_h), mode="replicate")
        blocks = channel.unfold(2, self.block_size, self.block_size).unfold(
            3, self.block_size, self.block_size
        )
        matrix = self.dct_matrix.to(device=channel.device, dtype=channel.dtype)
        coefficients = torch.einsum("ki,bchwij,lj->bchwkl", matrix, blocks, matrix)
        block_h, block_w = coefficients.shape[2:4]
        return coefficients.permute(0, 1, 4, 5, 2, 3).reshape(1, 64, block_h, block_w).contiguous()

    @staticmethod
    def _chroma_pooling(subsampling_row: Tensor | None) -> tuple[int, int]:
        if subsampling_row is None or float(subsampling_row.sum()) <= 0.0:
            return (1, 1)
        mode = int(subsampling_row.argmax())
        # Metadata order: 4:4:4, 4:2:2, 4:2:0, other/unknown.
        return {0: (1, 1), 1: (1, 2), 2: (2, 2)}.get(mode, (1, 1))

    def forward(
        self,
        rgb: Tensor,
        qtables: Tensor | None = None,
        subsampling: Tensor | None = None,
    ) -> Tensor:
        """Return continuous quantized coefficients as ``[B,3,64,Hb,Wb]``.

        Division by the final JPEG quantization tables places the fallback in
        the same approximate domain as exact integer coefficients. Rounding is
        intentionally omitted so gradients remain useful.
        """

        if not rgb.is_floating_point():
            raise TypeError("rgb must be a floating-point tensor normalized to [0, 1]")
        batch = rgb.shape[0]
        ycbcr = self.rgb_to_ycbcr(rgb)
        centered = ycbcr * 255.0 - 128.0
        if subsampling is not None and subsampling.shape != (batch, 4):
            raise ValueError("subsampling must have shape [B, 4]")

        # JPEG chroma planes can have fewer coefficient blocks than luma. Match
        # that process before DCT, then replicate chroma blocks to the luma grid
        # just as ExactJPEGDCTReader does.
        samples: list[Tensor] = []
        for batch_index in range(batch):
            y_coefficients = self._channel_dct(centered[batch_index : batch_index + 1, 0:1])
            target_size = y_coefficients.shape[-2:]
            sample_coefficients = [y_coefficients]
            pooling = self._chroma_pooling(
                None if subsampling is None else subsampling[batch_index]
            )
            for component_index in (1, 2):
                component = centered[
                    batch_index : batch_index + 1,
                    component_index : component_index + 1,
                ]
                if pooling != (1, 1):
                    component = F.avg_pool2d(
                        component,
                        kernel_size=pooling,
                        stride=pooling,
                        ceil_mode=True,
                    )
                component_coefficients = self._channel_dct(component)
                if component_coefficients.shape[-2:] != target_size:
                    component_coefficients = F.interpolate(
                        component_coefficients,
                        size=target_size,
                        mode="nearest",
                    )
                sample_coefficients.append(component_coefficients)
            samples.append(torch.stack(sample_coefficients, dim=1))
        coefficients = torch.cat(samples, dim=0)

        if qtables is None:
            qtables = torch.ones(batch, 3, 8, 8, device=rgb.device, dtype=rgb.dtype)
        if qtables.shape != (batch, 3, 8, 8):
            raise ValueError("qtables must have shape [B, 3, 8, 8]")
        divisors = qtables.to(device=rgb.device, dtype=rgb.dtype).reshape(batch, 3, 64, 1, 1)
        return coefficients / divisors.clamp_min(1.0)


class _ColourFrequencyStem(nn.Module):
    """A shallow, channel-specific encoder for one YCbCr component."""

    def __init__(self, stem_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(128, stem_channels, 1),
            ResidualBlock(stem_channels),
        )

    def forward(self, coefficients: Tensor, qtable: Tensor) -> Tensor:
        coefficient_scale = math.log1p(2048.0)
        normalized_coefficients = (
            torch.sign(coefficients) * torch.log1p(coefficients.abs()) / coefficient_scale
        )
        normalized_qtable = torch.log1p(qtable.clamp_min(1.0)) / math.log(256.0)
        qtable_map = normalized_qtable.expand(-1, -1, *coefficients.shape[-2:])
        return self.layers(torch.cat((normalized_coefficients, qtable_map), dim=1))


@dataclass
class DCTBranchOutput:
    """Dense maps exposed to fusion plus the weighted agreement objective."""

    features: dict[str, Tensor]
    consistency_loss: Tensor
    source: str


class DCTFeaturePyramid(nn.Module):
    """Separate Y/Cb/Cr stems, learned fusion, and a configurable pyramid.

    The first output is on the native 8x8 JPEG-block grid (stride 8 relative to
    the image). Every subsequent level downsamples by two. Default levels are
    therefore compatible with common CNN/ViT fusion strides 8, 16, 32, and 64.
    """

    def __init__(
        self,
        *,
        stem_channels: int = 32,
        out_channels: Sequence[int] = (64, 128, 256, 384),
        level_names: Sequence[str] = ("s8", "s16", "s32", "s64"),
        consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not out_channels or len(out_channels) != len(level_names):
            raise ValueError("out_channels and level_names must be non-empty and equal length")
        if consistency_weight < 0:
            raise ValueError("consistency_weight cannot be negative")

        self.out_channels = tuple(int(value) for value in out_channels)
        self.level_names = tuple(str(value) for value in level_names)
        self.consistency_weight = float(consistency_weight)
        self.pixel_dct = PixelDomainDCT()
        self.colour_stems = nn.ModuleList([_ColourFrequencyStem(stem_channels) for _ in range(3)])
        self.initial_fusion = nn.Sequential(
            ConvNormAct(3 * stem_channels, self.out_channels[0], 1),
            ResidualBlock(self.out_channels[0]),
        )
        transitions: list[nn.Module] = []
        for in_channels, next_channels in zip(
            self.out_channels, self.out_channels[1:], strict=False
        ):
            transitions.append(
                nn.Sequential(
                    ConvNormAct(in_channels, next_channels, 3, stride=2),
                    ResidualBlock(next_channels),
                )
            )
        self.transitions = nn.ModuleList(transitions)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DCTFeaturePyramid:
        return cls(
            stem_channels=int(config.get("stem_channels", 32)),
            out_channels=tuple(config.get("out_channels", (64, 128, 256, 384))),
            level_names=tuple(config.get("level_names", ("s8", "s16", "s32", "s64"))),
            consistency_weight=float(config.get("consistency_weight", 0.1)),
        )

    def _encode(self, coefficients: Tensor, qtables: Tensor) -> dict[str, Tensor]:
        if coefficients.ndim != 5 or coefficients.shape[1:3] != (3, 64):
            raise ValueError("coefficients must have shape [B, 3, 64, Hb, Wb]")
        batch = coefficients.shape[0]
        if qtables.shape != (batch, 3, 8, 8):
            raise ValueError("qtables must have shape [B, 3, 8, 8]")

        stems: list[Tensor] = []
        for index, stem in enumerate(self.colour_stems):
            table = qtables[:, index].reshape(batch, 64, 1, 1)
            stems.append(stem(coefficients[:, index], table))
        current = self.initial_fusion(torch.cat(stems, dim=1))
        features = {self.level_names[0]: current}
        for name, transition in zip(self.level_names[1:], self.transitions, strict=True):
            current = transition(current)
            features[name] = current
        return features

    @staticmethod
    def _agreement(exact: Mapping[str, Tensor], fallback: Mapping[str, Tensor]) -> Tensor:
        losses: list[Tensor] = []
        for name, exact_feature in exact.items():
            fallback_feature = fallback[name]
            teacher = F.normalize(exact_feature, dim=1).detach()
            student = F.normalize(fallback_feature, dim=1)
            losses.append(F.smooth_l1_loss(student, teacher))
        return torch.stack(losses).mean()

    def forward(
        self,
        rgb: Tensor,
        *,
        metadata: JPEGMetadata | None = None,
        exact_dct: ExactDCTBatch | None = None,
        compute_consistency: bool | None = None,
    ) -> DCTBranchOutput:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        if metadata is None:
            metadata = (
                exact_dct.metadata
                if exact_dct is not None
                else JPEGMetadata.for_non_jpeg(rgb.shape[0], device=rgb.device, dtype=rgb.dtype)
            )
        metadata = metadata.to(device=rgb.device, dtype=rgb.dtype)
        metadata.validate()
        if metadata.batch_size != rgb.shape[0]:
            raise ValueError("RGB and metadata batch sizes do not match")

        if compute_consistency is None:
            compute_consistency = self.training and exact_dct is not None

        if exact_dct is None:
            fallback_coefficients = self.pixel_dct(rgb, metadata.qtables, metadata.subsampling)
            features = self._encode(fallback_coefficients, metadata.qtables)
            return DCTBranchOutput(features, rgb.new_zeros(()), "pixel_fallback")

        exact_dct.validate()
        exact_coefficients = exact_dct.coefficients.to(device=rgb.device, dtype=rgb.dtype)
        fallback_coefficients = self.pixel_dct(rgb, metadata.qtables, metadata.subsampling)
        if fallback_coefficients.shape != exact_coefficients.shape:
            raise ValueError(
                "Exact and pixel DCT grids differ. Exact coefficients are only valid "
                "for an unchanged, 8-pixel-aligned image geometry."
            )
        valid = (
            torch.ones(rgb.shape[0], device=rgb.device, dtype=rgb.dtype)
            if exact_dct.valid is None
            else exact_dct.valid.to(device=rgb.device, dtype=rgb.dtype)
        )
        mixed_coefficients = torch.where(
            valid[:, None, None, None, None].bool(),
            exact_coefficients,
            fallback_coefficients,
        )
        exact_features = self._encode(mixed_coefficients, metadata.qtables)
        consistency_loss = rgb.new_zeros(())
        if compute_consistency and bool(valid.sum() > 0):
            fallback_features = self._encode(fallback_coefficients, metadata.qtables)
            exact_only = {name: value[valid.bool()] for name, value in exact_features.items()}
            fallback_only = {name: value[valid.bool()] for name, value in fallback_features.items()}
            consistency_loss = self.consistency_weight * self._agreement(exact_only, fallback_only)
        source = "exact_jpeg" if bool(valid.all()) else "mixed_exact_and_fallback"
        return DCTBranchOutput(exact_features, consistency_loss, source)
