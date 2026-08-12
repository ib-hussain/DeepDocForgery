"""Regional JPEG/noise degradation estimation for DeepDocForgery.

The estimator consumes all three agreed inputs: normalized RGB, learned DCT
feature maps, and non-label JPEG metadata. It emits configurable multi-scale
regional maps for JPEG quality, double-compression probability, noise type, and
noise strength. No global-only prediction is used in this milestone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .freqFeatures import (
    ConvNormAct,
    DCTBranchOutput,
    DCTFeaturePyramid,
    ExactDCTBatch,
    JPEGMetadata,
    ResidualBlock,
)


@dataclass
class DegradationLevelOutput:
    """Predictions for one regional scale."""

    jpeg_quality: Tensor
    double_compression_logits: Tensor
    noise_type_logits: Tensor
    noise_strength: Tensor

    def gate_tensor(self, maximum_noise_strength: float) -> Tensor:
        """Return normalized probabilities/values ready for the later gate."""

        return torch.cat(
            (
                (self.jpeg_quality - 1.0) / 99.0,
                self.double_compression_logits.sigmoid(),
                self.noise_type_logits.softmax(dim=1),
                self.noise_strength / maximum_noise_strength,
            ),
            dim=1,
        )


@dataclass
class DegradationEstimatorOutput:
    """Named regional outputs aligned with the DCT feature pyramid."""

    levels: dict[str, DegradationLevelOutput]
    noise_types: tuple[str, ...]
    maximum_noise_strength: float

    def gate_features(self) -> dict[str, Tensor]:
        return {
            name: value.gate_tensor(self.maximum_noise_strength)
            for name, value in self.levels.items()
        }


@dataclass
class DegradationTargets:
    """Full-resolution training targets; the loss resamples them per level."""

    jpeg_quality: Tensor
    double_compression: Tensor
    noise_type: Tensor
    noise_strength: Tensor
    jpeg_valid: Tensor | None = None
    noise_valid: Tensor | None = None

    def to(self, *args: Any, **kwargs: Any) -> DegradationTargets:
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        return DegradationTargets(
            jpeg_quality=self.jpeg_quality.to(*args, **kwargs),
            double_compression=self.double_compression.to(*args, **kwargs),
            noise_type=self.noise_type.to(*args, **kwargs),
            noise_strength=self.noise_strength.to(*args, **kwargs),
            jpeg_valid=move(self.jpeg_valid),
            noise_valid=move(self.noise_valid),
        )


class _RGBFeaturePyramid(nn.Module):
    """Lightweight RGB stream aligned to DCT strides 8, 16, 32, ..."""

    def __init__(self, channels: Sequence[int], level_names: Sequence[str]) -> None:
        super().__init__()
        if len(channels) != len(level_names):
            raise ValueError("RGB channels and level names must have equal length")
        first = int(channels[0])
        hidden = max(8, first // 2)
        self.level_names = tuple(level_names)
        self.stem = nn.Sequential(
            ConvNormAct(3, hidden, 3, stride=2),
            ConvNormAct(hidden, first, 3, stride=2),
            ConvNormAct(first, first, 3, stride=2),
            ResidualBlock(first),
        )
        transitions: list[nn.Module] = []
        for previous, current in zip(channels, channels[1:], strict=False):
            transitions.append(
                nn.Sequential(
                    ConvNormAct(int(previous), int(current), 3, stride=2),
                    ResidualBlock(int(current)),
                )
            )
        self.transitions = nn.ModuleList(transitions)

    def forward(self, rgb: Tensor) -> dict[str, Tensor]:
        current = self.stem(rgb)
        outputs = {self.level_names[0]: current}
        for name, transition in zip(self.level_names[1:], self.transitions, strict=True):
            current = transition(current)
            outputs[name] = current
        return outputs


class _MetadataEncoder(nn.Module):
    """Encode quantization tables and JPEG container facts without target leakage."""

    INPUT_DIMENSION = 3 * 8 * 8 + 3 + 1 + 4

    def __init__(self, output_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_DIMENSION, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.SiLU(inplace=True),
            nn.Linear(output_dimension, output_dimension),
            nn.SiLU(inplace=True),
        )

    def forward(self, metadata: JPEGMetadata) -> Tensor:
        metadata.validate()
        qtables = torch.log1p(metadata.qtables.clamp_min(1.0)) / torch.log(
            metadata.qtables.new_tensor(256.0)
        )
        vector = torch.cat(
            (
                qtables.flatten(1),
                metadata.qtable_valid,
                metadata.is_jpeg,
                metadata.subsampling,
            ),
            dim=1,
        )
        return self.layers(vector)


class _RegionalPredictionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        number_of_noise_types: int,
        maximum_noise_strength: float,
    ) -> None:
        super().__init__()
        self.number_of_noise_types = number_of_noise_types
        self.maximum_noise_strength = maximum_noise_strength
        self.predictor = nn.Sequential(
            ResidualBlock(channels),
            nn.Conv2d(channels, 3 + number_of_noise_types, kernel_size=1),
        )

    def forward(self, features: Tensor) -> DegradationLevelOutput:
        raw = self.predictor(features)
        quality_raw = raw[:, 0:1]
        double_logits = raw[:, 1:2]
        noise_logits = raw[:, 2 : 2 + self.number_of_noise_types]
        strength_raw = raw[:, 2 + self.number_of_noise_types :]
        return DegradationLevelOutput(
            jpeg_quality=1.0 + 99.0 * quality_raw.sigmoid(),
            double_compression_logits=double_logits,
            noise_type_logits=noise_logits,
            noise_strength=self.maximum_noise_strength * strength_raw.sigmoid(),
        )


class RegionalDegradationEstimator(nn.Module):
    """Fuse RGB, DCT, and metadata into multi-scale regional degradation maps."""

    def __init__(
        self,
        *,
        dct_channels: Sequence[int] = (64, 128, 256, 384),
        level_names: Sequence[str] = ("s8", "s16", "s32", "s64"),
        metadata_dimension: int = 32,
        fusion_channels: Sequence[int] | None = None,
        noise_types: Sequence[str] = ("none", "gaussian", "poisson", "speckle"),
        maximum_noise_strength: float = 0.2,
    ) -> None:
        super().__init__()
        if len(dct_channels) != len(level_names):
            raise ValueError("dct_channels and level_names must have equal length")
        if not noise_types:
            raise ValueError("At least one noise type is required")
        if maximum_noise_strength <= 0:
            raise ValueError("maximum_noise_strength must be positive")
        if fusion_channels is None:
            fusion_channels = dct_channels
        if len(fusion_channels) != len(level_names):
            raise ValueError("fusion_channels and level_names must have equal length")

        self.dct_channels = tuple(int(value) for value in dct_channels)
        self.level_names = tuple(str(value) for value in level_names)
        self.fusion_channels = tuple(int(value) for value in fusion_channels)
        self.noise_types = tuple(str(value) for value in noise_types)
        self.maximum_noise_strength = float(maximum_noise_strength)

        self.rgb_pyramid = _RGBFeaturePyramid(self.dct_channels, self.level_names)
        self.metadata_encoder = _MetadataEncoder(metadata_dimension)
        self.fusions = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for name, dct_width, output_width in zip(
            self.level_names, self.dct_channels, self.fusion_channels, strict=True
        ):
            self.fusions[name] = nn.Sequential(
                ConvNormAct(2 * dct_width + metadata_dimension, output_width, 1),
                ResidualBlock(output_width),
            )
            self.heads[name] = _RegionalPredictionHead(
                output_width,
                len(self.noise_types),
                self.maximum_noise_strength,
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        dct_channels: Sequence[int],
        level_names: Sequence[str],
    ) -> RegionalDegradationEstimator:
        configured_fusion = config.get("fusion_channels")
        return cls(
            dct_channels=dct_channels,
            level_names=level_names,
            metadata_dimension=int(config.get("metadata_dimension", 32)),
            fusion_channels=None if configured_fusion is None else tuple(configured_fusion),
            noise_types=tuple(
                config.get("noise_types", ("none", "gaussian", "poisson", "speckle"))
            ),
            maximum_noise_strength=float(config.get("maximum_noise_strength", 0.2)),
        )

    def forward(
        self,
        rgb: Tensor,
        dct: DCTBranchOutput | Mapping[str, Tensor],
        metadata: JPEGMetadata,
    ) -> DegradationEstimatorOutput:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        dct_features = dct.features if isinstance(dct, DCTBranchOutput) else dict(dct)
        missing = set(self.level_names).difference(dct_features)
        if missing:
            raise ValueError(f"Missing DCT pyramid levels: {sorted(missing)}")

        metadata = metadata.to(device=rgb.device, dtype=rgb.dtype)
        rgb_features = self.rgb_pyramid(rgb)
        metadata_vector = self.metadata_encoder(metadata)
        outputs: dict[str, DegradationLevelOutput] = {}
        for name, expected_channels in zip(self.level_names, self.dct_channels, strict=True):
            frequency = dct_features[name]
            if frequency.shape[1] != expected_channels:
                raise ValueError(
                    f"DCT level {name} has {frequency.shape[1]} channels; "
                    f"expected {expected_channels}"
                )
            spatial = rgb_features[name]
            if spatial.shape[-2:] != frequency.shape[-2:]:
                spatial = F.interpolate(
                    spatial, size=frequency.shape[-2:], mode="bilinear", align_corners=False
                )
            metadata_map = metadata_vector[:, :, None, None].expand(-1, -1, *frequency.shape[-2:])
            fused = self.fusions[name](torch.cat((frequency, spatial, metadata_map), dim=1))
            outputs[name] = self.heads[name](fused)
        return DegradationEstimatorOutput(
            levels=outputs,
            noise_types=self.noise_types,
            maximum_noise_strength=self.maximum_noise_strength,
        )


class MultiScaleDegradationLoss(nn.Module):
    """Masked supervision for all estimator tasks at every requested scale."""

    def __init__(
        self,
        *,
        quality_weight: float = 1.0,
        double_compression_weight: float = 1.0,
        noise_type_weight: float = 1.0,
        noise_strength_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = {
            "quality": float(quality_weight),
            "double_compression": float(double_compression_weight),
            "noise_type": float(noise_type_weight),
            "noise_strength": float(noise_strength_weight),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MultiScaleDegradationLoss:
        return cls(
            quality_weight=float(config.get("quality_weight", 1.0)),
            double_compression_weight=float(config.get("double_compression_weight", 1.0)),
            noise_type_weight=float(config.get("noise_type_weight", 1.0)),
            noise_strength_weight=float(config.get("noise_strength_weight", 1.0)),
        )

    @staticmethod
    def _resize_map(target: Tensor, size: tuple[int, int], *, nearest: bool = False) -> Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != 4:
            raise ValueError("Regional targets must have shape [B,H,W] or [B,C,H,W]")
        if nearest:
            return F.interpolate(target.float(), size=size, mode="nearest")
        return F.interpolate(target.float(), size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return values.mean()
        mask = mask.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(1)
        if mask.shape != values.shape:
            mask = mask.expand_as(values)
        denominator = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denominator

    def forward(
        self,
        predictions: DegradationEstimatorOutput,
        targets: DegradationTargets,
    ) -> dict[str, Tensor]:
        if not predictions.levels:
            raise ValueError("At least one degradation prediction level is required")

        component_lists: dict[str, list[Tensor]] = {
            "quality": [],
            "double_compression": [],
            "noise_type": [],
            "noise_strength": [],
        }
        for output in predictions.levels.values():
            size = output.jpeg_quality.shape[-2:]
            quality_target = self._resize_map(targets.jpeg_quality, size)
            double_target = self._resize_map(targets.double_compression, size, nearest=True)
            noise_target = (
                self._resize_map(targets.noise_type, size, nearest=True).squeeze(1).long()
            )
            strength_target = self._resize_map(targets.noise_strength, size)
            jpeg_mask = (
                None
                if targets.jpeg_valid is None
                else self._resize_map(targets.jpeg_valid, size, nearest=True)
            )
            noise_mask = (
                None
                if targets.noise_valid is None
                else self._resize_map(targets.noise_valid, size, nearest=True)
            )

            quality_error = (output.jpeg_quality - quality_target).abs() / 99.0
            component_lists["quality"].append(self._masked_mean(quality_error, jpeg_mask))
            double_error = F.binary_cross_entropy_with_logits(
                output.double_compression_logits, double_target, reduction="none"
            )
            component_lists["double_compression"].append(self._masked_mean(double_error, jpeg_mask))
            noise_error = F.cross_entropy(output.noise_type_logits, noise_target, reduction="none")
            squeezed_noise_mask = None if noise_mask is None else noise_mask.squeeze(1)
            component_lists["noise_type"].append(
                self._masked_mean(noise_error, squeezed_noise_mask)
            )
            strength_error = (
                output.noise_strength - strength_target
            ).abs() / predictions.maximum_noise_strength
            component_lists["noise_strength"].append(self._masked_mean(strength_error, noise_mask))

        components = {name: torch.stack(values).mean() for name, values in component_lists.items()}
        total = sum(self.weights[name] * value for name, value in components.items())
        return {"total": total, **components}


@dataclass
class FrontEndOutput:
    dct: DCTBranchOutput
    degradation: DegradationEstimatorOutput


class InputForensicsFrontEnd(nn.Module):
    """Milestone wrapper connecting the DCT branch to degradation estimation."""

    def __init__(
        self,
        dct_branch: DCTFeaturePyramid,
        degradation_estimator: RegionalDegradationEstimator,
    ) -> None:
        super().__init__()
        if dct_branch.level_names != degradation_estimator.level_names:
            raise ValueError("DCT and degradation-estimator level names must match")
        if dct_branch.out_channels != degradation_estimator.dct_channels:
            raise ValueError("DCT output channels must match estimator DCT channels")
        self.dct_branch = dct_branch
        self.degradation_estimator = degradation_estimator

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> InputForensicsFrontEnd:
        dct_branch = DCTFeaturePyramid.from_config(config.get("dct", {}))
        estimator = RegionalDegradationEstimator.from_config(
            config.get("degradation_estimator", {}),
            dct_channels=dct_branch.out_channels,
            level_names=dct_branch.level_names,
        )
        return cls(dct_branch, estimator)

    def forward(
        self,
        rgb: Tensor,
        *,
        metadata: JPEGMetadata | None = None,
        exact_dct: ExactDCTBatch | None = None,
        compute_consistency: bool | None = None,
    ) -> FrontEndOutput:
        if metadata is None:
            metadata = (
                exact_dct.metadata
                if exact_dct is not None
                else JPEGMetadata.for_non_jpeg(rgb.shape[0], device=rgb.device, dtype=rgb.dtype)
            )
        metadata = metadata.to(device=rgb.device, dtype=rgb.dtype)
        dct_output = self.dct_branch(
            rgb,
            metadata=metadata,
            exact_dct=exact_dct,
            compute_consistency=compute_consistency,
        )
        degradation_output = self.degradation_estimator(rgb, dct_output, metadata)
        return FrontEndOutput(dct=dct_output, degradation=degradation_output)
