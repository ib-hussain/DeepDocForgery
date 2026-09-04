"""Region-conditioned attention fusion for DeepDocForgery.

Every pyramid level receives three evidence streams:

* spatial ViT+ADN features,
* block-frequency DCT features, and
* regional degradation estimates.

The streams are projected to a common width.  A local attention head and a
global context head jointly predict a softmax distribution over the three
branches for every channel and spatial region.  Consequently, the weights are
ordinary trainable model parameters, receive gradients from downstream losses,
and can be inspected after every forward pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from deepdocforgery.degradation import (
    DegradationEstimatorOutput,
    FrontEndOutput,
    InputForensicsFrontEnd,
)
from deepdocforgery.frequency import (
    ConvNormAct,
    DCTBranchOutput,
    ExactDCTBatch,
    JPEGMetadata,
    ResidualBlock,
)
from deepdocforgery.spatial import SpatialBranchOutput, SpatialFeaturePyramid


class _RegionConditionedAttentionHead(nn.Module):
    """Predict per-region, per-channel weights over three feature branches."""

    number_of_branches = 3

    def __init__(
        self,
        channels: int,
        *,
        hidden_ratio: float = 0.5,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("attention channels must be positive")
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive")
        joined_channels = self.number_of_branches * channels
        hidden = max(8, int(round(channels * hidden_ratio)))
        self.channels = channels
        self.local = nn.Sequential(
            ConvNormAct(joined_channels, channels, kernel_size=3),
            nn.Conv2d(channels, joined_channels, kernel_size=1),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(joined_channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, joined_channels, kernel_size=1),
        )

        # Equal one-third routing is a defensible neutral prior.  Both heads
        # start at zero and learn deviations solely from training evidence.
        for head in (self.local[-1], self.global_context[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        initial_raw = math.log(math.expm1(1.0))
        temperature = torch.tensor(initial_raw, dtype=torch.float32)
        if learnable_temperature:
            self.raw_temperature = nn.Parameter(temperature)
        else:
            self.register_buffer("raw_temperature", temperature)

    def forward(self, joined: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.local(joined) + self.global_context(joined)
        batch, _, height, width = logits.shape
        logits = logits.reshape(
            batch,
            self.number_of_branches,
            self.channels,
            height,
            width,
        )
        temperature = F.softplus(self.raw_temperature).clamp_min(1e-4)
        weights = (logits / temperature).softmax(dim=1)
        return logits, weights, temperature


@dataclass
class FusionLevelOutput:
    """Fused representation and routing evidence for one spatial scale."""

    features: Tensor
    attention_logits: Tensor
    attention_weights: Tensor
    branch_features: dict[str, Tensor]
    temperature: Tensor


@dataclass
class AttentionFusionOutput:
    """Named multi-scale fusion maps with auditable branch weights."""

    levels: dict[str, FusionLevelOutput]
    branch_names: tuple[str, ...]

    @property
    def features(self) -> dict[str, Tensor]:
        return {name: value.features for name, value in self.levels.items()}

    def mean_attention(self) -> dict[str, dict[str, Tensor]]:
        """Return mean routing weights by level and evidence branch."""

        result: dict[str, dict[str, Tensor]] = {}
        for level_name, level in self.levels.items():
            means = level.attention_weights.mean(dim=(0, 2, 3, 4))
            result[level_name] = {
                branch_name: means[index] for index, branch_name in enumerate(self.branch_names)
            }
        return result


class MultiScaleAttentionFusion(nn.Module):
    """Trainable three-way spatial/frequency/degradation fusion."""

    branch_names = ("spatial", "frequency", "degradation")

    def __init__(
        self,
        *,
        dct_channels: Sequence[int],
        spatial_channels: Sequence[int],
        degradation_channels: int,
        level_names: Sequence[str],
        fusion_channels: Sequence[int] | None = None,
        attention_hidden_ratio: float = 0.5,
        learnable_temperature: bool = True,
        refinement_depth: int = 1,
        enabled_branches: Sequence[str] = ("spatial", "frequency", "degradation"),
    ) -> None:
        super().__init__()
        number_of_levels = len(level_names)
        if number_of_levels < 1:
            raise ValueError("At least one fusion level is required")
        if len(dct_channels) != number_of_levels:
            raise ValueError("dct_channels must match level_names")
        if len(spatial_channels) != number_of_levels:
            raise ValueError("spatial_channels must match level_names")
        if degradation_channels < 1:
            raise ValueError("degradation_channels must be positive")
        if fusion_channels is None:
            fusion_channels = spatial_channels
        if len(fusion_channels) != number_of_levels:
            raise ValueError("fusion_channels must match level_names")
        if refinement_depth < 1:
            raise ValueError("refinement_depth must be positive")

        self.level_names = tuple(str(value) for value in level_names)
        self.dct_channels = tuple(int(value) for value in dct_channels)
        self.spatial_channels = tuple(int(value) for value in spatial_channels)
        self.degradation_channels = int(degradation_channels)
        self.out_channels = tuple(int(value) for value in fusion_channels)
        unknown = set(enabled_branches).difference(self.branch_names)
        if unknown or not enabled_branches:
            raise ValueError(f"Invalid enabled fusion branches: {sorted(unknown)}")
        self.enabled_branches = tuple(str(value) for value in enabled_branches)

        self.spatial_projections = nn.ModuleDict()
        self.frequency_projections = nn.ModuleDict()
        self.degradation_projections = nn.ModuleDict()
        self.attention_heads = nn.ModuleDict()
        self.refinements = nn.ModuleDict()
        for name, dct_width, spatial_width, output_width in zip(
            self.level_names,
            self.dct_channels,
            self.spatial_channels,
            self.out_channels,
            strict=True,
        ):
            self.spatial_projections[name] = ConvNormAct(
                spatial_width,
                output_width,
                kernel_size=1,
            )
            self.frequency_projections[name] = ConvNormAct(
                dct_width,
                output_width,
                kernel_size=1,
            )
            self.degradation_projections[name] = ConvNormAct(
                self.degradation_channels,
                output_width,
                kernel_size=1,
            )
            self.attention_heads[name] = _RegionConditionedAttentionHead(
                output_width,
                hidden_ratio=attention_hidden_ratio,
                learnable_temperature=learnable_temperature,
            )
            self.refinements[name] = nn.Sequential(
                *[ResidualBlock(output_width) for _ in range(refinement_depth)]
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        dct_channels: Sequence[int],
        spatial_channels: Sequence[int],
        degradation_channels: int,
        level_names: Sequence[str],
    ) -> MultiScaleAttentionFusion:
        configured_channels = config.get("out_channels")
        return cls(
            dct_channels=dct_channels,
            spatial_channels=spatial_channels,
            degradation_channels=degradation_channels,
            level_names=level_names,
            fusion_channels=(None if configured_channels is None else tuple(configured_channels)),
            attention_hidden_ratio=float(config.get("attention_hidden_ratio", 0.5)),
            learnable_temperature=bool(config.get("learnable_temperature", True)),
            refinement_depth=int(config.get("refinement_depth", 1)),
            enabled_branches=tuple(config.get("enabled_branches", cls.branch_names)),
        )

    @staticmethod
    def _resize(inputs: Tensor, size: tuple[int, int]) -> Tensor:
        if inputs.shape[-2:] == size:
            return inputs
        return F.interpolate(inputs, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        spatial: SpatialBranchOutput | Mapping[str, Tensor],
        dct: DCTBranchOutput | Mapping[str, Tensor],
        degradation: DegradationEstimatorOutput,
    ) -> AttentionFusionOutput:
        spatial_features = (
            spatial.features if isinstance(spatial, SpatialBranchOutput) else dict(spatial)
        )
        dct_features = dct.features if isinstance(dct, DCTBranchOutput) else dict(dct)
        degradation_features = degradation.gate_features()
        for label, values in (
            ("spatial", spatial_features),
            ("DCT", dct_features),
            ("degradation", degradation_features),
        ):
            missing = set(self.level_names).difference(values)
            if missing:
                raise ValueError(f"Missing {label} pyramid levels: {sorted(missing)}")

        outputs: dict[str, FusionLevelOutput] = {}
        for index, name in enumerate(self.level_names):
            spatial_value = spatial_features[name]
            frequency_value = dct_features[name]
            degradation_value = degradation_features[name]
            if spatial_value.shape[1] != self.spatial_channels[index]:
                raise ValueError(
                    f"Spatial level {name} has {spatial_value.shape[1]} channels; "
                    f"expected {self.spatial_channels[index]}"
                )
            if frequency_value.shape[1] != self.dct_channels[index]:
                raise ValueError(
                    f"DCT level {name} has {frequency_value.shape[1]} channels; "
                    f"expected {self.dct_channels[index]}"
                )
            if degradation_value.shape[1] != self.degradation_channels:
                raise ValueError(
                    f"Degradation level {name} has {degradation_value.shape[1]} channels; "
                    f"expected {self.degradation_channels}"
                )

            size = spatial_value.shape[-2:]
            spatial_projected = self.spatial_projections[name](spatial_value)
            frequency_projected = self.frequency_projections[name](
                self._resize(frequency_value, size)
            )
            degradation_projected = self.degradation_projections[name](
                self._resize(degradation_value, size)
            )
            branches = {
                "spatial": spatial_projected,
                "frequency": frequency_projected,
                "degradation": degradation_projected,
            }
            for branch_name in self.branch_names:
                if branch_name not in self.enabled_branches:
                    branches[branch_name] = torch.zeros_like(branches[branch_name])
            joined = torch.cat(tuple(branches.values()), dim=1)
            logits, weights, temperature = self.attention_heads[name](joined)
            if len(self.enabled_branches) != len(self.branch_names):
                disabled = [
                    index
                    for index, branch_name in enumerate(self.branch_names)
                    if branch_name not in self.enabled_branches
                ]
                logits = logits.clone()
                logits[:, disabled] = -1.0e4
                weights = (logits / temperature).softmax(dim=1)
            stacked = torch.stack(tuple(branches.values()), dim=1)
            fused = self.refinements[name]((weights * stacked).sum(dim=1))
            outputs[name] = FusionLevelOutput(
                features=fused,
                attention_logits=logits,
                attention_weights=weights,
                branch_features=branches,
                temperature=temperature,
            )
        return AttentionFusionOutput(outputs, self.branch_names)


@dataclass
class DeepDocForgeryFusionOutput:
    """Everything produced by the pipeline through the fusion boundary."""

    forensics: FrontEndOutput
    spatial: SpatialBranchOutput
    fusion: AttentionFusionOutput


class DeepDocForgeryFusionFrontEnd(nn.Module):
    """DCT + degradation + ViT/ADN + learned attention fusion wrapper."""

    def __init__(
        self,
        forensics: InputForensicsFrontEnd,
        spatial: SpatialFeaturePyramid,
        fusion: MultiScaleAttentionFusion,
    ) -> None:
        super().__init__()
        dct = forensics.dct_branch
        if dct.level_names != spatial.level_names or dct.level_names != fusion.level_names:
            raise ValueError("DCT, spatial, and fusion level names must match")
        if dct.out_channels != fusion.dct_channels:
            raise ValueError("DCT channels must match the fusion contract")
        if spatial.out_channels != fusion.spatial_channels:
            raise ValueError("Spatial channels must match the fusion contract")
        expected_degradation_channels = len(forensics.degradation_estimator.noise_types) + 3
        if fusion.degradation_channels != expected_degradation_channels:
            raise ValueError("Degradation gate width must match the configured noise taxonomy")
        self.forensics = forensics
        self.spatial = spatial
        self.fusion = fusion

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, load_pretrained: bool = True
    ) -> DeepDocForgeryFusionFrontEnd:
        forensics = InputForensicsFrontEnd.from_config(config)
        spatial_config = dict(config.get("spatial", {}))
        spatial_config.setdefault("out_channels", forensics.dct_branch.out_channels)
        spatial_config.setdefault("level_names", forensics.dct_branch.level_names)
        spatial = SpatialFeaturePyramid.from_config(spatial_config, load_pretrained=load_pretrained)
        fusion = MultiScaleAttentionFusion.from_config(
            config.get("fusion", {}),
            dct_channels=forensics.dct_branch.out_channels,
            spatial_channels=spatial.out_channels,
            degradation_channels=len(forensics.degradation_estimator.noise_types) + 3,
            level_names=forensics.dct_branch.level_names,
        )
        return cls(forensics, spatial, fusion)

    def forward(
        self,
        rgb: Tensor,
        *,
        metadata: JPEGMetadata | None = None,
        exact_dct: ExactDCTBatch | None = None,
        compute_consistency: bool | None = None,
    ) -> DeepDocForgeryFusionOutput:
        forensics = self.forensics(
            rgb,
            metadata=metadata,
            exact_dct=exact_dct,
            compute_consistency=compute_consistency,
        )
        spatial = self.spatial(rgb)
        fusion = self.fusion(spatial, forensics.dct, forensics.degradation)
        return DeepDocForgeryFusionOutput(
            forensics=forensics,
            spatial=spatial,
            fusion=fusion,
        )
