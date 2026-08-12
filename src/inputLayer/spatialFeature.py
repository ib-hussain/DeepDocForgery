"""Spatial ViT + Artifact Decouple Network features for DeepDocForgery.

The official DanceText paper describes a hierarchical ViT backbone running in
parallel with a ConvNeXt-based Artifact Decouple Network (ADN).  Their
multi-scale maps are concatenated, channel-reweighted, and reduced with a 3x3
convolution.  DanceText's DS-Net source was not public when this module was
implemented, so this is a method-level implementation rather than copied code.

Both backbones are implemented with stock PyTorch operations.  That keeps the
same source usable on CPU and CUDA, under Arch Linux/WSL and Ubuntu Desktop,
without requiring torchvision, timm, or an online pretrained-weight download.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .freqFeatures import ConvNormAct, ResidualBlock


def _pad_bottom_right(inputs: Tensor, multiple: int) -> Tensor:
    """Replicate-pad spatial dimensions to a positive ``multiple``."""

    if multiple < 1:
        raise ValueError("multiple must be positive")
    height, width = inputs.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate")
    return inputs


class _LayerNorm2d(nn.Module):
    """Layer normalization over channels for an ``NCHW`` tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.normalization(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _PatchStem(nn.Module):
    """Non-overlapping 4x4 image patches followed by channel normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=4,
            bias=False,
        )
        self.normalization = _LayerNorm2d(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.normalization(self.projection(_pad_bottom_right(inputs, 4)))


class _PatchDownsample(nn.Module):
    """Halve spatial resolution while retaining odd border rows/columns."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.normalization = _LayerNorm2d(in_channels)
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.normalization(inputs)
        return self.projection(_pad_bottom_right(inputs, 2))


class _WindowTransformerBlock(nn.Module):
    """Pre-normalized self-attention over local two-dimensional windows."""

    def __init__(
        self,
        channels: int,
        number_of_heads: int,
        *,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % number_of_heads:
            raise ValueError(
                f"Transformer width {channels} must be divisible by {number_of_heads} heads"
            )
        if window_size < 1:
            raise ValueError("window_size must be positive")
        hidden = max(channels, int(round(channels * mlp_ratio)))
        self.window_size = int(window_size)
        self.position = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            number_of_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.position(inputs)
        batch, channels, height, width = inputs.shape
        window = self.window_size
        values = inputs.permute(0, 2, 3, 1)
        pad_h = (-height) % window
        pad_w = (-width) % window
        if pad_h or pad_w:
            values = F.pad(values, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = values.shape[1:3]
        grid_h, grid_w = padded_h // window, padded_w // window
        windows = (
            values.reshape(batch, grid_h, window, grid_w, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window, channels)
        )

        normalized = self.attention_norm(windows)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        windows = windows + attended
        windows = windows + self.mlp(self.mlp_norm(windows))

        values = (
            windows.reshape(batch, grid_h, grid_w, window, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, padded_h, padded_w, channels)
        )
        values = values[:, :height, :width]
        return values.permute(0, 3, 1, 2).contiguous()


class _TransformerStage(nn.Sequential):
    def __init__(
        self,
        channels: int,
        depth: int,
        number_of_heads: int,
        *,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        if depth < 1:
            raise ValueError("Every ViT stage requires at least one block")
        super().__init__(
            *[
                _WindowTransformerBlock(
                    channels,
                    number_of_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )


class HierarchicalViTBackbone(nn.Module):
    """A compact hierarchical, window-attention Vision Transformer.

    A stride-4 stem is retained internally.  Public features begin at stride 8
    so their grids line up with the JPEG block-DCT branch.
    """

    def __init__(
        self,
        *,
        out_channels: Sequence[int],
        level_names: Sequence[str],
        stem_channels: int = 48,
        stem_depth: int = 1,
        depths: Sequence[int],
        number_of_heads: Sequence[int],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not out_channels or len(out_channels) != len(level_names):
            raise ValueError("ViT out_channels and level_names must be non-empty and equal")
        if len(depths) != len(level_names) or len(number_of_heads) != len(level_names):
            raise ValueError("ViT depths and number_of_heads must match level_names")
        if stem_channels < 1:
            raise ValueError("stem_channels must be positive")

        self.out_channels = tuple(int(value) for value in out_channels)
        self.level_names = tuple(str(value) for value in level_names)
        self.patch_stem = _PatchStem(3, stem_channels)
        stem_heads = _largest_divisor(stem_channels, max(1, number_of_heads[0]))
        self.stem_stage = _TransformerStage(
            stem_channels,
            stem_depth,
            stem_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        downsamplers: list[nn.Module] = []
        stages: list[nn.Module] = []
        previous = stem_channels
        for channels, depth, heads in zip(
            self.out_channels,
            depths,
            number_of_heads,
            strict=True,
        ):
            downsamplers.append(_PatchDownsample(previous, channels))
            stages.append(
                _TransformerStage(
                    channels,
                    int(depth),
                    int(heads),
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )
            previous = channels
        self.downsamplers = nn.ModuleList(downsamplers)
        self.stages = nn.ModuleList(stages)

    def forward(self, rgb: Tensor) -> dict[str, Tensor]:
        current = self.stem_stage(self.patch_stem(rgb))
        outputs: dict[str, Tensor] = {}
        for name, downsample, stage in zip(
            self.level_names,
            self.downsamplers,
            self.stages,
            strict=True,
        ):
            current = stage(downsample(current))
            outputs[name] = current
        return outputs


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style depthwise block with learnable layer scaling."""

    def __init__(self, channels: int, *, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.normalization = nn.LayerNorm(channels)
        self.pointwise_1 = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.pointwise_2 = nn.Linear(hidden, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        values = self.depthwise(inputs).permute(0, 2, 3, 1)
        values = self.pointwise_2(self.activation(self.pointwise_1(self.normalization(values))))
        values = values * self.layer_scale
        return residual + values.permute(0, 3, 1, 2)


class _ConvNeXtStage(nn.Sequential):
    def __init__(self, channels: int, depth: int) -> None:
        if depth < 1:
            raise ValueError("Every ADN stage requires at least one block")
        super().__init__(*[_ConvNeXtBlock(channels) for _ in range(depth)])


class ArtifactDecoupleNetwork(nn.Module):
    """ConvNeXt-style ADN producing artifact maps and an auxiliary mask."""

    def __init__(
        self,
        *,
        out_channels: Sequence[int],
        level_names: Sequence[str],
        stem_channels: int = 48,
        stem_depth: int = 1,
        depths: Sequence[int],
    ) -> None:
        super().__init__()
        if not out_channels or len(out_channels) != len(level_names):
            raise ValueError("ADN out_channels and level_names must be non-empty and equal")
        if len(depths) != len(level_names):
            raise ValueError("ADN depths must match level_names")

        self.out_channels = tuple(int(value) for value in out_channels)
        self.level_names = tuple(str(value) for value in level_names)
        self.patch_stem = _PatchStem(3, stem_channels)
        self.stem_stage = _ConvNeXtStage(stem_channels, stem_depth)

        downsamplers: list[nn.Module] = []
        stages: list[nn.Module] = []
        previous = stem_channels
        for channels, depth in zip(self.out_channels, depths, strict=True):
            downsamplers.append(_PatchDownsample(previous, channels))
            stages.append(_ConvNeXtStage(channels, int(depth)))
            previous = channels
        self.downsamplers = nn.ModuleList(downsamplers)
        self.stages = nn.ModuleList(stages)
        self.auxiliary_head = nn.Conv2d(self.out_channels[-1], 1, kernel_size=1)

    def forward(self, rgb: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        current = self.stem_stage(self.patch_stem(rgb))
        outputs: dict[str, Tensor] = {}
        for name, downsample, stage in zip(
            self.level_names,
            self.downsamplers,
            self.stages,
            strict=True,
        ):
            current = stage(downsample(current))
            outputs[name] = current
        return outputs, self.auxiliary_head(current)


class _ChannelAttentionFusion(nn.Module):
    """DanceText-style channel attention over paired ViT and ADN maps."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("channel-attention reduction must be positive")
        joined_channels = 2 * channels
        hidden = max(8, joined_channels // reduction)
        self.excitation = nn.Sequential(
            nn.Conv2d(joined_channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, joined_channels, kernel_size=1),
        )
        # Neutral 0.5 attention at initialization; subsequent gradients choose
        # the useful ViT/ADN channels.
        nn.init.zeros_(self.excitation[-1].weight)
        nn.init.zeros_(self.excitation[-1].bias)
        self.reduction = nn.Sequential(
            ConvNormAct(joined_channels, channels, kernel_size=3),
            ResidualBlock(channels),
        )
        self.channels = channels

    def forward(self, vit: Tensor, adn: Tensor) -> tuple[Tensor, Tensor]:
        if vit.shape != adn.shape:
            raise ValueError("ViT and ADN features must have identical shapes before fusion")
        joined = torch.cat((vit, adn), dim=1)
        attention = self.excitation(F.adaptive_avg_pool2d(joined, 1)).sigmoid()
        fused = self.reduction(joined * attention)
        branch_attention = attention.reshape(
            attention.shape[0],
            2,
            self.channels,
            1,
            1,
        )
        return fused, branch_attention


@dataclass
class SpatialBranchOutput:
    """Public spatial pyramid plus auditable ViT/ADN intermediate maps."""

    features: dict[str, Tensor]
    vit_features: dict[str, Tensor]
    adn_features: dict[str, Tensor]
    backbone_attention: dict[str, Tensor]
    adn_auxiliary_logits: Tensor

    def auxiliary_probability(self, size: tuple[int, int] | None = None) -> Tensor:
        probability = self.adn_auxiliary_logits.sigmoid()
        if size is not None and probability.shape[-2:] != size:
            probability = F.interpolate(
                probability,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return probability


class SpatialFeaturePyramid(nn.Module):
    """Parallel hierarchical ViT + ADN with learned channel attention."""

    backbone_names = ("vit", "adn")

    def __init__(
        self,
        *,
        out_channels: Sequence[int] = (64, 128, 256, 384),
        level_names: Sequence[str] = ("s8", "s16", "s32", "s64"),
        vit_stem_channels: int = 48,
        vit_stem_depth: int = 1,
        vit_depths: Sequence[int] = (1, 2, 4, 1),
        vit_number_of_heads: Sequence[int] = (2, 4, 8, 12),
        vit_window_size: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
        adn_stem_channels: int = 48,
        adn_stem_depth: int = 1,
        adn_depths: Sequence[int] = (1, 2, 4, 1),
        channel_attention_reduction: int = 4,
    ) -> None:
        super().__init__()
        if not out_channels or len(out_channels) != len(level_names):
            raise ValueError("spatial out_channels and level_names must be non-empty and equal")
        self.out_channels = tuple(int(value) for value in out_channels)
        self.level_names = tuple(str(value) for value in level_names)

        self.vit = HierarchicalViTBackbone(
            out_channels=self.out_channels,
            level_names=self.level_names,
            stem_channels=vit_stem_channels,
            stem_depth=vit_stem_depth,
            depths=vit_depths,
            number_of_heads=vit_number_of_heads,
            window_size=vit_window_size,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
        )
        self.adn = ArtifactDecoupleNetwork(
            out_channels=self.out_channels,
            level_names=self.level_names,
            stem_channels=adn_stem_channels,
            stem_depth=adn_stem_depth,
            depths=adn_depths,
        )
        self.fusions = nn.ModuleDict(
            {
                name: _ChannelAttentionFusion(channels, channel_attention_reduction)
                for name, channels in zip(self.level_names, self.out_channels, strict=True)
            }
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SpatialFeaturePyramid:
        out_channels = tuple(config.get("out_channels", (64, 128, 256, 384)))
        level_names = tuple(config.get("level_names", ("s8", "s16", "s32", "s64")))
        number_of_levels = len(level_names)
        vit = config.get("vit", {})
        adn = config.get("adn", {})
        default_depths = _resize_tuple((1, 2, 4, 1), number_of_levels)
        default_heads = tuple(_default_attention_heads(int(width)) for width in out_channels)
        return cls(
            out_channels=out_channels,
            level_names=level_names,
            vit_stem_channels=int(vit.get("stem_channels", 48)),
            vit_stem_depth=int(vit.get("stem_depth", 1)),
            vit_depths=tuple(vit.get("depths", default_depths)),
            vit_number_of_heads=tuple(vit.get("number_of_heads", default_heads)),
            vit_window_size=int(vit.get("window_size", 8)),
            vit_mlp_ratio=float(vit.get("mlp_ratio", 4.0)),
            vit_dropout=float(vit.get("dropout", 0.0)),
            adn_stem_channels=int(adn.get("stem_channels", 48)),
            adn_stem_depth=int(adn.get("stem_depth", 1)),
            adn_depths=tuple(adn.get("depths", default_depths)),
            channel_attention_reduction=int(config.get("channel_attention_reduction", 4)),
        )

    def forward(self, rgb: Tensor) -> SpatialBranchOutput:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        if not rgb.is_floating_point():
            raise TypeError("rgb must be a floating-point tensor normalized to [0, 1]")

        vit_features = self.vit(rgb)
        adn_features, auxiliary_logits = self.adn(rgb)
        features: dict[str, Tensor] = {}
        attention: dict[str, Tensor] = {}
        for name in self.level_names:
            features[name], attention[name] = self.fusions[name](
                vit_features[name],
                adn_features[name],
            )
        return SpatialBranchOutput(
            features=features,
            vit_features=vit_features,
            adn_features=adn_features,
            backbone_attention=attention,
            adn_auxiliary_logits=auxiliary_logits,
        )


@dataclass
class ADNSupervision:
    """Targets for the ADN auxiliary and representation objectives.

    ``domain_index`` uses 0 for text images and 1 for non-text images.
    ``region_type`` uses 0 for authentic, 1 for edited, and 2 for removed
    regions.  The latter two tensors are optional so ordinary segmentation
    batches can still train the auxiliary ADN head.
    """

    artifact_mask: Tensor
    valid_mask: Tensor | None = None
    domain_index: Tensor | None = None
    region_type: Tensor | None = None

    def to(self, *args: Any, **kwargs: Any) -> ADNSupervision:
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        return ADNSupervision(
            artifact_mask=self.artifact_mask.to(*args, **kwargs),
            valid_mask=move(self.valid_mask),
            domain_index=move(self.domain_index),
            region_type=move(self.region_type),
        )


class ArtifactDecouplingLoss(nn.Module):
    """Paper-inspired supervision hooks for the Artifact Decouple Network.

    This loss exposes four separately reported terms corresponding to the roles
    of DanceText's text BCE, non-text shuffled-patch BCE, text-region
    decoupling, and cross-domain alignment.  It intentionally does not pretend
    to reproduce unpublished source details.
    """

    def __init__(
        self,
        *,
        text_bce_weight: float = 1.0,
        nontext_bce_weight: float = 1.0,
        text_decoupling_weight: float = 1.0,
        domain_alignment_weight: float = 1.0,
        separation_margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.weights = {
            "text_bce": float(text_bce_weight),
            "nontext_bce": float(nontext_bce_weight),
            "text_decoupling": float(text_decoupling_weight),
            "domain_alignment": float(domain_alignment_weight),
        }
        self.separation_margin = float(separation_margin)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ArtifactDecouplingLoss:
        return cls(
            text_bce_weight=float(config.get("text_bce_weight", 1.0)),
            nontext_bce_weight=float(config.get("nontext_bce_weight", 1.0)),
            text_decoupling_weight=float(config.get("text_decoupling_weight", 1.0)),
            domain_alignment_weight=float(config.get("domain_alignment_weight", 1.0)),
            separation_margin=float(config.get("separation_margin", 0.2)),
        )

    @staticmethod
    def _map(target: Tensor, size: tuple[int, int]) -> Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != 4:
            raise ValueError("ADN maps must have shape [B,H,W] or [B,1,H,W]")
        return F.interpolate(target.float(), size=size, mode="nearest")

    @staticmethod
    def _prototype(features: Tensor, mask: Tensor) -> Tensor | None:
        selected = features.permute(0, 2, 3, 1)[mask.squeeze(1).bool()]
        if selected.numel() == 0:
            return None
        return F.normalize(selected.mean(dim=0), dim=0)

    def _attract_and_separate(
        self,
        first: Tensor | None,
        second: Tensor | None,
        authentic: Tensor | None,
        zero: Tensor,
    ) -> Tensor:
        available = [value for value in (first, second) if value is not None]
        if not available:
            return zero
        loss = zero
        if first is not None and second is not None:
            loss = loss + 1.0 - F.cosine_similarity(first, second, dim=0)
        fake = F.normalize(torch.stack(available).mean(dim=0), dim=0)
        if authentic is not None:
            similarity = F.cosine_similarity(fake, authentic, dim=0)
            loss = loss + F.relu(similarity - self.separation_margin)
        return loss

    def forward(
        self,
        output: SpatialBranchOutput,
        targets: ADNSupervision,
    ) -> dict[str, Tensor]:
        logits = output.adn_auxiliary_logits
        artifact = self._map(targets.artifact_mask, logits.shape[-2:])
        valid = (
            torch.ones_like(artifact)
            if targets.valid_mask is None
            else self._map(targets.valid_mask, logits.shape[-2:])
        )
        pixel_loss = F.binary_cross_entropy_with_logits(logits, artifact, reduction="none")
        sample_loss = (pixel_loss * valid).flatten(1).sum(dim=1) / valid.flatten(1).sum(
            dim=1
        ).clamp_min(1.0)
        zero = logits.sum() * 0.0

        if targets.domain_index is None:
            text_bce = sample_loss.mean()
            nontext_bce = zero
            domain = None
        else:
            domain = targets.domain_index.to(device=logits.device).flatten().long()
            if domain.shape[0] != logits.shape[0]:
                raise ValueError("domain_index must contain one value per image")
            text_bce = sample_loss[domain == 0].mean() if bool((domain == 0).any()) else zero
            nontext_bce = (
                sample_loss[domain == 1].mean() if bool((domain == 1).any()) else zero
            )

        deepest = output.adn_features[next(reversed(output.adn_features))]
        artifact_deep = self._map(targets.artifact_mask, deepest.shape[-2:]) > 0.5
        authentic = self._prototype(deepest, ~artifact_deep)

        text_decoupling = zero
        if targets.region_type is not None:
            region_type = self._map(targets.region_type, deepest.shape[-2:]).long()
            edited = self._prototype(deepest, region_type == 1)
            removed = self._prototype(deepest, region_type == 2)
            text_decoupling = self._attract_and_separate(
                edited,
                removed,
                authentic,
                zero,
            )

        domain_alignment = zero
        if domain is not None and bool((domain == 0).any()) and bool((domain == 1).any()):
            domain_map = domain[:, None, None, None].expand_as(artifact_deep)
            text_fake = self._prototype(deepest, artifact_deep & (domain_map == 0))
            nontext_fake = self._prototype(deepest, artifact_deep & (domain_map == 1))
            domain_alignment = self._attract_and_separate(
                text_fake,
                nontext_fake,
                authentic,
                zero,
            )

        components = {
            "text_bce": text_bce,
            "nontext_bce": nontext_bce,
            "text_decoupling": text_decoupling,
            "domain_alignment": domain_alignment,
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return {"total": total, **components}


def _largest_divisor(value: int, maximum: int) -> int:
    for candidate in range(min(value, maximum), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _default_attention_heads(channels: int) -> int:
    desired = max(1, min(12, channels // 32))
    return _largest_divisor(channels, desired)


def _resize_tuple(values: Sequence[int], length: int) -> tuple[int, ...]:
    if length < 1:
        raise ValueError("At least one pyramid level is required")
    result = list(values[:length])
    while len(result) < length:
        result.append(result[-1])
    return tuple(result)
