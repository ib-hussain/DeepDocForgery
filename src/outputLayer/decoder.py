"""Classification-localization synergy decoder for DeepDocForgery.

DanceText describes a Synergy Denoising Decoder but, at the time this module
was written, its official repository did not publish DS-Net model source.  The
implementation below is therefore a clean-room, method-level realization of
the published idea rather than a source reproduction.

The decoder is deliberately bidirectional:

* global classification context conditions every top-down localization stage;
* the tentative localization mask pools suspicious and background evidence
  back into the final image classifier; and
* the final image decision supplies a learned prior to the full-resolution
  mask head.

All synergy strengths are trainable, bounded, and returned for inspection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.inputLayer.freqFeatures import ConvNormAct, ResidualBlock


class _TopDownRefinement(nn.Module):
    """Merge one lateral pyramid level with conditioned decoder state."""

    def __init__(self, channels: int, class_dimension: int, depth: int) -> None:
        super().__init__()
        self.mask_hint = ConvNormAct(1, channels, kernel_size=3)
        self.class_affine = nn.Linear(class_dimension, 2 * channels)
        self.merge = ConvNormAct(3 * channels, channels, kernel_size=3)
        self.refinement = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])

        # Start as an ordinary top-down decoder. Classification conditioning is
        # learned only when it helps the downstream objectives.
        nn.init.zeros_(self.class_affine.weight)
        nn.init.zeros_(self.class_affine.bias)

    def forward(
        self,
        lateral: Tensor,
        previous: Tensor,
        previous_mask_logits: Tensor,
        class_context: Tensor,
    ) -> Tensor:
        size = lateral.shape[-2:]
        previous = F.interpolate(previous, size=size, mode="bilinear", align_corners=False)
        mask_hint = F.interpolate(
            previous_mask_logits.sigmoid(), size=size, mode="bilinear", align_corners=False
        )
        gamma, beta = self.class_affine(class_context).chunk(2, dim=1)
        conditioned = previous * (1.0 + torch.tanh(gamma)[:, :, None, None])
        conditioned = conditioned + beta[:, :, None, None]
        merged = torch.cat((lateral, conditioned, self.mask_hint(mask_hint)), dim=1)
        return self.refinement(self.merge(merged))


@dataclass
class SynergyDecoderOutput:
    """Predictions and interpretable intermediate evidence from the decoder."""

    mask_logits: Tensor
    boundary_logits: Tensor
    image_logits: Tensor
    evidence_confidence_logits: Tensor
    auxiliary_mask_logits: dict[str, Tensor]
    decoded_features: dict[str, Tensor]
    localization_attention: Tensor
    initial_image_logits: Tensor
    classification_to_localization_strength: Tensor
    denoising_strength: Tensor

    @property
    def mask_probability(self) -> Tensor:
        return self.mask_logits.sigmoid()

    @property
    def boundary_probability(self) -> Tensor:
        return self.boundary_logits.sigmoid()

    @property
    def image_probability(self) -> Tensor:
        return self.image_logits.sigmoid()

    @property
    def evidence_confidence(self) -> Tensor:
        return self.evidence_confidence_logits.sigmoid()


class SynergyDenoisingDecoder(nn.Module):
    """Top-down decoder with trainable two-way classification/localization synergy."""

    def __init__(
        self,
        *,
        fusion_channels: Sequence[int],
        level_names: Sequence[str],
        decoder_channels: int = 128,
        class_dimension: int = 256,
        refinement_depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not level_names:
            raise ValueError("At least one decoder level is required")
        if len(fusion_channels) != len(level_names):
            raise ValueError("fusion_channels must match level_names")
        if min(decoder_channels, class_dimension, refinement_depth) < 1:
            raise ValueError("Decoder widths and refinement_depth must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.level_names = tuple(str(name) for name in level_names)
        self.fusion_channels = tuple(int(value) for value in fusion_channels)
        self.decoder_channels = int(decoder_channels)
        self.class_dimension = int(class_dimension)

        self.lateral = nn.ModuleDict(
            {
                name: ConvNormAct(width, decoder_channels, kernel_size=1)
                for name, width in zip(self.level_names, self.fusion_channels, strict=True)
            }
        )
        self.seed_refinement = nn.Sequential(
            *[ResidualBlock(decoder_channels) for _ in range(refinement_depth)]
        )
        self.refinements = nn.ModuleDict(
            {
                name: _TopDownRefinement(decoder_channels, class_dimension, refinement_depth)
                for name in self.level_names[:-1]
            }
        )
        self.auxiliary_heads = nn.ModuleDict(
            {name: nn.Conv2d(decoder_channels, 1, kernel_size=1) for name in self.level_names}
        )

        self.class_seed = nn.Sequential(
            nn.Linear(decoder_channels, class_dimension),
            nn.LayerNorm(class_dimension),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.initial_classifier = nn.Linear(class_dimension, 1)
        self.localization_feedback = nn.Sequential(
            nn.Linear(3 * decoder_channels, class_dimension),
            nn.LayerNorm(class_dimension),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.final_classifier = nn.Linear(2 * class_dimension, 1)
        self.class_to_localization = nn.Linear(class_dimension, decoder_channels)

        self.nuisance_predictor = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                groups=decoder_channels,
                bias=False,
            ),
            nn.GroupNorm(1, decoder_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1),
        )
        nn.init.zeros_(self.nuisance_predictor[-1].weight)
        nn.init.zeros_(self.nuisance_predictor[-1].bias)

        self.segmentation_head = nn.Sequential(
            ResidualBlock(decoder_channels),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.boundary_head = nn.Sequential(
            ResidualBlock(decoder_channels),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.confidence_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

        # Small initial coupling prevents either task from overwhelming the
        # other before useful representations have formed.
        self.raw_classification_to_localization = nn.Parameter(torch.tensor(-3.0))
        self.raw_denoising_strength = nn.Parameter(torch.tensor(-3.0))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        fusion_channels: Sequence[int],
        level_names: Sequence[str],
    ) -> SynergyDenoisingDecoder:
        return cls(
            fusion_channels=fusion_channels,
            level_names=level_names,
            decoder_channels=int(config.get("channels", 128)),
            class_dimension=int(config.get("class_dimension", 256)),
            refinement_depth=int(config.get("refinement_depth", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )

    @staticmethod
    def _weighted_pool(features: Tensor, weights: Tensor) -> Tensor:
        numerator = (features * weights).sum(dim=(-2, -1))
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        fused_features: Mapping[str, Tensor],
        *,
        output_size: tuple[int, int],
    ) -> SynergyDecoderOutput:
        missing = set(self.level_names).difference(fused_features)
        if missing:
            raise ValueError(f"Missing fused pyramid levels: {sorted(missing)}")
        if min(output_size) < 1:
            raise ValueError("output_size dimensions must be positive")

        deepest_name = self.level_names[-1]
        deepest = self.seed_refinement(self.lateral[deepest_name](fused_features[deepest_name]))
        global_pool = deepest.mean(dim=(-2, -1))
        class_context = self.class_seed(global_pool)
        initial_image_logits = self.initial_classifier(class_context)

        decoded: dict[str, Tensor] = {deepest_name: deepest}
        auxiliary: dict[str, Tensor] = {}
        state = deepest
        mask_logits = self.auxiliary_heads[deepest_name](state)
        auxiliary[deepest_name] = mask_logits

        for name in reversed(self.level_names[:-1]):
            lateral = self.lateral[name](fused_features[name])
            state = self.refinements[name](lateral, state, mask_logits, class_context)
            mask_logits = self.auxiliary_heads[name](state)
            decoded[name] = state
            auxiliary[name] = mask_logits

        localization_attention = mask_logits.sigmoid()
        suspicious = self._weighted_pool(state, localization_attention)
        background = self._weighted_pool(state, 1.0 - localization_attention)
        local_global = state.mean(dim=(-2, -1))
        feedback = self.localization_feedback(torch.cat((suspicious, background, local_global), dim=1))
        image_logits = self.final_classifier(torch.cat((class_context, feedback), dim=1))

        class_strength = self.raw_classification_to_localization.sigmoid()
        class_map = self.class_to_localization(feedback)[:, :, None, None]
        synergized = state + class_strength * image_logits.sigmoid()[:, :, None, None] * class_map

        denoising_strength = self.raw_denoising_strength.sigmoid()
        nuisance = torch.tanh(self.nuisance_predictor(synergized))
        denoised = synergized - denoising_strength * nuisance

        final_mask = self.segmentation_head(denoised)
        boundary = self.boundary_head(denoised)
        confidence = self.confidence_head(denoised)

        resize = lambda value: F.interpolate(  # noqa: E731
            value, size=output_size, mode="bilinear", align_corners=False
        )
        return SynergyDecoderOutput(
            mask_logits=resize(final_mask),
            boundary_logits=resize(boundary),
            image_logits=image_logits,
            evidence_confidence_logits=resize(confidence),
            auxiliary_mask_logits=auxiliary,
            decoded_features=decoded,
            localization_attention=resize(localization_attention),
            initial_image_logits=initial_image_logits,
            classification_to_localization_strength=class_strength,
            denoising_strength=denoising_strength,
        )
