"""Complete DeepDocForgery model from input evidence through final outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from deepdocforgery.decoder import SynergyDecoderOutput, SynergyDenoisingDecoder
from deepdocforgery.frequency import ExactDCTBatch, JPEGMetadata
from deepdocforgery.fusion import (
    DeepDocForgeryFusionFrontEnd,
    DeepDocForgeryFusionOutput,
)


@dataclass
class DeepDocForgeryOutput:
    """All final predictions plus the evidence needed for loss/audit tooling."""

    front_end: DeepDocForgeryFusionOutput
    decoder: SynergyDecoderOutput

    @property
    def mask_logits(self) -> Tensor:
        return self.decoder.mask_logits

    @property
    def image_logits(self) -> Tensor:
        return self.decoder.image_logits


class DeepDocForgeryModel(nn.Module):
    """End-to-end trainable model with no temporary probe heads."""

    def __init__(
        self,
        front_end: DeepDocForgeryFusionFrontEnd,
        decoder: SynergyDenoisingDecoder,
    ) -> None:
        super().__init__()
        if front_end.fusion.level_names != decoder.level_names:
            raise ValueError("Fusion and decoder level names must match")
        if front_end.fusion.out_channels != decoder.fusion_channels:
            raise ValueError("Fusion output channels must match the decoder contract")
        self.front_end = front_end
        self.decoder = decoder

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, load_pretrained: bool = True
    ) -> DeepDocForgeryModel:
        front_end = DeepDocForgeryFusionFrontEnd.from_config(
            config, load_pretrained=load_pretrained
        )
        decoder = SynergyDenoisingDecoder.from_config(
            config.get("decoder", {}),
            fusion_channels=front_end.fusion.out_channels,
            level_names=front_end.fusion.level_names,
        )
        return cls(front_end, decoder)

    def forward(
        self,
        rgb: Tensor,
        *,
        metadata: JPEGMetadata | None = None,
        exact_dct: ExactDCTBatch | None = None,
        compute_consistency: bool | None = None,
        classification_valid: Tensor | None = None,
    ) -> DeepDocForgeryOutput:
        front_end = self.front_end(
            rgb,
            metadata=metadata,
            exact_dct=exact_dct,
            compute_consistency=compute_consistency,
        )
        decoder = self.decoder(
            front_end.fusion.features,
            output_size=rgb.shape[-2:],
            rgb=rgb,
            classification_valid=classification_valid,
        )
        return DeepDocForgeryOutput(front_end=front_end, decoder=decoder)
