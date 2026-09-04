"""One criterion joining every supervised DeepDocForgery component."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from deepdocforgery.degradation import (
    DegradationTargets,
    MultiScaleDegradationLoss,
)
from deepdocforgery.losses import SynergyMultiTaskLoss, SynergySupervision
from deepdocforgery.model import DeepDocForgeryOutput
from deepdocforgery.spatial import (
    ADNSupervision,
    ArtifactDecouplingLoss,
)


@dataclass
class DeepDocForgerySupervision:
    tamper_mask: Tensor
    image_label: Tensor
    valid_mask: Tensor | None = None
    image_valid: Tensor | None = None
    degradation: DegradationTargets | None = None
    adn: ADNSupervision | None = None

    def to(self, *args: Any, **kwargs: Any) -> DeepDocForgerySupervision:
        return DeepDocForgerySupervision(
            tamper_mask=self.tamper_mask.to(*args, **kwargs),
            image_label=self.image_label.to(*args, **kwargs),
            valid_mask=(None if self.valid_mask is None else self.valid_mask.to(*args, **kwargs)),
            image_valid=(
                None if self.image_valid is None else self.image_valid.to(*args, **kwargs)
            ),
            degradation=(
                None if self.degradation is None else self.degradation.to(*args, **kwargs)
            ),
            adn=None if self.adn is None else self.adn.to(*args, **kwargs),
        )


class DeepDocForgeryCriterion(nn.Module):
    """Weighted end-to-end objective with individually reported components."""

    def __init__(
        self,
        *,
        main: SynergyMultiTaskLoss | None = None,
        degradation: MultiScaleDegradationLoss | None = None,
        adn: ArtifactDecouplingLoss | None = None,
        main_weight: float = 1.0,
        degradation_weight: float = 0.25,
        adn_weight: float = 0.25,
        dct_consistency_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.main = main or SynergyMultiTaskLoss()
        self.degradation = degradation or MultiScaleDegradationLoss()
        self.adn = adn or ArtifactDecouplingLoss()
        self.weights = {
            "main": float(main_weight),
            "degradation": float(degradation_weight),
            "adn": float(adn_weight),
            "dct_consistency": float(dct_consistency_weight),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DeepDocForgeryCriterion:
        return cls(
            main=SynergyMultiTaskLoss.from_config(config.get("main", config)),
            degradation=MultiScaleDegradationLoss.from_config(config),
            adn=ArtifactDecouplingLoss.from_config(config.get("adn", {})),
            main_weight=float(config.get("main_weight", 1.0)),
            degradation_weight=float(config.get("degradation_weight", 0.25)),
            adn_weight=float(config.get("adn_weight", 0.25)),
            dct_consistency_weight=float(config.get("dct_consistency_weight", 1.0)),
        )

    def forward(
        self,
        predictions: DeepDocForgeryOutput,
        supervision: DeepDocForgerySupervision,
    ) -> dict[str, Tensor]:
        main_losses = self.main(
            predictions.decoder,
            SynergySupervision(
                tamper_mask=supervision.tamper_mask,
                image_label=supervision.image_label,
                valid_mask=supervision.valid_mask,
                image_valid=supervision.image_valid,
            ),
        )
        zero = main_losses["total"].new_zeros(())
        if supervision.adn is None:
            adn_losses: dict[str, Tensor] = {
                "total": zero,
                "text_bce": zero,
                "nontext_bce": zero,
                "text_decoupling": zero,
                "domain_alignment": zero,
            }
        else:
            adn_losses = self.adn(predictions.front_end.spatial, supervision.adn)
        if supervision.degradation is None:
            degradation_losses: dict[str, Tensor] = {
                "total": zero,
                "quality": zero,
                "double_compression": zero,
                "noise_type": zero,
                "noise_strength": zero,
            }
        else:
            degradation_losses = self.degradation(
                predictions.front_end.forensics.degradation,
                supervision.degradation,
            )
        dct_consistency = predictions.front_end.forensics.dct.consistency_loss
        total = (
            self.weights["main"] * main_losses["total"]
            + self.weights["degradation"] * degradation_losses["total"]
            + self.weights["adn"] * adn_losses["total"]
            + self.weights["dct_consistency"] * dct_consistency
        )
        result = {"total": total, "dct_consistency": dct_consistency}
        result.update({f"main/{name}": value for name, value in main_losses.items()})
        result.update({f"degradation/{name}": value for name, value in degradation_losses.items()})
        result.update({f"adn/{name}": value for name, value in adn_losses.items()})
        return result
