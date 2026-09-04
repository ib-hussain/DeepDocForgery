"""Joint localization, classification, boundary, and synergy objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import SynergyDecoderOutput


@dataclass
class SynergySupervision:
    """Ground truth for the final DeepDocForgery tasks."""

    tamper_mask: Tensor
    image_label: Tensor
    valid_mask: Tensor | None = None
    image_valid: Tensor | None = None

    def to(self, *args: Any, **kwargs: Any) -> SynergySupervision:
        return SynergySupervision(
            tamper_mask=self.tamper_mask.to(*args, **kwargs),
            image_label=self.image_label.to(*args, **kwargs),
            valid_mask=(None if self.valid_mask is None else self.valid_mask.to(*args, **kwargs)),
            image_valid=(
                None if self.image_valid is None else self.image_valid.to(*args, **kwargs)
            ),
        )


def _validate_supervision(
    supervision: SynergySupervision,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    target = supervision.tamper_mask.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("tamper_mask must have shape [B,1,H,W] or [B,H,W]")
    labels = supervision.image_label.float()
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.shape != (target.shape[0], 1):
        raise ValueError("image_label must have shape [B] or [B,1]")
    if supervision.valid_mask is None:
        valid = torch.ones_like(target)
    else:
        valid = supervision.valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape != target.shape:
            raise ValueError("valid_mask must have the same shape as tamper_mask")
    if supervision.image_valid is None:
        image_valid = torch.ones_like(labels)
    else:
        image_valid = supervision.image_valid.float()
        if image_valid.ndim == 1:
            image_valid = image_valid.unsqueeze(1)
        if image_valid.shape != labels.shape:
            raise ValueError("image_valid must have shape [B] or [B,1]")
    return (
        target.clamp(0.0, 1.0),
        labels.clamp(0.0, 1.0),
        valid.clamp(0.0, 1.0),
        image_valid.clamp(0.0, 1.0),
    )


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    if valid.shape != values.shape:
        valid = valid.expand_as(values)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def _dice_loss(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _boundary_target(mask: Tensor, valid: Tensor) -> Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0) * valid


class SynergyMultiTaskLoss(nn.Module):
    """Supervise both tasks and the information exchanged between them."""

    def __init__(
        self,
        *,
        mask_bce_weight: float = 1.0,
        mask_dice_weight: float = 1.0,
        image_weight: float = 1.0,
        boundary_weight: float = 0.25,
        auxiliary_weight: float = 0.4,
        confidence_weight: float = 0.1,
        denoising_weight: float = 0.1,
        agreement_weight: float = 0.2,
        topk_fraction: float = 0.01,
        positive_pixel_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        if positive_pixel_weight <= 0:
            raise ValueError("positive_pixel_weight must be positive")
        self.weights = {
            "mask_bce": float(mask_bce_weight),
            "mask_dice": float(mask_dice_weight),
            "image": float(image_weight),
            "boundary": float(boundary_weight),
            "auxiliary": float(auxiliary_weight),
            "confidence": float(confidence_weight),
            "denoising": float(denoising_weight),
            "agreement": float(agreement_weight),
        }
        self.topk_fraction = float(topk_fraction)
        self.positive_pixel_weight = float(positive_pixel_weight)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SynergyMultiTaskLoss:
        return cls(
            mask_bce_weight=float(config.get("mask_bce_weight", 1.0)),
            mask_dice_weight=float(config.get("mask_dice_weight", 1.0)),
            image_weight=float(config.get("image_weight", 1.0)),
            boundary_weight=float(config.get("boundary_weight", 0.25)),
            auxiliary_weight=float(config.get("auxiliary_weight", 0.4)),
            confidence_weight=float(config.get("confidence_weight", 0.1)),
            denoising_weight=float(config.get("denoising_weight", 0.1)),
            agreement_weight=float(config.get("agreement_weight", 0.2)),
            topk_fraction=float(config.get("topk_fraction", 0.01)),
            positive_pixel_weight=float(config.get("positive_pixel_weight", 1.0)),
        )

    def _mask_presence(self, probability: Tensor, valid: Tensor) -> Tensor:
        flattened = (probability * valid).flatten(1)
        valid_counts = valid.flatten(1).sum(dim=1).clamp_min(1.0)
        # A single batch-wide k keeps this operation vectorized. Padded pixels
        # have zero probability and therefore do not enter the top values.
        k = max(1, int(round(float(valid_counts.max()) * self.topk_fraction)))
        k = min(k, flattened.shape[1])
        return flattened.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def forward(
        self,
        predictions: SynergyDecoderOutput,
        supervision: SynergySupervision,
    ) -> dict[str, Tensor]:
        target, labels, valid, image_valid = _validate_supervision(supervision)
        if predictions.mask_logits.shape != target.shape:
            raise ValueError("Full-resolution mask prediction and target shapes must match")
        if predictions.image_logits.shape != labels.shape:
            raise ValueError("Image prediction and label shapes must match")

        mask_bce = _masked_mean(
            F.binary_cross_entropy_with_logits(
                predictions.mask_logits,
                target,
                reduction="none",
                pos_weight=target.new_tensor(self.positive_pixel_weight),
            ),
            valid,
        )
        mask_dice = _dice_loss(predictions.mask_logits, target, valid)
        final_image = _masked_mean(
            F.binary_cross_entropy_with_logits(predictions.image_logits, labels, reduction="none"),
            image_valid,
        )
        initial_image = _masked_mean(
            F.binary_cross_entropy_with_logits(
                predictions.initial_image_logits, labels, reduction="none"
            ),
            image_valid,
        )
        image = final_image + 0.25 * initial_image

        boundary_target = _boundary_target(target, valid)
        boundary = _masked_mean(
            F.binary_cross_entropy_with_logits(
                predictions.boundary_logits, boundary_target, reduction="none"
            ),
            valid,
        )

        auxiliary_losses: list[Tensor] = []
        consistency_losses: list[Tensor] = []
        final_probability = predictions.mask_probability
        for auxiliary_logits in predictions.auxiliary_mask_logits.values():
            size = auxiliary_logits.shape[-2:]
            auxiliary_target = F.interpolate(target, size=size, mode="nearest")
            auxiliary_valid = F.interpolate(valid, size=size, mode="nearest")
            auxiliary_losses.append(
                _masked_mean(
                    F.binary_cross_entropy_with_logits(
                        auxiliary_logits,
                        auxiliary_target,
                        reduction="none",
                        pos_weight=auxiliary_target.new_tensor(self.positive_pixel_weight),
                    ),
                    auxiliary_valid,
                )
                + _dice_loss(auxiliary_logits, auxiliary_target, auxiliary_valid)
            )
            final_at_scale = F.interpolate(
                final_probability.detach(), size=size, mode="bilinear", align_corners=False
            )
            consistency_losses.append(
                _masked_mean((auxiliary_logits.sigmoid() - final_at_scale).abs(), auxiliary_valid)
            )
        auxiliary = torch.stack(auxiliary_losses).mean()
        denoising = torch.stack(consistency_losses).mean()

        correctness = 1.0 - (final_probability.detach() - target).abs()
        confidence = _masked_mean(
            F.binary_cross_entropy_with_logits(
                predictions.evidence_confidence_logits,
                correctness.clamp(0.0, 1.0),
                reduction="none",
            ),
            valid,
        )

        mask_presence = self._mask_presence(final_probability, valid).clamp(1e-6, 1.0 - 1e-6)
        classification_probability = predictions.image_probability
        agreement_per_sample = F.binary_cross_entropy(mask_presence, labels, reduction="none")
        agreement_per_sample = agreement_per_sample + F.smooth_l1_loss(
            mask_presence, classification_probability, reduction="none"
        )
        localization_valid = (valid.flatten(1).sum(dim=1, keepdim=True) > 0).to(
            agreement_per_sample.dtype
        ) * image_valid
        agreement = (agreement_per_sample * localization_valid).sum()
        agreement = agreement / localization_valid.sum().clamp_min(1.0)

        components = {
            "mask_bce": mask_bce,
            "mask_dice": mask_dice,
            "image": image,
            "boundary": boundary,
            "auxiliary": auxiliary,
            "confidence": confidence,
            "denoising": denoising,
            "agreement": agreement,
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return {
            "total": total,
            **components,
            "image_final": final_image,
            "image_initial": initial_image,
        }
