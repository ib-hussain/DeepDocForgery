"""Trainable gates and fusion layers for DeepDocForgery."""

from .fusion import (
    AttentionFusionOutput,
    DeepDocForgeryFusionFrontEnd,
    DeepDocForgeryFusionOutput,
    FusionLevelOutput,
    MultiScaleAttentionFusion,
)

__all__ = [
    "AttentionFusionOutput",
    "DeepDocForgeryFusionFrontEnd",
    "DeepDocForgeryFusionOutput",
    "FusionLevelOutput",
    "MultiScaleAttentionFusion",
]
