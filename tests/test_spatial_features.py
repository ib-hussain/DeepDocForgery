from __future__ import annotations

import torch
from torch.nn import functional as F

from deepdocforgery.spatial import (
    ADNSupervision,
    ArtifactDecouplingLoss,
    SpatialFeaturePyramid,
)


def _small_spatial() -> SpatialFeaturePyramid:
    return SpatialFeaturePyramid(
        out_channels=(16, 24, 32),
        level_names=("s8", "s16", "s32"),
        vit_stem_channels=12,
        vit_depths=(1, 1, 1),
        vit_number_of_heads=(2, 3, 4),
        vit_window_size=4,
        adn_stem_channels=12,
        adn_depths=(1, 1, 1),
    )


def test_spatial_pyramid_aligns_odd_image_sizes_and_exposes_attention() -> None:
    model = _small_spatial().eval()
    rgb = torch.rand(2, 3, 65, 81)
    with torch.no_grad():
        output = model(rgb)

    expected = {
        "s8": (2, 16, 9, 11),
        "s16": (2, 24, 5, 6),
        "s32": (2, 32, 3, 3),
    }
    for name, shape in expected.items():
        assert output.features[name].shape == shape
        assert output.vit_features[name].shape == shape
        assert output.adn_features[name].shape == shape
        assert output.backbone_attention[name].shape == (2, 2, shape[1], 1, 1)
        torch.testing.assert_close(
            output.backbone_attention[name],
            torch.full_like(output.backbone_attention[name], 0.5),
        )
    assert output.adn_auxiliary_logits.shape == (2, 1, 3, 3)
    assert output.auxiliary_probability((65, 81)).shape == (2, 1, 65, 81)


def test_artifact_decoupling_objectives_and_spatial_gradient_flow() -> None:
    model = _small_spatial().train()
    rgb = torch.rand(2, 3, 64, 80)
    mask = torch.zeros(2, 1, 64, 80)
    mask[:, :, 16:48, 20:60] = 1.0
    region_type = torch.zeros(2, 64, 80, dtype=torch.long)
    region_type[0, 16:48, 20:60] = 1
    region_type[1, 16:48, 20:60] = 2
    output = model(rgb)
    targets = ADNSupervision(
        artifact_mask=mask,
        domain_index=torch.tensor([0, 1]),
        region_type=region_type,
    )
    losses = ArtifactDecouplingLoss()(output, targets)
    resized_mask = F.interpolate(mask, size=output.features["s8"].shape[-2:], mode="nearest")
    spatial_probe = F.binary_cross_entropy_with_logits(
        output.features["s8"].mean(dim=1, keepdim=True),
        resized_mask,
    )
    total = losses["total"] + spatial_probe
    assert bool(torch.isfinite(total))
    total.backward()

    vit_gradient = model.vit.patch_stem.projection.weight.grad
    adn_gradient = model.adn.patch_stem.projection.weight.grad
    attention_gradient = model.fusions["s8"].reduction[0][0].weight.grad
    for gradient in (vit_gradient, adn_gradient, attention_gradient):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0
    assert set(losses) == {
        "total",
        "text_bce",
        "nontext_bce",
        "text_decoupling",
        "domain_alignment",
    }


def test_spatial_config_defaults_resize_to_custom_level_count() -> None:
    model = SpatialFeaturePyramid.from_config(
        {
            "out_channels": [8],
            "level_names": ["s8"],
            "vit": {"stem_channels": 8, "number_of_heads": [2], "window_size": 4},
            "adn": {"stem_channels": 8},
        }
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 32, 40))
    assert output.features["s8"].shape == (1, 8, 4, 5)
