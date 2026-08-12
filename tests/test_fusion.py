from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.hiddenLayers.fusion import DeepDocForgeryFusionFrontEnd
from src.inputLayer.dataModelling import make_synthetic_batch
from src.inputLayer.degradationEstimator import MultiScaleDegradationLoss
from src.inputLayer.spatialFeature import ADNSupervision, ArtifactDecouplingLoss


def _small_model() -> DeepDocForgeryFusionFrontEnd:
    return DeepDocForgeryFusionFrontEnd.from_config(
        {
            "dct": {
                "stem_channels": 8,
                "out_channels": [16, 24, 32],
                "level_names": ["s8", "s16", "s32"],
                "consistency_weight": 0.1,
            },
            "degradation_estimator": {
                "metadata_dimension": 8,
                "fusion_channels": [16, 24, 32],
                "noise_types": ["none", "gaussian", "poisson", "speckle"],
                "maximum_noise_strength": 0.2,
            },
            "spatial": {
                "out_channels": [16, 24, 32],
                "level_names": ["s8", "s16", "s32"],
                "vit": {
                    "stem_channels": 12,
                    "depths": [1, 1, 1],
                    "number_of_heads": [2, 3, 4],
                    "window_size": 4,
                },
                "adn": {"stem_channels": 12, "depths": [1, 1, 1]},
            },
            "fusion": {
                "out_channels": [16, 24, 32],
                "attention_hidden_ratio": 0.5,
                "learnable_temperature": True,
                "refinement_depth": 1,
            },
        }
    )


def test_full_frontend_shapes_and_neutral_attention_initialization() -> None:
    model = _small_model().eval()
    batch = make_synthetic_batch(batch_size=2, height=65, width=81, seed=41)
    with torch.no_grad():
        output = model(batch.rgb, metadata=batch.metadata)

    expected = {
        "s8": (2, 16, 9, 11),
        "s16": (2, 24, 5, 6),
        "s32": (2, 32, 3, 3),
    }
    assert output.fusion.branch_names == ("spatial", "frequency", "degradation")
    for name, feature_shape in expected.items():
        level = output.fusion.levels[name]
        assert level.features.shape == feature_shape
        assert level.attention_weights.shape == (
            feature_shape[0],
            3,
            feature_shape[1],
            feature_shape[2],
            feature_shape[3],
        )
        torch.testing.assert_close(
            level.attention_weights.sum(dim=1),
            torch.ones_like(level.attention_weights[:, 0]),
        )
        torch.testing.assert_close(
            level.attention_weights,
            torch.full_like(level.attention_weights, 1.0 / 3.0),
            atol=1e-6,
            rtol=1e-6,
        )
        assert set(level.branch_features) == {
            "spatial",
            "frequency",
            "degradation",
        }


def test_downstream_loss_updates_all_branches_and_attention_gate() -> None:
    model = _small_model().train()
    probe = nn.Conv2d(16, 1, kernel_size=1)
    batch = make_synthetic_batch(batch_size=2, height=64, width=80, seed=43)
    output = model(batch.rgb, metadata=batch.metadata)
    degradation_losses = MultiScaleDegradationLoss()(
        output.forensics.degradation,
        batch.targets,
    )
    adn_losses = ArtifactDecouplingLoss()(
        output.spatial,
        ADNSupervision(artifact_mask=batch.tamper_mask),
    )
    mask = F.interpolate(
        batch.tamper_mask,
        size=output.fusion.features["s8"].shape[-2:],
        mode="nearest",
    )
    fusion_loss = F.binary_cross_entropy_with_logits(
        probe(output.fusion.features["s8"]),
        mask,
    )
    total = degradation_losses["total"] + adn_losses["total"] + fusion_loss
    assert bool(torch.isfinite(total))
    total.backward()

    gradients = {
        "vit": model.spatial.vit.patch_stem.projection.weight.grad,
        "adn": model.spatial.adn.patch_stem.projection.weight.grad,
        "dct": model.forensics.dct_branch.initial_fusion[0][0].weight.grad,
        "degradation": model.fusion.degradation_projections["s8"][0].weight.grad,
        "attention": model.fusion.attention_heads["s8"].local[-1].weight.grad,
    }
    for name, gradient in gradients.items():
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name
        assert float(gradient.abs().sum()) > 0.0, name


def test_attention_weights_change_after_gate_parameters_change() -> None:
    model = _small_model().eval()
    batch = make_synthetic_batch(batch_size=1, height=64, width=64, seed=47)
    with torch.no_grad():
        neutral = model(batch.rgb, metadata=batch.metadata)
        gate = model.fusion.attention_heads["s8"].local[-1]
        gate.bias[:16].fill_(1.0)
        routed = model(batch.rgb, metadata=batch.metadata)
    neutral_spatial = neutral.fusion.levels["s8"].attention_weights[:, 0].mean()
    routed_spatial = routed.fusion.levels["s8"].attention_weights[:, 0].mean()
    assert float(routed_spatial) > float(neutral_spatial)
    assert float(routed_spatial) > 1.0 / 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_full_fusion_cuda_forward_and_backward() -> None:
    device = torch.device("cuda")
    model = _small_model().to(device).train()
    batch = make_synthetic_batch(
        batch_size=1,
        height=64,
        width=64,
        device=device,
        seed=53,
    )
    output = model(batch.rgb, metadata=batch.metadata)
    loss = sum(value.features.square().mean() for value in output.fusion.levels.values())
    loss.backward()
    gradient = model.fusion.attention_heads["s8"].local[-1].weight.grad
    assert gradient is not None
    assert gradient.device.type == "cuda"
    assert bool(torch.isfinite(gradient).all())

