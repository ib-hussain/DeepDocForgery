from __future__ import annotations

import pytest
import torch

from deepdocforgery.degradation import (
    InputForensicsFrontEnd,
    MultiScaleDegradationLoss,
)
from deepdocforgery.io import make_synthetic_batch


def _small_model() -> InputForensicsFrontEnd:
    return InputForensicsFrontEnd.from_config(
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
        }
    )


def test_regional_output_shapes_ranges_and_gate_contract() -> None:
    model = _small_model().eval()
    batch = make_synthetic_batch(batch_size=2, height=64, width=80, seed=11)
    with torch.no_grad():
        output = model(batch.rgb, metadata=batch.metadata)

    expected_sizes = {"s8": (8, 10), "s16": (4, 5), "s32": (2, 3)}
    for name, size in expected_sizes.items():
        level = output.degradation.levels[name]
        assert level.jpeg_quality.shape == (2, 1, *size)
        assert level.double_compression_logits.shape == (2, 1, *size)
        assert level.noise_type_logits.shape == (2, 4, *size)
        assert level.noise_strength.shape == (2, 1, *size)
        assert bool((level.jpeg_quality >= 1.0).all())
        assert bool((level.jpeg_quality <= 100.0).all())
        assert bool((level.noise_strength >= 0.0).all())
        assert bool((level.noise_strength <= 0.2).all())
        assert output.degradation.gate_features()[name].shape == (2, 7, *size)


def test_multitask_loss_backpropagates_through_both_branches() -> None:
    model = _small_model().train()
    batch = make_synthetic_batch(batch_size=2, height=64, width=80, seed=17)
    output = model(batch.rgb, metadata=batch.metadata)
    losses = MultiScaleDegradationLoss()(output.degradation, batch.targets)
    total = losses["total"] + output.dct.consistency_loss
    assert bool(torch.isfinite(total))
    total.backward()

    dct_gradient = model.dct_branch.initial_fusion[0][0].weight.grad
    head_gradient = model.degradation_estimator.heads["s8"].predictor[-1].weight.grad
    assert dct_gradient is not None and bool(torch.isfinite(dct_gradient).all())
    assert head_gradient is not None and bool(torch.isfinite(head_gradient).all())
    assert float(dct_gradient.abs().sum()) > 0.0
    assert float(head_gradient.abs().sum()) > 0.0
    assert set(losses) == {
        "total",
        "quality",
        "double_compression",
        "noise_type",
        "noise_strength",
    }


def test_custom_noise_taxonomy_changes_head_and_gate_width() -> None:
    model = InputForensicsFrontEnd.from_config(
        {
            "dct": {"out_channels": [8], "level_names": ["s8"], "stem_channels": 4},
            "degradation_estimator": {
                "metadata_dimension": 4,
                "noise_types": ["none", "gaussian"],
                "fusion_channels": [8],
            },
        }
    ).eval()
    batch = make_synthetic_batch(
        batch_size=1,
        height=32,
        width=32,
        number_of_noise_types=2,
        seed=3,
    )
    with torch.no_grad():
        output = model(batch.rgb, metadata=batch.metadata)
    assert output.degradation.levels["s8"].noise_type_logits.shape == (1, 2, 4, 4)
    assert output.degradation.gate_features()["s8"].shape == (1, 5, 4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_forward_and_backward() -> None:
    device = torch.device("cuda")
    model = _small_model().to(device).train()
    batch = make_synthetic_batch(
        batch_size=1,
        height=64,
        width=64,
        device=device,
        seed=31,
    )
    output = model(batch.rgb, metadata=batch.metadata)
    losses = MultiScaleDegradationLoss()(output.degradation, batch.targets)
    losses["total"].backward()
    gradient = model.dct_branch.initial_fusion[0][0].weight.grad
    assert gradient is not None
    assert gradient.device.type == "cuda"
    assert bool(torch.isfinite(gradient).all())
