from __future__ import annotations

import torch

from src.inputLayer.dataModelling import make_synthetic_batch
from src.model import DeepDocForgeryModel
from src.training.objectives import DeepDocForgeryCriterion, DeepDocForgerySupervision


def _small_config() -> dict[str, object]:
    return {
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
            "refinement_depth": 1,
        },
        "decoder": {
            "channels": 24,
            "class_dimension": 32,
            "refinement_depth": 1,
            "dropout": 0.0,
        },
    }


def test_complete_model_outputs_full_resolution_predictions() -> None:
    model = DeepDocForgeryModel.from_config(_small_config()).eval()
    batch = make_synthetic_batch(batch_size=2, height=65, width=81, seed=71)
    with torch.no_grad():
        output = model(batch.rgb, metadata=batch.metadata)
    assert output.decoder.mask_logits.shape == (2, 1, 65, 81)
    assert output.decoder.boundary_logits.shape == (2, 1, 65, 81)
    assert output.decoder.evidence_confidence_logits.shape == (2, 1, 65, 81)
    assert output.decoder.image_logits.shape == (2, 1)
    assert set(output.decoder.auxiliary_mask_logits) == {"s8", "s16", "s32"}
    assert 0.0 < float(output.decoder.classification_to_localization_strength) < 0.1
    assert 0.0 < float(output.decoder.denoising_strength) < 0.1
    assert bool((output.decoder.mask_probability >= 0.0).all())
    assert bool((output.decoder.mask_probability <= 1.0).all())


def test_joint_objective_reaches_decoder_and_fusion_attention() -> None:
    model = DeepDocForgeryModel.from_config(_small_config()).train()
    criterion = DeepDocForgeryCriterion()
    batch = make_synthetic_batch(batch_size=2, height=64, width=80, seed=73)
    labels = torch.ones(2, 1)
    supervision = DeepDocForgerySupervision(
        tamper_mask=batch.tamper_mask,
        image_label=labels,
        valid_mask=torch.ones_like(batch.tamper_mask),
        degradation=batch.targets,
    )
    output = model(batch.rgb, metadata=batch.metadata)
    losses = criterion(output, supervision)
    assert bool(torch.isfinite(losses["total"]))
    losses["total"].backward()
    gradients = {
        "decoder": model.decoder.segmentation_head[-1].weight.grad,
        "classification": model.decoder.final_classifier.weight.grad,
        "fusion": model.front_end.fusion.attention_heads["s8"].local[-1].weight.grad,
        "dct": model.front_end.forensics.dct_branch.initial_fusion[0][0].weight.grad,
        "vit": model.front_end.spatial.vit.patch_stem.projection.weight.grad,
    }
    for name, gradient in gradients.items():
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name
        assert float(gradient.abs().sum()) > 0.0, name


def test_unlabelled_mask_regions_are_excluded_from_localization_loss() -> None:
    model = DeepDocForgeryModel.from_config(_small_config()).eval()
    batch = make_synthetic_batch(batch_size=1, height=64, width=64, seed=79)
    output = model(batch.rgb, metadata=batch.metadata)
    criterion = DeepDocForgeryCriterion()
    supervision = DeepDocForgerySupervision(
        tamper_mask=torch.zeros_like(batch.tamper_mask),
        image_label=torch.ones(1, 1),
        valid_mask=torch.zeros_like(batch.tamper_mask),
    )
    losses = criterion(output, supervision)
    assert float(losses["main/mask_bce"]) == 0.0
    assert float(losses["main/mask_dice"]) == 0.0
    assert bool(torch.isfinite(losses["total"]))


def test_complete_model_state_dict_round_trip() -> None:
    first = DeepDocForgeryModel.from_config(_small_config()).eval()
    second = DeepDocForgeryModel.from_config(_small_config()).eval()
    second.load_state_dict(first.state_dict())
    batch = make_synthetic_batch(batch_size=1, height=64, width=64, seed=83)
    with torch.no_grad():
        first_output = first(batch.rgb, metadata=batch.metadata)
        second_output = second(batch.rgb, metadata=batch.metadata)
    torch.testing.assert_close(first_output.mask_logits, second_output.mask_logits)
    torch.testing.assert_close(first_output.image_logits, second_output.image_logits)
