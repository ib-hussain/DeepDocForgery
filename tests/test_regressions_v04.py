from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from deepdocforgery.frequency import ExactDCTBatch
from deepdocforgery.io import load_yaml, make_synthetic_batch
from deepdocforgery.metrics import StreamingForgeryMetrics
from deepdocforgery.model import DeepDocForgeryModel
from deepdocforgery.objectives import DeepDocForgeryCriterion, DeepDocForgerySupervision
from deepdocforgery.runtime import CosineEpochScheduler


def _model_config() -> dict[str, object]:
    return deepcopy(load_yaml("configs/cpu/smoke.yaml")["model"])


def test_missing_adn_labels_do_not_alias_tamper_mask() -> None:
    model = DeepDocForgeryModel.from_config(_model_config()).eval()
    batch = make_synthetic_batch(batch_size=1, height=64, width=64, seed=31)
    output = model(batch.rgb, metadata=batch.metadata)
    supervision = DeepDocForgerySupervision(
        tamper_mask=batch.tamper_mask,
        image_label=torch.ones(1, 1),
        valid_mask=torch.ones_like(batch.tamper_mask),
        adn=None,
    )
    losses = DeepDocForgeryCriterion()(output, supervision)
    assert float(losses["adn/total"]) == 0.0
    assert float(losses["main/mask_bce"].detach()) > 0.0


def test_invalid_classifier_cannot_condition_localization() -> None:
    model = DeepDocForgeryModel.from_config(_model_config()).eval()
    batch = make_synthetic_batch(batch_size=1, height=64, width=64, seed=33)
    invalid = torch.zeros(1, 1)
    with torch.no_grad():
        model.decoder.final_classifier.bias.fill_(-10.0)
        first = model(batch.rgb, metadata=batch.metadata, classification_valid=invalid).mask_logits
        model.decoder.final_classifier.bias.fill_(10.0)
        second = model(batch.rgb, metadata=batch.metadata, classification_valid=invalid).mask_logits
    torch.testing.assert_close(first, second)


def test_detail_path_reaches_stride_two() -> None:
    model = DeepDocForgeryModel.from_config(_model_config()).eval()
    batch = make_synthetic_batch(batch_size=1, height=65, width=81, seed=35)
    with torch.no_grad():
        output = model(batch.rgb, metadata=batch.metadata)
    assert output.decoder.decoded_features["detail_s2"].shape[-2:] == (33, 41)
    assert output.mask_logits.shape[-2:] == (65, 81)


def test_fusion_branch_ablation_is_structural() -> None:
    config = _model_config()
    config["fusion"]["enabled_branches"] = ["spatial"]
    model = DeepDocForgeryModel.from_config(config).eval()
    batch = make_synthetic_batch(batch_size=1, height=64, width=64, seed=36)
    with torch.no_grad():
        output = model(batch.rgb, metadata=batch.metadata)
    attention = output.front_end.fusion.mean_attention()["s8"]
    assert float(attention["spatial"]) == 1.0
    assert float(attention["frequency"]) == 0.0
    assert float(attention["degradation"]) == 0.0


def test_mixed_exact_and_pixel_dct_batch() -> None:
    model = DeepDocForgeryModel.from_config(_model_config()).eval()
    batch = make_synthetic_batch(batch_size=2, height=64, width=64, seed=37)
    dct = model.front_end.forensics.dct_branch
    coefficients = dct.pixel_dct(
        batch.rgb, batch.metadata.qtables, batch.metadata.subsampling
    ).detach()
    exact = ExactDCTBatch(
        coefficients=coefficients,
        metadata=batch.metadata,
        valid=torch.tensor([1.0, 0.0]),
    )
    with torch.no_grad():
        output = model(
            batch.rgb,
            metadata=batch.metadata,
            exact_dct=exact,
            compute_consistency=True,
        )
    assert output.front_end.forensics.dct.source == "mixed_exact_and_fallback"
    assert bool(torch.isfinite(output.front_end.forensics.dct.consistency_loss))


def test_failure_tail_and_small_area_metrics_are_reported() -> None:
    metrics = StreamingForgeryMetrics(catastrophic_f1_threshold=0.1)
    target = torch.zeros(2, 1, 20, 20)
    target[0, :, :2, :2] = 1.0
    target[1, :, :5, :5] = 1.0
    prediction = torch.zeros_like(target)
    prediction[1] = target[1]
    metrics.update(
        mask_probability=prediction,
        tamper_mask=target,
        image_probability=torch.tensor([[0.2], [0.8]]),
        image_label=torch.ones(2, 1),
        valid_mask=torch.ones_like(target),
        image_valid=torch.zeros(2, 1),
    )
    result = metrics.compute()
    assert result["failure/catastrophic_miss_rate"] == 0.5
    assert result["area/lt_2pct/images"] == 1.0
    assert result["pixel/all_background_accuracy"] == 771 / 800
    assert result["image/auroc"] is None


def test_absolute_cosine_schedule_does_not_rebound_on_resume() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    first = CosineEpochScheduler(optimizer, total_epochs=10, minimum_learning_rate=1e-5)
    first.step(8)
    state = first.state_dict()
    learning_rate_at_eight = optimizer.param_groups[0]["lr"]
    resumed = CosineEpochScheduler(optimizer, total_epochs=10, minimum_learning_rate=1e-5)
    resumed.load_state_dict(state)
    resumed.step(9)
    assert optimizer.param_groups[0]["lr"] < learning_rate_at_eight


def test_cosine_schedule_rejects_changed_horizon() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    first = CosineEpochScheduler(optimizer, total_epochs=10, minimum_learning_rate=1e-5)
    with pytest.raises(ValueError, match="changed total epoch"):
        CosineEpochScheduler(
            optimizer, total_epochs=20, minimum_learning_rate=1e-5
        ).load_state_dict(first.state_dict())


def test_cpu_and_cuda_profiles_are_separate() -> None:
    cpu = load_yaml("configs/cpu/sample.yaml")
    cuda = load_yaml("configs/cuda/full.yaml")
    assert cpu["data"]["manifest"].endswith("cpu.jsonl")
    assert cuda["data"]["manifest"].endswith("cuda.jsonl")
    assert cpu["data"]["manifest"].startswith("output/manifests/")
    assert cuda["data"]["manifest"].startswith("output/manifests/")
    assert cpu["model"]["spatial"]["vit"]["backend"] == "native"
    assert cuda["model"]["spatial"]["vit"]["backend"] == "timm"
    assert cuda["data"]["batch_size"] > cpu["data"]["batch_size"]
    assert cuda["training"]["amp"] is True
