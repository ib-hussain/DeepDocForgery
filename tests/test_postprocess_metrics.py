from __future__ import annotations

import torch

from src.evaluation.metrics import StreamingForgeryMetrics, binary_auroc
from src.outputLayer.postprocess import extract_instances


def test_extract_instances_filters_specks_and_returns_boxes() -> None:
    probability = torch.zeros(1, 1, 20, 24)
    probability[:, :, 2:8, 3:10] = 0.9
    probability[:, :, 12:18, 14:22] = 0.8
    probability[:, :, 0, 0] = 1.0
    instances = extract_instances(probability, threshold=0.5, minimum_area=4)[0]
    assert len(instances) == 2
    assert instances[0].box_xyxy == (3, 2, 10, 8)
    assert instances[1].box_xyxy == (14, 12, 22, 18)


def test_binary_auroc_is_exact_for_separated_scores() -> None:
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert binary_auroc([0.1, 0.2], [0, 0]) is None


def test_streaming_metrics_are_perfect_for_perfect_predictions() -> None:
    target = torch.zeros(2, 1, 24, 24)
    target[1, :, 5:15, 8:18] = 1.0
    metrics = StreamingForgeryMetrics(minimum_instance_area=4)
    metrics.update(
        mask_probability=target,
        tamper_mask=target,
        image_probability=torch.tensor([[0.0], [1.0]]),
        image_label=torch.tensor([[0.0], [1.0]]),
        valid_mask=torch.ones_like(target),
    )
    result = metrics.compute()
    assert result["pixel/f1"] == 1.0
    assert result["pixel/iou"] == 1.0
    assert result["image/accuracy"] == 1.0
    assert result["image/auroc"] == 1.0
    assert result["instance/f1"] == 1.0
