"""Dependency-light metrics with explicit zero-denominator behavior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.outputLayer.postprocess import InstancePrediction, extract_instances


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def binary_auroc(scores: list[float], labels: list[int]) -> float | None:
    """Exact trapezoidal ROC AUC; return None when only one class is present."""

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_tpr = 0.0
    previous_fpr = 0.0
    area = 0.0
    index = 0
    while index < len(order):
        threshold = scores[order[index]]
        while index < len(order) and scores[order[index]] == threshold:
            label = labels[order[index]]
            true_positives += int(label == 1)
            false_positives += int(label == 0)
            index += 1
        tpr = true_positives / positives
        fpr = false_positives / negatives
        area += (fpr - previous_fpr) * (tpr + previous_tpr) * 0.5
        previous_tpr, previous_fpr = tpr, fpr
    return area


@dataclass
class _Confusion:
    true_positive: float = 0.0
    false_positive: float = 0.0
    true_negative: float = 0.0
    false_negative: float = 0.0

    def update(self, prediction: Tensor, target: Tensor, valid: Tensor) -> None:
        prediction = prediction.bool()
        target = target.bool()
        valid = valid.bool()
        self.true_positive += float((prediction & target & valid).sum())
        self.false_positive += float((prediction & ~target & valid).sum())
        self.true_negative += float((~prediction & ~target & valid).sum())
        self.false_negative += float((~prediction & target & valid).sum())

    def metrics(self, prefix: str) -> dict[str, float]:
        tp, fp, tn, fn = (
            self.true_positive,
            self.false_positive,
            self.true_negative,
            self.false_negative,
        )
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        return {
            f"{prefix}/precision": precision,
            f"{prefix}/recall": recall,
            f"{prefix}/f1": _safe_divide(2.0 * precision * recall, precision + recall),
            f"{prefix}/iou": _safe_divide(tp, tp + fp + fn),
            f"{prefix}/accuracy": _safe_divide(tp + tn, tp + fp + tn + fn),
        }


def _box_iou(first: InstancePrediction, second: InstancePrediction) -> float:
    ax1, ay1, ax2, ay2 = first.box_xyxy
    bx1, by1, bx2, by2 = second.box_xyxy
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_box_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_box_area + second_box_area - intersection
    return _safe_divide(float(intersection), float(union))


class StreamingForgeryMetrics:
    """Accumulate fixed-threshold pixel, image, instance, and image-AUC metrics."""

    def __init__(
        self,
        *,
        mask_threshold: float = 0.5,
        image_threshold: float = 0.5,
        minimum_instance_area: int = 16,
        instance_iou_threshold: float = 0.5,
    ) -> None:
        self.mask_threshold = float(mask_threshold)
        self.image_threshold = float(image_threshold)
        self.minimum_instance_area = int(minimum_instance_area)
        self.instance_iou_threshold = float(instance_iou_threshold)
        self.pixel = _Confusion()
        self.image = _Confusion()
        self.image_scores: list[float] = []
        self.image_labels: list[int] = []
        self.instance_tp = 0.0
        self.instance_fp = 0.0
        self.instance_fn = 0.0

    def update(
        self,
        *,
        mask_probability: Tensor,
        tamper_mask: Tensor,
        image_probability: Tensor,
        image_label: Tensor,
        valid_mask: Tensor | None = None,
    ) -> None:
        mask_probability = mask_probability.detach().float().cpu()
        tamper_mask = tamper_mask.detach().float().cpu()
        if tamper_mask.ndim == 3:
            tamper_mask = tamper_mask.unsqueeze(1)
        valid = (
            torch.ones_like(tamper_mask)
            if valid_mask is None
            else valid_mask.detach().float().cpu()
        )
        self.pixel.update(
            mask_probability >= self.mask_threshold,
            tamper_mask >= 0.5,
            valid >= 0.5,
        )

        image_probability = image_probability.detach().float().cpu().reshape(-1)
        image_label = image_label.detach().float().cpu().reshape(-1)
        self.image.update(
            image_probability >= self.image_threshold,
            image_label >= 0.5,
            torch.ones_like(image_label, dtype=torch.bool),
        )
        self.image_scores.extend(float(value) for value in image_probability)
        self.image_labels.extend(int(value >= 0.5) for value in image_label)

        for index in range(mask_probability.shape[0]):
            if float(valid[index].sum()) <= 0.0:
                continue
            predicted = extract_instances(
                mask_probability[index : index + 1] * valid[index : index + 1],
                threshold=self.mask_threshold,
                minimum_area=self.minimum_instance_area,
            )[0]
            expected = extract_instances(
                tamper_mask[index : index + 1] * valid[index : index + 1],
                threshold=0.5,
                minimum_area=self.minimum_instance_area,
            )[0]
            candidates = sorted(
                (
                    (_box_iou(prediction, truth), pred_index, truth_index)
                    for pred_index, prediction in enumerate(predicted)
                    for truth_index, truth in enumerate(expected)
                ),
                reverse=True,
            )
            used_predictions: set[int] = set()
            used_truth: set[int] = set()
            for iou, pred_index, truth_index in candidates:
                if iou < self.instance_iou_threshold:
                    break
                if pred_index in used_predictions or truth_index in used_truth:
                    continue
                used_predictions.add(pred_index)
                used_truth.add(truth_index)
            self.instance_tp += len(used_predictions)
            self.instance_fp += len(predicted) - len(used_predictions)
            self.instance_fn += len(expected) - len(used_truth)

    def compute(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        result.update(self.pixel.metrics("pixel"))
        result.update(self.image.metrics("image"))
        result["image/auroc"] = binary_auroc(self.image_scores, self.image_labels)
        precision = _safe_divide(self.instance_tp, self.instance_tp + self.instance_fp)
        recall = _safe_divide(self.instance_tp, self.instance_tp + self.instance_fn)
        result.update(
            {
                "instance/precision": precision,
                "instance/recall": recall,
                "instance/f1": _safe_divide(
                    2.0 * precision * recall, precision + recall
                ),
            }
        )
        return result
