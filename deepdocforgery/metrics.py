"""Pixel micro/macro, failure-tail, image, and instance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from deepdocforgery.postprocess import InstancePrediction, extract_instances


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def binary_auroc(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positives = sum(value == 1 for value in labels)
    negatives = sum(value == 0 for value in labels)
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = false_positive = 0
    previous_tpr = previous_fpr = area = 0.0
    index = 0
    while index < len(order):
        threshold = scores[order[index]]
        while index < len(order) and scores[order[index]] == threshold:
            label = labels[order[index]]
            true_positive += int(label == 1)
            false_positive += int(label == 0)
            index += 1
        tpr = true_positive / positives
        fpr = false_positive / negatives
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
        prediction, target, valid = prediction.bool(), target.bool(), valid.bool()
        self.true_positive += float((prediction & target & valid).sum())
        self.false_positive += float((prediction & ~target & valid).sum())
        self.true_negative += float((~prediction & ~target & valid).sum())
        self.false_negative += float((~prediction & target & valid).sum())

    def metrics(self, prefix: str, *, null_if_empty: bool = False) -> dict[str, float | None]:
        tp, fp, tn, fn = (
            self.true_positive,
            self.false_positive,
            self.true_negative,
            self.false_negative,
        )
        if null_if_empty and tp + fp + tn + fn == 0:
            return {
                f"{prefix}/precision": None,
                f"{prefix}/recall": None,
                f"{prefix}/f1": None,
                f"{prefix}/iou": None,
                f"{prefix}/accuracy": None,
                f"{prefix}/specificity": None,
                f"{prefix}/false_positive_rate": None,
                f"{prefix}/false_negative_rate": None,
            }
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        return {
            f"{prefix}/precision": precision,
            f"{prefix}/recall": recall,
            f"{prefix}/f1": _safe_divide(2 * precision * recall, precision + recall),
            f"{prefix}/iou": _safe_divide(tp, tp + fp + fn),
            f"{prefix}/accuracy": _safe_divide(tp + tn, tp + fp + tn + fn),
            f"{prefix}/specificity": specificity,
            f"{prefix}/false_positive_rate": _safe_divide(fp, fp + tn),
            f"{prefix}/false_negative_rate": _safe_divide(fn, fn + tp),
        }


def _box_iou(first: InstancePrediction, second: InstancePrediction) -> float:
    ax1, ay1, ax2, ay2 = first.box_xyxy
    bx1, by1, bx2, by2 = second.box_xyxy
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return _safe_divide(float(intersection), float(first_area + second_area - intersection))


class StreamingForgeryMetrics:
    """Keep aggregate performance and the catastrophic failure tail visible."""

    def __init__(
        self,
        *,
        mask_threshold: float = 0.5,
        image_threshold: float = 0.5,
        minimum_instance_area: int = 16,
        instance_iou_threshold: float = 0.5,
        catastrophic_f1_threshold: float = 0.1,
    ) -> None:
        self.mask_threshold = float(mask_threshold)
        self.image_threshold = float(image_threshold)
        self.minimum_instance_area = int(minimum_instance_area)
        self.instance_iou_threshold = float(instance_iou_threshold)
        self.catastrophic_f1_threshold = float(catastrophic_f1_threshold)
        self.pixel = _Confusion()
        self.image = _Confusion()
        self.image_scores: list[float] = []
        self.image_labels: list[int] = []
        self.per_image_f1: list[float] = []
        self.forged_image_f1: list[float] = []
        self.area_f1: dict[str, list[float]] = {
            "lt_2pct": [],
            "2_to_5pct": [],
            "5_to_10pct": [],
            "ge_10pct": [],
        }
        self.instance_tp = self.instance_fp = self.instance_fn = 0.0

    @staticmethod
    def _image_f1(prediction: Tensor, target: Tensor, valid: Tensor) -> float:
        prediction, target, valid = prediction.bool(), target.bool(), valid.bool()
        tp = float((prediction & target & valid).sum())
        fp = float((prediction & ~target & valid).sum())
        fn = float((~prediction & target & valid).sum())
        if tp + fp + fn == 0:
            return 1.0
        return _safe_divide(2 * tp, 2 * tp + fp + fn)

    @staticmethod
    def _area_bucket(fraction: float) -> str:
        if fraction < 0.02:
            return "lt_2pct"
        if fraction < 0.05:
            return "2_to_5pct"
        if fraction < 0.10:
            return "5_to_10pct"
        return "ge_10pct"

    def update(
        self,
        *,
        mask_probability: Tensor,
        tamper_mask: Tensor,
        image_probability: Tensor,
        image_label: Tensor,
        valid_mask: Tensor | None = None,
        image_valid: Tensor | None = None,
    ) -> None:
        probability = mask_probability.detach().float().cpu()
        target = tamper_mask.detach().float().cpu()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        valid = torch.ones_like(target) if valid_mask is None else valid_mask.detach().float().cpu()
        prediction = probability >= self.mask_threshold
        truth = target >= 0.5
        self.pixel.update(prediction, truth, valid >= 0.5)

        image_probability = image_probability.detach().float().cpu().reshape(-1)
        image_label = image_label.detach().float().cpu().reshape(-1)
        image_valid_flat = (
            torch.ones_like(image_label, dtype=torch.bool)
            if image_valid is None
            else image_valid.detach().float().cpu().reshape(-1) >= 0.5
        )
        self.image.update(
            image_probability >= self.image_threshold,
            image_label >= 0.5,
            image_valid_flat,
        )
        self.image_scores.extend(float(value) for value in image_probability[image_valid_flat])
        self.image_labels.extend(int(value >= 0.5) for value in image_label[image_valid_flat])

        for index in range(probability.shape[0]):
            valid_pixels = float(valid[index].sum())
            if valid_pixels <= 0:
                continue
            f1 = self._image_f1(prediction[index], truth[index], valid[index] >= 0.5)
            self.per_image_f1.append(f1)
            tampered_pixels = float((truth[index] & (valid[index] >= 0.5)).sum())
            if tampered_pixels > 0:
                self.forged_image_f1.append(f1)
                fraction = tampered_pixels / valid_pixels
                self.area_f1[self._area_bucket(fraction)].append(f1)

            predicted_instances = extract_instances(
                probability[index : index + 1] * valid[index : index + 1],
                threshold=self.mask_threshold,
                minimum_area=self.minimum_instance_area,
            )[0]
            expected_instances = extract_instances(
                target[index : index + 1] * valid[index : index + 1],
                threshold=0.5,
                minimum_area=self.minimum_instance_area,
            )[0]
            candidates = sorted(
                (
                    (_box_iou(predicted_item, expected_item), pred_index, truth_index)
                    for pred_index, predicted_item in enumerate(predicted_instances)
                    for truth_index, expected_item in enumerate(expected_instances)
                ),
                reverse=True,
            )
            used_prediction: set[int] = set()
            used_truth: set[int] = set()
            for iou, pred_index, truth_index in candidates:
                if iou < self.instance_iou_threshold:
                    break
                if pred_index in used_prediction or truth_index in used_truth:
                    continue
                used_prediction.add(pred_index)
                used_truth.add(truth_index)
            self.instance_tp += len(used_prediction)
            self.instance_fp += len(predicted_instances) - len(used_prediction)
            self.instance_fn += len(expected_instances) - len(used_truth)

    def compute(self) -> dict[str, float | None]:
        micro = self.pixel.metrics("pixel_micro")
        result: dict[str, float | None] = dict(micro)
        # Compatibility aliases remain explicit in the report.
        for metric in ("precision", "recall", "f1", "iou", "accuracy"):
            result[f"pixel/{metric}"] = micro[f"pixel_micro/{metric}"]
        total = (
            self.pixel.true_positive
            + self.pixel.false_positive
            + self.pixel.true_negative
            + self.pixel.false_negative
        )
        positives = self.pixel.true_positive + self.pixel.false_negative
        background_accuracy = _safe_divide(total - positives, total)
        result.update(
            {
                "pixel_macro/f1_all": _mean(self.per_image_f1),
                "pixel_macro/f1_forged": _mean(self.forged_image_f1),
                "pixel/prevalence": _safe_divide(positives, total),
                "pixel/all_background_accuracy": background_accuracy,
                "pixel/accuracy_gain_over_background": (
                    float(result["pixel/accuracy"]) - background_accuracy
                ),
                "failure/catastrophic_miss_rate": (
                    _safe_divide(
                        sum(
                            value < self.catastrophic_f1_threshold for value in self.forged_image_f1
                        ),
                        len(self.forged_image_f1),
                    )
                    if self.forged_image_f1
                    else None
                ),
            }
        )
        for bucket, values in self.area_f1.items():
            result[f"area/{bucket}/f1"] = _mean(values)
            result[f"area/{bucket}/images"] = float(len(values))
        result.update(self.image.metrics("image", null_if_empty=True))
        result["image/auroc"] = binary_auroc(self.image_scores, self.image_labels)
        result["image/supervised_samples"] = float(len(self.image_labels))
        result["image/supervised_authentic"] = float(sum(label == 0 for label in self.image_labels))
        result["image/supervised_forged"] = float(sum(label == 1 for label in self.image_labels))
        precision = _safe_divide(self.instance_tp, self.instance_tp + self.instance_fp)
        recall = _safe_divide(self.instance_tp, self.instance_tp + self.instance_fn)
        result.update(
            {
                "instance/precision": precision,
                "instance/recall": recall,
                "instance/f1": _safe_divide(2 * precision * recall, precision + recall),
            }
        )
        return result
