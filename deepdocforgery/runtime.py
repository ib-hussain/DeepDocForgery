"""Shared deterministic data, checkpoint, and evaluation runtime."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from deepdocforgery.data import (
    ForgeryManifestDataset,
    ManifestBatch,
    collate_manifest_samples,
)
from deepdocforgery.metrics import StreamingForgeryMetrics
from deepdocforgery.model import DeepDocForgeryModel
from deepdocforgery.objectives import DeepDocForgeryCriterion


class CosineEpochScheduler:
    """Absolute-epoch cosine schedule that remains monotonic after resume."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        minimum_learning_rate: float,
    ) -> None:
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = 0

    def step(self, completed_epochs: int) -> None:
        self.last_epoch = int(completed_epochs)
        progress = min(1.0, max(0.0, self.last_epoch / self.total_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = self.minimum_learning_rate + (base - self.minimum_learning_rate) * cosine

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_epoch": self.last_epoch,
            "total_epochs": self.total_epochs,
            "base_lrs": self.base_lrs,
            "minimum_learning_rate": self.minimum_learning_rate,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_total = int(state.get("total_epochs", self.total_epochs))
        if saved_total != self.total_epochs:
            raise ValueError(
                "Cannot resume with a changed total epoch count; start a documented fresh "
                "fine-tuning phase to avoid a learning-rate rebound"
            )
        self.last_epoch = int(state.get("last_epoch", 0))
        saved_base = state.get("base_lrs")
        if isinstance(saved_base, list) and len(saved_base) == len(self.base_lrs):
            self.base_lrs = [float(value) for value in saved_base]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    config: Mapping[str, Any],
    *,
    split: str,
    shuffle: bool,
) -> DataLoader[ManifestBatch]:
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    degradation_config = model_config.get("degradation_estimator", {})
    manifest = data_config.get("manifest")
    if manifest is None:
        raise ValueError("Configuration data.manifest is required")
    size = data_config.get("image_size", [512, 512])
    augmentation = data_config.get("augmentation", {}) if split == "train" else {}
    dataset = ForgeryManifestDataset(
        manifest,
        split=split,
        image_size=(int(size[0]), int(size[1])),
        noise_types=tuple(
            degradation_config.get("noise_types", ["none", "gaussian", "poisson", "speckle"])
        ),
        augment=split == "train" and bool(augmentation.get("enabled", False)),
        horizontal_flip_probability=float(augmentation.get("horizontal_flip_probability", 0.0)),
        jpeg_probability=float(augmentation.get("jpeg_probability", 0.0)),
        jpeg_quality_range=tuple(
            int(value) for value in augmentation.get("jpeg_quality_range", (55, 95))
        ),
        gaussian_noise_probability=float(augmentation.get("gaussian_noise_probability", 0.0)),
        gaussian_noise_maximum=float(augmentation.get("gaussian_noise_maximum", 0.03)),
        crop_probability=float(augmentation.get("crop_probability", 0.0)),
        crop_scale_range=tuple(
            float(value) for value in augmentation.get("crop_scale_range", (0.6, 1.0))
        ),
        exact_dct_probability=float(data_config.get("exact_dct_probability", 0.0)),
    )
    seed = int(training_config.get("seed", 7))
    generator = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[split])
    sampler = None
    sampling = data_config.get("sampling", {})
    dataset_weights = sampling.get("dataset_weights", {}) if split == "train" else {}
    if dataset_weights:
        counts: dict[str, int] = defaultdict(int)
        for record in dataset.records:
            counts[record.dataset] += 1
        unknown = set(dataset_weights).difference(counts)
        if unknown:
            raise ValueError(f"Sampling weights reference absent datasets: {sorted(unknown)}")
        weights = [
            float(dataset_weights.get(record.dataset, 1.0)) / counts[record.dataset]
            for record in dataset.records
        ]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(sampling.get("samples_per_epoch", len(dataset))),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=int(data_config.get(f"{split}_batch_size", data_config.get("batch_size", 4))),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=(
            bool(data_config.get("persistent_workers", True))
            and int(data_config.get("num_workers", 0)) > 0
        ),
        collate_fn=collate_manifest_samples,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=bool(data_config.get("drop_last", False)) if split == "train" else False,
    )


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must be a dictionary containing a model state")
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


class _LossAverages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.examples = 0

    def update(self, losses: Mapping[str, torch.Tensor], batch_size: int) -> None:
        self.examples += batch_size
        for name, value in losses.items():
            self.sums[name] += float(value.detach().cpu()) * batch_size

    def compute(self, prefix: str = "loss") -> dict[str, float]:
        denominator = max(1, self.examples)
        return {f"{prefix}/{name}": value / denominator for name, value in self.sums.items()}


@torch.no_grad()
def evaluate_model(
    model: DeepDocForgeryModel,
    loader: DataLoader[ManifestBatch],
    *,
    device: torch.device,
    criterion: DeepDocForgeryCriterion | None = None,
    use_amp: bool = False,
    maximum_batches: int | None = None,
    metric_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    metric_config = metric_config or {}
    metrics = StreamingForgeryMetrics(
        mask_threshold=float(metric_config.get("mask_threshold", 0.5)),
        image_threshold=float(metric_config.get("image_threshold", 0.5)),
        minimum_instance_area=int(metric_config.get("minimum_instance_area", 16)),
        instance_iou_threshold=float(metric_config.get("instance_iou_threshold", 0.5)),
        catastrophic_f1_threshold=float(metric_config.get("catastrophic_f1_threshold", 0.1)),
    )
    benchmark_metrics: dict[str, StreamingForgeryMetrics] = {}
    exact_samples = 0
    total_samples = 0
    losses = _LossAverages()
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            output = model(
                batch.rgb,
                metadata=batch.metadata,
                exact_dct=batch.exact_dct,
                classification_valid=batch.supervision.image_valid,
            )
            if criterion is not None:
                batch_losses = criterion(output, batch.supervision)
                losses.update(batch_losses, batch.rgb.shape[0])
        metrics.update(
            mask_probability=output.decoder.mask_probability,
            tamper_mask=batch.supervision.tamper_mask,
            image_probability=output.decoder.image_probability,
            image_label=batch.supervision.image_label,
            valid_mask=batch.supervision.valid_mask,
            image_valid=batch.supervision.image_valid,
        )
        total_samples += batch.rgb.shape[0]
        if batch.exact_dct is not None and batch.exact_dct.valid is not None:
            exact_samples += int(batch.exact_dct.valid.sum().item())
        benchmark_names = sorted({record.benchmark for record in batch.records})
        for benchmark in benchmark_names:
            indices = [
                index for index, record in enumerate(batch.records) if record.benchmark == benchmark
            ]
            current = benchmark_metrics.setdefault(
                benchmark,
                StreamingForgeryMetrics(
                    mask_threshold=float(metric_config.get("mask_threshold", 0.5)),
                    image_threshold=float(metric_config.get("image_threshold", 0.5)),
                    minimum_instance_area=int(metric_config.get("minimum_instance_area", 16)),
                    instance_iou_threshold=float(metric_config.get("instance_iou_threshold", 0.5)),
                    catastrophic_f1_threshold=float(
                        metric_config.get("catastrophic_f1_threshold", 0.1)
                    ),
                ),
            )
            current.update(
                mask_probability=output.decoder.mask_probability[indices],
                tamper_mask=batch.supervision.tamper_mask[indices],
                image_probability=output.decoder.image_probability[indices],
                image_label=batch.supervision.image_label[indices],
                valid_mask=(
                    None
                    if batch.supervision.valid_mask is None
                    else batch.supervision.valid_mask[indices]
                ),
                image_valid=(
                    None
                    if batch.supervision.image_valid is None
                    else batch.supervision.image_valid[indices]
                ),
            )
    result = metrics.compute()
    result["benchmarks"] = {
        name: value.compute() for name, value in sorted(benchmark_metrics.items())
    }
    result["evidence/exact_dct_samples"] = float(exact_samples)
    result["evidence/total_samples"] = float(total_samples)
    result["evidence/exact_dct_fraction"] = exact_samples / total_samples if total_samples else 0.0
    if criterion is not None:
        result.update(losses.compute())
    return result
