"""Shared deterministic data, checkpoint, and evaluation runtime."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.manifest import (
    ForgeryManifestDataset,
    ManifestBatch,
    collate_manifest_samples,
)
from src.evaluation.metrics import StreamingForgeryMetrics
from src.model import DeepDocForgeryModel
from src.training.objectives import DeepDocForgeryCriterion


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
    dataset = ForgeryManifestDataset(
        manifest,
        split=split,
        image_size=(int(size[0]), int(size[1])),
        noise_types=tuple(
            degradation_config.get(
                "noise_types", ["none", "gaussian", "poisson", "speckle"]
            )
        ),
    )
    seed = int(training_config.get("seed", 7))
    generator = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[split])
    return DataLoader(
        dataset,
        batch_size=int(
            data_config.get(
                f"{split}_batch_size", data_config.get("batch_size", 4)
            )
        ),
        shuffle=shuffle,
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
) -> dict[str, float | None]:
    model.eval()
    metric_config = metric_config or {}
    metrics = StreamingForgeryMetrics(
        mask_threshold=float(metric_config.get("mask_threshold", 0.5)),
        image_threshold=float(metric_config.get("image_threshold", 0.5)),
        minimum_instance_area=int(metric_config.get("minimum_instance_area", 16)),
        instance_iou_threshold=float(metric_config.get("instance_iou_threshold", 0.5)),
    )
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
            output = model(batch.rgb, metadata=batch.metadata)
            if criterion is not None:
                batch_losses = criterion(output, batch.supervision)
                losses.update(batch_losses, batch.rgb.shape[0])
        metrics.update(
            mask_probability=output.decoder.mask_probability,
            tamper_mask=batch.supervision.tamper_mask,
            image_probability=output.decoder.image_probability,
            image_label=batch.supervision.image_label,
            valid_mask=batch.supervision.valid_mask,
        )
    result = metrics.compute()
    if criterion is not None:
        result.update(losses.compute())
    return result
