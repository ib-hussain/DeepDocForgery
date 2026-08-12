"""Manifest-backed data loading for real and generated document datasets."""

from .manifest import (
    ForgeryManifestDataset,
    ManifestBatch,
    ManifestRecord,
    collate_manifest_samples,
    load_manifest,
    summarize_manifest,
    validate_group_disjointness,
)

__all__ = [
    "ForgeryManifestDataset",
    "ManifestBatch",
    "ManifestRecord",
    "collate_manifest_samples",
    "load_manifest",
    "summarize_manifest",
    "validate_group_disjointness",
]
