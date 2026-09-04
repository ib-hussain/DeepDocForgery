from __future__ import annotations

from pathlib import Path

import pytest

from deepdocforgery.prepare import assigned_split, infer_subset


@pytest.mark.parametrize(
    ("directory", "expected"),
    (
        ("DocTamperV1-TrainingSet", "training"),
        ("DocTamperV1-TestingSet", "testing"),
        ("DocTamperV1-FCD", "fcd"),
        ("DocTamperV1-SCD", "scd"),
    ),
)
def test_infer_official_doctamper_subset(directory: str, expected: str) -> None:
    assert infer_subset(Path(directory), "auto") == expected


def test_official_test_subsets_are_never_randomly_split() -> None:
    for subset in ("testing", "fcd", "scd"):
        for index in range(20):
            assert assigned_split(f"sample-{index}", subset, 7, 0.05) == "test"


def test_training_subset_has_no_test_assignment() -> None:
    assignments = {assigned_split(f"sample-{index}", "training", 7, 0.25) for index in range(100)}
    assert assignments == {"train", "val"}
