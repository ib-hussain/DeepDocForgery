from __future__ import annotations

import pytest
import torch

from src.inputLayer.freqFeatures import (
    DCTFeaturePyramid,
    ExactDCTBatch,
    JPEGMetadata,
    PixelDomainDCT,
)


def test_dct_matrix_is_orthonormal() -> None:
    matrix = PixelDomainDCT().dct_matrix
    identity = matrix @ matrix.transpose(0, 1)
    torch.testing.assert_close(identity, torch.eye(8), atol=1e-6, rtol=1e-6)


def test_rgb_to_ycbcr_known_black_and_white() -> None:
    rgb = torch.tensor(
        [
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[1.0]], [[1.0]], [[1.0]]],
        ]
    )
    converted = PixelDomainDCT.rgb_to_ycbcr(rgb)
    torch.testing.assert_close(converted[0, :, 0, 0], torch.tensor([0.0, 0.5, 0.5]))
    torch.testing.assert_close(converted[1, :, 0, 0], torch.tensor([1.0, 0.5, 0.5]))


def test_fallback_pyramid_shapes_for_non_multiple_image_size() -> None:
    model = DCTFeaturePyramid(
        stem_channels=8,
        out_channels=(12, 20, 28),
        level_names=("s8", "s16", "s32"),
    )
    rgb = torch.rand(2, 3, 65, 81)
    output = model(rgb)
    assert output.source == "pixel_fallback"
    assert output.consistency_loss.shape == torch.Size([])
    assert output.features["s8"].shape == (2, 12, 9, 11)
    assert output.features["s16"].shape == (2, 20, 5, 6)
    assert output.features["s32"].shape == (2, 28, 3, 3)


def test_exact_and_identical_fallback_have_zero_consistency_loss() -> None:
    model = DCTFeaturePyramid(
        stem_channels=8,
        out_channels=(16, 24),
        level_names=("s8", "s16"),
        consistency_weight=0.5,
    ).train()
    rgb = torch.rand(1, 3, 32, 40)
    metadata = JPEGMetadata(
        qtables=torch.full((1, 3, 8, 8), 10.0),
        qtable_valid=torch.ones(1, 3),
        is_jpeg=torch.ones(1, 1),
        subsampling=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    coefficients = model.pixel_dct(rgb, metadata.qtables, metadata.subsampling).detach()
    exact = ExactDCTBatch(coefficients=coefficients, metadata=metadata)
    output = model(
        rgb,
        metadata=metadata,
        exact_dct=exact,
        compute_consistency=True,
    )
    assert output.source == "exact_jpeg"
    torch.testing.assert_close(output.consistency_loss, torch.tensor(0.0), atol=1e-8, rtol=0)


def test_consistency_rejects_misaligned_exact_grid() -> None:
    model = DCTFeaturePyramid(
        stem_channels=8,
        out_channels=(16,),
        level_names=("s8",),
    ).train()
    rgb = torch.rand(1, 3, 32, 32)
    metadata = JPEGMetadata.for_non_jpeg(1)
    exact = ExactDCTBatch(
        coefficients=torch.zeros(1, 3, 64, 3, 4),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="8-pixel-aligned crop"):
        model(
            rgb,
            metadata=metadata,
            exact_dct=exact,
            compute_consistency=True,
        )


def test_pixel_dct_models_declared_chroma_subsampling() -> None:
    transform = PixelDomainDCT()
    rgb = torch.rand(1, 3, 32, 48)
    tables = torch.ones(1, 3, 8, 8)
    coefficients_444 = transform(rgb, tables, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    coefficients_420 = transform(rgb, tables, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    assert coefficients_444.shape == coefficients_420.shape == (1, 3, 64, 4, 6)
    assert not torch.allclose(coefficients_444[:, 1:], coefficients_420[:, 1:])
