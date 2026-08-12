from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.inputLayer.dataModelling import (
    estimate_quality_from_qtables,
    jpeg_quantization_tables,
    make_synthetic_batch,
    read_jpeg_metadata,
    shuffle_artifact_patches,
)
from src.inputLayer.freqFeatures import ExactJPEGDCTReader


def test_quantization_tables_track_quality_and_round_trip_estimate() -> None:
    quality = torch.tensor([25.0, 75.0, 95.0])
    tables = jpeg_quantization_tables(quality)
    assert float(tables[0].mean()) > float(tables[1].mean()) > float(tables[2].mean())
    torch.testing.assert_close(estimate_quality_from_qtables(tables), quality)


def test_pillow_metadata_is_natural_order_not_double_zigzagged(tmp_path) -> None:
    path = tmp_path / "quality50.jpg"
    pixels = np.full((32, 32, 3), 127, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=50, subsampling=0)
    metadata = read_jpeg_metadata(path)
    metadata.validate()
    assert metadata.is_jpeg.item() == 1.0
    assert metadata.qtable_valid.sum().item() == 3.0
    assert metadata.subsampling[0, 0].item() == 1.0
    assert metadata.qtables[0, 0, 0, 0].item() == 16.0
    assert metadata.qtables[0, 0, 0, 1].item() == 11.0


def test_non_jpeg_metadata_uses_neutral_fallback(tmp_path) -> None:
    path = tmp_path / "document.png"
    Image.new("RGB", (16, 16), "white").save(path)
    metadata = read_jpeg_metadata(path)
    assert metadata.is_jpeg.item() == 0.0
    assert metadata.qtable_valid.sum().item() == 0.0
    assert bool((metadata.qtables == 1.0).all())


def test_synthetic_batch_contract() -> None:
    batch = make_synthetic_batch(batch_size=2, height=64, width=80, seed=23)
    assert batch.rgb.shape == (2, 3, 64, 80)
    assert batch.targets.jpeg_quality.shape == (2, 1, 64, 80)
    assert batch.targets.double_compression.shape == (2, 1, 64, 80)
    assert batch.targets.noise_type.shape == (2, 64, 80)
    assert batch.targets.noise_strength.shape == (2, 1, 64, 80)
    assert batch.tamper_mask.shape == (2, 1, 64, 80)
    assert set(batch.tamper_mask.unique().tolist()).issubset({0.0, 1.0})
    assert batch.metadata.qtables.shape == (2, 3, 8, 8)
    assert bool((batch.rgb >= 0.0).all() and (batch.rgb <= 1.0).all())


def test_exact_reader_aligns_subsampled_chroma_to_luma_grid(monkeypatch) -> None:
    fake_jpeg = SimpleNamespace(
        coef_arrays=[
            np.arange(32 * 48).reshape(32, 48),
            np.arange(16 * 24).reshape(16, 24),
            np.arange(16 * 24).reshape(16, 24),
        ],
        quant_tables=[
            np.full((8, 8), 10),
            np.full((8, 8), 20),
        ],
        comp_info=[
            SimpleNamespace(h_samp_factor=2, v_samp_factor=2, quant_tbl_no=0),
            SimpleNamespace(h_samp_factor=1, v_samp_factor=1, quant_tbl_no=1),
            SimpleNamespace(h_samp_factor=1, v_samp_factor=1, quant_tbl_no=1),
        ],
    )
    reader = ExactJPEGDCTReader()
    monkeypatch.setattr(
        reader,
        "_import_jpegio",
        lambda: SimpleNamespace(read=lambda _: fake_jpeg),
    )
    result = reader.read("synthetic.jpg")
    assert result.coefficients.shape == (1, 3, 64, 4, 6)
    assert result.metadata.subsampling.tolist() == [[0.0, 0.0, 1.0, 0.0]]
    assert result.metadata.qtables[0, 0].unique().item() == 10.0
    assert result.metadata.qtables[0, 1].unique().item() == 20.0


def test_internal_patch_shuffle_keeps_image_and_mask_permutations_aligned() -> None:
    patch_values = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    rgb = patch_values.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3).repeat(1, 3, 1, 1)
    mask = patch_values.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
    shuffled = shuffle_artifact_patches(
        rgb,
        mask,
        patch_size=8,
        mode="internal",
        generator=torch.Generator().manual_seed(5),
    )
    torch.testing.assert_close(shuffled.rgb[:, 0:1], shuffled.artifact_mask)
    for sample in range(2):
        assert sorted(shuffled.rgb[sample, 0].unique().tolist()) == sorted(
            rgb[sample, 0].unique().tolist()
        )


def test_external_patch_shuffle_can_exchange_patches_across_images() -> None:
    rgb = torch.zeros(2, 3, 16, 16)
    rgb[1] = 1.0
    mask = rgb[:, 0:1].clone()
    shuffled = shuffle_artifact_patches(
        rgb,
        mask,
        patch_size=8,
        mode="external",
        generator=torch.Generator().manual_seed(3),
    )
    torch.testing.assert_close(shuffled.rgb[:, 0:1], shuffled.artifact_mask)
    assert set(shuffled.rgb.unique().tolist()) == {0.0, 1.0}
    assert any(len(shuffled.rgb[index].unique()) > 1 for index in range(2))
