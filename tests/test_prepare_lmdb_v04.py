from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from deepdocforgery.prepare import _doctamper_lmdb_records

lmdb = pytest.importorskip("lmdb")


def _bytes(image: Image.Image, format_name: str, **kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize("offset", [0, 1])
def test_lmdb_export_is_index_safe_restartable_and_preserves_jpeg(
    tmp_path: Path, offset: int
) -> None:
    source = tmp_path / "lmdb"
    source.mkdir()
    image_bytes = _bytes(Image.new("RGB", (32, 32), "white"), "JPEG", quality=81)
    mask = Image.new("L", (32, 32), 0)
    mask.putpixel((10, 10), 255)
    mask_bytes = _bytes(mask, "PNG")
    environment = lmdb.open(str(source), map_size=8 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        transaction.put(b"num-samples", b"1")
        transaction.put(f"image-{offset:09d}".encode(), image_bytes)
        transaction.put(f"label-{offset:09d}".encode(), mask_bytes)
    environment.close()

    output = tmp_path / "prepared"
    first = _doctamper_lmdb_records(
        source,
        output,
        benchmark="doctamper-training",
        limit=None,
    )
    second = _doctamper_lmdb_records(
        source,
        output,
        benchmark="doctamper-training",
        limit=None,
    )
    assert first == second
    assert Path(first[0].image).read_bytes() == image_bytes
    assert first[0].label == 1
    assert first[0].classification_supervised is False
