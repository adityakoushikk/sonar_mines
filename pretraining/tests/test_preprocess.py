"""Tests for the BenthiCat -> WebDataset shard preprocessing step.

These exercise the lossless uint8/PNG conversion contract and the shard
manifest. They depend only on numpy + PIL + webdataset (no torch), so they run
even in a minimal environment.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("webdataset")

from pretraining.preprocess import _tile_to_png_bytes, preprocess


def _write_tile(path: Path, arr: np.ndarray) -> None:
    """Persist a float32 tile to ``path`` (mimics one BenthiCat .npy)."""
    np.save(path, arr.astype(np.float32))


def test_tile_to_png_is_lossless_uint8() -> None:
    """(arr*255).clip.astype(uint8) must survive the PNG round-trip exactly."""
    rng = np.random.default_rng(0)
    arr = rng.random((384, 384), dtype=np.float32)  # ~[0, 1)

    png = _tile_to_png_bytes(arr)
    decoded = np.asarray(Image.open(io.BytesIO(png)))

    expected = (arr * 255.0).clip(0, 255).astype(np.uint8)
    assert decoded.shape == (384, 384)
    assert decoded.dtype == np.uint8
    # PNG is lossless => exact match, not just close.
    np.testing.assert_array_equal(decoded, expected)


def test_tile_to_png_squeezes_trailing_singleton() -> None:
    """HxWx1 input collapses to a 2-D mode-L PNG."""
    arr = np.zeros((8, 8, 1), dtype=np.float32)
    decoded = np.asarray(Image.open(io.BytesIO(_tile_to_png_bytes(arr))))
    assert decoded.shape == (8, 8)


def test_tile_to_png_clips_out_of_range() -> None:
    """Values >1 clip to 255 rather than overflowing uint8."""
    arr = np.full((4, 4), 1.5, dtype=np.float32)
    decoded = np.asarray(Image.open(io.BytesIO(_tile_to_png_bytes(arr))))
    assert (decoded == 255).all()


def test_preprocess_manifest_and_keys(tmp_path: Path) -> None:
    """preprocess() walks <sector>/*.npy, writes shards, preserves full stems."""
    in_dir = tmp_path / "benthicat"
    (in_dir / "sectorA").mkdir(parents=True)
    (in_dir / "sectorB").mkdir(parents=True)

    rng = np.random.default_rng(1)
    stems = [
        "sectorA/N02_4_220807052300_xtf-CH12_batch0_ch0_r0_c0",
        "sectorA/N02_4_220807052300_xtf-CH12_batch0_ch0_r0_c1",
        "sectorB/N03_1_220101000000_xtf-CH11_batch2_ch1_r3_c4",
    ]
    expected_keys = set()
    for stem in stems:
        _write_tile(in_dir / f"{stem}.npy", rng.random((16, 16)))
        expected_keys.add(Path(stem).stem)  # full filename stem, no parent

    out_dir = tmp_path / "shards"
    manifest = preprocess(in_dir, out_dir, maxcount=2)  # force >1 shard

    assert manifest["n_tiles"] == 3
    assert manifest["n_shards"] >= 2  # maxcount=2 over 3 tiles -> 2 shards
    assert manifest["total_bytes"] > 0

    # Pull every __key__ back out of the written tars and compare to expected.
    found_keys: set[str] = set()
    for shard in sorted(out_dir.glob("benthicat-*.tar")):
        with tarfile.open(shard) as tar:
            for member in tar.getnames():
                # members look like "<key>.png"; strip the extension.
                if member.endswith(".png"):
                    found_keys.add(member[: -len(".png")])
    assert found_keys == expected_keys


def test_preprocess_limit(tmp_path: Path) -> None:
    """--limit N stops after N tiles."""
    in_dir = tmp_path / "b"
    in_dir.mkdir()
    for i in range(5):
        _write_tile(in_dir / f"tile_{i}.npy", np.zeros((4, 4), dtype=np.float32))

    manifest = preprocess(in_dir, tmp_path / "out", limit=2)
    assert manifest["n_tiles"] == 2
