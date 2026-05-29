"""Smoke test for the WebDataset BYOL loader.

Builds a tiny real shard on disk with ``preprocess`` and iterates one batch with
``num_workers=0`` (the contract the smoke test must satisfy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("webdataset")
pytest.importorskip("albumentations")
pytest.importorskip("torch")

import torch

from pretraining.preprocess import preprocess
from pretraining.ssl.data import build_webdataset_loader
from pretraining.ssl.transforms import SonarBYOLTransform


def _make_shard(tmp_path: Path, n: int = 8) -> str:
    """Write ``n`` random tiles into a single shard; return a brace-glob path."""
    in_dir = tmp_path / "tiles"
    in_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        np.save(in_dir / f"tile_{i:03d}.npy", rng.random((32, 32)).astype(np.float32))

    out_dir = tmp_path / "shards"
    preprocess(in_dir, out_dir, maxcount=1000)
    # Single shard => index 000000.
    return str(out_dir / "benthicat-000000.tar")


def test_loader_yields_two_view_batch(tmp_path: Path) -> None:
    """One batch is a 2-tuple of (B, 3, 224, 224) float tensors."""
    shards = _make_shard(tmp_path, n=8)
    loader = build_webdataset_loader(
        shards,
        transform=SonarBYOLTransform(),
        batch_size=4,
        num_workers=0,  # must work per the contract
        shardshuffle=False,
    )

    batch = next(iter(loader))
    assert isinstance(batch, tuple) and len(batch) == 2
    v0, v1 = batch
    assert v0.shape == (4, 3, 224, 224)
    assert v1.shape == (4, 3, 224, 224)
    assert v0.dtype == torch.float32
    # The two views in a batch come from independent augmentation draws.
    assert not torch.equal(v0, v1)


def test_with_epoch_limits_samples(tmp_path: Path) -> None:
    """epoch_length caps the number of samples seen per epoch."""
    shards = _make_shard(tmp_path, n=8)
    loader = build_webdataset_loader(
        shards,
        transform=SonarBYOLTransform(),
        batch_size=2,
        num_workers=0,
        epoch_length=6,
        shardshuffle=False,
    )
    n_seen = sum(v0.shape[0] for v0, _ in loader)
    assert n_seen == 6
