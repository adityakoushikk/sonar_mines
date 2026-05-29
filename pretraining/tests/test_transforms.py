"""Tests for the BYOL two-view transform.

Requires albumentations + torch. Verifies shapes, channel replication, and that
the two views are produced by *independent* random draws.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("albumentations")
pytest.importorskip("torch")

import torch

from pretraining.ssl.transforms import SonarBYOLTransform


@pytest.fixture(scope="module")
def transform() -> SonarBYOLTransform:
    return SonarBYOLTransform()


@pytest.mark.parametrize(
    "shape",
    [(384, 384), (384, 384, 1), (384, 384, 3)],
    ids=["HxW", "HxWx1", "HxWx3"],
)
def test_output_shape_and_dtype(transform: SonarBYOLTransform, shape) -> None:
    """Any accepted input shape yields two (3, 224, 224) float tensors."""
    img = (np.random.default_rng(0).random(shape) * 255).astype(np.uint8)
    v0, v1 = transform(img)
    for v in (v0, v1):
        assert isinstance(v, torch.Tensor)
        assert v.shape == (3, 224, 224)
        assert v.dtype == torch.float32


def test_views_are_independent(transform: SonarBYOLTransform) -> None:
    """Two views of one tile must differ (independent crops/flips/noise)."""
    img = (np.random.default_rng(1).random((256, 256)) * 255).astype(np.uint8)
    v0, v1 = transform(img)
    # Astronomically unlikely to be identical given RandomResizedCrop alone.
    assert not torch.equal(v0, v1)


def test_rejects_bad_channel_count(transform: SonarBYOLTransform) -> None:
    """A 2-channel image is neither grayscale nor RGB and must raise."""
    img = np.zeros((16, 16, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        transform(img)
