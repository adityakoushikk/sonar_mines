"""BYOL two-view augmentation pipeline for side-scan-sonar tiles.

BYOL learns representations invariant to the augmentations we apply, so the
augmentation set *defines* the invariances. We deliberately include SSS-specific
nuisances — speckle and range falloff — so the backbone learns to ignore them,
but we EXCLUDE acoustic shadow: shadow is the downstream detection *signal*, and
making the backbone shadow-invariant would hurt transfer to mine detection.
"""
from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.augmentations.range_falloff import RangeFalloff
from src.augmentations.speckle import SpeckleNoise


def _random_resized_crop() -> A.RandomResizedCrop:
    """Build ``RandomResizedCrop(224, 224, scale=(0.4, 1.0))`` version-robustly.

    Albumentations changed this constructor around 1.4.15: older releases accept
    positional ``(height, width)``; newer ones require a single ``size=(h, w)``
    tuple and reject the positional ints. ``requirements.txt`` pins only
    ``albumentations>=1.4`` (no upper bound), so a fresh Modal install may land on
    either side of that break — we try the new signature first, then fall back.
    """
    scale = (0.4, 1.0)
    try:
        return A.RandomResizedCrop(size=(224, 224), scale=scale, p=1.0)
    except TypeError:
        return A.RandomResizedCrop(224, 224, scale=scale, p=1.0)


class SonarBYOLTransform:
    """Produce two independently-augmented (3,224,224) views of one tile.

    The same ``albumentations.Compose`` is applied twice with independent random
    draws, yielding the two correlated views BYOL contrasts. Speckle and range
    falloff are SSS-aware; acoustic shadow is intentionally omitted (see module
    docstring).
    """

    def __init__(self) -> None:
        # One pipeline, called twice — each call samples its own random params.
        self.transform = A.Compose(
            [
                _random_resized_crop(),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                SpeckleNoise(p=0.5),
                # nadir along x: range increases horizontally for SSS tiles.
                RangeFalloff(nadir_axis="x", falloff_range=(0.3, 1.0), p=0.5),
                A.Normalize(mean=(0.55, 0.55, 0.55), std=(0.20, 0.20, 0.20)),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def _to_3ch(image: np.ndarray) -> np.ndarray:
        """Coerce HxW / HxWx1 / HxWx3 uint8 input to a 3-channel HxWx3 array.

        SSS tiles are single-channel; we replicate to 3 channels because the
        YOLOv8 stem expects RGB-shaped input.
        """
        if image.ndim == 2:  # HxW
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"expected HxW or HxWxC image, got shape {image.shape}")
        c = image.shape[2]
        if c == 1:
            image = np.repeat(image, 3, axis=2)
        elif c != 3:
            raise ValueError(f"expected 1 or 3 channels, got {c}")
        return image

    def __call__(self, image: np.ndarray) -> tuple[Tensor, Tensor]:
        """Return two independently-augmented float views of ``image``."""
        image = self._to_3ch(image)
        v0 = self.transform(image=image)["image"]
        v1 = self.transform(image=image)["image"]
        return v0, v1
