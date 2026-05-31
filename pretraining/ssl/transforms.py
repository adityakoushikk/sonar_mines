"""BYOL two-view augmentation for SSS tiles.

The augmentation set defines BYOL's learned invariances, so we include SSS
nuisances (speckle, range falloff) but EXCLUDE acoustic shadow — shadow is the
downstream detection signal, so making the backbone shadow-invariant would hurt.
"""
from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.augmentations.range_falloff import RangeFalloff
from src.augmentations.speckle import SpeckleNoise


def _random_resized_crop() -> A.RandomResizedCrop:
    """RandomResizedCrop(224), robust to the albumentations ~1.4.15 size-arg change."""
    scale = (0.4, 1.0)
    try:
        return A.RandomResizedCrop(size=(224, 224), scale=scale, p=1.0)
    except TypeError:
        return A.RandomResizedCrop(224, 224, scale=scale, p=1.0)


class SonarBYOLTransform:
    """Produce two independently-augmented (3,224,224) views of one tile."""

    def __init__(self) -> None:
        self.transform = A.Compose(
            [
                _random_resized_crop(),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                SpeckleNoise(p=0.5),
                RangeFalloff(nadir_axis="x", falloff_range=(0.3, 1.0), p=0.5),
                A.Normalize(mean=(0.55, 0.55, 0.55), std=(0.20, 0.20, 0.20)),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def _to_3ch(image: np.ndarray) -> np.ndarray:
        """Coerce HxW / HxWx1 single-channel uint8 to HxWx3 (YOLO stem wants RGB)."""
        if image.ndim == 2:
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
        image = self._to_3ch(image)
        v0 = self.transform(image=image)["image"]
        v1 = self.transform(image=image)["image"]
        return v0, v1
