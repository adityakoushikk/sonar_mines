"""Speckle-noise augmentation for SSS images.

SSS imagery is dominated by multiplicative speckle (a coherent-imaging
phenomenon). We model it as ``I' = I * S`` where ``S`` is sampled per-pixel
from a Rayleigh / Gamma / log-normal distribution centred at 1.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform


class SpeckleNoise(ImageOnlyTransform):
    """Multiply each pixel by a speckle field with mean 1.

    Args:
        sigma: Width parameter of the speckle distribution.
        distribution: One of ``"rayleigh"``, ``"gamma"``, ``"lognormal"``.
        always_apply: Albumentations base-class flag.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        sigma: float = 0.0,
        distribution: str = "rayleigh",
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma
        self.distribution = distribution

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Return ``img`` multiplied by a speckle field with mean 1."""
        # TODO: sample speckle field S of shape img.shape from self.distribution
        # TODO: return clip(img * S, valid range for img.dtype)
        raise NotImplementedError("TODO: implement speckle multiplication")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Albumentations hook for round-tripping config -> object."""
        return ("sigma", "distribution")
