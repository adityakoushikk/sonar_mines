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
        sigma: Width parameter of the speckle distribution. Larger = noisier.
            Typical values: 0.1 (subtle) to 0.4 (heavy). For "gamma", sigma
            controls 1/sqrt(shape), so sigma=0.2 corresponds to shape=25.
        distribution: One of ``"rayleigh"``, ``"gamma"``, ``"lognormal"``.
        always_apply: Albumentations base-class flag.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        sigma: float = 0.2,
        distribution: str = "rayleigh",
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=1.0 if always_apply else p)
        if distribution not in {"rayleigh", "gamma", "lognormal"}:
            raise ValueError(
                f"distribution must be one of rayleigh/gamma/lognormal, got {distribution!r}"
            )
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)
        self.distribution = distribution

    def _sample_speckle_field(self, shape: tuple[int, ...]) -> np.ndarray:
        """Sample a multiplicative speckle field S with mean ~1."""
        if self.sigma == 0.0:
            return np.ones(shape, dtype=np.float32)

        if self.distribution == "rayleigh":
            # Rayleigh(scale=s) has mean = s * sqrt(pi/2).
            # We pick s so that E[S] = 1, then control width via sigma by
            # blending with the identity: S = 1 + sigma * (R - 1).
            r = np.random.rayleigh(scale=1.0, size=shape).astype(np.float32)
            r /= np.sqrt(np.pi / 2.0)  # now E[R] = 1
            s = 1.0 + self.sigma * (r - 1.0)

        elif self.distribution == "gamma":
            # Gamma with shape k, scale 1/k has mean 1 and variance 1/k.
            # We want Var[S] ~ sigma^2, so k = 1/sigma^2.
            k = 1.0 / (self.sigma ** 2)
            s = np.random.gamma(shape=k, scale=1.0 / k, size=shape).astype(np.float32)

        else:  # lognormal
            # LogNormal(mu, sig) has mean exp(mu + sig^2/2).
            # Set mu = -sigma^2 / 2 so that E[S] = 1; sigma controls width.
            s = np.random.lognormal(
                mean=-(self.sigma ** 2) / 2.0,
                sigma=self.sigma,
                size=shape,
            ).astype(np.float32)

        return s

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Return ``img`` multiplied by a speckle field with mean 1."""
        s = self._sample_speckle_field(img.shape)

        # Multiply in float, clip to the original dtype's valid range.
        out = img.astype(np.float32) * s

        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            out = np.clip(out, info.min, info.max)
        else:
            # assume float images live in [0, 1]
            out = np.clip(out, 0.0, 1.0)

        return out.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Albumentations hook for round-tripping config -> object."""
        return ("sigma", "distribution")
