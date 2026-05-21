"""Range-dependent intensity falloff augmentation.

In SSS the received signal attenuates with distance from the nadir line,
typically as a power law (and with TVG correction artifacts). We multiply
the image by a 1-D falloff profile along the range axis to synthesize
under-/over-correction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from albumentations import ImageOnlyTransform


class RangeFalloff(ImageOnlyTransform):
    """Apply a 1-D intensity falloff along the range axis.

    Args:
        nadir_axis: ``"x"`` or ``"y"`` — image axis along which range increases.
        falloff_exponent: Default exponent for the I(r) ~ (1 + r)^-exp model.
        falloff_range: ``(low, high)`` range from which the exponent is sampled
            per call. If both equal ``falloff_exponent``, behaviour is deterministic.
        always_apply: Albumentations flag.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        nadir_axis: str = "y",
        falloff_exponent: float = 0.0,
        falloff_range: tuple[float, float] = (0.0, 0.0),
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=1.0 if always_apply else p)
        if nadir_axis not in {"x", "y"}:
            raise ValueError(f"nadir_axis must be 'x' or 'y', got {nadir_axis!r}")
        self.nadir_axis = nadir_axis
        self.falloff_exponent = float(falloff_exponent)
        self.falloff_range = (float(falloff_range[0]), float(falloff_range[1]))

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Return ``img`` modulated by a falloff profile along ``self.nadir_axis``."""
        # 1. Sample an exponent for this image.
        low, high = self.falloff_range
        if low == high:
            exponent = self.falloff_exponent
        else:
            exponent = float(np.random.uniform(low, high))

        # No-op fast path
        if exponent == 0.0:
            return img

        h, w = img.shape[:2]

        # 2. Build a 1-D normalized range axis r in [0, 1] along the chosen axis.
        if self.nadir_axis == "y":

            # range = |distance from horizontal nadir line at image center|
            r = np.abs(np.linspace(-1.0, 1.0, h, dtype=np.float32))   # shape (H,), V-shape
            profile = (1.0 + r) ** (-exponent)
            profile = profile[:, None]
        else:  # "x"
            # range = |distance from vertical nadir line at image center|
            r = np.abs(np.linspace(-1.0, 1.0, w, dtype=np.float32))   # shape (W,), V-shape
            profile = (1.0 + r) ** (-exponent)
            profile = profile[None, :]

        # 3. Broadcast to (H, W) (and (H, W, 1) for multi-channel images).
        if img.ndim == 3:
            profile = profile[..., None]                            # shape (H, 1, 1) or (1, W, 1)

        # 4. Apply: I' = I * profile, then clip to dtype range.
        out = img.astype(np.float32) * profile
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            out = np.clip(out, info.min, info.max)
        else:
            out = np.clip(out, 0.0, 1.0)
        return out.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("nadir_axis", "falloff_exponent", "falloff_range")