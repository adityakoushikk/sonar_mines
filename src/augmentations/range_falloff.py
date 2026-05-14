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
        super().__init__(always_apply=always_apply, p=p)
        self.nadir_axis = nadir_axis
        self.falloff_exponent = falloff_exponent
        self.falloff_range = falloff_range

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Return ``img`` modulated by a falloff profile along ``self.nadir_axis``."""
        # TODO: sample exponent from self.falloff_range
        # TODO: build 1-D profile of shape (H,) or (W,) and broadcast over img
        # TODO: return clip(img * profile, valid range for img.dtype)
        raise NotImplementedError("TODO: implement range falloff")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("nadir_axis", "falloff_exponent", "falloff_range")
