"""Acoustic-shadow augmentation.

Objects on the seafloor block insonification of the area immediately behind
them (relative to the sensor), creating dark shadow regions whose length
encodes object height. We synthesize / strengthen these shadows.

Implemented as a DualTransform because shadow geometry can change the
effective extent of objects and may interact with bounding-box augmentation
in downstream code.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from albumentations import DualTransform


class AcousticShadow(DualTransform):
    """Synthesize acoustic-shadow strips behind detected/annotated objects.

    Args:
        nadir_axis: ``"x"`` or ``"y"`` — axis along which shadows extend.
        shadow_length_range: ``(low, high)`` shadow length as fraction of image.
        shadow_intensity: Multiplier applied inside the shadow region (0..1).
        always_apply: Albumentations flag.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        nadir_axis: str = "y",
        shadow_length_range: tuple[float, float] = (0.0, 0.0),
        shadow_intensity: float = 0.0,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.nadir_axis = nadir_axis
        self.shadow_length_range = shadow_length_range
        self.shadow_intensity = shadow_intensity

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Darken pixels in the shadow region(s) of ``img``."""
        # TODO: for each bbox passed via params, compute shadow polygon along
        # nadir_axis with length sampled from shadow_length_range
        # TODO: multiply pixels inside the polygon by shadow_intensity
        raise NotImplementedError("TODO: implement shadow synthesis for image")

    def apply_to_bbox(self, bbox: tuple[float, float, float, float], **params: Any) -> tuple[float, float, float, float]:
        """Return the bbox possibly adjusted to account for the added shadow.

        Most call sites will leave bboxes untouched; this hook exists in case
        we ever choose to extend the box to include the shadow.
        """
        # TODO: decide whether to extend bbox to include shadow; default = identity
        raise NotImplementedError("TODO: implement bbox handling for shadow")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("nadir_axis", "shadow_length_range", "shadow_intensity")
