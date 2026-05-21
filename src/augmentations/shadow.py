"""Acoustic-shadow augmentation."""
from __future__ import annotations

from typing import Any

import numpy as np
from albumentations import DualTransform


class AcousticShadow(DualTransform):
    """Synthesize acoustic-shadow strips behind annotated objects.

    Args:
        nadir_axis: ``"x"`` or ``"y"`` — axis along which shadows extend.
        shadow_length_range: ``(low, high)`` shadow length as fraction of image
            size along ``nadir_axis``. Sampled per bbox.
        shadow_intensity: Multiplier applied inside the shadow region (0..1).
            0.0 = pure black, 1.0 = no effect.
        nadir_position: Where the nadir line lives along ``nadir_axis``, as a
            fraction in [0, 1]. 0.5 for centered (dual-channel images like the
            Pessanha Santos dataset); 0.0 or 1.0 for edge-nadir layouts.
        always_apply: Albumentations flag.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        nadir_axis: str = "x",
        shadow_length_range: tuple[float, float] = (0.05, 0.20),
        shadow_intensity: float = 0.1,
        nadir_position: float = 0.5,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=1.0 if always_apply else p)
        if nadir_axis not in {"x", "y"}:
            raise ValueError(f"nadir_axis must be 'x' or 'y', got {nadir_axis!r}")
        if not 0.0 <= shadow_intensity <= 1.0:
            raise ValueError(f"shadow_intensity must be in [0, 1], got {shadow_intensity}")
        if not 0.0 <= nadir_position <= 1.0:
            raise ValueError(f"nadir_position must be in [0, 1], got {nadir_position}")
        self.nadir_axis = nadir_axis
        self.shadow_length_range = (float(shadow_length_range[0]), float(shadow_length_range[1]))
        self.shadow_intensity = float(shadow_intensity)
        self.nadir_position = float(nadir_position)

    @property
    def targets_as_params(self) -> list[str]:
        """Tell Albumentations we need access to bboxes when generating params."""
        return ["bboxes"]

    def _sample_shadow_params(self, bboxes: list[tuple]) -> dict[str, Any]:
        """Sample one shadow length per bbox, so image+bbox apply() see the same draws."""
        n = len(bboxes)
        if n == 0:
            return {
                "shadow_bboxes": bboxes,
                "shadow_lengths": np.array([], dtype=np.float32),
            }
        low, high = self.shadow_length_range
        shadow_lengths = np.random.uniform(low, high, size=n).astype(np.float32)
        return {"shadow_bboxes": bboxes, "shadow_lengths": shadow_lengths}

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        bboxes = data.get("bboxes", [])
        return self._sample_shadow_params(bboxes)

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        bboxes = params.get("bboxes", [])
        return self._sample_shadow_params(bboxes)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Shadows change image intensities only; bounding boxes stay unchanged."""
        return bboxes

    def apply(
        self,
        img: np.ndarray,
        shadow_bboxes: list[tuple] = (),
        shadow_lengths: np.ndarray = None,
        **params: Any,
    ) -> np.ndarray:
        """Darken pixels in the shadow region behind each bbox with a soft trailing edge."""
        if len(shadow_bboxes) == 0 or shadow_lengths is None or len(shadow_lengths) == 0:
            return img

        h, w = img.shape[:2]
        out = img.astype(np.float32).copy()

        if self.nadir_axis == "x":
            axis_len = w
        else:
            axis_len = h
        nadir_coord = self.nadir_position * axis_len

        for bbox, shadow_frac in zip(shadow_bboxes, shadow_lengths):
            x_min_n, y_min_n, x_max_n, y_max_n = bbox[:4]
            x_min, x_max = int(x_min_n * w), int(x_max_n * w)
            y_min, y_max = int(y_min_n * h), int(y_max_n * h)

            shadow_len_px = int(shadow_frac * axis_len)
            if shadow_len_px <= 0:
                continue

            if self.nadir_axis == "x":
                bbox_center_x = 0.5 * (x_min + x_max)
                outward_is_positive = bbox_center_x >= nadir_coord
                if outward_is_positive:
                    sx_start, sx_end = x_max, min(w, x_max + shadow_len_px)
                else:
                    sx_start, sx_end = max(0, x_min - shadow_len_px), x_min
                sy_start, sy_end = y_min, y_max
            else:  # "y"
                bbox_center_y = 0.5 * (y_min + y_max)
                outward_is_positive = bbox_center_y >= nadir_coord
                if outward_is_positive:
                    sy_start, sy_end = y_max, min(h, y_max + shadow_len_px)
                else:
                    sy_start, sy_end = max(0, y_min - shadow_len_px), y_min
                sx_start, sx_end = x_min, x_max

            if sx_end <= sx_start or sy_end <= sy_start:
                continue

            # --- Build a soft 1-D falloff profile along the shadow direction ---
            # darkest (= shadow_intensity) at the bbox-adjacent end,
            # back to 1.0 (no effect) at the far end.
            length = sx_end - sx_start if self.nadir_axis == "x" else sy_end - sy_start
            ramp = np.linspace(self.shadow_intensity, 1.0, length, dtype=np.float32)

            # If the shadow extends in the negative direction (toward port / toward top),
            # the darkest pixel is at the END of the slice, not the start.
            if not outward_is_positive:
                ramp = ramp[::-1]

            # --- Broadcast the 1-D ramp to the 2-D shadow region ---
            if self.nadir_axis == "x":
                mask = ramp[None, :]          # shape (1, length)
            else:
                mask = ramp[:, None]          # shape (length, 1)

            if out.ndim == 3:
                mask = mask[..., None]        # add channel dim if needed

            out[sy_start:sy_end, sx_start:sx_end] *= mask

        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            out = np.clip(out, info.min, info.max)
        else:
            out = np.clip(out, 0.0, 1.0)
        return out.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "nadir_axis",
            "shadow_length_range",
            "shadow_intensity",
            "nadir_position",
        )
