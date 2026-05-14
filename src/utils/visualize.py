"""Bounding-box and augmentation visualization helpers.

Used both interactively (in `notebooks/`) and from `src.train.py` to attach
example images to the W&B run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def plot_bbox_predictions(
    image: np.ndarray,
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]] | None = None,
    class_names: Sequence[str] | None = None,
    save_path: Path | str | None = None,
) -> Any:
    """Render an image with predicted boxes (and optional GT) overlaid.

    Args:
        image: HxW or HxWxC numpy array.
        predictions: Detections to draw (bbox, score, class).
        targets: Optional ground-truth boxes drawn in a contrasting colour.
        class_names: Index -> name mapping for labels.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure (so the caller can attach it to W&B).
    """
    # TODO: build matplotlib Figure with image + boxes; optionally save
    raise NotImplementedError("TODO: implement plot_bbox_predictions")


def plot_augmented_samples(
    image: np.ndarray,
    pipeline: Any,
    n_samples: int = 8,
    save_path: Path | str | None = None,
) -> Any:
    """Render a grid of N augmentation samples from ``pipeline`` applied to ``image``.

    Args:
        image: Source image to repeatedly augment.
        pipeline: An Albumentations Compose to call ``n_samples`` times.
        n_samples: Grid size.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    # TODO: build n_samples grid; call pipeline(image) repeatedly; assemble figure
    raise NotImplementedError("TODO: implement plot_augmented_samples")
