"""mAP computation and W&B metric logging helpers.

Ultralytics' `model.val()` already returns mAP — these helpers exist for
(a) the classical baseline, which has no Ultralytics validator, and
(b) post-hoc analysis that joins predictions from multiple runs.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def compute_map(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    iou_thresholds: Sequence[float] = (0.5,),
) -> float:
    """Mean Average Precision over a dataset.

    Args:
        predictions: Per-image detection lists (bbox, score, class).
        targets: Per-image ground-truth lists (bbox, class).
        iou_thresholds: IoU thresholds to average AP over (e.g. (0.5,) for mAP50,
            ``tuple(np.arange(0.5, 1.0, 0.05))`` for mAP50-95).

    Returns:
        mAP averaged across classes and IoU thresholds.
    """
    # TODO: compute per-class AP at each IoU; average across classes and thresholds
    raise NotImplementedError("TODO: implement compute_map")


def compute_per_class_ap(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[int, float]:
    """Per-class Average Precision at a single IoU threshold."""
    # TODO: sort detections by score desc; compute precision/recall curve per class
    raise NotImplementedError("TODO: implement compute_per_class_ap")


def log_metrics_to_wandb(metrics: Mapping[str, Any], step: int | None = None) -> None:
    """Push a metrics dict to the active W&B run."""
    # TODO: import wandb; call wandb.log(dict(metrics), step=step) if a run is active
    raise NotImplementedError("TODO: implement log_metrics_to_wandb")
