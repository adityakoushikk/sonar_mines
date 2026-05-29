"""Extract and checkpoint a YOLOv8 backbone for self-supervised pretraining.

The trick that makes SSL transfer trivial later: we build the YOLO model from
its *architecture-only* yaml and wrap the first ``cut`` layer modules in an
``nn.Sequential``. Because ``nn.Sequential`` holds references to the *same*
module objects as ``yolo.model.model``, optimizing the sequential mutates the
parent YOLO in place. We therefore checkpoint by reading ``yolo.model``'s
state_dict with its native, non-re-indexed keys ("model.0.*" .. "model.9.*"),
which is exactly what ``YOLO.model.load_state_dict`` expects downstream.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


def extract_yolo_backbone(
    variant: str = "yolov8n",
    cut: int = 10,
) -> tuple[YOLO, nn.Sequential, int]:
    """Build a YOLO from yaml and return (yolo, backbone, out_channels).

    Args:
        variant: e.g. ``"yolov8n"``. Loaded as ``YOLO(f"{variant}.yaml")`` —
            architecture only, so nothing is downloaded. Never pass a ``.pt``.
        cut: Number of leading layer modules to treat as the "backbone".
            ``cut=10`` keeps stem .. SPPF for YOLOv8 (indices 0..9).

    Returns:
        ``(yolo, backbone, out_channels)`` where ``backbone`` reuses the same
        layer modules as ``yolo.model.model`` (training one updates the other),
        and ``out_channels`` is the channel count of the backbone's output,
        inferred by a forward pass (do NOT hardcode 256).
    """
    yolo = YOLO(f"{variant}.yaml")  # architecture only — bundled yaml, no download
    backbone = nn.Sequential(*list(yolo.model.model[:cut]))

    # Infer output channels with a dummy forward; 224 is the SSL crop size.
    backbone.eval()
    with torch.no_grad():
        feats = backbone(torch.zeros(1, 3, 224, 224))
    out_channels = int(feats.shape[1])

    return yolo, backbone, out_channels


def backbone_state_dict(yolo: YOLO, cut: int = 10) -> dict:
    """Return only the backbone layers' weights, keyed by full-model names.

    Keeps entries whose key matches ``model.<single-digit>.`` — i.e. layers
    0..9. NOTE: this single-digit regex assumes ``cut == 10``; for a different
    cut the pattern would need to widen to multi-digit indices.

    Args:
        yolo: The model returned by :func:`extract_yolo_backbone`.
        cut: Backbone cut point (documented assumption: 10).
    """
    if cut != 10:
        raise ValueError(
            f"backbone_state_dict's single-digit regex assumes cut==10, got {cut}"
        )
    return {
        k: v
        for k, v in yolo.model.state_dict().items()
        if re.match(r"model\.[0-9]\.", k)
    }


def save_backbone_checkpoint(
    yolo: YOLO,
    path: str | Path,
    *,
    channels: int,
    epoch: int,
    variant: str = "yolov8n",
) -> None:
    """Write the checkpoint-contract dict to ``path`` via ``torch.save``.

    The dict is the single source of truth shared by the trainer, the
    downstream loader (``load_ssl_benthicat``), and the tests:
    ``{"backbone_state_dict", "model_variant", "channels", "epoch"}``.

    Args:
        yolo: Model whose backbone weights to persist.
        path: Destination ``.pt`` path; parent dirs are created.
        channels: SPPF output channel count (256 for yolov8n).
        epoch: Epoch index recorded in the checkpoint.
        variant: Model variant string recorded in the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "backbone_state_dict": backbone_state_dict(yolo),
        "model_variant": variant,
        "channels": int(channels),
        "epoch": int(epoch),
    }
    torch.save(ckpt, str(path))
