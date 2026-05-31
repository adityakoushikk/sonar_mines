"""Extract and checkpoint a YOLO backbone for self-supervised pretraining.

The trick that makes SSL transfer trivial later: we build the YOLO model from
its *architecture-only* yaml and wrap the first ``cut`` layer modules in an
``nn.Sequential``. Because ``nn.Sequential`` holds references to the *same*
module objects as ``yolo.model.model``, optimizing the sequential mutates the
parent YOLO in place. We therefore checkpoint by reading ``yolo.model``'s
state_dict with its native, non-re-indexed keys ("model.0.*" .. ), which is
exactly what ``YOLO.model.load_state_dict`` expects downstream.

The backbone cut is **variant-agnostic**: rather than hardcoding ``10`` (the
YOLOv8 stem..SPPF boundary) we read the ``backbone:`` section length straight
from the architecture yaml, so the same code extracts the right prefix for
``yolov8n``, ``yolo26n``, or any other variant. The round-trip test is the
empirical gate that the chosen prefix actually reloads cleanly.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


def _infer_backbone_cut(yolo: YOLO) -> int:
    """Number of leading modules that form the backbone, from the model yaml.

    Ultralytics architecture yamls split layers into explicit ``backbone:`` and
    ``head:`` lists; each ``backbone`` entry yields exactly one top-level module
    in ``yolo.model.model``, so the backbone is its first ``len(backbone)``
    modules. Reading it here keeps the cut correct across variants (yolov8n = 10
    ending at SPPF; yolo26n differs) instead of hardcoding a constant.
    """
    yaml_cfg = getattr(yolo.model, "yaml", None)
    if not isinstance(yaml_cfg, dict) or "backbone" not in yaml_cfg:
        raise ValueError(
            "cannot infer backbone cut: model.yaml has no 'backbone' section"
        )
    return len(yaml_cfg["backbone"])


def _layer_index(key: str) -> int | None:
    """Top-level layer index from a state_dict key, e.g. ``model.9.cv1...`` -> 9.

    Matches multi-digit indices (``model.10.`` .. ``model.22.``) so it works for
    necks/heads beyond layer 9, not just the single-digit backbone.
    """
    match = re.match(r"model\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def extract_yolo_backbone(
    variant: str = "yolov8n",
    cut: int | None = None,
) -> tuple[YOLO, nn.Sequential, int]:
    """Build a YOLO from yaml and return (yolo, backbone, out_channels).

    Args:
        variant: e.g. ``"yolov8n"`` or ``"yolo26n"``. Loaded as
            ``YOLO(f"{variant}.yaml")`` — architecture only, so nothing is
            downloaded. Never pass a ``.pt``.
        cut: Number of leading layer modules to treat as the "backbone". When
            ``None`` (default) it is inferred from the architecture yaml's
            ``backbone:`` section, which is correct for any variant (10 for
            yolov8n). Pass an explicit int only to override.

    Returns:
        ``(yolo, backbone, out_channels)`` where ``backbone`` reuses the same
        layer modules as ``yolo.model.model`` (training one updates the other),
        and ``out_channels`` is the channel count of the backbone's output,
        inferred by a forward pass (do NOT hardcode 256 — it is variant-specific).
    """
    yolo = YOLO(f"{variant}.yaml")  # architecture only — bundled yaml, no download
    if cut is None:
        cut = _infer_backbone_cut(yolo)
    backbone = nn.Sequential(*list(yolo.model.model[:cut]))

    # Infer output channels with a dummy forward; 224 is the SSL crop size.
    backbone.eval()
    with torch.no_grad():
        feats = backbone(torch.zeros(1, 3, 224, 224))
    out_channels = int(feats.shape[1])

    return yolo, backbone, out_channels


def backbone_state_dict(yolo: YOLO, cut: int | None = None) -> dict:
    """Return only the backbone layers' weights, keyed by full-model names.

    Keeps entries whose top-level layer index is ``< cut`` (i.e. the backbone
    prefix), leaving neck/head keys out. The index match is multi-digit, so the
    cut may exceed 9 for variants whose backbone is longer than YOLOv8's.

    Args:
        yolo: The model returned by :func:`extract_yolo_backbone`.
        cut: Backbone cut point. When ``None`` it is inferred from the model
            yaml's ``backbone:`` section (same value extract uses).
    """
    if cut is None:
        cut = _infer_backbone_cut(yolo)
    return {
        k: v
        for k, v in yolo.model.state_dict().items()
        if (idx := _layer_index(k)) is not None and idx < cut
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
    ``{"backbone_state_dict", "model_variant", "channels", "epoch"}``. The
    backbone slice is taken with the variant-inferred cut, so this is correct
    for yolov8n, yolo26n, etc. without a per-variant code path.

    Args:
        yolo: Model whose backbone weights to persist.
        path: Destination ``.pt`` path; parent dirs are created.
        channels: Backbone output channel count (256 for yolov8n).
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
