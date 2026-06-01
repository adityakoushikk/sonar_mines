"""Extract and checkpoint a YOLO backbone for self-supervised pretraining.

The backbone ``nn.Sequential`` shares modules with ``yolo.model.model``, so
training it mutates the parent; the cut is read from the yaml's ``backbone:``
section so it is correct for any variant (yolov8n, yolo26n, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


def _infer_backbone_cut(yolo: YOLO) -> int:
    yaml_cfg = getattr(yolo.model, "yaml", None)
    if not isinstance(yaml_cfg, dict) or "backbone" not in yaml_cfg:
        raise ValueError("cannot infer backbone cut: model.yaml has no 'backbone'")
    return len(yaml_cfg["backbone"])


def _layer_index(key: str) -> int | None:
    match = re.match(r"model\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def extract_yolo_backbone(
    variant: str = "yolov8n",
    cut: int | None = None,
) -> tuple[YOLO, nn.Sequential, int]:
    """Build a YOLO from yaml; return (yolo, backbone sharing its modules, out_channels)."""
    yolo = YOLO(f"{variant}.yaml")
    if cut is None:
        cut = _infer_backbone_cut(yolo)
    backbone = nn.Sequential(*list(yolo.model.model[:cut]))
    backbone.eval()
    with torch.no_grad():
        feats = backbone(torch.zeros(1, 3, 224, 224))
    return yolo, backbone, int(feats.shape[1])


def backbone_state_dict(yolo: YOLO, cut: int | None = None) -> dict:
    """Backbone weights only (layer index < cut), keyed by full-model names."""
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
    """Save the {backbone_state_dict, model_variant, channels, epoch} contract dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state_dict": backbone_state_dict(yolo),
            "model_variant": variant,
            "channels": int(channels),
            "epoch": int(epoch),
        },
        str(path),
    )
