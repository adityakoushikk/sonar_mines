"""Tests for YOLO backbone extraction and the checkpoint contract.

Requires torch + ultralytics. ``YOLO(f"{variant}.yaml")`` builds from a bundled
yaml (no network), so these are hermetic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import torch

from pretraining.ssl.backbone import (
    backbone_state_dict,
    extract_yolo_backbone,
    save_backbone_checkpoint,
)


def test_extract_returns_inferred_channels() -> None:
    """out_channels is inferred by a forward pass; yolov8n's SPPF gives 256."""
    yolo, backbone, channels = extract_yolo_backbone("yolov8n", cut=10)

    assert isinstance(channels, int)
    assert channels == 256  # documented value for yolov8n at cut=10

    # The backbone reuses the SAME module objects as yolo.model.model[:10], so a
    # forward through either should produce the same channel count.
    with torch.no_grad():
        feats = backbone(torch.zeros(1, 3, 224, 224))
    assert feats.shape[1] == channels


def test_backbone_shares_modules_with_yolo() -> None:
    """Training the backbone must mutate the parent YOLO (shared modules)."""
    yolo, backbone, _ = extract_yolo_backbone("yolov8n")
    # Identity, not just equality: the first backbone layer IS yolo's layer 0.
    assert backbone[0] is yolo.model.model[0]


def test_backbone_cut_is_inferred_from_yaml() -> None:
    """The cut is read from the architecture yaml: yolov8n's backbone is 10."""
    from pretraining.ssl.backbone import _infer_backbone_cut

    yolo, _, _ = extract_yolo_backbone("yolov8n")
    assert _infer_backbone_cut(yolo) == 10  # stem..SPPF are layers 0..9


def test_backbone_state_dict_within_inferred_cut() -> None:
    """All kept keys are backbone layers (index < the inferred cut)."""
    from pretraining.ssl.backbone import _infer_backbone_cut, _layer_index

    yolo, _, _ = extract_yolo_backbone("yolov8n")
    cut = _infer_backbone_cut(yolo)
    sd = backbone_state_dict(yolo)  # cut inferred from yaml, not hardcoded

    assert sd, "expected a non-empty backbone state dict"
    for key in sd:
        idx = _layer_index(key)
        assert idx is not None and idx < cut, f"non-backbone key {key!r}"


def test_backbone_state_dict_explicit_cut_is_subset() -> None:
    """An explicit smaller cut yields a strict subset (no longer raises)."""
    yolo, _, _ = extract_yolo_backbone("yolov8n")
    full = set(backbone_state_dict(yolo))            # inferred cut == 10
    smaller = set(backbone_state_dict(yolo, cut=8))  # layers 0..7 only
    assert smaller, "expected a non-empty subset"
    assert smaller < full


def test_save_checkpoint_matches_contract(tmp_path: Path) -> None:
    """save_backbone_checkpoint writes the exact 4-key contract dict."""
    yolo, _, channels = extract_yolo_backbone("yolov8n")
    path = tmp_path / "nested" / "byol_benthicat_backbone.pt"

    save_backbone_checkpoint(yolo, path, channels=channels, epoch=7, variant="yolov8n")

    assert path.exists()  # parent dirs were created
    ckpt = torch.load(path, map_location="cpu")
    assert set(ckpt) == {"backbone_state_dict", "model_variant", "channels", "epoch"}
    assert ckpt["model_variant"] == "yolov8n"
    assert ckpt["channels"] == channels
    assert ckpt["epoch"] == 7
    # Loadable back onto a fresh model's backbone with no unexpected keys.
    fresh, _, _ = extract_yolo_backbone("yolov8n")
    missing, unexpected = fresh.model.load_state_dict(
        ckpt["backbone_state_dict"], strict=False
    )
    assert not unexpected
