"""Empirical gate for the YOLOv8 backbone layer cut + checkpoint round-trip.

This is the load-bearing contract test between the SSL trainer and the
supervised loader: it proves that weights *trained* in ``pretraining/`` arrive
*intact* in a fresh detector built by ``src/``. We simulate training by nudging
every backbone parameter, serialize via the shared checkpoint contract, reload
through the real :func:`load_ssl_benthicat` seam, and then assert three things:

  (a) every saved backbone tensor reappears byte-for-byte in the reloaded model
      (the cut + key naming actually round-trips, not just "loads without error");
  (b) a strict=False load on a *fresh* model has no unexpected keys and every
      *missing* key is a neck/head layer (index >= 10) — i.e. the cut at 10 keeps
      exactly the backbone and nothing leaks the other way;
  (c) the reloaded detector records the requested class count.

Builds YOLO from the bundled ``*.yaml`` (no network), so this stays hermetic.
"""
from __future__ import annotations

import re
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
from src.models.load_pretrained import load_ssl_benthicat

# "model.<idx>." — capture the integer layer index between the first two dots.
_LAYER_IDX = re.compile(r"^model\.(\d+)\.")


def _layer_index(key: str) -> int:
    """Return the layer index N parsed from a ``model.N.*`` state-dict key."""
    match = _LAYER_IDX.match(key)
    assert match is not None, f"key not in expected 'model.N.' form: {key!r}"
    return int(match.group(1))


def test_backbone_checkpoint_roundtrips_through_load_seam(tmp_path: Path) -> None:
    """Trained backbone weights survive save -> load_ssl_benthicat exactly."""
    yolo, backbone, channels = extract_yolo_backbone("yolov8n")
    assert channels > 0  # inferred by a forward pass, never hardcoded

    # Simulate a training step: shift every backbone weight off its init so a
    # byte-for-byte match later can only come from our checkpoint, not from the
    # fresh model happening to share the architecture-only initialization.
    for param in backbone.parameters():
        param.data.add_(0.01)

    # Snapshot the (now-perturbed) backbone weights via the contract helper, and
    # clone so later in-place ops cannot alias what we compare against.
    saved = {k: v.clone() for k, v in backbone_state_dict(yolo).items()}
    assert saved, "expected a non-empty backbone state dict"

    ckpt_path = tmp_path / "checkpoints" / "byol_benthicat_backbone.pt"
    save_backbone_checkpoint(
        yolo, ckpt_path, channels=channels, epoch=3, variant="yolov8n"
    )

    # The real downstream seam: build a fresh detector and load the SSL backbone.
    model = load_ssl_benthicat(ckpt_path, num_classes=2, model_variant="yolov8n")

    # (a) Every saved key must reappear byte-for-byte in the reloaded model.
    reloaded = model.model.state_dict()
    for key, tensor in saved.items():
        assert key in reloaded, f"saved backbone key missing after reload: {key!r}"
        assert torch.equal(reloaded[key], tensor), f"weight changed on round-trip: {key!r}"

    # (b) A strict=False load on a FRESH model: nothing unexpected, and every
    # missing key is a neck/head layer (index >= 10) — the cut keeps backbone
    # layers 0..9 and only those.
    fresh_yolo, _, _ = extract_yolo_backbone("yolov8n")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = fresh_yolo.model.load_state_dict(
        ckpt["backbone_state_dict"], strict=False
    )
    assert unexpected == [], f"unexpected keys when loading backbone slice: {unexpected}"
    assert missing, "expected neck/head keys to be reported missing"
    for key in missing:
        assert _layer_index(key) >= 10, f"missing key inside the backbone cut: {key!r}"

    # (c) The reloaded detector records the requested class count.
    assert model.model.nc == 2
