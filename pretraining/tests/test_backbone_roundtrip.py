"""Contract gate: backbone weights trained in pretraining/ reload intact via the
src/ seam. Asserts (a) byte-for-byte round-trip, (b) a fresh strict=False load
has no unexpected keys and only neck/head (index >= cut) missing, (c) nc is set.

Parametrized over yolov8n/yolo26n. A variant whose yaml is absent is skipped; one
that builds but fails to round-trip is a real failure (not hidden).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import torch
from ultralytics import YOLO

from pretraining.ssl.backbone import (
    _infer_backbone_cut,
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


def _require_variant(variant: str) -> None:
    """Skip iff this ultralytics cannot even build the variant's yaml.

    Separated from extraction so that a *buildable* variant which then fails to
    extract or round-trip surfaces as a real failure rather than a silent skip —
    that failure mode is exactly what this gate exists to catch (e.g. a variant
    whose backbone is not a clean sequential prefix).
    """
    try:
        YOLO(f"{variant}.yaml")
    except Exception as exc:  # noqa: BLE001 - variant yaml absent in this install
        pytest.skip(f"variant {variant!r} yaml unavailable in this ultralytics: {exc}")


@pytest.mark.parametrize("variant", ["yolov8n", "yolo26n"])
def test_backbone_checkpoint_roundtrips_through_load_seam(
    tmp_path: Path, variant: str
) -> None:
    """Trained backbone weights survive save -> load_ssl_benthicat exactly."""
    _require_variant(variant)
    yolo, backbone, channels = extract_yolo_backbone(variant)
    cut = _infer_backbone_cut(yolo)  # variant-specific backbone boundary
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
        yolo, ckpt_path, channels=channels, epoch=3, variant=variant
    )

    # The real downstream seam: build a fresh detector and load the SSL backbone.
    model = load_ssl_benthicat(ckpt_path, num_classes=2, model_variant=variant)

    # (a) Every saved key must reappear byte-for-byte in the reloaded model.
    reloaded = model.model.state_dict()
    for key, tensor in saved.items():
        assert key in reloaded, f"saved backbone key missing after reload: {key!r}"
        assert torch.equal(reloaded[key], tensor), f"weight changed on round-trip: {key!r}"

    # (b) A strict=False load on a FRESH model: nothing unexpected, and every
    # missing key is a neck/head layer (index >= cut) — the cut keeps exactly the
    # backbone layers 0..cut-1 and only those.
    fresh_yolo, _, _ = extract_yolo_backbone(variant)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = fresh_yolo.model.load_state_dict(
        ckpt["backbone_state_dict"], strict=False
    )
    assert unexpected == [], f"unexpected keys when loading backbone slice: {unexpected}"
    assert missing, "expected neck/head keys to be reported missing"
    for key in missing:
        assert _layer_index(key) >= cut, f"missing key inside the backbone cut: {key!r}"

    # (c) The reloaded detector records the requested class count.
    assert model.model.nc == 2
