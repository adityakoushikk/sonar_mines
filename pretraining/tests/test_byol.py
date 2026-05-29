"""End-to-end CPU smoke test for BYOL training + the downstream load seam.

Trains for a couple of epochs on a tiny shard with a single CPU process
(accelerate degrades to no-mixed-precision), then verifies the emitted
checkpoint loads back onto a fresh detector via the real supervised loader.
This is the contract that ties the SSL half to ``src/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("lightly")
pytest.importorskip("accelerate")
pytest.importorskip("webdataset")
pytest.importorskip("albumentations")

import torch

from pretraining.preprocess import preprocess


def _tiny_shards(tmp_path: Path, n: int = 8) -> str:
    in_dir = tmp_path / "tiles"
    in_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        np.save(in_dir / f"t{i:03d}.npy", rng.random((48, 48)).astype(np.float32))
    out_dir = tmp_path / "shards"
    preprocess(in_dir, out_dir, maxcount=1000)
    return str(out_dir / "benthicat-000000.tar")


def test_train_byol_cpu_smoke_and_load(tmp_path: Path) -> None:
    """train_byol runs on CPU and the checkpoint loads onto a fresh YOLO."""
    from pretraining.ssl.byol import TrainConfig, train_byol

    cfg = TrainConfig(
        shards=_tiny_shards(tmp_path, n=8),
        out_dir=str(tmp_path / "ckpts"),
        variant="yolov8n",
        epochs=2,
        batch_size=2,
        num_workers=0,            # single-process CPU smoke
        epoch_length=4,           # 2 batches/epoch -> fast
        ckpt_every_epochs=1,
        wandb_enabled=False,
        mixed_precision="bf16",   # forced to "no" on CPU inside train_byol
    )

    ckpt_path = train_byol(cfg)
    assert Path(ckpt_path).name == "byol_benthicat_backbone.pt"
    assert Path(ckpt_path).exists()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert set(ckpt) == {"backbone_state_dict", "model_variant", "channels", "epoch"}
    assert ckpt["channels"] == 256

    # The real downstream seam: load_ssl_benthicat must accept this checkpoint.
    from src.models import load_ssl_benthicat

    model = load_ssl_benthicat(ckpt_path, num_classes=2, model_variant="yolov8n")
    assert model.model.nc == 2
    assert model.model.names == {0: "class_0", 1: "class_1"}


def test_byol_module_forward_shapes() -> None:
    """BYOL.forward returns prediction p; forward_momentum returns detached z."""
    from pretraining.ssl.backbone import extract_yolo_backbone
    from pretraining.ssl.byol import BYOL

    _, backbone, channels = extract_yolo_backbone("yolov8n")
    model = BYOL(backbone, in_dim=channels)

    x = torch.zeros(2, 3, 224, 224)
    p = model(x)
    z = model.forward_momentum(x)
    assert p.shape == (2, 256)      # prediction head output dim
    assert z.shape == (2, 256)      # projection head output dim
    assert not z.requires_grad      # target projection is detached
