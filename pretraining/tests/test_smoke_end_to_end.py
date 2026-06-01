"""Synthetic end-to-end gate for the B6 pipeline, on CPU with no real data/Modal.

Fabricate BenthiCat-shaped tiles -> preprocess to shards -> train_byol one epoch
-> reload via load_ssl_benthicat -> run one model.train step on a synthetic
2-class set. Stage 5 is what the unit tests skip: that the SSL backbone slots
into Ultralytics' training graph without shape/key mismatches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("lightly")
pytest.importorskip("webdataset")
pytest.importorskip("albumentations")

import torch

from pretraining.preprocess import preprocess

_CONTRACT_KEYS = {"backbone_state_dict", "model_variant", "channels", "epoch"}
# 64px is the smallest that trains cleanly at yolov8n's stride of 32.
_IMG = 64


def _make_benthicat_tiles(tmp_path: Path, n: int = 8) -> Path:
    """Stage 1: write ``n`` 384x384 float32 tiles under a nested ``<sector>/``."""
    sector = tmp_path / "tiles" / "sectorA"
    sector.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        stem = f"N02_4_220807052300_xtf-CH12_batch0_ch0_r0_c{i}"
        np.save(sector / f"{stem}.npy", rng.random((384, 384), dtype=np.float32))
    return tmp_path / "tiles"


def _make_detection_dataset(tmp_path: Path) -> Path:
    """Stage 5 fixture: a minimal 2-class YOLO detection set + ``data.yaml``."""
    root = tmp_path / "det"
    rng = np.random.default_rng(1)
    label_line = "0 0.5 0.5 0.2 0.2\n"  # "<cls> <cx> <cy> <w> <h>"

    from PIL import Image

    for split, count in (("train", 2), ("val", 1)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(count):
            arr = (rng.random((_IMG, _IMG)) * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").convert("RGB").save(img_dir / f"{i}.png")
            (lbl_dir / f"{i}.txt").write_text(label_line)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: milco\n"
        "  1: nombo\n".format(root=root)
    )
    return data_yaml


def test_b6_end_to_end_cpu(tmp_path: Path) -> None:
    """preprocess -> train_byol -> load_ssl_benthicat -> model.train, on CPU."""
    from pretraining.ssl.byol import TrainConfig, train_byol

    # Stage 1+2: tiles -> shards
    input_dir = _make_benthicat_tiles(tmp_path, n=8)
    output_dir = tmp_path / "shards"
    manifest = preprocess(input_dir, output_dir, limit=8)

    assert manifest["n_tiles"] == 8
    shards = sorted(output_dir.glob("benthicat-*.tar"))
    assert manifest["n_shards"] >= 1
    assert len(shards) >= 1

    # Stage 3: BYOL pretraining on CPU
    cfg = TrainConfig(
        shards=str(shards[0]),
        out_dir=str(tmp_path / "ckpts"),
        variant="yolov8n",
        epochs=1,
        batch_size=2,
        num_workers=0,
        epoch_length=2,
        ckpt_every_epochs=1,
        wandb_enabled=False,
        mixed_precision="no",
    )
    ckpt_path = train_byol(cfg)

    assert Path(ckpt_path).name == "byol_benthicat_yolov8n.pt"
    assert Path(ckpt_path).exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert set(ckpt) == _CONTRACT_KEYS
    assert ckpt["model_variant"] == "yolov8n"
    assert ckpt["channels"] == 256
    assert all(k.startswith("model.") for k in ckpt["backbone_state_dict"])

    # Stage 4: downstream seam loads the checkpoint
    from src.models import load_ssl_benthicat

    model = load_ssl_benthicat(ckpt_path, num_classes=2, model_variant="yolov8n")
    assert model.model.nc == 2
    assert model.model.names == {0: "class_0", 1: "class_1"}

    # Stage 5: the SSL-initialised detector actually trains (no raise = wired right)
    data_yaml = _make_detection_dataset(tmp_path)
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=_IMG,
        batch=2,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
        project=str(tmp_path / "runs"),
        name="b6_smoke",
    )
