"""Synthetic end-to-end gate for the B6 BYOL pretraining pipeline.

This is the *full* B6 contract exercised on the CPU with zero real data and no
Modal: from raw BenthiCat-shaped ``.npy`` tiles all the way to a YOLO detector
that actually takes a training step on the SSL-initialised weights.

The five stages mirror the production flow:

1. fabricate BenthiCat-layout tiles (``<sector>/<file>.npy``, 384x384 float32);
2. ``preprocess`` them into WebDataset shards (lossless uint8 PNG);
3. ``train_byol`` for one tiny epoch on a single CPU process and emit the
   checkpoint-contract ``.pt``;
4. reload that checkpoint onto a fresh detector via the real ``src/`` seam
   (``load_ssl_benthicat``);
5. *prove the weights are trainable* by running one ``model.train`` step on a
   synthetic 2-class detection set.

Stage 5 is the part the narrower unit/integration tests skip: it closes the
loop by confirming the SSL backbone slots into Ultralytics' training graph
without shape or key mismatches. Everything is sized to run in seconds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# B6 spans torch + ultralytics + lightly + webdataset + albumentations; skip the
# whole module cleanly in a minimal env rather than erroring on import.
pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("lightly")
pytest.importorskip("webdataset")
pytest.importorskip("albumentations")

import torch

from pretraining.preprocess import preprocess

# The checkpoint-contract keys shared by trainer, loader, and these tests.
_CONTRACT_KEYS = {"backbone_state_dict", "model_variant", "channels", "epoch"}

# A 1x1-ish PNG is too small for Ultralytics' letterbox/stride math; 64px is the
# smallest size that trains cleanly at the n-variant stride of 32.
_IMG = 64


def _make_benthicat_tiles(tmp_path: Path, n: int = 8) -> Path:
    """Stage 1: write ``n`` BenthiCat-shaped tiles under a fake ``<sector>/``.

    Tiles are 384x384 float32 in [0, 1) — the real BenthiCat tile shape/range,
    so the uint8 quantization in ``preprocess`` exercises its true code path.
    The nested sector dir verifies ``preprocess``'s recursive ``rglob`` walk.
    """
    sector = tmp_path / "tiles" / "sectorA"
    sector.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        # Full BenthiCat-style stem so the preserved __key__ is realistic.
        stem = f"N02_4_220807052300_xtf-CH12_batch0_ch0_r0_c{i}"
        np.save(sector / f"{stem}.npy", rng.random((384, 384), dtype=np.float32))
    return tmp_path / "tiles"


def _make_detection_dataset(tmp_path: Path) -> Path:
    """Stage 5 fixture: a minimal 2-class YOLO detection set + ``data.yaml``.

    Two train images and one val image, each 64x64 with a single centred box
    labelled class 0. The label/class names match Santos (MILCO/NOMBO) so the
    detector head is rebuilt for the same 2-class problem the SSL init targets.
    Returns the path to the written ``data.yaml``.
    """
    root = tmp_path / "det"
    rng = np.random.default_rng(1)
    # One YOLO-format line: "<cls> <cx> <cy> <w> <h>" — a centred 0.2x0.2 box.
    label_line = "0 0.5 0.5 0.2 0.2\n"

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
    # Ultralytics resolves train/val relative to ``path``; names give the 2 classes.
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
    """preprocess -> train_byol -> load_ssl_benthicat -> model.train, on CPU.

    The single gate that proves the whole B6 chain is wired correctly: the
    emitted checkpoint honours the contract, the downstream loader accepts it,
    and the resulting detector takes a real training step without raising.
    """
    from pretraining.ssl.byol import TrainConfig, train_byol

    # --- Stage 1+2: tiles -> shards -----------------------------------------
    input_dir = _make_benthicat_tiles(tmp_path, n=8)
    output_dir = tmp_path / "shards"
    manifest = preprocess(input_dir, output_dir, limit=8)

    assert manifest["n_tiles"] == 8
    # At least one shard tar must land on disk.
    shards = sorted(output_dir.glob("benthicat-*.tar"))
    assert manifest["n_shards"] >= 1
    assert len(shards) >= 1

    # --- Stage 3: BYOL pretraining on CPU -----------------------------------
    cfg = TrainConfig(
        shards=str(shards[0]),
        out_dir=str(tmp_path / "ckpts"),
        variant="yolov8n",
        epochs=1,
        batch_size=2,
        num_workers=0,            # single-process CPU smoke
        epoch_length=2,           # 1 batch/epoch -> seconds
        ckpt_every_epochs=1,
        wandb_enabled=False,
        mixed_precision="no",     # explicit CPU path (no bf16/fp16 on CPU)
    )
    ckpt_path = train_byol(cfg)

    # Checkpoint exists with the canonical name and full contract.
    assert Path(ckpt_path).name == "byol_benthicat_backbone.pt"
    assert Path(ckpt_path).exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert set(ckpt) == _CONTRACT_KEYS
    assert ckpt["model_variant"] == "yolov8n"
    assert ckpt["channels"] == 256  # SPPF output channels for yolov8n
    # Backbone-only state: keys are full-model names for layers 0..9.
    assert all(k.startswith("model.") for k in ckpt["backbone_state_dict"])

    # --- Stage 4: downstream seam loads the checkpoint ----------------------
    from src.models import load_ssl_benthicat

    model = load_ssl_benthicat(ckpt_path, num_classes=2, model_variant="yolov8n")
    assert model.model.nc == 2
    assert model.model.names == {0: "class_0", 1: "class_1"}

    # --- Stage 5: the SSL-initialised detector actually trains --------------
    data_yaml = _make_detection_dataset(tmp_path)
    # If this returns without raising, the SSL backbone integrated into the
    # Ultralytics training graph (head rebuilt for 2 classes) and a step ran.
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
