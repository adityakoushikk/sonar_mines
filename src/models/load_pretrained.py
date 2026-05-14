"""Backbone/checkpoint loaders for the B1–B5 initialization ablation.

Every loader returns an `ultralytics.YOLO` model whose head is reset to the
number of target classes (2 for Santos: MILCO, NOMBO). Callers should not
assume any further reshaping has happened.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def load_random(model_variant: str, num_classes: int) -> YOLO:
    """B1: build a YOLOv8 from architecture only, with random weights everywhere.

    Args:
        model_variant: e.g. ``"yolov8n"``.
        num_classes: Number of detection classes for the new head.
    """
    # TODO: build YOLO(f"{model_variant}.yaml") to get architecture-only init
    # TODO: reset detection head for num_classes
    raise NotImplementedError("TODO: implement load_random")


def load_imagenet_backbone(model_variant: str, num_classes: int) -> YOLO:
    """B2: ImageNet-pretrained backbone, neck+head trained from scratch.

    Args:
        model_variant: e.g. ``"yolov8n"``.
        num_classes: Number of detection classes for the new head.
    """
    # TODO: build architecture-only YOLO, then load ImageNet backbone weights
    # (torchvision Resnet/EfficientNet -> YOLO backbone layer mapping)
    # TODO: reset detection head for num_classes
    raise NotImplementedError("TODO: implement load_imagenet_backbone")


def load_coco_full(weights_path: str | Path, num_classes: int) -> YOLO:
    """B3: full COCO-pretrained checkpoint, head re-shaped to num_classes.

    Args:
        weights_path: Path / Ultralytics name of the .pt checkpoint.
        num_classes: Number of detection classes for the new head.
    """
    # TODO: YOLO(weights_path); reset detection head for num_classes
    raise NotImplementedError("TODO: implement load_coco_full")


def load_sonar_fls(weights_path: str | Path, num_classes: int) -> YOLO:
    """B4: Valdenegro-Toro forward-look-sonar pretrained YOLOv8.

    Args:
        weights_path: Path to the FLS-pretrained .pt checkpoint.
        num_classes: Number of detection classes for the new head.
    """
    # TODO: YOLO(weights_path); reset head; sanity-check tensor shapes
    raise NotImplementedError("TODO: implement load_sonar_fls")


def load_sonar_uatd(weights_path: str | Path, num_classes: int) -> YOLO:
    """B5: our UATD-pretrained YOLOv8 (matched modality).

    Args:
        weights_path: Path to the UATD-pretrained .pt checkpoint.
        num_classes: Number of detection classes for the new head.
    """
    # TODO: YOLO(weights_path); reset head; sanity-check tensor shapes
    raise NotImplementedError("TODO: implement load_sonar_uatd")
