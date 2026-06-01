"""Backbone/checkpoint loaders for the B1–B6 initialization ablation.

Every loader returns an `ultralytics.YOLO` model whose head is reset to the
number of target classes (2 for Santos: MILCO, NOMBO). Callers should not
assume any further reshaping has happened.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO


def _validate_num_classes(num_classes: int) -> None:
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")


def _model_yaml_name(model_variant: str) -> str:
    path = Path(model_variant)
    if path.suffix in {".yaml", ".yml"}:
        return str(path)
    return f"{model_variant}.yaml"


def _class_names(num_classes: int) -> dict[int, str]:
    return {i: f"class_{i}" for i in range(num_classes)}


def _apply_class_metadata(model: YOLO, num_classes: int) -> YOLO:
    model.model.nc = num_classes
    model.model.names = _class_names(num_classes)
    if hasattr(model.model, "yaml"):
        model.model.yaml["nc"] = num_classes
    return model


def _detection_model_from_yaml(yaml_cfg: dict, num_classes: int):
    """Build an Ultralytics DetectionModel with a target class count."""
    from ultralytics.nn.tasks import DetectionModel

    cfg = deepcopy(yaml_cfg)
    cfg["nc"] = num_classes
    return DetectionModel(cfg, nc=num_classes, verbose=False)


def _replace_with_target_nc_model(model: YOLO, num_classes: int) -> YOLO:
    """Replace the wrapped module with the same architecture at target nc."""
    old_model = model.model
    target_model = _detection_model_from_yaml(old_model.yaml, num_classes)
    target_model.args = getattr(old_model, "args", {})
    target_model.task = getattr(old_model, "task", "detect")
    model.model = target_model
    model.task = "detect"
    model.overrides["task"] = "detect"
    return _apply_class_metadata(model, num_classes)


def _randomize_detector(model: YOLO, num_classes: int) -> YOLO:
    """Keep compatible non-detector tensors and replace Detect with random init."""
    old_model = model.model
    target_model = _detection_model_from_yaml(old_model.yaml, num_classes)
    target_state = target_model.state_dict()
    old_state = old_model.state_dict()
    detector_prefix = f"model.{len(old_model.model) - 1}."

    compatible_state = {
        key: value
        for key, value in old_state.items()
        if (
            key in target_state
            and target_state[key].shape == value.shape
            and not key.startswith(detector_prefix)
        )
    }
    target_model.load_state_dict(compatible_state, strict=False)
    target_model.args = getattr(old_model, "args", {})
    target_model.task = getattr(old_model, "task", "detect")
    model.model = target_model
    model.task = "detect"
    model.overrides["task"] = "detect"
    return _apply_class_metadata(model, num_classes)


def _load_compatible_non_detector_weights(target_model, source_model) -> None:
    """Copy matching non-detector tensors from source_model into target_model."""
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()
    source_detector_prefix = f"model.{len(source_model.model) - 1}."
    target_detector_prefix = f"model.{len(target_model.model) - 1}."

    compatible_state = {
        key: value
        for key, value in source_state.items()
        if (
            key in target_state
            and target_state[key].shape == value.shape
            and not key.startswith(source_detector_prefix)
            and not key.startswith(target_detector_prefix)
        )
    }
    target_model.load_state_dict(compatible_state, strict=False)


def load_random(model_variant: str, num_classes: int) -> YOLO:
    """B1: build a YOLOv8 from architecture only, with random weights everywhere.

    Args:
        model_variant: e.g. ``"yolov8n"``.
        num_classes: Number of detection classes for the new head.
    """
    _validate_num_classes(num_classes)
    model = YOLO(_model_yaml_name(model_variant))
    return _replace_with_target_nc_model(model, num_classes)


def load_random_detector(weights_path: str | Path, num_classes: int) -> YOLO:
    """Load pretrained weights except for a fresh random Detect head.

    This preserves the checkpoint's backbone and neck tensors, then rebuilds
    the detector for ``num_classes`` so box/classification detector weights are
    randomly initialized.

    Args:
        weights_path: Path to a local ``.pt`` checkpoint or an Ultralytics
            model name (e.g. ``"yolov8n.pt"``) that the library will fetch.
        num_classes: Number of detection classes for the new head.
    """
    _validate_num_classes(num_classes)
    model = YOLO(str(weights_path))
    return _randomize_detector(model, num_classes)


def load_random_head(weights_path: str | Path, num_classes: int) -> YOLO:
    """Backward-compatible alias for detector-only randomization."""
    return load_random_detector(weights_path=weights_path, num_classes=num_classes)


def load_coco_full(weights_path: str | Path, num_classes: int) -> YOLO:
    """B3: full COCO-pretrained checkpoint, head re-shaped to num_classes.

    Args:
        weights_path: Path to a local ``.pt`` checkpoint or an Ultralytics
            model name (e.g. ``"yolov8n.pt"``) that the library will fetch.
        num_classes: Number of detection classes for the new head.
    """
    _validate_num_classes(num_classes)

    model = YOLO(str(weights_path))
    # Ultralytics rebuilds the Detect head (keeping backbone+neck weights, new
    # randomly-initialised classification branch) when model.train(data=...)
    # sees a dataset whose class count differs from the checkpoint's. Recording
    # the target nc on the underlying nn.Module lets callers introspect it.
    return _apply_class_metadata(model, num_classes)


def load_coco_partial_arch(
    model_variant: str,
    weights_path: str | Path,
    num_classes: int,
) -> YOLO:
    """Build an architecture YAML and copy compatible COCO checkpoint tensors.

    This is useful for YOLO architecture variants that do not have released
    pretrained weights, such as YOLO26 P2. The detector is intentionally left
    randomly initialized because the target architecture/head differs from the
    checkpoint.

    Args:
        model_variant: YAML architecture name, e.g. ``"yolo26n-p2"``.
        weights_path: Pretrained checkpoint to partially transfer from, e.g.
            ``"yolo26n.pt"``.
        num_classes: Number of detection classes for the new head.
    """
    _validate_num_classes(num_classes)
    target = YOLO(_model_yaml_name(model_variant))
    source = YOLO(str(weights_path))
    target = _replace_with_target_nc_model(target, num_classes)
    _load_compatible_non_detector_weights(target.model, source.model)
    return _apply_class_metadata(target, num_classes)


def load_ssl_benthicat(
    weights_path: str | Path,
    num_classes: int,
    model_variant: str = "yolov8n",
) -> YOLO:
    """B6: BYOL self-supervised backbone (BenthiCat SSS) loaded into a fresh detector.

    Loads only the backbone slice (strict=False leaves neck+head random);
    architecture-agnostic, so ``model_variant`` may be ``"yolov8n"`` or ``"yolo26n"``.

    Args:
        weights_path: Path to the ``byol_benthicat_<variant>.pt`` checkpoint.
        num_classes: Number of detection classes for the new head.
        model_variant: YAML architecture name; must match the checkpoint's variant.
    """
    _validate_num_classes(num_classes)
    model = YOLO(_model_yaml_name(model_variant))
    ckpt = torch.load(weights_path, map_location="cpu")
    _missing, unexpected = model.model.load_state_dict(
        ckpt["backbone_state_dict"], strict=False
    )
    if unexpected:  # unexpected key => variant/architecture mismatch, not partial load
        raise ValueError(f"unexpected keys when loading SSL backbone: {unexpected}")
    return _apply_class_metadata(model, num_classes)
