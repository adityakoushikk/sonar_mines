"""Hydra entrypoint for training a YOLOv8 model on the Santos SSS dataset.

Current scope: just stand up the base COCO-init YOLO training run. Everything
else (Albumentations pipeline, custom test-split eval, W&B init + artifact
push, full B1–B5 loader dispatch) is left as TODO and will be filled in once
the base run is green.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ultralytics import settings as ultralytics_settings

from src.data import (
    SantosDataset,
    cross_year_split,
    random_split,
    stratified_random_split,
)
from src.models import load_pretrained

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("PROJECT_ROOT") is None:
    os.environ["PROJECT_ROOT"] = str(_PROJECT_ROOT)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_splits(cfg: DictConfig, ds: SantosDataset) -> dict[str, list[int]]:
    split = cfg.data.split
    if split.kind == "random":
        return random_split(
            n_items=len(ds),
            train_frac=split.train_frac,
            val_frac=split.val_frac,
            test_frac=split.test_frac,
            seed=split.seed,
        )
    if split.kind == "stratified_random":
        return stratified_random_split(
            strata=ds.strata,
            train_frac=split.train_frac,
            val_frac=split.val_frac,
            test_frac=split.test_frac,
            seed=split.seed,
        )
    if split.kind == "cross_year":
        return cross_year_split(
            years_per_item=ds.years,
            train_years=list(split.train_years),
            test_years=list(split.test_years),
            val_frac_of_train=split.val_frac_of_train,
            seed=split.seed,
        )
    raise ValueError(f"unknown split kind: {split.kind}")



def _extract_box_metrics(results: Any, prefix: str) -> dict[str, float]:
    """Pull mAP50 / mAP50-95 off an Ultralytics results object.

    Defensive against version drift in Ultralytics' results schema — missing
    attributes silently produce a smaller dict instead of crashing.
    """
    box = getattr(results, "box", None)
    out: dict[str, float] = {}
    if box is not None:
        if hasattr(box, "map"):
            out[f"{prefix}/mAP50-95"] = float(box.map)
        if hasattr(box, "map50"):
            out[f"{prefix}/mAP50"] = float(box.map50)
        if hasattr(box, "map75"):
            out[f"{prefix}/mAP75"] = float(box.map75)

        names = getattr(results, "names", {})
        ap_class_index = getattr(box, "ap_class_index", [])
        ap = getattr(box, "ap", [])
        ap50 = getattr(box, "ap50", [])
        per_class_ap: dict[str, float] = {}
        per_class_ap50: dict[str, float] = {}
        per_class_count: dict[str, int] = {}
        ap_position = 0
        for class_id in ap_class_index:
            class_name = names.get(int(class_id), str(class_id))
            if class_name not in per_class_count:
                per_class_count[class_name] = 0
                per_class_ap[class_name] = 0.0
                per_class_ap50[class_name] = 0.0

            per_class_count[class_name] += 1
            if ap_position < len(ap):
                per_class_ap[class_name] += float(ap[ap_position])
            if ap_position < len(ap50):
                per_class_ap50[class_name] += float(ap50[ap_position])
            ap_position += 1

        for class_name, ap_sum in per_class_ap.items():
            out[f"{prefix}/ap/{class_name}"] = (
                ap_sum / per_class_count[class_name]
            )
        for class_name, ap50_sum in per_class_ap50.items():
            out[f"{prefix}/ap50/{class_name}"] = (
                ap50_sum / per_class_count[class_name]
            )
    return out


def _log_metrics_to_wandb(
    metrics: dict[str, Any],
    cfg: DictConfig,
    run_id: str | None,
) -> None:
    """Log project-owned metrics, resuming the active W&B run if needed."""
    if not cfg.logging.wandb.enabled or not metrics:
        return

    if wandb.run is None:
        wandb.init(
            project=cfg.logging.wandb.project,
            id=run_id,
            resume="allow" if run_id is not None else None,
        )
    wandb.log(metrics)
    wandb.run.summary.update(metrics)


def _wandb_hyperparameters(cfg: DictConfig, model: torch.nn.Module | None = None) -> dict[str, Any]:
    """Build a W&B config dict that mirrors the resolved Hydra config."""
    hparams = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(hparams, dict):
        raise TypeError(f"expected resolved cfg to be a dict, got {type(hparams).__name__}")

    if model is not None:
        hparams["model"] = {
            "params": {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        }

    return hparams


def _disable_ultralytics_default_albumentations() -> None:
    """Replace Ultralytics' built-in Albumentations pipeline with a no-op.

    Ultralytics 8.4 unconditionally applies Blur / MedianBlur / ToGray / CLAHE
    at p=0.01 inside its dataloader. Setting `transform = None` short-circuits
    the class's __call__, giving the A0 condition a literal zero-augmentation
    floor. Verified against ultralytics==8.4.50.
    """
    from ultralytics.data import augment as ul_augment

    def _noop_init(self, p: float = 1.0, transforms=None) -> None:
        del transforms  # accepted for signature compatibility, intentionally unused
        self.p = p
        self.transform = None
        self.contains_spatial = False

    ul_augment.Albumentations.__init__ = _noop_init


def _transform_needs_bboxes(transform: Any) -> bool:
    """Return True when a custom Albumentations transform needs bbox data."""
    try:
        targets_as_params = transform.targets_as_params
    except Exception:
        return False
    return "bboxes" in targets_as_params


def _patch_ultralytics_albumentations_for_bbox_transforms() -> None:
    """Teach Ultralytics to pass bboxes to custom bbox-dependent transforms.

    Ultralytics decides whether to pass bboxes by checking transform class names
    against its own hardcoded list of spatial Albumentations transforms. Custom
    transforms like AcousticShadow are invisible to that check, so we rebuild
    the composed Albumentations transform with bbox_params when any custom
    transform declares that it needs ``bboxes`` in ``targets_as_params``.
    """
    from ultralytics.data import augment as ul_augment

    if getattr(ul_augment.Albumentations, "_sonar_bbox_patch", False):
        return

    original_init = ul_augment.Albumentations.__init__

    def _patched_init(self, p: float = 1.0, transforms=None) -> None:
        original_init(self, p=p, transforms=transforms)

        if transforms is None or not any(_transform_needs_bboxes(t) for t in transforms):
            return

        try:
            import albumentations as A
        except ImportError:
            return

        self.contains_spatial = True
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(torch.initial_seed())

    ul_augment.Albumentations.__init__ = _patched_init
    ul_augment.Albumentations._sonar_bbox_patch = True


def _build_model(cfg_init: DictConfig):
    """Dispatch to the init loader named in cfg.init.loader.

    B3 (load_coco_full) and B6 (load_ssl_benthicat) are wired up; the other
    B1/B2/B4/B5 loaders are still stubs in src.models.load_pretrained.
    """
    if cfg_init.loader == "load_coco_full":
        return load_pretrained.load_coco_full(
            weights_path=cfg_init.weights_path,
            num_classes=cfg_init.num_classes,
        )
    elif cfg_init.loader == "load_ssl_benthicat":
        return load_pretrained.load_ssl_benthicat(
            weights_path=cfg_init.weights_path,
            num_classes=cfg_init.num_classes,
            model_variant=cfg_init.get("model_variant", "yolov8n"),
        )
    raise NotImplementedError(
        f"init.loader={cfg_init.loader!r} is not implemented yet; "
        f"only 'load_coco_full' and 'load_ssl_benthicat' are wired into train.py"
    )


def _build_albumentations_pipeline(cfg_augmentation: DictConfig) -> list[Any] | None:
    """Instantiate custom Albumentations transforms from augmentation.pipeline."""
    pipeline_cfg = cfg_augmentation.get("pipeline")
    if not pipeline_cfg:
        return None
    return [hydra.utils.instantiate(transform_cfg) for transform_cfg in pipeline_cfg]


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run a training cycle and return the metric dict.

    Args:
        cfg: Composed Hydra config (see `configs/config.yaml`).

    Returns:
        Dict of metric_name -> value. Must contain `cfg.optimized_metric`.
    """
    _seed_everything(cfg.seed)

    wandb_run_id: str | None = None
    if cfg.logging.wandb.enabled:
        # Ultralytics 8.4+ ships a W&B callback but leaves it off by default;
        # flip the setting on so model.train() logs metrics to the active run.
        ultralytics_settings.update({"wandb": True})
        run = wandb.init(
            project=cfg.logging.wandb.project,
            config=_wandb_hyperparameters(cfg),
            tags=list(cfg.tags)
        )
        wandb_run_id = run.id
    else:
        ultralytics_settings.update({"wandb": False})

    if cfg.augmentation.name == "none":
        _disable_ultralytics_default_albumentations()

    ds = SantosDataset(
        root=cfg.data.root,
        images_dir=cfg.data.images_dir,
        labels_dir=cfg.data.labels_dir,
        class_names=list(cfg.data.class_names),
    )
    ds.split_indices = _build_splits(cfg, ds)

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    data_yaml = ds.to_ultralytics_yaml(run_dir / "data")

    # Model.
    model = _build_model(cfg.init)
    if wandb.run is not None:
        wandb.config.update(_wandb_hyperparameters(cfg, model), allow_val_change=True)

    # Train. Pass training cfg directly through to Ultralytics.
    train_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
    augmentation_train_args_cfg = cfg.augmentation.get("train_args")
    augmentation_train_args = (
        OmegaConf.to_container(augmentation_train_args_cfg, resolve=True)
        if augmentation_train_args_cfg is not None
        else {}
    )
    if not isinstance(augmentation_train_args, dict):
        raise TypeError(
            "cfg.augmentation.train_args must be a mapping of Ultralytics "
            f"train() keyword arguments, got {type(augmentation_train_args).__name__}"
        )
    train_kwargs.update(augmentation_train_args)
    albumentations_pipeline = _build_albumentations_pipeline(cfg.augmentation)
    if albumentations_pipeline is not None:
        if any(_transform_needs_bboxes(t) for t in albumentations_pipeline):
            _patch_ultralytics_albumentations_for_bbox_transforms()
        train_kwargs["augmentations"] = albumentations_pipeline

    results = model.train(
        data=str(data_yaml),
        project=str(run_dir),
        name="ultralytics",
        **train_kwargs,
    )

    # Ultralytics returns a results object with .box.map (mAP50-95) and
    # .box.map50 (mAP50). The train() call exposes val-split metrics; a
    # separate model.val(split="test") run gives the held-out test mAP.
    test_results = model.val(
        data=str(data_yaml),
        split="test",
        project=str(run_dir),
        name="ultralytics_test",
    )

    metrics: dict[str, Any] = {}
    metrics.update(_extract_box_metrics(results, prefix="metrics"))
    test_metrics = _extract_box_metrics(test_results, prefix="test")
    metrics.update(test_metrics)
    _log_metrics_to_wandb(test_metrics, cfg, wandb_run_id)

    # Close the W&B run explicitly so the next Hydra multirun job starts a
    # fresh one (newer wandb versions return the previous active run from
    # wandb.init() unless this is called first).
    if wandb.run is not None:
        wandb.finish()

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Hydra entrypoint. Composes the config and dispatches to `train`."""
    print(OmegaConf.to_yaml(cfg))
    metrics = train(cfg)
    return metrics.get(cfg.optimized_metric)


if __name__ == "__main__":
    main()
