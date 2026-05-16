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
    return out


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


def _build_model(cfg_init: DictConfig):
    """Dispatch to the init loader named in cfg.init.loader.

    Only B3 (load_coco_full) is wired up for now; the other B1/B2/B4/B5
    loaders are still stubs in src.models.load_pretrained.
    """
    if cfg_init.loader == "load_coco_full":
        return load_pretrained.load_coco_full(
            weights_path=cfg_init.weights_path,
            num_classes=cfg_init.num_classes,
        )
    raise NotImplementedError(
        f"init.loader={cfg_init.loader!r} is not implemented yet; "
        f"only 'load_coco_full' is wired into train.py"
    )


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run a training cycle and return the metric dict.

    Args:
        cfg: Composed Hydra config (see `configs/config.yaml`).

    Returns:
        Dict of metric_name -> value. Must contain `cfg.optimized_metric`.
    """
    _seed_everything(cfg.seed)

    if cfg.logging.wandb.enabled:
        # Ultralytics 8.4+ ships a W&B callback but leaves it off by default;
        # flip the setting on so model.train() logs metrics to the active run.
        ultralytics_settings.update({"wandb": True})
        wandb.init(project=cfg.logging.wandb.project)

    if cfg.augmentation.name == "none":
        _disable_ultralytics_default_albumentations()

    # TODO: build Albumentations pipeline from cfg.augmentation.pipeline and
    #       hook it into the Ultralytics dataloader (custom-dataset path).
    # TODO: collect slice-level test metrics (per year, per class, by bbox size)
    #       via src.utils.eval.compute_map and push artifacts to W&B.

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

    # Train. Pass training cfg directly through to Ultralytics.
    train_kwargs = OmegaConf.to_container(cfg.training, resolve=True)
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
    metrics.update(_extract_box_metrics(test_results, prefix="test"))

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
