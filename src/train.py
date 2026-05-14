"""Hydra entrypoint for training a YOLOv8 model on the Santos SSS dataset.

Intended flow inside `train(cfg)`:
    1. Seed everything from cfg.seed.
    2. Initialize the W&B run from cfg.logging.wandb (if enabled).
    3. Build the train/val/test split via src.data.splits + src.data.dataset,
       writing an Ultralytics-style dataset YAML to the run output dir.
    4. Build the Albumentations pipeline from cfg.augmentation.pipeline and
       wire it into the Ultralytics dataloader (custom-dataset hook).
    5. Instantiate the YOLO model via the loader named in cfg.init.loader
       (one of src.models.load_pretrained.load_{random,imagenet_backbone,
       coco_full,sonar_fls,sonar_uatd}).
    6. Call model.train(...) with cfg.training kwargs.
    7. Call model.val(...) on the held-out test split; collect mAP metrics.
    8. Push metrics + artifacts to W&B and return the optimized metric.

The Hydra main() returns the optimized metric so multirun sweeps
(`-m experiment=a0,a1,...`) can rank configs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("PROJECT_ROOT") is None:
    os.environ["PROJECT_ROOT"] = str(_PROJECT_ROOT)


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run a full training + evaluation cycle and return the metric dict.

    Args:
        cfg: Composed Hydra config (see `configs/config.yaml`).

    Returns:
        Dict of metric_name -> value. Must contain `cfg.optimized_metric`.
    """
    # TODO: seed everything (numpy, torch, random) from cfg.seed
    # TODO: init W&B run from cfg.logging.wandb if enabled
    # TODO: build splits + write Ultralytics dataset YAML (src.data)
    # TODO: build Albumentations pipeline from cfg.augmentation.pipeline
    # TODO: load YOLO model via cfg.init.loader -> src.models.load_pretrained
    # TODO: call model.train(**cfg.training) with the augmentation hook
    # TODO: evaluate on the held-out split and collect metrics
    # TODO: log metrics + artifacts to W&B
    # TODO: return metric dict
    raise NotImplementedError("TODO: implement training entrypoint")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Hydra entrypoint. Composes the config and dispatches to `train`.

    Returns the value of `cfg.optimized_metric` so Hydra multirun
    sweeps can rank configurations.
    """
    print(OmegaConf.to_yaml(cfg))
    # TODO: call train(cfg) and return metric_dict[cfg.optimized_metric]
    raise NotImplementedError("TODO: wire main() to train()")


if __name__ == "__main__":
    main()
