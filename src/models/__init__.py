"""Model factories for the init-ablation suite (B1–B5) and classical baseline."""

from src.models.classical_baseline import HOGClassifier
from src.models.load_pretrained import (
    load_coco_full,
    load_imagenet_backbone,
    load_random,
    load_sonar_fls,
    load_sonar_uatd,
    load_ssl_benthicat,
)

__all__ = [
    "load_random",
    "load_imagenet_backbone",
    "load_coco_full",
    "load_sonar_fls",
    "load_sonar_uatd",
    "load_ssl_benthicat",
    "HOGClassifier",
]
