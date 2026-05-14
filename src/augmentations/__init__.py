"""Sonar-physics-informed Albumentations transforms.

All transforms are Albumentations-compatible so they compose with the generic
CV transforms in `configs/augmentation/generic_cv.yaml`.

- SpeckleNoise:   multiplicative noise model for SSS speckle (ImageOnlyTransform)
- RangeFalloff:   range-dependent intensity attenuation     (ImageOnlyTransform)
- AcousticShadow: shadow synthesis behind seafloor objects  (DualTransform —
                  may interact with bounding boxes)
"""

from src.augmentations.range_falloff import RangeFalloff
from src.augmentations.shadow import AcousticShadow
from src.augmentations.speckle import SpeckleNoise

__all__ = ["SpeckleNoise", "RangeFalloff", "AcousticShadow"]
