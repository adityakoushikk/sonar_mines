"""sonar-mines: YOLOv8 fine-tuning for side-scan sonar mine detection.

This package contains the training entrypoint and supporting modules:
- src.data         dataset wrappers and train/val/test splitting
- src.augmentations sonar-physics-informed Albumentations transforms
- src.models       backbone loaders for the B1-B5 init ablations + classical baselines
- src.utils        evaluation and visualization helpers
"""
