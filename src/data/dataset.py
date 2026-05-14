"""Santos 2024 SSS dataset wrapper.

Ultralytics' YOLO trainer consumes a `data.yaml` file pointing at folders of
images + label files (YOLO format). This module owns the in-memory
representation of the dataset and the on-disk YAML emission step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


class SantosDataset:
    """In-memory view of the Santos SSS dataset.

    Holds image paths, label paths, optional per-image metadata (year,
    sonar parameters), and the active split assignment. The Ultralytics
    trainer is fed a YAML pointing at a flat directory structure that this
    class materializes via `to_ultralytics_yaml`.
    """

    def __init__(
        self,
        root: Path | str,
        images_dir: str,
        labels_dir: str,
        class_names: Sequence[str],
        split_indices: dict[str, Sequence[int]] | None = None,
    ) -> None:
        """Index the dataset on disk and store split assignments.

        Args:
            root: Path to the dataset root (e.g. ``data/santos``).
            images_dir: Subdirectory containing image files.
            labels_dir: Subdirectory containing YOLO-format label files.
            class_names: Ordered class list; indices must match label files.
            split_indices: Optional mapping of split name -> indices into the
                indexed image list. If None, the whole dataset is one split.
        """
        # TODO: scan root/images_dir, build the (image, label) pair list
        # TODO: validate that every image has a matching label file (or no objects)
        # TODO: store split_indices for later YAML emission
        raise NotImplementedError("TODO: implement SantosDataset.__init__")

    def __len__(self) -> int:
        """Number of (image, label) pairs in the dataset."""
        raise NotImplementedError("TODO: implement __len__")

    def to_ultralytics_yaml(self, out_dir: Path | str) -> Path:
        """Write an Ultralytics-style dataset YAML and return its path.

        The YAML will reference `train`, `val`, and `test` image-list files
        written into ``out_dir``. The class list, paths, and names are filled
        from this dataset's state.
        """
        # TODO: write {train,val,test}.txt image lists into out_dir
        # TODO: write data.yaml referencing them; return the YAML path
        raise NotImplementedError("TODO: implement to_ultralytics_yaml")
