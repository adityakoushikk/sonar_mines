"""Santos 2024 SSS dataset wrapper.

Ultralytics' YOLO trainer consumes a `data.yaml` file pointing at folders of
images + label files (YOLO format). This module owns the in-memory
representation of the dataset and the on-disk YAML emission step.

Reference: https://docs.ultralytics.com/datasets/detect
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

_SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _compute_stratum(label_path: Path, class_names: Sequence[str]) -> str:
    """Map a YOLO label file to a stratification bucket.

    Returns ``"unlabeled"`` for empty files, ``"<name>_only"`` for single-class
    images, and ``"+".join(sorted(...))`` for multi-class images. Used by
    `random_split` to keep per-class image counts proportional across splits.
    """
    classes_present: set[int] = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            classes_present.add(int(line.split()[0]))
    if not classes_present:
        return "unlabeled"
    if len(classes_present) == 1:
        return f"{class_names[next(iter(classes_present))].lower()}_only"
    return "+".join(sorted(class_names[c].lower() for c in classes_present))


class SantosDataset:
    """In-memory view of the Santos SSS dataset.

    Indexes (image, label) pairs on disk, parses the collection year from each
    filename's ``_YYYY`` suffix, and emits an Ultralytics-style dataset YAML
    plus per-split image-list files via :meth:`to_ultralytics_yaml`.
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
                indexed image list. If None, the whole dataset is exposed as a
                single ``"all"`` split.
        """
        self.root = Path(root).resolve()
        self.images_dir = self.root / images_dir
        self.labels_dir = self.root / labels_dir
        self.class_names = list(class_names)

        image_paths: list[Path] = []
        for ext in _SUPPORTED_IMAGE_EXTS:
            image_paths.extend(self.images_dir.glob(f"*{ext}"))
        self.image_paths: list[Path] = sorted(image_paths)
        if not self.image_paths:
            raise FileNotFoundError(f"no images found under {self.images_dir}")

        self.label_paths: list[Path] = [
            self.labels_dir / f"{p.stem}.txt" for p in self.image_paths
        ]
        missing = [p for p in self.label_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no label file "
                f"(e.g. {missing[0].name}). YOLO expects an empty .txt for "
                f"images with no annotated objects."
            )

        # Year parsed from the "<stem>_<YYYY>.<ext>" filename convention so
        # cross_year_split can bucket items without consulting an external map.
        self.years: list[int] = [
            int(p.stem.rsplit("_", 1)[-1]) for p in self.image_paths
        ]

        # Per-image stratum (class-presence bucket) consumed by stratified
        # `random_split`. Computed once at index time; label files are tiny.
        self.strata: list[str] = [
            _compute_stratum(p, self.class_names) for p in self.label_paths
        ]

        self.split_indices: dict[str, list[int]] = (
            {k: list(v) for k, v in split_indices.items()}
            if split_indices is not None
            else {"all": list(range(len(self.image_paths)))}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def to_ultralytics_yaml(self, out_dir: Path | str) -> Path:
        """Write an Ultralytics-style dataset YAML and return its path.

        For each split in ``self.split_indices`` writes a ``<split>.txt`` file
        of absolute image paths into ``out_dir``, then writes ``data.yaml``
        referencing them. ``names`` is emitted as a ``{int: str}`` dict per the
        Ultralytics spec; ``nc`` is omitted (the trainer derives it).
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        split_files: dict[str, Path] = {}
        for split_name, idxs in self.split_indices.items():
            txt_path = out_dir / f"{split_name}.txt"
            txt_path.write_text(
                "\n".join(str(self.image_paths[i]) for i in idxs) + "\n"
            )
            split_files[split_name] = txt_path

        data: dict = {
            "path": str(self.root),
            "names": {i: name for i, name in enumerate(self.class_names)},
        }
        for split_name in ("train", "val", "test"):
            if split_name in split_files:
                data[split_name] = str(split_files[split_name])

        yaml_path = out_dir / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
        return yaml_path
