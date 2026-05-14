# sonar-mines

YOLOv8 fine-tuning for **side-scan sonar mine detection** on the Santos et al. 2024 dataset (Figshare DOI: 10.6084/m9.figshare.24574879).

YOLOv8 was chosen because its CNN detection head is small enough to fine-tune from a few hundred annotated SSS images — transformer detectors over-fit at this label budget.

Configuration is managed by **Hydra** (run/multirun) and experiment tracking is on **Weights & Biases**. This project does **not** use PyTorch Lightning — Ultralytics has its own training loop.

> Status: scaffolding only. Every Python function raises `NotImplementedError` and every config value is a `TODO` placeholder. Fill in incrementally.

## Layout

| Path | What lives here |
| --- | --- |
| [`configs/`](configs/) | Hydra config groups: `data/`, `init/`, `augmentation/`, `experiment/`, plus `training.yaml`, `logging.yaml`, and the top-level `config.yaml` |
| [`src/train.py`](src/train.py) | Hydra entrypoint |
| [`src/data/`](src/data/) | Santos dataset wrapper + train/val/test splits |
| [`src/augmentations/`](src/augmentations/) | Sonar-physics-informed Albumentations transforms (speckle, range-falloff, shadow) |
| [`src/models/`](src/models/) | Backbone loaders for the **B1–B5** init ablation + HOG+SVM classical baseline |
| [`src/utils/`](src/utils/) | mAP helpers, W&B logging, bbox & augmentation visualisation |
| [`data/santos/`](data/santos/) | Placeholder for the Santos 2024 dataset |
| [`data/uatd/`](data/uatd/) | Placeholder for UATD (used to pretrain the **B5** init) |
| [`notebooks/`](notebooks/) | Data exploration, results analysis, error analysis |
| [`outputs/`](outputs/) | Hydra run directories (gitignored except `.gitkeep`) |

## Experiments

**Augmentation ablation (init held at COCO):**

| ID | Augmentation |
| --- | --- |
| A0 | none |
| A1 | generic CV (flips, rotations, brightness/contrast) |
| A2 | speckle noise |
| A3 | range-dependent intensity falloff |
| A4 | acoustic shadow |
| A5 | full sonar-physics stack (A2 + A3 + A4) |

**Initialization ablation:**

| ID | Init |
| --- | --- |
| B1 | random |
| B2 | ImageNet backbone weights only |
| B3 | full YOLOv8 COCO checkpoint |
| B4 | Valdenegro-Toro forward-look-sonar pretrained |
| B5 | our UATD-pretrained checkpoint |

## Running

Single experiment:

```bash
python -m src.train experiment=a3 seed=0
```

Multirun sweep (all augmentation ablations, three seeds each):

```bash
python -m src.train -m experiment=a0,a1,a2,a3,a4,a5 seed=0,1,2
```

## Install

Requires Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume the venv is active and you're running from the project root. The `python -m src.foo` invocation puts the repo root on `sys.path`, so `from src.x import y` resolves without any install step.

For notebooks in [`notebooks/`](notebooks/) to import from `src`, either launch Jupyter from the repo root (`jupyter lab`) or add `sys.path.insert(0, "..")` at the top of the notebook.
