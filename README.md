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

## Examples

Quick CPU smoke test before committing GPU time — runs the A0 (COCO init, no augmentation) experiment for 3 epochs across 3 seeds at a tiny image size, with metrics tracked in a separate `sonar-test` W&B project so the real `sonar` project stays clean:

```bash
python -m src.train -m experiment=a0 seed=0,1,2 \
  training.epochs=3 \
  training.device=cpu \
  training.batch=2 \
  training.imgsz=320 \
  training.workers=0 \
  logging.wandb.project=sonar-test
```

Every value here is a Hydra override, so the same pattern works for any (config-group, seed, hyperparameter) combination.

A0 baseline run — full GPU training, three seeds, default hyperparameters (50 epochs, batch 16, imgsz 800), logged to the main `sonar` W&B project. This is the "true" baseline that every augmentation/init ablation is compared against:

```bash
python -m src.train -m experiment=a0 seed=0,1,2 \
  training.epochs=50 \
  training.batch=16 \
  training.imgsz=800 \
  training.device=cuda \
  training.workers=8
```

Note on A0: Ultralytics 8.4's YOLO dataloader unconditionally bakes in four Albumentations transforms (`Blur`, `MedianBlur`, `ToGray`, `CLAHE` at `p=0.01` each) plus a few others at `p=0.0`. To keep A0 a true zero-augmentation baseline, `src/train.py` monkey-patches `ultralytics.data.augment.Albumentations.__init__` to leave `self.transform = None` whenever `cfg.augmentation.name == "none"`, which short-circuits the class's `__call__`. No batches get the Ultralytics defaults under A0. Verified against `ultralytics==8.4.50` — if you bump versions, re-check that the class API hasn't drifted.

## Install

Requires Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume the venv is active and you're running from the project root. The `python -m src.foo` invocation puts the repo root on `sys.path`, so `from src.x import y` resolves without any install step.

For notebooks in [`notebooks/`](notebooks/) to import from `src`, either launch Jupyter from the repo root (`jupyter lab`) or add `sys.path.insert(0, "..")` at the top of the notebook.
