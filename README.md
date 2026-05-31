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
| A6 | generic CV + full sonar-physics stack |
| A7 | generic CV + speckle + range falloff, no shadow |

**Initialization ablation:**

| ID | Init |
| --- | --- |
| B1 | random |
| B2 | COCO backbone/neck with random detector weights |
| B3 | full YOLOv8 COCO checkpoint |
| B4 | Valdenegro-Toro forward-look-sonar pretrained |
| B5 | our UATD-pretrained checkpoint |
| B6 | self-supervised BYOL pretraining on BenthiCat SSS |

## Running

Single experiment:

```bash
python -m src.train experiment=a3 seed=0
```

Multirun sweep (all augmentation ablations, three seeds each):

```bash
python -m src.train -m experiment=a0,a1,a2,a3,a4,a5,a6,a7 seed=0,1,2
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

Full augmentation-ablation sweep — all A-series experiments, three seeds, both random and stratified-random splits:

```bash
python -m src.train -m experiment=a0,a1,a2,a3,a4,a5,a6,a7 data=santos,santos_stratified seed=0,1,2 training.epochs=100 training.batch=16 training.imgsz=800 training.device=cuda training.workers=8
```

### Conservative Augmentation Sweep

For configs that do not use generic CV (`A0`, `A2`, `A3`, `A4`, `A5`), the conservative physics defaults already live in the YAML files. This sweep uses only the stratified split:

```bash
python -m src.train -m experiment=a0,a2,a3,a4,a5 data=santos_stratified seed=0,1,2 training.epochs=100 training.batch=16 training.imgsz=800 training.device=cuda training.workers=5 logging.wandb.project="sonar conservative aug"
```

For configs that use generic CV (`A1`, `A6`, `A7`), this keeps flips enabled but reduces the stronger geometry/mosaic settings, also using only the stratified split:

```bash
python -m src.train -m experiment=a1,a6,a7 data=santos_stratified seed=0,1,2 augmentation.train_args.mosaic=0.5 augmentation.train_args.translate=0.05 augmentation.train_args.scale=0.1 training.epochs=100 training.batch=16 training.imgsz=800 training.device=cuda training.workers=5 logging.wandb.project="sonar conservative aug"
```

### Generic CV Strength Sweep

Focused sweep on the `A6` all-augmentation condition over mosaic probability, the generic CV knob most likely to affect tiny sonar targets. This is `9` runs (`3 seeds * 3 mosaic values`) and keeps the other generic CV and sonar-physics augmentations fixed.

```bash
python -m src.train -m experiment=a6 data=santos_stratified seed=0,1,2 augmentation.train_args.mosaic=0.25,0.5,1.0 training.epochs=600 training.batch=16 training.imgsz=800 training.device=cuda training.workers=5 logging.wandb.project="sonar generic cv sweep"
```

### Training-Hyperparameter Sweep

Focused training sweep on the `A6` all-augmentation condition (`generic CV + speckle + range falloff + shadow`) over optimizer, learning rate, and cosine LR scheduling. This is `36` runs (`3 seeds * 2 optimizers * 3 LRs * 2 cos_lr settings`) and keeps regularization fixed at the current defaults.

```bash
python -m src.train -m experiment=a6 data=santos_stratified seed=0,1,2 training.optimizer=MuSGD,AdamW training.lr0=0.0001,0.0005,0.001 training.cos_lr=true,false training.epochs=600 training.batch=16 training.imgsz=800 training.device=cuda training.workers=5 logging.wandb.project="sonar training sweep real"
```

### Random vs Init Weights

This compares initialization/model variants under the strongest augmentation condition: `A6` uses the YOLOv8 COCO baseline, `B1` uses random weights everywhere, `B2` uses COCO backbone/neck weights with a random detector, `B3` uses YOLO26 COCO weights, and `B4` uses YOLO26 P2 with compatible COCO weights.

The run keeps the training setup aligned with the sweep above (`epochs=1000`, `batch=16`, `imgsz=800`, `device=cuda`) and uses 5 dataloader workers.

```bash
python -m src.train -m experiment=a6,b1,b2,b3,b4 data=santos_stratified seed=0,1,2 training.epochs=1000 training.batch=16 training.imgsz=800 training.device=cuda training.workers=5 logging.wandb.project="sonar random vs init weights"
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
