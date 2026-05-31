# pretraining — B6 self-supervised pretraining

Self-supervised **BYOL** pretraining of a YOLOv8 backbone on the unlabeled
**BenthiCat** side-scan-sonar dataset, producing the **B6** initialization for the
Santos mine-detection fine-tune (see the [root README](../README.md) init
ablation). The hypothesis: ~950K in-distribution SSS tiles teach sonar texture
priors (speckle, range falloff, seafloor structure) that transfer better than
COCO init at the small Santos label budget.

This package is **self-contained** — it carries its own
[`requirements.txt`](requirements.txt) so the heavy SSL/cloud deps stay out of the
repo-root environment. The only interface back into the main repo is one loader
(`load_ssl_benthicat`) plus two configs (`configs/init/ssl_benthicat.yaml`,
`configs/experiment/b6.yaml`).

> Compute runs on **Modal** (serverless GPU). The loop is plain `accelerate` — no
> PyTorch Lightning. Default is a single A100; multi-GPU is opt-in (see below).

## Layout

| Path | What lives here |
| --- | --- |
| [`preprocess.py`](preprocess.py) | `.npy` tiles → uint8 PNG → WebDataset shards (CLI + importable) |
| [`ssl/transforms.py`](ssl/transforms.py) | Two-view BYOL augmentation: speckle + range-falloff (**no** shadow — it is the downstream signal) |
| [`ssl/backbone.py`](ssl/backbone.py) | Extract/checkpoint the YOLOv8 backbone (layers 0–9, ending at SPPF) |
| [`ssl/data.py`](ssl/data.py) | WebDataset → DataLoader of view pairs |
| [`ssl/byol.py`](ssl/byol.py) | BYOL model + `accelerate` training loop (`train_byol`) |
| [`modal_app.py`](modal_app.py) | Modal image, Volume, and the `preprocess_volume` / `train` functions |
| [`tests/`](tests/) | Backbone round-trip + synthetic end-to-end smoke (CPU, no real data) |

## Quick check (no data, no Modal)

The synthetic end-to-end smoke proves the whole pipeline before you spend any
bandwidth or money — it generates random tiles, packs shards, runs one CPU BYOL
epoch, and reloads the checkpoint through `load_ssl_benthicat`:

```bash
pip install -r pretraining/requirements.txt   # once, into your project env
pytest pretraining/
```

## Full workflow

The recommended path runs **entirely on Modal** — download, extract, and
preprocessing all happen server-side, so nothing large ever touches your machine
(the extracted `.npy` is ~560 GB, so don't extract it locally). Your laptop just
issues `modal run` commands.

### 0. One-time setup

```bash
modal token new                                # Modal auth
modal secret create wandb WANDB_API_KEY=<key>  # W&B key for run logging
```

### 1. Get the archives onto the Volume (~59 GB)

"BenthiCat – SSS Pre-training", Harvard Dataverse DOI
[`10.7910/DVN/2VCN7Y`](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2VCN7Y)
— public (CC BY-NC-SA 4.0), 16 sectors (N01–N10, N12, N15, N16, S01–S03) shipped
as **multi-volume 7-Zip splits** (`N04.7z.001`, `N04.7z.002`, …; 32 files total).

**Recommended — pull straight from Dataverse to the Volume (nothing local):**

```bash
modal run pretraining/modal_app.py::download_to_volume
```

Streams every file to `/data/raw` and commits per file, so a re-run resumes.

**Alternative — download locally, then upload:** grab the `.7z.*` files into a
folder (e.g. `~/benthicat_7z/`) and push them:

```bash
modal volume put sonar-ssl-data ~/benthicat_7z /raw
```

### 2. Preprocess on Modal (extract + pack, server-side)

```bash
modal run pretraining/modal_app.py::preprocess_volume
modal run pretraining/modal_app.py::inspect_volume     # verify shards/benthicat-*.tar
```

`preprocess_volume` walks the archives one at a time (extract → pack → delete →
commit), so transient Volume disk stays ~one sector rather than the full ~560 GB.
Shards land at `/data/shards/benthicat-*.tar`.

> **Prefer to preprocess locally?** Pack shards on your machine and upload those
> instead — fine if you have the disk:
> ```bash
> python -m pretraining.preprocess --input data/benthicat --output data/benthicat_shards
> modal volume put sonar-ssl-data data/benthicat_shards /shards
> ```

### 3. Train

A cheap calibration pass first (measures real throughput for ~$1), then the full
run. Loss logs to the `sonar-ssl` W&B project.

```bash
modal run pretraining/modal_app.py --epochs 1 --epoch-length 5000 --batch-size 64   # smoke
modal run pretraining/modal_app.py --epochs 100                                      # full run
```

**Choosing the architecture (v8 or v26).** Default is `yolov8n`; pass
`--variant yolo26n` to pretrain a YOLO26 backbone instead (the backbone cut is
inferred from the variant's yaml, so nothing else changes). The checkpoint name
encodes the variant, so the two never collide:

```bash
modal run pretraining/modal_app.py --epochs 100 --variant yolo26n   # -> byol_benthicat_yolo26n.pt
```

### 4. Pull the checkpoint

```bash
modal volume get sonar-ssl-data /checkpoints/byol_benthicat_yolov8n.pt \
  pretraining/checkpoints/byol_benthicat_yolov8n.pt
# yolo26n run instead: swap both names to byol_benthicat_yolo26n.pt
```

These are the paths baked into `configs/init/ssl_benthicat.yaml` (v8) and
`configs/init/ssl_benthicat_yolo26.yaml` (v26).

### 5. Fine-tune B6 on Santos

Pick the init that matches the backbone you pretrained — `experiment=b6` (YOLOv8n,
compare against A6/B1/B2) or `experiment=b6_yolo26` (YOLO26, compare against
B3/B4):

```bash
python -m src.train -m experiment=a6,b6 seed=0,1,2 \
  training.epochs=100 training.batch=16 training.imgsz=800 training.device=cuda
# YOLO26: python -m src.train -m experiment=b3,b6_yolo26 seed=0,1,2 ...
```

## Cost & time

The backbone is tiny, so the run is **data-bound** (CPU augmentation, not GPU) —
a single A100 is the right default; more GPUs cost the same total and only buy
wall-clock.

| Epochs | Wall time (1×A100) | ~Cost |
| --- | --- | --- |
| 50 | ~5 h | ~$15 |
| 100 | ~10 h | ~$26 |

Preprocessing adds ~$1–2 (CPU-only). Figures assume ~2,500 tiles/s with ~8
dataloader workers; the step-4 smoke measures the real rate. The `train` timeout
is 12 h, so a 100-epoch single-A100 run fits.

## Multi-GPU (opt-in, not yet wired)

`gpu="A100:N"` alone would **not** speed things up: the Modal function runs as a
single process, so `accelerate` would use one GPU and bill for N. Real scaling
needs multi-process launch (`notebook_launcher`) plus a fixed `--epoch-length`
(≈ tiles/rank) so DDP doesn't hang on uneven shards — `train_byol` fails fast if
you pass multiple processes without it. Ask before relying on this; validate with
a 1-epoch multi-GPU smoke first.
