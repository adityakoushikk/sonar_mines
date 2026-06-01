"""Modal app for running BYOL self-supervised pretraining (B6) on GPUs.

Packages ``src/`` + ``pretraining/`` into a Modal image, mounts a persistent
Volume at ``/data`` for shards + checkpoints, and runs ``train_byol`` on an A100.
All Modal/cluster plumbing lives here; the trainer itself has no Modal dependency.

    modal run pretraining/modal_app.py                      # launch training
    modal run pretraining/modal_app.py::inspect_volume      # list the Volume
"""
from __future__ import annotations

import modal

# Build ordering matters: Modal forbids build steps (apt/pip/run) AFTER
# add_local_*, so all package layers go on `_base` and `_add_local()` is last.
_base = (
    modal.Image.debian_slim()
    # ultralytics imports cv2, which needs libGL.so.1 + libgthread — absent in slim.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("pretraining/requirements.txt")
    # /root on sys.path so the mounted src/ and pretraining/ dirs are importable.
    .env({"PYTHONPATH": "/root"})
)


def _add_local(img: "modal.Image") -> "modal.Image":
    """Mount src/ and pretraining/ onto the import path (MUST be the last build step)."""
    return (
        img
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("pretraining", remote_path="/root/pretraining")
    )


image = _add_local(_base)
# Preprocess needs p7zip (the `7za` extractor); apt layer goes BEFORE the mount.
preprocess_image = _add_local(_base.apt_install("p7zip-full"))

app = modal.App("sonar-ssl", image=image)

vol = modal.Volume.from_name("sonar-ssl-data", create_if_missing=True)
_DATA_DIR = "/data"
_SHARDS_DIR = f"{_DATA_DIR}/shards"
_OUT_DIR = f"{_DATA_DIR}/checkpoints"


@app.function(
    # Single A100. For multi-GPU use the separate `train_multi` function below —
    # setting "A100:N" here would NOT parallelize (one process uses 1 GPU, bills N);
    # real multi-GPU needs the notebook_launcher in train_multi.
    gpu="A100",
    # The run is augmentation-bound (the A100 sits ~95% idle waiting on CPU-side
    # speckle/range-falloff/crops), so give the DataLoader workers real cores —
    # throughput scales ~linearly with them. Keep num_workers <= cpu and <= the
    # shard count (~80-100) so no worker starves.
    cpu=32.0,
    # 24h is Modal's max; at ~717 samples/s (~22 min per full pass) that fits
    # ~60 epochs. Default is 300s, so this must be set explicitly.
    timeout=24 * 60 * 60,
    volumes={_DATA_DIR: vol},
    secrets=[modal.Secret.from_name("wandb")],  # WANDB_API_KEY from the "wandb" secret
)
def train(
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 32,
    wandb_enabled: bool = True,
    epoch_length: int | None = None,
    variant: str = "yolov8n",
) -> str:
    """Run BYOL pretraining on the GPU box and return the checkpoint path.

    ``variant`` (``yolov8n``/``yolo26n``) picks the architecture; the downstream
    B6 fine-tune must use the same one (``init.model_variant``).
    """
    import glob

    from pretraining.ssl.byol import TrainConfig, train_byol

    vol.reload()
    shards = sorted(glob.glob(f"{_SHARDS_DIR}/benthicat-*.tar"))
    if not shards:
        raise FileNotFoundError(
            f"no benthicat-*.tar shards under {_SHARDS_DIR}; "
            "upload them with `modal volume put` first"
        )

    cfg = TrainConfig(
        shards=shards,
        out_dir=_OUT_DIR,
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        wandb_project="sonar-ssl",
        epoch_length=epoch_length,
        mixed_precision="bf16",
    )

    # Commit after every checkpoint so a mid-run crash still leaves it durable.
    ckpt_path = train_byol(cfg, on_checkpoint=lambda _path: vol.commit())
    vol.commit()  # Volume writes are buffered; commit makes the final .pt visible.
    return ckpt_path


# Number of GPUs for the multi-GPU function. Static because @app.function's gpu=
# is fixed at decoration time, so multi-GPU lives in its own function.
_N_GPUS = 4


@app.function(
    gpu=f"A100:{_N_GPUS}",
    # ~16 cores/GPU to feed each process's dataloader workers. The run is
    # augmentation-bound, so total workers (N_GPUS x num_workers) drive throughput.
    cpu=float(_N_GPUS * 16),
    timeout=24 * 60 * 60,
    volumes={_DATA_DIR: vol},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_multi(
    epochs: int = 40,
    batch_size: int = 256,
    num_workers: int = 16,
    wandb_enabled: bool = True,
    total_tiles: int = 957_040,
    variant: str = "yolov8n",
) -> str:
    """Multi-GPU (DDP) BYOL pretraining via ``accelerate.notebook_launcher``.

    Spawns ``_N_GPUS`` processes, each on one A100 with its own dataloader workers,
    so the augmentation-bound throughput scales ~with GPU count. ``total_tiles`` (the
    count preprocess_volume printed) sets epoch_length for ~one full pass per epoch.
    """
    import glob

    # CUDA-free import: ddp.py pulls in no torch/ultralytics at top level, so the
    # parent stays clean and the spawn workers do the heavy imports themselves.
    from pretraining.ssl.ddp import launch

    vol.reload()
    shards = sorted(glob.glob(f"{_SHARDS_DIR}/benthicat-*.tar"))
    if not shards:
        raise FileNotFoundError(
            f"no benthicat-*.tar shards under {_SHARDS_DIR}; preprocess first"
        )

    # Each shard is read by exactly one (process, worker), so cap workers at
    # shards-per-process or some workers starve ("fewer shards than workers").
    workers = max(1, min(num_workers, len(shards) // _N_GPUS))
    # with_epoch is per-worker, so samples/epoch = N_GPUS * workers * epoch_length;
    # target ~one full pass over total_tiles each epoch.
    epoch_length = max(1, total_tiles // (_N_GPUS * workers))
    print(
        f"[train_multi] {_N_GPUS} GPUs x {workers} workers; epoch_length={epoch_length} "
        f"(~{_N_GPUS * workers * epoch_length} samples/epoch over {len(shards)} shards)"
    )

    # Pass a plain dict (not a TrainConfig) so the parent never imports the
    # ultralytics-bearing byol module; each spawned worker reconstructs TrainConfig.
    cfg_dict = dict(
        shards=shards,
        out_dir=_OUT_DIR,
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=workers,
        wandb_enabled=wandb_enabled,
        wandb_project="sonar-ssl",
        epoch_length=epoch_length,
        mixed_precision="bf16",
    )

    # Spawn (not fork) _N_GPUS processes under DDP; rank 0 writes checkpoints to
    # the mounted Volume, and we commit once all workers return.
    launch(cfg_dict, _N_GPUS)

    vol.commit()
    return f"{_OUT_DIR}/byol_benthicat_{variant}.pt"


@app.function(volumes={_DATA_DIR: vol})
def inspect_volume() -> list[str]:
    """List the Volume contents (shards + checkpoints) for a quick sanity check."""
    import os

    vol.reload()
    found: list[str] = []
    for root, _dirs, files in os.walk(_DATA_DIR):
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, _DATA_DIR)
            found.append(f"{rel}\t{os.path.getsize(full)}")
    for line in found:
        print(line)
    return found


@app.function(volumes={_DATA_DIR: vol}, timeout=6 * 60 * 60)
def download_to_volume() -> dict:
    """Download every BenthiCat archive from Harvard Dataverse straight to ``/raw``.

    The dataset (DOI ``10.7910/DVN/2VCN7Y``) is public, so the access API serves
    each file directly (303 -> pre-signed S3, which urllib follows). Commits per
    file, so a re-run resumes (files already present at the manifest size skip).

        modal run pretraining/modal_app.py::download_to_volume
    """
    import json
    import os
    import shutil
    import urllib.request

    _DOI = "doi:10.7910/DVN/2VCN7Y"
    _BASE = "https://dataverse.harvard.edu"
    list_url = (
        f"{_BASE}/api/datasets/:persistentId/versions/:latest/files"
        f"?persistentId={_DOI}"
    )

    def _req(url: str) -> "urllib.request.Request":
        return urllib.request.Request(url, headers={"User-Agent": "sonar-ssl/1.0"})

    vol.reload()
    raw_dir = f"{_DATA_DIR}/raw"
    os.makedirs(raw_dir, exist_ok=True)

    with urllib.request.urlopen(_req(list_url)) as resp:
        listing = json.load(resp)["data"]

    n_downloaded = 0
    total_bytes = 0
    for entry in listing:
        df = entry["dataFile"]
        fid = df["id"]
        name = df["filename"]
        expected = int(df.get("filesize", 0))
        dest = f"{raw_dir}/{name}"

        if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
            print(f"[download] skip {name} (already {expected} bytes)")
            total_bytes += expected
            continue

        url = f"{_BASE}/api/access/datafile/{fid}"
        print(f"[download] {name} ({expected / 1e9:.2f} GB) <- datafile/{fid}")
        # Stream to .part then rename, so an interrupted file never looks complete.
        part = f"{dest}.part"
        with urllib.request.urlopen(_req(url)) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        os.replace(part, dest)
        n_downloaded += 1
        total_bytes += os.path.getsize(dest)
        vol.commit()  # persist each archive as it lands (crash-safe / resumable)

    vol.commit()
    print(
        f"[download] DONE: {len(listing)} files on Volume "
        f"({total_bytes / 1e9:.1f} GB total)"
    )
    return {
        "n_files": len(listing),
        "n_downloaded": n_downloaded,
        "total_bytes": total_bytes,
    }


@app.function(
    image=preprocess_image,
    volumes={_DATA_DIR: vol},
    cpu=4.0,
    timeout=12 * 60 * 60,
)
def preprocess_volume() -> dict:
    """Extract the .7z archives in ``/raw`` and pack them into shards, on the Volume.

    Prereq: archives in ``/data/raw`` (via ``download_to_volume`` or ``modal volume
    put``). They ship as multi-volume splits (``N04.7z.001``, ``.002``, ...); 7za
    reassembles a set from its ``.001`` volume, so we iterate first volumes only.
    One sector at a time (extract -> pack -> delete) keeps transient disk ~1 sector.
    """
    import glob
    import os
    import shutil
    import subprocess

    from pretraining.preprocess import preprocess

    vol.reload()
    raw_dir = f"{_DATA_DIR}/raw"
    os.makedirs(_SHARDS_DIR, exist_ok=True)

    # ".7z.001" first-volumes + any plain single-file ".7z" (globs don't overlap).
    archives = sorted(
        glob.glob(f"{raw_dir}/*.7z.001") + glob.glob(f"{raw_dir}/*.7z")
    )
    if not archives:
        raise FileNotFoundError(
            f"no .7z / .7z.001 archives under {raw_dir}; get them onto the Volume "
            "with `modal run pretraining/modal_app.py::download_to_volume` "
            "(or `modal volume put sonar-ssl-data <dir> /raw`) first"
        )

    total = 0
    for arc in archives:
        # "N04.7z.001" -> "N04", "S02.7z" -> "S02" (splitext would give "N04.7z").
        sector = os.path.basename(arc).split(".7z")[0]
        tmp = f"{_DATA_DIR}/_extract/{sector}"
        os.makedirs(tmp, exist_ok=True)
        subprocess.run(["7za", "x", arc, f"-o{tmp}", "-y"], check=True)
        result = preprocess(tmp, _SHARDS_DIR, pattern=f"benthicat-{sector}-%06d.tar")
        total += int(result["n_tiles"])
        shutil.rmtree(tmp)
        vol.commit()
        print(
            f"[preprocess_volume] {sector}: {result['n_tiles']} tiles "
            f"(running total {total})"
        )

    shutil.rmtree(f"{_DATA_DIR}/_extract", ignore_errors=True)
    vol.commit()
    print(f"[preprocess_volume] DONE: {total} tiles -> {_SHARDS_DIR}")
    return {"n_tiles": total}


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 32,
    wandb_enabled: bool = True,
    epoch_length: int | None = None,
    variant: str = "yolov8n",
) -> None:
    """Launch remote training from your laptop. ``--variant yolo26n`` for YOLO26."""
    ckpt_path = train.remote(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        epoch_length=epoch_length,
        variant=variant,
    )
    print(f"[modal] BYOL pretraining finished; checkpoint on Volume at: {ckpt_path}")
