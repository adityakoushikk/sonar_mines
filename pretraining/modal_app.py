"""Modal app for running BYOL self-supervised pretraining (B6) on GPUs.

This is the cloud entry point for the SSL stage. It packages the repo's `src/`
augmentations and the `pretraining/` package into a Modal image, mounts a
persistent Volume at ``/data`` for shards + checkpoints, and runs
:func:`pretraining.ssl.byol.train_byol` on a multi-GPU A100 box.

Local code is kept *out* of the trainer modules and concentrated here so the
GPU/cluster plumbing stays in one auditable place. The trainer itself
(``pretraining/ssl/byol.py``) has no Modal dependency and runs identically on a
laptop CPU for smoke tests.

Run::

    # one-off training launch (reads/writes the Volume at /data):
    modal run pretraining/modal_app.py

    # inspect what is on the Volume without launching a GPU:
    modal run pretraining/modal_app.py::inspect_volume
"""
from __future__ import annotations

import modal

# --- Image -----------------------------------------------------------------
# Build from the *isolated* SSL requirements (lightly/webdataset/accelerate/
# modal live here, separate from the repo-root requirements.txt) so the GPU
# image matches what the trainer imports.
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("pretraining/requirements.txt")
    # Put /root on sys.path so the src/ and pretraining/ dirs copied below are
    # importable: add_local_dir copies the files but does not itself extend the
    # import path, and the remote trainer does `import src` / `import pretraining`.
    .env({"PYTHONPATH": "/root"})
    # GOTCHA: code imported by the remote function is NOT auto-shipped unless it
    # lives next to this file. The trainer does `from src.augmentations import
    # ...` and `from pretraining.ssl import ...`, so both top-level package dirs
    # must be present on the container's import path. We mount them under /root
    # (the container CWD, which is on sys.path) so `import src` / `import
    # pretraining` resolve exactly as they do locally from the repo root.
    #
    # add_local_dir copies a directory tree into the image at build/deploy time.
    # (On modal >= 0.63 the modern equivalent for pure-Python packages is
    # `image.add_local_python_source("src", "pretraining")`; we use add_local_dir
    # so non-.py assets like the YOLO yaml shipped with ultralytics' install are
    # unaffected and the mount is explicit.)
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("pretraining", remote_path="/root/pretraining")
)

app = modal.App("sonar-ssl", image=image)

# --- Persistent storage ----------------------------------------------------
# A named Volume survives across runs; create_if_missing makes the first run
# self-bootstrapping. Put the WebDataset shards under /data/shards (uploaded
# out-of-band, e.g. `modal volume put`) and write checkpoints under
# /data/checkpoints so they persist after the function returns.
vol = modal.Volume.from_name("sonar-ssl-data", create_if_missing=True)

# Mount point inside the container for the Volume above.
_DATA_DIR = "/data"
# Shards live under /data/shards (uploaded out-of-band via `modal volume put`);
# train() globs the actual files at run time rather than assuming a brace range.
_SHARDS_DIR = f"{_DATA_DIR}/shards"
_OUT_DIR = f"{_DATA_DIR}/checkpoints"


@app.function(
    # Default to a SINGLE A100: the yolov8n backbone is tiny and data-bound, so one
    # GPU is the right first run, and it avoids the multi-GPU DDP path that can't be
    # smoke-tested locally. SCALING: set "A100:2"/"A100:4" AND pass --epoch-length
    # (multi-GPU needs equal batches/rank; train_byol enforces this).
    gpu="A100",
    # GOTCHA: Modal's default function timeout is 300s (5 min). SSL pretraining
    # runs for hours, so we raise it to the 8h ceiling; without this the run is
    # killed mid-epoch.
    timeout=8 * 60 * 60,
    # Mount the persistent Volume so shards are readable and checkpoints persist.
    volumes={_DATA_DIR: vol},
    # GOTCHA: the W&B API key is injected from a Modal Secret literally named
    # "wandb" (create once with `modal secret create wandb WANDB_API_KEY=...`).
    # It lands as the WANDB_API_KEY env var inside the container.
    secrets=[modal.Secret.from_name("wandb")],
)
def train(
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 8,
    wandb_enabled: bool = True,
    epoch_length: int | None = None,
) -> str:
    """Run BYOL pretraining on the GPU box and persist the checkpoint.

    Args:
        epochs: Number of training epochs.
        batch_size: Per-process batch size.
        num_workers: DataLoader workers per process.
        wandb_enabled: Log metrics to W&B (uses the mounted "wandb" secret).
        epoch_length: Samples/epoch for the streaming dataset (None = one full
            pass over the shards).

    Returns:
        The checkpoint path on the Volume (under ``/data/checkpoints``).
    """
    # Import inside the function: these heavy deps only exist in the remote image,
    # not necessarily in the local launching environment.
    import glob

    from pretraining.ssl.byol import TrainConfig, train_byol

    # Resolve the real shard files on the Volume at run time. WebDataset opens
    # every URL it is handed, so a fixed over-wide brace range would error on the
    # shards that don't exist; reload() first so freshly-put shards are visible.
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
        variant="yolov8n",
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        wandb_project="sonar-ssl",
        epoch_length=epoch_length,
        # bf16 is safe + fast on A100; train_byol falls back to "no" off-CUDA.
        mixed_precision="bf16",
    )

    # Commit the Volume after every checkpoint write (not just at the end) so a
    # crash during a multi-hour run still leaves the latest backbone durable.
    ckpt_path = train_byol(cfg, on_checkpoint=lambda _path: vol.commit())

    # GOTCHA: writes to a Volume are buffered — without commit() the checkpoint is
    # NOT visible to other functions / future runs / `modal volume get`. Commit
    # after the trainer has finished writing the final .pt.
    vol.commit()
    return ckpt_path


@app.function(volumes={_DATA_DIR: vol})
def inspect_volume() -> list[str]:
    """List the Volume contents (shards + checkpoints) for a quick sanity check.

    Handy after `modal volume put` to confirm shards landed, and after a run to
    confirm the checkpoint was committed. Returns relative paths under /data.
    """
    import os

    # reload() pulls in any commits made by other functions since this container
    # started, so we see freshly-written checkpoints.
    vol.reload()

    found: list[str] = []
    for root, _dirs, files in os.walk(_DATA_DIR):
        for name in sorted(files):
            full = os.path.join(root, name)
            size = os.path.getsize(full)
            rel = os.path.relpath(full, _DATA_DIR)
            found.append(f"{rel}\t{size}")
    for line in found:
        print(line)
    return found


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 256,
    num_workers: int = 8,
    wandb_enabled: bool = True,
    epoch_length: int | None = None,
) -> None:
    """Local entry point: kick off remote GPU training and print the result.

    Runs on your laptop (no GPU needed); ``.remote()`` ships the call to Modal.
    """
    ckpt_path = train.remote(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        epoch_length=epoch_length,
    )
    print(f"[modal] BYOL pretraining finished; checkpoint on Volume at: {ckpt_path}")
