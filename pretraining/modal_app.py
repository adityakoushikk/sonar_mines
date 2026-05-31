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
#
# IMPORTANT build ordering: Modal forbids any build step (apt_install/pip/
# run_commands) *after* `add_local_*`. So all package/apt layers go on `_base`
# first, and `_add_local()` (which mounts the source dirs) is always the LAST
# step. The preprocess image adds its apt layer to `_base` *before* that mount.
_base = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("pretraining/requirements.txt")
    # Put /root on sys.path so the src/ and pretraining/ dirs mounted below are
    # importable: add_local_dir copies the files but does not itself extend the
    # import path, and the remote trainer does `import src` / `import pretraining`.
    .env({"PYTHONPATH": "/root"})
)


def _add_local(img: "modal.Image") -> "modal.Image":
    """Mount src/ and pretraining/ onto the container import path.

    MUST be the final build step on any image (Modal disallows build steps after
    add_local_*; that constraint is exactly what this helper centralizes).

    GOTCHA: code imported by the remote function is NOT auto-shipped unless it
    lives next to this file. The trainer does `from src.augmentations import ...`
    and `from pretraining.ssl import ...`, so both top-level package dirs must be
    on the container's import path. We mount them under /root (the container CWD,
    which is on sys.path) so the imports resolve exactly as they do locally from
    the repo root. add_local_dir (vs add_local_python_source) keeps non-.py
    assets — e.g. the YOLO yaml shipped with ultralytics — explicit.
    """
    return (
        img
        .add_local_dir("src", remote_path="/root/src")
        .add_local_dir("pretraining", remote_path="/root/pretraining")
    )


# Default training image: base deps + local source.
image = _add_local(_base)
# Preprocess image: base deps + 7-Zip (apt layer BEFORE the local mount), then
# local source. p7zip-full provides the `7za` extractor for the .7z archives.
preprocess_image = _add_local(_base.apt_install("p7zip-full"))

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
    # runs for hours (a single-A100 100-epoch run is ~10h), so we set 12h of
    # headroom; without this the run is killed mid-epoch. (Modal max is 24h.)
    timeout=12 * 60 * 60,
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
    variant: str = "yolov8n",
) -> str:
    """Run BYOL pretraining on the GPU box and persist the checkpoint.

    Args:
        epochs: Number of training epochs.
        batch_size: Per-process batch size.
        num_workers: DataLoader workers per process.
        wandb_enabled: Log metrics to W&B (uses the mounted "wandb" secret).
        epoch_length: Samples/epoch for the streaming dataset (None = one full
            pass over the shards).
        variant: YOLO architecture to pretrain the backbone of, e.g.
            ``"yolov8n"`` or ``"yolo26n"``. The backbone cut is inferred from the
            variant's yaml, so no other change is needed to switch. The downstream
            B6 fine-tune must use the same variant (``init.model_variant``).

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
        variant=variant,
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


@app.function(
    # Network-bound, not compute-bound: just streams files from Dataverse to the
    # Volume. The training image already has urllib (stdlib); no 7-Zip needed here.
    volumes={_DATA_DIR: vol},
    timeout=6 * 60 * 60,
)
def download_to_volume() -> dict:
    """Download every BenthiCat archive from Harvard Dataverse straight to the
    Volume's ``/raw`` — no large download ever touches your local machine.

    The dataset (DOI ``10.7910/DVN/2VCN7Y``, CC BY-NC-SA 4.0) is public with no
    access request, so the Dataverse access API serves each file directly (via a
    303 redirect to pre-signed S3 that urllib follows). We list the dataset's
    files, then stream each one to ``/data/raw`` and commit per file, so a re-run
    resumes — already-complete files (size matches the manifest) are skipped.

    Run::

        modal run pretraining/modal_app.py::download_to_volume

    Then pack shards with ``preprocess_volume`` (it handles the multi-volume
    ``.7z.001``/``.002``/... splits these archives ship as).
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
        # A descriptive User-Agent avoids occasional bot filtering; urllib follows
        # the 303 redirect from the access endpoint to S3 automatically.
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

        # Resume: a file already fully present (size matches the manifest) is skipped.
        if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
            print(f"[download] skip {name} (already {expected} bytes)")
            total_bytes += expected
            continue

        url = f"{_BASE}/api/access/datafile/{fid}"
        print(f"[download] {name} ({expected / 1e9:.2f} GB) <- datafile/{fid}")
        # Stream to a .part file, then atomically rename — so an interrupted
        # download never looks complete to the resume check above.
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
    # Only this function needs 7-Zip, so it uses the dedicated preprocess_image
    # (base deps + p7zip-full, built BEFORE the local mount); the training image
    # stays unchanged.
    image=preprocess_image,
    volumes={_DATA_DIR: vol},
    cpu=4.0,
    timeout=12 * 60 * 60,
)
def preprocess_volume() -> dict:
    """Extract BenthiCat .7z archives and pack them into WebDataset shards
    entirely on the Volume — no large local disk, no shard upload.

    Prereq: get the archives onto /data/raw first, either directly from Dataverse::

        modal run pretraining/modal_app.py::download_to_volume

    or from a local copy::

        modal volume put sonar-ssl-data <local-dir-of-.7z> /raw

    Then run::

        modal run pretraining/modal_app.py::preprocess_volume

    These archives ship as MULTI-VOLUME 7-Zip splits (``N04.7z.001``,
    ``N04.7z.002``, ...). ``7za`` reassembles a whole set when pointed at its
    ``.001`` volume (it auto-joins the rest from the same dir), so we iterate over
    first volumes only. Processes one sector at a time (extract -> pack -> delete
    -> commit) so transient .npy on the Volume stays ~one sector, not the full
    ~560 GB.
    """
    import glob
    import os
    import shutil
    import subprocess

    from pretraining.preprocess import preprocess

    vol.reload()
    raw_dir = f"{_DATA_DIR}/raw"
    os.makedirs(_SHARDS_DIR, exist_ok=True)

    # First volumes of each set (".7z.001") plus any plain single-file ".7z".
    # glob "*.7z" excludes split parts, whose names end in ".001"/".002"/... — so
    # the two globs don't overlap.
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
        # Sector = everything before ".7z": "N04.7z.001" -> "N04", "S02.7z" ->
        # "S02". (splitext would wrongly yield "N04.7z" for a split's .001 part.)
        sector = os.path.basename(arc).split(".7z")[0]
        tmp = f"{_DATA_DIR}/_extract/{sector}"
        os.makedirs(tmp, exist_ok=True)
        # p7zip-full's `7za` extracts .7z. -o<dir> takes no space; -y = assume yes.
        subprocess.run(["7za", "x", arc, f"-o{tmp}", "-y"], check=True)
        # Sector-unique shard pattern so every shard still matches benthicat-*.tar.
        result = preprocess(tmp, _SHARDS_DIR, pattern=f"benthicat-{sector}-%06d.tar")
        total += int(result["n_tiles"])
        shutil.rmtree(tmp)        # free this sector's .npy before the next one
        vol.commit()              # persist shards incrementally (crash-safe)
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
    num_workers: int = 8,
    wandb_enabled: bool = True,
    epoch_length: int | None = None,
    variant: str = "yolov8n",
) -> None:
    """Local entry point: kick off remote GPU training and print the result.

    Runs on your laptop (no GPU needed); ``.remote()`` ships the call to Modal.
    Pass ``--variant yolo26n`` to pretrain a YOLO26 backbone instead of YOLOv8n.
    """
    ckpt_path = train.remote(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=wandb_enabled,
        epoch_length=epoch_length,
        variant=variant,
    )
    print(f"[modal] BYOL pretraining finished; checkpoint on Volume at: {ckpt_path}")
