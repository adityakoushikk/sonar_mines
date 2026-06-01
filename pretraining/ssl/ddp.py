"""Spawn-based multi-GPU launch for DDP BYOL pretraining (Modal-safe).

We use the *spawn* start method, not fork. Modal's GPU container has already
touched CUDA by the time our function runs, so a fork-based launcher
(``accelerate.notebook_launcher``) crashes every child with "Cannot re-initialize
CUDA in forked subprocess". Spawned children are fresh interpreters, so the
parent's CUDA state doesn't matter.

This module imports nothing heavy at top level (no torch, ultralytics, or modal),
so the parent can import it while staying CUDA-clean; the per-process imports
happen inside the spawned workers.
"""
from __future__ import annotations

import os


def ddp_worker(rank: int, cfg_dict: dict, world_size: int, master_port: int) -> None:
    """Per-rank entry: set the torchrun-style env, then run train_byol under DDP."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)

    # Imported here so ultralytics' import-time CUDA probe runs inside this fresh
    # spawned process, and accelerate picks up the env vars set above.
    from pretraining.ssl.byol import TrainConfig, train_byol

    train_byol(TrainConfig(**cfg_dict))


def launch(cfg_dict: dict, world_size: int, master_port: int = 29500) -> None:
    """Spawn ``world_size`` processes running :func:`ddp_worker` under DDP."""
    import torch.multiprocessing as mp

    mp.start_processes(
        ddp_worker,
        args=(cfg_dict, world_size, master_port),
        nprocs=world_size,
        start_method="spawn",
    )
