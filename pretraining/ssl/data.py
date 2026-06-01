"""WebDataset input pipeline for BYOL pretraining on sharded SSS tiles."""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import torch
import webdataset as wds
from PIL import Image

from pretraining.ssl.transforms import SonarBYOLTransform


def _decode_sample(
    sample: dict, transform: SonarBYOLTransform
) -> tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(io.BytesIO(sample["png"]))
    return transform(np.asarray(img))  # HxW uint8 for mode "L"


def _collate_two_views(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    view0 = torch.stack([b[0] for b in batch], dim=0)
    view1 = torch.stack([b[1] for b in batch], dim=0)
    return view0, view1


def build_webdataset_loader(
    shards,
    transform: SonarBYOLTransform,
    batch_size: int,
    num_workers: int,
    *,
    world_size: int = 1,
    epoch_length: int | None = None,
    shardshuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """DataLoader over WebDataset shards yielding ((B,3,224,224), (B,3,224,224)) view pairs.

    For ``world_size > 1`` shards are split per node so each rank sees a disjoint
    subset; ``epoch_length`` fixes samples/epoch (IterableDatasets have no length).
    """
    nodesplitter = wds.split_by_node if world_size > 1 else None
    # webdataset>=1.0 wants an int shuffle-buffer, not a bare True.
    shardshuffle_arg = 1000 if shardshuffle is True else shardshuffle

    dataset = wds.WebDataset(
        shards,
        shardshuffle=shardshuffle_arg,
        nodesplitter=nodesplitter,
    )
    dataset = dataset.map(lambda s: _decode_sample(s, transform))

    if epoch_length is not None:
        dataset = dataset.with_epoch(epoch_length)

    # Force fork for the DataLoader workers: under a spawn-launched DDP process the
    # default context becomes spawn, which would need to pickle the (lambda-bearing)
    # WebDataset pipeline. Workers do only CPU augmentation (never touch CUDA), so
    # forking from a CUDA-initialized process is safe — same as the single-GPU path.
    mp_context = "fork" if num_workers > 0 else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_two_views,
        drop_last=True,
        multiprocessing_context=mp_context,
    )
