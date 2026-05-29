"""WebDataset input pipeline for BYOL pretraining on sharded SSS tiles.

We decode the ``png`` field ourselves with PIL (rather than WebDataset's
built-in decoders) so the bytes -> uint8 ndarray path is explicit and matches
the preprocessing contract (mode "L", single channel). Each sample becomes a
two-view tuple via :class:`SonarBYOLTransform`, and we collate to a plain
2-tuple of ``(B, 3, 224, 224)`` tensors that the BYOL trainer consumes.
"""
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
    """Decode one WebDataset sample's PNG and return its two augmented views."""
    img = Image.open(io.BytesIO(sample["png"]))
    arr = np.asarray(img)  # HxW uint8 for mode "L"
    return transform(arr)


def _collate_two_views(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of (v0, v1) pairs into ((B,3,224,224), (B,3,224,224))."""
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
    """Build a DataLoader over WebDataset shards yielding BYOL view pairs.

    Args:
        shards: Brace-glob path (e.g. ``".../benthicat-{000000..000009}.tar"``)
            or an explicit list of shard paths.
        transform: Two-view transform applied per tile.
        batch_size: Samples per batch.
        num_workers: DataLoader workers; ``0`` must work (used by the smoke test).
        world_size: When > 1, shards are split across nodes via
            ``wds.split_by_node`` so each rank sees a disjoint subset.
        epoch_length: If given, ``.with_epoch(epoch_length)`` fixes the number of
            samples per epoch (IterableDatasets otherwise have no inherent length).
        shardshuffle: Whether to shuffle shard order each epoch.

    Returns:
        A ``torch.utils.data.DataLoader`` over an ``IterableDataset``; batches are
        2-tuples of ``(B, 3, 224, 224)`` float tensors.
    """
    # split_by_worker spreads shards across DataLoader workers; add node splitting
    # only for distributed runs so single-process smoke tests stay simple.
    nodesplitter = wds.split_by_node if world_size > 1 else None

    # webdataset>=1.0 wants an int shuffle-buffer size (or 0/False), and warns on a
    # bare ``True``. Keep the public ``shardshuffle: bool`` contract but translate
    # True -> a buffer large enough to shuffle whole shard lists; older 0.2.x
    # accepts the same int, so this is forward/backward compatible.
    shardshuffle_arg = 1000 if shardshuffle is True else shardshuffle

    dataset = wds.WebDataset(
        shards,
        shardshuffle=shardshuffle_arg,
        nodesplitter=nodesplitter,
    )
    dataset = dataset.map(lambda s: _decode_sample(s, transform))

    if epoch_length is not None:
        dataset = dataset.with_epoch(epoch_length)

    # Let the DataLoader batch the IterableDataset; our custom collate turns the
    # list of (v0, v1) pairs into the 2-tuple of stacked view tensors BYOL wants.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_two_views,
        drop_last=True,
    )
