"""Convert BenthiCat ``.npy`` SSS tiles into WebDataset shards for SSL.

The self-supervised BYOL pretraining stage (B6) streams unlabeled side-scan
sonar tiles from sharded ``.tar`` files (WebDataset) rather than touching the
filesystem per-sample — this is what keeps shuffled, multi-worker reads fast on
a cluster filesystem like Sherlock's. Raw BenthiCat tiles ship as one
``float32`` array per file in a ``<sector>/<file>.npy`` layout; here we walk that
tree, quantize each tile to ``uint8``, encode it as a *lossless* single-channel
PNG, and append it to a shard.

We encode the FULL original filename stem as the sample ``__key__`` because that
stem embeds the sector / survey line / channel (e.g.
``N02_4_220807052300_xtf-CH12_batch0_ch0_r0_c0``); preserving it lets the
downstream loader stratify or de-duplicate by acquisition later, even though SSL
itself ignores it.

Run as a module from the repo root::

    python -m pretraining.preprocess --input data/benthicat --output data/benthicat_shards
    python -m pretraining.preprocess --input ... --output ... --limit 1000
"""
from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import webdataset as wds
from PIL import Image

# BenthiCat tiles are nominally 384x384 float32 in ~[0, 1]; we do not enforce
# the spatial size (the SSL transform resizes/crops to 224), only the dtype and
# value-range conversion below.


def _iter_npy_files(input_dir: Path) -> Iterator[Path]:
    """Yield every ``*.npy`` file under ``input_dir`` recursively, sorted.

    Sorting makes ``--limit`` deterministic (same first-N tiles across runs) and
    keeps shard contents reproducible. Non-``.npy`` files are skipped by the glob
    itself, so callers can point at a mixed directory tree safely.
    """
    # rglob walks arbitrarily nested <sector>/.../<file>.npy without us having to
    # know the depth; sorted() pins iteration order for reproducible shards.
    yield from sorted(input_dir.rglob("*.npy"))


def _tile_to_png_bytes(arr: np.ndarray) -> bytes:
    """Quantize a float SSS tile to ``uint8`` and encode as a lossless PNG.

    Args:
        arr: A single tile. ``HxW`` or ``HxWx1`` (a trailing singleton channel is
            squeezed); values are float ~[0, 1]. Other dtypes/shapes raise.

    Returns:
        The PNG-encoded bytes of a mode-``"L"`` (single-channel) image.
    """
    # Drop a trailing singleton channel so HxWx1 collapses to HxW; PIL mode "L"
    # requires a 2-D array. We deliberately reject genuine multi-channel tiles
    # rather than silently picking a channel.
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(
            f"expected a HxW (or HxWx1) tile, got array with shape {arr.shape}"
        )

    # Float [0, 1] -> uint8 [0, 255]. clip BEFORE the cast so out-of-range values
    # (e.g. a stray 1.0001) wrap-safely to 255/0 instead of overflowing uint8.
    quantized = (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(quantized, mode="L")
    buffer = io.BytesIO()
    # PNG is lossless; this preserves the exact uint8 quantization for BYOL.
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def preprocess(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    limit: int | None = None,
    maxsize: float = 2e9,
    maxcount: int = 100_000,
    pattern: str = "benthicat-%06d.tar",
) -> dict:
    """Pack BenthiCat ``.npy`` tiles into WebDataset shards under ``output_dir``.

    Streams one tile at a time into :class:`webdataset.ShardWriter` so peak memory
    stays flat regardless of dataset size. Each sample is
    ``{"__key__": <full stem>, "png": <bytes>}``.

    Args:
        input_dir: Root of the BenthiCat ``<sector>/<file>.npy`` tree (searched
            recursively).
        output_dir: Directory to receive ``<pattern>`` shards; created if absent.
        limit: If given, process only the first ``limit`` tiles (sorted order).
        maxsize: Soft per-shard byte cap handed to ``ShardWriter``.
        maxcount: Per-shard sample-count cap handed to ``ShardWriter``.
        pattern: ``ShardWriter`` filename pattern (must contain a ``%d`` field).

    Returns:
        A manifest dict ``{"n_tiles", "n_shards", "total_bytes"}`` where
        ``total_bytes`` is the on-disk size of all written shards.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_pattern = os.path.join(str(output_dir), pattern)

    n_tiles = 0
    start = time.perf_counter()

    # ShardWriter rotates to a new .tar whenever maxsize OR maxcount is hit, so we
    # never hold more than one tile's bytes in memory at once.
    with wds.ShardWriter(shard_pattern, maxsize=maxsize, maxcount=maxcount) as sink:
        for npy_path in _iter_npy_files(input_dir):
            if limit is not None and n_tiles >= limit:
                break

            # Load one tile at a time and hand its bytes straight to the writer,
            # so peak memory stays flat regardless of how many tiles we process.
            arr = np.asarray(np.load(npy_path, allow_pickle=False))
            png_bytes = _tile_to_png_bytes(arr)

            # FULL stem (no parent, no suffix) — encodes sector/line/channel for
            # later stratification; ".npy" is the only suffix so .stem is exact.
            sink.write({"__key__": npy_path.stem, "png": png_bytes})
            n_tiles += 1

    elapsed = time.perf_counter() - start
    throughput = n_tiles / elapsed if elapsed > 0 else float("nan")

    # Sum the bytes of exactly the shards we just wrote (pattern's prefix), not
    # any pre-existing files in output_dir.
    written_shards = sorted(output_dir.glob(pattern.replace("%06d", "*").replace("%d", "*")))
    total_bytes = sum(p.stat().st_size for p in written_shards)
    n_shards = len(written_shards)

    print(
        f"[preprocess] {n_tiles} tiles -> {n_shards} shards "
        f"({total_bytes / 1e6:.1f} MB) in {elapsed:.1f}s "
        f"({throughput:.1f} tiles/s)"
    )

    return {"n_tiles": n_tiles, "n_shards": n_shards, "total_bytes": total_bytes}


def main() -> None:
    """CLI entry point: parse args and run :func:`preprocess`."""
    parser = argparse.ArgumentParser(
        description="Pack BenthiCat .npy SSS tiles into WebDataset shards for "
        "self-supervised pretraining.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root dir of the BenthiCat <sector>/<file>.npy tree (recursive).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dir for the WebDataset .tar shards (created if absent).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tiles (sorted order). Default: all.",
    )
    parser.add_argument(
        "--maxsize",
        type=float,
        default=2e9,
        help="Soft per-shard byte cap (default: 2e9).",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=100_000,
        help="Per-shard sample-count cap (default: 100000).",
    )
    parser.add_argument(
        "--pattern",
        default="benthicat-%06d.tar",
        help="ShardWriter filename pattern (default: benthicat-%%06d.tar).",
    )
    args = parser.parse_args()

    preprocess(
        args.input,
        args.output,
        limit=args.limit,
        maxsize=args.maxsize,
        maxcount=args.maxcount,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
