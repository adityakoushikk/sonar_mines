"""Convert BenthiCat ``.npy`` SSS tiles into WebDataset shards for SSL.

Walks a ``<sector>/<file>.npy`` tree, quantizes each float tile to uint8, encodes
a lossless single-channel PNG, and streams it into ``.tar`` shards. The sample
``__key__`` is the full filename stem (it embeds sector/line/channel).

    python -m pretraining.preprocess --input data/benthicat --output data/benthicat_shards
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


def _iter_npy_files(input_dir: Path) -> Iterator[Path]:
    """Yield every ``*.npy`` under ``input_dir`` recursively, sorted (deterministic)."""
    yield from sorted(input_dir.rglob("*.npy"))


def _tile_to_png_bytes(arr: np.ndarray) -> bytes:
    """Quantize a float [0,1] HxW(x1) tile to uint8 and encode a lossless mode-L PNG."""
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(
            f"expected a HxW (or HxWx1) tile, got array with shape {arr.shape}"
        )

    # clip before the cast so stray out-of-range values don't wrap the uint8.
    quantized = (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(quantized, mode="L").save(buffer, format="PNG")
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
    """Pack BenthiCat ``.npy`` tiles into WebDataset shards; one tile in memory at a time.

    Each sample is ``{"__key__": <stem>, "png": <bytes>}``. Returns a manifest
    ``{"n_tiles", "n_shards", "total_bytes"}``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_pattern = os.path.join(str(output_dir), pattern)
    n_tiles = 0
    start = time.perf_counter()

    with wds.ShardWriter(shard_pattern, maxsize=maxsize, maxcount=maxcount) as sink:
        for npy_path in _iter_npy_files(input_dir):
            if limit is not None and n_tiles >= limit:
                break
            arr = np.asarray(np.load(npy_path, allow_pickle=False))
            sink.write({"__key__": npy_path.stem, "png": _tile_to_png_bytes(arr)})
            n_tiles += 1

    elapsed = time.perf_counter() - start
    throughput = n_tiles / elapsed if elapsed > 0 else float("nan")

    written_shards = sorted(
        output_dir.glob(pattern.replace("%06d", "*").replace("%d", "*"))
    )
    total_bytes = sum(p.stat().st_size for p in written_shards)
    n_shards = len(written_shards)

    print(
        f"[preprocess] {n_tiles} tiles -> {n_shards} shards "
        f"({total_bytes / 1e6:.1f} MB) in {elapsed:.1f}s ({throughput:.1f} tiles/s)"
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
