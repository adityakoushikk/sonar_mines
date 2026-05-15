"""Train/val/test split strategies for the Santos dataset."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def random_split(
    n_items: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    """Shuffle and partition [0, n_items) into train/val/test by fraction.

    Args:
        n_items: Number of indexable items in the dataset.
        train_frac, val_frac, test_frac: Must sum to 1.0.
        seed: RNG seed for reproducibility.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"fractions must sum to 1.0, got "
            f"{train_frac} + {val_frac} + {test_frac} "
            f"= {train_frac + val_frac + test_frac}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items).tolist()

    n_train = int(round(n_items * train_frac))
    n_val = int(round(n_items * val_frac))
    # Remainder absorbs rounding so no items are lost.
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def cross_year_split(
    years_per_item: Sequence[int],
    train_years: Sequence[int],
    test_years: Sequence[int],
    val_frac_of_train: float,
    seed: int,
) -> dict[str, list[int]]:
    """Split by collection year for the Experiment 3 generalization test.

    Items whose year is in ``test_years`` go to test; items in ``train_years``
    are further split into train + val by ``val_frac_of_train``. Items in
    neither set are dropped.

    Args:
        years_per_item: Year associated with each item (same length as dataset).
        train_years, test_years: Disjoint sets of years.
        val_frac_of_train: Fraction of train_years items reserved for validation.
        seed: RNG seed for the train/val split.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    train_years_set = set(train_years)
    test_years_set = set(test_years)
    overlap = train_years_set & test_years_set
    if overlap:
        raise ValueError(f"train_years and test_years overlap: {sorted(overlap)}")
    if not 0.0 <= val_frac_of_train <= 1.0:
        raise ValueError(
            f"val_frac_of_train must be in [0, 1], got {val_frac_of_train}"
        )

    train_pool: list[int] = []
    test_indices: list[int] = []
    for i, y in enumerate(years_per_item):
        if y in train_years_set:
            train_pool.append(i)
        elif y in test_years_set:
            test_indices.append(i)

    rng = np.random.default_rng(seed)
    train_pool = rng.permutation(train_pool).tolist()
    n_val = int(round(len(train_pool) * val_frac_of_train))

    return {
        "train": train_pool[n_val:],
        "val": train_pool[:n_val],
        "test": test_indices,
    }
