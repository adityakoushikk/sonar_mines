"""Train/val/test split strategies for the Santos dataset."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"fractions must sum to 1.0, got "
            f"{train_frac} + {val_frac} + {test_frac} "
            f"= {train_frac + val_frac + test_frac}"
        )


def random_split(
    n_items: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    """Uniform random split of ``[0, n_items)`` into train/val/test.

    Args:
        n_items: Number of indexable items in the dataset.
        train_frac, val_frac, test_frac: Must sum to 1.0.
        seed: RNG seed for reproducibility.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    _validate_fractions(train_frac, val_frac, test_frac)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items).tolist()

    n_train = int(round(n_items * train_frac))
    n_val = int(round(n_items * val_frac))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def stratified_random_split(
    strata: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    """Stratified random split keyed by per-item ``strata`` labels.

    Shuffles and slices each unique stratum independently by the same
    fractions, so per-stratum counts in train/val/test track the dataset
    population (up to rounding). Eliminates the dominant source of
    seed-to-seed variance when minority strata are small — for Santos, the 49
    NOMBO-only images would otherwise be allocated to val with a 4x spread
    across seeds.

    Args:
        strata: Per-item stratum label; the split length matches its length.
        train_frac, val_frac, test_frac: Must sum to 1.0.
        seed: RNG seed for reproducibility.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    _validate_fractions(train_frac, val_frac, test_frac)

    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)

    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for indices in groups.values():
        perm = rng.permutation(indices).tolist()
        n = len(perm)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train.extend(perm[:n_train])
        val.extend(perm[n_train : n_train + n_val])
        test.extend(perm[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


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
