"""Train/val/test split strategies for the Santos dataset."""
from __future__ import annotations

from typing import Sequence


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
    # TODO: shuffle indices with a seeded RNG, slice by fractions, return dict
    raise NotImplementedError("TODO: implement random_split")


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
    # TODO: bucket indices by year, carve val from train, return dict
    raise NotImplementedError("TODO: implement cross_year_split")
