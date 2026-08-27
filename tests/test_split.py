"""Tests for the chronological 70/15/15 temporal split (ADR-0003, ticket 02).

Seam A: `ieee_cis_fraud_detection.modeling.split.temporal_split_70_15_15`.
"""

import numpy as np
import pandas as pd
import pytest

from ieee_cis_fraud_detection.modeling.split import temporal_split_70_15_15


def make_frame(n: int = 100, *, seed: int = 0) -> pd.DataFrame:
    """A synthetic transaction frame ordered (mostly) by TransactionDT."""
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 1_000_000, size=n))
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": dt,
            "isFraud": rng.integers(0, 2, size=n),
        }
    )


def test_split_returns_three_chronological_non_overlapping_slices() -> None:
    df = make_frame(n=100)
    train, test, stream = temporal_split_70_15_15(df)

    assert len(train) == 70
    assert len(test) == 15
    assert len(stream) == 15

    # No overlap and no rows lost: the ordered concatenation equals the input.
    rebuilt = pd.concat([train, test, stream])
    assert len(rebuilt) == len(df)
    assert rebuilt["TransactionID"].is_unique

    # Chronological ordering across slices (train oldest, stream newest).
    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
    assert test["TransactionDT"].max() <= stream["TransactionDT"].min()


def test_split_sorts_input_by_transactiondt_internally() -> None:
    # Intentionally shuffled input must still be split chronologically.
    df = make_frame(n=90).sample(frac=1.0, random_state=1).reset_index(drop=True)
    train, test, stream = temporal_split_70_15_15(df)

    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
    assert test["TransactionDT"].max() <= stream["TransactionDT"].min()
    assert train["TransactionDT"].is_monotonic_increasing


def test_split_boundary_matches_exact_fraction_rows() -> None:
    # n=1000 -> 700 / 150 / 150 exactly.
    df = make_frame(n=1000)
    train, test, stream = temporal_split_70_15_15(df)
    assert (len(train), len(test), len(stream)) == (700, 150, 150)


def test_split_accepts_custom_fractions() -> None:
    df = make_frame(n=100)
    train, test, stream = temporal_split_70_15_15(df, train_frac=0.5, test_frac=0.3)
    assert (len(train), len(test), len(stream)) == (50, 30, 20)


def test_split_rejects_fractions_that_leave_no_stream() -> None:
    df = make_frame(n=100)
    with pytest.raises(ValueError, match="leave a production stream"):
        temporal_split_70_15_15(df, train_frac=0.7, test_frac=0.3)
    with pytest.raises(ValueError, match="leave a production stream"):
        temporal_split_70_15_15(df, train_frac=1.0, test_frac=0.0)


def test_split_requires_transactiondt_column() -> None:
    df = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0]})
    with pytest.raises(ValueError, match="TransactionDT"):
        temporal_split_70_15_15(df)
