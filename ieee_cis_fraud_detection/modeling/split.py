"""Chronological 70/15/15 temporal split (ADR-0003).

The data is time-ordered by ``TransactionDT`` (a timedelta in seconds), so the
production-relevant split is chronological, never random:

- **Train** (70%): the oldest slice, used to fit the model.
- **Test** (15%): the next slice, used only for offline evaluation —
  operating-threshold selection and champion-vs-challenger comparison.
- **Production stream** (15%): the newest slice, held out label-free at serve
  time and replayed through the serving stack to drive the demo and drift
  monitoring.
"""

from __future__ import annotations

import pandas as pd

TRAIN_FRAC = 0.70
TEST_FRAC = 0.15
STREAM_FRAC = 0.15

_TIME_COLUMN = "TransactionDT"


def temporal_split_70_15_15(
    df: pd.DataFrame,
    *,
    train_frac: float = TRAIN_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a transaction frame chronologically into train/test/stream.

    Rows are sorted by ``TransactionDT`` (the input order is ignored), then cut
    at ``train_frac`` and ``train_frac + test_frac`` of the rows. Returns
    ``(train, test, stream)`` — three disjoint slices whose ordered
    concatenation is the sorted input.

    Raises:
        ValueError: if ``TransactionDT`` is missing, or the fractions leave no
            rows for the production stream.
    """
    if _TIME_COLUMN not in df.columns:
        raise ValueError(f"Missing time column {_TIME_COLUMN!r}; cannot split chronologically")
    if train_frac + test_frac >= 1.0:
        raise ValueError("train_frac + test_frac must be < 1.0 to leave a production stream")

    ordered = df.sort_values(_TIME_COLUMN).reset_index(drop=True)
    n = len(ordered)
    i_test = round(n * train_frac)
    i_stream = round(n * (train_frac + test_frac))
    return (
        ordered.iloc[:i_test],
        ordered.iloc[i_test:i_stream],
        ordered.iloc[i_stream:],
    )
