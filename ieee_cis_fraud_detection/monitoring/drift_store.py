"""Drift current-window store (tickets 05/08).

The honest data source for monitoring: every batch-scored transaction lands
here as ``(TransactionID, score, decision)``. The monitoring stack (ticket 08)
time-slices this store into the drift window's "current" distribution and
compares it to the training reference.

The store is a pure append log: each scored batch is appended verbatim, so it
reflects exactly what was scored. Window semantics (time-slicing, de-duplicating
a replayed stream) belong to the drift monitor that reads it, not to the write
path — re-scoring a transaction never silently drops the newer score.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The canonical shape of a scored transaction, shared with the batch scorer's
# CSV output (ieee_cis_fraud_detection.serving.batch.OUTPUT_COLUMNS).
STORE_COLUMNS = ("TransactionID", "score", "decision")


def read_store(store_path: Path) -> pd.DataFrame:
    """The current store as a DataFrame (empty if it does not exist yet)."""
    if not Path(store_path).exists():
        return pd.DataFrame(columns=list(STORE_COLUMNS))
    return pd.read_csv(store_path)


def append_scores(scored: pd.DataFrame, store_path: Path) -> int:
    """Append ``(TransactionID, score, decision)`` rows to the store.

    Returns the number of rows appended. The store is created (with a header)
    on first append.
    """
    store_path = Path(store_path)
    rows = scored[list(STORE_COLUMNS)].copy()
    if len(rows) == 0:
        return 0
    store_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(store_path, mode="a", header=not store_path.exists(), index=False)
    return len(rows)


def scored_ids(store: pd.DataFrame, *, id_column: str = "TransactionID") -> set[int]:
    """The set of ``TransactionID``s present in a store frame (empty if none).

    The honest record of what has actually been scored, shared by the
    retraining corpus assembly and the monitoring flow so both read the same
    set of scored transactions.
    """
    if len(store) == 0 or id_column not in store.columns:
        return set()
    ids = pd.to_numeric(store[id_column], errors="coerce")
    return {int(x) for x in ids.dropna().unique()}
