"""Batch serving surface (ticket 05).

A thin adapter over the scoring & decision boundary (Seam 1, ticket 03): a CLI
that scores a CSV of transactions (``TransactionID`` + the 218 feature columns)
through the boundary and writes a CSV out with ``TransactionID``, ``score``,
``decision``. The scored output is appended to the drift current-window store
(``ieee_cis_fraud_detection.monitoring.drift_store``), making batch scoring the
honest data source for drift monitoring. Runs in parallel with the real-time
API.

The feature contract is enforced by the boundary, not here — a violating CSV
(missing column, extra column, wrong dtype, or NaN) surfaces as a
:class:`~ieee_cis_fraud_detection.serving.scoring.ContractError`. The only
shape this adapter adds is the row key: the input CSV must carry a
``TransactionID`` column so scores can be written back per transaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import typer

from ieee_cis_fraud_detection.config import DRIFT_STORE_PATH
from ieee_cis_fraud_detection.monitoring.drift_store import STORE_COLUMNS, append_scores
from ieee_cis_fraud_detection.serving.scoring import ScoringBoundary, load_model

app = typer.Typer()

TRANSACTION_ID_COL = "TransactionID"
# The scored-row shape is the drift store's schema — the batch output and the
# store it feeds are the same record, defined once in monitoring.drift_store.
OUTPUT_COLUMNS = STORE_COLUMNS


class BatchError(ValueError):
    """The input CSV cannot be scored (e.g. it lacks a TransactionID column)."""


def score_frame(frame: pd.DataFrame, boundary: ScoringBoundary) -> pd.DataFrame:
    """Score a frame of ``TransactionID`` + 218 features -> ``{TransactionID, score, decision}``.

    ``TransactionID`` is the row key, not a feature, so it is set aside before
    the frame crosses the boundary (whose contract is exactly the 218 feature
    columns) and joined back onto the scores.
    """
    if TRANSACTION_ID_COL not in frame.columns:
        raise BatchError(
            f"input CSV must include the {TRANSACTION_ID_COL!r} column so "
            f"scores can be written back per transaction; found columns: "
            f"{list(frame.columns)}"
        )
    features = frame.drop(columns=[TRANSACTION_ID_COL])
    scored = boundary.score(features)
    return pd.DataFrame(
        {
            TRANSACTION_ID_COL: frame[TRANSACTION_ID_COL],
            "score": scored["score"],
            "decision": scored["decision"],
        },
        index=frame.index,
    )


def score_csv(
    input_path: Path,
    output_path: Path,
    *,
    boundary: ScoringBoundary | None = None,
    drift_store_path: Path | None = None,
) -> pd.DataFrame:
    """Score a CSV of transactions and write ``{TransactionID, score, decision}`` out.

    ``boundary`` defaults to the committed champion (``load_model()``). When
    ``drift_store_path`` is given, the scored rows are appended to the drift
    current-window store there — the CLI wires it to the real store, and
    library callers opt in explicitly so scoring has no hidden side effects.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame = pd.read_csv(input_path)
    resolved = boundary if boundary is not None else load_model()
    out = score_frame(frame, resolved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    if drift_store_path is not None:
        appended = append_scores(out, drift_store_path)
        logger.info(f"Appended {appended} rows to drift store {drift_store_path}")
    logger.success(f"Scored {len(out)} transactions -> {output_path}")
    return out


@app.command()
def main(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input CSV: TransactionID + 218 feature columns"),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output CSV path (TransactionID, score, decision)"),
    ],
    drift_store: Annotated[
        Path,
        typer.Option(
            "--drift-store", help="Drift current-window store to append the scored rows to"
        ),
    ] = DRIFT_STORE_PATH,
    no_append: Annotated[
        bool,
        typer.Option("--no-append", help="Do not append the scored rows to the drift store"),
    ] = False,
) -> None:
    """Score a CSV of transactions through the champion and write the decisions."""
    score_csv(input_path, output_path, drift_store_path=None if no_append else drift_store)


if __name__ == "__main__":
    app()
