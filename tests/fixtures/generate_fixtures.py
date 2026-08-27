"""Generate the committed Seam-1 fixture (ticket 03).

Two committed files back the "score correct on a small committed fixture"
test:

- ``transactions.csv`` — 200 real, fully-populated rows sliced from the
  processed features (``TransactionID`` + the exact 218 feature columns in
  training order). "Fully-populated" means no NaN anywhere in the 218 features,
  so the rows are valid under the strict feature contract and can be scored.
- ``expected_scores.csv`` — the champion's fraud score and decision for each
  row (``TransactionID`` + ``score`` + ``decision``), the committed reference
  the boundary must reproduce. ``decision`` is computed directly from
  ``score >= threshold`` (independent of the boundary), so the reference pins
  the operating threshold on real data.

Regenerate after the champion changes (e.g. after a promotion):

    MLFLOW_LOG_UV_FILES=false .venv/bin/python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

from ieee_cis_fraud_detection.config import SEED_MODEL_PATH
from ieee_cis_fraud_detection.features import TRANSACTION_FEATURES_PATH
from ieee_cis_fraud_detection.serving.scoring import load_model

FIXTURES_DIR = Path(__file__).resolve().parent
N_ROWS = 200
RANDOM_STATE = 42


def main() -> None:
    transaction = pd.read_parquet(TRANSACTION_FEATURES_PATH)

    # The boundary exposes the champion contract through its public interface;
    # the raw pyfunc is loaded separately only to produce the independent
    # score reference.
    boundary = load_model()
    feature_columns = list(boundary.feature_columns)
    threshold = boundary.threshold
    champion = mlflow.pyfunc.load_model(str(SEED_MODEL_PATH))

    # Real rows that already satisfy the strict contract (no NaN in the 218
    # features) — sampled deterministically so the fixture is reproducible.
    populated = transaction[transaction[feature_columns].notna().all(axis=1)]
    sample = populated.sample(n=N_ROWS, random_state=RANDOM_STATE)

    # Input fixture: TransactionID + exactly the 218 feature columns, written
    # as CSV (categoricals become plain strings, so tests also exercise the
    # string -> `category` coercion path).
    input_frame = sample[["TransactionID"] + feature_columns]
    input_frame.to_csv(FIXTURES_DIR / "transactions.csv", index=False)

    # Reference: the champion's score + decision per row, scored from the
    # parquet slice (category dtypes intact); the boundary test re-scores the
    # CSV round-trip and must match to within float noise.
    score = champion.predict(sample[feature_columns])["score"]
    decision = np.where(score.to_numpy() >= threshold, "block", "allow")
    expected = pd.DataFrame(
        {
            "TransactionID": sample["TransactionID"].to_numpy(),
            "score": score.to_numpy(),
            "decision": decision,
        }
    )
    expected.to_csv(FIXTURES_DIR / "expected_scores.csv", index=False)

    print(
        f"Wrote {FIXTURES_DIR / 'transactions.csv'} "
        f"({len(sample)} rows x {len(input_frame.columns)} cols)"
    )
    print(f"Wrote {FIXTURES_DIR / 'expected_scores.csv'} ({len(expected)} rows)")
    print(
        f"Expected score range: [{score.min():.4f}, {score.max():.4f}] "
        f"threshold={threshold:.4f}"
    )


if __name__ == "__main__":
    main()
