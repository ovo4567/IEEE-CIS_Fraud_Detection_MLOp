"""Shared hermetic scaffolding for the serving-seam test modules.

The scoring & decision boundary's tiny contract (4 features, 2 categoricals)
plus stub helpers under it. The three serving surfaces (scoring boundary, real-
time API, batch scorer) all exercise the same seam (Seam 1, ticket 03), so the
tiny contract is defined once here instead of being copy-pasted per module.
"""

from __future__ import annotations

import os

# The retraining-flow tests import Prefect, whose telemetry client prints a
# banner and spawns background threads at interpreter shutdown. Opt out before
# any test module imports it.
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
# Prefect 3.x starts an ephemeral subprocess server per flow call; keep its
# lifecycle logs (and a known shutdown-order logging error) out of test output.
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "CRITICAL")

import pandas as pd

from ieee_cis_fraud_detection.serving.scoring import ModelContract, ScoringBoundary

FEATURE_COLUMNS = ("TransactionDT", "amount", "cat_a", "cat_b")
CATEGORICAL_COLUMNS = ("cat_a", "cat_b")
THRESHOLD = 0.5


def make_contract() -> ModelContract:
    return ModelContract(
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        threshold=THRESHOLD,
    )


def make_boundary(score_fn=None) -> ScoringBoundary:
    """A hermetic boundary under the tiny contract, so no model is loaded."""
    if score_fn is None:
        score_fn = lambda frame: pd.Series(0.42, index=frame.index)
    return ScoringBoundary(score_fn=score_fn, contract=make_contract())


def make_frame(n: int = 2) -> pd.DataFrame:
    """A valid frame under the tiny contract (categoricals as plain strings)."""
    return pd.DataFrame(
        {
            "TransactionDT": [100 + i for i in range(n)],
            "amount": [10.0 + i for i in range(n)],
            "cat_a": [("W", "H", "C")[i % 3] for i in range(n)],
            "cat_b": [("a", "b", "c")[i % 3] for i in range(n)],
        }
    )
