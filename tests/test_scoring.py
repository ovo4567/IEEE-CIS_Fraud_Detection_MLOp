"""Tests for the scoring & decision boundary (Seam 1, ticket 03).

Seam: `ieee_cis_fraud_detection.serving.scoring.ScoringBoundary` — the shared
interface the real-time API and the batch scorer are thin adapters over. It
enforces the 218-column feature contract (missing column, extra column, wrong
dtype, NaN all rejected with a precise `ContractError`), coerces the 9
categorical columns to `category`, scores through the pyfunc model, and turns
the score into a decision at the operating threshold.

Hermetic unit tests inject a tiny `ModelContract` + a stub score function; one
integration test scores the committed 218-column fixture through the real seed
champion and checks the scores reproduce the committed reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import CATEGORICAL_COLUMNS, THRESHOLD, make_boundary, make_frame
from ieee_cis_fraud_detection.serving.scoring import ContractError, ScoringBoundary, load_model

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------- #
# Contract enforcement: rejecting violating payloads
# --------------------------------------------------------------------------- #


def test_missing_column_rejected() -> None:
    boundary = make_boundary()
    frame = make_frame().drop(columns=["cat_b"])
    with pytest.raises(ContractError, match="missing.*cat_b"):
        boundary.score(frame)


def test_extra_column_rejected() -> None:
    boundary = make_boundary()
    frame = make_frame()
    frame["not_a_feature"] = 0
    with pytest.raises(ContractError, match="not_a_feature"):
        boundary.score(frame)


def test_wrong_dtype_rejected() -> None:
    boundary = make_boundary()
    frame = make_frame()
    frame["amount"] = frame["amount"].astype("object")
    with pytest.raises(ContractError, match="amount.*dtype"):
        boundary.score(frame)


def test_nan_rejected() -> None:
    boundary = make_boundary()
    frame = make_frame()
    frame.loc[0, "amount"] = np.nan
    with pytest.raises(ContractError, match="NaN.*amount"):
        boundary.score(frame)


def test_nan_in_categorical_column_rejected() -> None:
    boundary = make_boundary()
    frame = make_frame()
    frame.loc[1, "cat_a"] = np.nan
    with pytest.raises(ContractError, match="NaN.*cat_a"):
        boundary.score(frame)


def test_non_dataframe_rejected() -> None:
    boundary = make_boundary()
    with pytest.raises(ContractError, match="DataFrame"):
        boundary.score({"TransactionDT": 100, "amount": 10.0})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Category coercion and scoring
# --------------------------------------------------------------------------- #


def test_categoricals_coerced_to_category_before_scoring() -> None:
    seen: dict[str, dict[str, str]] = {}

    def score_fn(frame: pd.DataFrame) -> pd.Series:
        seen["dtypes"] = {c: frame[c].dtype.name for c in CATEGORICAL_COLUMNS}
        return pd.Series(0.1, index=frame.index)

    boundary = make_boundary(score_fn)
    boundary.score(make_frame())  # categoricals arrive as plain strings
    assert seen["dtypes"] == {"cat_a": "category", "cat_b": "category"}


def test_score_passthrough_and_output_columns() -> None:
    def score_fn(frame: pd.DataFrame) -> pd.Series:
        return pd.Series([0.2, 0.8], index=frame.index)

    boundary = make_boundary(score_fn)
    out = boundary.score(make_frame())
    assert list(out.columns) == ["score", "decision", "threshold"]
    assert out["score"].tolist() == [0.2, 0.8]
    assert out["threshold"].tolist() == [THRESHOLD, THRESHOLD]


def test_decision_flips_at_the_operating_threshold() -> None:
    def score_fn(frame: pd.DataFrame) -> pd.Series:
        return pd.Series([0.49, 0.50, 0.51], index=frame.index)

    boundary = make_boundary(score_fn)
    out = boundary.score(make_frame(n=3))
    # At exactly the threshold the transaction is blocked.
    assert out["decision"].tolist() == ["allow", "block", "block"]


def test_output_preserves_input_index() -> None:
    def score_fn(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(0.1, index=frame.index)

    boundary = make_boundary(score_fn)
    frame = make_frame()
    frame.index = ["txn-1", "txn-2"]
    out = boundary.score(frame)
    assert list(out.index) == ["txn-1", "txn-2"]


# --------------------------------------------------------------------------- #
# Real champion on the committed fixture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def champion_boundary() -> ScoringBoundary:
    """The committed seed champion, loaded once per module."""
    return load_model()


def test_score_correct_on_committed_fixture(champion_boundary: ScoringBoundary) -> None:
    """The boundary reproduces the committed reference scores (regression)."""
    fixture = pd.read_csv(FIXTURES_DIR / "transactions.csv")
    expected = pd.read_csv(FIXTURES_DIR / "expected_scores.csv")

    assert len(fixture) == len(expected) == 200
    X = fixture[list(champion_boundary.feature_columns)]
    assert X.shape == (200, 218)

    out = champion_boundary.score(X)
    np.testing.assert_allclose(
        out["score"].to_numpy(),
        expected["score"].to_numpy(),
        rtol=1e-6,
        atol=1e-9,
    )
    # Decisions reproduce the committed reference (pinning the threshold on
    # real data) and the fixture exercises both classes.
    assert out["decision"].tolist() == expected["decision"].tolist()
    assert set(expected["decision"]) == {"block", "allow"}


def test_contract_rejects_id_columns_besides_features(champion_boundary: ScoringBoundary) -> None:
    """The full fixture includes TransactionID — an extra column for the boundary."""
    fixture = pd.read_csv(FIXTURES_DIR / "transactions.csv")
    with pytest.raises(ContractError, match="TransactionID"):
        champion_boundary.score(fixture)
