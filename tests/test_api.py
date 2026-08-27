"""Tests for the real-time API (ticket 04).

Seam: the FastAPI HTTP surface (``POST /predict``) via an in-process test
client — a thin adapter over the scoring & decision boundary (Seam 1, ticket
03). Hermetic tests inject a tiny ``ScoringBoundary`` into ``create_app()`` so
no model is loaded; one integration smoke test boots the app with the real
committed champion and scores a transaction from the committed fixture.

The feature contract is enforced entirely by the boundary, not the API: any
payload that deviates (missing column, extra column, wrong dtype, NaN) surfaces
as a precise HTTP 400 through the adapter.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from conftest import make_boundary
from ieee_cis_fraud_detection.serving.api import create_app
from ieee_cis_fraud_detection.serving.scoring import load_model

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def make_payload(**overrides: object) -> dict[str, object]:
    """A valid payload under the tiny contract."""
    payload: dict[str, object] = {
        "TransactionDT": 100,
        "amount": 10.0,
        "cat_a": "W",
        "cat_b": "a",
    }
    payload.update(overrides)
    return payload


def post(client: TestClient, payload: dict[str, object]):
    return client.post("/predict", json=payload)


def fixture_payload(row: pd.Series, feature_columns: tuple[str, ...]) -> dict[str, object]:
    """A JSON-encodable payload for one fixture row (numpy scalars -> Python)."""
    return {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in feature_columns}


# --------------------------------------------------------------------------- #
# Successful prediction: {score, decision, threshold}
# --------------------------------------------------------------------------- #


def test_predict_returns_score_decision_threshold() -> None:
    app = create_app(boundary=make_boundary())
    client = TestClient(app)
    resp = post(client, make_payload())
    assert resp.status_code == 200
    # score 0.42 < threshold 0.5 -> allow
    assert resp.json() == {"score": 0.42, "decision": "allow", "threshold": 0.5}


def test_predict_blocks_when_score_at_or_above_threshold() -> None:
    app = create_app(boundary=make_boundary(score_fn=lambda f: pd.Series(0.6, index=f.index)))
    resp = post(TestClient(app), make_payload())
    assert resp.status_code == 200
    assert resp.json()["decision"] == "block"


# --------------------------------------------------------------------------- #
# Contract violations surface as a precise 400
# --------------------------------------------------------------------------- #


def test_predict_returns_400_for_missing_column() -> None:
    app = create_app(boundary=make_boundary())
    payload = make_payload()
    del payload["cat_b"]
    resp = post(TestClient(app), payload)
    assert resp.status_code == 400
    assert "cat_b" in resp.json()["detail"]


def test_predict_returns_400_for_extra_column() -> None:
    app = create_app(boundary=make_boundary())
    resp = post(TestClient(app), make_payload(not_a_feature=0))
    assert resp.status_code == 400
    assert "not_a_feature" in resp.json()["detail"]


def test_predict_returns_400_for_wrong_dtype() -> None:
    app = create_app(boundary=make_boundary())
    resp = post(TestClient(app), make_payload(amount="ten"))
    assert resp.status_code == 400
    assert "amount" in resp.json()["detail"]


def test_predict_returns_400_for_nan() -> None:
    app = create_app(boundary=make_boundary())
    # JSON null in a categorical column is the NaN case a client can send.
    resp = post(TestClient(app), make_payload(cat_a=None))
    assert resp.status_code == 400
    assert "NaN" in resp.json()["detail"]
    assert "cat_a" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Real champion: in-process smoke test (integration)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def champion_app():
    """The app wired to the real committed champion, loaded once."""
    boundary = load_model()
    return create_app(boundary=boundary), boundary


def test_predict_smoke_scores_a_real_transaction(champion_app) -> None:
    """POSTing one committed fixture transaction reproduces its reference score."""
    app, boundary = champion_app
    fixture = pd.read_csv(FIXTURES_DIR / "transactions.csv")
    expected = pd.read_csv(FIXTURES_DIR / "expected_scores.csv")

    row = fixture.iloc[0]
    payload = fixture_payload(row, boundary.feature_columns)
    assert len(payload) == 218

    resp = post(TestClient(app), payload)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"score", "decision", "threshold"}
    np.testing.assert_allclose(
        body["score"], float(expected["score"].iloc[0]), rtol=1e-6, atol=1e-9
    )
    assert body["decision"] == expected["decision"].iloc[0]
    assert body["threshold"] == pytest.approx(boundary.threshold)


def test_predict_smoke_rejects_a_violating_payload(champion_app) -> None:
    """A payload with an extra column is rejected with a precise 400."""
    app, boundary = champion_app
    fixture = pd.read_csv(FIXTURES_DIR / "transactions.csv")
    row = fixture.iloc[0]
    payload = fixture_payload(row, boundary.feature_columns)
    payload["sneaky_extra"] = 0

    resp = post(TestClient(app), payload)
    assert resp.status_code == 400
    assert "sneaky_extra" in resp.json()["detail"]
