"""Tests for the batch scorer (ticket 05).

Seam: the batch serving surface — ``score_csv`` and its CLI (``python -m
ieee_cis_fraud_detection.serving.batch``) — a thin adapter over the scoring &
decision boundary (Seam 1, ticket 03). Hermetic tests inject a tiny
``ScoringBoundary`` so no model is loaded; one integration smoke test runs the
CLI on the committed 218-column fixture through the real champion and asserts
the output columns and that the scores reproduce the committed reference. The
drift current-window store is exercised through a tmp path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import make_boundary
from ieee_cis_fraud_detection.monitoring.drift_store import read_store
from ieee_cis_fraud_detection.serving.batch import OUTPUT_COLUMNS, BatchError, app, score_csv
from ieee_cis_fraud_detection.serving.scoring import ContractError

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RUNNER = CliRunner()


def write_input_csv(path: Path, n: int = 3) -> pd.DataFrame:
    """A CSV with TransactionID + the tiny contract columns."""
    frame = pd.DataFrame(
        {
            "TransactionID": [1000 + i for i in range(n)],
            "TransactionDT": [100 + i for i in range(n)],
            "amount": [10.0 + i for i in range(n)],
            "cat_a": [("W", "H", "C")[i % 3] for i in range(n)],
            "cat_b": [("a", "b", "c")[i % 3] for i in range(n)],
        }
    )
    frame.to_csv(path, index=False)
    return frame


# --------------------------------------------------------------------------- #
# CSV in (218 cols) -> CSV out with TransactionID, score, decision
# --------------------------------------------------------------------------- #


def test_score_csv_writes_transactionid_score_decision(tmp_path: Path) -> None:
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    write_input_csv(input_path, n=3)

    out = score_csv(
        input_path,
        output_path,
        boundary=make_boundary(),
        drift_store_path=tmp_path / "store.csv",
    )

    # The returned frame carries exactly the contract output columns.
    assert list(out.columns) == ["TransactionID", "score", "decision"]
    assert list(out["TransactionID"]) == [1000, 1001, 1002]
    # score 0.42 < threshold 0.5 -> allow
    assert out["decision"].tolist() == ["allow", "allow", "allow"]
    np.testing.assert_allclose(out["score"], 0.42)

    # The CSV on disk has the same shape.
    written = pd.read_csv(output_path)
    assert list(written.columns) == list(OUTPUT_COLUMNS)
    assert len(written) == 3


def test_score_csv_decisions_flip_at_threshold(tmp_path: Path) -> None:
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    # Two rows: first below threshold, second at/above -> mixed decisions.
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 200],
            "amount": [10.0, 20.0],
            "cat_a": ["W", "H"],
            "cat_b": ["a", "b"],
        }
    )
    frame.to_csv(input_path, index=False)
    boundary = make_boundary(score_fn=lambda f: pd.Series([0.4, 0.5], index=f.index))

    out = score_csv(input_path, output_path, boundary=boundary)

    assert out["decision"].tolist() == ["allow", "block"]


def test_score_csv_rejects_missing_transactionid(tmp_path: Path) -> None:
    input_path = tmp_path / "in.csv"
    # No TransactionID: just the 4 feature columns.
    pd.DataFrame(
        {
            "TransactionDT": [100],
            "amount": [10.0],
            "cat_a": ["W"],
            "cat_b": ["a"],
        }
    ).to_csv(input_path, index=False)

    with pytest.raises(BatchError, match="TransactionID"):
        score_csv(input_path, tmp_path / "out.csv", boundary=make_boundary())


def test_score_csv_surfaces_contract_violation(tmp_path: Path) -> None:
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    frame = write_input_csv(input_path, n=1)
    frame["sneaky_extra"] = 0
    frame.to_csv(input_path, index=False)

    with pytest.raises(ContractError, match="sneaky_extra"):
        score_csv(input_path, output_path, boundary=make_boundary())


def test_score_csv_surfaces_nan_in_features(tmp_path: Path) -> None:
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    frame = write_input_csv(input_path, n=1)
    frame.loc[0, "amount"] = np.nan
    frame.to_csv(input_path, index=False)

    with pytest.raises(ContractError, match="NaN"):
        score_csv(input_path, output_path, boundary=make_boundary())


# --------------------------------------------------------------------------- #
# Drift current-window store: the honest data source for monitoring
# --------------------------------------------------------------------------- #


def test_drift_store_reads_empty_when_absent(tmp_path: Path) -> None:
    store = read_store(tmp_path / "nope.csv")
    assert list(store.columns) == ["TransactionID", "score", "decision"]
    assert len(store) == 0


def test_score_csv_appends_to_drift_store(tmp_path: Path) -> None:
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    store_path = tmp_path / "current_window.csv"
    write_input_csv(input_path, n=3)

    score_csv(input_path, output_path, boundary=make_boundary(), drift_store_path=store_path)

    store = read_store(store_path)
    assert list(store.columns) == ["TransactionID", "score", "decision"]
    assert list(store["TransactionID"]) == [1000, 1001, 1002]
    np.testing.assert_allclose(store["score"], 0.42)
    assert store["decision"].tolist() == ["allow", "allow", "allow"]


def test_drift_store_accumulates_across_batches(tmp_path: Path) -> None:
    store_path = tmp_path / "current_window.csv"
    input_a = tmp_path / "a.csv"
    input_b = tmp_path / "b.csv"
    write_input_csv(input_a, n=2)
    # A second, distinct batch of transactions.
    pd.DataFrame(
        {
            "TransactionID": [2000, 2001],
            "TransactionDT": [300, 400],
            "amount": [30.0, 40.0],
            "cat_a": ["C", "W"],
            "cat_b": ["c", "a"],
        }
    ).to_csv(input_b, index=False)

    score_csv(
        input_a,
        tmp_path / "a_out.csv",
        boundary=make_boundary(),
        drift_store_path=store_path,
    )
    score_csv(
        input_b,
        tmp_path / "b_out.csv",
        boundary=make_boundary(),
        drift_store_path=store_path,
    )

    # The store is a pure append log: both batches are present verbatim.
    store = read_store(store_path)
    assert list(store["TransactionID"]) == [1000, 1001, 2000, 2001]
    assert len(store) == 4


# --------------------------------------------------------------------------- #
# CLI: `python -m ieee_cis_fraud_detection.serving.batch`
# --------------------------------------------------------------------------- #


def test_cli_writes_output_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI loads the champion by default; stub only that heavy dependency so
    # the command under test runs hermetically against the tiny boundary.
    monkeypatch.setattr("ieee_cis_fraud_detection.serving.batch.load_model", make_boundary)
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    store_path = tmp_path / "current_window.csv"
    write_input_csv(input_path, n=2)

    result = RUNNER.invoke(
        app,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--drift-store",
            str(store_path),
        ],
    )

    assert result.exit_code == 0, result.output
    written = pd.read_csv(output_path)
    assert list(written.columns) == ["TransactionID", "score", "decision"]
    assert len(written) == 2


def test_cli_no_append_skips_drift_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ieee_cis_fraud_detection.serving.batch.load_model", make_boundary)
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    store_path = tmp_path / "current_window.csv"
    write_input_csv(input_path, n=2)

    result = RUNNER.invoke(
        app,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--drift-store",
            str(store_path),
            "--no-append",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not store_path.exists()


# --------------------------------------------------------------------------- #
# Real champion: CLI smoke test on the committed fixture (integration)
# --------------------------------------------------------------------------- #


def test_cli_smoke_scores_real_fixture(tmp_path: Path) -> None:
    """Running the CLI on the committed 218-column fixture reproduces the refs."""
    fixture = pd.read_csv(FIXTURES_DIR / "transactions.csv")
    expected = pd.read_csv(FIXTURES_DIR / "expected_scores.csv")
    assert len(fixture.columns) == 219  # TransactionID + 218 features

    input_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "scored.csv"
    store_path = tmp_path / "current_window.csv"
    fixture.to_csv(input_path, index=False)

    result = RUNNER.invoke(
        app,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--drift-store",
            str(store_path),
        ],
    )

    assert result.exit_code == 0, result.output
    scored = pd.read_csv(output_path)
    # The ticket's contract: output columns are TransactionID, score, decision.
    assert list(scored.columns) == ["TransactionID", "score", "decision"]
    assert len(scored) == len(expected)
    assert list(scored["TransactionID"]) == list(expected["TransactionID"])
    np.testing.assert_allclose(scored["score"], expected["score"], rtol=1e-6, atol=1e-9)
    assert scored["decision"].tolist() == expected["decision"].tolist()
    # And the scored output is the drift store's content.
    store = read_store(store_path)
    assert list(store.columns) == ["TransactionID", "score", "decision"]
    assert len(store) == len(expected)
