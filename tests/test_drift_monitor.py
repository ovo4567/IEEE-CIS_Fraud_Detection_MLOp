"""Tests for the drift monitor (Seam 3, ticket 08).

Seam: `ieee_cis_fraud_detection.monitoring.drift_monitor` — the pure,
infra-free drift computation the monitoring flows wrap (mirroring
`control_plane` for ticket 06):

- `slice_current_window`: the "current" half of the drift window — the store's
  scored rows joined back to their features, de-duplicated (a replayed stream
  keeps its newest score), time-sliced to recent activity.
- `build_reference`: the "reference" half — the training features scored
  through the model, with an optional size cap.
- `compute_drift`: runs Evidently (feature p-values, LOW = drifted) and a
  score-distance (2-sample KS statistic, HIGH = drifted).
- `evaluate_drift`: the aggregate alarm rule over the computed statistics
  (delegates to `control_plane.decide_drift`).
- `run_drift_monitor`: compute + evaluate in one call.

Hermetic: small synthetic frames; the Evidently integration is exercised in
two real-`compute_drift` tests, the rest avoid importing Evidently entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from ieee_cis_fraud_detection.monitoring.drift_monitor import (
    DriftMetrics,
    DriftReport,
    build_reference,
    compute_drift,
    evaluate_drift,
    no_drift_report,
    run_drift_monitor,
    slice_current_window,
)
from ieee_cis_fraud_detection.orchestration.control_plane import (
    DEFAULT_FEATURE_FRACTION_THRESHOLD,
    DEFAULT_FEATURE_PVALUE_THRESHOLD,
    DriftDecision,
    decide_drift,
)

FEATURE_COLUMNS = ("TransactionDT", "amount", "cat_a")


def make_features(n: int, *, start_dt: int = 0, dt_step: int = 100) -> pd.DataFrame:
    """A time-ordered transaction frame under a tiny 3-feature contract."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": [1000 + i for i in range(n)],
            "TransactionDT": [start_dt + i * dt_step for i in range(n)],
            "amount": rng.uniform(0, 1000, size=n),
            "cat_a": pd.Series(rng.choice(["W", "H", "C"], size=n), dtype="category"),
            "isFraud": rng.integers(0, 2, size=n),
        }
    )


def make_store(ids: list[int], scores: list[float] | None = None) -> pd.DataFrame:
    """A drift current-window store under the store's (id, score, decision) shape."""
    if scores is None:
        scores = [0.3] * len(ids)
    return pd.DataFrame(
        {
            "TransactionID": ids,
            "score": scores,
            "decision": ["allow"] * len(ids),
        }
    )


# --------------------------------------------------------------------------- #
# slice_current_window — the "current" half of the drift window
# --------------------------------------------------------------------------- #


def test_slice_returns_feature_columns_plus_score() -> None:
    features = make_features(5)
    store = make_store(ids=[1000, 1001, 1002], scores=[0.1, 0.2, 0.3])

    current = slice_current_window(
        store, features, feature_columns=FEATURE_COLUMNS, window_rows=100
    )

    assert list(current.columns) == ["TransactionDT", "amount", "cat_a", "score"]
    assert list(current["score"]) == [0.1, 0.2, 0.3]
    # The join recovered the features of the scored transactions.
    assert list(current["TransactionDT"]) == [0, 100, 200]


def test_slice_drops_store_and_non_feature_columns() -> None:
    features = make_features(3)
    store = make_store(ids=[1000, 1001, 1002])

    current = slice_current_window(
        store, features, feature_columns=FEATURE_COLUMNS, window_rows=100
    )

    assert "TransactionID" not in current.columns
    assert "decision" not in current.columns
    assert "isFraud" not in current.columns


def test_slice_de_duplicates_replayed_stream_keeping_newest_score() -> None:
    # A transaction re-scored by a later pass is appended again; the window
    # keeps the newest score (the store is a pure append log).
    features = make_features(2)
    store = make_store(ids=[1000, 1000, 1001], scores=[0.1, 0.9, 0.2])

    current = slice_current_window(
        store, features, feature_columns=FEATURE_COLUMNS, window_rows=100
    )

    # The replayed row (1000 scored twice) appears once, with its newest score.
    assert len(current) == 2
    assert list(current["score"]) == [0.9, 0.2]


def test_slice_takes_the_most_recent_window_rows() -> None:
    features = make_features(20)
    store = make_store(ids=list(range(1000, 1020)))

    current = slice_current_window(store, features, feature_columns=FEATURE_COLUMNS, window_rows=5)

    # The 5 most recent transactions by time (TransactionDT is a feature here).
    assert len(current) == 5
    assert set(current["TransactionDT"]) == {1500, 1600, 1700, 1800, 1900}


def test_slice_window_days_restricts_to_recent_activity() -> None:
    # dt = i*100 seconds; a 300s window with now = 1900 keeps rows dt > 1600.
    features = make_features(20)
    store = make_store(ids=list(range(1000, 1020)))

    current = slice_current_window(
        store,
        features,
        feature_columns=FEATURE_COLUMNS,
        window_days=pd.Timedelta(seconds=300),
        now=1_900,
    )

    assert len(current) == 3
    assert set(current["TransactionDT"]) == {1700, 1800, 1900}  # dt 1700, 1800, 1900


def test_slice_empty_store_returns_empty_frame() -> None:
    features = make_features(3)
    current = slice_current_window(
        pd.DataFrame(columns=["TransactionID", "score", "decision"]),
        features,
        feature_columns=FEATURE_COLUMNS,
        window_rows=100,
    )
    assert len(current) == 0
    assert list(current.columns) == ["TransactionDT", "amount", "cat_a", "score"]


def test_slice_unknown_ids_are_dropped() -> None:
    features = make_features(2)
    store = make_store(ids=[1000, 9999])  # 9999 is not in the features frame
    current = slice_current_window(
        store, features, feature_columns=FEATURE_COLUMNS, window_rows=100
    )
    assert current["TransactionDT"].tolist() == [0]  # only the known id survived
    assert current["score"].tolist() == [0.3]


# --------------------------------------------------------------------------- #
# build_reference — the "reference" half of the drift window
# --------------------------------------------------------------------------- #


def test_build_reference_appends_scores() -> None:
    features = make_features(4).drop(columns=["isFraud", "TransactionID"])
    reference = build_reference(
        features, score_fn=lambda f: pd.Series([0.5, 0.6, 0.7, 0.8], index=f.index)
    )
    assert list(reference.columns) == ["TransactionDT", "amount", "cat_a", "score"]
    assert list(reference["score"]) == [0.5, 0.6, 0.7, 0.8]
    # The features themselves are unchanged.
    np.testing.assert_allclose(reference["amount"], features["amount"])


def test_build_reference_caps_at_max_rows() -> None:
    features = make_features(50).drop(columns=["isFraud", "TransactionID"])
    reference = build_reference(
        features,
        score_fn=lambda f: pd.Series(0.5, index=f.index),
        max_rows=10,
    )
    assert len(reference) == 10
    # Features and scores stay aligned after the cap.
    assert set(reference["score"]) == {0.5}
    assert reference["score"].isna().sum() == 0


def test_build_reference_without_cap_keeps_all_rows() -> None:
    features = make_features(7).drop(columns=["isFraud", "TransactionID"])
    reference = build_reference(
        features,
        score_fn=lambda f: pd.Series(0.5, index=f.index),
        max_rows=None,
    )
    assert len(reference) == 7


# --------------------------------------------------------------------------- #
# compute_drift — Evidently feature p-values + score distance (real integration)
# --------------------------------------------------------------------------- #


def _drift_frame(n: int = 300, *, amount_shift: float = 0.0, score_shift: float = 0.0):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n) * 100,
            "amount": rng.normal(100 + amount_shift, 20, size=n),
            "cat_a": pd.Series(rng.choice(["W", "H", "C"], size=n), dtype="category"),
            "score": rng.uniform(0 + score_shift, 1 + score_shift, size=n).clip(0, 1),
        }
    )


def test_compute_drift_detects_drifted_features_and_score() -> None:
    reference = _drift_frame()
    current = _drift_frame(amount_shift=60, score_shift=0.5)

    metrics, snapshot = compute_drift(reference, current)

    # Feature p-values are keyed by feature column and LOW means drifted.
    assert set(metrics.feature_pvalues) == {"TransactionDT", "amount", "cat_a"}
    assert metrics.feature_pvalues["amount"] < DEFAULT_FEATURE_PVALUE_THRESHOLD
    # The score distance is the 2-sample KS statistic, HIGH means drifted.
    expected = ks_2samp(reference["score"], current["score"]).statistic
    assert metrics.score_distance == pytest.approx(expected)
    assert metrics.score_distance > 0.5
    # The snapshot can be persisted as the human-readable Evidently report.
    assert hasattr(snapshot, "save_html")
    assert hasattr(snapshot, "save_json")


def test_compute_drift_no_drift_when_distributions_match() -> None:
    # Identical distributions (here literally the same frame): no feature
    # p-value is below the threshold and the score distance is ~0.
    reference = _drift_frame()
    current = reference.copy()

    metrics, _snapshot = compute_drift(reference, current)

    assert metrics.feature_pvalues["amount"] > DEFAULT_FEATURE_PVALUE_THRESHOLD
    assert metrics.feature_pvalues["cat_a"] > DEFAULT_FEATURE_PVALUE_THRESHOLD
    assert metrics.score_distance == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# evaluate_drift — the aggregate alarm rule over computed statistics
# --------------------------------------------------------------------------- #


def test_evaluate_drift_no_drift_no_alarm() -> None:
    metrics = DriftMetrics(
        feature_pvalues={"a": 0.4, "b": 0.6, "c": 0.8},
        score_distance=0.02,
    )
    decision = evaluate_drift(metrics)
    assert isinstance(decision, DriftDecision)
    assert decision.alarm is False


def test_evaluate_drift_feature_fraction_alarms() -> None:
    # 2 of 20 features drifted (10%) -> exactly the >= 10% threshold.
    pvalues = {f"f{i}": 0.001 if i < 2 else 0.5 for i in range(20)}
    metrics = DriftMetrics(feature_pvalues=pvalues, score_distance=0.0)
    decision = evaluate_drift(metrics)
    assert decision.alarm is True
    assert decision.feature_alarm is True
    assert decision.feature_fraction == pytest.approx(0.10)


def test_evaluate_drift_score_distance_alarms_on_its_own() -> None:
    metrics = DriftMetrics(feature_pvalues={"a": 0.9, "b": 0.8}, score_distance=0.8)
    decision = evaluate_drift(metrics, score_distance_threshold=0.1)
    assert decision.alarm is True
    assert decision.score_alarm is True
    assert decision.feature_alarm is False


def test_evaluate_drift_matches_control_plane_rule() -> None:
    metrics = DriftMetrics(
        feature_pvalues={"a": 0.001, "b": 0.002, "c": 0.9},
        score_distance=0.05,
    )
    expected = decide_drift(
        [0.001, 0.002, 0.9],
        score_distance=0.05,
        score_distance_threshold=0.1,
        feature_pvalue_threshold=DEFAULT_FEATURE_PVALUE_THRESHOLD,
        feature_fraction_threshold=DEFAULT_FEATURE_FRACTION_THRESHOLD,
    )
    decision = evaluate_drift(metrics)
    assert decision == expected
    assert decision.alarm is True  # 2/3 features drifted


# --------------------------------------------------------------------------- #
# run_drift_monitor — compute + evaluate in one call
# --------------------------------------------------------------------------- #


def test_run_drift_monitor_computes_and_evaluates() -> None:
    reference = _drift_frame()
    current = _drift_frame(amount_shift=60, score_shift=0.5)

    report = run_drift_monitor(reference, current)

    assert isinstance(report, DriftReport)
    assert isinstance(report.metrics, DriftMetrics)
    assert isinstance(report.decision, DriftDecision)
    assert report.decision.alarm is True
    assert hasattr(report.snapshot, "save_html")


def test_no_drift_report_never_alarms_and_has_no_snapshot() -> None:
    # An empty current window means there is no evidence of drift: every
    # feature p-value is 1.0, the score distance is 0, and no report exists.
    report = no_drift_report(FEATURE_COLUMNS)

    assert isinstance(report, DriftReport)
    assert report.snapshot is None
    assert report.metrics.feature_pvalues == {c: 1.0 for c in FEATURE_COLUMNS}
    assert report.metrics.score_distance == 0.0
    assert report.decision.alarm is False
    assert report.decision.feature_alarm is False
    assert report.decision.score_alarm is False
