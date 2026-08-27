"""Tests for the closed-loop control-plane logic (Seam 2, ticket 06).

Seam: `ieee_cis_fraud_detection.orchestration.control_plane` — the pure,
infra-free functions that encode the loop decisions:

- `build_retraining_corpus`: history + scored stream with labels revealed per
  the reveal lag (7 days by default, ADR-0003); stream rows newer than
  ``now - reveal_lag`` stay label-free and are excluded.
- `decide_promotion`: statistical gate on the shared test set (ADR-0004) —
  DeLong test for the AUC difference between champion and challenger; promote
  only when the challenger is significantly better at level ``alpha``.
- `decide_drift`: aggregate alarm rule — >= 10% of features drift OR the
  score-distance threshold is crossed.

Everything here is hermetic: no Prefect, no Docker, no model loading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ieee_cis_fraud_detection.orchestration.control_plane import (
    DriftDecision,
    PromotionDecision,
    build_retraining_corpus,
    decide_drift,
    decide_promotion,
)

# --------------------------------------------------------------------------- #
# build_retraining_corpus — reveal-lag cutoff
# --------------------------------------------------------------------------- #


def make_history(n: int = 3) -> pd.DataFrame:
    """A small labeled history frame (all labels known, all included)."""
    return pd.DataFrame(
        {
            "TransactionID": [100 + i for i in range(n)],
            "TransactionDT": [1_000 * (i + 1) for i in range(n)],
            "amount": [10.0 + i for i in range(n)],
            "isFraud": [0, 1, 0][:n],
        }
    )


def make_stream(n: int = 4, *, start_dt: int = 10_000) -> pd.DataFrame:
    """A scored-stream frame with labels present (revealed per the lag)."""
    return pd.DataFrame(
        {
            "TransactionID": [200 + i for i in range(n)],
            "TransactionDT": [start_dt + i * 100 for i in range(n)],
            "amount": [20.0 + i for i in range(n)],
            "isFraud": [0, 0, 1, 0][:n],
        }
    )


def test_corpus_is_history_plus_only_revealed_stream_rows() -> None:
    # now = 10_300, lag = 1 day -> cutoff = 10_300 - 86_400 < 0, so NO stream
    # rows are old enough -> corpus is exactly the history.
    history = make_history()
    stream = make_stream()
    corpus = build_retraining_corpus(history, stream, now=10_300)

    assert len(corpus) == len(history)
    assert set(corpus["TransactionID"]) == set(history["TransactionID"])


def test_reveal_lag_cutoff_reveals_rows_older_than_lag() -> None:
    # now = 10_300, lag = 200 -> cutoff = 10_100: stream rows at dt <= 10_100
    # (10_000, 10_100) are revealed; rows 10_200, 10_300 stay label-free.
    history = make_history()
    stream = make_stream()
    corpus = build_retraining_corpus(
        history, stream, reveal_lag=pd.Timedelta(seconds=200), now=10_300
    )

    revealed_ids = {200, 201}
    assert len(corpus) == len(history) + len(revealed_ids)
    assert set(corpus["TransactionID"]) == set(history["TransactionID"]) | revealed_ids


def test_reveal_cutoff_boundary_is_inclusive() -> None:
    # A row exactly at now - reveal_lag has served its full lag -> revealed.
    history = make_history()
    stream = make_stream(start_dt=10_000)
    now = 10_200
    lag = 100  # cutoff = 10_100
    corpus = build_retraining_corpus(
        history, stream, reveal_lag=pd.Timedelta(seconds=lag), now=now
    )
    # Rows at 10_000 and 10_100 (exactly the cutoff) are in; 10_200+ are not.
    assert set(corpus["TransactionID"]) == {100, 101, 102, 200, 201}


def test_default_now_is_newest_timestamp_in_data() -> None:
    # Without an explicit `now`, the newest transaction in the combined data is
    # the reference: with a 200-day lag nothing in the stream is revealed.
    history = make_history()
    stream = make_stream()
    corpus = build_retraining_corpus(history, stream, reveal_lag=pd.Timedelta(days=200))
    assert len(corpus) == len(history)


def test_zero_lag_reveals_the_whole_stream() -> None:
    # reveal_lag = 0 -> cutoff = now -> every stream row (all <= now) revealed.
    history = make_history()
    stream = make_stream()
    corpus = build_retraining_corpus(
        history, stream, reveal_lag=pd.Timedelta(seconds=0), now=10_300
    )
    assert len(corpus) == len(history) + len(stream)
    assert set(corpus["TransactionID"]) == set(history["TransactionID"]) | set(
        stream["TransactionID"]
    )


def test_history_rows_are_never_dropped() -> None:
    # History rows are always in the corpus regardless of their timestamp
    # relative to the reveal cutoff.
    history = make_history()
    stream = make_stream()
    corpus = build_retraining_corpus(history, stream, reveal_lag=pd.Timedelta(days=7), now=0)
    assert set(corpus["TransactionID"]) == set(history["TransactionID"])


def test_reveal_lag_accepts_plain_seconds() -> None:
    # The reveal lag may also be given as plain seconds (a config value), not
    # only as a pd.Timedelta: same cutoff as the Timedelta(seconds=200) case.
    history = make_history()
    stream = make_stream()
    corpus = build_retraining_corpus(history, stream, reveal_lag=200, now=10_300)
    assert set(corpus["TransactionID"]) == {100, 101, 102, 200, 201}


def test_reveal_lag_must_be_non_negative() -> None:
    history = make_history()
    stream = make_stream()
    with pytest.raises(ValueError, match="reveal_lag"):
        build_retraining_corpus(history, stream, reveal_lag=pd.Timedelta(days=-1), now=10_300)


def test_corpus_requires_label_column() -> None:
    history = make_history().drop(columns=["isFraud"])
    stream = make_stream()
    with pytest.raises(ValueError, match="isFraud"):
        build_retraining_corpus(history, stream, now=10_300)


def test_corpus_requires_time_column_in_both_frames() -> None:
    history = make_history()
    stream = make_stream().drop(columns=["TransactionDT"])
    with pytest.raises(ValueError, match="TransactionDT"):
        build_retraining_corpus(history, stream, now=10_300)


# --------------------------------------------------------------------------- #
# decide_promotion — DeLong statistical gate (ADR-0004)
# --------------------------------------------------------------------------- #


def test_significant_improvement_promotes() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([0, 1] * (n // 2))
    # Champion: random scores (AUC ~ 0.5). Challenger: perfect separation
    # (AUC ~ 1.0) -> the difference is unambiguously significant.
    champion = rng.uniform(0, 1, size=n)
    challenger = np.where(y == 1, 0.9, 0.1).astype(float)

    decision = decide_promotion(y, champion, challenger, alpha=0.05)

    assert isinstance(decision, PromotionDecision)
    assert decision.promote is True
    assert decision.auc_challenger > decision.auc_champion
    assert decision.p_value < 0.05
    assert decision.alpha == 0.05


def test_insignificant_difference_does_not_promote() -> None:
    rng = np.random.default_rng(1)
    n = 300
    y = np.array([0, 1] * (n // 2))
    base = np.where(y == 1, 1.0, 0.0) + rng.normal(0, 1.0, size=n)
    champion = base + rng.normal(0, 0.01, size=n)
    challenger = base + rng.normal(0, 0.01, size=n)

    decision = decide_promotion(y, champion, challenger, alpha=0.05)

    assert decision.promote is False
    assert decision.p_value > 0.05


def test_worse_challenger_is_never_promoted() -> None:
    n = 200
    y = np.array([0, 1] * (n // 2))
    champion = np.where(y == 1, 0.9, 0.1).astype(float)
    challenger = np.where(y == 1, 0.2, 0.8).astype(float)  # reversed: worse

    decision = decide_promotion(y, champion, challenger, alpha=0.05)

    assert decision.promote is False
    assert decision.auc_challenger < decision.auc_champion
    assert decision.p_value > 0.5


def test_identical_scores_do_not_promote() -> None:
    rng = np.random.default_rng(3)
    n = 100
    y = np.array([0, 1] * (n // 2))
    scores = rng.uniform(0, 1, size=n)

    decision = decide_promotion(y, scores, scores.copy(), alpha=0.05)

    assert decision.promote is False
    assert decision.auc_champion == decision.auc_challenger
    assert decision.p_value == 1.0


def test_alpha_tightens_the_gate() -> None:
    rng = np.random.default_rng(4)
    n = 200
    y = np.array([0, 1] * (n // 2))
    champion = rng.uniform(0, 1, size=n)
    challenger = np.where(y == 1, 0.9, 0.1).astype(float)

    # Significant at a normal level, but a strict enough alpha blocks it.
    assert decide_promotion(y, champion, challenger, alpha=0.05).promote is True
    assert decide_promotion(y, champion, challenger, alpha=1e-200).promote is False


def test_promotion_length_mismatch_raises() -> None:
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="same length"):
        decide_promotion(y, np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3, 0.4]))


def test_promotion_requires_both_classes_for_variance() -> None:
    y = np.zeros(10, dtype=int)  # no positives
    scores = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="positive"):
        decide_promotion(y, scores, scores.copy())


def test_promotion_requires_enough_of_each_class() -> None:
    y = np.array([0, 0, 1])  # only one positive
    scores = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="at least two"):
        decide_promotion(y, scores, scores.copy())


def test_promotion_alpha_must_be_in_open_unit_interval() -> None:
    rng = np.random.default_rng(5)
    n = 20
    y = np.array([0, 1] * (n // 2))
    scores = rng.uniform(0, 1, size=n)
    with pytest.raises(ValueError, match="alpha"):
        decide_promotion(y, scores, scores.copy(), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        decide_promotion(y, scores, scores.copy(), alpha=1.0)


# --------------------------------------------------------------------------- #
# decide_drift — aggregate alarm rule
# --------------------------------------------------------------------------- #


def test_no_drift_no_alarm() -> None:
    decision = decide_drift([0.4, 0.6, 0.8], score_distance=0.02, score_distance_threshold=0.5)

    assert decision.alarm is False
    assert decision.feature_alarm is False
    assert decision.score_alarm is False
    assert decision.n_drifted == 0
    assert decision.n_features == 3
    assert decision.feature_fraction == 0.0


def test_feature_fraction_at_or_above_threshold_alarms() -> None:
    # 2 of 20 features drift (10%) -> exactly the >= 10% threshold -> alarm.
    pvalues = [0.001, 0.002] + [0.5] * 18
    decision = decide_drift(pvalues, score_distance=0.0, score_distance_threshold=0.5)

    assert decision.alarm is True
    assert decision.feature_alarm is True
    assert decision.n_drifted == 2
    assert decision.n_features == 20
    assert decision.feature_fraction == pytest.approx(0.10)
    assert decision.score_alarm is False


def test_feature_fraction_below_threshold_no_alarm() -> None:
    # 1 of 20 features drifts (5%) and the score distance is below its
    # threshold -> no alarm.
    pvalues = [0.001] + [0.5] * 19
    decision = decide_drift(pvalues, score_distance=0.1, score_distance_threshold=0.5)

    assert decision.alarm is False
    assert decision.feature_alarm is False
    assert decision.score_alarm is False


def test_score_distance_crossing_threshold_alarms_on_its_own() -> None:
    # No feature drifts, but the score distance crosses its threshold.
    decision = decide_drift([0.5, 0.6], score_distance=0.8, score_distance_threshold=0.5)

    assert decision.alarm is True
    assert decision.feature_alarm is False
    assert decision.score_alarm is True


def test_score_distance_boundary_is_inclusive() -> None:
    # distance exactly at the threshold counts as crossed (>=).
    decision = decide_drift([0.9], score_distance=0.5, score_distance_threshold=0.5)
    assert decision.score_alarm is True
    assert decision.alarm is True


def test_score_distance_threshold_none_disables_score_leg() -> None:
    # With no score-distance threshold configured, only the feature rule runs.
    decision = decide_drift([0.5], score_distance=99.0, score_distance_threshold=None)
    assert decision.score_alarm is False
    assert decision.alarm is False


def test_empty_feature_list_never_feature_alarms() -> None:
    decision = decide_drift([], score_distance=0.0, score_distance_threshold=0.5)
    assert decision.feature_alarm is False
    assert decision.n_features == 0
    assert decision.feature_fraction == 0.0
    assert decision.alarm is False


def test_drift_decision_reports_breakdown() -> None:
    decision = decide_drift([0.001, 0.002, 0.9], score_distance=0.1, score_distance_threshold=0.05)
    assert isinstance(decision, DriftDecision)
    assert decision.n_drifted == 2
    assert decision.n_features == 3
    assert decision.feature_fraction == pytest.approx(2 / 3)
    assert decision.feature_alarm is True  # 66% >= 10%


def test_drift_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="feature_fraction_threshold"):
        decide_drift(
            [0.5], score_distance=0.0, score_distance_threshold=0.5, feature_fraction_threshold=1.5
        )
    with pytest.raises(ValueError, match="feature_pvalue_threshold"):
        decide_drift(
            [0.5], score_distance=0.0, score_distance_threshold=0.5, feature_pvalue_threshold=-0.1
        )
    with pytest.raises(ValueError, match="score_distance"):
        decide_drift([0.5], score_distance=-1.0, score_distance_threshold=0.5)
