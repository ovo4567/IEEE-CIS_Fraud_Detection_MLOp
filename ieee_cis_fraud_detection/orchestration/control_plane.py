"""Closed-loop control-plane logic (Seam 2, ticket 06).

The pure, infra-free functions that encode the retraining loop's decisions:

- :func:`build_retraining_corpus` — assemble the retraining corpus from the
  historical training data plus the scored production stream whose labels have
  been revealed per the reveal lag (ADR-0003).
- :func:`decide_promotion` — the statistical promotion gate (ADR-0004): a
  DeLong test for the champion-vs-challenger AUC difference on the shared test
  set; a challenger is promoted only when it is significantly better.
- :func:`decide_drift` — the aggregate drift alarm rule: >= 10% of features
  drift OR the score-distance threshold is crossed.

Deliberately no Prefect, no Docker, no model loading: these are testable
decisions that the Prefect flows (tickets 07/08) wrap, so the loop's rules can
be verified in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_REVEAL_LAG = pd.Timedelta(days=7)
DEFAULT_ALPHA = 0.05
DEFAULT_FEATURE_PVALUE_THRESHOLD = 0.05
DEFAULT_FEATURE_FRACTION_THRESHOLD = 0.10
TIME_COLUMN = "TransactionDT"
LABEL_COLUMN = "isFraud"


def _as_seconds(value: pd.Timedelta | float) -> float:
    """A reveal lag (or any delta) as plain seconds for timestamp math."""
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return float(value)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} must have columns {list(columns)}; missing {missing}")


# --------------------------------------------------------------------------- #
# Retraining corpus (reveal lag)
# --------------------------------------------------------------------------- #


def build_retraining_corpus(
    history: pd.DataFrame,
    scored_stream: pd.DataFrame,
    *,
    reveal_lag: pd.Timedelta | float = DEFAULT_REVEAL_LAG,
    now: float | None = None,
    time_column: str = TIME_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """All historical rows plus the scored-stream rows whose labels are revealed.

    A stream row's label is usable once it is older than the reveal lag: rows
    with ``time_column <= now - reveal_lag`` join the corpus with their labels;
    rows newer than the cutoff remain label-free and are excluded. ``now`` is
    the reference "current time" in the same units as ``time_column`` (seconds
    for ``TransactionDT``), defaulting to the newest timestamp in the combined
    data. Every ``history`` row is always included.

    Raises:
        ValueError: if either frame lacks the time/label columns, or
            ``reveal_lag`` is negative.
    """
    _require_columns(history, [time_column, label_column], "history")
    _require_columns(scored_stream, [time_column, label_column], "scored_stream")

    lag_seconds = _as_seconds(reveal_lag)
    if lag_seconds < 0:
        raise ValueError(f"reveal_lag must be non-negative, got {lag_seconds!r}")

    if now is None:
        newest = max(
            float(history[time_column].max()),
            float(scored_stream[time_column].max()),
        )
    else:
        newest = float(now)

    cutoff = newest - lag_seconds
    revealed = scored_stream[scored_stream[time_column] <= cutoff]
    return pd.concat([history, revealed], ignore_index=True)


# --------------------------------------------------------------------------- #
# Promotion gate (DeLong test, ADR-0004)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the champion-vs-challenger statistical gate."""

    promote: bool
    auc_champion: float
    auc_challenger: float
    p_value: float
    alpha: float


def decide_promotion(
    y_true,
    champion_scores,
    challenger_scores,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> PromotionDecision:
    """Whether a challenger is significantly better than the champion.

    Runs the DeLong test (DeLong et al., 1988) for the difference between two
    correlated AUCs on the shared test set (ADR-0004): the structural-
    components variance of ``AUC_challenger - AUC_champion``, turned into a
    one-sided z-test for "the challenger is better". The challenger is
    promoted only when ``p_value <= alpha``. Identical scores give ``se = 0``
    and ``p_value = 1`` (never promoted).

    Raises:
        ValueError: if the inputs differ in length, fewer than two rows of
            either class are present, or ``alpha`` is outside ``(0, 1)``.
    """
    y_true = np.asarray(y_true, dtype=int)
    champion = np.asarray(champion_scores, dtype=float)
    challenger = np.asarray(challenger_scores, dtype=float)
    if not (len(champion) == len(challenger) == len(y_true)):
        raise ValueError("y_true, champion_scores, and challenger_scores must be the same length")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    pos = y_true == 1
    neg = y_true == 0
    if not pos.any() or not neg.any():
        raise ValueError("the test set needs at least one positive and one negative label")
    if pos.sum() < 2 or neg.sum() < 2:
        raise ValueError("the DeLong test needs at least two positives and two negatives")

    auc_champion, auc_challenger, se = _delong_aucs(pos, neg, champion, challenger)

    diff = auc_challenger - auc_champion
    if se == 0.0 or diff <= 0.0:
        p_value = 1.0
    else:
        p_value = float(norm.sf(diff / se))  # one-sided: challenger better

    return PromotionDecision(
        promote=p_value <= alpha,
        auc_champion=auc_champion,
        auc_challenger=auc_challenger,
        p_value=p_value,
        alpha=alpha,
    )


def _delong_aucs(
    pos: np.ndarray,
    neg: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
) -> tuple[float, float, float]:
    """DeLong estimate of ``(AUC_A, AUC_B, se(AUC_B - AUC_A))``.

    The structural-components variance (DeLong et al., 1988) for two correlated
    ROC curves computed on the same cases and controls. ``pos``/``neg`` are the
    boolean masks for the positive and negative rows of the shared test set, so
    both scorers are indexed identically.
    """
    scores_champ_pos = champion[pos]
    scores_champ_neg = champion[neg]
    scores_chall_pos = challenger[pos]
    scores_chall_neg = challenger[neg]

    def placements(scores_pos, scores_neg) -> tuple[float, np.ndarray, np.ndarray]:
        """AUC plus the per-case/control placement values (U-statistic kernel)."""
        # P[i, j] = 1 if positive i outranks negative j, 0.5 on ties, else 0.
        kernel = (scores_pos[:, None] > scores_neg[None, :]).astype(float)
        kernel += 0.5 * (scores_pos[:, None] == scores_neg[None, :]).astype(float)
        auc = float(kernel.mean())
        placement_pos = kernel.mean(axis=1)  # how high each case ranks among controls
        placement_neg = kernel.mean(axis=0)  # how low each control ranks among cases
        return auc, placement_pos, placement_neg

    auc_a, pp_a, pn_a = placements(scores_champ_pos, scores_champ_neg)
    auc_b, pp_b, pn_b = placements(scores_chall_pos, scores_chall_neg)

    n = len(scores_champ_pos)  # number of positives (cases)
    m = len(scores_champ_neg)  # number of negatives (controls)

    def component(x, auc_x, y, auc_y, denom: int) -> float:
        """A DeLong structural component: deviation product over the placements.

        With ``x is y`` this is the within-model variance component (S_10 over
        the cases or S_01 over the controls); with ``x != y`` it is the
        cross-model covariance component shared by the two AUCs.
        """
        return float(((x - auc_x) * (y - auc_y)).sum() / denom)

    var_a = (
        component(pp_a, auc_a, pp_a, auc_a, n - 1) / n
        + component(pn_a, auc_a, pn_a, auc_a, m - 1) / m
    )
    var_b = (
        component(pp_b, auc_b, pp_b, auc_b, n - 1) / n
        + component(pn_b, auc_b, pn_b, auc_b, m - 1) / m
    )
    cov_ab = (
        component(pp_a, auc_a, pp_b, auc_b, n - 1) / n
        + component(pn_a, auc_a, pn_b, auc_b, m - 1) / m
    )
    se = float(np.sqrt(max(var_a + var_b - 2 * cov_ab, 0.0)))
    return auc_a, auc_b, se


# --------------------------------------------------------------------------- #
# Drift aggregate alarm rule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriftDecision:
    """Outcome of the aggregate drift alarm rule."""

    alarm: bool
    n_drifted: int
    n_features: int
    feature_fraction: float
    feature_alarm: bool
    score_alarm: bool


def decide_drift(
    feature_pvalues: Sequence[float],
    *,
    score_distance: float,
    score_distance_threshold: float | None = None,
    feature_pvalue_threshold: float = DEFAULT_FEATURE_PVALUE_THRESHOLD,
    feature_fraction_threshold: float = DEFAULT_FEATURE_FRACTION_THRESHOLD,
) -> DriftDecision:
    """Whether the aggregate drift rule fires for the constructed stats.

    A feature is drifted when its p-value is below ``feature_pvalue_threshold``
    (smaller p-value = more drift). The feature leg alarms when the drifted
    fraction is at least ``feature_fraction_threshold`` (default 10%). The
    score leg alarms when ``score_distance >= score_distance_threshold``; a
    ``None`` threshold disables the score leg. The overall rule is an OR of the
    two legs.

    Raises:
        ValueError: if ``feature_pvalue_threshold``/``feature_fraction_threshold``
            are outside ``[0, 1]``, ``score_distance`` is negative, or any
            p-value is outside ``[0, 1]``.
    """
    if not 0 <= feature_pvalue_threshold <= 1:
        raise ValueError(
            f"feature_pvalue_threshold must be in [0, 1], got {feature_pvalue_threshold!r}"
        )
    if not 0 <= feature_fraction_threshold <= 1:
        raise ValueError(
            f"feature_fraction_threshold must be in [0, 1], got {feature_fraction_threshold!r}"
        )
    if score_distance < 0:
        raise ValueError(f"score_distance must be non-negative, got {score_distance!r}")
    if any(not 0 <= p <= 1 for p in feature_pvalues):
        raise ValueError("feature p-values must lie in [0, 1]")

    n_features = len(feature_pvalues)
    n_drifted = sum(p < feature_pvalue_threshold for p in feature_pvalues)
    feature_fraction = n_drifted / n_features if n_features else 0.0

    feature_alarm = feature_fraction >= feature_fraction_threshold
    score_alarm = (
        score_distance_threshold is not None and score_distance >= score_distance_threshold
    )

    return DriftDecision(
        alarm=bool(feature_alarm or score_alarm),
        n_drifted=n_drifted,
        n_features=n_features,
        feature_fraction=float(feature_fraction),
        feature_alarm=bool(feature_alarm),
        score_alarm=bool(score_alarm),
    )
