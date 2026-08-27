"""Drift monitor (ticket 08).

The pure, infra-free drift computation that the monitoring flows (ticket 08's
``orchestration.monitoring``) wrap — mirroring :mod:`control_plane` for ticket
06. It builds the two halves of the **drift window** and turns them into the
aggregate drift alarm:

- :func:`slice_current_window` — the "current" half: the drift current-window
  store's scored rows joined back to their features, de-duplicated (a replayed
  stream keeps its newest score) and time-sliced to recent activity.
- :func:`build_reference` — the "reference" half: the training features scored
  through the model, with an optional size cap to keep the scheduled pass fast.
- :func:`compute_drift` — Evidently compares the two halves: per-feature drift
  **p-values** (K-S for numerical, chi-square for categorical — LOW means
  drifted, matching :func:`control_plane.decide_drift`) plus a **score
  distance** (the 2-sample Kolmogorov-Smirnov statistic — HIGH means drifted).
- :func:`evaluate_drift` — the aggregate alarm rule (>= 10% of features drift
  OR the score-distance threshold is crossed) via
  :func:`control_plane.decide_drift`.
- :func:`run_drift_monitor` — compute + evaluate in one call.

Deliberately no Prefect, no model loading, no network: the monitor takes plain
DataFrames in and returns drift statistics, the aggregate decision, and the
Evidently snapshot (for saving the report artifact). Evidently is imported
lazily (its import is heavy) so tests of everything except
:func:`compute_drift` never touch it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ieee_cis_fraud_detection.orchestration.control_plane import (
    DEFAULT_FEATURE_FRACTION_THRESHOLD,
    DEFAULT_FEATURE_PVALUE_THRESHOLD,
    DriftDecision,
    decide_drift,
)

# The drift-window schema (shared with the store + the batch scorer's output).
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"
SCORE_COLUMN = "score"

# The current window is "a time slice of recent stream activity" (spec, story
# 22): the most recent ``DEFAULT_WINDOW_ROWS`` scored transactions.
DEFAULT_WINDOW_ROWS = 5_000

# The reference is capped (subsampled) at this many rows so a scheduled drift
# pass stays fast; features and scores are subsampled together.
DEFAULT_REFERENCE_MAX_ROWS = 10_000

# The score leg of the alarm rule: the 2-sample KS statistic (0..1) at or above
# this counts as score drift.
DEFAULT_SCORE_DISTANCE_THRESHOLD = 0.1

# Evidently drift methods: forced to p-value tests so LOW always means drifted,
# regardless of row count (Evidently's auto-selection switches to a distance
# method for > 1000 rows, which would invert the semantics).
_NUMERIC_DRIFT_METHOD = "ks"
_CATEGORICAL_DRIFT_METHOD = "chisquare"


@dataclass(frozen=True)
class DriftMetrics:
    """The drift statistics the aggregate rule reads.

    ``feature_pvalues`` maps each feature column to its drift p-value (LOW =
    drifted). ``score_distance`` is the 2-sample KS statistic on the score
    column (HIGH = drifted).
    """

    feature_pvalues: dict[str, float]
    score_distance: float


@dataclass(frozen=True)
class DriftReport:
    """A full drift-monitoring result: statistics + decision + report snapshot."""

    metrics: DriftMetrics
    decision: DriftDecision
    snapshot: Any  # the Evidently Snapshot, saved as the human-readable report


# --------------------------------------------------------------------------- #
# The current-window half
# --------------------------------------------------------------------------- #


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} must have columns {list(columns)}; missing {missing}")


def _as_seconds(value: pd.Timedelta | float) -> float:
    """A window length (or any delta) as plain seconds for timestamp math."""
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return float(value)


def slice_current_window(
    store: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    window_rows: int = DEFAULT_WINDOW_ROWS,
    window_days: pd.Timedelta | float | None = None,
    now: float | None = None,
    id_column: str = ID_COLUMN,
    time_column: str = TIME_COLUMN,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    """The "current" half of the drift window: recent scored features + score.

    Joins the store's ``(id, score)`` rows back to ``features`` to recover the
    feature values of what was actually scored, de-duplicates a replayed stream
    (a transaction re-scored by a later pass keeps its newest score, matching
    the store's append-only contract), and time-slices to recent activity: the
    most recent ``window_rows`` transactions by ``time_column``, optionally
    further restricted to the last ``window_days`` before ``now`` (defaulting
    ``now`` to the newest timestamp in the joined data).

    Returns a frame with exactly ``feature_columns`` plus ``score_column``
    (the store's id/decision and any non-feature columns are dropped).

    Raises:
        ValueError: if ``features`` lacks ``time_column``, the store lacks the
            id/score columns, or ``window_rows``/``window_days`` are invalid.
    """
    _require_columns(features, [time_column], "features")
    _require_columns(store, [id_column, score_column], "store")
    if window_rows <= 0:
        raise ValueError(f"window_rows must be positive, got {window_rows!r}")
    if window_days is not None and _as_seconds(window_days) < 0:
        raise ValueError(f"window_days must be non-negative, got {window_days!r}")

    out_columns = [*feature_columns, score_column]
    if len(store) == 0:
        return pd.DataFrame(columns=out_columns)

    # The time column is often itself a feature column, so dedupe the selection.
    select_cols = list(dict.fromkeys([id_column, time_column, *feature_columns]))
    joined = store[[id_column, score_column]].merge(
        features[select_cols],
        on=id_column,
        how="inner",
    )
    # A replayed stream: keep the newest append per id (the store is append-only).
    joined = joined.drop_duplicates(subset=[id_column], keep="last")

    if window_days is not None:
        newest = float(now) if now is not None else float(joined[time_column].max())
        cutoff = newest - _as_seconds(window_days)
        joined = joined[joined[time_column] > cutoff]

    recent = joined.sort_values(time_column).tail(window_rows)
    return recent[out_columns].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The reference half
# --------------------------------------------------------------------------- #


def build_reference(
    features: pd.DataFrame,
    *,
    score_fn: Callable[[pd.DataFrame], pd.Series],
    max_rows: int | None = DEFAULT_REFERENCE_MAX_ROWS,
    score_column: str = SCORE_COLUMN,
) -> pd.DataFrame:
    """The "reference" half of the drift window: training features + score.

    Scores ``features`` through ``score_fn`` (the served model) and appends the
    result as ``score_column``. When ``max_rows`` is given and the frame is
    larger, it is subsampled deterministically so a scheduled drift pass stays
    fast — the cap applies to the scored frame, keeping features and scores
    aligned.
    """
    frame = features.copy()
    frame[score_column] = np.asarray(score_fn(features), dtype=float)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=0)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# compute_drift — Evidently feature p-values + score distance
# --------------------------------------------------------------------------- #


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _drift_method(series: pd.Series) -> str:
    """A p-value drift method for a column (LOW = drifted) by its dtype."""
    return _NUMERIC_DRIFT_METHOD if _is_numeric(series) else _CATEGORICAL_DRIFT_METHOD


def _safe_pvalue(value: Any) -> float:
    """A drift p-value coerced to ``[0, 1]``; a NaN/None result means no drift."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(v):
        return 1.0
    return float(min(max(v, 0.0), 1.0))


def _extract_feature_pvalues(
    metric_results: dict[str, Any],
    feature_columns: Sequence[str],
) -> dict[str, float]:
    """Per-feature drift p-values from an Evidently snapshot's metric results.

    Each ``ValueDrift`` result reports the column it measured in its metric
    params and its drift score (the p-value) in ``value``. A feature Evidently
    did not measure (degenerate column) counts as no drift (p-value 1.0).
    """
    out: dict[str, float] = {}
    for result in metric_results.values():
        column = result.metric_value_location.metric.params.get("column")
        if column in feature_columns:
            out[column] = _safe_pvalue(result.value)
    return {column: out.get(column, 1.0) for column in feature_columns}


def _score_distance(reference_scores, current_scores) -> float:
    """The 2-sample KS statistic between the score distributions (0..1, HIGH=drifted)."""
    a = np.asarray(reference_scores, dtype=float)
    b = np.asarray(current_scores, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0  # insufficient data to measure a distance
    return float(ks_2samp(a, b).statistic)


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    score_column: str = SCORE_COLUMN,
) -> tuple[DriftMetrics, Any]:
    """Compare the reference and current halves with Evidently.

    Returns ``(metrics, snapshot)``:

    - ``metrics.feature_pvalues`` — per-feature drift p-values (K-S for
      numerical columns, chi-square for categorical; LOW = drifted).
    - ``metrics.score_distance`` — the 2-sample KS statistic on the score
      column (HIGH = drifted).
    - ``snapshot`` — the Evidently ``Snapshot``, saved by the caller as the
      human-readable drift report (HTML/JSON).

    Evidently is imported lazily here (it is heavy) so the rest of the module
    stays light to import.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift

    feature_columns = [c for c in reference.columns if c != score_column]
    numerical = [c for c in feature_columns if _is_numeric(reference[c])] + [score_column]
    categorical = [c for c in feature_columns if not _is_numeric(reference[c])]
    data_definition = DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
    )

    metrics_list = [
        ValueDrift(column=column, method=_drift_method(reference[column]))
        for column in feature_columns
    ]
    # The score column joins the report too (its panel is part of the artifact);
    # the alarm reads the KS statistic below, not this panel's value.
    metrics_list.append(ValueDrift(column=score_column))

    report = Report(metrics_list)
    snapshot = report.run(
        Dataset.from_pandas(reference, data_definition=data_definition),
        Dataset.from_pandas(current, data_definition=data_definition),
    )

    pvalues = _extract_feature_pvalues(snapshot.metric_results, feature_columns)
    metrics = DriftMetrics(
        feature_pvalues=pvalues,
        score_distance=_score_distance(reference[score_column], current[score_column]),
    )
    return metrics, snapshot


# --------------------------------------------------------------------------- #
# The aggregate alarm rule
# --------------------------------------------------------------------------- #


def evaluate_drift(
    metrics: DriftMetrics,
    *,
    score_distance_threshold: float = DEFAULT_SCORE_DISTANCE_THRESHOLD,
    feature_pvalue_threshold: float = DEFAULT_FEATURE_PVALUE_THRESHOLD,
    feature_fraction_threshold: float = DEFAULT_FEATURE_FRACTION_THRESHOLD,
) -> DriftDecision:
    """The aggregate drift alarm over the computed statistics.

    Delegates to :func:`control_plane.decide_drift`: a feature is drifted when
    its p-value is below ``feature_pvalue_threshold``; the feature leg alarms
    when the drifted fraction is at least ``feature_fraction_threshold``
    (default 10%); the score leg alarms when ``score_distance`` is at or above
    ``score_distance_threshold``; the overall rule is an OR of the two legs.
    """
    return decide_drift(
        list(metrics.feature_pvalues.values()),
        score_distance=metrics.score_distance,
        score_distance_threshold=score_distance_threshold,
        feature_pvalue_threshold=feature_pvalue_threshold,
        feature_fraction_threshold=feature_fraction_threshold,
    )


def run_drift_monitor(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    score_column: str = SCORE_COLUMN,
    score_distance_threshold: float = DEFAULT_SCORE_DISTANCE_THRESHOLD,
    feature_pvalue_threshold: float = DEFAULT_FEATURE_PVALUE_THRESHOLD,
    feature_fraction_threshold: float = DEFAULT_FEATURE_FRACTION_THRESHOLD,
) -> DriftReport:
    """Compute the drift statistics and the aggregate decision in one call."""
    metrics, snapshot = compute_drift(reference, current, score_column=score_column)
    decision = evaluate_drift(
        metrics,
        score_distance_threshold=score_distance_threshold,
        feature_pvalue_threshold=feature_pvalue_threshold,
        feature_fraction_threshold=feature_fraction_threshold,
    )
    return DriftReport(metrics=metrics, decision=decision, snapshot=snapshot)


def no_drift_report(
    feature_columns: Sequence[str],
    *,
    score_distance_threshold: float = DEFAULT_SCORE_DISTANCE_THRESHOLD,
    feature_pvalue_threshold: float = DEFAULT_FEATURE_PVALUE_THRESHOLD,
    feature_fraction_threshold: float = DEFAULT_FEATURE_FRACTION_THRESHOLD,
) -> DriftReport:
    """A no-drift report for an empty current window (nothing scored yet).

    With no scored window there is no evidence of drift: every feature reports
    a p-value of 1.0 and the score distance is 0.0, so the aggregate rule never
    alarms. The snapshot is ``None`` — there is no Evidently run to save.
    """
    metrics = DriftMetrics(
        feature_pvalues={column: 1.0 for column in feature_columns},
        score_distance=0.0,
    )
    decision = evaluate_drift(
        metrics,
        score_distance_threshold=score_distance_threshold,
        feature_pvalue_threshold=feature_pvalue_threshold,
        feature_fraction_threshold=feature_fraction_threshold,
    )
    return DriftReport(metrics=metrics, decision=decision, snapshot=None)
