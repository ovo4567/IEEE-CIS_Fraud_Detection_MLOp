"""Monitoring flows (ticket 08).

The monitoring half of the closed loop, as Prefect flows:

- :func:`simulator_flow` — replays the production-stream slice through the
  real-time API at accelerated cadence for the live demo (story 15).
- :func:`monitoring_flow` — the scheduled drift-monitoring pass: batch-scores
  the next unseen chunk of the production stream (accumulating the drift
  current-window store), builds the training reference, time-slices the
  current window, runs the Evidently drift report, evaluates the aggregate
  alarm, and feeds the retraining trigger when it fires.

The pure, infra-free drift computation lives in
:mod:`ieee_cis_fraud_detection.monitoring.drift_monitor` (Seam 3, ticket 08);
this module wraps it in the Prefect flows plus the model/serving mechanics,
mirroring how :mod:`retraining` wraps :mod:`control_plane`. All steps are
callable as plain functions so the tests stay hermetic (no Prefect server, no
Docker, no network).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Annotated, Any

import httpx
from loguru import logger
import pandas as pd
from prefect import flow
import typer

from ieee_cis_fraud_detection.config import (
    DRIFT_REPORT_PATH,
    DRIFT_STORE_PATH,
)
from ieee_cis_fraud_detection.features import TRANSACTION_FEATURES_PATH
from ieee_cis_fraud_detection.modeling.split import temporal_split_70_15_15
from ieee_cis_fraud_detection.monitoring.drift_monitor import (
    DEFAULT_REFERENCE_MAX_ROWS,
    DEFAULT_SCORE_DISTANCE_THRESHOLD,
    DEFAULT_WINDOW_ROWS,
    DriftMetrics,
    build_reference,
    no_drift_report,
    run_drift_monitor,
    slice_current_window,
)
from ieee_cis_fraud_detection.monitoring.drift_store import (
    append_scores,
    read_store,
    scored_ids,
)
from ieee_cis_fraud_detection.orchestration.control_plane import (
    DEFAULT_FEATURE_FRACTION_THRESHOLD,
    DEFAULT_FEATURE_PVALUE_THRESHOLD,
    DriftDecision,
)
from ieee_cis_fraud_detection.orchestration.retraining import retraining_flow as _retraining_flow
from ieee_cis_fraud_detection.serving.batch import TRANSACTION_ID_COL, score_frame
from ieee_cis_fraud_detection.serving.scoring import ScoringBoundary, load_model

SIMULATOR_FLOW_NAME = "ieee-fraud-stream-simulator"
MONITORING_FLOW_NAME = "ieee-fraud-monitoring"

# The real-time API the simulator replays the stream through (the Compose stack
# exposes it at this service name, ticket 09; overridable for a local run).
DEFAULT_API_URL = "http://localhost:8000"

# The accelerated cadence: how long to pause between replayed requests so the
# ~30.8-day stream replays in demo time.
DEFAULT_CADENCE_SECONDS = 1.0

# How many stream rows a single monitoring pass batch-scores and appends to the
# drift store (each scheduled pass accumulates the window further).
DEFAULT_CHUNK_ROWS = 500

# Columns that are never features and must not cross the scoring contract or
# the API: the row key and the label.
NON_FEATURE_COLUMNS = ("TransactionID", "isFraud")


@dataclass(frozen=True)
class SimulationOutcome:
    """The result of a stream-simulator run."""

    replayed: int
    total: int


@dataclass(frozen=True)
class MonitoringOutcome:
    """The result of one scheduled drift-monitoring pass."""

    decision: DriftDecision
    metrics: DriftMetrics
    n_scored: int
    n_reference: int
    n_current: int
    report_path: Path | None
    triggered_retraining: bool


# --------------------------------------------------------------------------- #
# Stream simulator (live demo)
# --------------------------------------------------------------------------- #


@flow(
    name=SIMULATOR_FLOW_NAME,
    description="Replay the production stream through the real-time API.",
)
def simulator_flow(
    *,
    transaction: pd.DataFrame | None = None,
    features_path: Path = TRANSACTION_FEATURES_PATH,
    api_url: str = DEFAULT_API_URL,
    cadence_seconds: float = DEFAULT_CADENCE_SECONDS,
    max_transactions: int | None = None,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> int:
    """Replay the production stream through the real-time API at accelerated cadence.

    Chronologically splits the processed features and posts each transaction of
    the newest 15% (the production stream) to ``POST {api_url}/predict`` as its
    218 feature fields, pausing ``cadence_seconds`` between requests so the
    ~30.8-day stream replays in demo time and live scoring is visible. Returns
    the number of transactions replayed.

    ``client`` is an optional ``httpx.Client``-like object (tests inject a
    recorder); when omitted a real client is opened against ``api_url``.
    ``sleep_fn`` overrides the pause (tests inject a recorder); it defaults to
    ``time.sleep``. ``max_transactions`` caps the replay (e.g. a short demo).
    """
    if transaction is None:
        logger.info(f"Reading transactions from {features_path}")
        transaction = pd.read_parquet(features_path)
    _train, _test, stream = temporal_split_70_15_15(transaction)
    payload = stream.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in stream.columns])
    # The real-time API enforces the strict no-NaN contract (ticket 03), so the
    # simulator replays only contract-valid (fully populated) transactions for
    # live scoring; NaN-bearing rows (which the NaN-native model would still
    # score internally) are skipped rather than failing the demo with 400s.
    n_skipped = int(payload.isna().any(axis=1).sum())
    payload = payload.dropna()
    if n_skipped:
        logger.info(f"Simulator skipping {n_skipped} stream rows with NaN (API contract)")
    if max_transactions is not None:
        payload = payload.head(max_transactions)

    sleep = sleep_fn if sleep_fn is not None else time.sleep
    own_client = client is None
    if own_client:
        client = httpx.Client(base_url=api_url, timeout=10.0)
    try:
        # The pause is the inter-request gap, so no sleep before the first
        # request nor after the last.
        for index, row in enumerate(payload.to_dict("records")):
            if index > 0 and cadence_seconds:
                sleep(cadence_seconds)
            client.post("/predict", json=row).raise_for_status()
    finally:
        if own_client:
            client.close()

    logger.success(f"Replayed {len(payload)} transactions through {api_url}/predict")
    return len(payload)


# --------------------------------------------------------------------------- #
# The scheduled drift-monitoring pass
# --------------------------------------------------------------------------- #


@flow(
    name=MONITORING_FLOW_NAME,
    description="One scheduled drift-monitoring pass over the closed loop.",
)
def monitoring_flow(
    *,
    transaction: pd.DataFrame | None = None,
    features_path: Path = TRANSACTION_FEATURES_PATH,
    drift_store_path: Path = DRIFT_STORE_PATH,
    report_path: Path | None = DRIFT_REPORT_PATH,
    boundary: ScoringBoundary | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    window_days: pd.Timedelta | float | None = None,
    reference: pd.DataFrame | None = None,
    reference_max_rows: int = DEFAULT_REFERENCE_MAX_ROWS,
    score_distance_threshold: float = DEFAULT_SCORE_DISTANCE_THRESHOLD,
    feature_pvalue_threshold: float = DEFAULT_FEATURE_PVALUE_THRESHOLD,
    feature_fraction_threshold: float = DEFAULT_FEATURE_FRACTION_THRESHOLD,
    now: float | None = None,
    trigger_retraining: bool = True,
) -> MonitoringOutcome:
    """Run one scheduled drift-monitoring pass over the closed loop.

    1. Chronologically split the processed features 70/15/15; the newest 15% is
       the production stream.
    2. Batch-score the next unseen chunk of the stream through ``boundary`` and
       append the scored rows to the drift current-window store (accumulating
       the window).
    3. Build the training reference — the 70% train slice scored through the
       model, capped at ``reference_max_rows`` — or use the caller's
       precomputed ``reference`` (features + ``score``).
    4. Time-slice the store into the current window (join back to features,
       de-duplicate a replayed stream, keep the most recent ``window_rows``,
       optionally within ``window_days``).
    5. Run the Evidently drift report comparing current vs reference, save it
       as HTML at ``report_path``, and evaluate the aggregate alarm.
    6. When the alarm fires and ``trigger_retraining`` is set, feed the
       retraining trigger (``retraining_flow(drift_alarm=True)``).

    ``boundary`` defaults to the served champion scored on its NATIVE input
    space (``load_model(require_complete=False)``, so NaN-bearing stream rows
    are scored rather than rejected); tests inject a hermetic stub. Returns a
    :class:`MonitoringOutcome`. When no transactions have been scored yet the
    pass reports no drift (no alarm) and skips the report.
    """
    if transaction is None:
        logger.info(f"Reading transactions from {features_path}")
        transaction = pd.read_parquet(features_path)
    if boundary is None:
        # Score the model's NATIVE input space (NaN-native LightGBM champion,
        # ADR-0002): the drift window must reflect the real production stream,
        # not an imputed stand-in. The strict no-NaN contract (ticket 03) stays
        # the external serving surfaces' rule (API + batch CLI).
        boundary = load_model(require_complete=False)
    train_df, _test, stream = temporal_split_70_15_15(transaction)
    feature_columns = boundary.feature_columns

    # 2. Score the next unseen stream chunk -> drift store (accumulate window).
    store_df = read_store(drift_store_path)
    seen_ids = scored_ids(store_df)
    unseen = stream[~stream[TRANSACTION_ID_COL].isin(seen_ids)]
    chunk = unseen.head(chunk_rows)[[TRANSACTION_ID_COL, *feature_columns]]
    n_scored = len(chunk)
    if n_scored:
        scored = score_frame(chunk, boundary)
        append_scores(scored, drift_store_path)
        logger.info(
            f"Scored {n_scored} stream rows; "
            f"drift store now holds {len(read_store(drift_store_path))}"
        )

    # 3. Reference: the train slice scored through the model.
    if reference is None:
        reference_features = train_df[list(feature_columns)]
        reference = build_reference(
            reference_features,
            score_fn=lambda f: boundary.score(f)["score"],
            max_rows=reference_max_rows,
        )

    # 4. Current window: time-slice the store back to recent activity.
    current = slice_current_window(
        read_store(drift_store_path),
        transaction,
        feature_columns=feature_columns,
        window_rows=window_rows,
        window_days=window_days,
        now=now,
    )

    # 5. Drift report + aggregate alarm.
    if len(current) == 0:
        logger.info("No scored current window yet; reporting no drift")
        report = no_drift_report(
            feature_columns,
            score_distance_threshold=score_distance_threshold,
            feature_pvalue_threshold=feature_pvalue_threshold,
            feature_fraction_threshold=feature_fraction_threshold,
        )
        metrics, decision = report.metrics, report.decision
        saved_path = None
    else:
        report = run_drift_monitor(
            reference,
            current,
            score_distance_threshold=score_distance_threshold,
            feature_pvalue_threshold=feature_pvalue_threshold,
            feature_fraction_threshold=feature_fraction_threshold,
        )
        metrics, decision = report.metrics, report.decision
        saved_path = None
        if report_path is not None:
            saved_path = Path(report_path)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            report.snapshot.save_html(str(saved_path))
            report.snapshot.save_json(str(saved_path.with_suffix(".json")))
        logger.info(
            f"Drift: {decision.n_drifted}/{decision.n_features} features drifted, "
            f"score distance {metrics.score_distance:.4f} -> alarm={decision.alarm}"
        )

    # 6. Feed the retraining trigger on alarm.
    triggered = False
    if decision.alarm and trigger_retraining:
        logger.warning("Drift alarm -> triggering the retraining flow")
        _retraining_flow(drift_alarm=True)
        triggered = True

    return MonitoringOutcome(
        decision=decision,
        metrics=metrics,
        n_scored=n_scored,
        n_reference=len(reference),
        n_current=len(current),
        report_path=saved_path,
        triggered_retraining=triggered,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

app = typer.Typer()


@app.command()
def simulate(
    features_path: Annotated[Path, typer.Option()] = TRANSACTION_FEATURES_PATH,
    api_url: Annotated[str, typer.Option()] = DEFAULT_API_URL,
    cadence_seconds: Annotated[float, typer.Option()] = DEFAULT_CADENCE_SECONDS,
    max_transactions: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Replay the production stream through the real-time API once."""
    replayed = simulator_flow(
        features_path=features_path,
        api_url=api_url,
        cadence_seconds=cadence_seconds,
        max_transactions=max_transactions,
    )
    logger.success(f"Simulator replayed {replayed} transactions")


@app.command()
def monitor(
    features_path: Annotated[Path, typer.Option()] = TRANSACTION_FEATURES_PATH,
    drift_store_path: Annotated[Path, typer.Option()] = DRIFT_STORE_PATH,
    chunk_rows: Annotated[int, typer.Option()] = DEFAULT_CHUNK_ROWS,
    trigger_retraining: Annotated[bool, typer.Option()] = True,
) -> None:
    """Run one scheduled drift-monitoring pass."""
    outcome = monitoring_flow(
        features_path=features_path,
        drift_store_path=drift_store_path,
        chunk_rows=chunk_rows,
        trigger_retraining=trigger_retraining,
    )
    logger.success(
        f"Monitoring pass done: alarm={outcome.decision.alarm} "
        f"(scored {outcome.n_scored}, reference {outcome.n_reference}, "
        f"current {outcome.n_current}); triggered_retraining={outcome.triggered_retraining}"
    )


if __name__ == "__main__":
    app()
