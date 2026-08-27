"""Prefect worker entrypoint for the self-contained demo stack (ticket 09).

Registers the two scheduled flows as Prefect deployments against the in-stack
Prefect server and serves them in-process (``prefect.serve``):

- ``stream-simulator`` — replays a bounded slice of the production stream
  through the real-time API at accelerated cadence (live demo, story 15);
- ``drift-monitoring`` — one scheduled drift pass: batch scores the next
  unseen stream chunk into the drift current-window store, builds the training
  reference, time-slices the current window, runs the Evidently drift report,
  evaluates the aggregate alarm, and feeds the retraining trigger on alarm
  (stories 13/14).

Both deployments get an interval schedule AND an explicit immediate run on
startup (via the Prefect REST API), so the demo starts scoring right away and
repeats on the interval. The retraining flow itself is triggered by the
monitoring flow on alarm, and remains re-triggerable from the Prefect UI.

The simulator targets the API at ``API_URL`` (default ``http://api:8000``, the
Compose service name behind ``DEFAULT_API_URL`` — ticket 08 handoff).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import os
import time
from uuid import UUID

import httpx
from loguru import logger
from prefect import serve
from prefect.client.schemas.schedules import IntervalSchedule

from ieee_cis_fraud_detection.orchestration.monitoring import (
    monitoring_flow,
    simulator_flow,
)

API_URL = os.environ.get("API_URL", "http://api:8000")
SIMULATOR_MAX_TRANSACTIONS = int(os.environ.get("SIMULATOR_MAX_TRANSACTIONS", "300"))
SIMULATOR_CADENCE_SECONDS = float(os.environ.get("SIMULATOR_CADENCE_SECONDS", "0.2"))
SIMULATOR_INTERVAL_SECONDS = int(os.environ.get("SIMULATOR_INTERVAL_SECONDS", "600"))
MONITOR_INTERVAL_SECONDS = int(os.environ.get("MONITOR_INTERVAL_SECONDS", "60"))
MONITOR_CHUNK_ROWS = int(os.environ.get("MONITOR_CHUNK_ROWS", "200"))
# The demo is turnkey: the monitoring pass scores, reports, and ALARMS on drift,
# but does NOT auto-run the heavy retraining flow (re-reads the full data + fits
# a 539-tree LightGBM on 413k rows) — that is triggered on demand from the
# Prefect UI / `make retrain`. Flip MONITOR_TRIGGER_RETRAINING=true to enable
# the automatic drift -> retrain loop.
MONITOR_TRIGGER_RETRAINING = os.environ.get("MONITOR_TRIGGER_RETRAINING", "false").lower() in (
    "1",
    "true",
    "yes",
)

_MAX_SERVE_RETRIES = 10
_RETRY_DELAY_SECONDS = 5.0


def _immediate_schedules(interval_seconds: int) -> list[IntervalSchedule]:
    """A single interval schedule anchored at now (first tick ~now, then each interval)."""
    return [
        IntervalSchedule(
            interval=timedelta(seconds=interval_seconds),
            anchor_date=datetime.now(UTC),
        )
    ]


def build_deployments() -> list:
    """The demo deployments: the stream simulator and the drift-monitoring pass."""
    return [
        simulator_flow.to_deployment(
            name="stream-simulator",
            schedules=_immediate_schedules(SIMULATOR_INTERVAL_SECONDS),
            parameters={
                "api_url": API_URL,
                "cadence_seconds": SIMULATOR_CADENCE_SECONDS,
                "max_transactions": SIMULATOR_MAX_TRANSACTIONS,
            },
            description=(
                "Replay a bounded slice of the production stream through the "
                "real-time API at accelerated cadence (live demo)."
            ),
        ),
        monitoring_flow.to_deployment(
            name="drift-monitoring",
            schedules=_immediate_schedules(MONITOR_INTERVAL_SECONDS),
            parameters={
                "chunk_rows": MONITOR_CHUNK_ROWS,
                "trigger_retraining": MONITOR_TRIGGER_RETRAINING,
            },
            description=(
                "Batch-score the next unseen stream chunk, run the Evidently "
                "drift report, and alarm (optionally feeding the retraining "
                "trigger)."
            ),
        ),
    ]


def _trigger_immediate_runs(deployment_ids: list[UUID]) -> None:
    """Create one run of each deployment right away, so the demo starts scoring
    immediately rather than waiting for the first schedule tick."""
    api_url = os.environ.get("PREFECT_API_URL", "http://prefect-server:4200/api").rstrip("/")
    with httpx.Client(timeout=10.0) as client:
        for deployment_id in deployment_ids:
            response = client.post(
                f"{api_url}/deployments/{deployment_id}/create_flow_run", json={}
            )
            response.raise_for_status()
            logger.info(f"Triggered an immediate run of deployment {deployment_id}")


def main() -> None:
    logger.info(
        "Registering demo deployments against the Prefect server: "
        f"simulator every {SIMULATOR_INTERVAL_SECONDS}s "
        f"(max {SIMULATOR_MAX_TRANSACTIONS} @ {SIMULATOR_CADENCE_SECONDS}s cadence), "
        f"monitoring every {MONITOR_INTERVAL_SECONDS}s (chunk {MONITOR_CHUNK_ROWS})"
    )
    deployments = build_deployments()

    # Register + trigger immediate runs EXACTLY once, retrying only until the
    # server accepts them. (A retry of `serve()` below must not re-fire the
    # immediate runs — that would enqueue duplicate runs on each attempt.)
    for attempt in range(1, _MAX_SERVE_RETRIES + 1):
        try:
            deployment_ids = [deployment.apply() for deployment in deployments]
            break
        except Exception as exc:  # pragma: no cover - server-availability retry
            logger.warning(f"registration attempt {attempt}/{_MAX_SERVE_RETRIES} failed: {exc}")
            if attempt == _MAX_SERVE_RETRIES:
                raise
            time.sleep(_RETRY_DELAY_SECONDS)
    _trigger_immediate_runs(deployment_ids)

    for attempt in range(1, _MAX_SERVE_RETRIES + 1):
        try:
            # limit=1 serializes the flow runs: each run reads the full
            # processed features (~2.8 GB peak with the temporal split), so
            # running the simulator + monitoring concurrently exceeds the
            # worker's memory and gets OOM-killed.
            serve(*deployments, limit=1)
            return  # serve() blocks until shutdown
        except Exception as exc:  # pragma: no cover - server-availability retry
            logger.warning(f"serve attempt {attempt}/{_MAX_SERVE_RETRIES} failed: {exc}")
            if attempt == _MAX_SERVE_RETRIES:
                raise
            time.sleep(_RETRY_DELAY_SECONDS)


if __name__ == "__main__":
    main()
