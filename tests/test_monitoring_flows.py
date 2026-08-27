"""Tests for the monitoring flows (ticket 08).

Seam: `ieee_cis_fraud_detection.orchestration.monitoring` — the Prefect flows
that wrap the drift monitor (Seam 3):

- `simulator_flow`: replays the production stream through the real-time API at
  accelerated cadence for the live demo. An injected recording client keeps it
  hermetic.
- `monitoring_flow`: batch-scores the next unseen stream chunk (accumulating
  the drift current-window store), builds the training reference, time-slices
  the current window, runs the Evidently drift report, and on the aggregate
  alarm feeds the retraining trigger.

Hermetic: small synthetic frames, the tiny conftest boundary, tmp-path stores
and reports. The Evidently computation is stubbed out of the flow tests (it is
covered by `test_drift_monitor.py`); the retraining feed is stubbed too.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from conftest import make_boundary
import numpy as np
import pandas as pd

from ieee_cis_fraud_detection.monitoring.drift_monitor import DriftMetrics, DriftReport
from ieee_cis_fraud_detection.monitoring.drift_store import read_store
from ieee_cis_fraud_detection.orchestration.control_plane import DriftDecision
import ieee_cis_fraud_detection.orchestration.monitoring as monitoring_module
from ieee_cis_fraud_detection.orchestration.monitoring import (
    monitoring_flow,
    simulator_flow,
)

FEATURE_COLUMNS = ("TransactionDT", "amount", "cat_a", "cat_b")


def make_transaction(n: int = 300) -> pd.DataFrame:
    """A time-ordered transaction frame under the tiny conftest contract."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 100,
            "isFraud": rng.integers(0, 2, size=n),
            "amount": rng.uniform(0, 1000, size=n),
            "cat_a": rng.choice(["W", "H", "C"], size=n),
            "cat_b": rng.choice(["a", "b", "c"], size=n),
        }
    )


class RecordingClient:
    """A stand-in httpx client that records every ``post`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict | None = None, **kwargs):
        self.calls.append((url, json))
        return SimpleNamespace(raise_for_status=lambda: None)


def make_decision(alarm: bool) -> DriftDecision:
    if alarm:
        return DriftDecision(
            alarm=True,
            n_drifted=2,
            n_features=len(FEATURE_COLUMNS),
            feature_fraction=0.5,
            feature_alarm=True,
            score_alarm=False,
        )
    return DriftDecision(
        alarm=False,
        n_drifted=0,
        n_features=len(FEATURE_COLUMNS),
        feature_fraction=0.0,
        feature_alarm=False,
        score_alarm=False,
    )


def make_metrics() -> DriftMetrics:
    return DriftMetrics(
        feature_pvalues={c: 0.5 for c in FEATURE_COLUMNS},
        score_distance=0.02,
    )


class FakeSnapshot:
    """A stand-in Evidently snapshot that records saves to a file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved: list[str] = []

    def save_html(self, path: str) -> None:
        self.saved.append(path)
        Path(path).write_text("<html>drift report</html>")

    def save_json(self, path: str) -> None:
        self.saved.append(path)
        Path(path).write_text("{}")


def fake_report(alarm: bool, report_path: Path | None = None) -> DriftReport:
    snapshot = (
        FakeSnapshot(report_path)
        if report_path is not None
        else FakeSnapshot(Path("/tmp/none.html"))
    )
    return DriftReport(metrics=make_metrics(), decision=make_decision(alarm), snapshot=snapshot)


# --------------------------------------------------------------------------- #
# simulator_flow — replay the production stream through the API
# --------------------------------------------------------------------------- #


def test_simulator_replays_every_stream_transaction(monkeypatch) -> None:
    frame = make_transaction(n=300)  # train 0..209, test 210..254, stream 255..299
    recorder = RecordingClient()

    replayed = simulator_flow(transaction=frame, client=recorder, cadence_seconds=0)

    assert replayed == 45
    assert len(recorder.calls) == 45
    # Every call hits the /predict endpoint.
    assert all(url == "/predict" for url, _json in recorder.calls)
    # The stream slice (ids 255..299) is what was replayed, in time order.
    dts = [payload["TransactionDT"] for _url, payload in recorder.calls]
    assert dts == [i * 100 for i in range(255, 300)]  # dt(row i) = i*100
    assert dts == sorted(dts)


def test_simulator_payloads_are_the_feature_columns_only() -> None:
    frame = make_transaction(n=300)
    recorder = RecordingClient()

    simulator_flow(transaction=frame, client=recorder, cadence_seconds=0)

    for _url, payload in recorder.calls:
        assert set(payload) == set(FEATURE_COLUMNS)
        assert "TransactionID" not in payload
        assert "isFraud" not in payload


def test_simulator_respects_max_transactions() -> None:
    frame = make_transaction(n=300)
    recorder = RecordingClient()

    replayed = simulator_flow(
        transaction=frame, client=recorder, cadence_seconds=0, max_transactions=5
    )

    assert replayed == 5
    assert len(recorder.calls) == 5


def test_simulator_pauses_at_cadence() -> None:
    frame = make_transaction(n=300)
    recorder = RecordingClient()
    sleeps: list[float] = []

    simulator_flow(
        transaction=frame, client=recorder, cadence_seconds=0.25, sleep_fn=sleeps.append
    )

    # One pause after each replayed request except the last.
    assert sleeps == [0.25] * 44


# --------------------------------------------------------------------------- #
# monitoring_flow — the scheduled drift-monitoring pass
# --------------------------------------------------------------------------- #


def test_monitoring_flow_scores_next_unseen_chunk_into_store(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "store.csv"
    monkeypatch.setattr(
        monitoring_module, "run_drift_monitor", lambda *a, **k: fake_report(alarm=False)
    )

    outcome = monitoring_flow(
        transaction=make_transaction(n=300),
        drift_store_path=store,
        boundary=make_boundary(),
        chunk_rows=20,
        trigger_retraining=False,
    )

    assert outcome.n_scored == 20
    # The first 20 unseen stream rows (ids 255..274) landed in the store.
    stored = read_store(store)
    assert list(stored["TransactionID"]) == list(range(255, 275))
    assert stored["decision"].tolist() == ["allow"] * 20


def test_monitoring_flow_does_not_rescore_already_scored(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "store.csv"
    pd.DataFrame(
        {
            "TransactionID": list(range(255, 265)),  # 10 stream rows already scored
            "score": [0.3] * 10,
            "decision": ["allow"] * 10,
        }
    ).to_csv(store, index=False)
    monkeypatch.setattr(
        monitoring_module, "run_drift_monitor", lambda *a, **k: fake_report(alarm=False)
    )

    outcome = monitoring_flow(
        transaction=make_transaction(n=300),
        drift_store_path=store,
        boundary=make_boundary(),
        chunk_rows=20,
        trigger_retraining=False,
    )

    # The next 20 unseen rows (265..284) are scored, on top of the 10 already there.
    assert outcome.n_scored == 20
    stored = read_store(store)
    assert len(stored) == 30
    assert list(stored["TransactionID"]) == list(range(255, 285))


def test_monitoring_flow_builds_reference_and_current(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, pd.DataFrame] = {}
    frame = make_transaction(n=300)
    boundary = make_boundary()

    def capture(reference, current, **kwargs):
        captured["reference"] = reference
        captured["current"] = current
        return fake_report(alarm=False)

    monkeypatch.setattr(monitoring_module, "run_drift_monitor", capture)

    outcome = monitoring_flow(
        transaction=frame,
        drift_store_path=tmp_path / "store.csv",
        boundary=boundary,
        chunk_rows=20,
        trigger_retraining=False,
    )

    # Reference: the train slice's features + score (capped at default).
    assert set(captured["reference"].columns) == set(FEATURE_COLUMNS) | {"score"}
    assert len(captured["reference"]) == 210  # the whole 70% train slice
    assert captured["reference"]["score"].notna().all()
    # Current: the scored window's features + score.
    assert set(captured["current"].columns) == set(FEATURE_COLUMNS) | {"score"}
    assert len(captured["current"]) == 20
    # Outcome mirrors the stubbed report + counts.
    assert outcome.decision.alarm is False
    assert outcome.n_reference == 210
    assert outcome.n_current == 20
    assert outcome.triggered_retraining is False


def test_monitoring_flow_saves_the_report(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "reports" / "latest.html"
    monkeypatch.setattr(
        monitoring_module,
        "run_drift_monitor",
        lambda *a, **k: fake_report(alarm=False, report_path=report_path),
    )

    outcome = monitoring_flow(
        transaction=make_transaction(n=300),
        drift_store_path=tmp_path / "store.csv",
        boundary=make_boundary(),
        chunk_rows=10,
        report_path=report_path,
        trigger_retraining=False,
    )

    assert outcome.report_path == report_path
    assert report_path.exists()
    assert "drift report" in report_path.read_text()
    # The machine-readable JSON snapshot is saved next to the HTML report.
    assert report_path.with_suffix(".json").exists()


def test_monitoring_flow_alarm_triggers_retraining(tmp_path: Path, monkeypatch) -> None:
    called: list[dict] = []

    def fake_retraining(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(monitoring_module, "_retraining_flow", fake_retraining)
    monkeypatch.setattr(
        monitoring_module, "run_drift_monitor", lambda *a, **k: fake_report(alarm=True)
    )

    outcome = monitoring_flow(
        transaction=make_transaction(n=300),
        drift_store_path=tmp_path / "store.csv",
        boundary=make_boundary(),
        chunk_rows=10,
        trigger_retraining=True,
    )

    assert outcome.decision.alarm is True
    assert outcome.triggered_retraining is True
    assert called == [{"drift_alarm": True}]


def test_monitoring_flow_no_alarm_does_not_trigger_retraining(tmp_path: Path, monkeypatch) -> None:
    called: list[dict] = []
    monkeypatch.setattr(monitoring_module, "_retraining_flow", lambda **k: called.append(k))
    monkeypatch.setattr(
        monitoring_module, "run_drift_monitor", lambda *a, **k: fake_report(alarm=False)
    )

    outcome = monitoring_flow(
        transaction=make_transaction(n=300),
        drift_store_path=tmp_path / "store.csv",
        boundary=make_boundary(),
        chunk_rows=10,
        trigger_retraining=True,
    )

    assert outcome.decision.alarm is False
    assert outcome.triggered_retraining is False
    assert called == []


def test_monitoring_flow_empty_window_reports_no_drift(tmp_path: Path, monkeypatch) -> None:
    # chunk_rows=0 -> nothing is scored -> the current window is empty.
    called: list[dict] = []
    monkeypatch.setattr(monitoring_module, "_retraining_flow", lambda **k: called.append(k))

    outcome = monitoring_flow(
        transaction=make_transaction(n=300),
        drift_store_path=tmp_path / "missing_store.csv",
        boundary=make_boundary(),
        chunk_rows=0,
        trigger_retraining=True,
    )

    assert outcome.n_scored == 0
    assert outcome.n_current == 0
    assert outcome.decision.alarm is False
    assert outcome.report_path is None
    assert outcome.triggered_retraining is False
    assert called == []
