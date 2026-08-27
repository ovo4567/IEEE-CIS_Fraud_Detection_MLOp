# 08: Stream simulator + drift monitoring (Evidently)

**What to build:** The monitoring half of the closed loop: a stream simulator (Prefect flow) that replays the production stream slice through the real-time API at accelerated cadence for the live demo, alongside scheduled batch scoring that accumulates a time-sliced drift window; Evidently reports comparing feature + score distributions of the current window against the training reference; the aggregate drift alarm feeds the retraining trigger.

**Blocked by:** 04: Real-time API, 05: Batch scorer, 06: Control-plane logic (Seam 2)

**Status:** resolved

- [x] Simulator replays the stream through the API (live scoring visible)
- [x] Scheduled batch scoring accumulates the time-sliced current window
- [x] Evidently report compares current window vs training reference (feature + score drift)
- [x] Aggregate drift alarm feeds the retraining trigger

## Answer

**Seam 3 — drift monitor (pure, mirrors `control_plane`/Seam 2)**
`ieee_cis_fraud_detection/monitoring/drift_monitor.py` — infra-free drift
computation the flows wrap:

- `slice_current_window(store, features, *, feature_columns, window_rows,
  window_days, now)` — the "current" half: joins the drift current-window
  store's `(TransactionID, score)` back to the features, de-duplicates a
  replayed stream (keeps the newest append), and time-slices to recent
  activity (most recent `window_rows`, optionally within `window_days`).
- `build_reference(features, *, score_fn, max_rows)` — the "reference" half:
  the training slice scored through the served model, deterministically capped
  at `DEFAULT_REFERENCE_MAX_ROWS` to keep the scheduled pass fast.
- `compute_drift(reference, current)` — Evidently (`ValueDrift`) per-feature
  drift **p-values** (K-S for numerical, chi-square for categorical; LOW =
  drifted, matching `decide_drift`) plus a **score distance** (the 2-sample KS
  statistic via scipy; HIGH = drifted); returns `(DriftMetrics, snapshot)` and
  imports Evidently lazily (heavy).
- `evaluate_drift(metrics, *, thresholds)` — the aggregate alarm rule
  (>= 10% features drift OR score-distance threshold crossed) via
  `control_plane.decide_drift`; `run_drift_monitor` = compute + evaluate.

**Flows (thin wrappers, mirror `retraining.py`)**
`ieee_cis_fraud_detection/orchestration/monitoring.py`:

- `simulator_flow` — splits the features 70/15/15, posts each transaction of
  the production stream to `POST {api_url}/predict` (218 feature fields,
  `TransactionID`/`isFraud` stripped) at `cadence_seconds` inter-request
  cadence for the live demo; injectable httpx-like `client` + `sleep_fn` keep
  tests hermetic.
- `monitoring_flow` — one scheduled pass: batch-scores the next unseen stream
  chunk through the boundary and appends it to the drift store (accumulating
  the window), builds the reference, time-slices the current window, runs the
  Evidently report (saved as HTML to `data/monitoring/reports/`), evaluates
  the aggregate alarm, and on alarm calls `retraining_flow(drift_alarm=True)`
  (`trigger_retraining`, default on). Empty window -> no-drift no-op.

**Config**: `DRIFT_REPORTS_DIR`, `DRIFT_REPORT_PATH`, `DRIFT_REPORT_JSON_PATH`
under `data/monitoring/`. Dep: `evidently>=0.7.21` added (0.7.x rewrite: no
classic `Report`/`DataDriftTable`; used `Dataset.from_pandas` + `Report([ValueDrift...])`
+ `snapshot.metric_results[mid].value`, methods forced to p-value tests so the
alarm semantics hold at any row count).

**Tests (TDD, 28 new, suite 132 passing)**: `tests/test_drift_monitor.py`
(window join/de-dup/time-slice, reference capping, evaluate/run wiring, two
real-Evidently integration tests) and `tests/test_monitoring_flows.py`
(simulator payloads/cadence/cap; monitoring pass scoring, window accumulation,
report save, alarm -> retraining feed, empty-window no-op). Lint (ruff) and
typecheck (Pylance) clean.

**Handoff to 09**: the flows are plain callable `@flow`s + CLIs/Makefile targets;
the actual *scheduling* (Prefect deployment/cron for `monitoring_flow` and
`simulator_flow` against the running stack) is wired by ticket 09's Compose
orchestration, as is the API service name behind `DEFAULT_API_URL`. Review
hardening: the empty-window no-drift semantics moved into the seam
(`no_drift_report`), scored-ID extraction deduplicated into
`drift_store.scored_ids` (shared with `retraining.scored_transaction_ids`), and
the monitoring pass now writes both the HTML and JSON Evidently report
(`DRIFT_REPORT_PATH` + its `.json` sibling).

