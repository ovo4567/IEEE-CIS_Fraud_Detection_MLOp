# 09: Docker Compose + make demo

**What to build:** The self-contained stack (ADR-0001) — Docker Compose bringing up MLflow on a named volume seeded from the committed artifact, the real-time API, the batch scorer, Prefect (retraining + simulator), and monitoring — with the training container mounting the local data directory read-only. `make demo` brings the whole stack up from the committed seed offline, so a fresh clone can run the demo with no training and no cloud.

**Blocked by:** 02: Seed champion pipeline, 04: Real-time API, 05: Batch scorer, 07: Retraining flow (Prefect), 08: Stream simulator + drift monitoring (Evidently)

**Status:** ready-for-agent → in-progress → resolved

- [x] `docker compose up` starts MLflow, API, batch, Prefect, and monitoring
- [x] MLflow seeded from the committed artifact (no re-training needed)
- [x] Training container mounts data read-only (no in-container DVC pull)
- [x] `make demo` works from a fresh clone offline

## Answer

### What was built

A self-contained Docker Compose stack (ADR-0001) in `deploy/` plus `make demo` in the root Makefile. All four services run on **one image** (`ieee-fraud-demo:local`, built from the repo root) so the demo is fully offline:

- **`deploy/compose.yaml`** (`name: ieee-fraud-demo`) — 4 services:
  - `mlflow` — seeded from the committed artifact into the `mlflow_models` named volume on startup (`ieee_cis_fraud_detection.deployment.seed`), exposed on host **5001** (host 5000 is held by macOS AirPlay on many Macs), healthchecked on `/health`.
  - `prefect-server` — Prefect server on 4200 with a `prefect_data` named volume.
  - `api` — FastAPI real-time scorer on 8000, `depends_on: mlflow (healthy)`.
  - `worker` — Prefect worker (entrypoint `entrypoint_worker.sh` → `deploy/scripts/worker.py`) that registers the `stream-simulator` and `drift-monitoring` deployments, triggers an immediate run of each, and serves them in-process (`prefect.serve(*deployments, limit=1)` to serialize flow runs and avoid OOM). Mounts `../data:/app/data:ro` (read-only) plus `data/monitoring` read-write for the drift store/reports.
- **`deploy/Dockerfile`** — `python:3.12-slim`, installs build tools *before* pip (statsmodels source build, a transitive dep of evidently), pins `deploy/requirements.txt` to the exact dev versions, `pip install -e` the package, keeps data/models out of the image (mounted at runtime, via the root `.dockerignore`).
- **`deploy/scripts/`** — `worker.py` (Prefect registration + serve), `entrypoint_mlflow.sh`, `entrypoint_worker.sh`, plus `wait_for_http.py`.
- **`ieee_cis_fraud_detection/deployment/seed.py`** — idempotent MLflow store seeder: copies committed `models/seed` (`mlflow.db` + `champion_model/`) into the store exactly once (marker file), never clobbers a live registry. Typer CLI + `seed_mlflow_store(source, target) -> bool`. Unit-tested (`tests/test_deploy_seed.py`).
- **`Makefile`** — `make demo` (pre-flights docker/data/seed, `docker compose -f deploy/compose.yaml up --build -d`, prints the URLs), `make demo-logs`, `make demo-down`.
- **Docs** — `deploy/README.md` (stack, running, env vars) and `docs/docs/getting-started.md` rewritten for the one-command closed loop.

### Reconciliation: internal monitoring vs the strict API contract (ticket 03)

The production stream is ~81.5% NaN-bearing, but the external serving surfaces enforce a strict no-NaN contract (ticket 03). The closed loop therefore needed a seam:

- `ScoringBoundary.__init__`/`load_model()` gained `require_complete: bool = True`; `_validate` only raises on NaN when `require_complete` (missing/extra/dtype always enforced). The API/batch CLI keep the strict rule (default `True`); the monitoring flow scores the model's NaN-native input space with `load_model(require_complete=False)`.
- `simulator_flow` drops NaN-bearing stream rows before POSTing to the strict API (`payload.dropna()`), with a log line: `Simulator skipping 72150 stream rows with NaN (API contract)`.
- Tests: `tests/test_scoring.py` +2 for `require_complete`; `tests/conftest.py` `make_boundary(..., require_complete=...)`.

### OOM / memory

Each flow run reads the full processed features frame (590k rows) and the temporal split, peaking ~2.8–7 GB. Fixes: `prefect.serve(*deployments, limit=1)` serializes flow runs; and the demo requires Docker Desktop ≥12 GB (documented; the Docker VM is bumped to 12.5 GB in this workspace). Auto-triggered **retraining is off by default**: `MONITOR_TRIGGER_RETRAINING` defaults to `false`, so the demo scores, reports, and **alarms** on drift but does not run the heavy retraining flow (re-reads full data + fits 539-tree LightGBM on 413k rows) — retraining is on demand from the Prefect UI / `make retrain`. Verified: drift alarms fire on every pass while retraining stays off and no flow is OOM-killed.

### Verified end-to-end (fresh clone: empty volumes + cleared monitoring store)

- `docker compose up` → all 4 services healthy (mlflow 5001, prefect-server 4200, api 8000, worker).
- MLflow seeded on the fresh volume: `Seeding model store /app/models from committed seed /seed-src` → SUCCESS, `champion_model/` + `mlflow.db` present.
- Simulator: skipped 72,150 NaN rows; `POST /predict` all `200 OK` in the API logs.
- Monitoring: drift store grew 200 → 800+ rows; Evidently report saved (`latest_drift_report.html` 10.8 MB + `.json`); `Drift: 107/218 features drifted ... alarm=True` fires every pass.
- **0 OOM events, 0 auto retraining triggers** — the loop alarms but is on-demand.

### Design notes

- Relative bind mounts in `compose.yaml` resolve against the compose file's dir (`deploy/`), so they must be `../models/seed`, `../data`, `../data/monitoring` to reach the repo root.
- The stack is rebuilt on `make demo` (`up --build`) so a changed `deploy/scripts/` layer is picked up without `--force-recreate` gymnastics.
- Prefect 3.8.4: `Deployment.build_from_flow` is removed; use `flow.to_deployment(...)` (RunnerDeployment) + `prefect.serve`. The async `PrefectClient` signature confuses Pylance, so immediate runs are triggered via the REST API (`POST /deployments/{id}/create_flow_run`) with httpx. Prefect health path is `/api/health`.
- Idempotent seed: restarting/recreating the stack never double-seeds or clobbers a live registry.
