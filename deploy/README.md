# deploy

Home of the self-contained local-Docker deployment of the MLOps stack
(ADR-0001): the Compose file and container image for MLflow (seeded from the
committed champion artifact), the real-time API, the batch scorer + monitoring
(scheduled Prefect flows), and the Prefect orchestrator. `make demo` brings the
whole stack up from the committed seed, offline — no cloud, no registry auth,
no re-training on a fresh clone.

## The stack

| Service          | What it runs                                                  | Port |
|------------------|---------------------------------------------------------------|------|
| `mlflow`         | MLflow tracking/registry server on a **named volume**, seeded on first boot from the committed seed (`models/seed`) — champion v1 in the UI, no re-training | 5001 (host) |
| `prefect-server` | Prefect server (UI + scheduling) on a named volume            | 4200 |
| `api`            | Real-time FastAPI surface (`POST /predict` → `{score, decision, threshold}`) | 8000 |
| `worker`         | A Prefect `serve` process running the scheduled flows: the **stream simulator** (replays the production stream through the API for the live demo) and the **drift-monitoring pass** (batch-scores the next chunk → drift store → Evidently report → alarm → retraining trigger) | — |

All four run the same image (`deploy/Dockerfile`) with different entrypoints;
only the *runtime* dependencies are installed (no notebooks/docs tooling).

**Where's the batch scorer?** The batch scorer (ticket 05) is not a standalone
always-on service — `docker compose up` starts MLflow, API, Prefect, and
monitoring, and the batch surface is the `drift-monitoring` pass inside the
worker (it batch-scores the next unseen stream chunk into the drift store on a
schedule). The batch CLI also remains available inside the image
(`python -m ieee_cis_fraud_detection.serving.batch`) for on-demand runs.

## How it stays self-contained (ADR-0001, ticket 09)

- **One shared model store.** `mlflow`, `api`, and `worker` all mount the same
  named volume (`mlflow_models`) at `/app/models`. On first boot the `mlflow`
  container seeds it from the committed `models/seed` (via
  `ieee_cis_fraud_detection.deployment.seed` — idempotent, never clobbers a
  live registry), so a fresh clone serves the champion with no re-training.
  Because the worker registers challengers into that same store and publishes
  promoted models to `/app/models/serving/champion_model`, the MLflow UI shows
  the same registry the API serves from — a promotion needs no redeploy.
- **Data mounted read-only.** `./data` is mounted `:ro` into the worker (the
  training container) — no in-container DVC pull. Only the runtime monitoring
  outputs (`./data/monitoring`: drift store, Evidently reports, retrain state)
  are mounted read-write, so the closed loop can accumulate.
- **Offline.** Images build locally; no cloud service, no GHCR auth, and no
  in-container DVC pull (data is mounted read-only). "Offline" means the
  *running stack* needs no training and no cloud; the repo-standard one-time
  data step (`features.py`, which materializes `data/processed/*.parquet`
  from the raw CSVs in `data/raw/`) remains the fresh-clone prerequisite,
  exactly as it is for every other `make` target. The committed seed
  (`models/seed`, already in git) means no re-training.

## CI/CD (ticket 10)

GitHub Actions publishes this image as a real artifact pipeline, but GHCR is
the *publish target, not the runtime* (ADR-0001) — `make demo` always builds
and runs locally. One workflow (`.github/workflows/ci.yml`) holds both jobs:

- **`ci`** (every push + PR): `uv sync --frozen` → `make lint` → `make contract`
  (the feature-contract check) → `make test` → `docker compose config --quiet`.
  Runs on committed inputs only — no DVC pull.
- **`cd`** (`needs: ci`, default-branch pushes only): builds the same
  `deploy/Dockerfile` and pushes to `ghcr.io/<owner>/ieee-fraud-serving` tagged
  `sha-<short>` (this commit) + `latest` — CD only publishes what CI validated.
- **`make contract`** — the feature-contract check: loads the committed seed
  artifact and asserts it carries the exact production contract (218 features /
  9 categoricals / threshold in (0,1)) and scores a contract-shaped row. Runs
  locally too: `uv run make contract`.

## Running the demo

```bash
# one-time prerequisites (repo standard): data + committed seed
# no DVC remote is configured — download from Kaggle and copy the two train
# CSVs into data/raw/ first (see the root README → "Get the dataset")
.venv/bin/python -m ieee_cis_fraud_detection.features  # build processed features (gitignored, only if data/processed is missing)
make seed          # only if models/seed is absent (it is committed, so usually not needed)

# bring the whole stack up from the committed seed, offline
make demo

# watch live scoring + monitoring passes
make demo-logs

# stop the stack
make demo-down
```

`make demo` pre-flights Docker, the processed features, and the committed seed,
then `docker compose -f deploy/compose.yaml up --build -d` and prints the URLs:

- **MLflow UI** — http://localhost:5001 (champion v1; challengers appear as the
  loop retrains; host 5001 → container 5000 because macOS AirPlay claims 5000)
- **Prefect UI** — http://localhost:4200 (the `stream-simulator` and
  `drift-monitoring` deployments; re-trigger the simulator or the monitoring
  pass from here)
- **API** — http://localhost:8000 (`POST /predict`; the simulator streams live
  scoring into the API logs)

The drift reports land in `data/monitoring/reports/latest_drift_report.html`
(and `.json`) — open the HTML in a browser to see feature + score drift of the
current window against the training reference.

Tune the demo via environment variables (e.g. `SIMULATOR_CADENCE_SECONDS=0.5
make demo`): `SIMULATOR_MAX_TRANSACTIONS`, `SIMULATOR_CADENCE_SECONDS`,
`SIMULATOR_INTERVAL_SECONDS`, `MONITOR_INTERVAL_SECONDS`, `MONITOR_CHUNK_ROWS`.

**Drift → retrain is off by default in the demo.** The monitoring pass scores,
reports, and *alarms* on drift (the committed seed's train slice genuinely
drifts from the production stream over time), but it does not auto-run the
heavy retraining flow (re-reads the full data + fits a 539-tree LightGBM on
~413k rows). Trigger a retrain on demand from the Prefect UI or `make retrain`;
set `MONITOR_TRIGGER_RETRAINING=true make demo` to enable the automatic
drift→retrain loop.

**Memory**: the flows replay/score the full 590k-row processed frame (a single
monitoring pass peaks ~7 GB), so give Docker Desktop ≥ 12 GB of memory
(Settings → Resources → Memory). The worker serializes flow runs (`limit=1`) so
concurrent runs never stack their peaks.

## Files

- `compose.yaml` — the stack (services, volumes, healthchecks, dependencies).
- `Dockerfile` — the single runtime image (deploy + `../` context; see the
  root `.dockerignore`).
- `requirements.txt` — the curated runtime dependencies.
- `scripts/` — container entrypoints: `entrypoint_mlflow.sh` (seed + serve),
  `entrypoint_worker.sh` (wait + serve the scheduled flows), `worker.py`
  (registers the deployments and serves them), `wait_for_http.py`.
- `ieee_cis_fraud_detection/deployment/seed.py` — the idempotent store-seeding
  rule (unit-tested in `tests/test_deploy_seed.py`).

Compose and the image are verified by *running* the stack (`make demo`), per
the master spec's testing decision.
