Status: ready-for-agent

# IEEE-CIS Fraud Detection — MLOps Deployment

## Problem Statement

The repo has a strong model (`finetuned_lgbm`, best val AUC 0.9210 on the old 80/20 split) but **no way to deploy it**: the model exists only as MLflow params, no train→register→serve path exists, the MLflow registry is empty, tests are a failing placeholder, and there is no serving, monitoring, or retraining infrastructure. The user wants a portfolio-grade MLOps deployment demonstrating the full loop — train, register, serve (real-time + batch), monitor for drift, and retrain on a trigger — reproducible locally via Docker, with CI/CD on GitHub.

## Solution

A local-Docker MLOps stack that ships a **reproducibly re-fit** `finetuned_lgbm` seed model and runs the full closed loop: a 70/15/15 temporal split reserves a **production stream** that is replayed through the serving stack to simulate live traffic, feed drift monitoring, and (after a 7-day **reveal lag**) join the **retraining corpus**. A FastAPI service and a batch scorer share one MLflow `pyfunc` model behind a strict **feature contract**. Drift or accumulated-volume triggers a Prefect retraining flow; a statistical promotion gate (ADR-0004) decides whether the challenger becomes the champion. GitHub Actions runs CI (lint, tests, contract) and CD (build + publish image to GHCR). `make demo` brings the whole stack up from the committed seed artifact.

## User Stories

1. As a fraud operations engineer, I want to POST a single transaction to a real-time endpoint and get back `{score, decision, threshold}`, so that I can block or allow a transaction immediately.
2. As a fraud operations engineer, I want the API to reject malformed input with a precise error, so that I never silently score a transaction that violates the feature contract.
3. As a batch analytics user, I want to score a CSV of transactions and get back a CSV with `TransactionID`, `score`, and `decision`, so that I can run bulk scoring.
4. As a data scientist, I want the model artifact to carry the full feature transform + booster + threshold, so that serving never re-implements training logic.
5. As a data scientist, I want `train.py` to re-fit the `finetuned_lgbm` recipe from its registered hyperparameters on the 70% training split, so that the seed model is reproducible.
6. As a data scientist, I want the seed artifact committed to the repo, so that `make demo` works offline on a fresh clone without re-training.
7. As a platform user, I want the model registered in an MLflow registry with stage transitions, so that promotion is auditable.
8. As a platform user, I want the champion and challenger evaluated on the same test set, so that comparisons are apples-to-apples.
9. As a platform user, I want promotion decided by a statistical significance test, so that a challenger is promoted only when it is genuinely better.
10. As a platform user, I want a challenger that is not significantly better to remain in Staging, so that nothing regresses automatically.
11. As a platform user, I want retraining to trigger on accumulated stream volume or on drift, so that the model stays current.
12. As a platform user, I want retraining to fold the scored stream into the retraining corpus once labels are revealed (7-day lag), so that the model improves from production feedback.
13. As a monitoring user, I want drift reports comparing the training distribution to the production-stream distribution (feature drift + score drift), so that I can see when the data or the model's output shifts.
14. As a monitoring user, I want drift to alarm on an aggregate rule (≥10% of features drift OR score drift), so that I am not spammed by single-feature noise.
15. As a demo presenter, I want a stream simulator that replays the production stream through the API, so that I can show live scoring.
16. As a demo presenter, I want `make demo` to bring up MLflow, the API, the batch scorer, the orchestrator, and monitoring from the committed seed, so that the demo is turnkey.
17. As a CI user, I want lint, tests, and a feature-contract check to run on push, so that regressions are caught early.
18. As a CI user, I want CD to build and publish the serving image to GHCR tagged by commit, so that the repo shows a real artifact pipeline.
19. As a maintainer, I want the data split to be 70% train / 15% test / 15% production stream (chronological), so that the stream simulates unseen production traffic.
20. As a maintainer, I want the production stream to enter the stack label-free, so that the simulation is honest about label arrival.
21. As a maintainer, I want the operating threshold chosen from a 10:1 (missed-fraud : blocked-good) cost ratio on the test set, so that the decision point reflects the business asymmetry.
22. As a maintainer, I want the drift window to be a time slice of recent stream activity, so that monitoring reflects a recent period rather than all history.
23. As a maintainer, I want Docker Compose to run MLflow in-stack with a named volume, so that the demo is self-contained.
24. As a maintainer, I want the training container to mount the local data directory read-only, so that no in-container data pull is needed.
25. As a reviewer, I want the repo to contain CONTEXT.md, ADRs, and the spec, so that the design decisions are recorded and defensible.

## Implementation Decisions

- **Seed model**: `train.py` reads the `finetuned_lgbm` hyperparameters from the MLflow run (experiment `ieee-fraud-detection-finetune`, run `7c925b…`), re-fits on the 70% training split, and logs via `mlflow.lightgbm.log_model` (pyfunc flavor) with the operating threshold stored as a model tag. The committed seed artifact is the output of this single pipeline (ADR-0002).
- **Registry**: start a **clean deployment registry** (the old `mlruns` store, with its `notebooks/mlruns` artifact split-brain and FAILED runs, is archaeology — do not migrate it). MLflow runs in-stack as a Compose service with a named volume, seeded from the committed artifact.
- **Feature contract**: exactly the 218 training columns with the 9 categorical columns as `category` dtype, enforced in the pyfunc wrapper and re-checked by the API. Reject (400) on missing column, extra column, wrong dtype, or NaN. `TransactionDT` is a training feature and is required in requests.
- **Serving surfaces**:
  - Real-time: `POST /predict`, one transaction's 218 fields → `{score, decision, threshold}`.
  - Batch: CSV in (218 cols) → CSV out with `TransactionID`, `score`, `decision`. Batch output feeds the drift monitor (the "current" window source).
- **Data split (ADR-0003)**: chronological 70/15/15 — train / test / production stream. The test set is used for operating-threshold selection and champion-vs-challenger comparison only. The production stream is held out, label-free at serve time, and replayed.
- **Operating threshold**: derived by minimizing expected cost on the test set under a 10:1 (missed-fraud : blocked-good) ratio. Stored with the model.
- **Promotion (ADR-0004)**: statistical gate — DeLong test / paired bootstrap on the shared test set. Significantly better → promote to Production; else remain in Staging. Champion is re-baselined on the new split (the old 0.9210 was 80/20 and is no longer canonical).
- **Retraining trigger**: drift alarm OR accumulated-volume threshold (default ~5,000 scored transactions, configurable) since the last retrain.
- **Retraining corpus**: all historical training data + accumulated scored stream whose labels have been revealed (7-day reveal lag). Each retrain re-splits the enlarged corpus into a fresh 70/15/15.
- **Drift monitoring**: Evidently comparing a time-sliced current window (recent simulated activity) to the training reference. Signals: feature drift + score-distribution drift. Alarm rule: ≥10% of features drift OR score-distance threshold crossed.
- **Stream simulator**: Prefect flow that replays the production stream slice through the API at accelerated cadence (live demo) alongside scheduled batch scoring that accumulates the drift window.
- **CI/CD**: GitHub Actions — CI = ruff, pytest, feature-contract check; CD = build & push serving image to GHCR tagged by commit. `make demo` builds and runs locally (no registry auth needed).
- **Module layout** (restructure): serving (API + batch) sharing the pyfunc model; orchestration flows; monitoring/drift; training pipeline (`train.py`); deployment compose + Dockerfile; CI/CD workflows; tests. Existing `ieee_cis_fraud_detection` package (features, config) is preserved and reused.
- **Dependencies**: add `fastapi`, `uvicorn`, `prefect`, `evidently`, and Docker tooling to the project dependencies.

## Testing Decisions

- **What makes a good test here**: assert external behavior only — the observable contract (inputs accepted/rejected, score/decision produced, promotion/drift decided) — never implementation details. Fast, hermetic, no network.
- **Seam 1 — scoring & decision boundary (primary)**: unit-test the shared interface that enforces the feature contract and applies the model + threshold. Cases: missing column rejected, extra column rejected, wrong dtype rejected, NaN rejected, category coercion correct, score correct on a small committed fixture, decision flips at the operating threshold. Both serving surfaces inherit this correctness by being thin adapters.
- **Seam 2 — control-plane logic (secondary)**: pure-function tests for `build_retraining_corpus` (reveal-lag cutoff boundary), `decide_promotion` (constructed significant vs insignificant score pairs), and `decide_drift` (aggregate rule on constructed stats).
- **Integration smoke tests**: boot the API (in-process test client) and hit `POST /predict`; run the batch CLI on a small CSV and assert the output columns. Docker Compose and GitHub Actions correctness is verified by running the stack / workflow, not by unit tests.
- **Prior art**: the repo currently has no working tests (only a failing placeholder) — this spec introduces the test suite from scratch. No existing fixture data; a small committed fixture (a few hundred rows matching the 218-col contract) is added for Seam 1 and the smoke tests.

## Out of Scope

- Identity-feature engineering and any v2 model improvements (feature engineering is a separate future effort; identity raw join was dropped as it hurt AUC).
- Cloud deployment (ADR-0001: local Docker only; GHCR is the publish target, not a runtime).
- Migration of the historical MLflow runs into the new clean registry.
- Prometheus/Grafana observability (monitoring is Evidently reports + drift threshold).
- Online learning / real-time model updates (retraining is batch, triggered, not streaming).
- The 70/15/15 split changes to the original notebook modeling pipeline (notebooks remain reference-only).

## Further Notes

- **Reveal lag constraint (fact-checked)**: the dataset spans 182 days; the production stream is the last 30.8 days. A 7-day reveal lag reveals ~77% of stream labels while keeping the newest simulated days label-free. A 120-day lag was rejected because it reveals zero stream labels.
- **DVC**: raw data remains DVC-tracked with the local `storage` remote; the training container mounts the materialized `data/` directory read-only (no in-container `dvc pull`).
- **Git hygiene**: the restructure and this deployment land as sequenced commits; `HANDOFF.md`'s deployment section is updated to reflect the new state.
- The old `mlruns` store (tracking DB at `mlruns/mlflow.db` but artifact URIs pointing into `notebooks/mlruns/`) is known-broken and intentionally not reused; the deployment uses a fresh store seeded from the committed artifact.
- No specific file paths are named in this spec by design — they will change; the ticket breakdown (`.scratch/mlops-deployment/issues/`) carries the concrete layout.
