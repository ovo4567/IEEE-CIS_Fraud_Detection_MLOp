# 02: Seed champion pipeline

**What to build:** The reproducible path that turns raw processed features into a registered, committed seed model — the champion. Chronologically split the data 70% train / 15% test / 15% production stream (ADR-0003); re-fit the `finetuned_lgbm` hyperparameters from the MLflow experiment on the 70% split; compute the operating threshold from the 10:1 (missed-fraud : blocked-good) cost ratio on the test set; log the model as an MLflow `pyfunc` (ADR-0002) carrying the full feature transform + booster + threshold; register it as champion version 1 in the clean registry; and commit the seed artifact so `make demo` works offline on a fresh clone.

**Blocked by:** 01: Repo restructure into MLOps layout

**Status:** resolved

- [x] Chronological 70/15/15 split produced from the processed features
- [x] `finetuned_lgbm` re-fit on the 70% split from registered hyperparameters
- [x] Operating threshold computed on the test set under the 10:1 cost ratio
- [x] Model registered as champion v1 in a clean registry
- [x] Seed artifact committed to the repo (loadable offline without re-training)

## Answer

Implemented on `main` (commits `…` / this ticket). The reproducible path from
processed features to the committed champion:

- `ieee_cis_fraud_detection/modeling/split.py` — `temporal_split_70_15_15()`,
  chronological 70/15/15 split by `TransactionDT` (ADR-0003); reusable by the
  retraining flow (ticket 07).
- `ieee_cis_fraud_detection/modeling/threshold.py` — `operating_threshold()`,
  minimizes expected cost on the scored set under the 10:1
  (missed-fraud : blocked-good) ratio; lowest threshold at the minimum cost.
- `ieee_cis_fraud_detection/modeling/pyfunc.py` — `ChampionModel`
  (`mlflow.pyfunc.PythonModel`): carries the 218-feature transform (column
  selection in training order + the 9 categoricals coerced to `category`), the
  LightGBM booster, and the operating threshold (ADR-0002).
- `ieee_cis_fraud_detection/modeling/train.py` — `run_seed_pipeline()` + the
  `make seed` CLI: reads the `finetuned_lgbm` recipe from the original MLflow
  experiment, re-fits on the 70% split, computes the threshold on the test
  set, saves the pyfunc to `models/seed/champion_model`, and registers
  champion v1 in the clean registry (`models/seed/mlflow.db`).
- `ieee_cis_fraud_detection/config.py` — `SEED_REGISTRY_DIR` /
  `SEED_TRACKING_URI` / `SEED_MODEL_PATH` / `LEGACY_TRACKING_URI` +
  `tracking_uri_for()`.
- Tests (TDD, 23 passing): split boundaries, threshold cost minimization
  (incl. a genuine 10:1-vs-0.5:1 discrimination case), pyfunc transform/dtype
  contract, an end-to-end seed-pipeline run on a synthetic frame (champion v1
  registered + artifact loadable offline), and the recipe-fallback regression.
- `Makefile` — `make seed`.

**Reproducibility**: `make seed` pins the spec's recipe run (`7c925b…`) in the
legacy store; if that store is absent (fresh clone — `mlruns/` is gitignored),
it falls back to the recipe logged with the seed in the clean registry, so the
pipeline is re-runnable offline (data via `dvc pull`).

**Seed numbers** (committed): re-fit `finetuned_lgbm` on the 413,378-row 70%
split → test AUC **0.9286** (re-baselined on the new 70/15/15 split per
ADR-0003), operating threshold **0.0551** (expected cost 13,310 on the
88,581-row test set), champion registered as version 1.

**Mechanism note**: ADR-0002 named `mlflow.lightgbm.log_model`, but that path
cannot wrap a feature transform and its pyfunc returns class labels, not fraud
probabilities. Implemented via `mlflow.pyfunc.save_model` with the embedded
booster — pyfunc flavor carrying transform + booster + threshold — and
ADR-0002 was amended to match. The committed seed (`models/seed/`) is loadable
offline via `mlflow.pyfunc.load_model`, so `make demo` (ticket 09) serves it
without re-training.

**Contract note**: the pyfunc enforces the 218-column / 9-categorical contract
*by construction* (selection in training order + dtype coercion). Strict
rejection of a violating payload (missing column, extra column, wrong dtype,
NaN → precise 400) is the scoring-boundary seam — ticket 03 — per the spec's
Seam-1 testing decision and the ticket breakdown.
