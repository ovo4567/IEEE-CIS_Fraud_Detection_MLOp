# HANDOFF — IEEE-CIS Fraud Detection

- **Last session:** 2026-08-27 (end of day)
- **Next session:** 2026-08-28 — add `identity` to LightGBM / XGBoost
- **Git:** commit the notebook changes from this session before starting

---

## Current state

| Area | Status |
|------|--------|
| Raw data | DVC-tracked (`data/raw.dvc`, remote `storage` → `/Users/alex/dvc-remote/ieee-cis-fraud`) |
| Processed features | `data/processed/*.parquet` (gitignored, regenerable via `features.py`) |
| EDA | Complete in `notebooks/Data_Exploaration.ipynb` (incl. persist-features step) |
| Modeling harness | `notebooks/Modeling.ipynb` — temporal split + MLflow |
| MLflow store | `mlruns/mlflow.db` (SQLite, local) |
| Baselines (missingness-aware) | see leaderboard below — LightGBM ≈ 0.901, XGBoost ≈ 0.896, RF ≈ 0.890, CatBoost ≈ 0.870; classical ≈ 0.80–0.82 |

### Repo map
- `ieee_cis_fraud_detection/features.py` — feature pipeline (load → drop >50% missing → category dtype → parquet)
- `ieee_cis_fraud_detection/config.py` — paths (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, …)
- `notebooks/Modeling.ipynb` — harness: `temporal_split()`, `train_and_eval()`, classical `preprocessor`, tree track
- `notebooks/Data_Exploaration.ipynb` — EDA
- `references/data-description.md` — dataset overview
- `mlruns/` — MLflow experiments (gitignored)

### Data shapes (after >50%-missing filter)
- `train_transaction`: 590,540 × 220
- `train_identity`: 144,233 × 29
- Joined on `TransactionID`; identity present for only ~25% of transactions.

---

## ✅ Resolved this session — missingness-aware imputation (MNAR)

Missing values carry signal in this data (V2/V9 missing → 2.7× fraud lift; addr1/2 missing → 4.8×). The classical-track `preprocessor` is now missingness-aware:

- **numerics** → `SimpleImputer(strategy="median", add_indicator=True)` — median fill + binary `is_missing` indicator per column (the indicator carries the signal)
- **categoricals** → constant `"missing"` fill + `OneHotEncoder(handle_unknown="ignore")` — NaN becomes its own one-hot category (= implicit indicator)
- **trees** → untouched, NaN native (LightGBM/XGBoost); CatBoost needs special handling (see gotchas)

Also done this session: `LinearSVC` `decision_function` fallback in `train_and_eval`, `roc_auc_score` kwarg fixed (`y_score`, see gotchas), full baseline suite run & logged to MLflow, MLflow cleanup (purged stale/duplicate runs, renamed runs to `baseline_*`, compare cell dedupes by best-per-name).

---

## Baseline leaderboard (transaction-only, temporal split, missingness-aware)

| Model | val AUC |
|---|---|
| LightGBM (200, lr 0.05) | 0.9006 |
| XGBoost (200, lr 0.05, `enable_categorical`) | 0.8960 |
| RandomForest (200 trees, via preprocessor) | 0.8902 |
| CatBoost (200, lr 0.05) | 0.8696 |
| LogisticRegression | 0.8181 |
| LinearSVC | 0.8137 |
| LDA | 0.8027 |

Decision: **commit to tree-based models** — LightGBM + XGBoost are the top 2.

---

## Plan for next session (priority order)

1. **Add `identity` to LightGBM & XGBoost** (transaction-only has plateaued ≈ 0.90)
   - Left-join `identity` onto `transaction` on `TransactionID` (`how="left"`).
   - **Join BEFORE the temporal split**, then re-derive `X_train/X_val` with the existing `prepare_data`/`temporal_split`. No cross-row aggregates → no leakage.
   - Identity covers only ~25% of rows — that's fine for trees: identity-column NaN on the other 75% is itself a strong MNAR signal (device/identity info absent).
   - **Cast identity categoricals to pandas `category` dtype** (`DeviceType`, `DeviceInfo` — DeviceInfo is very high-cardinality, do NOT one-hot). Keep NaN untouched.
   - Expected: modest lift vs transaction-only (often +0.01–0.03 in this comp).

2. **New baseline cells** (same hyperparams as transaction-only runs for a fair compare):
   - `baseline_lgbm_identity` — `LGBMClassifier(n_estimators=200, learning_rate=0.05)`
   - `baseline_xgb_identity` — `XGBClassifier(n_estimators=200, learning_rate=0.05, enable_categorical=True)`

3. **Compare** all runs via the MLflow compare-runs cell (already dedupes by best-per-name, skips NaN).

4. Later: hyperparameter tuning (via MLflow), feature engineering (time features from `TransactionDT`, card/addr/email aggregations), RF label-encoding variant, neural network.

---

## Commands

```bash
uv sync                      # install deps
uv add <pkg>                 # add a dependency
.venv/bin/python -m ieee_cis_fraud_detection.features   # rebuild processed features
dvc pull                     # pull raw data from remote
mlflow ui                    # open MLflow UI (store: mlruns/mlflow.db)
make lint / make format      # ruff
```

## Environment / gotchas

- Python 3.12 in `.venv`; scikit-learn 1.9.0, lightgbm 4.7, xgboost 3.4.1, catboost 1.2.10, mlflow 3.15.2, pyarrow 25.
- **MLflow 3.x**: the plain `./mlruns` file store is deprecated → use `sqlite:///mlruns/mlflow.db` (already set in the notebook).
- **sklearn 1.9.0 (this venv)**: `roc_auc_score(y_true, y_score, ...)` — the kwarg is STILL `y_score`; `y_proba=` raises `TypeError`. (Earlier handoff said `y_proba` — that was WRONG; verified empirically.)
- **XGBoost**: does not use the `category` dtype by default → `XGBClassifier(enable_categorical=True)`.
- **CatBoost**: needs `cat_features=<cols>` listed AND categorical NaN converted to a string (e.g. `fillna("missing")`) — it REJECTS NaN in categorical features (unlike LightGBM/XGBoost); numeric NaN is native.
- **MLflow**: compare cell dedupes by best `val_auc` per run_name & skips NaN; stale runs soft-deleted via `client.delete_run(run_id)`, permanently via `mlflow gc --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Parquet** preserves the `category` dtype; CSV does not — keep using parquet for processed features.
