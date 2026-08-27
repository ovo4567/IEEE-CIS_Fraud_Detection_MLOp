"""Seed champion pipeline (ticket 02).

The reproducible path that turns the processed transaction features into the
committed seed model — the champion:

1. Chronologically split the processed features 70% train / 15% test / 15%
   production stream (ADR-0003).
2. Re-fit the ``finetuned_lgbm`` recipe — read from the original MLflow
   experiment — on the 70% train split.
3. Choose the operating threshold on the test set under the 10:1
   (missed-fraud : blocked-good) cost ratio.
4. Save the model as an MLflow pyfunc carrying the full 218-feature transform
   + the LightGBM booster + the threshold (ADR-0002).
5. Register it as champion version 1 in the clean deployment registry and
   commit the artifact so ``make demo`` works offline on a fresh clone.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from math import isfinite
import os
from pathlib import Path

import lightgbm as lgb
from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import typer

from ieee_cis_fraud_detection.config import (
    LEGACY_TRACKING_URI,
    SEED_MODEL_PATH,
    SEED_REGISTRY_DIR,
    tracking_uri_for,
)
from ieee_cis_fraud_detection.features import TRANSACTION_FEATURES_PATH
from ieee_cis_fraud_detection.modeling.pyfunc import ChampionModel
from ieee_cis_fraud_detection.modeling.split import temporal_split_70_15_15
from ieee_cis_fraud_detection.modeling.threshold import (
    DEFAULT_COST_RATIO,
    expected_cost,
    operating_threshold,
)

TARGET = "isFraud"
DROP_COLS = ["TransactionID"]  # pure ID column, never a feature

# The spec pins the finetuned_lgbm recipe to this original run (spec.md).
FINETUNE_RUN_ID = "7c925b549fa249f9add7072a5b4d7d37"
FINETUNE_EXPERIMENT = "ieee-fraud-detection-finetune"
FINETUNE_RUN_NAME = "finetuned_lgbm"
CHAMPION_EXPERIMENT = "ieee-fraud-champion"
CHAMPION_REGISTERED_NAME = "ieee-fraud-champion"
SEED_RUN_NAME = "seed_champion"

_INT_PARAMS = {
    "n_estimators",
    "num_leaves",
    "min_child_samples",
    "subsample_freq",
    "random_state",
    "n_jobs",
    "verbose",
}
_FLOAT_PARAMS = {"learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}

app = typer.Typer()


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate features (the 218-column contract) and labels."""
    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[TARGET] + DROP_COLS)
    return X, y


def read_finetuned_params(
    tracking_uri: str,
    *,
    run_id: str | None = None,
    experiment_name: str = FINETUNE_EXPERIMENT,
    run_name: str = FINETUNE_RUN_NAME,
) -> dict:
    """Read the ``finetuned_lgbm`` hyperparameters from an MLflow run.

    Uses the pinned ``run_id`` when given (the spec's original run); otherwise
    the run named ``run_name`` with the highest logged ``val_auc`` in
    ``experiment_name``. Only parameters ``LGBMClassifier`` accepts are kept
    and re-typed to int/float so they can be passed straight to the model.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    if run_id is not None:
        run = client.get_run(run_id)
    else:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise FileNotFoundError(
                f"No MLflow experiment {experiment_name!r} at {tracking_uri!r}"
            )
        best_run, best_auc = None, float("-inf")
        first_match = None
        for candidate in client.search_runs([experiment.experiment_id]):
            if candidate.data.tags.get("mlflow.runName") != run_name:
                continue
            first_match = first_match or candidate
            auc = candidate.data.metrics.get("val_auc", float("nan"))
            if isfinite(auc) and auc > best_auc:  # skip failed runs (NaN val_auc)
                best_run, best_auc = candidate, auc
        # No run logged val_auc (e.g. the clean registry's seed run logs
        # test_auc): fall back to the first run with the matching name.
        if best_run is None:
            best_run = first_match
        if best_run is None:
            raise RuntimeError(f"No run named {run_name!r} in experiment {experiment_name!r}")
        run = best_run

    params: dict = {}
    for key, value in run.data.params.items():
        if key not in _VALID_LGBM_PARAMS:
            continue
        if key in _INT_PARAMS:
            params[key] = int(float(value))
        elif key in _FLOAT_PARAMS:
            params[key] = float(value)
        else:
            params[key] = value
    logger.info(f"finetuned_lgbm recipe ({len(params)} params) from run {run.info.run_id}")
    return params


_VALID_LGBM_PARAMS = set(inspect.signature(lgb.LGBMClassifier).parameters)


def resolve_finetuned_params(
    finetune_tracking_uri: str,
    registry_dir: Path,
) -> dict:
    """Best-effort source of the finetuned recipe, in order:

    1. the spec-pinned run in the legacy notebook store;
    2. the best ``val_auc`` run there; then
    3. the clean registry's own seed recipe (logged with the seed, so
       ``make seed`` still works on a fresh clone where ``mlruns/`` is
       gitignored).
    """
    sources = [
        ("legacy pinned run", finetune_tracking_uri, {"run_id": FINETUNE_RUN_ID}),
        (
            "legacy best run",
            finetune_tracking_uri,
            {"experiment_name": FINETUNE_EXPERIMENT, "run_name": FINETUNE_RUN_NAME},
        ),
        (
            "clean registry seed recipe",
            tracking_uri_for(registry_dir),
            {"experiment_name": CHAMPION_EXPERIMENT, "run_name": SEED_RUN_NAME},
        ),
    ]
    for label, uri, kwargs in sources:
        try:
            return read_finetuned_params(uri, **kwargs)
        except (MlflowException, FileNotFoundError, RuntimeError) as exc:
            logger.warning(f"Recipe not available from {label}: {exc}")
    raise RuntimeError("No finetuned recipe available; run the FineTuning notebook first")


@dataclass(frozen=True)
class SeedResult:
    """Outcome of the seed pipeline."""

    run_id: str
    model_version: int
    threshold: float
    test_auc: float
    expected_cost: float
    model_path: Path
    registered_name: str


def run_seed_pipeline(
    transaction: pd.DataFrame,
    *,
    params: dict | None = None,
    cost_ratio: float = DEFAULT_COST_RATIO,
    registry_dir: Path = SEED_REGISTRY_DIR,
    experiment_name: str = CHAMPION_EXPERIMENT,
    registered_name: str = CHAMPION_REGISTERED_NAME,
    model_path: Path = SEED_MODEL_PATH,
    finetune_tracking_uri: str = LEGACY_TRACKING_URI,
) -> SeedResult:
    """Run the full seed pipeline and register the champion.

    ``transaction`` is the processed transaction frame (with ``TransactionID``,
    ``TransactionDT`` and ``isFraud``). When ``params`` is omitted the
    ``finetuned_lgbm`` recipe is read from ``finetune_tracking_uri``.
    """
    train_df, test_df, stream_df = temporal_split_70_15_15(transaction)
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    logger.info(
        f"Split: train={len(train_df):,} test={len(test_df):,} "
        f"stream={len(stream_df):,} (70/15/15 chronological)"
    )

    if params is None:
        params = resolve_finetuned_params(finetune_tracking_uri, registry_dir)

    booster = lgb.LGBMClassifier(**params)
    booster.fit(X_train, y_train)
    logger.info("Re-fit finetuned_lgbm recipe on the 70% train split")

    y_score_test = booster.predict_proba(X_test)[:, 1]
    # sklearn 1.9.0 kwarg is `y_score`, not `y_proba`.
    test_auc = float(roc_auc_score(y_test, y_score=y_score_test))
    threshold = operating_threshold(y_test, y_score_test, cost_ratio=cost_ratio)
    min_cost = expected_cost(y_test, y_score_test, threshold, cost_ratio)
    logger.info(
        f"Test set: AUC={test_auc:.4f} operating threshold={threshold:.4f} "
        f"expected cost={min_cost:.0f} (cost_ratio={cost_ratio:g})"
    )

    feature_columns = X_train.columns.tolist()
    categorical_columns = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    if len(feature_columns) != 218:
        logger.warning(f"Expected 218 feature columns, got {len(feature_columns)}")

    # --- Clean deployment registry ----------------------------------------
    registry_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_uri_for(registry_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name, artifact_location=str(registry_dir / "artifacts")
        )
    mlflow.set_experiment(experiment_name)

    champion = ChampionModel(booster, feature_columns, categorical_columns, threshold)
    with mlflow.start_run(run_name=SEED_RUN_NAME) as run:
        mlflow.log_params({**params, "cost_ratio": cost_ratio, "threshold": threshold})
        mlflow.log_metrics(
            {
                "test_auc": test_auc,
                "threshold": threshold,
                "expected_cost": min_cost,
                "n_train": float(len(train_df)),
                "n_test": float(len(test_df)),
                "n_stream": float(len(stream_df)),
            }
        )
        mlflow.set_tag("operating_threshold", str(threshold))
        # Keep the committed artifact lean and self-describing: don't copy the
        # project's uv.lock/pyproject.toml into it, and record the runtime deps
        # explicitly.
        os.environ["MLFLOW_LOG_UV_FILES"] = "false"
        mlflow.pyfunc.save_model(
            path=str(model_path),
            python_model=champion,
            pip_requirements=["mlflow", "lightgbm", "pandas", "numpy", "scikit-learn"],
        )
        run_id = run.info.run_id

    # Register as champion version 1 (creating the registered model if needed).
    try:
        client.get_registered_model(registered_name)
    except MlflowException:
        client.create_registered_model(registered_name)
    version = client.create_model_version(registered_name, source=str(model_path), run_id=run_id)
    client.set_model_version_tag(
        registered_name, str(version.version), "operating_threshold", str(threshold)
    )
    logger.success(
        f"Champion {registered_name!r} v{version.version} registered; artifact at {model_path}"
    )

    return SeedResult(
        run_id=run_id,
        model_version=version.version,
        threshold=threshold,
        test_auc=test_auc,
        expected_cost=min_cost,
        model_path=model_path,
        registered_name=registered_name,
    )


@app.command()
def main(
    features_path: Path = TRANSACTION_FEATURES_PATH,
    cost_ratio: float = DEFAULT_COST_RATIO,
    registry_dir: Path = SEED_REGISTRY_DIR,
    model_path: Path = SEED_MODEL_PATH,
) -> None:
    """Build and register the committed seed champion model."""
    transaction = pd.read_parquet(features_path)
    result = run_seed_pipeline(
        transaction,
        cost_ratio=cost_ratio,
        registry_dir=registry_dir,
        model_path=model_path,
    )
    logger.success(
        f"Seed champion ready: v{result.model_version}, "
        f"threshold={result.threshold:.4f}, test_auc={result.test_auc:.4f}, "
        f"model={result.model_path}"
    )


if __name__ == "__main__":
    app()
