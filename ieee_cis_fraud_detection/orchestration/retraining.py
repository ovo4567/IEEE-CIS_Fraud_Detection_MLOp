"""Retraining flow (ticket 07).

The retraining half of the closed loop, as a Prefect flow. On a trigger —
accumulated scored volume since the last retrain (default ~5,000, configurable)
OR a drift alarm — it:

1. builds the **retraining corpus** (all history + the scored-stream rows whose
   labels are revealed per the reveal lag, ADR-0003);
2. trains a **challenger** on a fresh 70/15/15 re-split of that corpus;
3. evaluates it against the champion on the **shared test set** through the
   statistical promotion gate (DeLong test, ADR-0004);
4. registers the challenger and **transitions the registry stage** —
   ``Production`` when promoted (and the served model is updated), ``Staging``
   otherwise.

The pure, infra-free decisions live in :mod:`control_plane` (Seam 2, ticket
06); this module wraps them in the Prefect flow plus the MLflow/registry and
serving-path mechanics. All the steps are callable as plain functions so the
tests stay hermetic (no Prefect server, no Docker).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from typing import Annotated

import lightgbm as lgb
from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from prefect import flow
from sklearn.metrics import roc_auc_score
import typer

from ieee_cis_fraud_detection.config import (
    DRIFT_STORE_PATH,
    LEGACY_TRACKING_URI,
    RETRAIN_STATE_PATH,
    SEED_MODEL_PATH,
    SEED_REGISTRY_DIR,
    SERVING_MODEL_PATH,
    tracking_uri_for,
)
from ieee_cis_fraud_detection.features import TRANSACTION_FEATURES_PATH
from ieee_cis_fraud_detection.modeling.pyfunc import ChampionModel
from ieee_cis_fraud_detection.modeling.split import temporal_split_70_15_15
from ieee_cis_fraud_detection.modeling.threshold import (
    DEFAULT_COST_RATIO,
    operating_threshold,
)
from ieee_cis_fraud_detection.modeling.train import prepare_data, resolve_finetuned_params
from ieee_cis_fraud_detection.monitoring.drift_store import read_store
from ieee_cis_fraud_detection.orchestration.control_plane import (
    DEFAULT_ALPHA,
    DEFAULT_REVEAL_LAG,
    PromotionDecision,
    build_retraining_corpus,
    decide_promotion,
)

# The trigger's accumulated-volume default (spec: ~5,000 scored transactions
# since the last retrain, configurable).
DEFAULT_VOLUME_THRESHOLD = 5_000

CHAMPION_REGISTERED_NAME = "ieee-fraud-champion"
RETRAIN_EXPERIMENT = "ieee-fraud-retrain"
RETRAIN_RUN_NAME = "challenger"

_PIP_REQUIREMENTS = ["mlflow", "lightgbm", "pandas", "numpy", "scikit-learn"]


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrainTrigger:
    """Whether the retrain trigger fires, and why."""

    retrain: bool
    volume_since_last: int
    volume_threshold: int
    volume_triggered: bool
    drift_triggered: bool


def should_retrain(
    *,
    scored_volume_since_last: int,
    volume_threshold: int = DEFAULT_VOLUME_THRESHOLD,
    drift_alarm: bool = False,
) -> RetrainTrigger:
    """Whether the retraining trigger fires.

    The rule is an OR: the accumulated scored volume since the last retrain is
    at least ``volume_threshold``, or the drift alarm is raised. The volume is
    ``scored_volume_since_last`` (the caller measures it against the drift
    current-window store).

    Raises:
        ValueError: if ``scored_volume_since_last`` is negative or
            ``volume_threshold`` is not positive.
    """
    if scored_volume_since_last < 0:
        raise ValueError(
            f"scored_volume_since_last must be non-negative, got {scored_volume_since_last!r}"
        )
    if volume_threshold <= 0:
        raise ValueError(f"volume_threshold must be positive, got {volume_threshold!r}")

    volume_triggered = scored_volume_since_last >= volume_threshold
    return RetrainTrigger(
        retrain=bool(volume_triggered or drift_alarm),
        volume_since_last=scored_volume_since_last,
        volume_threshold=volume_threshold,
        volume_triggered=bool(volume_triggered),
        drift_triggered=bool(drift_alarm),
    )


# --------------------------------------------------------------------------- #
# Accumulated volume since the last retrain
# --------------------------------------------------------------------------- #


def read_retrain_state(state_path: Path) -> dict:
    """The persisted retrain state (empty defaults if it does not exist yet).

    The state records the drift-store row count at the last retrain
    (``store_row_count``) so the accumulated volume can be measured "since the
    last retrain", plus the registered challenger version.
    """
    if not Path(state_path).exists():
        return {"store_row_count": 0}
    with open(state_path) as handle:
        return json.load(handle)


def write_retrain_state(
    state_path: Path,
    *,
    store_row_count: int,
    challenger_version: int | None = None,
) -> None:
    """Persist the retrain state after a successful retrain."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "store_row_count": int(store_row_count),
        "at": datetime.now(UTC).isoformat(),
    }
    if challenger_version is not None:
        state["challenger_version"] = int(challenger_version)
    with open(state_path, "w") as handle:
        json.dump(state, handle, indent=2)


def accumulated_volume_since_last_retrain(store_path: Path, state_path: Path) -> int:
    """Rows appended to the drift store since the last retrain.

    The drift store is a pure append log; the state holds the row count at the
    last retrain, so the volume since then is ``current - last`` (never
    negative). Without a state file the whole store counts.
    """
    current = len(read_store(store_path))
    last = int(read_retrain_state(state_path).get("store_row_count", 0))
    return max(current - last, 0)


# --------------------------------------------------------------------------- #
# Retraining corpus
# --------------------------------------------------------------------------- #


def scored_transaction_ids(store_path: Path) -> set[int]:
    """The set of ``TransactionID``s present in the drift current-window store.

    This is the honest record of what has actually been scored, so retraining
    folds in exactly the production transactions that entered the serving
    stack — never stream rows that were never scored.
    """
    store = read_store(store_path)
    if "TransactionID" not in store.columns or len(store) == 0:
        return set()
    ids = pd.to_numeric(store["TransactionID"], errors="coerce")
    return {int(x) for x in ids.dropna().unique()}


def prepare_retraining_inputs(
    transaction: pd.DataFrame,
    *,
    scored_ids: set[int] | None = None,
    reveal_lag: pd.Timedelta | float = DEFAULT_REVEAL_LAG,
    now: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the full frame and assemble the corpus.

    Returns ``(corpus, shared_test, stream)``:

    - ``corpus`` — the historical train slice plus the scored-stream rows whose
      labels are revealed per the reveal lag (via
      :func:`control_plane.build_retraining_corpus`);
    - ``shared_test`` — the original test slice, the fixed benchmark the
      champion and challenger are both scored on (ADR-0004);
    - ``stream`` — the full production-stream slice (scored or not).
    """
    train_df, test_df, stream_df = temporal_split_70_15_15(transaction)
    scored = stream_df[stream_df["TransactionID"].isin(scored_ids or set())]
    corpus = build_retraining_corpus(train_df, scored, reveal_lag=reveal_lag, now=now)
    return corpus, test_df, stream_df


# --------------------------------------------------------------------------- #
# Challenger training
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Challenger:
    """A retrained model ready for evaluation against the champion."""

    booster: object
    threshold: float
    feature_columns: list[str]
    categorical_columns: list[str]
    test_auc: float
    n_train: int
    n_corpus: int


def train_challenger(
    corpus: pd.DataFrame,
    *,
    params: dict,
    cost_ratio: float = DEFAULT_COST_RATIO,
) -> Challenger:
    """Fit a challenger on a fresh 70/15/15 re-split of the corpus.

    Mirrors the seed pipeline's fit + threshold logic: chronological fresh
    split, LightGBM fit on the fresh train, operating threshold chosen on the
    fresh test slice under the cost ratio. The threshold is tuned on the
    challenger's own fresh test slice — never on the shared benchmark the
    promotion gate reads — so the decision point cannot leak into the
    champion-vs-challenger comparison (ADR-0004). ``params`` are the LightGBM
    hyperparameters (defaults to the ``finetuned_lgbm`` recipe in the flow).
    """
    train_df, test_df, _ = temporal_split_70_15_15(corpus)
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)

    booster = lgb.LGBMClassifier(**params)
    booster.fit(X_train, y_train)

    test_scores = booster.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, y_score=test_scores))
    threshold = operating_threshold(y_test, test_scores, cost_ratio=cost_ratio)

    feature_columns = X_train.columns.tolist()
    categorical_columns = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    logger.info(
        f"Challenger fit on {len(train_df):,} rows (corpus {len(corpus):,}): "
        f"fresh-test AUC={test_auc:.4f} threshold={threshold:.4f}"
    )
    return Challenger(
        booster=booster,
        threshold=threshold,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        test_auc=test_auc,
        n_train=len(train_df),
        n_corpus=len(corpus),
    )


# --------------------------------------------------------------------------- #
# Registry + served model
# --------------------------------------------------------------------------- #


def _save_pyfunc(challenger: Challenger, model_path: Path) -> None:
    """Save the challenger as a champion-format pyfunc artifact."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    python_model = ChampionModel(
        challenger.booster,
        challenger.feature_columns,
        challenger.categorical_columns,
        challenger.threshold,
    )
    os.environ["MLFLOW_LOG_UV_FILES"] = "false"
    mlflow.pyfunc.save_model(
        path=str(model_path),
        python_model=python_model,
        pip_requirements=_PIP_REQUIREMENTS,
    )


def publish_served_model(challenger: Challenger, served_model_path: Path) -> None:
    """Publish a promoted challenger to the served-model path.

    Writing here is what makes the served model pick up the update:
    :func:`serving.scoring.load_model` prefers this path over the committed
    seed once it exists. A previous served artifact (from an earlier promotion)
    is replaced.
    """
    served_model_path = Path(served_model_path)
    if served_model_path.exists():
        shutil.rmtree(served_model_path)
    _save_pyfunc(challenger, served_model_path)
    logger.success(f"Served model updated at {served_model_path}")


def load_champion(model_path: Path):
    """Load the current champion pyfunc from an artifact path."""
    return mlflow.pyfunc.load_model(str(model_path))


def register_challenger(
    challenger: Challenger,
    *,
    registry_dir: Path,
    experiment_name: str = RETRAIN_EXPERIMENT,
    registered_name: str = CHAMPION_REGISTERED_NAME,
    params: dict,
    cost_ratio: float,
    promote: bool,
    decision: PromotionDecision,
    corpus_size: int,
    shared_test_size: int,
    served_model_path: Path,
) -> tuple[int, str]:
    """Register the challenger as the next model version and set its stage.

    Logs the run, saves the pyfunc to a version-unique artifact path, registers
    the next version in the registry, then transitions stages: on promotion the
    challenger becomes ``Production``, superseded versions (the prior
    ``Production`` champion and the seed's unstaged version) are archived while
    challengers still under review in ``Staging`` are left alone, and the
    served model is published; otherwise the challenger stays in ``Staging``
    and nothing is served.

    Returns ``(version, stage)``.
    """
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_uri_for(registry_dir)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name, artifact_location=str(registry_dir / "artifacts")
        )
    mlflow.set_experiment(experiment_name)

    try:
        client.get_registered_model(registered_name)
    except MlflowException:
        client.create_registered_model(registered_name)
    # Each version's artifact lives at its own path so the registry's audit
    # trail stays intact: a later retrain never overwrites an earlier version.
    existing = client.search_model_versions(f"name='{registered_name}'")
    next_version = max((int(v.version) for v in existing), default=0) + 1
    artifact_path = registry_dir / "challengers" / f"v{next_version}"

    with mlflow.start_run(run_name=RETRAIN_RUN_NAME) as run:
        mlflow.log_params({**params, "cost_ratio": cost_ratio, "threshold": challenger.threshold})
        mlflow.log_metrics(
            {
                "test_auc": challenger.test_auc,
                "threshold": challenger.threshold,
                "n_corpus": float(corpus_size),
                "n_train": float(challenger.n_train),
                "n_test": float(shared_test_size),
                "auc_champion": decision.auc_champion,
                "auc_challenger": decision.auc_challenger,
                "promotion_pvalue": decision.p_value,
            }
        )
        mlflow.set_tag("operating_threshold", str(challenger.threshold))
        mlflow.set_tag("promoted", str(promote))
        _save_pyfunc(challenger, artifact_path)
        run_id = run.info.run_id

    version = client.create_model_version(
        registered_name, source=str(artifact_path), run_id=run_id
    )
    client.set_model_version_tag(
        registered_name, str(version.version), "operating_threshold", str(challenger.threshold)
    )

    if promote:
        # The new champion supersedes the prior Production champion (and the
        # seed's unstaged version) — but not challengers still under review.
        for other in client.search_model_versions(f"name='{registered_name}'"):
            if other.version != version.version and other.current_stage not in (
                "Staging",
                "Archived",
            ):
                client.transition_model_version_stage(registered_name, other.version, "Archived")
        client.transition_model_version_stage(registered_name, version.version, "Production")
        stage = "Production"
        publish_served_model(challenger, served_model_path)
    else:
        client.transition_model_version_stage(registered_name, version.version, "Staging")
        stage = "Staging"

    logger.success(
        f"Challenger {registered_name!r} v{version.version} registered as {stage} "
        f"(promote={promote})"
    )
    return int(version.version), stage


# --------------------------------------------------------------------------- #
# The Prefect flow
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrainingOutcome:
    """The result of a retraining-flow run."""

    trigger: RetrainTrigger
    corpus_size: int
    shared_test_size: int
    challenger_version: int | None
    stage: str | None
    decision: PromotionDecision | None


@flow(name="ieee-fraud-retrain", description="Close the loop's retraining half.")
def retraining_flow(
    *,
    transaction: pd.DataFrame | None = None,
    features_path: Path = TRANSACTION_FEATURES_PATH,
    drift_store_path: Path = DRIFT_STORE_PATH,
    state_path: Path = RETRAIN_STATE_PATH,
    registry_dir: Path = SEED_REGISTRY_DIR,
    seed_model_path: Path = SEED_MODEL_PATH,
    served_model_path: Path = SERVING_MODEL_PATH,
    champion_source: Path | None = None,
    volume_threshold: int = DEFAULT_VOLUME_THRESHOLD,
    drift_alarm: bool = False,
    reveal_lag: pd.Timedelta | float = DEFAULT_REVEAL_LAG,
    cost_ratio: float = DEFAULT_COST_RATIO,
    alpha: float = DEFAULT_ALPHA,
    now: float | None = None,
    params: dict | None = None,
    experiment_name: str = RETRAIN_EXPERIMENT,
    registered_name: str = CHAMPION_REGISTERED_NAME,
) -> RetrainingOutcome:
    """Run one retraining pass over the closed loop.

    When ``transaction`` is omitted the full processed features are read from
    ``features_path``. The champion is loaded from ``champion_source``, or the
    served model when present, or the committed seed. ``params`` are the
    challenger's LightGBM hyperparameters; when omitted the ``finetuned_lgbm``
    recipe is resolved from the registries. Returns a
    :class:`RetrainingOutcome` describing what happened (a no-op when the
    trigger does not fire).
    """
    if transaction is None:
        logger.info(f"Reading transactions from {features_path}")
        transaction = pd.read_parquet(features_path)

    volume_since_last = accumulated_volume_since_last_retrain(drift_store_path, state_path)
    trigger = should_retrain(
        scored_volume_since_last=volume_since_last,
        volume_threshold=volume_threshold,
        drift_alarm=drift_alarm,
    )
    logger.info(
        f"Trigger: retrain={trigger.retrain} volume_since_last={trigger.volume_since_last} "
        f"volume_threshold={trigger.volume_threshold} drift={trigger.drift_triggered}"
    )
    if not trigger.retrain:
        return RetrainingOutcome(
            trigger=trigger,
            corpus_size=0,
            shared_test_size=0,
            challenger_version=None,
            stage=None,
            decision=None,
        )

    scored_ids = scored_transaction_ids(drift_store_path)
    corpus, shared_test, _stream = prepare_retraining_inputs(
        transaction, scored_ids=scored_ids, reveal_lag=reveal_lag, now=now
    )
    logger.info(f"Retraining corpus: {len(corpus):,} rows (history + revealed scored stream)")

    if params is None:
        params = resolve_finetuned_params(LEGACY_TRACKING_URI, registry_dir)
    challenger = train_challenger(corpus, params=params, cost_ratio=cost_ratio)

    if champion_source is None:
        champion_source = (
            served_model_path if Path(served_model_path).exists() else seed_model_path
        )
    champion = load_champion(champion_source)
    X_test, y_test = prepare_data(shared_test)
    champion_scores = np.asarray(champion.predict(X_test)["score"], dtype=float)
    challenger_scores = challenger.booster.predict_proba(X_test)[:, 1]
    decision = decide_promotion(y_test, champion_scores, challenger_scores, alpha=alpha)
    logger.info(
        f"Promotion gate on the shared test set ({len(shared_test):,} rows): "
        f"champion_auc={decision.auc_champion:.4f} challenger_auc={decision.auc_challenger:.4f} "
        f"p={decision.p_value:.4g} -> {'PROMOTE' if decision.promote else 'keep in Staging'}"
    )

    version, stage = register_challenger(
        challenger,
        registry_dir=registry_dir,
        experiment_name=experiment_name,
        registered_name=registered_name,
        params=params,
        cost_ratio=cost_ratio,
        promote=decision.promote,
        decision=decision,
        corpus_size=len(corpus),
        shared_test_size=len(shared_test),
        served_model_path=served_model_path,
    )
    write_retrain_state(
        state_path,
        store_row_count=len(read_store(drift_store_path)),
        challenger_version=version,
    )

    return RetrainingOutcome(
        trigger=trigger,
        corpus_size=len(corpus),
        shared_test_size=len(shared_test),
        challenger_version=version,
        stage=stage,
        decision=decision,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

app = typer.Typer()


@app.command()
def main(
    features_path: Annotated[Path, typer.Option()] = TRANSACTION_FEATURES_PATH,
    drift_store_path: Annotated[Path, typer.Option()] = DRIFT_STORE_PATH,
    volume_threshold: Annotated[int, typer.Option()] = DEFAULT_VOLUME_THRESHOLD,
    drift_alarm: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the retraining flow once (trigger -> corpus -> challenger -> gate)."""
    outcome = retraining_flow(
        features_path=features_path,
        drift_store_path=drift_store_path,
        volume_threshold=volume_threshold,
        drift_alarm=drift_alarm,
    )
    if outcome.challenger_version is None:
        logger.info("No retrain trigger; nothing to do")
    else:
        logger.success(
            f"Challenger v{outcome.challenger_version} -> {outcome.stage} "
            f"(corpus={outcome.corpus_size:,} rows)"
        )


if __name__ == "__main__":
    app()
