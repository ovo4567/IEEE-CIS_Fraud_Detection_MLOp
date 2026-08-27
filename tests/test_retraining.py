"""Tests for the retraining flow (ticket 07).

Seam: `ieee_cis_fraud_detection.orchestration.retraining` — the Prefect flow
that closes the loop's retraining half: the trigger (drift alarm OR
accumulated volume since the last retrain), the retraining corpus (history +
revealed scored stream per the reveal lag), the challenger trained and
evaluated against the champion on the shared test set, the promotion decision
via the statistical gate (ADR-0004), the registry stage transition, and the
served-model update.

Hermetic: small synthetic frames, tiny hyperparameters, tmp-path registries and
stores. The Prefect flow is called as a plain function (no Prefect server).
"""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from ieee_cis_fraud_detection.config import tracking_uri_for
from ieee_cis_fraud_detection.modeling.train import run_seed_pipeline
from ieee_cis_fraud_detection.orchestration.control_plane import PromotionDecision
from ieee_cis_fraud_detection.orchestration.retraining import (
    DEFAULT_VOLUME_THRESHOLD,
    accumulated_volume_since_last_retrain,
    prepare_retraining_inputs,
    publish_served_model,
    read_retrain_state,
    register_challenger,
    retraining_flow,
    scored_transaction_ids,
    should_retrain,
    train_challenger,
    write_retrain_state,
)
from ieee_cis_fraud_detection.serving import scoring as scoring_module
from ieee_cis_fraud_detection.serving.scoring import load_model

FAST_PARAMS = {
    "n_estimators": 5,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample_freq": 1,
    "random_state": 42,
    "verbose": -1,
}

REGISTERED_NAME = "ieee-fraud-champion"


def make_ordered_frame(n: int = 300) -> pd.DataFrame:
    """A time-ordered synthetic transaction frame with labels."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 100,
            "isFraud": rng.integers(0, 2, size=n),
            "amount": rng.uniform(0, 1000, size=n),
            "cat_a": rng.choice(["W", "H", "C", "S"], size=n),
        }
    ).assign(cat_a=lambda df: df["cat_a"].astype("category"))


def make_band_frame(n: int = 600) -> pd.DataFrame:
    """A frame where fraud is a noisy band over ``amount`` (200 < amount < 800).

    A single split cannot represent a band, so a stump champion is genuinely
    weak while a forest challenger captures it — a robust, seed-insensitive
    champion-vs-challenger gap on the shared test set.
    """
    rng = np.random.default_rng(3)
    amount = rng.uniform(0, 1000, size=n)
    in_band = ((amount > 200) & (amount < 800)).astype(float)
    is_fraud = rng.binomial(1, 0.25 + 0.7 * in_band, size=n)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 100,
            "isFraud": is_fraud,
            "amount": amount,
            "cat_a": rng.choice(["W", "H", "C", "S"], size=n),
        }
    ).assign(cat_a=lambda df: df["cat_a"].astype("category"))


def write_store(path: Path, ids: list[int]) -> None:
    """Write a drift current-window store with the given scored TransactionIDs."""
    frame = pd.DataFrame(
        {
            "TransactionID": ids,
            "score": [0.1] * len(ids),
            "decision": ["allow"] * len(ids),
        }
    )
    frame.to_csv(path, index=False)


def seed_champion(frame: pd.DataFrame, tmp_path: Path, *, params: dict = FAST_PARAMS) -> Path:
    """Register champion v1 in a tmp registry and return its artifact path."""
    registry_dir = tmp_path / "registry"
    model_path = registry_dir / "champion_model"
    run_seed_pipeline(
        frame,
        params=params,
        cost_ratio=10.0,
        registry_dir=registry_dir,
        experiment_name="ieee-fraud-champion",
        registered_name=REGISTERED_NAME,
        model_path=model_path,
    )
    return model_path


# --------------------------------------------------------------------------- #
# should_retrain — the trigger rule
# --------------------------------------------------------------------------- #


def test_should_retrain_volume_at_threshold_fires() -> None:
    trigger = should_retrain(scored_volume_since_last=5_000)
    assert trigger.retrain is True
    assert trigger.volume_triggered is True
    assert trigger.drift_triggered is False
    assert trigger.volume_since_last == 5_000
    assert trigger.volume_threshold == DEFAULT_VOLUME_THRESHOLD


def test_should_retrain_volume_below_threshold_does_not_fire() -> None:
    trigger = should_retrain(scored_volume_since_last=DEFAULT_VOLUME_THRESHOLD - 1)
    assert trigger.retrain is False
    assert trigger.volume_triggered is False


def test_should_retrain_drift_alarm_fires_without_volume() -> None:
    trigger = should_retrain(scored_volume_since_last=0, drift_alarm=True)
    assert trigger.retrain is True
    assert trigger.drift_triggered is True
    assert trigger.volume_triggered is False


def test_should_retrain_default_threshold_is_5000() -> None:
    assert DEFAULT_VOLUME_THRESHOLD == 5_000


def test_should_retrain_rejects_negative_volume() -> None:
    with pytest.raises(ValueError, match="volume"):
        should_retrain(scored_volume_since_last=-1)


def test_should_retrain_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        should_retrain(scored_volume_since_last=1, volume_threshold=0)


# --------------------------------------------------------------------------- #
# Accumulated volume since the last retrain
# --------------------------------------------------------------------------- #


def test_read_retrain_state_defaults_to_zero(tmp_path: Path) -> None:
    assert read_retrain_state(tmp_path / "missing.json")["store_row_count"] == 0


def test_write_then_read_retrain_state(tmp_path: Path) -> None:
    path = tmp_path / "retrain_state.json"
    write_retrain_state(path, store_row_count=123, challenger_version=2)
    state = read_retrain_state(path)
    assert state["store_row_count"] == 123
    assert state["challenger_version"] == 2


def test_accumulated_volume_counts_rows_since_last_retrain(tmp_path: Path) -> None:
    store = tmp_path / "store.csv"
    state = tmp_path / "retrain_state.json"
    write_store(store, ids=[1, 2, 3, 4, 5])
    write_retrain_state(state, store_row_count=3)
    assert accumulated_volume_since_last_retrain(store, state) == 2


def test_accumulated_volume_is_full_count_without_state(tmp_path: Path) -> None:
    store = tmp_path / "store.csv"
    write_store(store, ids=[1, 2, 3])
    assert accumulated_volume_since_last_retrain(store, tmp_path / "missing.json") == 3


# --------------------------------------------------------------------------- #
# The drift store as the set of scored transactions
# --------------------------------------------------------------------------- #


def test_scored_transaction_ids_reads_store(tmp_path: Path) -> None:
    store = tmp_path / "store.csv"
    write_store(store, ids=[100, 101, 102])
    assert scored_transaction_ids(store) == {100, 101, 102}


def test_scored_transaction_ids_missing_store_is_empty(tmp_path: Path) -> None:
    assert scored_transaction_ids(tmp_path / "missing.csv") == set()


# --------------------------------------------------------------------------- #
# prepare_retraining_inputs — corpus vs shared test
# --------------------------------------------------------------------------- #


def test_prepare_retraining_inputs_keeps_history_and_revealed_scored_stream() -> None:
    # n=300 -> train 0..209, shared test 210..254, stream 255..299.
    frame = make_ordered_frame(n=300)
    scored_ids = set(range(255, 285))  # a 30-row subset of the stream, all scored
    corpus, test_df, stream_df = prepare_retraining_inputs(
        frame, scored_ids=scored_ids, reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    # reveal_lag=0 -> cutoff = now -> every scored stream row joins the corpus.
    assert len(corpus) == 210 + 30
    assert set(test_df["TransactionID"]) == set(range(210, 255))  # shared test slice
    assert set(stream_df["TransactionID"]) == set(range(255, 300))  # full stream


def test_prepare_retraining_inputs_honors_reveal_lag() -> None:
    # dt(row i) = i*100; max dt = 299*100 = 29_900. A 5_000s lag -> cutoff
    # 24_900, below every stream row (255*100=25_500) -> no stream row revealed.
    frame = make_ordered_frame(n=300)
    corpus, _test_df, _stream_df = prepare_retraining_inputs(
        frame,
        scored_ids=set(range(255, 300)),
        reveal_lag=5_000,
        now=float(frame["TransactionDT"].max()),
    )
    assert len(corpus) == 210  # exactly the history slice


# --------------------------------------------------------------------------- #
# train_challenger
# --------------------------------------------------------------------------- #


def test_train_challenger_fits_on_fresh_split_of_corpus() -> None:
    frame = make_ordered_frame(n=400)
    corpus, _test_df, _stream_df = prepare_retraining_inputs(
        frame, scored_ids=set(), reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    challenger = train_challenger(corpus, params=FAST_PARAMS, cost_ratio=10.0)

    assert challenger.n_corpus == len(corpus)
    assert challenger.n_train > 0
    assert 0.0 < challenger.threshold < 1.0
    assert 0.0 <= challenger.test_auc <= 1.0
    # TransactionID/isFraud are dropped; the rest are the features.
    assert challenger.feature_columns == ["TransactionDT", "amount", "cat_a"]
    assert challenger.categorical_columns == ["cat_a"]
    # The booster is scorable on the feature contract.
    X = corpus.drop(columns=["isFraud", "TransactionID"])
    scores = challenger.booster.predict_proba(X.head(5))[:, 1]
    assert scores.shape == (5,)


# --------------------------------------------------------------------------- #
# register_challenger — stage transitions
# --------------------------------------------------------------------------- #


def test_register_challenger_without_promotion_stays_in_staging(tmp_path: Path) -> None:
    frame = make_ordered_frame(n=300)
    seed_path = seed_champion(frame, tmp_path)
    registry_dir = tmp_path / "registry"
    corpus, shared_test, _stream_df = prepare_retraining_inputs(
        frame, scored_ids=set(), reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    challenger = train_challenger(corpus, params=FAST_PARAMS)
    decision = PromotionDecision(
        promote=False, auc_champion=0.5, auc_challenger=0.5, p_value=1.0, alpha=0.05
    )

    version, stage = register_challenger(
        challenger,
        registry_dir=registry_dir,
        experiment_name="ieee-fraud-retrain",
        registered_name=REGISTERED_NAME,
        params=FAST_PARAMS,
        cost_ratio=10.0,
        promote=False,
        decision=decision,
        corpus_size=len(corpus),
        shared_test_size=len(shared_test),
        served_model_path=tmp_path / "serving" / "champion_model",
    )

    assert version == 2
    assert stage == "Staging"
    client = MlflowClient(tracking_uri=tracking_uri_for(registry_dir))
    v2 = client.get_model_version(REGISTERED_NAME, "2")
    assert v2.current_stage == "Staging"
    # A non-promoted challenger is never published to the served path.
    assert not (tmp_path / "serving").exists()
    # The seed champion artifact is untouched by this registration.
    assert seed_path.is_dir()


def test_register_challenger_promotion_transitions_stage_and_publishes(
    tmp_path: Path,
) -> None:
    frame = make_ordered_frame(n=300)
    seed_champion(frame, tmp_path)
    registry_dir = tmp_path / "registry"
    corpus, _test_df, _stream_df = prepare_retraining_inputs(
        frame, scored_ids=set(), reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    challenger = train_challenger(corpus, params=FAST_PARAMS)
    served_path = tmp_path / "serving" / "champion_model"
    decision = PromotionDecision(
        promote=True, auc_champion=0.5, auc_challenger=0.8, p_value=0.001, alpha=0.05
    )

    version, stage = register_challenger(
        challenger,
        registry_dir=registry_dir,
        experiment_name="ieee-fraud-retrain",
        registered_name=REGISTERED_NAME,
        params=FAST_PARAMS,
        cost_ratio=10.0,
        promote=True,
        decision=decision,
        corpus_size=len(corpus),
        shared_test_size=len(corpus),
        served_model_path=served_path,
    )

    assert version == 2
    assert stage == "Production"
    client = MlflowClient(tracking_uri=tracking_uri_for(registry_dir))
    assert client.get_model_version(REGISTERED_NAME, "2").current_stage == "Production"
    # The previous champion (v1) is archived once the challenger is promoted.
    assert client.get_model_version(REGISTERED_NAME, "1").current_stage == "Archived"
    # The promoted challenger is published so the served model picks it up.
    assert (served_path / "MLmodel").exists()
    loaded = mlflow.pyfunc.load_model(str(served_path))
    assert loaded._model_impl.python_model.threshold == challenger.threshold


def test_register_challenger_artifacts_are_version_unique(tmp_path: Path) -> None:
    """Each registered version keeps its own artifact (audit trail, story 7)."""
    frame = make_ordered_frame(n=300)
    seed_champion(frame, tmp_path)
    registry_dir = tmp_path / "registry"
    corpus, shared_test, _stream_df = prepare_retraining_inputs(
        frame, scored_ids=set(), reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    served_path = tmp_path / "serving" / "champion_model"

    challenger_a = train_challenger(corpus, params=FAST_PARAMS)
    challenger_b = train_challenger(corpus, params={**FAST_PARAMS, "n_estimators": 9})
    decision_a = PromotionDecision(True, 0.5, 0.8, 0.001, 0.05)
    decision_b = PromotionDecision(True, 0.5, 0.9, 0.0001, 0.05)

    version_a, _ = register_challenger(
        challenger_a,
        registry_dir=registry_dir,
        registered_name=REGISTERED_NAME,
        params=FAST_PARAMS,
        cost_ratio=10.0,
        promote=True,
        decision=decision_a,
        corpus_size=len(corpus),
        shared_test_size=len(shared_test),
        served_model_path=served_path,
    )
    version_b, _ = register_challenger(
        challenger_b,
        registry_dir=registry_dir,
        registered_name=REGISTERED_NAME,
        params={**FAST_PARAMS, "n_estimators": 9},
        cost_ratio=10.0,
        promote=True,
        decision=decision_b,
        corpus_size=len(corpus),
        shared_test_size=len(shared_test),
        served_model_path=served_path,
    )

    client = MlflowClient(tracking_uri=tracking_uri_for(registry_dir))
    assert (version_a, version_b) == (2, 3)
    source_a = client.get_model_version(REGISTERED_NAME, str(version_a)).source
    source_b = client.get_model_version(REGISTERED_NAME, str(version_b)).source
    # Distinct artifact paths, and each still loads its own challenger.
    assert source_a != source_b
    loaded_a = mlflow.pyfunc.load_model(source_a)
    loaded_b = mlflow.pyfunc.load_model(source_b)
    assert loaded_a._model_impl.python_model.threshold == challenger_a.threshold
    assert loaded_b._model_impl.python_model.threshold == challenger_b.threshold


def test_promotion_preserves_challengers_still_in_staging(tmp_path: Path) -> None:
    """A later promotion archives superseded champions, not Staging challengers."""
    frame = make_ordered_frame(n=300)
    seed_champion(frame, tmp_path)
    registry_dir = tmp_path / "registry"
    corpus, shared_test, _stream_df = prepare_retraining_inputs(
        frame, scored_ids=set(), reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    served_path = tmp_path / "serving" / "champion_model"
    pending = PromotionDecision(False, 0.5, 0.5, 1.0, 0.05)
    promoted = PromotionDecision(True, 0.5, 0.9, 0.0001, 0.05)

    def register(promote: bool, decision: PromotionDecision, n_est: int) -> int:
        challenger = train_challenger(corpus, params={**FAST_PARAMS, "n_estimators": n_est})
        version, _ = register_challenger(
            challenger,
            registry_dir=registry_dir,
            registered_name=REGISTERED_NAME,
            params={**FAST_PARAMS, "n_estimators": n_est},
            cost_ratio=10.0,
            promote=promote,
            decision=decision,
            corpus_size=len(corpus),
            shared_test_size=len(shared_test),
            served_model_path=served_path,
        )
        return version

    v2 = register(False, pending, 5)  # v2 stays under review in Staging
    v3 = register(True, promoted, 7)  # v3 is promoted to Production

    client = MlflowClient(tracking_uri=tracking_uri_for(registry_dir))
    assert client.get_model_version(REGISTERED_NAME, str(v2)).current_stage == "Staging"
    assert client.get_model_version(REGISTERED_NAME, str(v3)).current_stage == "Production"
    # The seed v1 (unstaged) is superseded by the first promotion.
    assert client.get_model_version(REGISTERED_NAME, "1").current_stage == "Archived"


# --------------------------------------------------------------------------- #
# Served model picks up the update (scoring.load_model resolution)
# --------------------------------------------------------------------------- #


def test_load_model_prefers_served_model_when_present(tmp_path: Path, monkeypatch) -> None:
    frame = make_ordered_frame(n=400)
    seed_path = seed_champion(frame, tmp_path)
    assert load_model(seed_path).threshold != 0.5  # the committed seed loads fine

    corpus, _test_df, _stream_df = prepare_retraining_inputs(
        frame, scored_ids=set(), reveal_lag=0, now=float(frame["TransactionDT"].max())
    )
    challenger = train_challenger(corpus, params=FAST_PARAMS)
    served_path = tmp_path / "serving" / "champion_model"
    publish_served_model(challenger, served_path)

    monkeypatch.setattr(scoring_module, "SERVING_MODEL_PATH", served_path)
    # With a served model present, load_model() serves the promoted challenger.
    assert load_model().threshold == challenger.threshold
    assert load_model().feature_columns == tuple(challenger.feature_columns)


# --------------------------------------------------------------------------- #
# The Prefect flow, end to end
# --------------------------------------------------------------------------- #


def test_retraining_flow_no_trigger_is_noop(tmp_path: Path) -> None:
    frame = make_ordered_frame(n=300)
    seed_champion(frame, tmp_path)
    registry_dir = tmp_path / "registry"
    store = tmp_path / "store.csv"
    write_store(store, ids=[])  # empty store -> zero accumulated volume

    outcome = retraining_flow(
        transaction=frame,
        drift_store_path=store,
        state_path=tmp_path / "retrain_state.json",
        registry_dir=registry_dir,
        seed_model_path=registry_dir / "champion_model",
        served_model_path=tmp_path / "serving" / "champion_model",
        volume_threshold=DEFAULT_VOLUME_THRESHOLD,
        drift_alarm=False,
        params=FAST_PARAMS,
        now=float(frame["TransactionDT"].max()),
    )

    assert outcome.trigger.retrain is False
    assert outcome.challenger_version is None
    assert outcome.stage is None
    assert outcome.decision is None
    # No challenger version was registered.
    client = MlflowClient(tracking_uri=tracking_uri_for(registry_dir))
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    assert sorted(v.version for v in versions) == [1]


def test_retraining_flow_promotes_better_challenger(tmp_path: Path) -> None:
    frame = make_band_frame(n=600)
    # A stump champion: one 2-leaf tree cannot represent the band pattern, so
    # the forest challenger is genuinely better on the shared test set.
    weak_params = {**FAST_PARAMS, "n_estimators": 1, "num_leaves": 2}
    seed_path = seed_champion(frame, tmp_path, params=weak_params)
    registry_dir = tmp_path / "registry"
    strong_params = {**FAST_PARAMS, "n_estimators": 40}

    store = tmp_path / "store.csv"
    write_store(store, ids=list(range(510, 600)))  # the full stream slice is scored
    state_path = tmp_path / "retrain_state.json"
    served_path = tmp_path / "serving" / "champion_model"

    outcome = retraining_flow(
        transaction=frame,
        drift_store_path=store,
        state_path=state_path,
        registry_dir=registry_dir,
        seed_model_path=seed_path,
        served_model_path=served_path,
        volume_threshold=1,  # any scored volume triggers
        drift_alarm=False,
        reveal_lag=0,  # labels are "revealed" immediately in the synthetic sim
        params=strong_params,
        cost_ratio=10.0,
        now=float(frame["TransactionDT"].max()),
    )

    assert outcome.trigger.retrain is True
    assert outcome.challenger_version == 2
    assert outcome.stage == "Production"
    assert outcome.decision is not None
    assert outcome.decision.promote is True
    assert outcome.decision.auc_challenger > outcome.decision.auc_champion

    # Registry: v2 Production, v1 archived.
    client = MlflowClient(tracking_uri=tracking_uri_for(registry_dir))
    assert client.get_model_version(REGISTERED_NAME, "2").current_stage == "Production"
    assert client.get_model_version(REGISTERED_NAME, "1").current_stage == "Archived"

    # The served model was published and scores a contract-shaped row.
    boundary = load_model(served_path)
    sample = pd.DataFrame({"TransactionDT": [100], "amount": [800.0], "cat_a": ["W"]})
    out = boundary.score(sample)
    assert "score" in out.columns and "decision" in out.columns

    # The retrain state records the store volume at the retrain.
    state = read_retrain_state(state_path)
    assert state["store_row_count"] == 90
    assert state["challenger_version"] == 2
