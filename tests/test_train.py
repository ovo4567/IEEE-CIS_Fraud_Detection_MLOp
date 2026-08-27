"""End-to-end test of the seed champion pipeline (ticket 02).

Seam D: `ieee_cis_fraud_detection.modeling.train.run_seed_pipeline` — re-fits
the finetuned recipe on the 70% train split, picks the operating threshold on
the test set, saves the pyfunc to a committed-style path, and registers it as
champion version 1 in a clean registry. Run on a small synthetic frame with
explicit hyperparameters so it is fast and hermetic.
"""

import numpy as np
import pandas as pd

import mlflow.pyfunc

from ieee_cis_fraud_detection.modeling.train import resolve_finetuned_params, run_seed_pipeline

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


def make_transaction_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(86_400, 15_000_000, size=n)),
            "isFraud": rng.integers(0, 2, size=n),
            "amount": rng.uniform(0, 1000, size=n),
            "cat_a": rng.choice(["W", "H", "C", "S"], size=n),
        }
    ).assign(cat_a=lambda df: df["cat_a"].astype("category"))


def test_run_seed_pipeline_registers_champion_and_saves_loadable_artifact(
    tmp_path,
) -> None:
    registry_dir = tmp_path / "registry"
    model_path = registry_dir / "champion_model"

    result = run_seed_pipeline(
        make_transaction_frame(),
        params=FAST_PARAMS,
        cost_ratio=10.0,
        registry_dir=registry_dir,
        experiment_name="ieee-fraud-champion",
        registered_name="ieee-fraud-champion",
        model_path=model_path,
    )

    # The pyfunc is saved to the committed-style path and loads offline.
    assert model_path.is_dir()
    loaded = mlflow.pyfunc.load_model(str(model_path))
    assert list(loaded.predict(make_transaction_frame().drop(columns=["isFraud", "TransactionID"]).head(2)).columns) == ["score"]

    # Champion registered as version 1 in the clean registry.
    assert result.model_version == 1
    assert result.threshold > 0.0
    assert result.test_auc == result.test_auc  # not NaN


def test_committed_seed_artifact_loads_offline() -> None:
    """The committed seed (ticket 02) loads without re-training.

    Guards the whole point of the seed: `make demo` on a fresh clone serves
    this artifact. Replaces the original placeholder test.
    """
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient

    from ieee_cis_fraud_detection.config import SEED_MODEL_PATH, SEED_TRACKING_URI

    loaded = mlflow.pyfunc.load_model(str(SEED_MODEL_PATH))
    python_model = loaded._model_impl.python_model

    # The artifact carries the full feature contract + the operating threshold.
    assert len(python_model.feature_columns) == 218
    assert len(python_model.categorical_columns) == 9
    assert 0.0 < python_model.threshold < 1.0

    # The clean registry records the champion as version 1 with the threshold.
    client = MlflowClient(tracking_uri=SEED_TRACKING_URI)
    version = client.get_model_version("ieee-fraud-champion", "1")
    assert version.tags.get("operating_threshold") == str(python_model.threshold)

    # It scores a single contract-shaped row.
    sample = pd.DataFrame({c: [0.0] for c in python_model.feature_columns})
    out = loaded.predict(sample)
    assert out["score"].between(0, 1).all()


def test_recipe_resolves_from_clean_registry_when_legacy_store_missing(
    tmp_path,
) -> None:
    """`make seed` on a fresh clone still finds the finetuned recipe.

    The legacy notebook store (`mlruns/`) is gitignored, so on a fresh clone
    the recipe must fall back to the recipe logged with the seed itself in the
    clean registry.
    """
    registry_dir = tmp_path / "registry"
    run_seed_pipeline(
        make_transaction_frame(),
        params=FAST_PARAMS,
        registry_dir=registry_dir,
        experiment_name="ieee-fraud-champion",
        registered_name="ieee-fraud-champion",
        model_path=registry_dir / "champion_model",
    )

    # Point at a non-existent legacy store: the resolver must fall back to the
    # clean registry's seed recipe, with the params correctly re-typed.
    params = resolve_finetuned_params(str(tmp_path / "missing" / "mlflow.db"), registry_dir)
    assert params["n_estimators"] == FAST_PARAMS["n_estimators"]
    assert isinstance(params["num_leaves"], int)
    assert isinstance(params["learning_rate"], float)
