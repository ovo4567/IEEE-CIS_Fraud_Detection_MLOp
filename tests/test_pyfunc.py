"""Tests for the champion pyfunc model (ADR-0002, ticket 02).

Seam C: `ieee_cis_fraud_detection.modeling.pyfunc.ChampionModel` — the single
artifact carrying the 218-feature transform + the LightGBM booster + the
operating threshold. Uses a tiny fitted booster so the tests stay fast and
hermetic.
"""

import numpy as np
import pandas as pd
import pytest

from ieee_cis_fraud_detection.modeling.pyfunc import ChampionModel

FEATURE_COLUMNS = ["TransactionDT", "amount", "cat_a", "cat_b"]
CATEGORICAL_COLUMNS = ["cat_a", "cat_b"]


def make_booster_and_frame(n: int = 60) -> tuple:
    """A tiny LGBMClassifier fitted on a frame with the contract's shape."""
    import lightgbm as lgb

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "TransactionDT": rng.integers(86_400, 15_000_000, size=n),
            "amount": rng.uniform(0, 1000, size=n),
            "cat_a": rng.choice(["W", "H", "C", "S"], size=n),
            "cat_b": rng.choice(["a", "b", "c"], size=n),
            "isFraud": rng.integers(0, 2, size=n),
        }
    )
    for c in CATEGORICAL_COLUMNS:
        frame[c] = frame[c].astype("category")
    y = frame["isFraud"].to_numpy()
    X = frame[FEATURE_COLUMNS]
    booster = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
    return ChampionModel(booster, FEATURE_COLUMNS, CATEGORICAL_COLUMNS, threshold=0.09), X


def test_transform_selects_exactly_the_feature_columns_in_order() -> None:
    model, X = make_booster_and_frame()
    # Extra columns must be dropped, and the 4 feature columns kept in order.
    noisy = X.copy()
    noisy["extra_col"] = 1.0
    transformed = model.transform(noisy)
    assert list(transformed.columns) == FEATURE_COLUMNS


def test_transform_coerces_categoricals_to_category_dtype() -> None:
    model, X = make_booster_and_frame()
    # Categoricals arrive as plain object/string columns; the transform must
    # coerce exactly them (and only them) to the pandas `category` dtype.
    raw = X.copy()
    for c in CATEGORICAL_COLUMNS:
        raw[c] = raw[c].astype("object")
    transformed = model.transform(raw)
    for c in CATEGORICAL_COLUMNS:
        assert transformed[c].dtype.name == "category"
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    for c in numeric:
        assert transformed[c].dtype.name != "category"


def test_predict_returns_one_score_per_row() -> None:
    model, X = make_booster_and_frame()
    out = model.predict(None, X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["score"]
    assert len(out) == len(X)
    assert out["score"].between(0, 1).all()


def test_model_carries_the_operating_threshold() -> None:
    model, _ = make_booster_and_frame()
    assert model.threshold == 0.09


def test_transform_requires_the_feature_columns() -> None:
    model, X = make_booster_and_frame()
    with pytest.raises(KeyError):
        model.transform(X.drop(columns=["amount"]))
