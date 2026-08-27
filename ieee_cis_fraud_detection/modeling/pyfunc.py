"""The champion pyfunc model (ADR-0002, ticket 02).

One model artifact serves both the real-time API and the batch scorer. It
carries:

- the **feature transform** — coercing input to the exact training
  representation: the 218 feature columns in training order, with the 9
  categorical columns as the pandas ``category`` dtype;
- the **LightGBM booster** that produces the fraud score;
- the **operating threshold** chosen on the test set.

Embedding the transform in the artifact is what prevents train/serve drift:
serving surfaces load this pyfunc and never re-implement training logic.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlflow.pyfunc
import pandas as pd


def apply_transform(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Coerce input to the exact training representation.

    Selects precisely the training feature columns in training order and casts
    the categorical columns to the pandas ``category`` dtype, so a booster only
    ever sees the representation it was fit on. The champion artifact and the
    scoring boundary (ticket 03) share this one function, so the feature
    representation has a single owner (ADR-0002).
    """
    out = frame.loc[:, list(feature_columns)].copy()
    for column in categorical_columns:
        out[column] = out[column].astype("category")
    return out


class ChampionModel(mlflow.pyfunc.PythonModel):
    """The champion model as an MLflow pyfunc.

    ``predict`` returns a DataFrame with one ``score`` column (the fraud
    probability) per input row; the operating ``threshold`` is carried as an
    attribute so the scoring boundary can turn a score into a decision.
    """

    def __init__(
        self,
        booster,
        feature_columns: list[str],
        categorical_columns: list[str],
        threshold: float,
    ) -> None:
        self.booster = booster
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)
        self.threshold = float(threshold)

    def transform(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """Coerce input to the exact training representation (see :func:`apply_transform`)."""
        return apply_transform(model_input, self.feature_columns, self.categorical_columns)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Return the fraud score (probability) for each input row."""
        features = self.transform(model_input)
        score = self.booster.predict_proba(features)[:, 1]
        return pd.DataFrame({"score": score}, index=model_input.index)
