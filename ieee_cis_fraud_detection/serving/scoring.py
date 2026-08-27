"""Scoring & decision boundary (Seam 1, ticket 03).

The deep module both serving surfaces share: a single interface that takes a
transaction (or a batch) and returns ``{score, decision, threshold}``. It
enforces the strict 218-column feature contract, loads the MLflow ``pyfunc``
champion, and applies the operating threshold. The real-time API (ticket 04)
and the batch scorer (ticket 05) are thin adapters over this module.

Contract enforcement lives here, not in the API: any payload that deviates
from the exact feature representation the champion was trained on — a missing
column, an extra column, a wrong dtype, or NaN — is rejected with a precise
:class:`ContractError` before anything is scored.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

from ieee_cis_fraud_detection.config import SEED_MODEL_PATH, SERVING_MODEL_PATH
from ieee_cis_fraud_detection.modeling.pyfunc import apply_transform


class ContractError(ValueError):
    """A payload violates the feature contract (surfaces as an HTTP 400)."""


@dataclass(frozen=True)
class ModelContract:
    """The exact feature representation a champion was trained on.

    ``feature_columns`` are the 218 training columns in training order. The 9
    categorical columns among them are listed in ``categorical_columns`` and
    coerced to the pandas ``category`` dtype at score time; the remaining
    columns must already be numeric.
    """

    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    threshold: float

    @classmethod
    def from_python_model(cls, model: object) -> ModelContract:
        """Read the contract off the champion PythonModel (raw or pyfunc-wrapped).

        The contract travels inside the committed pyfunc artifact (it is
        pickled with the PythonModel in ``python_model.pkl``), so the serving
        surfaces never hard-code the 218 columns.
        """
        python_model = _unwrap_python_model(model)
        try:
            return cls(
                feature_columns=tuple(python_model.feature_columns),
                categorical_columns=tuple(python_model.categorical_columns),
                threshold=float(python_model.threshold),
            )
        except AttributeError as exc:
            raise TypeError(
                "the model does not expose the champion contract "
                "(feature_columns, categorical_columns, threshold)"
            ) from exc


def _unwrap_python_model(model: object) -> object:
    """The underlying PythonModel of a pyfunc wrapper (or ``model`` itself).

    ``mlflow.pyfunc.load_model`` returns a ``PyFuncModel`` wrapper; the
    ChampionModel instance lives on ``_model_impl.python_model``. This is the
    single place the boundary touches that wrapper's internals.
    """
    impl = getattr(model, "_model_impl", None)
    python_model = getattr(impl, "python_model", None)
    return python_model if python_model is not None else model


class ScoringBoundary:
    """Score transactions and turn scores into decisions at the threshold.

    ``score`` accepts one transaction (a single-row DataFrame) or a batch and
    returns a DataFrame with a ``score``, a ``decision`` ("block"/"allow") and
    the shared ``threshold`` column per input row. Any contract violation
    raises :class:`ContractError` before scoring.
    """

    def __init__(
        self,
        *,
        score_fn: Callable[[pd.DataFrame], pd.Series],
        contract: ModelContract,
    ) -> None:
        self._score_fn = score_fn
        self.contract = contract

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.contract.feature_columns

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return self.contract.categorical_columns

    @property
    def threshold(self) -> float:
        return self.contract.threshold

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score one or more transactions under the feature contract.

        Returns a DataFrame with one row per input row and columns
        ``["score", "decision", "threshold"]``. ``decision`` is "block" when
        ``score >= threshold`` and "allow" otherwise.
        """
        self._validate(frame)
        prepared = self._prepare(frame)
        score = self._score_fn(prepared)
        decision = np.where(score.to_numpy() >= self.threshold, "block", "allow")
        return pd.DataFrame(
            {
                "score": np.asarray(score.to_numpy(), dtype=float),
                "decision": decision,
                "threshold": float(self.threshold),
            },
            index=frame.index,
        )

    def _validate(self, frame: pd.DataFrame) -> None:
        """Reject payloads that deviate from the exact feature contract."""
        if not isinstance(frame, pd.DataFrame):
            raise ContractError(
                f"expected a pandas DataFrame of {len(self.feature_columns)} "
                f"feature columns, got {type(frame).__name__}"
            )
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ContractError(
                f"missing {len(missing)} of {len(self.feature_columns)} feature columns: {missing}"
            )
        extra = [c for c in frame.columns if c not in self.feature_columns]
        if extra:
            raise ContractError(
                f"unexpected {len(extra)} column(s) outside the "
                f"{len(self.feature_columns)}-column feature contract: {extra}"
            )
        for column in self.feature_columns:
            if column in self.categorical_columns:
                continue
            dtype = frame[column].dtype
            numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
                dtype
            )
            if not numeric:
                raise ContractError(
                    f"column {column!r} has dtype {dtype}; expected a numeric dtype"
                )
        nan_columns = [c for c in self.feature_columns if frame[c].isna().any()]
        if nan_columns:
            raise ContractError(
                f"NaN not allowed in the feature contract; found in columns: {nan_columns}"
            )

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The validated frame in exact training order with category coercion."""
        return apply_transform(frame, self.feature_columns, self.categorical_columns)


def load_model(model_path: Path | None = None) -> ScoringBoundary:
    """Load the current champion pyfunc and wrap it as a :class:`ScoringBoundary`.

    With no explicit path, the served model (``models/serving/champion_model``
    — written by the retraining flow on promotion) is preferred; when it does
    not exist yet the committed seed artifact (``models/seed/champion_model``)
    is served instead, so serving surfaces call ``load_model()`` with no
    arguments and pick up a promoted model automatically.
    """
    if model_path is None:
        model_path = SERVING_MODEL_PATH if SERVING_MODEL_PATH.exists() else SEED_MODEL_PATH
    loaded = mlflow.pyfunc.load_model(str(model_path))
    contract = ModelContract.from_python_model(loaded)
    return ScoringBoundary(
        score_fn=lambda frame: loaded.predict(frame)["score"],
        contract=contract,
    )
