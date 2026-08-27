"""Tests for the CI feature-contract check (ticket 10).

`check_seed_contract` is the pure gate CI runs over the committed seed
artifact: it loads a :class:`ScoringBoundary` (the committed seed in CI, an
injected tiny one here) and asserts the artifact carries the exact production
feature contract — 218 feature columns, 9 categoricals, a sane threshold — and
that a contract-shaped row actually scores to ``{score, decision, threshold}``.

Hermetic unit tests drive a synthetic 218-column boundary with a stub score
function (no model load); one integration test runs the real CLI over the
committed seed artifact, which is git-tracked and loads offline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ieee_cis_fraud_detection.deployment.contract_check import (
    EXPECTED_CATEGORICAL_COUNT,
    EXPECTED_FEATURE_COUNT,
    SeedContractError,
    check_seed_contract,
)
from ieee_cis_fraud_detection.serving.scoring import ModelContract, ScoringBoundary


def make_boundary(
    n_features: int = EXPECTED_FEATURE_COUNT,
    n_categoricals: int = EXPECTED_CATEGORICAL_COUNT,
    *,
    threshold: float = 0.5,
    score_value: float = 0.42,
    duplicate_feature: bool = False,
    categorical_columns: tuple[str, ...] | None = None,
) -> ScoringBoundary:
    """A hermetic boundary under a (default valid) 218-column contract."""
    features = [f"f{i:03d}" for i in range(n_features)]
    if duplicate_feature:
        # Replace (not insert) so the column count stays 218 and only the
        # uniqueness check trips, not the count check.
        features[1] = features[0]
    if categorical_columns is None:
        categorical_columns = tuple(features[:n_categoricals])
    contract = ModelContract(
        feature_columns=tuple(features),
        categorical_columns=tuple(categorical_columns),
        threshold=threshold,
    )
    return ScoringBoundary(
        score_fn=lambda frame: pd.Series(score_value, index=frame.index),
        contract=contract,
    )


# --------------------------------------------------------------------------- #
# The happy path: a valid 218/9 contract passes and reports its shape
# --------------------------------------------------------------------------- #


def test_check_seed_contract_passes_on_valid_218_contract() -> None:
    boundary = make_boundary()

    contract = check_seed_contract(boundary)

    assert len(contract.feature_columns) == 218
    assert len(contract.categorical_columns) == 9
    assert contract.threshold == 0.5


# --------------------------------------------------------------------------- #
# Shape violations: each rejected with a precise, named error
# --------------------------------------------------------------------------- #


def test_rejects_wrong_feature_count() -> None:
    with pytest.raises(SeedContractError, match=r"218"):
        check_seed_contract(make_boundary(n_features=217))


def test_rejects_too_many_features() -> None:
    with pytest.raises(SeedContractError, match=r"218"):
        check_seed_contract(make_boundary(n_features=219))


def test_rejects_wrong_categorical_count() -> None:
    with pytest.raises(SeedContractError, match=r"9"):
        check_seed_contract(make_boundary(n_categoricals=8))


def test_rejects_categorical_column_outside_features() -> None:
    with pytest.raises(SeedContractError, match=r"not.*feature|outside"):
        check_seed_contract(
            make_boundary(categorical_columns=("f000", "not_a_feature", *("f001",) * 7))
        )


def test_rejects_duplicate_feature_columns() -> None:
    with pytest.raises(SeedContractError, match=r"duplicate|unique"):
        check_seed_contract(make_boundary(duplicate_feature=True))


# --------------------------------------------------------------------------- #
# Threshold violations
# --------------------------------------------------------------------------- #


def test_rejects_non_finite_threshold() -> None:
    with pytest.raises(SeedContractError, match=r"threshold"):
        check_seed_contract(make_boundary(threshold=float("nan")))


def test_rejects_threshold_outside_unit_interval() -> None:
    with pytest.raises(SeedContractError, match=r"threshold"):
        check_seed_contract(make_boundary(threshold=1.5))


# --------------------------------------------------------------------------- #
# Round-trip: a contract-shaped row must actually score
# --------------------------------------------------------------------------- #


def test_rejects_score_outside_unit_interval() -> None:
    with pytest.raises(SeedContractError, match=r"score"):
        check_seed_contract(make_boundary(score_value=1.7))


def test_categoricals_coerced_to_category_dtype() -> None:
    # The round-trip verifies the feature contract's ``category`` dtype promise
    # through the shared transform: every categorical must come out as the
    # pandas ``category`` dtype on a contract-shaped row.
    boundary = make_boundary()
    from ieee_cis_fraud_detection.modeling.pyfunc import apply_transform

    frame = pd.DataFrame({c: [0.0] for c in boundary.feature_columns})
    for c in boundary.categorical_columns:
        frame[c] = "0"

    prepared = apply_transform(frame, boundary.feature_columns, boundary.categorical_columns)

    for column in boundary.categorical_columns:
        assert isinstance(prepared[column].dtype, pd.CategoricalDtype)


# --------------------------------------------------------------------------- #
# Integration: the real CLI over the committed seed (git-tracked, offline)
# --------------------------------------------------------------------------- #


def test_cli_passes_on_committed_seed() -> None:
    """The gate CI runs accepts the committed seed artifact as-is.

    Guards the CI feature-contract check end-to-end: the committed
    ``models/seed/champion_model`` (git-tracked, loads offline) must satisfy
    the exact 218/9 production contract. Fails if the seed was regenerated
    with a different feature set.
    """
    from typer.testing import CliRunner

    from ieee_cis_fraud_detection.deployment.contract_check import app

    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "Feature contract OK" in result.output
