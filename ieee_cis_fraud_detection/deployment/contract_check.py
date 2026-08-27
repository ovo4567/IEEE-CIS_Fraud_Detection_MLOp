"""CI feature-contract check (ticket 10).

The gate CI runs over the committed seed artifact: it loads the champion
boundary and asserts the artifact carries the exact production feature
contract — 218 feature columns with the 9 categoricals as ``category`` dtype,
plus a sane operating threshold — and that a contract-shaped row actually
scores to ``{score, decision, threshold}``. It runs offline on a fresh clone
(the seed artifact is git-tracked, ticket 02), so no data or registry access
is needed. Any deviation fails with a precise message and a non-zero exit.

The pure check (:func:`check_seed_contract`) takes an injected boundary so it
is unit-testable hermetically; the CLI (:func:`main`) wires it to the
committed seed. Invoked via ``python -m`` (the deployment-package convention),
so it is NOT re-exported from the package ``__init__``.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Annotated

from loguru import logger
import pandas as pd
import typer

from ieee_cis_fraud_detection.config import SEED_MODEL_PATH
from ieee_cis_fraud_detection.modeling.pyfunc import apply_transform
from ieee_cis_fraud_detection.serving.scoring import (
    ModelContract,
    ScoringBoundary,
    load_model,
)

# The production feature contract (spec.md): exactly these shapes, or CI fails.
EXPECTED_FEATURE_COUNT = 218
EXPECTED_CATEGORICAL_COUNT = 9


class SeedContractError(ValueError):
    """The seed artifact deviates from the expected production feature contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedContractError(message)


def check_seed_contract(boundary: ScoringBoundary) -> ModelContract:
    """Validate a champion boundary against the production feature contract.

    Returns the validated :class:`ModelContract` on success; raises
    :class:`SeedContractError` naming the first deviation otherwise. Reads the
    contract through the boundary's public properties so the contract stays
    hidden behind the seam it exists to hide.
    """
    contract = boundary.contract
    features = boundary.feature_columns
    categoricals = boundary.categorical_columns

    _require(
        len(features) == EXPECTED_FEATURE_COUNT,
        f"expected exactly {EXPECTED_FEATURE_COUNT} feature columns, got {len(features)}",
    )
    _require(
        len(set(features)) == len(features),
        f"feature columns are not unique ({len(features) - len(set(features))} duplicate(s))",
    )
    _require(
        len(categoricals) == EXPECTED_CATEGORICAL_COUNT,
        f"expected exactly {EXPECTED_CATEGORICAL_COUNT} categorical columns, "
        f"got {len(categoricals)}",
    )
    outside = [column for column in categoricals if column not in set(features)]
    _require(
        not outside,
        f"categorical columns outside the feature set: {outside}",
    )
    threshold = boundary.threshold
    _require(
        isfinite(threshold) and 0.0 < threshold < 1.0,
        f"operating threshold must be finite and in (0, 1), got {threshold!r}",
    )

    _check_round_trip(boundary, contract)
    return contract


def _check_round_trip(boundary: ScoringBoundary, contract: ModelContract) -> None:
    """A single contract-shaped row must score under the strict boundary."""
    frame = pd.DataFrame({column: [0.0] for column in contract.feature_columns})
    for column in contract.categorical_columns:
        frame[column] = "0"

    # The feature contract promises the 9 categoricals are coerced to the
    # pandas ``category`` dtype by the shared transform (ADR-0002) — assert it
    # through the same public function the artifact uses, not an ad-hoc cast.
    prepared = apply_transform(frame, contract.feature_columns, contract.categorical_columns)
    for column in contract.categorical_columns:
        dtype = prepared[column].dtype
        _require(
            isinstance(dtype, pd.CategoricalDtype),
            f"categorical column {column!r} is not the `category` dtype "
            f"after the transform (got {dtype})",
        )

    try:
        out = boundary.score(frame)
    except Exception as exc:
        # Surface any scoring failure as a precise contract error (the gate's
        # whole job is to name the first deviation from the contract).
        raise SeedContractError(f"contract-shaped row failed to score: {exc}") from exc
    _require(
        list(out.columns) == ["score", "decision", "threshold"],
        f"scoring a contract-shaped row returned columns {list(out.columns)}; "
        "expected [score, decision, threshold]",
    )
    _require(
        out["score"].between(0, 1).all(),
        f"score out of the unit interval: {out['score'].tolist()}",
    )


app = typer.Typer()


@app.command()
def main(
    model_path: Annotated[
        Path | None,
        typer.Option(
            help="Seed artifact to check (default: the committed models/seed/champion_model)"
        ),
    ] = None,
) -> None:
    """Verify the committed seed artifact's feature contract (CI gate)."""
    path = Path(model_path) if model_path is not None else SEED_MODEL_PATH
    try:
        boundary = load_model(path)
        contract = check_seed_contract(boundary)
    except SeedContractError as exc:
        logger.error(f"Feature contract FAILED: {exc}")
        raise typer.Exit(1)
    logger.success(
        f"Feature contract OK: {len(contract.feature_columns)} features, "
        f"{len(contract.categorical_columns)} categoricals, "
        f"threshold={contract.threshold:.4f}"
    )


if __name__ == "__main__":
    app()
