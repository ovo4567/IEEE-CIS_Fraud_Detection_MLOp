"""Real-time serving surface (ticket 04).

A thin FastAPI adapter over the scoring & decision boundary (Seam 1, ticket
03): ``POST /predict`` accepts one transaction's 218 feature fields as a JSON
object and returns ``{score, decision, threshold}``.

The feature contract is enforced by the boundary, not here — any payload that
deviates (a missing column, an extra column, a wrong dtype, or NaN) raises a
:class:`ContractError` inside the boundary, which this adapter surfaces as a
precise HTTP 400. The endpoint itself only converts the JSON body into a
single-row frame and maps the boundary's output back to JSON.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd

from ieee_cis_fraud_detection.serving.scoring import ContractError, ScoringBoundary, load_model


def create_app(boundary: ScoringBoundary | None = None) -> FastAPI:
    """Build the FastAPI app; ``boundary`` defaults to the committed champion.

    Tests inject a stub boundary to stay hermetic; production calls
    ``create_app()`` with no arguments, lazily loading the real champion on the
    first request.
    """
    resolved = boundary

    def _get_boundary() -> ScoringBoundary:
        nonlocal resolved
        if resolved is None:
            resolved = load_model()
        return resolved

    app = FastAPI(title="IEEE-CIS Fraud Detection API", version="0.1.0")

    @app.exception_handler(ContractError)
    async def contract_error_handler(request: Request, exc: ContractError) -> JSONResponse:
        """Any feature-contract violation surfaces as a precise HTTP 400."""
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        """Score one transaction: its 218 feature fields -> {score, decision, threshold}."""
        boundary = _get_boundary()
        frame = pd.DataFrame([payload])
        row = boundary.score(frame).iloc[0]
        return {
            "score": float(row["score"]),
            "decision": str(row["decision"]),
            "threshold": float(row["threshold"]),
        }

    return app


# The ASGI entry point: `uvicorn ieee_cis_fraud_detection.serving.api:app`.
app = create_app()
