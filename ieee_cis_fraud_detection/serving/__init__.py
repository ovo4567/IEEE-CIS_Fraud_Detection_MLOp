"""Real-time and batch model serving surfaces.

The real-time API and the batch scorer are thin adapters over the shared
scoring module, which enforces the 218-column feature contract, loads the
MLflow ``pyfunc`` model, and applies the operating threshold (ADR-0002).
"""

from ieee_cis_fraud_detection.serving.scoring import (
    ContractError,
    ModelContract,
    ScoringBoundary,
    load_model,
)

__all__ = ["ContractError", "ModelContract", "ScoringBoundary", "load_model"]
