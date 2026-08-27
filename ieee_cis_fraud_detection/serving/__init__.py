"""Real-time and batch model serving surfaces.

The real-time API and the batch scorer are thin adapters over the shared
scoring module, which enforces the 218-column feature contract, loads the
MLflow ``pyfunc`` model, and applies the operating threshold (ADR-0002).

The batch scorer is deliberately NOT re-exported here: it is a CLI module run
via ``python -m ieee_cis_fraud_detection.serving.batch`` (the repo's CLI
convention), and eagerly importing it from the package ``__init__`` would make
runpy execute the module twice and emit a double-execution warning. Import
``score_csv`` / ``BatchError`` from ``ieee_cis_fraud_detection.serving.batch``
directly.
"""

from ieee_cis_fraud_detection.serving.api import create_app
from ieee_cis_fraud_detection.serving.scoring import (
    ContractError,
    ModelContract,
    ScoringBoundary,
    load_model,
)

__all__ = [
    "ContractError",
    "ModelContract",
    "ScoringBoundary",
    "create_app",
    "load_model",
]
