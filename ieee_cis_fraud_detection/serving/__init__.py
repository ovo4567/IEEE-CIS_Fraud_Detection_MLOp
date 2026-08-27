"""Real-time and batch model serving surfaces.

The real-time API and the batch scorer are thin adapters over the shared
scoring module, which enforces the 218-column feature contract, loads the
MLflow ``pyfunc`` model, and applies the operating threshold (ADR-0002).
"""
