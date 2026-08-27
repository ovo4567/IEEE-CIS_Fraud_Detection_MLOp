"""Drift monitoring.

Evidently reports comparing the training reference distribution to the
production-stream drift window (feature drift + score drift), feeding the
aggregate drift alarm that can trigger retraining. The drift current-window
store (the accumulation of batch-scored stream rows) is the honest data source
the reports read.
"""

from ieee_cis_fraud_detection.monitoring.drift_store import append_scores, read_store

__all__ = ["append_scores", "read_store"]
