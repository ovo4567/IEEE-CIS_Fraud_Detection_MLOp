"""Drift monitoring.

Evidently reports comparing the training reference distribution to the
production-stream drift window (feature drift + score drift), feeding the
aggregate drift alarm that can trigger retraining.
"""
