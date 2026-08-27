"""Prefect flows for the closed loop.

The retraining flow (trigger -> retraining corpus -> challenger -> promotion
gate) and the stream simulator that replays the production stream through the
serving stack to drive the live demo and drift monitoring.
"""
