"""Deployment plumbing for the self-contained Docker demo stack (ticket 09).

Small, testable helpers the Compose container entrypoints call — chiefly
seeding the MLflow named-volume store from the committed seed artifact so
``make demo`` serves the champion with no re-training and no cloud. The stack
itself is declared in ``deploy/compose.yaml`` and verified by running it (per
the master spec's testing decision, Compose correctness is not unit-tested).

The CLI modules here are invoked via ``python -m``, so they are NOT re-exported
from this package (eager imports make runpy double-execute them — the same
convention as ``serving.batch``, ticket 05).
"""
