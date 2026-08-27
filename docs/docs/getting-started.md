# Getting started

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/) (`uv sync` installs the
  project and its dependencies).
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for the
  self-contained demo stack, ticket 09 / ADR-0001).
- The raw data via [DVC](https://dvc.org/): `dvc pull` (remote `storage` →
  local DVC cache).

## Set up the environment

```bash
uv sync                 # install dependencies into .venv
dvc pull                # fetch the raw data (data/raw/*)
.venv/bin/python -m ieee_cis_fraud_detection.features  # build processed features (only if data/processed is missing — gitignored)
make seed               # (only if models/seed is absent — it is committed)
make test               # run the test suite
make lint               # ruff check + format check
make contract           # feature-contract check (verifies the committed seed)
```

## CI/CD (GitHub Actions)

Pushing to `main` runs CI (ruff, the feature-contract check, the full test
suite, and a Compose-manifest validation) and then CD builds the serving image
and publishes it to GHCR tagged by commit (`ghcr.io/<owner>/ieee-fraud-serving`).
GHCR is the publish target only — `make demo` builds and runs the same image
locally with no registry auth (ADR-0001). See `deploy/README.md`.

## Run the self-contained demo stack (Docker Compose, offline)

`make demo` brings up the whole MLOps stack from the committed seed — MLflow
(seeded on a named volume), the real-time API, the Prefect orchestrator, and
the worker that runs the stream simulator + drift-monitoring pass — with no
training and no cloud:

```bash
make demo            # build + up (pre-flights Docker, data, seed), prints URLs
make demo-logs       # tail the stack logs (live scoring + monitoring passes)
make demo-down       # stop the stack
```

See `deploy/README.md` for the full walkthrough (services, URLs, tuning env
vars, and how the named-volume seeding keeps it self-contained).

## The closed loop, in one command each (no Docker)

```bash
make simulate    # replay the production stream through the real-time API
make monitor     # one drift-monitoring pass (batch-score -> Evidently -> alarm)
make retrain     # one retraining pass (trigger -> challenger -> promotion gate)
```

## Docs

This project is documented with MkDocs (`mkdocs serve` from `docs/`), the
design decisions live in `docs/adr/`, and the domain vocabulary in
`CONTEXT.md`.

