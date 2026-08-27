# IEEE-CIS Fraud Detection — MLOps

An end-to-end MLOps deployment of a fine-tuned LightGBM fraud model: a
reproducible training pipeline, an MLflow registry, real-time + batch serving
behind a strict feature contract, drift monitoring, and triggered retraining
with a statistical promotion gate — all reproducible locally with Docker and
wired to GitHub Actions CI/CD.

One command — `make demo` — brings up the whole stack (MLflow, FastAPI API,
Prefect, and the monitoring worker) from a committed seed artifact, offline.

## Explore

- **[Portfolio](portfolio.md)** — the MLOps story: problem, decisions, and the
  closed loop, written for reviewers.
- **[Getting started](getting-started.md)** — prerequisites and the one-command
  walkthrough (`make demo`, `make simulate`, `make monitor`, `make retrain`).
- **[Design decisions](../adr/)** — ADRs 0001–0004: local Docker, the pyfunc
  artifact, the 70/15/15 split, and the statistical promotion gate.
- **[Domain vocabulary](../../CONTEXT.md)** — the precise terms used across the
  codebase and docs.

## Commands

The Makefile holds the central entry points:

```bash
make seed       # re-fit + commit the seed champion (reproducible)
make contract   # feature-contract check on the committed seed
make test       # full hermetic test suite
make demo       # bring up the whole Docker stack from the committed seed
make demo-logs  # tail the stack logs
make demo-down  # stop the stack
```

See the repository `README.md` for the architecture and quickstart.

