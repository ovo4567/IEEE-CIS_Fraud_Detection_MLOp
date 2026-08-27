# 09: Docker Compose + make demo

**What to build:** The self-contained stack (ADR-0001) — Docker Compose bringing up MLflow on a named volume seeded from the committed artifact, the real-time API, the batch scorer, Prefect (retraining + simulator), and monitoring — with the training container mounting the local data directory read-only. `make demo` brings the whole stack up from the committed seed offline, so a fresh clone can run the demo with no training and no cloud.

**Blocked by:** 02: Seed champion pipeline, 04: Real-time API, 05: Batch scorer, 07: Retraining flow (Prefect), 08: Stream simulator + drift monitoring (Evidently)

**Status:** ready-for-agent

- [ ] `docker compose up` starts MLflow, API, batch, Prefect, and monitoring
- [ ] MLflow seeded from the committed artifact (no re-training needed)
- [ ] Training container mounts data read-only (no in-container DVC pull)
- [ ] `make demo` works from a fresh clone offline
