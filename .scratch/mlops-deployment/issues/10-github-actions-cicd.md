# 10: GitHub Actions CI/CD

**What to build:** CI (ruff, pytest, and a feature-contract check on push) and CD (build the serving image and push it to GHCR tagged by commit) so the repo shows a real artifact pipeline. `make demo` still builds and runs locally — GHCR is the publish target, not the runtime (ADR-0001).

**Blocked by:** 03: Scoring & decision boundary (Seam 1), 06: Control-plane logic (Seam 2), 09: Docker Compose + make demo

**Status:** ready-for-agent

- [ ] CI runs ruff, pytest, and the feature-contract check on push
- [ ] CD builds the serving image and pushes it to GHCR tagged by commit
- [ ] `make demo` remains a local build (no registry auth required to run)
