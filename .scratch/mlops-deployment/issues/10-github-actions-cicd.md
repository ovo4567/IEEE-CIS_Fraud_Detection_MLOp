# 10: GitHub Actions CI/CD

**What to build:** CI (ruff, pytest, and a feature-contract check on push) and CD (build the serving image and push it to GHCR tagged by commit) so the repo shows a real artifact pipeline. `make demo` still builds and runs locally — GHCR is the publish target, not the runtime (ADR-0001).

**Blocked by:** 03: Scoring & decision boundary (Seam 1), 06: Control-plane logic (Seam 2), 09: Docker Compose + make demo

**Status:** ready-for-agent → in-progress → resolved

- [x] CI runs ruff, pytest, and the feature-contract check on push
- [x] CD builds the serving image and pushes it to GHCR tagged by commit
- [x] `make demo` remains a local build (no registry auth required to run)

## Answer

### What was built

One GitHub Actions workflow (`.github/workflows/ci.yml`) with two jobs plus a
first-class feature-contract gate, all working offline on a fresh clone (only
git-tracked inputs: the committed seed artifact + the hermetic test suite).

- **`ci` job** — on every push and pull request:
  1. `uv sync --frozen` (locked deps via the committed `uv.lock`);
  2. `uv run make lint` (ruff format --check + ruff check);
  3. `uv run make contract` — the **feature-contract check**;
  4. `uv run make test` (full suite, 151 passing);
  5. `docker compose -f deploy/compose.yaml config --quiet` — validates the
     Compose manifest (no Docker build needed in CI).
- **`cd` job** — `needs: ci` and only on pushes to the default branch: builds
  the same `deploy/Dockerfile` (repo-root context, `.dockerignore` respected)
  and pushes it to GHCR (`ghcr.io/<owner>/ieee-fraud-serving`) tagged
  `sha-<short>` (this commit) + `latest`. `needs: ci` guarantees CD only
  publishes what CI validated; `permissions: packages: write`, login via
  `GITHUB_TOKEN`, Buildx layer cache (`type=gha`). `concurrency` cancels
  superseded runs per branch.
- **Feature-contract check** (`ieee_cis_fraud_detection/deployment/contract_check.py`
  + `make contract` + `tests/test_contract_check.py`): a dedicated CI gate that
  loads the committed seed artifact and asserts it carries the exact production
  contract — **218 feature columns, 9 categoricals** (all inside the feature
  set, unique), a **finite threshold in (0, 1)** — and that a contract-shaped
  row actually scores to `{score, decision, threshold}` with a score in the
  unit interval. Any deviation raises a precise `SeedContractError` and exits
  non-zero. Pure `check_seed_contract(boundary)` is hermetically unit-tested
  (11 tests) with an injected 218-column boundary + stub scorer; the CLI wraps
  it around the committed seed. TDD: tests written first (RED), module added
  (GREEN).

### Why the contract check is a separate CI step

`tests/test_train.py::test_committed_seed_artifact_loads_offline` already
guards the seed's shape inside the suite, but the ticket asks for a
first-class **feature-contract check** as a named CI stage — a focused gate
with its own precise output and non-zero exit, run *before* the (slower) test
suite so a bad seed fails fast and loudly. It is pure + CLI, so CI can invoke
it directly (`make contract`) and it doubles as an on-demand local check.

### Verification

- `uv run make lint` green (ruff format + check).
- `uv run make contract` → `Feature contract OK: 218 features, 9 categoricals,
  threshold=0.0551`, exit 0.
- `uv run make test` → **151 passed** (140 baseline + 11 new contract-check
  tests).
- `docker compose -f deploy/compose.yaml config --quiet` passes (same as CI).
- The workflow YAML parses; Dockerfile COPY sources all present (CD build
  context is complete).
- `make demo` untouched — `compose.yaml` still builds `ieee-fraud-demo:local`
  locally; GHCR is the publish target, not the runtime (ADR-0001).

### Design notes

- `uv run make ...` puts the project venv on `PATH`, so the Makefile's bare
  `python`/`ruff` resolve to the locked environment — CI and local use the same
  commands (`make lint`/`make test` were run via `uv run` to confirm).
- The CD image name is explicit and lowercase (`ieee-fraud-serving`) because
  GHCR requires lowercase and `github.repository` keeps the repo's uppercase;
  the owner comes from `github.repository_owner`.
- GHCR push is not runnable from this repo (needs registry auth) — correctness
  of the *build* is inherited from ticket 09 (`make demo` builds the same
  Dockerfile) and the build-context inputs are verified present; the push
  itself runs on the first push to `main`.
- CI needs no DVC pull: `data/` is DVC-tracked and gitignored, so the contract
  check and tests were designed to run on committed inputs only.
- **Review hardening**: CD is gated on CI (`needs: ci` + a default-branch-only
  `if`) so a failing commit can never publish; the contract gate now also
  asserts the feature contract's ``category`` dtype promise through the shared
  transform (spec.md); CI runs on *every* push (spec story 17), not just main.

### Verified end-to-end on GitHub (2026-08-27)

The first push (`5bccbf0..3a42b21`) triggered run **33080345424**, both jobs
green:

- **`ci`** (lint + contract + tests) completed in **3m50s** — `uv sync
  --frozen` → `make lint` → `make contract` (`Feature contract OK: 218
  features, 9 categoricals, threshold=0.0551`) → `make test` (151 passed) →
  compose validation. This confirmed the **CI cold start** works on fresh
  GitHub runners with no caches.
- **`cd`** (build + push serving image) completed in **5m50s** — Buildx build
  of `deploy/Dockerfile` from the repo root, GHCR login via `GITHUB_TOKEN`,
  and push of `ghcr.io/ovo4567/ieee-fraud-serving` tagged `sha-3a42b21` +
  `latest` (digest `sha256:f04badc2…`, `pushed_at` set in the logs).

Both previously-untestable unknowns are now verified. Only annotation on the
run: **Node.js 20 deprecation** warnings on `actions/checkout@v4`,
`actions/setup-python@v5`, `astral-sh/setup-uv@v5` (forced onto Node 24) —
non-blocking; bump those actions in a future cleanup.
