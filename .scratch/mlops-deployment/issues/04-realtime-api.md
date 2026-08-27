# 04: Real-time API

**What to build:** A real-time serving surface — `POST /predict` accepting one transaction's 218 fields and returning `{score, decision, threshold}`, implemented as a thin adapter over the scoring & decision boundary. Requests that violate the feature contract return a precise 400 error rather than being scored.

**Blocked by:** 03: Scoring & decision boundary (Seam 1)

**Status:** resolved

- [x] `POST /predict` returns `{score, decision, threshold}` for a valid transaction
- [x] Contract violations return a precise 400
- [x] In-process smoke test exercises a successful request and a rejection

## Answer

Implemented on `main` (this ticket).

- `ieee_cis_fraud_detection/serving/api.py` — `create_app(boundary=None)` builds the FastAPI app; `POST /predict` accepts one transaction's 218 feature fields as a JSON object and returns `{score, decision, threshold}`. It is a thin adapter: the body becomes a single-row frame passed straight to `ScoringBoundary.score()` (Seam 1), and any `ContractError` — missing column, extra column, wrong dtype, or NaN — surfaces as a precise HTTP 400 via a registered exception handler, so the adapter never re-implements contract logic. `create_app()` takes an injectable boundary (tests stub it; production calls it with no args, lazily loading the committed champion on first request). Module-level `app = create_app()` is the ASGI entry point for uvicorn.
- `ieee_cis_fraud_detection/serving/__init__.py` — re-exports `create_app` alongside the scoring seam.
- Dependencies (`pyproject.toml`, re-synced via `uv sync`): added `fastapi`, `uvicorn`, and `httpx` (the latter is what FastAPI's `TestClient` needs to run the suite).
- Tests (TDD, 8 new, suite at 43 passing): hermetic unit tests inject a tiny `ScoringBoundary` and assert `POST /predict` returns `{score, decision, threshold}`, decision flips to `"block"` at/above the threshold, and each contract violation (missing column, extra column, wrong dtype, JSON `null`/NaN) returns 400 naming the offending columns. Two in-process smoke tests boot the app wired to the real committed champion and score a committed fixture transaction (200, reproducing the reference score/decision/threshold) and reject a violating payload (400) — the ticket's "successful request and a rejection".

**Design notes / reconciliations with the master spec** (`spec.md`): contract enforcement lives at the boundary, not the API — the master spec's "enforced in the pyfunc wrapper and re-checked by the API" is superseded by ticket 03's Seam-1 decision (boundary is the gate; both adapters route through it), and the outcome is identical: violations return a precise 400. Non-object request bodies (e.g. a JSON array) are rejected by FastAPI with a precise 422 before scoring, which satisfies User Story 2 ("reject malformed input"). `httpx` sits in the runtime deps because the project has no dev-dependency group and `make test` runs after a plain `uv sync`; it is required for the test suite, not used by the serving path.
