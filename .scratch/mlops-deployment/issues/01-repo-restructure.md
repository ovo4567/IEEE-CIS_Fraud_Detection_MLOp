# 01: Repo restructure into MLOps layout

**What to build:** Restructure the repository into a clean MLOps layout (serving / orchestration / monitoring / deploy / train) while preserving the existing `ieee_cis_fraud_detection` package and its feature pipeline. This is the prefactor that makes the rest of the work easy — "make the change easy, then make the easy change."

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Repo has distinct areas for serving, orchestration, monitoring, deployment, and training
- [x] Existing feature pipeline and config remain intact and importable
- [x] Lint and existing tests still pass after the move

## Answer

Restructured into a clean MLOps layout (commits `e27c7c3`, `aee871f` on `main`):

- **Area subpackages** under the existing `ieee_cis_fraud_detection` package (flit ships the whole tree, so they're in the wheel with zero packaging config):
  - `ieee_cis_fraud_detection/serving/` — real-time API + batch scorer (built in tickets 04/05 over the Seam-1 module from 03)
  - `ieee_cis_fraud_detection/orchestration/` — Prefect flows (07 retraining, 08 stream simulator)
  - `ieee_cis_fraud_detection/monitoring/` — Evidently drift (08)
  - training stays at the existing `ieee_cis_fraud_detection/modeling/` (`train.py` rewritten in 02)
- **Infra areas**: top-level `deploy/` (README; Compose/Dockerfile land here in 09) and `.github/workflows/` (CI/CD land here in 10)
- **Preserved**: `ieee_cis_fraud_detection` core package (`config`, `features`, `dataset`, `plots`) untouched and importable — verified via import check.
- **Lint**: fixed pre-existing `features.py` ruff format + import-order debt (behavior-preserving) so `make lint` is green (11 files formatted, `ruff check` clean). Verified `uv build` wheel contains the new subpackages.
- **Tests**: `python -m pytest tests` shows exactly one failure — the pre-existing placeholder `test_code_is_tested` (`assert False`), documented prior art the spec replaces from scratch in the Seam tickets (03/06). No regression introduced by this move.
- Reviewed via two-axis code-review (Standards + Spec): no hard violations; Spec criteria fully met; judgment-call docstring/README wording tightened accordingly.
