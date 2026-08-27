# 01: Repo restructure into MLOps layout

**What to build:** Restructure the repository into a clean MLOps layout (serving / orchestration / monitoring / deploy / train) while preserving the existing `ieee_cis_fraud_detection` package and its feature pipeline. This is the prefactor that makes the rest of the work easy — "make the change easy, then make the easy change."

**Blocked by:** None (can start immediately).

**Status:** claimed

- [ ] Repo has distinct areas for serving, orchestration, monitoring, deployment, and training
- [ ] Existing feature pipeline and config remain intact and importable
- [ ] Lint and existing tests still pass after the move
