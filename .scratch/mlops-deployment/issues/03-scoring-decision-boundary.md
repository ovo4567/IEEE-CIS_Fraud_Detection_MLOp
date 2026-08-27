# 03: Scoring & decision boundary (Seam 1)

**What to build:** The deep module both serving surfaces share — a single interface that takes a transaction (or batch) and returns `{score, decision, threshold}`. It enforces the strict 218-column feature contract (reject missing column, extra column, wrong dtype, or NaN with a precise error), loads the pyfunc model, and applies the operating threshold. Both the real-time API and the batch scorer become thin adapters over it.

**Blocked by:** 02: Seed champion pipeline

**Status:** resolved

- [x] Missing column rejected
- [x] Extra column rejected
- [x] Wrong dtype rejected
- [x] NaN rejected
- [x] Category coercion applied correctly (9 categorical columns to `category`)
- [x] Score correct on a small committed fixture
- [x] Decision flips at the operating threshold

## Answer

Implemented on `main` (this ticket). The deep module both serving surfaces
share, at the spec's pre-agreed Seam 1:

- `ieee_cis_fraud_detection/serving/scoring.py` — `ScoringBoundary` (the seam):
  `score(frame)` takes one transaction (a single-row DataFrame) or a batch and
  returns `{score, decision, threshold}` per row. It enforces the strict
  218-column feature contract — missing column, extra column, wrong dtype
  (numeric features must be numeric; bool excluded), or NaN each raise a
  precise `ContractError` (surfaces as HTTP 400 in ticket 04) naming the
  offending columns — then coerces the 9 categorical columns to `category`,
  scores through the pyfunc, and applies the operating threshold
  (`score >= threshold` → `"block"`, else `"allow"`). `load_model()` loads the
  committed champion pyfunc and wraps it; `ModelContract` carries the 218 /
  9 / threshold read off the artifact so serving never hard-codes the columns.
- `ieee_cis_fraud_detection/modeling/pyfunc.py` — extracted `apply_transform()`
  (selection in training order + category coercion), the single owner of the
  feature representation (ADR-0002); both `ChampionModel.transform` and
  `ScoringBoundary` call it, so the boundary does not duplicate the transform.
- `ieee_cis_fraud_detection/serving/__init__.py` — re-exports the seam.
- Fixture (`tests/fixtures/`): `transactions.csv` (200 real, fully-populated
  rows: `TransactionID` + the exact 218 features, CSV so categoricals arrive as
  strings) and `expected_scores.csv` (the champion's `score` + `decision` per
  row, computed independently of the boundary so it pins the threshold on real
  data); `generate_fixtures.py` regenerates both (run it after a promotion).
- Tests (TDD, 12 new, suite at 35 passing): contract rejection (missing / extra
  / wrong dtype / NaN incl. NaN-in-categorical / non-DataFrame), category
  coercion before scoring, score passthrough + output columns, decision flips
  at the threshold (0.49→allow, 0.50→block, 0.51→block), and the committed
  fixture reproduces the reference scores and decisions through the real
  champion.

**Design notes**: contract enforcement lives at the boundary (not the API), so
tickets 04/05 become thin adapters — `POST /predict` returns the single row's
`{score, decision, threshold}`; the batch scorer scores a 218-col frame and
re-attaches `TransactionID` itself. The boundary reads the contract off the
committed pyfunc via `_model_impl.python_model` — the one place it touches
MLflow internals (isolated in `_unwrap_python_model`; fragile across MLflow
version bumps, acceptable for the pinned 3.15.x environment). Strict rejection
is the boundary's job by design (ticket 02's answer); the pyfunc remains
permissive-by-construction and the boundary is the gate both adapters route
through.
