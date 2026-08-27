# 05: Batch scorer

**What to build:** A batch serving surface — a CLI that scores a CSV of transactions (218 columns) through the scoring & decision boundary and writes a CSV out with `TransactionID`, `score`, and `decision`. Its output appends to the drift current-window store, making it the honest data source for monitoring. Runs in parallel with the real-time API.

**Blocked by:** 03: Scoring & decision boundary (Seam 1)

**Status:** resolved

- [x] CSV in (218 cols) → CSV out with `TransactionID`, `score`, `decision`
- [x] Output appended to the drift current-window store
- [x] Smoke test runs the CLI on a small CSV and asserts the output columns

## Answer

Implemented on `main` (this ticket). The batch serving surface runs in
parallel with the real-time API, as a thin adapter over Seam 1:

- `ieee_cis_fraud_detection/serving/batch.py` — `score_csv()` and its typer CLI
  (`python -m ieee_cis_fraud_detection.serving.batch --input <csv> --output
  <csv> [--drift-store <path>] [--no-append]`). The input CSV carries
  `TransactionID` + the exact 218 feature columns; `score_frame()` sets the
  row key aside (it is not a feature), routes the rest through
  `ScoringBoundary.score()` (Seam 1), and writes a CSV out with exactly
  `TransactionID`, `score`, `decision`. Contract violations (missing column,
  extra column, wrong dtype, NaN) surface as `ContractError` from the boundary;
  a missing `TransactionID` raises a precise `BatchError`.
- `ieee_cis_fraud_detection/monitoring/drift_store.py` — the drift
  current-window store: `append_scores()`/`read_store()` over
  `data/monitoring/current_window.csv` (gitignored runtime data). The scored
  output is appended verbatim as a pure append log — the honest data source
  ticket 08 time-slices into the "current" half of the drift window. `score_csv`
  appends only when a store path is given, so the CLI wires the real store and
  library callers opt in (no hidden side effects). The scored-row schema
  (`TransactionID`, `score`, `decision`) is defined once as `STORE_COLUMNS` and
  shared with the batch output (`OUTPUT_COLUMNS`), so the store and the output
  it feeds cannot drift apart.
- `tests/conftest.py` — the tiny-contract serving scaffold
  (`FEATURE_COLUMNS`/`CATEGORICAL_COLUMNS`/`THRESHOLD`/`make_boundary`/
  `make_frame`) extracted so the scoring, API, and batch test modules no longer
  copy-paste it.
- `CONTEXT.md` — added the "Drift current-window store" glossary term.
- Tests (TDD, 11 new, suite at 54 passing): hermetic tests inject a tiny
  `ScoringBoundary` and assert the output columns/decisions, decisions flip at
  the threshold, missing `TransactionID` → `BatchError`, contract violations
  (extra column, NaN) → `ContractError`, and the drift store accumulates
  across batches. A CLI smoke test runs the command (via typer `CliRunner`)
  on the committed 218-column fixture through the real champion and asserts
  the output columns exactly `TransactionID, score, decision`, the scores
  reproduce `expected_scores.csv` (rtol 1e-6), and the scored output is what
  lands in the drift store.

**Design notes / reconciliations with the master spec**: the ticket's "CSV in
(218 cols)" is read as the 218 feature columns plus the `TransactionID` row key
(User Story 3 requires `TransactionID` in the output, so the key must be present
in the input); the CLI help says so explicitly. The drift store is deliberately
a pure append log — de-duplication/windowing is ticket 08's concern, so
re-scoring never silently drops a newer score (would mask drift). The batch
scorer is not re-exported from `serving/__init__.py`: eagerly importing a module
designed for `python -m` invocation makes runpy double-execute it; `score_csv`
/`BatchError` are imported from the submodule directly (documented in the
package docstring).
