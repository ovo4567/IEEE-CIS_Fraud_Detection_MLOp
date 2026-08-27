# 05: Batch scorer

**What to build:** A batch serving surface — a CLI that scores a CSV of transactions (218 columns) through the scoring & decision boundary and writes a CSV out with `TransactionID`, `score`, and `decision`. Its output appends to the drift current-window store, making it the honest data source for monitoring. Runs in parallel with the real-time API.

**Blocked by:** 03: Scoring & decision boundary (Seam 1)

**Status:** ready-for-agent

- [ ] CSV in (218 cols) → CSV out with `TransactionID`, `score`, `decision`
- [ ] Output appended to the drift current-window store
- [ ] Smoke test runs the CLI on a small CSV and asserts the output columns
