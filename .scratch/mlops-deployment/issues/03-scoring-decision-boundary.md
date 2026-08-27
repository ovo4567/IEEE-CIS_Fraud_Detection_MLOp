# 03: Scoring & decision boundary (Seam 1)

**What to build:** The deep module both serving surfaces share — a single interface that takes a transaction (or batch) and returns `{score, decision, threshold}`. It enforces the strict 218-column feature contract (reject missing column, extra column, wrong dtype, or NaN with a precise error), loads the pyfunc model, and applies the operating threshold. Both the real-time API and the batch scorer become thin adapters over it.

**Blocked by:** 02: Seed champion pipeline

**Status:** ready-for-agent

- [ ] Missing column rejected
- [ ] Extra column rejected
- [ ] Wrong dtype rejected
- [ ] NaN rejected
- [ ] Category coercion applied correctly (9 categorical columns to `category`)
- [ ] Score correct on a small committed fixture
- [ ] Decision flips at the operating threshold
