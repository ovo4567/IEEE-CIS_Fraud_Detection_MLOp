# 04: Real-time API

**What to build:** A real-time serving surface — `POST /predict` accepting one transaction's 218 fields and returning `{score, decision, threshold}`, implemented as a thin adapter over the scoring & decision boundary. Requests that violate the feature contract return a precise 400 error rather than being scored.

**Blocked by:** 03: Scoring & decision boundary (Seam 1)

**Status:** ready-for-agent

- [ ] `POST /predict` returns `{score, decision, threshold}` for a valid transaction
- [ ] Contract violations return a precise 400
- [ ] In-process smoke test exercises a successful request and a rejection
