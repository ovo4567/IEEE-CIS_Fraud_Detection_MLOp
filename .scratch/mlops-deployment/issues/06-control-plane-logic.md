# 06: Control-plane logic (Seam 2)

**What to build:** The pure, infra-free functions that encode the loop decisions: `build_retraining_corpus` (history + scored stream with labels revealed per the 7-day reveal lag), `decide_promotion` (statistical gate — DeLong test / paired bootstrap on the shared test set, per ADR-0004), and `decide_drift` (aggregate rule: ≥10% of features drift OR score-distance threshold crossed). No Prefect, no Docker — just testable logic.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Reveal-lag cutoff boundary correct (labels for stream rows older than the lag)
- [ ] Promotion decides correctly on constructed significant vs insignificant score pairs
- [ ] Drift aggregate rule decides correctly on constructed stats
