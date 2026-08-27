# 08: Stream simulator + drift monitoring (Evidently)

**What to build:** The monitoring half of the closed loop: a stream simulator (Prefect flow) that replays the production stream slice through the real-time API at accelerated cadence for the live demo, alongside scheduled batch scoring that accumulates a time-sliced drift window; Evidently reports comparing feature + score distributions of the current window against the training reference; the aggregate drift alarm feeds the retraining trigger.

**Blocked by:** 04: Real-time API, 05: Batch scorer, 06: Control-plane logic (Seam 2)

**Status:** ready-for-agent

- [ ] Simulator replays the stream through the API (live scoring visible)
- [ ] Scheduled batch scoring accumulates the time-sliced current window
- [ ] Evidently report compares current window vs training reference (feature + score drift)
- [ ] Aggregate drift alarm feeds the retraining trigger
