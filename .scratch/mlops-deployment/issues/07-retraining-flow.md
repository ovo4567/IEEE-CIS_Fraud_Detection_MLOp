# 07: Retraining flow (Prefect)

**What to build:** The retraining half of the closed loop, as a Prefect flow: on a trigger (drift alarm OR accumulated-volume ≥ default ~5,000 scored transactions, configurable), build the retraining corpus (history + revealed labels from the scored stream), train a challenger, evaluate it against the champion on the shared test set, decide promotion via the control-plane logic, transition the registry stage, and make the served model pick up the update.

**Blocked by:** 02: Seed champion pipeline, 05: Batch scorer, 06: Control-plane logic (Seam 2)

**Status:** ready-for-agent

- [ ] Trigger fires on accumulated-volume threshold
- [ ] Retraining corpus built with labels revealed per the reveal lag
- [ ] Challenger trained and evaluated against the champion on the same test set
- [ ] Promotion/Staging decision applied via the statistical gate
- [ ] Registry stage transitions and the served model updates
