# 07: Retraining flow (Prefect)

**What to build:** The retraining half of the closed loop, as a Prefect flow: on a trigger (drift alarm OR accumulated-volume ≥ default ~5,000 scored transactions, configurable), build the retraining corpus (history + revealed labels from the scored stream), train a challenger, evaluate it against the champion on the shared test set, decide promotion via the control-plane logic, transition the registry stage, and make the served model pick up the update.

**Blocked by:** 02: Seed champion pipeline, 05: Batch scorer, 06: Control-plane logic (Seam 2)

**Status:** resolved

- [x] Trigger fires on accumulated-volume threshold
- [x] Retraining corpus built with labels revealed per the reveal lag
- [x] Challenger trained and evaluated against the champion on the same test set
- [x] Promotion/Staging decision applied via the statistical gate
- [x] Registry stage transitions and the served model updates

## Answer

Implemented on `main` (this ticket). The retraining half of the closed loop,
as a Prefect flow wrapping the pure control-plane decisions (Seam 2, ticket
06):

- `ieee_cis_fraud_detection/orchestration/retraining.py` — `retraining_flow`
  (Prefect `@flow`) plus the callable steps, all hermetic-testable:
  `should_retrain` (accumulated volume since the last retrain >= default 5,000
  OR drift alarm), `accumulated_volume_since_last_retrain` (drift-store rows
  since the retrain-state checkpoint), `scored_transaction_ids`,
  `prepare_retraining_inputs` (history + revealed scored stream -> corpus; the
  original test slice is the shared benchmark), `train_challenger` (fresh
  70/15/15 of the corpus, operating threshold under the cost ratio),
  `register_challenger` (next registry version + stage transition + serve on
  promotion), `publish_served_model`. CLI: `make retrain` / `python -m
  ieee_cis_fraud_detection.orchestration.retraining`.
- **Registry stage transitions**: promoted challenger -> `Production`, every
  previous version archived; non-promoted -> `Staging`. Metrics logged:
  `val_auc`, `auc_champion`, `auc_challenger`, `promotion_pvalue`.
- **Served model picks up the update**: promotion publishes the pyfunc to
  `models/serving/champion_model` (gitignored); `serving.scoring.load_model()`
  prefers that path over the committed seed once it exists, so the API and
  batch scorer serve a promoted model with no redeploy.
- **Retrain state** (`data/monitoring/retrain_state.json`) records the
  drift-store row count at each retrain so the accumulated volume is measured
  "since the last retrain".
- Dep: `prefect>=3.8.4` added. Tests (TDD, 22 new, suite 104 passing):
  trigger rule, volume checkpointing, corpus/reveal-lag, challenger fit,
  Staging-vs-Production transitions, served-model resolution, and an
  end-to-end flow run that promotes a 40-tree challenger over a stump champion
  (DeLong p < 0.007 across seeds).
- Review hardening: each version's artifact is version-unique
  (`challengers/v{N}`) so the registry audit trail survives repeated
  retrains; a promotion archives the superseded champion + seed's unstaged
  version but leaves Staging challengers under review alone; republishing the
  served model replaces the prior artifact.
- Known trade-off: MLflow 3.x deprecates registry stages (alias migration is a
  future cleanup); the flow uses stages per the ticket + ADR-0004 vocabulary.
