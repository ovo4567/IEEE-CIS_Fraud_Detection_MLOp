# 02: Seed champion pipeline

**What to build:** The reproducible path that turns raw processed features into a registered, committed seed model — the champion. Chronologically split the data 70% train / 15% test / 15% production stream (ADR-0003); re-fit the `finetuned_lgbm` hyperparameters from the MLflow experiment on the 70% split; compute the operating threshold from the 10:1 (missed-fraud : blocked-good) cost ratio on the test set; log the model as an MLflow `pyfunc` (ADR-0002) carrying the full feature transform + booster + threshold; register it as champion version 1 in the clean registry; and commit the seed artifact so `make demo` works offline on a fresh clone.

**Blocked by:** 01: Repo restructure into MLOps layout

**Status:** ready-for-agent

- [ ] Chronological 70/15/15 split produced from the processed features
- [ ] `finetuned_lgbm` re-fit on the 70% split from registered hyperparameters
- [ ] Operating threshold computed on the test set under the 10:1 cost ratio
- [ ] Model registered as champion v1 in a clean registry
- [ ] Seed artifact committed to the repo (loadable offline without re-training)
