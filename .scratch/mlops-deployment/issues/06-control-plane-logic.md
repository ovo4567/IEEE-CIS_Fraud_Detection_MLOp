# 06: Control-plane logic (Seam 2)

**What to build:** The pure, infra-free functions that encode the loop decisions: `build_retraining_corpus` (history + scored stream with labels revealed per the 7-day reveal lag), `decide_promotion` (statistical gate — DeLong test / paired bootstrap on the shared test set, per ADR-0004), and `decide_drift` (aggregate rule: ≥10% of features drift OR score-distance threshold crossed). No Prefect, no Docker — just testable logic.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Reveal-lag cutoff boundary correct (labels for stream rows older than the lag)
- [x] Promotion decides correctly on constructed significant vs insignificant score pairs
- [x] Drift aggregate rule decides correctly on constructed stats

## Answer

Implemented on `main` (this ticket). The pure, infra-free decision functions
live in `ieee_cis_fraud_detection/orchestration/control_plane.py` — no Prefect,
no Docker, no model loading, so the loop's rules are testable in isolation
(Seam 2). The Prefect flows (tickets 07/08) wrap these.

- **`build_retraining_corpus(history, scored_stream, *, reveal_lag=7d, now=None)`** —
  assembles the retraining corpus: every history row, plus the scored-stream
  rows whose labels are revealed per the reveal lag. A stream row's label is
  usable once it is older than the lag: rows with `TransactionDT <= now -
  reveal_lag` join the corpus with their labels; newer rows stay label-free and
  are excluded. `now` is the reference "current time" (seconds, defaulting to
  the newest timestamp in the combined data). The cutoff boundary is inclusive
  (a row exactly at `now - reveal_lag` has served its full lag and is
  revealed). Raises `ValueError` on a negative lag or missing time/label
  columns. Fact-checked against the real data: a 7-day lag over the 30.8-day
  production stream reveals ~80% of stream labels while keeping the newest
  simulated days label-free (spec said ~77% — same regime).
- **`decide_promotion(y_true, champion_scores, challenger_scores, *, alpha=0.05)`** —
  the statistical promotion gate (ADR-0004): the **DeLong test** (DeLong et
  al., 1988, structural-components variance) for the correlated AUC difference
  on the shared test set, turned into a one-sided z-test for "challenger is
  better". Promotes only when `p_value <= alpha`; identical scores give `se =
  0` and `p_value = 1` (never promoted). Returns a `PromotionDecision` with
  `promote`, `auc_champion`, `auc_challenger`, `p_value`, `alpha`. Only DeLong
  is implemented (the ADR's "DeLong test / paired bootstrap" allows either —
  DeLong is deterministic, which suits the constructed-pair acceptance test).
  Independently verified: AUCs match `sklearn.metrics.roc_auc_score` exactly,
  and the one-sided p-values agree with a paired-bootstrap 95% CI of the AUC
  difference (significant pair CI excludes 0 → promote; near-identical pair CI
  spans 0 → no promote).
- **`decide_drift(feature_pvalues, *, score_distance, score_distance_threshold=None, …)`** —
  the aggregate alarm rule: a feature drifts when its p-value is below
  `feature_pvalue_threshold` (0.05); the feature leg alarms when the drifted
  fraction is at least `feature_fraction_threshold` (0.10, inclusive); the
  score leg alarms when `score_distance >= score_distance_threshold` (`None`
  disables that leg); the overall rule is an OR. Returns a `DriftDecision`
  breakdown (`alarm`, `n_drifted`, `n_features`, `feature_fraction`,
  `feature_alarm`, `score_alarm`). Raises on out-of-range thresholds / negative
  distance.

- **Tests** (TDD, 28 new, suite at 82 passing): `tests/test_control_plane.py`
  is hermetic and covers the three acceptance criteria — reveal-lag cutoff
  boundary (exact-boundary row included, zero-lag reveals all, default `now`,
  history never dropped, negative lag / missing columns rejected),
  promotion on constructed significant vs insignificant pairs (perfect-vs-
  random promotes, near-identical and worse challengers do not, identical
  scores `p = 1`, `alpha` tightens the gate, length/class/alpha validation),
  and the drift aggregate rule on constructed stats (10%-exact-threshold
  alarms, below-threshold does not, score-distance-only alarm, boundary
  inclusive, `None` threshold disables the score leg, empty feature list,
  threshold validation). Lint clean (`ruff`); no new dependencies (`scipy`
  arrives transitively via scikit-learn and was already present).
