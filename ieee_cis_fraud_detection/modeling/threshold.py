"""Operating-threshold selection under a (missed-fraud : blocked-good) cost
ratio (ADR-0003, ticket 02).

The operating threshold is the decision cut-off on the fraud score, chosen on
the test set and stored with the model. Here it is the score that minimizes
expected cost on the scored set under the business asymmetry: missing a fraud
is ``cost_ratio`` times more expensive than blocking a good transaction.
"""

from __future__ import annotations

import numpy as np

DEFAULT_COST_RATIO = 10.0  # a missed fraud costs 10x a blocked good transaction


def operating_threshold(
    y_true,
    y_score,
    *,
    cost_ratio: float = DEFAULT_COST_RATIO,
) -> float:
    """Return the fraud-score threshold minimizing expected cost.

    With ``cost_ratio = C_FN / C_FP`` (C_FP normalized to 1), the expected cost
    of flagging rows with ``score >= t`` is::

        C_FN * (# fraud rows with score < t) + (# good rows with score >= t)

    Candidate thresholds are the distinct observed scores. Among thresholds
    achieving the minimum cost, the lowest is chosen — it leans toward catching
    fraud, the costlier failure mode.

    Raises:
        ValueError: if ``cost_ratio <= 0`` or the input is empty.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_score) == 0:
        raise ValueError("operating_threshold requires at least one score")
    if cost_ratio <= 0:
        raise ValueError("cost_ratio must be positive")

    candidates = np.unique(y_score)
    fraud = y_true == 1
    good = ~fraud

    costs = np.empty(len(candidates), dtype=float)
    for i, t in enumerate(candidates):
        missed = (fraud & (y_score < t)).sum()
        blocked = (good & (y_score >= t)).sum()
        costs[i] = missed * cost_ratio + blocked

    # Lowest threshold achieving the minimum cost (ties lean toward catching
    # fraud, the costlier failure mode).
    best = int(np.where(costs == costs.min())[0][0])
    return float(candidates[best])


def expected_cost(
    y_true,
    y_score,
    threshold: float,
    cost_ratio: float = DEFAULT_COST_RATIO,
) -> float:
    """Total expected cost of flagging rows with ``score >= threshold``.

    Shares the cost model with :func:`operating_threshold`: a missed fraud
    costs ``cost_ratio`` and a blocked good costs 1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    missed = (y_true == 1) & (y_score < threshold)
    blocked = (y_true == 0) & (y_score >= threshold)
    return float(missed.sum() * cost_ratio + blocked.sum())
