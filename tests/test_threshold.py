"""Tests for operating-threshold selection under the (missed-fraud :
blocked-good) cost ratio (ticket 02, ADR-0003).

Seam B: `ieee_cis_fraud_detection.modeling.threshold.operating_threshold`.

The threshold minimizes expected cost on the given scored set. With
``cost_ratio = C_FN / C_FP`` (default 10), a missed fraud costs 10x a blocked
good transaction; candidate thresholds are the distinct observed scores, and
the lowest threshold achieving the minimum cost is chosen.
"""

import numpy as np
import pytest

from ieee_cis_fraud_detection.modeling.threshold import operating_threshold


def test_perfect_separation_thresholds_at_the_fraud_score() -> None:
    # Frauds score 0.9, goods score 0.1: flagging at 0.9 costs nothing.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.9, 0.1, 0.1])
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.9


def test_block_aggressive_when_missing_fraud_is_expensive() -> None:
    # A fraud scores just below a good (mis-ordered relative to labels). At
    # 10:1 the block-aggressive lower threshold (0.07) is chosen: missing the
    # fraud costs 10 while blocking the one good costs 1.
    y_true = np.array([1, 0])
    y_score = np.array([0.07, 0.08])
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.07


def test_cost_ratio_changes_the_choice() -> None:
    # A low-scoring fraud (0.2) sits below a good (0.5). When a missed fraud is
    # expensive (10:1) we block the good to catch the fraud (threshold 0.2);
    # when a missed fraud is cheap (0.5:1) we spare the good and accept the
    # miss (threshold 0.9).
    y_true = np.array([1, 1, 0])
    y_score = np.array([0.9, 0.2, 0.5])
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.2
    assert operating_threshold(y_true, y_score, cost_ratio=0.5) == 0.9


def test_threshold_flags_frauds_and_spares_goods() -> None:
    # Frauds at 0.5 (above the 1/11 Bayes boundary), goods at 0.05 (below it):
    # the empirical minimizer flags the frauds and spares the goods.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.5, 0.5, 0.05, 0.05])
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.5


def test_tie_break_picks_lowest_threshold_at_minimum_cost() -> None:
    # cost_ratio=10: t=0.2 and t=0.9 both cost 1 (the minimum); the lowest is
    # chosen, leaning toward catching the fraud.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.2


def test_all_good_blocks_fewest_rows() -> None:
    y_true = np.zeros(5, dtype=int)
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    # Every threshold blocks some goods; the highest score (0.9) blocks the
    # fewest at the minimum cost.
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.9


def test_all_fraud_flags_everything() -> None:
    y_true = np.ones(5, dtype=int)
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Flagging at the lowest score catches every fraud at zero cost.
    assert operating_threshold(y_true, y_score, cost_ratio=10) == 0.1


def test_cost_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cost_ratio"):
        operating_threshold(np.array([1]), np.array([0.5]), cost_ratio=0)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one score"):
        operating_threshold(np.array([], dtype=int), np.array([], dtype=float))
