"""Tests for thresholds.py: confusion counts, costs, selection rule, theory."""

import numpy as np
import pytest

from churn_platform.thresholds import (
    confusion_at,
    cost_sensitivity,
    select_threshold,
    theoretical_threshold,
    threshold_table,
)

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.6, 0.2, 0.7, 0.4, 0.3, 0.1, 0.05])


def test_confusion_at_threshold():
    assert confusion_at(Y, P, 0.5) == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    # >= 0.15 flags 0.9, 0.6, 0.2 (churners) and 0.7, 0.4, 0.3 (non-churners)
    assert confusion_at(Y, P, 0.15) == {"tp": 3, "fp": 3, "fn": 0, "tn": 2}


def test_threshold_is_inclusive():
    assert confusion_at([1], [0.5], 0.5)["tp"] == 1


def test_cost_formula():
    table = threshold_table(Y, P, fp_cost=2, fn_cost=7, thresholds=[0.5, 0.15])
    row = table.set_index("threshold").loc[0.5]
    assert row["cost"] == 1 * 2 + 1 * 7
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(2 / 3)
    assert table.set_index("threshold").loc[0.15, "cost"] == 3 * 2 + 0 * 7


def test_precision_undefined_when_nobody_flagged():
    row = threshold_table(Y, P, thresholds=[0.95]).iloc[0]
    assert row["flagged"] == 0 and np.isnan(row["precision"]) and row["f1"] == 0.0


def test_select_threshold_minimum_cost_ties_to_higher_threshold():
    table = threshold_table(Y, P, fp_cost=1, fn_cost=1, thresholds=[0.35, 0.5, 0.55, 0.59, 0.65])
    # 0.5, 0.55 and 0.59 all flag {0.9, 0.6, 0.7}: TP 2, FP 1, FN 1 -> cost 2 (the minimum).
    # 0.35 also flags 0.4 (cost 3); 0.65 drops 0.6 (cost 3). The rule picks the highest tied threshold.
    best = select_threshold(table)
    assert best["threshold"] == 0.59
    assert best["cost"] == 2
    assert best["near_optimal_min"] == 0.5 and best["near_optimal_max"] == 0.59


def test_theoretical_threshold():
    assert theoretical_threshold(1, 1) == 0.5
    assert theoretical_threshold(1, 5) == pytest.approx(1 / 6)


def test_selected_threshold_close_to_theory_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, 50_000) < p).astype(int)   # labels drawn FROM p -> perfectly calibrated
    s = cost_sensitivity(y, p, fp_cost=1, ratios=(1, 3, 5))
    assert (s["selected_threshold"] - s["theoretical_threshold"]).abs().max() < 0.03


def test_higher_fn_cost_never_raises_threshold():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5_000)
    y = (rng.uniform(0, 1, 5_000) < p).astype(int)
    s = cost_sensitivity(y, p, ratios=(1, 2, 3, 5, 10))
    assert s["selected_threshold"].is_monotonic_decreasing
