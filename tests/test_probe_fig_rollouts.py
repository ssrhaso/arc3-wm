"""Tests for ``scripts/probe_fig_rollouts.py`` - the qualitative rollout strip.

The figure itself is a plotting job, but the *window choice* is a claim: the
paper's point is that the imagined board locks onto a wrong configuration and
holds it while the truth moves, and that reads correctly only on a
representative window. The busiest window flatters the model. So the selection
rule is pinned here rather than left to whatever ``argsort`` happens to do.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.probe_fig_rollouts import build_figure, select_window


def _synthetic(churns: list[int], horizon: int = 4, hw: int = 8):
    """Windows whose ground truth differs from its context in ``churn`` cells.

    All of the change lands on the first future frame; the rest of the horizon
    repeats it, so total churn over the window is exactly ``churn``.
    """
    n = len(churns)
    ctx = np.zeros((n, hw, hw), np.int16)
    true = np.zeros((n, horizon, hw, hw), np.int16)
    for i, c in enumerate(churns):
        flat = true[i, 0].reshape(-1)
        flat[:c] = 1
        true[i, 1:] = true[i, 0]
    return true, ctx


def test_median_picks_middle_of_the_churn_distribution():
    true, ctx = _synthetic([0, 5, 40, 9, 2])
    # sorted churn -> [0, 2, 5, 9, 40]; index 5 // 2 == 2 -> the value 5.
    assert select_window(true, ctx, "median") == 1


def test_max_picks_the_busiest_window():
    true, ctx = _synthetic([0, 5, 40, 9, 2])
    assert select_window(true, ctx, "max") == 2


def test_median_is_not_the_max_when_distribution_is_skewed():
    """One runaway window must not drag the representative choice with it."""
    true, ctx = _synthetic([1, 1, 1, 1, 63])
    assert select_window(true, ctx, "median") != select_window(true, ctx, "max")


def test_selection_is_deterministic():
    true, ctx = _synthetic([3, 3, 3, 3])
    assert select_window(true, ctx) == select_window(true, ctx)


def test_single_window_is_selectable():
    true, ctx = _synthetic([7])
    assert select_window(true, ctx, "median") == 0
    assert select_window(true, ctx, "max") == 0


def test_unknown_mode_raises():
    true, ctx = _synthetic([1, 2])
    with pytest.raises(ValueError, match="unknown window mode"):
        select_window(true, ctx, "busiest")


def test_default_mode_is_median():
    true, ctx = _synthetic([0, 5, 40, 9, 2])
    assert select_window(true, ctx) == select_window(true, ctx, "median")


def test_build_figure_lays_out_label_copy_and_both_rows():
    horizon = 8
    rng = np.random.default_rng(0)
    q_true = rng.integers(0, 16, (horizon, 8, 8), dtype=np.int16)
    q_pred = rng.integers(0, 16, (horizon, 8, 8), dtype=np.int16)
    q_ctx = rng.integers(0, 16, (8, 8), dtype=np.int16)
    fig = build_figure(q_true, q_pred, q_ctx, horizon=horizon)
    try:
        # label column + copy column + one column per horizon step, two rows.
        assert len(fig.axes) == 2 * (horizon + 2)
        # Both rows of frames, plus the copy baseline shown once. If the lower
        # copy cell were ever filled this would be 2 * horizon + 2.
        assert sum(len(ax.images) for ax in fig.axes) == 2 * horizon + 1
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
