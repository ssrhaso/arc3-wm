"""Tests for ``arc3_wm.rhae`` — Relative Human Action Efficiency.

Spec source: ``docs/arc-agi-3/methodology.md``. Per D6, this module is a
reference implementation: ``arc.get_scorecard().score`` is the
source of truth at training time, but we keep a pure-function
re-implementation for (a) per-checkpoint logging without re-instantiating
``Arcade`` and (b) a sanity-test fixture against worked examples.

Formula:

    level_score = min((human / ai)^2, 1.15)
    game_score  = sum(level_score_i * i, completed) / sum(i, all levels)
    total_score = mean(game_scores)

Key consequences locked in by these tests:
- Per-level cap at 1.15× human baseline.
- Failing the final level caps the game score (the largest-weight term
  drops from the numerator).
- Total = simple arithmetic mean across games, no weighting.
"""
from __future__ import annotations

import math

import pytest

from arc3_wm.rhae import (
    LEVEL_SCORE_CAP,
    game_score,
    level_score,
    total_score,
)


# ---------------------------------------------------------------------------
# level_score — per-level efficiency, capped at 1.15
# ---------------------------------------------------------------------------


def test_level_score_match_baseline():
    """Human=10, AI=10 → exactly 1.0 (100%)."""
    assert level_score(human_baseline_actions=10, ai_actions=10) == 1.0


def test_level_score_double_baseline():
    """Human=10, AI=20 → 0.25 (methodology.md example)."""
    assert level_score(10, 20) == 0.25


def test_level_score_ten_x_baseline():
    """Human=10, AI=100 → 0.01 (methodology.md example)."""
    assert level_score(10, 100) == pytest.approx(0.01)


def test_level_score_caps_at_1_15_when_ai_faster():
    """AI faster than human: uncapped (10/5)^2 = 4.0, cap kicks in to 1.15."""
    assert level_score(10, 5) == LEVEL_SCORE_CAP == 1.15


def test_level_score_caps_at_1_15_extreme():
    """Extreme shortcut: human=100, ai=1 → uncapped 10000, capped to 1.15."""
    assert level_score(100, 1) == LEVEL_SCORE_CAP


def test_level_score_just_below_cap_not_capped():
    """When (human/ai)^2 < 1.15, the cap doesn't apply."""
    # human=11, ai=10 → (11/10)^2 = 1.21 > 1.15 → capped
    assert level_score(11, 10) == LEVEL_SCORE_CAP
    # human=10, ai=10 → 1.0 < 1.15 → uncapped
    assert level_score(10, 10) == 1.0


def test_level_score_rejects_zero_or_negative_ai_actions():
    """AI with zero actions on a completed level is nonsensical."""
    with pytest.raises(ValueError, match="ai_actions"):
        level_score(10, 0)
    with pytest.raises(ValueError, match="ai_actions"):
        level_score(10, -1)


def test_level_score_rejects_zero_or_negative_human_baseline():
    """Baseline must be a positive action count."""
    with pytest.raises(ValueError, match="human_baseline_actions"):
        level_score(0, 10)
    with pytest.raises(ValueError, match="human_baseline_actions"):
        level_score(-5, 10)


# ---------------------------------------------------------------------------
# game_score — weighted mean, weights are 1-indexed level numbers
# ---------------------------------------------------------------------------


def test_game_score_methodology_example():
    """methodology.md: 5-level game, AI completes only first 4 perfectly →
    max_game_score = (1+2+3+4)/(1+2+3+4+5) = 10/15 = 0.6667."""
    score = game_score({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, total_levels=5)
    assert score == pytest.approx(10 / 15)


def test_game_score_all_levels_perfect():
    """All levels at 1.0 → game score 1.0 (denominator = numerator)."""
    score = game_score({i: 1.0 for i in range(1, 6)}, total_levels=5)
    assert score == 1.0


def test_game_score_no_levels_completed():
    """Empty dict → 0.0 (numerator empty, denominator > 0)."""
    assert game_score({}, total_levels=5) == 0.0


def test_game_score_only_first_level():
    """{1: 1.0} on a 5-level game → 1 / (1+2+3+4+5) = 1/15."""
    score = game_score({1: 1.0}, total_levels=5)
    assert score == pytest.approx(1 / 15)


def test_game_score_only_last_level_dominates():
    """{5: 1.0} on a 5-level game → 5/15 — the highest-weighted term."""
    score = game_score({5: 1.0}, total_levels=5)
    assert score == pytest.approx(5 / 15)


def test_game_score_failing_final_level_caps():
    """Locked-in consequence from methodology.md: missing the final level
    caps the game score regardless of efficiency on prior levels."""
    # 7-level game, levels 1..6 perfectly at cap (1.15 each), level 7 missed.
    completed = {i: LEVEL_SCORE_CAP for i in range(1, 7)}
    score_missing_7 = game_score(completed, total_levels=7)
    # Compare against completing all 7 perfectly at cap.
    completed_all = {i: LEVEL_SCORE_CAP for i in range(1, 8)}
    score_all_7 = game_score(completed_all, total_levels=7)
    assert score_missing_7 < score_all_7
    # And the cap from the missed-final logic: numerator omits 7*1.15.
    expected = sum(i * LEVEL_SCORE_CAP for i in range(1, 7)) / sum(range(1, 8))
    assert score_missing_7 == pytest.approx(expected)


def test_game_score_handles_none_dict():
    """``None`` is treated as an empty dict (no levels completed)."""
    assert game_score(None, total_levels=5) == 0.0


def test_game_score_rejects_invalid_total_levels():
    with pytest.raises(ValueError, match="total_levels"):
        game_score({}, total_levels=0)
    with pytest.raises(ValueError, match="total_levels"):
        game_score({}, total_levels=-1)


def test_game_score_rejects_out_of_range_level_keys():
    """Level index outside [1, total_levels] is a programming error."""
    with pytest.raises(ValueError, match="level"):
        game_score({0: 1.0}, total_levels=5)
    with pytest.raises(ValueError, match="level"):
        game_score({6: 1.0}, total_levels=5)


# ---------------------------------------------------------------------------
# total_score — arithmetic mean across games
# ---------------------------------------------------------------------------


def test_total_score_mean_of_games():
    assert total_score([0.5, 0.7, 0.3]) == pytest.approx(0.5)


def test_total_score_single_game():
    assert total_score([0.42]) == 0.42


def test_total_score_empty_returns_zero():
    """No games scored → 0% (per methodology.md interpretation table)."""
    assert total_score([]) == 0.0


def test_total_score_perfect():
    """All games at 1.0 → 1.0 (the 100% interpretation)."""
    assert total_score([1.0] * 25) == 1.0


def test_total_score_zero_when_no_levels_completed():
    """All games at 0.0 → 0.0 (the 0% interpretation)."""
    assert total_score([0.0] * 25) == 0.0


# ---------------------------------------------------------------------------
# End-to-end sanity — a worked scenario through all three functions
# ---------------------------------------------------------------------------


def test_end_to_end_two_games():
    """Composition smoke: build per-level scores, aggregate to game,
    aggregate games to total. Asserts the three functions compose
    without surprises (e.g. no double-capping, no off-by-one in weights)."""
    # Game A: 3 levels, AI matches human on level 1, doubles on level 2,
    # fails level 3. Baselines: [10, 20, 30]. AI actions: [10, 40, ∞].
    game_a_levels = {
        1: level_score(10, 10),  # 1.0
        2: level_score(20, 40),  # 0.25
    }
    game_a = game_score(game_a_levels, total_levels=3)
    # numerator = 1*1.0 + 2*0.25 = 1.5; denominator = 1+2+3 = 6 → 0.25
    assert game_a == pytest.approx(1.5 / 6)

    # Game B: 2 levels, both perfect.
    game_b = game_score({1: 1.0, 2: 1.0}, total_levels=2)
    assert game_b == 1.0

    total = total_score([game_a, game_b])
    assert total == pytest.approx((1.5 / 6 + 1.0) / 2)


def test_score_outputs_are_finite():
    """Defensive: under the cap and validated inputs, scores are always
    finite floats in [0, LEVEL_SCORE_CAP] (level) or [0, 1] (game/total
    given level scores ≤ 1)."""
    s = level_score(10, 1000)
    assert math.isfinite(s)
    assert 0 <= s <= LEVEL_SCORE_CAP
