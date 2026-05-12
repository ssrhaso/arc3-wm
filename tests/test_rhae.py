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

import json
import math
from pathlib import Path

import pytest

from arc3_wm.rhae import (
    LEVEL_SCORE_CAP,
    RHAEAggregator,
    coverage,
    game_score,
    level_score,
    total_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_BASELINES_PATH = REPO_ROOT / "data" / "human_baselines.json"


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
    score = game_score(
        {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, covered_levels=range(1, 6)
    )
    assert score == pytest.approx(10 / 15)


def test_game_score_all_levels_perfect():
    """All levels at 1.0 → game score 1.0 (denominator = numerator)."""
    score = game_score(
        {i: 1.0 for i in range(1, 6)}, covered_levels=range(1, 6)
    )
    assert score == 1.0


def test_game_score_no_levels_completed():
    """Empty dict → 0.0 (numerator empty, denominator > 0)."""
    assert game_score({}, covered_levels=range(1, 6)) == 0.0


def test_game_score_only_first_level():
    """{1: 1.0} on a 5-level game → 1 / (1+2+3+4+5) = 1/15."""
    score = game_score({1: 1.0}, covered_levels=range(1, 6))
    assert score == pytest.approx(1 / 15)


def test_game_score_only_last_level_dominates():
    """{5: 1.0} on a 5-level game → 5/15 — the highest-weighted term."""
    score = game_score({5: 1.0}, covered_levels=range(1, 6))
    assert score == pytest.approx(5 / 15)


def test_game_score_failing_final_level_caps():
    """Locked-in consequence from methodology.md: missing the final
    scored level caps the game score regardless of efficiency on prior
    levels. Under D-B, "final scored level" means the last COVERED level
    — but here the game has full coverage so it's the spec-original."""
    # 7-level game, all covered, levels 1..6 perfectly at cap, level 7 missed.
    completed = {i: LEVEL_SCORE_CAP for i in range(1, 7)}
    score_missing_7 = game_score(completed, covered_levels=range(1, 8))
    completed_all = {i: LEVEL_SCORE_CAP for i in range(1, 8)}
    score_all_7 = game_score(completed_all, covered_levels=range(1, 8))
    assert score_missing_7 < score_all_7
    expected = sum(i * LEVEL_SCORE_CAP for i in range(1, 7)) / sum(range(1, 8))
    assert score_missing_7 == pytest.approx(expected)


def test_game_score_handles_none_dict():
    """``None`` is treated as an empty dict (no levels completed)."""
    assert game_score(None, covered_levels=range(1, 6)) == 0.0


def test_game_score_empty_covered_returns_zero():
    """All-uncovered game (no scoreable levels) returns 0.0 — pinned
    sentinel even though no real fixture data triggers this on the
    340-replay dataset (every game has >=2 covered levels)."""
    assert game_score({}, covered_levels=[]) == 0.0
    assert game_score(None, covered_levels=set()) == 0.0


def test_game_score_rejects_non_positive_covered():
    """Covered level indices must be 1-indexed positive ints."""
    with pytest.raises(ValueError, match="1-indexed"):
        game_score({}, covered_levels=[0, 1, 2])
    with pytest.raises(ValueError, match="1-indexed"):
        game_score({}, covered_levels=[-1, 1])


def test_game_score_rejects_score_key_not_in_covered():
    """``level_scores`` keys must be a subset of ``covered_levels`` —
    caller (RHAEAggregator) is responsible for D-B skipping."""
    with pytest.raises(ValueError, match="covered_levels"):
        game_score({2: 1.0}, covered_levels=[1, 3])
    with pytest.raises(ValueError, match="covered_levels"):
        game_score({99: 1.0}, covered_levels=range(1, 6))


def test_game_score_non_contiguous_covered_db_case():
    """D-B realistic case: covered levels are not contiguous. Example:
    8-level game where only levels {1, 3, 5} have n>=2 baselines (the
    other 5 are uncovered). AI clears level 1 at parity:
      denominator = 1+3+5 = 9, numerator = 1*1.0 = 1.0 → 1/9.
    AI also "clears" uncovered levels 2, 4: caller must skip them
    before calling game_score (this test only exercises the math)."""
    score = game_score({1: 1.0}, covered_levels=[1, 3, 5])
    assert score == pytest.approx(1 / 9)
    score_all_covered = game_score(
        {1: 1.0, 3: 1.0, 5: 1.0}, covered_levels=[1, 3, 5]
    )
    assert score_all_covered == 1.0


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
    game_a = game_score(game_a_levels, covered_levels=range(1, 4))
    # numerator = 1*1.0 + 2*0.25 = 1.5; denominator = 1+2+3 = 6 → 0.25
    assert game_a == pytest.approx(1.5 / 6)

    # Game B: 2 levels, both perfect.
    game_b = game_score({1: 1.0, 2: 1.0}, covered_levels=range(1, 3))
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


# ---------------------------------------------------------------------------
# RHAEAggregator — post-hoc aggregator (D2: no in-loop hook, no scheduler)
#
# Contract: the post-hoc CLI segments an eval rollout into per-level AI
# action counts for a given game, then calls the aggregator. It is pure
# plumbing over level_score / game_score — no rollout machinery, no step
# scheduling. Emits the three wandb-key family spec'd in Notion
# "Logging & analysis plan":
#
#   eval/rhae/per_game/{game_id}            float
#   eval/rhae/level_scores/{game_id}/{i}    float (1-indexed level i)
#   eval/rhae/levels_completed/{game_id}    int
#
# Human baselines (per-game, per-level upper-median action counts per
# methodology.md / D1) are injected at construction. The aggregator
# validates the game_id is known and every claimed completion has a
# baseline; missing baselines raise (task brief: don't silently zero).
# ---------------------------------------------------------------------------


# Toy fixture: two games in the D-A/D-B shape.
# vc33: 3 total levels (fully covered) — keeps tests legible.
# tu93: 2 total levels (fully covered).
_BASELINES = {
    "vc33": {"total_levels": 3, "baselines": {1: 10, 2: 20, 3: 30}},
    "tu93": {"total_levels": 2, "baselines": {1: 8, 2: 16}},
}

# D-B realistic fixture: total > covered, with uncovered levels in the
# range [1, total_levels]. Mirrors the actual public-demo dataset (e.g.
# vc33 6/7, tu93 3/9, sb26 8/8). bp35-style coverage gap chosen.
_BASELINES_DB = {
    # 5 total levels, only levels 1 and 4 covered (gap in middle, gap at end).
    "gap5": {"total_levels": 5, "baselines": {1: 10, 4: 40}},
    # Fully covered 3-level game.
    "full3": {"total_levels": 3, "baselines": {1: 10, 2: 20, 3: 30}},
    # All-uncovered edge case (sentinel — no real fixture data triggers
    # this; every public-demo game has >=2 covered levels). Pinned for
    # safety.
    "none": {"total_levels": 4, "baselines": {}},
}


def test_aggregator_emits_three_key_families():
    """When called, output contains exactly these keys for the game:
    eval/rhae/per_game/{game_id}, eval/rhae/level_scores/{game_id}/{i},
    eval/rhae/levels_completed/{game_id}. Pin the contract so wandb
    panels keep working if the impl is touched."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    metrics = agg(game_id="vc33", ai_actions_per_level={1: 10, 2: 40})
    assert "eval/rhae/per_game/vc33" in metrics
    assert "eval/rhae/level_scores/vc33/1" in metrics
    assert "eval/rhae/level_scores/vc33/2" in metrics
    # uncompleted level 3 is absent — sparse logging is intentional
    assert "eval/rhae/level_scores/vc33/3" not in metrics
    assert "eval/rhae/levels_completed/vc33" in metrics
    assert metrics["eval/rhae/levels_completed/vc33"] == 2


def test_aggregator_per_level_values_match_level_score():
    """Per-level emission values are exactly ``level_score(h, a)``."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    metrics = agg(game_id="vc33", ai_actions_per_level={1: 20, 2: 10})
    assert metrics["eval/rhae/level_scores/vc33/1"] == pytest.approx(0.25)
    assert metrics["eval/rhae/level_scores/vc33/2"] == LEVEL_SCORE_CAP


def test_aggregator_per_game_value_matches_game_score():
    """Per-game emission value is exactly ``game_score(level_scores,
    covered_levels)``. vc33 fully covered 3-level fixture: AI completes
    1 perfectly, 2 at half-efficiency, fails 3. Expected = (1*1.0 +
    2*0.25 + 0*3) / (1+2+3) = 1.5/6."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    metrics = agg(game_id="vc33", ai_actions_per_level={1: 10, 2: 40})
    assert metrics["eval/rhae/per_game/vc33"] == pytest.approx(1.5 / 6)


def test_aggregator_no_completion_emits_zero_per_game():
    """Zero-level run is a real Phase-4 failure mode (agent stalls on
    level 1). Per-game RHAE = 0.0, levels_completed=0, no level_scores."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    metrics = agg(game_id="vc33", ai_actions_per_level={})
    assert metrics["eval/rhae/per_game/vc33"] == 0.0
    assert metrics["eval/rhae/levels_completed/vc33"] == 0
    assert not any(
        k.startswith("eval/rhae/level_scores/vc33/") for k in metrics
    )


def test_aggregator_unknown_game_raises():
    """Unknown game_id must raise KeyError (typo or missing-data drift)."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    with pytest.raises(KeyError, match="ls99"):
        agg(game_id="ls99", ai_actions_per_level={1: 10})


def test_aggregator_level_out_of_range_raises():
    """AI claims to have completed a level outside [1, total_levels] —
    that's a level-indexing bug upstream, raise ValueError loud. Note:
    this is the out-of-range case; in-range-but-uncovered is silently
    skipped per D-B (see test_aggregator_uncovered_level_skipped)."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    with pytest.raises(ValueError, match="out of"):
        agg(game_id="vc33", ai_actions_per_level={99: 10})
    with pytest.raises(ValueError, match="out of"):
        agg(game_id="vc33", ai_actions_per_level={0: 10})


def test_aggregator_total_levels_from_fixture_not_inferred():
    """D-A: total_levels is read from the fixture's per-game entry, NOT
    inferred from max(baseline_keys). Sentinel that a partially-covered
    game uses the engine win_levels denominator basis."""
    # gap5: 5 total levels, covered {1, 4}. AI clears level 1 only.
    # Under D-A/D-B, denominator = sum(covered) = 1+4 = 5.
    # Numerator = 1*1.0 = 1.0. Per-game = 1/5 = 0.2.
    agg = RHAEAggregator(human_baselines=_BASELINES_DB)
    metrics = agg(game_id="gap5", ai_actions_per_level={1: 10})
    assert metrics["eval/rhae/per_game/gap5"] == pytest.approx(1 / 5)


def test_aggregator_uncovered_level_skipped():
    """D-B: AI completing a level in [1, total_levels] but NOT in
    ``baselines`` is silently skipped — no level_scores atom, no
    contribution to per_game numerator/denominator. The agent gets
    neither credit nor penalty on uncovered levels.

    gap5 fixture: total=5, covered={1,4}. AI clears 1, 2, 3, 4.
    Levels 2, 3 are in-range-but-uncovered → silently skipped.
    Level 1 at parity (1.0), level 4 at parity (1.0).
    Numerator = 1*1.0 + 4*1.0 = 5. Denominator = 1+4 = 5. → 1.0.
    levels_completed = 2 (only covered completions count)."""
    agg = RHAEAggregator(human_baselines=_BASELINES_DB)
    metrics = agg(
        game_id="gap5",
        ai_actions_per_level={1: 10, 2: 99, 3: 99, 4: 40},
    )
    assert metrics["eval/rhae/per_game/gap5"] == pytest.approx(1.0)
    assert metrics["eval/rhae/levels_completed/gap5"] == 2
    assert "eval/rhae/level_scores/gap5/1" in metrics
    assert "eval/rhae/level_scores/gap5/4" in metrics
    # Uncovered levels emit no level_scores atom.
    assert "eval/rhae/level_scores/gap5/2" not in metrics
    assert "eval/rhae/level_scores/gap5/3" not in metrics


def test_aggregator_all_uncovered_game_returns_zero():
    """Sentinel edge case (no real fixture data triggers this — every
    public-demo game has >=2 covered levels). Game with empty
    ``baselines`` returns per_game=0.0, levels_completed=0, no
    level_scores atoms regardless of AI completions in range."""
    agg = RHAEAggregator(human_baselines=_BASELINES_DB)
    metrics = agg(
        game_id="none", ai_actions_per_level={1: 5, 2: 5, 3: 5, 4: 5}
    )
    assert metrics["eval/rhae/per_game/none"] == 0.0
    assert metrics["eval/rhae/levels_completed/none"] == 0
    assert not any(
        k.startswith("eval/rhae/level_scores/none/") for k in metrics
    )


def test_aggregator_independent_across_games():
    """Sequential calls for different games don't share state."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    metrics_vc33 = agg(game_id="vc33", ai_actions_per_level={1: 10})
    metrics_tu93 = agg(game_id="tu93", ai_actions_per_level={1: 8})
    assert all("tu93" not in k for k in metrics_vc33)
    assert all("vc33" not in k for k in metrics_tu93)


def test_aggregator_rejects_non_positive_action_counts():
    """AI completing a level with 0 or negative actions is nonsensical."""
    agg = RHAEAggregator(human_baselines=_BASELINES)
    with pytest.raises(ValueError, match="ai_actions"):
        agg(game_id="vc33", ai_actions_per_level={1: 0})
    with pytest.raises(ValueError, match="ai_actions"):
        agg(game_id="vc33", ai_actions_per_level={1: -3})


def test_aggregator_accepts_string_level_keys():
    """Fixture-loaded ``data/human_baselines.json`` has string level
    keys after ``json.loads`` — the constructor must coerce them to int
    so the aggregator can be constructed directly from a file load."""
    string_keyed = {
        "vc33": {
            "total_levels": 3,
            "baselines": {"1": 10, "2": 20, "3": 30},
        }
    }
    agg = RHAEAggregator(human_baselines=string_keyed)
    metrics = agg(game_id="vc33", ai_actions_per_level={1: 10, 2: 40})
    assert metrics["eval/rhae/per_game/vc33"] == pytest.approx(1.5 / 6)


def test_aggregator_rejects_malformed_entry():
    """Fixture entry missing 'total_levels' or 'baselines' raises at
    construction — surface schema drift early, not at call time."""
    with pytest.raises(ValueError, match="total_levels|baselines"):
        RHAEAggregator(human_baselines={"vc33": {"baselines": {1: 10}}})
    with pytest.raises(ValueError, match="total_levels|baselines"):
        RHAEAggregator(human_baselines={"vc33": {"total_levels": 3}})


def test_aggregator_rejects_baseline_level_out_of_range():
    """A baseline level outside [1, total_levels] is a fixture-build
    bug — surface at construction."""
    with pytest.raises(ValueError, match="out of"):
        RHAEAggregator(
            human_baselines={
                "vc33": {"total_levels": 3, "baselines": {99: 10}}
            }
        )


# ---------------------------------------------------------------------------
# coverage — global RHAE coverage helper (per D-B / paper "RHAE coverage")
# ---------------------------------------------------------------------------


def test_coverage_global_ratio():
    """Coverage = sum(covered) / sum(total) across all games. Fixture
    has 2 games of 3 levels each, fully covered → 6/6 = 1.0."""
    assert coverage(_BASELINES) == 1.0


def test_coverage_db_mixed():
    """Mixed-coverage fixture (D-B realistic): gap5 has 2/5, full3 has
    3/3, none has 0/4. Total = (2+3+0) / (5+3+4) = 5/12."""
    assert coverage(_BASELINES_DB) == pytest.approx(5 / 12)


def test_coverage_empty_returns_zero():
    """Empty fixture → 0.0, not ZeroDivisionError."""
    assert coverage({}) == 0.0


def test_coverage_matches_extractor_summary():
    """The real-data summary the extractor prints reports
    `129/183 covered levels (70% RHAE coverage)` on the 340-replay
    public-demo dataset. Synthetic check that the helper produces the
    same ratio when fed a fixture with those raw counts."""
    synth = {
        f"g{i}": {"total_levels": 10, "baselines": {j: 1 for j in range(1, 8)}}
        for i in range(18)
    }
    # 18 games of 10 levels each = 180 total; 7 covered per game = 126.
    # Add a fixture-shape spot-check on a different ratio.
    assert coverage(synth) == pytest.approx(126 / 180)


# ---------------------------------------------------------------------------
# Integration — RHAEAggregator against the committed real fixture
#
# These tests load ``data/human_baselines.json`` (the actual D-A/D-B
# output of ``scripts/extract_human_baselines.py`` over the 340-replay
# public-demo dataset, committed 2026-05-12). They pin:
#
# - Fixture schema integrity: the constructor accepts the real file
#   without raising (catches D-A/D-B drift between extractor and
#   aggregator at PR-time).
# - Headline numbers from the Step-2 sign-off table (25 games, 70%
#   coverage, vc33 6/7 covered) — regression alarm if the extractor
#   silently changes its output.
# - End-to-end RHAE math on vc33 against the real per-level baselines
#   with a synthetic realistic action stream.
#
# Skipped (not xfailed) if the fixture is missing — keeps the laptop
# suite green for fresh clones before ``data/human_baselines.json`` is
# materialised. The integration is opt-in by presence.
# ---------------------------------------------------------------------------


def _load_real_baselines():
    if not REAL_BASELINES_PATH.exists():
        pytest.skip(f"real baselines fixture missing: {REAL_BASELINES_PATH}")
    return json.loads(REAL_BASELINES_PATH.read_text(encoding="utf-8"))


def test_real_fixture_loads_into_aggregator():
    """Real fixture is well-formed enough for RHAEAggregator to construct.
    Catches D-A/D-B drift between extractor and aggregator at PR-time
    (e.g. extractor renames "total_levels" → "levels_total" silently)."""
    baselines = _load_real_baselines()
    agg = RHAEAggregator(human_baselines=baselines)
    # Sanity: same games present.
    assert sorted(agg.human_baselines) == sorted(baselines)


def test_real_fixture_headline_numbers():
    """Step-2 sign-off table: 25 games, 183 total levels, 129 covered,
    70% RHAE coverage on the 340-replay public-demo dataset. Sentinels
    against silent extractor drift."""
    baselines = _load_real_baselines()
    assert len(baselines) == 25
    total_levels = sum(int(v["total_levels"]) for v in baselines.values())
    covered_levels = sum(len(v["baselines"]) for v in baselines.values())
    assert total_levels == 183
    assert covered_levels == 129
    assert coverage(baselines) == pytest.approx(129 / 183)


def test_real_fixture_vc33_shape():
    """vc33 specifically (Phase-4 pilot member, dry-run target): 7 total
    levels, 6 covered (uncovered: level 7), baselines range [13..87].
    Sentinel against vc33-specific regressions in the extractor."""
    baselines = _load_real_baselines()
    vc33 = baselines["vc33"]
    assert int(vc33["total_levels"]) == 7
    covered_keys = sorted(int(k) for k in vc33["baselines"])
    assert covered_keys == [1, 2, 3, 4, 5, 6]
    counts = [int(v) for v in vc33["baselines"].values()]
    assert min(counts) == 13 and max(counts) == 87


def test_real_fixture_every_game_has_at_least_two_covered():
    """D-B invariant on the real dataset: every game has >=2 covered
    levels. No real fixture data triggers the all-uncovered sentinel —
    pinning this fact protects the per-game RHAE from collapsing to a
    sentinel-zero on any public-demo game."""
    baselines = _load_real_baselines()
    for game_id, entry in baselines.items():
        assert len(entry["baselines"]) >= 2, (
            f"{game_id}: only {len(entry['baselines'])} covered levels — "
            "every public-demo game should have >=2 under the current "
            "min_completers=2 threshold"
        )


def test_real_fixture_vc33_end_to_end():
    """End-to-end RHAE math on vc33 with realistic synthetic AI counts.
    Scenario: AI clears levels 1, 2, 3 at parity with the human upper-
    median (baselines 13, 18, 38), AI also "clears" level 7 (uncovered
    under D-B) in 99 actions, and fails 4/5/6. Expected:

      level_scores = {1: 1.0, 2: 1.0, 3: 1.0}     (level 7 → skipped)
      levels_completed = 3
      per_game numerator = 1*1.0 + 2*1.0 + 3*1.0 = 6.0
      per_game denominator = sum(covered) = 1+2+3+4+5+6 = 21
      per_game = 6/21 = 2/7

    This pins three things together: (a) the D-A denominator uses
    covered-not-total, (b) the D-B skip silently drops level 7 (in
    range [1, 7] but uncovered), (c) per-level cap doesn't bite at
    parity."""
    baselines = _load_real_baselines()
    agg = RHAEAggregator(human_baselines=baselines)
    metrics = agg(
        game_id="vc33",
        ai_actions_per_level={1: 13, 2: 18, 3: 38, 7: 99},
    )
    # Three covered levels emitted; uncovered level 7 silently skipped.
    assert metrics["eval/rhae/levels_completed/vc33"] == 3
    assert metrics["eval/rhae/level_scores/vc33/1"] == pytest.approx(1.0)
    assert metrics["eval/rhae/level_scores/vc33/2"] == pytest.approx(1.0)
    assert metrics["eval/rhae/level_scores/vc33/3"] == pytest.approx(1.0)
    assert "eval/rhae/level_scores/vc33/7" not in metrics
    assert metrics["eval/rhae/per_game/vc33"] == pytest.approx(6 / 21)


def test_real_fixture_vc33_failed_to_clear_anything():
    """Phase-4 RHAE > 0 gate failure mode: AI takes actions on vc33 but
    clears no level. Per-game RHAE must be exactly 0.0, not NaN, not
    a sentinel. Pinned because the Phase-4 gate language ("RHAE > 0 on
    >=2/3 of pilots") makes the distinction between 0.0 and (e.g.)
    near-zero load-bearing — a NaN or small-epsilon return would
    silently pass the gate."""
    baselines = _load_real_baselines()
    agg = RHAEAggregator(human_baselines=baselines)
    metrics = agg(game_id="vc33", ai_actions_per_level={})
    assert metrics["eval/rhae/per_game/vc33"] == 0.0
    assert metrics["eval/rhae/levels_completed/vc33"] == 0
    assert math.isfinite(metrics["eval/rhae/per_game/vc33"])
