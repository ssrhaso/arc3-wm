"""Tests for ``scripts/compute_rhae.py`` - Phase-4 post-hoc RHAE CLI.

Per D2 the Phase-4 pipeline computes RHAE post-hoc after the
``--script train_eval`` run completes. This CLI consumes a JSONL of
per-eval-episode reward streams (one episode per line, ``{"rewards":
[r0, r1, ...]}``), segments each episode by level via the cumulative
``r = delta levels_completed`` signal from ``arc3_wm/env.py:113``, takes
MIN action count per level across eval episodes (mirroring
``scripts.extract_human_baselines.extract_per_session_baselines``),
and feeds the result to ``arc3_wm.rhae.RHAEAggregator``.

Known gap (surfaced, not silently worked around): DV3's eval logfn
pops the per-step rewards stack before logging (see
``third_party/dreamerv3/embodied/run/train_eval.py:67``), so the
stream is NOT in ``eval/episode/rewards`` on wandb today. The CLI
therefore takes a local JSONL input by default; capturing the stream
during runs is a separate concern.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compute_rhae import (
    aggregate_eval_episodes,
    compute_rhae,
    format_summary,
    load_episodes_from_jsonl,
    segment_episode_actions_per_level,
)


# ===========================================================================
# segment_episode_actions_per_level - single eval episode -> cleared-level counts
# ===========================================================================


def test_segment_level_boundaries_from_synthetic_stream():
    """Brief test (2): level-boundary detection from cumulative-reward diffs.
    Reward stream encodes: 5 actions to clear lvl 1, 3 actions to clear lvl 2,
    3 actions on lvl 3 (didn't clear). Output: {1: 5, 2: 3} - level 3 is
    not in the output (only CLEARED levels)."""
    # Index 0 is the initial-obs reward (always 0 in DV3 convention).
    # Indices 1..5: 5 actions on level 1; index 5 = level-up reward.
    # Indices 6..8: 3 actions on level 2; index 8 = level-up reward.
    # Indices 9..11: 3 actions on level 3; no level-up.
    rewards = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    assert segment_episode_actions_per_level(rewards) == {1: 5, 2: 3}


def test_segment_all_levels_cleared():
    """Reward stream ends with a level-up - all levels cleared. The final
    level-up reward fires at the last step; completed_max counts it.
    10 actions cleared lvl 1, 8 cleared lvl 2."""
    rewards = [0] + [0] * 9 + [1] + [0] * 7 + [1]  # length 19
    assert segment_episode_actions_per_level(rewards) == {1: 10, 2: 8}


def test_segment_no_levels_cleared_returns_empty():
    """Brief test (3): 0-level-completed case returns empty dict (not NaN).
    Player died on level 1; their partial 5-action count does NOT contribute
    to any level baseline (only CLEARED levels)."""
    rewards = [0, 0, 0, 0, 0, 0]  # 5 actions on lvl 1, no level-up
    assert segment_episode_actions_per_level(rewards) == {}


def test_segment_single_action_episode():
    """Edge: episode with one action, no level-up. Returns empty dict
    (no cleared levels). Smoke-tests the rewards[1:] slice."""
    assert segment_episode_actions_per_level([0, 0]) == {}


def test_segment_empty_or_initial_only_returns_empty():
    """Defensive: a zero-length or one-step (initial obs only) reward
    stream has no actions to count."""
    assert segment_episode_actions_per_level([]) == {}
    assert segment_episode_actions_per_level([0]) == {}


def test_segment_rejects_negative_rewards():
    """Reward signal is r in {0, +1} per arc3_wm/env.py:113. A negative
    reward indicates either a different env or a parsing bug - surface."""
    with pytest.raises(ValueError, match="negative|non-binary"):
        segment_episode_actions_per_level([0, -1, 0, 0])


# ===========================================================================
# aggregate_eval_episodes - MIN across episodes per cleared level
# ===========================================================================


def test_aggregate_min_across_episodes():
    """Multiple eval episodes for one game: take MIN action count per
    level that was cleared by AT LEAST one episode. Mirrors the human
    baseline extractor's per-session MIN logic so agent and human are
    measured by the same 'best attempt at each level' framing."""
    # Episode A: 12 actions cleared lvl 1, 25 cleared lvl 2.
    ep_a = [0] + [0] * 11 + [1] + [0] * 24 + [1]
    # Episode B: 15 actions cleared lvl 1, died on lvl 2.
    ep_b = [0] + [0] * 14 + [1] + [0] * 8
    # Expected: level 1 = min(12, 15) = 12; level 2 = 25 (only A cleared it).
    assert aggregate_eval_episodes([ep_a, ep_b]) == {1: 12, 2: 25}


def test_aggregate_empty_episode_list():
    """No eval episodes at all -> empty per-level dict."""
    assert aggregate_eval_episodes([]) == {}


def test_aggregate_all_zero_level_runs():
    """Every eval episode died before clearing any level. Output: empty
    per-level dict. compute_rhae downstream emits per_game=0.0,
    levels_completed=0."""
    ep1 = [0, 0, 0, 0, 0]
    ep2 = [0, 0, 0]
    assert aggregate_eval_episodes([ep1, ep2]) == {}


# ===========================================================================
# compute_rhae - end-to-end with synthetic episodes + baselines
# ===========================================================================


_BASELINES = {
    "vc33": {"total_levels": 3, "baselines": {"1": 10, "2": 20, "3": 30}},
    "tu93": {"total_levels": 2, "baselines": {"1": 8, "2": 16}},
}


def test_compute_rhae_brief_synthetic_example():
    """Brief test (1): synthetic eval/episode/* series -> expected RHAE
    output. Two eval episodes on vc33:
    - A: cleared lvl 1 in 12 actions, lvl 2 in 25.
    - B: cleared lvl 1 in 15, died on lvl 2.
    MIN: lvl1=12, lvl2=25. Baselines: lvl1=10, lvl2=20, lvl3=30 (3 levels).
    Expected:
      level_score(10, 12) = (10/12)^2 ~ 0.6944
      level_score(20, 25) = (20/25)^2 = 0.64
      per_game = (1*0.6944 + 2*0.64) / (1+2+3) = (0.6944 + 1.28) / 6
      levels_completed = 2"""
    ep_a = [0] + [0] * 11 + [1] + [0] * 24 + [1]
    ep_b = [0] + [0] * 14 + [1] + [0] * 8
    metrics = compute_rhae(
        episodes_rewards=[ep_a, ep_b],
        game_id="vc33",
        baselines=_BASELINES,
    )
    assert metrics["eval/rhae/levels_completed/vc33"] == 2
    expected_lvl1 = (10 / 12) ** 2
    expected_lvl2 = (20 / 25) ** 2
    assert metrics["eval/rhae/level_scores/vc33/1"] == pytest.approx(
        expected_lvl1
    )
    assert metrics["eval/rhae/level_scores/vc33/2"] == pytest.approx(
        expected_lvl2
    )
    expected_pg = (1 * expected_lvl1 + 2 * expected_lvl2) / (1 + 2 + 3)
    assert metrics["eval/rhae/per_game/vc33"] == pytest.approx(expected_pg)


def test_compute_rhae_zero_levels_completed_returns_zero_not_nan():
    """Brief test (3): no eval episode cleared any level. Output:
    per_game=0.0 (exact zero, not NaN), levels_completed=0, no
    level_scores keys."""
    eps = [[0, 0, 0, 0], [0, 0, 0]]
    metrics = compute_rhae(
        episodes_rewards=eps, game_id="vc33", baselines=_BASELINES
    )
    assert metrics["eval/rhae/per_game/vc33"] == 0.0
    assert metrics["eval/rhae/levels_completed/vc33"] == 0
    assert not any(
        k.startswith("eval/rhae/level_scores/vc33/") for k in metrics
    )


def test_compute_rhae_handles_baselines_with_string_level_keys():
    """The extractor emits ``data/human_baselines.json`` with string
    level keys (per task brief output-shape spec). RHAEAggregator's
    constructor must coerce string level keys to ints. Regression
    guard for the fixture-loading path under the D-A/D-B shape."""
    string_keyed = {
        "vc33": {"total_levels": 2, "baselines": {"1": 10, "2": 20}}
    }
    ep_a = [0] + [0] * 11 + [1]
    metrics = compute_rhae(
        episodes_rewards=[ep_a],
        game_id="vc33",
        baselines=string_keyed,
    )
    assert metrics["eval/rhae/levels_completed/vc33"] == 1
    assert metrics["eval/rhae/level_scores/vc33/1"] == pytest.approx(
        (10 / 12) ** 2
    )


# ===========================================================================
# load_episodes_from_jsonl - input parsing
# ===========================================================================


def test_load_jsonl_one_episode_per_line(tmp_path: Path):
    """File contains one JSON object per line with a 'rewards' key.
    Returns a list-of-lists of floats, one per episode, in file order."""
    f = tmp_path / "eps.jsonl"
    f.write_text(
        json.dumps({"rewards": [0, 0, 1]})
        + "\n"
        + json.dumps({"rewards": [0, 1, 0]})
        + "\n",
        encoding="utf-8",
    )
    eps = load_episodes_from_jsonl(f)
    assert eps == [[0, 0, 1], [0, 1, 0]]


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    """Blank lines between episodes are ignored."""
    f = tmp_path / "eps.jsonl"
    f.write_text(
        json.dumps({"rewards": [0, 0]})
        + "\n\n"
        + json.dumps({"rewards": [0, 1]})
        + "\n",
        encoding="utf-8",
    )
    assert load_episodes_from_jsonl(f) == [[0, 0], [0, 1]]


def test_load_jsonl_missing_file_raises(tmp_path: Path):
    """Stop-point fail-clean: nonexistent file errors cleanly with the
    path in the message - caller (CLI) surfaces this verbatim."""
    with pytest.raises(FileNotFoundError, match="no_such"):
        load_episodes_from_jsonl(tmp_path / "no_such.jsonl")


def test_load_jsonl_empty_file_returns_empty_list(tmp_path: Path):
    """Empty file -> empty episodes list. compute_rhae then emits the
    0-levels-completed degenerate output. Distinguishes 'file exists,
    no eval data' from 'file not found'."""
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    assert load_episodes_from_jsonl(f) == []


def test_load_jsonl_missing_rewards_key_raises(tmp_path: Path):
    """A row without a 'rewards' key is a malformed eval log - surface
    rather than silently emit an empty episode."""
    f = tmp_path / "bad.jsonl"
    f.write_text(json.dumps({"score": 1, "length": 10}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rewards"):
        load_episodes_from_jsonl(f)


# ===========================================================================
# format_summary - the one-line stdout the brief specifies
# ===========================================================================


def test_format_summary_one_line_with_step():
    """Brief spec: 'vc33 @ 500k env steps: levels_completed=2,
    per_game_rhae=0.42'. Step shown in 500k / 1M format when divisible;
    raw int otherwise."""
    metrics = {
        "eval/rhae/per_game/vc33": 0.42,
        "eval/rhae/levels_completed/vc33": 2,
    }
    s = format_summary(game_id="vc33", step=500_000, metrics=metrics)
    assert "vc33" in s
    assert "500k" in s or "500000" in s
    assert "levels_completed=2" in s
    assert "per_game_rhae=0.42" in s


def test_format_summary_no_step_omits_step_field():
    """When step is None (e.g. file mode without explicit --step), the
    summary still works; 'env steps' field is absent or 'unknown'."""
    metrics = {
        "eval/rhae/per_game/vc33": 0.0,
        "eval/rhae/levels_completed/vc33": 0,
    }
    s = format_summary(game_id="vc33", step=None, metrics=metrics)
    assert "vc33" in s
    assert "levels_completed=0" in s
    assert "per_game_rhae=0.00" in s
