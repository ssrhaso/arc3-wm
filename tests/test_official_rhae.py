"""Tests for arc3_wm.official_rhae, including parity with the toolkit's own class."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc3_wm.official_rhae import (
    LEVEL_SCORE_CAP_PCT,
    compute_official_rhae,
    load_official_baselines,
    official_environment_score,
    official_level_score,
    score_episode,
    score_episodes,
)

REPO = Path(__file__).resolve().parents[1]


def _stream(*, level_lengths, tail=0):
    """Reward stream clearing each level in ``level_lengths`` actions, then ``tail`` misses."""
    rewards = [0.0]
    for n in level_lengths:
        rewards += [0.0] * (n - 1) + [1.0]
    rewards += [0.0] * tail
    return rewards


# --- per-level -------------------------------------------------------------

def test_level_score_is_squared_ratio_in_percent():
    assert official_level_score(20, 40, True) == pytest.approx(25.0)
    assert official_level_score(20, 20, True) == pytest.approx(100.0)


def test_level_score_caps_at_115():
    assert official_level_score(7, 3, True) == LEVEL_SCORE_CAP_PCT
    assert official_level_score(100, 1, True) == LEVEL_SCORE_CAP_PCT


def test_uncompleted_level_scores_zero_however_cheap():
    assert official_level_score(20, 1, False) == 0.0
    assert official_level_score(20, 0, True) == 0.0


# --- game aggregation ------------------------------------------------------

def test_game_cap_binds_when_only_early_levels_clear():
    """A capped level-1 clear on a 7-level game is worth 1/28, not 115/28."""
    scores = [LEVEL_SCORE_CAP_PCT] + [0.0] * 6
    assert official_environment_score(scores) == pytest.approx(100 / 28)


def test_perfect_game_scores_100():
    assert official_environment_score([100.0] * 5) == pytest.approx(100.0)


def test_weights_are_level_indices():
    # level 2 cleared at 100, others zero, on a 3-level game: 2/6 of the weight
    assert official_environment_score([0.0, 100.0, 0.0]) == pytest.approx(100 * 2 / 6)


def test_empty_inputs():
    assert official_environment_score([]) == 0.0
    with pytest.raises(ValueError):
        score_episode([0.0, 0.0], [])


# --- episode scoring -------------------------------------------------------

def test_score_episode_counts_actions_and_levels():
    ep = score_episode(_stream(level_lengths=[4, 6], tail=5), [8, 12, 20])
    assert ep.levels_completed == 2
    assert ep.actions == 15
    assert ep.level_scores[0] == pytest.approx(100 * (8 / 4) ** 2 if (8 / 4) ** 2 * 100 < 115 else 115)
    assert ep.level_scores[2] == 0.0
    assert 0.0 < ep.score_unit <= 1.0


def test_score_episode_with_no_clear_is_zero():
    ep = score_episode([0.0] * 51, [7, 18, 44])
    assert ep.levels_completed == 0 and ep.score == 0.0 and ep.actions == 50


def test_negative_reward_raises():
    with pytest.raises(ValueError):
        score_episode([0.0, -1.0], [10])


def test_score_episodes_takes_the_max_over_runs():
    """The toolkit's EnvironmentScoreList.score is max over plays, not the mean."""
    baselines = [10, 20]
    slow = _stream(level_lengths=[40])
    fast = _stream(level_lengths=[10])
    res = score_episodes([slow, fast], baselines)
    assert res["score"] == pytest.approx(score_episode(fast, baselines).score)
    assert res["n_episodes"] == 2 and res["n_cleared_episodes"] == 2
    assert res["mean_score_unit"] < res["score_unit"]


def test_score_episodes_empty():
    res = score_episodes([], [10])
    assert res["score"] == 0.0 and res["n_episodes"] == 0


# --- parity with the toolkit ----------------------------------------------

def test_matches_toolkit_environment_score_calculator():
    """Same inputs through arc_agi's own calculator must give the same score."""
    pytest.importorskip("arc_agi")
    from arc_agi.scorecard import EnvironmentScoreCalculator

    cases = [
        ([7, 18, 44, 61, 131, 34, 152], {1: 3}, 50),        # vc33 capped level-1 clear
        ([55, 8, 41, 21, 23, 23], {}, 100),                  # cd82 no clear
        ([18, 28, 18, 19, 31, 23, 58, 18], {1: 40, 2: 60}, 220),
        ([32, 81, 60], {1: 32, 2: 81, 3: 60}, 173),          # exactly human on every level
    ]
    for baselines, cleared, total_actions in cases:
        calc = EnvironmentScoreCalculator()
        spent = sum(cleared.values())
        for idx, base in enumerate(baselines, start=1):
            if idx in cleared:
                calc.add_level(level_index=idx, completed=True,
                               actions_taken=cleared[idx], baseline_actions=base)
            else:
                calc.add_level(level_index=idx, completed=False,
                               actions_taken=max(total_actions - spent, 0),
                               baseline_actions=base)
                spent = total_actions
        expected = calc.to_score().score

        ours = official_environment_score([
            official_level_score(b, cleared.get(i, 0), i in cleared)
            for i, b in enumerate(baselines, start=1)
        ])
        assert ours == pytest.approx(expected), (baselines, cleared)


def test_score_episode_matches_toolkit_on_a_real_stream():
    pytest.importorskip("arc_agi")
    from arc_agi.scorecard import EnvironmentScoreCalculator

    baselines = [7, 18, 44, 61, 131, 34, 152]
    rewards = _stream(level_lengths=[3], tail=47)
    ep = score_episode(rewards, baselines)
    calc = EnvironmentScoreCalculator()
    calc.add_level(level_index=1, completed=True, actions_taken=3, baseline_actions=7)
    for idx, base in enumerate(baselines[1:], start=2):
        calc.add_level(level_index=idx, completed=False,
                       actions_taken=(50 - 3 if idx == 2 else 0), baseline_actions=base)
    assert ep.score == pytest.approx(calc.to_score().score)
    assert ep.score_unit == pytest.approx(100 / 28 / 100)


# --- baselines and the metric family ---------------------------------------

def test_load_official_baselines_reads_metadata():
    if not (REPO / "environment_files" / "vc33").is_dir():
        pytest.skip("vc33 environment files not cached")
    b = load_official_baselines("vc33", REPO / "environment_files")
    assert b[0] == 7 and len(b) == 7 and all(isinstance(x, int) for x in b)


def test_load_official_baselines_missing_game():
    with pytest.raises(FileNotFoundError):
        load_official_baselines("zz99", REPO / "environment_files")


def test_official_baselines_differ_from_the_corpus_fixture():
    """The two metrics really are different; this is why the paper needs both."""
    fixture = REPO / "data" / "human_baselines.json"
    if not fixture.exists() or not (REPO / "environment_files" / "vc33").is_dir():
        pytest.skip("fixture or environment files unavailable")
    corpus = json.loads(fixture.read_text(encoding="utf-8"))
    official = load_official_baselines("vc33", REPO / "environment_files")
    assert official[0] == 7
    assert int(corpus["vc33"]["baselines"]["1"]) == 13
    assert len(official) >= len(corpus["vc33"]["baselines"])


def test_compute_official_rhae_key_family():
    m = compute_official_rhae(
        episodes_rewards=[_stream(level_lengths=[3], tail=47)],
        game_id="vc33",
        baselines=[7, 18, 44, 61, 131, 34, 152],
    )
    assert m["eval/rhae_official/per_game/vc33"] == pytest.approx(100 / 28 / 100)
    assert m["eval/rhae_official/levels_completed/vc33"] == 1.0
    assert m["eval/rhae_official/total_levels/vc33"] == 7.0
    assert m["eval/rhae_official/n_cleared_episodes/vc33"] == 1.0
