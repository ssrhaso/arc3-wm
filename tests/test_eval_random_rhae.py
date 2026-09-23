"""Tests for ``scripts/eval_random_rhae.py`` - random-policy offline RHAE.

The random-policy eval exists to answer one question for the paper: is the
trained per-game RHAE above what a uniform-random masked/unmasked agent
scores under the *same* eval protocol and the *same* RHAE aggregator?

Two contracts are pinned here:

1. The masked agent only ever emits flat indices that the per-step
   action mask marks valid (``sample_action`` + an end-to-end real-env run
   that asserts zero mask violations). The unmasked agent samples the full
   4102-way space and tolerates the 6 dead indices (arc_agi no-ops them) -
   this is what matches the trained vc33 eval, which used NO masking (D11).
2. RHAE is computed over the D-B *covered-level subset* via the exact same
   path that scored the trained runs (``scripts.compute_rhae`` ->
   ``arc3_wm.rhae.RHAEAggregator``): a clear on an uncovered level
   contributes nothing; covered-level denominators are preserved.

The integration tests use the real OFFLINE ``arc_agi`` env (no mocking -
CLAUDE.md "no mocking the environment when a real call would work").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arc3_wm.action_space import N_ACTIONS, build_mask
from scripts.eval_random_rhae import run_random_eval, sample_action, score_episodes


# ---------------------------------------------------------------------------
# Unit: sample_action masking contract (no env)
# ---------------------------------------------------------------------------


def test_sample_action_masked_only_returns_valid_indices():
    rng = np.random.default_rng(0)
    # A representative vc33-style mask: ACTION6 click grid live, types dead.
    mask = build_mask([6])
    valid = set(np.flatnonzero(mask).tolist())
    for _ in range(2000):
        a = sample_action(rng, mask, masked=True)
        assert a in valid


def test_sample_action_masked_respects_arbitrary_masks():
    rng = np.random.default_rng(1)
    for _ in range(50):
        # Random sparse boolean mask with at least one live index.
        mask = rng.integers(0, 2, size=N_ACTIONS).astype(bool)
        if not mask.any():
            mask[rng.integers(N_ACTIONS)] = True
        valid = set(np.flatnonzero(mask).tolist())
        for _ in range(20):
            a = sample_action(rng, mask, masked=True)
            assert a in valid


def test_sample_action_unmasked_ignores_mask_and_covers_dead_indices():
    rng = np.random.default_rng(2)
    # Only ACTION6 grid live (vc33). Unmasked sampling must still be able to
    # land on the 6 dead indices {0,1,2,3,4,4101} - that is the trained-eval
    # behaviour we replicate.
    mask = build_mask([6])
    seen_dead = False
    for _ in range(20000):
        a = sample_action(rng, mask, masked=False)
        assert 0 <= a < N_ACTIONS
        if not mask[a]:
            seen_dead = True
    assert seen_dead, "unmasked sampler never hit a dead index in 20k draws"


def test_sample_action_masked_empty_mask_raises():
    rng = np.random.default_rng(3)
    empty = np.zeros(N_ACTIONS, dtype=bool)
    with pytest.raises(ValueError):
        sample_action(rng, empty, masked=True)


# ---------------------------------------------------------------------------
# Unit: score_episodes uses the covered-level subset (same as trained pipeline)
# ---------------------------------------------------------------------------


def _vc33_fixture(tmp_path: Path) -> Path:
    # Mirrors data/human_baselines.json vc33 entry: total_levels=7, levels
    # 1..6 covered (n>=2), level 7 uncovered (D-B).
    fixture = {
        "vc33": {
            "total_levels": 7,
            "baselines": {"1": 13, "2": 18, "3": 38, "4": 50, "5": 87, "6": 44},
        }
    }
    p = tmp_path / "baselines.json"
    p.write_text(json.dumps(fixture), encoding="utf-8")
    return p


def test_score_episodes_skips_uncovered_level(tmp_path: Path):
    # One synthetic episode that clears levels 1..7 (the engine would emit a
    # +1 at each level-up). Level 7 is uncovered in the fixture and MUST be
    # skipped from level_scores; levels 1..6 score; denominator = sum(1..6).
    # rewards[0] is the initial-obs 0 (EvalRewardSink convention); then one
    # +1 per level cleared, here each on its own action.
    rewards = [0.0] + [1.0] * 7
    episodes_file = tmp_path / "eval_episodes.jsonl"
    episodes_file.write_text(json.dumps({"rewards": rewards}) + "\n", encoding="utf-8")
    baselines = _vc33_fixture(tmp_path)

    metrics, n_eps = score_episodes(
        sink_path=episodes_file, game_id="vc33", baselines_path=baselines
    )

    assert n_eps == 1
    scored_levels = {
        int(k.rsplit("/", 1)[1])
        for k in metrics
        if k.startswith("eval/rhae/level_scores/vc33/")
    }
    assert scored_levels == {1, 2, 3, 4, 5, 6}, scored_levels
    assert 7 not in scored_levels
    # levels_completed counts covered clears only (6, not 7).
    assert metrics["eval/rhae/levels_completed/vc33"] == 6
    assert metrics["eval/rhae/per_game/vc33"] > 0.0


def test_score_episodes_zero_clears_is_zero_not_nan(tmp_path: Path):
    # An episode that clears nothing -> per_game 0.0 (the Phase-4 gate-fail
    # shape), never NaN.
    episodes_file = tmp_path / "eval_episodes.jsonl"
    episodes_file.write_text(json.dumps({"rewards": [0.0, 0.0, 0.0]}) + "\n", encoding="utf-8")
    baselines = _vc33_fixture(tmp_path)
    metrics, _ = score_episodes(
        sink_path=episodes_file, game_id="vc33", baselines_path=baselines
    )
    assert metrics["eval/rhae/per_game/vc33"] == 0.0
    assert metrics["eval/rhae/levels_completed/vc33"] == 0


# ---------------------------------------------------------------------------
# Integration: real OFFLINE arc_agi env, small budget
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
def test_run_random_eval_masked_respects_mask_and_writes_artifact(tmp_path: Path):
    sink = tmp_path / "eval_episodes.jsonl"
    diag = run_random_eval(
        game_id="vc33",
        n_episodes=4,
        max_steps=60,
        masked=True,
        env_seed=0,
        action_seed=0,
        sink_path=sink,
    )
    # The masked agent must never emit an out-of-mask action.
    assert diag["invalid_action_count"] == 0
    assert diag["n_episodes"] == 4
    assert diag["total_steps"] >= 4

    lines = [l for l in sink.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4, "one JSONL line per eval episode"
    for line in lines:
        obj = json.loads(line)
        assert "rewards" in obj
        assert len(obj["rewards"]) >= 1
        assert obj["rewards"][0] == 0.0  # initial-obs reward, sink convention

    # And it scores cleanly through the shared aggregator.
    metrics, n_eps = score_episodes(
        sink_path=sink, game_id="vc33", baselines_path=Path("data/human_baselines.json")
    )
    assert n_eps == 4
    assert "eval/rhae/per_game/vc33" in metrics
    scored_levels = {
        int(k.rsplit("/", 1)[1])
        for k in metrics
        if k.startswith("eval/rhae/level_scores/vc33/")
    }
    # Whatever the random agent cleared, it can only be a covered level.
    assert scored_levels.issubset({1, 2, 3, 4, 5, 6})


@pytest.mark.timeout(180)
def test_run_random_eval_unmasked_runs_and_scores(tmp_path: Path):
    sink = tmp_path / "eval_episodes.jsonl"
    diag = run_random_eval(
        game_id="vc33",
        n_episodes=4,
        max_steps=60,
        masked=False,
        env_seed=0,
        action_seed=0,
        sink_path=sink,
    )
    assert diag["n_episodes"] == 4
    # Unmasked: invalid actions are allowed (arc_agi no-ops them); the diag
    # counter is only meaningful in masked mode, so we don't assert it == 0.
    lines = [l for l in sink.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4
    metrics, n_eps = score_episodes(
        sink_path=sink, game_id="vc33", baselines_path=Path("data/human_baselines.json")
    )
    assert n_eps == 4
    assert isinstance(metrics["eval/rhae/per_game/vc33"], float)
