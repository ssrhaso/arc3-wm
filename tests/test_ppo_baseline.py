"""Laptop-runnable tests for scripts/ppo_baseline.py helpers (no torch needed)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import scripts.ppo_baseline as P
from arc3_wm.action_space import N_ACTIONS, flat_to_arc


def test_factored_round_trips_every_flat_index():
    for idx in range(N_ACTIONS):
        t, x, y = P.flat_to_factored(idx)
        assert P.factored_to_flat(t, x, y) == idx, idx


def test_factored_click_matches_action_space_convention():
    idx = P.factored_to_flat(5, 3, 7)  # ACTION6 at x=3, y=7
    action, data = flat_to_arc(idx)
    assert action.value == 6 and data == {"x": 3, "y": 7}
    assert P.factored_to_flat(0, 40, 41) == 0        # ACTION1 ignores coordinates
    assert P.factored_to_flat(6, 0, 0) == N_ACTIONS - 1  # ACTION7 is the last index


def test_entropy_ceilings():
    assert P.entropy_ceiling("flat") == pytest.approx(math.log(4102))
    assert P.entropy_ceiling("factored") == pytest.approx(math.log(7) + 2 * math.log(64))
    with pytest.raises(ValueError):
        P.entropy_ceiling("mixed")


def test_eval_stream_line_uses_sink_convention():
    line = P.eval_stream_line([0.0, 1.0, 0.0])
    assert json.loads(line) == {"rewards": [0.0, 0.0, 1.0, 0.0]}


def test_to_flat_actions_for_both_heads():
    assert P.to_flat_actions("flat", np.array([[7], [4101]])) == [7, 4101]
    assert P.to_flat_actions("factored", np.array([[5, 0, 0], [6, 9, 9]])) == [5, 4101]


def test_parse_args_defaults_match_the_atari_recipe():
    a = P.parse_args(["--game", "vc33", "--logdir", "/tmp/x"])
    assert (a.head, a.total_steps, a.num_envs, a.num_steps) == ("flat", 500_000, 8, 128)
    assert (a.lr, a.gamma, a.gae_lambda, a.epochs, a.minibatches) == (2.5e-4, 0.99, 0.95, 4, 4)
    assert (a.clip, a.ent_coef, a.vf_coef, a.max_grad_norm) == (0.1, 0.01, 0.5, 0.5)
    assert a.eval_episodes == 100 and a.max_steps == 1000


def test_evaluation_env_starts_every_episode_at_level_one():
    """The eval env must set full_reset.

    The engine's plain RESET resumes the current level once one has been cleared, so an
    evaluation that does not force a fresh game silently measures level 2 onward after the
    first clear. Passing a new seed per episode happens to rebuild the game too, but this
    pins the intent rather than the side effect.
    """
    src = Path(P.__file__).read_text(encoding="utf-8")
    assert "full_reset=True" in src.split("def evaluate(", 1)[1].split("def main(", 1)[0],         "evaluate() must build its env with full_reset=True"


def test_env_pool_resets_and_records_episodes():
    pytest.importorskip("arc_agi")
    pool = P.EnvPool("vc33", seeds=[0, 1], max_steps=1000)
    assert pool.obs.shape == (2, 64, 64, 3) and pool.obs.dtype == np.uint8
    rng = np.random.default_rng(0)
    for _ in range(60):  # vc33 ends in GAME_OVER after 50 flat random actions
        obs, rew, done = pool.step(rng.integers(0, N_ACTIONS, size=2).tolist())
    assert obs.shape == (2, 64, 64, 3)
    assert len(pool.finished) >= 2
    assert all(length == 50 for _, length in pool.finished)
    pool.close()
