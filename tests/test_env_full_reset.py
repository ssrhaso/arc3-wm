"""Reset semantics of the substrate against the real OFFLINE engine.

Finding (2026-09-05, vc33): the engine's RESET restarts the *current level*
once a level has been cleared and the game has ended. After a level-1 clear
and a GAME_OVER on level 2, ``reset()`` returns the level-2 start frame with
``levels_completed == 1``. Only ``Arcade.make`` starts a fresh game. So with
the default env an instance offers level 1 only until it clears it once;
``full_reset=True`` re-makes the game on every reset so each episode is a
full attempt from level 1, which is what the RHAE harness assumes.

These tests drive a human replay of vc33 through the engine to reach level 2
and therefore need ``data/replays/vc33`` and ``environment_files/vc33``.
"""
from __future__ import annotations

import glob
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arc_agi")

from arc3_wm.action_space import N_ACTIONS  # noqa: E402
from arc3_wm.env import ARC3GymEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _human_level1_actions():
    from arc3_wm.replay_loader import load_replay_file

    files = sorted(glob.glob(str(REPO / "data" / "replays" / "vc33" / "*.jsonl")))
    if not files or not (REPO / "environment_files" / "vc33").is_dir():
        pytest.skip("vc33 replays or environment files not available")
    for f in files:
        for ep in load_replay_file(Path(f)):
            if sum(s["reward"] for s in ep) >= 1:
                return [s["action"] for s in ep if s["action"] is not None and s["action"] >= 0]
    pytest.skip("no vc33 replay with a level clear")


def _clear_level1_then_game_over(env: ARC3GymEnv, actions, rng):
    obs0, info = env.reset()
    assert info["levels_completed"] == 0
    for a in actions:
        obs, r, term, trunc, info = env.step(int(a))
        if info["levels_completed"] >= 1:
            break
    assert info["levels_completed"] == 1, "human replay did not clear level 1"
    level2_start = obs.copy()
    n = 0
    while not (term or trunc) and n < 2000:
        obs, r, term, trunc, info = env.step(int(rng.integers(0, N_ACTIONS)))
        n += 1
    assert term, "expected an engine GAME_OVER on level 2 under random play"
    assert info["levels_completed"] == 1
    return obs0, level2_start


def test_default_reset_resumes_at_level_2_after_a_clear():
    actions = _human_level1_actions()
    env = ARC3GymEnv(game_id="vc33", seed=0, max_steps=1000)
    obs0, level2_start = _clear_level1_then_game_over(env, actions, np.random.default_rng(0))
    obs, info = env.reset()
    assert info["levels_completed"] == 1
    assert np.array_equal(obs, level2_start)
    assert not np.array_equal(obs, obs0)
    env.close()


def test_full_reset_starts_a_fresh_game_at_level_1():
    actions = _human_level1_actions()
    env = ARC3GymEnv(game_id="vc33", seed=0, max_steps=1000, full_reset=True)
    obs0, level2_start = _clear_level1_then_game_over(env, actions, np.random.default_rng(0))
    obs, info = env.reset()
    assert info["levels_completed"] == 0
    assert np.array_equal(obs, obs0)
    assert not np.array_equal(obs, level2_start)
    # and the reward accounting starts from level 1 again
    for a in actions:
        obs, r, term, trunc, info = env.step(int(a))
        if r > 0:
            break
    assert r == 1.0 and info["levels_completed"] == 1
    env.close()


def test_full_reset_flag_is_off_by_default_and_visible_in_repr():
    env = ARC3GymEnv(game_id="vc33", seed=0)
    assert "full_reset=False" in repr(env)
    env.close()
    env = ARC3GymEnv(game_id="vc33", seed=0, full_reset=True)
    assert "full_reset=True" in repr(env)
    env.close()


def test_embodied_env_passes_full_reset_through():
    from arc3_wm.embodied_env import ARC3EmbodiedEnv

    env = ARC3EmbodiedEnv(game_id="vc33", seed=0, full_reset=True)
    assert env._gym._full_reset is True
    env.close()


def test_launcher_default_and_make_env_wiring():
    import scripts.launch_pergame as L

    assert L.DEFAULT_ARC3_ENV["full_reset"] is False
    src = Path(L.__file__).read_text(encoding="utf-8")
    assert 'full_reset = bool(arc3_cfg.get("full_reset", False))' in src
    assert "ARC3EmbodiedEnv(game_id=game_id, seed=seed, max_steps=max_steps, full_reset=full_reset)" in src
