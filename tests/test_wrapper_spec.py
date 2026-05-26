"""Gymnasium contract tests for ARC3GymEnv on vc33.

Asserts the wrapper meets the Gymnasium API surface DreamerV3 expects:
- observation_space and action_space are concrete spaces (not dynamic).
- reset() returns (obs, info); step() returns (obs, reward, terminated, truncated, info).
- obs has shape (64, 64, 3) and dtype uint8 in the documented value range.
- info["action_mask"] is a length-4102 bool array.
- terminated / truncated semantics: terminal states flip terminated, max_steps flips truncated.
"""
from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from arc3_wm.action_space import N_ACTIONS, build_mask
from arc3_wm.env import ARC3GymEnv


@pytest.fixture(scope="module")
def env() -> ARC3GymEnv:
    e = ARC3GymEnv(game_id="vc33", seed=0, max_steps=10)
    yield e
    e.close()


def test_is_gym_env(env):
    assert isinstance(env, gym.Env)


def test_action_space_is_discrete_4102(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == N_ACTIONS == 4102


def test_observation_space_is_box_uint8(env):
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)
    assert obs_space.shape == (64, 64, 3)
    assert obs_space.dtype == np.uint8
    assert obs_space.low.min() == 0
    assert obs_space.high.max() == 255


def test_reset_returns_obs_info(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    for key in ("available_actions", "action_mask", "levels_completed", "win_levels", "state", "guid", "steps"):
        assert key in info, f"info missing {key!r}"
    assert info["steps"] == 0
    assert info["state"] == "NOT_FINISHED"


def test_info_action_mask_shape_dtype(env):
    _, info = env.reset()
    mask = info["action_mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == bool
    # Every entry that's True must correspond to a valid action under the
    # current available_actions set.
    np.testing.assert_array_equal(mask, build_mask(info["available_actions"]))


def test_step_signature_and_obs(env):
    env.reset()
    # vc33 only allows ACTION6 -> sample a click cell.
    out = env.step(5)  # idx 5 = ACTION6 (x=0, y=0)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["steps"] == 1


def test_step_truncates_at_max_steps():
    """max_steps=3 -> after 3 steps without terminal, truncated must be True."""
    e = ARC3GymEnv(game_id="vc33", seed=0, max_steps=3)
    try:
        e.reset()
        for i in range(3):
            obs, _, terminated, truncated, info = e.step(5)  # ACTION6 (0,0)
            if terminated:
                # vc33 is fragile; if terminal arrives early the truncation
                # invariant doesn't apply on this trajectory. Re-run from a fresh reset.
                e.reset()
                continue
            if i < 2:
                assert not truncated
        # If we never hit terminal, last step must have flipped truncated.
        if not terminated:
            assert truncated
    finally:
        e.close()


def test_invalid_action_raises(env):
    env.reset()
    with pytest.raises(ValueError):
        env.step(N_ACTIONS)
    with pytest.raises(ValueError):
        env.step(-1)


def test_render_mode_none_returns_none():
    """Default (no render_mode) follows the Gymnasium contract: render() -> None."""
    e = ARC3GymEnv(game_id="vc33", seed=0, max_steps=10)
    try:
        assert e.render_mode is None
        e.reset()
        assert e.render() is None
    finally:
        e.close()


def test_render_rgb_array_matches_last_obs():
    """render_mode='rgb_array' returns the most recent decoded frame."""
    e = ARC3GymEnv(game_id="vc33", seed=0, max_steps=10, render_mode="rgb_array")
    try:
        assert "rgb_array" in e.metadata["render_modes"]
        obs, _ = e.reset()
        frame = e.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, obs)
        # Advances with stepping and stays in sync with the returned obs.
        obs2, *_ = e.step(5)  # ACTION6 (0, 0)
        np.testing.assert_array_equal(e.render(), obs2)
        # Returned frame is a copy: mutating it must not corrupt the env.
        frame2 = e.render()
        frame2[:] = 0
        np.testing.assert_array_equal(e.render(), obs2)
    finally:
        e.close()


def test_render_before_reset_is_black_frame():
    """rgb_array before the first reset returns a black frame, never None."""
    e = ARC3GymEnv(game_id="vc33", seed=0, render_mode="rgb_array")
    try:
        frame = e.render()
        assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
        assert not frame.any()
    finally:
        e.close()


def test_unsupported_render_mode_raises():
    with pytest.raises(ValueError, match="render_mode"):
        ARC3GymEnv(game_id="vc33", render_mode="terminal")


def test_offline_mode_required(monkeypatch):
    """Constructing the wrapper from a NORMAL-mode Arcade must raise."""
    import arc_agi
    # Simulate a NORMAL-mode arcade by passing one in explicitly.
    class _FakeArcade:
        operation_mode = arc_agi.OperationMode.NORMAL
        def make(self, *a, **k): return None
    with pytest.raises(RuntimeError, match="OFFLINE"):
        ARC3GymEnv(game_id="vc33", arcade=_FakeArcade())


def test_unknown_game_id_raises():
    """A game_id outside the 25 public games is rejected with a clear error."""
    with pytest.raises(ValueError, match="unknown game_id"):
        ARC3GymEnv(game_id="zz99")  # not a public ARC-AGI-3 game


def test_repr_reports_identity(env):
    r = repr(env)
    assert r.startswith("ARC3GymEnv(")
    assert "game_id='vc33'" in r
    assert "seed=0" in r
    assert "render_mode=None" in r


def test_repr_tracks_step_progress():
    e = ARC3GymEnv(game_id="vc33", seed=0, max_steps=10)
    e.reset()
    assert "step=0" in repr(e)
    e.step(e.action_space.sample())
    assert "step=1" in repr(e)

