"""Tests for arc3_wm.state_change_reward (laptop-runnable, no JAX)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from arc3_wm.state_change_reward import (
    BONUS_KEY,
    CHANGED_KEY,
    StateChangeRewardWrapper,
    grid_hash,
)


class _FakeEnv:
    """Replays a scripted list of (frame, reward, is_first, is_last)."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.obs_space = {"image": "img-space", "reward": "r-space"}
        self.act_space = {"action": "a-space"}

    def step(self, action):
        frame, reward, first, last = self._script[self._i]
        self._i += 1
        return {
            "image": frame,
            "reward": np.float32(reward),
            "is_first": np.bool_(first),
            "is_last": np.bool_(last),
            "is_terminal": np.bool_(last),
        }

    def __len__(self):
        return 1


def _frame(fill: int) -> np.ndarray:
    return np.full((64, 64, 3), fill, dtype=np.uint8)


def test_grid_hash_is_exact_and_shape_sensitive():
    a = _frame(1)
    b = _frame(1)
    c = _frame(2)
    assert grid_hash(a) == grid_hash(b)
    assert grid_hash(a) != grid_hash(c)
    d = a.copy()
    d[10, 10, 0] = 9
    assert grid_hash(a) != grid_hash(d)
    assert grid_hash(a.reshape(64, 192)) != grid_hash(a)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        StateChangeRewardWrapper(_FakeEnv([]), beta=-0.1)
    with pytest.raises(ValueError):
        StateChangeRewardWrapper(_FakeEnv([]), beta=0.1, gate="magnitude")


def test_no_bonus_on_reset_step_and_unchanged_frames():
    script = [
        (_frame(0), 0.0, True, False),   # reset step: never a bonus
        (_frame(0), 0.0, False, False),  # unchanged
        (_frame(0), 0.0, False, True),   # unchanged, episode ends
    ]
    env = StateChangeRewardWrapper(_FakeEnv(script), beta=0.5)
    outs = [env.step({"action": 0}) for _ in script]
    assert [float(o["reward"]) for o in outs] == [0.0, 0.0, 0.0]
    assert [float(o[BONUS_KEY]) for o in outs] == [0.0, 0.0, 0.0]
    assert [float(o[CHANGED_KEY]) for o in outs] == [0.0, 0.0, 0.0]
    assert env.stats()["changed_frac"] == 0.0


def test_novelty_gate_decays_with_visit_count():
    """First visit to a new grid pays beta; the n-th visit pays beta / sqrt(n)."""
    script = [
        (_frame(0), 0.0, True, False),
        (_frame(1), 0.0, False, False),  # new grid 1: beta
        (_frame(0), 0.0, False, False),  # back to grid 0: first *post-change* visit: beta
        (_frame(1), 0.0, False, False),  # grid 1 again: beta / sqrt(2)
        (_frame(0), 0.0, False, False),  # grid 0 again: beta / sqrt(2)
        (_frame(1), 0.0, False, True),   # grid 1 third time: beta / sqrt(3)
    ]
    beta = 0.1
    env = StateChangeRewardWrapper(_FakeEnv(script), beta=beta)
    bonuses = [float(env.step({"action": 0})[BONUS_KEY]) for _ in script]
    expected = [0.0, beta, beta, beta / math.sqrt(2), beta / math.sqrt(2), beta / math.sqrt(3)]
    assert bonuses == pytest.approx(expected, rel=1e-6)
    st = env.stats()
    assert st["distinct_grids"] == 2
    assert st["changed_frac"] == 1.0
    assert st["total_bonus"] == pytest.approx(sum(expected), rel=1e-6)


def test_binary_gate_pays_flat_beta_on_every_change():
    script = [
        (_frame(0), 0.0, True, False),
        (_frame(1), 0.0, False, False),
        (_frame(1), 0.0, False, False),
        (_frame(0), 0.0, False, True),
    ]
    env = StateChangeRewardWrapper(_FakeEnv(script), beta=0.25, gate="binary")
    bonuses = [float(env.step({"action": 0})[BONUS_KEY]) for _ in script]
    assert bonuses == [0.0, 0.25, 0.0, 0.25]


def test_native_reward_is_added_not_replaced_and_stays_float32():
    script = [
        (_frame(0), 0.0, True, False),
        (_frame(1), 1.0, False, True),   # level clear coincides with a change
    ]
    env = StateChangeRewardWrapper(_FakeEnv(script), beta=0.1)
    env.step({"action": 0})
    out = env.step({"action": 0})
    assert out["reward"].dtype == np.float32
    assert float(out["reward"]) == pytest.approx(1.1)


def test_beta_zero_is_a_no_op_on_rewards_but_still_logs_changes():
    script = [
        (_frame(0), 0.0, True, False),
        (_frame(1), 0.0, False, True),
    ]
    env = StateChangeRewardWrapper(_FakeEnv(script), beta=0.0)
    env.step({"action": 0})
    out = env.step({"action": 0})
    assert float(out["reward"]) == 0.0
    assert float(out[CHANGED_KEY]) == 1.0


def test_new_episode_resets_previous_frame_but_keeps_counts():
    """Counts are per env instance across episodes; the comparison frame is not."""
    script = [
        (_frame(0), 0.0, True, False),
        (_frame(1), 0.0, False, True),    # ep 1: grid 1 first visit -> beta
        (_frame(1), 0.0, True, False),    # ep 2 reset on grid 1: no bonus, no comparison
        (_frame(1), 0.0, False, False),   # unchanged -> 0
        (_frame(0), 0.0, False, False),   # grid 0 first post-change visit -> beta
        (_frame(1), 0.0, False, True),    # grid 1 second visit -> beta / sqrt(2)
    ]
    beta = 0.2
    env = StateChangeRewardWrapper(_FakeEnv(script), beta=beta)
    bonuses = [float(env.step({"action": 0})[BONUS_KEY]) for _ in script]
    assert bonuses == pytest.approx([0.0, beta, 0.0, 0.0, beta, beta / math.sqrt(2)], rel=1e-6)


def test_wrapper_duck_types_embodied_wrapper_surface():
    inner = _FakeEnv([])
    env = StateChangeRewardWrapper(inner, beta=0.1)
    assert env.act_space == inner.act_space
    assert len(env) == 1
    assert bool(env) is True
    with pytest.raises(ValueError):
        _ = env.no_such_attribute
