"""Tests for arc3_wm.behavior_mixture (laptop-runnable, no JAX)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from arc3_wm.behavior_mixture import (
    N_ACTIONS,
    TYPE_SLICES,
    TypeBalancedMixturePolicy,
    exposed_types_from_mask,
    sample_type_balanced,
)


def _mask(types):
    m = np.zeros(N_ACTIONS, dtype=bool)
    for t in types:
        lo, hi = TYPE_SLICES[t]
        m[lo:hi] = True
    return m


def test_type_slices_partition_the_flat_space():
    covered = np.zeros(N_ACTIONS, dtype=int)
    for lo, hi in TYPE_SLICES.values():
        covered[lo:hi] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_exposed_types_from_mask():
    assert exposed_types_from_mask(_mask([1, 2, 3, 4])) == [1, 2, 3, 4]        # ls20-like
    assert exposed_types_from_mask(_mask([6])) == [6]                          # click-only
    m = _mask([6]); m[5:4101] = False; m[5 + 64 * 10 + 3] = True                # one live cell
    assert exposed_types_from_mask(m) == [6]
    assert exposed_types_from_mask(np.zeros(N_ACTIONS, bool)) == []
    with pytest.raises(ValueError):
        exposed_types_from_mask(np.zeros(10, bool))


def test_sample_type_balanced_is_uniform_over_types_not_indices():
    rng = np.random.default_rng(0)
    types = [1, 2, 3, 4, 5, 6]
    draws = [sample_type_balanced(rng, types) for _ in range(60_000)]
    by_type = Counter()
    for a in draws:
        for t, (lo, hi) in TYPE_SLICES.items():
            if lo <= a < hi:
                by_type[t] += 1
    for t in types:
        assert abs(by_type[t] / len(draws) - 1 / 6) < 0.01, by_type
    clicks = [a for a in draws if 5 <= a < 4101]
    assert len(set(clicks)) > 3000, "clicks must spread over the 64x64 grid"
    with pytest.raises(ValueError):
        sample_type_balanced(rng, [])


class _FakeAgent:
    def __init__(self, n_envs, fixed_action=7):
        self.n = n_envs
        self.fixed = fixed_action
        self.calls = []

    def policy(self, carry, obs, mode="train"):
        self.calls.append(mode)
        acts = {"action": np.full(self.n, self.fixed, dtype=np.int32)}
        return carry, acts, {"logprob": np.zeros(self.n)}

    def init_policy(self, batch):
        return "carry"

    def save(self):
        return {"agent": 1}


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        TypeBalancedMixturePolicy(_FakeAgent(1), types=[])
    with pytest.raises(ValueError):
        TypeBalancedMixturePolicy(_FakeAgent(1), types=[9])
    with pytest.raises(ValueError):
        TypeBalancedMixturePolicy(_FakeAgent(1), types=[1], eps0=1.5)
    with pytest.raises(ValueError):
        TypeBalancedMixturePolicy(_FakeAgent(1), types=[1], anneal_steps=0)


def test_delegates_everything_but_policy():
    inner = _FakeAgent(4)
    mix = TypeBalancedMixturePolicy(inner, types=[1, 6])
    assert mix.init_policy(4) == "carry"
    assert mix.save() == {"agent": 1}
    with pytest.raises(AttributeError):
        _ = mix.__no_such__


def test_eval_mode_is_never_mixed():
    inner = _FakeAgent(8)
    mix = TypeBalancedMixturePolicy(inner, types=[1, 2, 3, 4, 5, 6], eps0=1.0)
    _, acts, _ = mix.policy(None, {}, mode="eval")
    assert (acts["action"] == 7).all()
    assert mix.stats()["env_steps"] == 0


def test_eps_one_replaces_every_training_action_and_keeps_dtype():
    inner = _FakeAgent(16)
    mix = TypeBalancedMixturePolicy(inner, types=[1, 2, 3, 4, 5, 6], eps0=1.0, anneal_steps=10**9, seed=1)
    _, acts, outs = mix.policy(None, {}, mode="train")
    assert acts["action"].dtype == np.int32
    assert acts["action"].shape == (16,)
    assert "logprob" in outs
    assert mix.stats()["n_redrawn"] == 16
    assert mix.stats()["env_steps"] == 16
    draws = np.concatenate([mix.policy(None, {}, mode="train")[1]["action"] for _ in range(500)])
    buttons = ((draws >= 0) & (draws < 5)).mean()
    assert 0.78 < buttons < 0.88, buttons  # 5 of 6 exposed types are buttons


def test_eps_zero_is_identity():
    inner = _FakeAgent(4)
    mix = TypeBalancedMixturePolicy(inner, types=[6], eps0=0.0)
    _, acts, _ = mix.policy(None, {}, mode="train")
    assert (acts["action"] == 7).all()
    assert mix.stats()["n_redrawn"] == 0


def test_eps_anneals_linearly_to_zero_in_env_steps():
    inner = _FakeAgent(10)
    mix = TypeBalancedMixturePolicy(inner, types=[6], eps0=0.3, anneal_steps=100)
    assert mix.eps() == pytest.approx(0.3)
    for _ in range(5):                      # 50 env steps
        mix.policy(None, {}, mode="train")
    assert mix.eps() == pytest.approx(0.15)
    for _ in range(5):                      # 100 env steps
        mix.policy(None, {}, mode="train")
    assert mix.eps() == 0.0
    mix.policy(None, {}, mode="train")      # past the schedule: identity
    assert mix.eps() == 0.0
    before = mix.stats()["n_redrawn"]
    mix.policy(None, {}, mode="train")
    assert mix.stats()["n_redrawn"] == before


def test_redraw_rate_matches_eps():
    inner = _FakeAgent(100)
    mix = TypeBalancedMixturePolicy(inner, types=[6], eps0=0.3, anneal_steps=10**9, seed=3)
    total = 0
    for _ in range(200):
        _, acts, _ = mix.policy(None, {}, mode="train")
        total += int((acts["action"] != 7).sum())
    assert abs(total / 20_000 - 0.3) < 0.02
