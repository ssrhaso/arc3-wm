"""Gymnasium registration contract for the 25 public ARC-AGI-3 games.

Pins the formalised ``gym.make`` entry point so the registered ids,
their entry point, and the per-game ``kwargs`` cannot drift silently.
"""
from __future__ import annotations

import gymnasium as gym
import pytest

from arc3_wm.registration import (
    PUBLIC_GAMES,
    env_id,
    register_envs,
)


def test_public_games_is_canonical_25():
    assert len(PUBLIC_GAMES) == 25
    assert len(set(PUBLIC_GAMES)) == 25, "duplicate game id in PUBLIC_GAMES"
    assert list(PUBLIC_GAMES) == sorted(PUBLIC_GAMES), "PUBLIC_GAMES not sorted"
    assert "vc33" in PUBLIC_GAMES and "sb26" in PUBLIC_GAMES and "cd82" in PUBLIC_GAMES


def test_public_games_matches_baseline_fixture():
    """If the human-baseline fixture is present, its keys must match.

    Skips (not fails) on a fresh clone with no data/ - registration must
    not depend on the fixture, but when both exist they must agree.
    """
    import json
    import pathlib

    fixture = pathlib.Path(__file__).resolve().parents[1] / "data" / "human_baselines.json"
    if not fixture.exists():
        pytest.skip("data/human_baselines.json absent (fresh clone)")
    keys = set(json.loads(fixture.read_text()).keys())
    assert keys == set(PUBLIC_GAMES)


def test_register_envs_registers_all_ids():
    ids = register_envs()
    assert ids == tuple(env_id(g) for g in PUBLIC_GAMES)
    for eid in ids:
        assert eid in gym.registry, f"{eid} not in gymnasium.registry"


def test_env_id_format():
    assert env_id("vc33") == "ARC3/vc33-v0"


def test_registered_spec_entry_point_and_kwargs():
    register_envs()
    spec = gym.registry[env_id("vc33")]
    assert spec.entry_point == "arc3_wm.env:ARC3GymEnv"
    assert spec.kwargs == {"game_id": "vc33"}
    # No double-truncation: the wrapper owns truncation, Gymnasium must not
    # also wrap a TimeLimit.
    assert spec.max_episode_steps is None


def test_register_envs_is_idempotent():
    first = register_envs()
    second = register_envs()
    assert first == second
    # No duplicate-registration exception, and the spec object is unchanged.
    assert gym.registry[env_id("vc33")].entry_point == "arc3_wm.env:ARC3GymEnv"


def test_importing_package_self_registers():
    """``import arc3_wm`` alone must make the ids resolvable."""
    import arc3_wm  # noqa: F401

    assert env_id("vc33") in gym.registry


def test_gym_make_yields_arc3gymenv():
    """End-to-end: gym.make on a cached game returns a real ARC3GymEnv."""
    from arc3_wm.env import ARC3GymEnv

    env = gym.make(env_id("vc33"), max_steps=5)
    try:
        assert isinstance(env.unwrapped, ARC3GymEnv)
        obs, info = env.reset()
        assert obs.shape == (64, 64, 3)
        assert "action_mask" in info
    finally:
        env.close()
