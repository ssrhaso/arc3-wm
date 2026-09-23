"""Laptop-runnable tests for the launcher's two intervention arms.

* State-change bonus: ``--env.arc3.shaping_beta`` wraps the *training* env
  factory in ``StateChangeRewardWrapper``; eval factories stay native.
* Type-balanced behavior mixture: ``--behavior-mixture type_balanced`` wraps
  the agent in ``TypeBalancedMixturePolicy`` over the game's exposed types.

No JAX: helpers are exercised with fakes, wiring is pinned by source
inspection, and the exposed-type lookup runs the real OFFLINE env.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import scripts.launch_pergame as L
from arc3_wm.behavior_mixture import TypeBalancedMixturePolicy
from arc3_wm.state_change_reward import StateChangeRewardWrapper


# --- CLI -----------------------------------------------------------------

def test_mixture_flags_default_off():
    args, _ = L.parse_args(["--logdir", "/tmp/r", "--task", "arc3_vc33"])
    assert args.behavior_mixture == "none"
    assert args.mixture_eps0 == 0.3
    assert args.mixture_anneal_steps == 200_000


def test_mixture_flags_captured():
    args, leftover = L.parse_args([
        "--logdir", "/tmp/r", "--task", "arc3_cd82",
        "--behavior-mixture", "type_balanced", "--mixture-eps0", "0.5",
        "--mixture-anneal-steps", "1000", "--env.arc3.shaping_beta", "0.1",
    ])
    assert args.behavior_mixture == "type_balanced"
    assert args.mixture_eps0 == 0.5
    assert args.mixture_anneal_steps == 1000
    assert "--env.arc3.shaping_beta" in leftover  # forwarded to elements.Flags


def test_mixture_flag_rejects_unknown_value():
    with pytest.raises(SystemExit):
        L.parse_args(["--logdir", "/tmp/r", "--task", "arc3_vc33", "--behavior-mixture", "uniform"])


def test_shaping_defaults_are_off():
    assert L.DEFAULT_ARC3_ENV["shaping_beta"] == 0.0
    assert L.DEFAULT_ARC3_ENV["shaping_gate"] == "novelty"


# --- shaping helper ----------------------------------------------------------

class _Env:
    obs_space = {}
    act_space = {}


def test_maybe_shape_env_only_wraps_training_envs_with_positive_beta():
    env = _Env()
    assert L.maybe_shape_env(env, {"shaping_beta": 0.0}, shaped=True) is env
    assert L.maybe_shape_env(env, {}, shaped=True) is env
    assert L.maybe_shape_env(env, {"shaping_beta": 0.1}, shaped=False) is env
    wrapped = L.maybe_shape_env(env, {"shaping_beta": 0.1, "shaping_gate": "binary"}, shaped=True)
    assert isinstance(wrapped, StateChangeRewardWrapper)
    assert wrapped.env is env
    assert wrapped.beta == 0.1
    assert wrapped.gate == "binary"


# --- mixture helper ----------------------------------------------------------

class _Agent:
    pass


def test_wrap_agent_with_mixture():
    agent = _Agent()
    assert L.wrap_agent_with_mixture(agent, "none", [], 0.3, 100, 0) is agent
    mix = L.wrap_agent_with_mixture(agent, "type_balanced", [1, 6], 0.2, 500, 7)
    assert isinstance(mix, TypeBalancedMixturePolicy)
    assert mix.types == [1, 6] and mix.eps0 == 0.2 and mix.anneal_steps == 500
    with pytest.raises(ValueError):
        L.wrap_agent_with_mixture(agent, "epsilon_greedy", [1], 0.3, 100, 0)


@pytest.mark.parametrize("game, expected", [
    ("vc33", [6]),                 # click-only
    ("cd82", [1, 2, 3, 4, 5, 6]),  # five buttons plus the click grid
    ("ls20", [1, 2, 3, 4]),        # four directional moves only
])
def test_exposed_types_for_game_reads_the_real_engine_mask(game, expected):
    pytest.importorskip("arc_agi")
    if not (Path(L._REPO_ROOT) / "environment_files" / game).is_dir():
        pytest.skip(f"environment files for {game} not cached")
    assert L.exposed_types_for_game(game) == expected


# --- wiring, by source inspection ------------------------------------------

def _branch_body(source: str, script_name: str) -> str:
    m = re.search(rf'config\.script == "{script_name}":(.*?)(?=\n    (?:elif|else)\b)', source, flags=re.S)
    assert m, script_name
    return m.group(1)


def test_training_factories_are_shaped_and_eval_factories_are_not():
    src = Path(L.__file__).read_text(encoding="utf-8")
    for name in ("train", "train_eval"):
        assert "bind(make_env, config, shaped=True)" in _branch_body(src, name), name
    assert "shaped=True" not in _branch_body(src, "eval_only")
    # the shared eval factory wraps make_env without the shaped switch
    assert "make_sinked_env_factory(make_env, eval_sink_path)" in src
    assert "def make_env(config, index: int, shaped: bool = False" in src


def test_agent_factory_applies_the_mixture_after_warm_start():
    src = Path(L.__file__).read_text(encoding="utf-8")
    body = src.split("def make_agent_with_seed():", 1)[1].split("run_args = elements.Config(", 1)[0]
    assert body.index("seed_wm_from_ckpt(") < body.index("wrap_agent_with_mixture(")
