"""Laptop-runnable tests for the launcher's eval-env sink plumbing.

``scripts/launch_pergame.py`` wraps the eval env factory in
``EvalRewardSink`` so every evaluation episode's reward stream lands in
``{logdir}/eval_episodes.jsonl``. Until 2026-09-05 only the ``train_eval``
branch did this; ``eval_only`` handed DreamerV3 the bare factory, so a
frozen-checkpoint evaluation produced no stream and no RHAE. These tests
pin the shared helpers and the branch wiring without importing JAX.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import scripts.launch_pergame as L
from arc3_wm.eval_reward_sink import EvalRewardSink


def test_eval_sink_path_for_joins_logdir(tmp_path):
    assert L.eval_sink_path_for(tmp_path / "run") == tmp_path / "run" / "eval_episodes.jsonl"
    assert L.eval_sink_path_for(str(tmp_path / "run")) == tmp_path / "run" / "eval_episodes.jsonl"


class _FakeEnv:
    """Minimal embodied-style env: dict transitions with is_first / is_last."""

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self._i = 0
        self.obs_space = {}
        self.act_space = {}

    def step(self, action):
        r = self._rewards[self._i]
        first = self._i == 0
        last = self._i == len(self._rewards) - 1
        self._i += 1
        return {"reward": r, "is_first": first, "is_last": last}

    def close(self):
        pass


def test_make_sinked_env_factory_wraps_and_forwards(tmp_path):
    calls = []

    def fake_make_env(cfg, index, **overrides):
        calls.append((cfg, index, overrides))
        return _FakeEnv([0.0])

    sink = tmp_path / "eval100" / "eval_episodes.jsonl"
    factory = L.make_sinked_env_factory(fake_make_env, sink)
    env = factory("cfg-object", 3, foo=1)

    assert isinstance(env, EvalRewardSink)
    assert calls == [("cfg-object", 3, {"foo": 1})]
    assert env._sink_path == sink
    assert sink.parent.is_dir(), "sink parent directory must be created eagerly"


def test_sinked_factory_records_one_line_per_episode(tmp_path):
    sink = tmp_path / "eval_episodes.jsonl"
    factory = L.make_sinked_env_factory(lambda cfg, index, **kw: _FakeEnv([0.0, 0.0, 1.0]), sink)
    env = factory(None, 0)
    for _ in range(3):
        env.step({"action": 0})
    lines = [json.loads(l) for l in sink.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines == [{"rewards": [0.0, 0.0, 1.0]}]


def _branch_body(source: str, script_name: str) -> str:
    """Source text of the ``config.script == "<script_name>"`` branch."""
    pattern = rf'config\.script == "{script_name}":(.*?)(?=\n    (?:elif|else)\b)'
    m = re.search(pattern, source, flags=re.S)
    assert m, f"{script_name} branch not found"
    return m.group(1)


def test_train_eval_and_eval_only_use_the_sinked_eval_factory():
    source = Path(L.__file__).read_text(encoding="utf-8")
    for script_name in ("train_eval", "eval_only"):
        body = _branch_body(source, script_name)
        assert "make_env_eval_with_sink" in body, f"{script_name} must wrap eval envs in the sink"
    eval_only = _branch_body(source, "eval_only")
    assert "bind(make_env, config)" not in eval_only, "eval_only must not hand DreamerV3 the bare env factory"
    train = _branch_body(source, "train")
    assert "make_env_eval_with_sink" not in train, "training rollouts must stay unwrapped"
