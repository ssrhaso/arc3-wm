"""Tests for ``arc3_wm.eval_reward_sink.EvalRewardSink``.

The wrapper's job is to record per-eval-episode reward streams to a
JSONL sink so ``scripts/compute_rhae.py`` can compute post-hoc RHAE.
Tests use a hand-rolled DummyEnv that exercises the wrapper's three
contract points:

  1. ``is_first`` resets the buffer.
  2. Every step appends the obs reward to the buffer (including the
     ``is_first`` step, where reward is the DV3-convention 0).
  3. ``is_last`` flushes one JSONL line and resets.

Format of each line is exactly what
``scripts.compute_rhae.load_episodes_from_jsonl`` expects:
``{"rewards": [r0, r1, ...]}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc3_wm.eval_reward_sink import EvalRewardSink


class DummyEnv:
    """Scripted env. ``transitions`` is a list of obs dicts the env
    returns from successive ``step()`` calls. ``self.steps_called``
    records every (action, returned-obs) pair so tests can assert the
    wrapper forwards correctly without inspecting it directly."""

    def __init__(self, transitions):
        self._transitions = list(transitions)
        self._i = 0
        self.steps_called: list[tuple] = []
        # Stub attributes the wrapper's __getattr__ should forward.
        self.obs_space = {"image": "stub"}
        self.act_space = {"action": "stub"}

    def step(self, action):
        if self._i >= len(self._transitions):
            raise RuntimeError("DummyEnv exhausted")
        obs = self._transitions[self._i]
        self._i += 1
        self.steps_called.append((action, obs))
        return obs

    def __len__(self):
        return len(self._transitions)

    def __bool__(self):
        return True


class DummyEnvWithInfo(DummyEnv):
    """``DummyEnv`` that also exposes an ``info`` dict, mimicking
    ``ARC3EmbodiedEnv.info`` (which carries ``info["state"] =
    fd.state.name``). ``EvalRewardSink._read_terminal_state`` reads
    ``self.env.info.get("state")`` at ``is_last`` to record the terminal
    cause. The info is fixed for the scripted episode - the wrapper only
    reads it at the terminal step, so a constant value is representative
    of what the real env reports there."""

    def __init__(self, transitions, info):
        super().__init__(transitions)
        self.info = dict(info)


def _episode(rewards, *, length=None):
    """Build a list of obs dicts that mimic a DV3 episode of given rewards.

    ``rewards[0]`` is the initial-obs reward (paired with ``is_first=True``);
    ``rewards[-1]`` is the terminal reward (paired with ``is_last=True``).
    """
    length = length or len(rewards)
    eps = []
    for i, r in enumerate(rewards):
        eps.append(
            {
                "reward": r,
                "is_first": i == 0,
                "is_last": i == length - 1,
                "is_terminal": i == length - 1,
            }
        )
    return eps


# ---------------------------------------------------------------------------
# Step forwarding + buffer assembly + JSONL flush
# ---------------------------------------------------------------------------


def test_flushes_one_line_per_episode(tmp_path: Path):
    """Single full episode -> exactly one JSONL line with the full reward
    stream including the initial-obs reward (0)."""
    sink = tmp_path / "eval_episodes.jsonl"
    env = DummyEnv(_episode([0, 0, 1, 0, 1]))  # 5-step episode
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(5):
        w.step({"action": 0, "reset": False})
    assert sink.exists()
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == {"rewards": [0.0, 0.0, 1.0, 0.0, 1.0]}


def test_multiple_episodes_one_line_each(tmp_path: Path):
    """Three consecutive episodes -> three JSONL lines, in order."""
    sink = tmp_path / "eval_episodes.jsonl"
    eps = (
        _episode([0, 1, 1])
        + _episode([0, 0, 0])
        + _episode([0, 1])
    )
    env = DummyEnv(eps)
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(len(eps)):
        w.step({"action": 0, "reset": False})
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"rewards": [0.0, 1.0, 1.0]}
    assert json.loads(lines[1]) == {"rewards": [0.0, 0.0, 0.0]}
    assert json.loads(lines[2]) == {"rewards": [0.0, 1.0]}


def test_is_first_resets_buffer_mid_episode(tmp_path: Path):
    """Pathological case: an in-progress episode is reset by is_first=True
    before is_last fires. The pre-reset rewards are discarded - the
    flushed line reflects only the post-reset run. Defends against a
    driver-side reset that doesn't go through is_last (e.g. preemption-
    style abort)."""
    sink = tmp_path / "eval_episodes.jsonl"
    # First three steps: an episode that "aborts" (no is_last); then
    # is_first triggers a clean restart, and a normal 3-step episode runs.
    transitions = [
        {"reward": 0, "is_first": True, "is_last": False, "is_terminal": False},
        {"reward": 5, "is_first": False, "is_last": False, "is_terminal": False},
        {"reward": 7, "is_first": False, "is_last": False, "is_terminal": False},
        # is_first=True here - buffer should drop the [0, 5, 7] and start fresh.
        {"reward": 0, "is_first": True, "is_last": False, "is_terminal": False},
        {"reward": 1, "is_first": False, "is_last": False, "is_terminal": False},
        {"reward": 1, "is_first": False, "is_last": True, "is_terminal": True},
    ]
    env = DummyEnv(transitions)
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(len(transitions)):
        w.step({"action": 0, "reset": False})
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"rewards": [0.0, 1.0, 1.0]}


def test_no_flush_until_is_last(tmp_path: Path):
    """Mid-episode: sink file should be empty (no premature flush)."""
    sink = tmp_path / "eval_episodes.jsonl"
    transitions = _episode([0, 0, 0, 1], length=5)  # length 5 but only 4 in list
    env = DummyEnv(transitions)
    w = EvalRewardSink(env, sink_path=sink)
    # Take 3 of 4 steps - episode not yet terminal.
    for _ in range(3):
        w.step({"action": 0, "reset": False})
    # Sink may or may not exist on disk depending on parent-dir creation,
    # but it must contain no flushed lines.
    if sink.exists():
        assert sink.read_text(encoding="utf-8") == ""


def test_action_is_forwarded_unchanged(tmp_path: Path):
    """The wrapper must NOT mutate the action dict. Otherwise the agent's
    own observations downstream would diverge."""
    sink = tmp_path / "eval_episodes.jsonl"
    env = DummyEnv(_episode([0, 1]))
    w = EvalRewardSink(env, sink_path=sink)
    action = {"action": 42, "reset": False}
    w.step(action)
    w.step(action)
    # The inner env saw exactly the actions we passed (same dict identity OK).
    seen_actions = [a for (a, _o) in env.steps_called]
    assert seen_actions == [action, action]


def test_returned_obs_is_unmodified(tmp_path: Path):
    """The wrapper returns the inner env's obs verbatim (same identity).
    The driver downstream expects the full obs dict to flow through."""
    sink = tmp_path / "eval_episodes.jsonl"
    transitions = _episode([0, 1])
    env = DummyEnv(transitions)
    w = EvalRewardSink(env, sink_path=sink)
    returned_obs_0 = w.step({"action": 0, "reset": False})
    returned_obs_1 = w.step({"action": 0, "reset": False})
    assert returned_obs_0 is transitions[0]
    assert returned_obs_1 is transitions[1]


# ---------------------------------------------------------------------------
# Wrapper duck-type surface (delegation to inner env)
# ---------------------------------------------------------------------------


def test_attribute_forwarding(tmp_path: Path):
    """The wrapper forwards attribute lookups to the inner env, matching
    embodied.core.wrappers.Wrapper.__getattr__ semantics. Driver code
    that calls e.g. ``env.obs_space`` or ``env.close()`` must keep
    working through the wrapper."""
    env = DummyEnv([])
    w = EvalRewardSink(env, sink_path=tmp_path / "eval_episodes.jsonl")
    assert w.obs_space == {"image": "stub"}
    assert w.act_space == {"action": "stub"}


def test_attribute_forwarding_missing_attribute_raises(tmp_path: Path):
    """Embodied's Wrapper raises ValueError for missing attributes
    (intentional: prevents silent typo bugs in driver code). Match."""
    env = DummyEnv([])
    w = EvalRewardSink(env, sink_path=tmp_path / "eval_episodes.jsonl")
    with pytest.raises(ValueError):
        _ = w.nonexistent_attribute


def test_len_and_bool_forward(tmp_path: Path):
    """``len(wrapper)`` and ``bool(wrapper)`` forward to the inner env -
    embodied driver / parallel infrastructure relies on this."""
    env = DummyEnv(_episode([0, 1, 1]))
    w = EvalRewardSink(env, sink_path=tmp_path / "eval_episodes.jsonl")
    assert len(w) == 3
    assert bool(w) is True


# ---------------------------------------------------------------------------
# Sink file creation
# ---------------------------------------------------------------------------


def test_creates_parent_dir(tmp_path: Path):
    """Construction creates the sink's parent dir if it doesn't exist -
    avoids a FileNotFoundError on first flush mid-run."""
    sink = tmp_path / "nested" / "deeper" / "eval_episodes.jsonl"
    env = DummyEnv(_episode([0, 1]))
    EvalRewardSink(env, sink_path=sink)
    assert sink.parent.is_dir()


def test_appends_to_existing_file(tmp_path: Path):
    """Two wrapper instances pointing at the same file produce two
    consecutive lines (append mode). Defends against the wrapper being
    re-instantiated mid-run (e.g. eval-env factory closure called more
    than once)."""
    sink = tmp_path / "eval_episodes.jsonl"
    env_a = DummyEnv(_episode([0, 1, 1]))
    w_a = EvalRewardSink(env_a, sink_path=sink)
    for _ in range(3):
        w_a.step({"action": 0, "reset": False})
    env_b = DummyEnv(_episode([0, 0, 1]))
    w_b = EvalRewardSink(env_b, sink_path=sink)
    for _ in range(3):
        w_b.step({"action": 0, "reset": False})
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"rewards": [0.0, 1.0, 1.0]}
    assert json.loads(lines[1]) == {"rewards": [0.0, 0.0, 1.0]}


# ---------------------------------------------------------------------------
# Integration with compute_rhae's loader (round-trip sanity)
# ---------------------------------------------------------------------------


def test_output_round_trips_through_compute_rhae_loader(tmp_path: Path):
    """End-to-end format check: a few episodes written through the sink
    are parsed cleanly by ``scripts.compute_rhae.load_episodes_from_jsonl``.
    Pinned because the sink and the loader live in different files -
    any drift in field name or shape ('rewards' -> 'reward', etc.)
    breaks the dry-run pipeline silently otherwise."""
    from scripts.compute_rhae import load_episodes_from_jsonl

    sink = tmp_path / "eval_episodes.jsonl"
    eps = _episode([0, 0, 1]) + _episode([0, 1, 0, 1])
    env = DummyEnv(eps)
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(len(eps)):
        w.step({"action": 0, "reset": False})
    loaded = load_episodes_from_jsonl(sink)
    assert loaded == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]


# ---------------------------------------------------------------------------
# terminal_state capture (info["state"]) - the terminal cause, recorded directly
# ---------------------------------------------------------------------------


def test_terminal_state_written_when_env_exposes_info(tmp_path: Path):
    """When the inner env exposes ``info["state"]`` (as ARC3EmbodiedEnv
    does), the flushed record carries it as ``terminal_state`` alongside
    the unchanged ``rewards`` stream."""
    sink = tmp_path / "eval_episodes.jsonl"
    env = DummyEnvWithInfo(_episode([0, 0, 1]), info={"state": "GAME_OVER"})
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(3):
        w.step({"action": 0, "reset": False})
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "rewards": [0.0, 0.0, 1.0],
        "terminal_state": "GAME_OVER",
    }


def test_terminal_state_records_win(tmp_path: Path):
    """A solved game reports ``WIN`` - recorded verbatim."""
    sink = tmp_path / "eval_episodes.jsonl"
    env = DummyEnvWithInfo(_episode([0, 1, 1]), info={"state": "WIN"})
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(3):
        w.step({"action": 0, "reset": False})
    assert json.loads(sink.read_text(encoding="utf-8").splitlines()[0]) == {
        "rewards": [0.0, 1.0, 1.0],
        "terminal_state": "WIN",
    }


def test_terminal_state_records_truncation_as_not_finished(tmp_path: Path):
    """A truncated episode (the wrapper sets ``truncated`` while
    ``fd.state`` stays ``NOT_FINISHED``) is recorded as ``NOT_FINISHED`` -
    the field distinguishes a real engine termination from a horizon
    truncation, which is exactly the distinction the post-hoc analysis
    previously had to infer."""
    sink = tmp_path / "eval_episodes.jsonl"
    env = DummyEnvWithInfo(_episode([0, 0, 0]), info={"state": "NOT_FINISHED"})
    w = EvalRewardSink(env, sink_path=sink)
    for _ in range(3):
        w.step({"action": 0, "reset": False})
    assert json.loads(sink.read_text(encoding="utf-8").splitlines()[0]) == {
        "rewards": [0.0, 0.0, 0.0],
        "terminal_state": "NOT_FINISHED",
    }
