"""Eval-env wrapper that records per-step rewards to a JSONL sink.

DV3's eval logfn pops the per-step rewards stack before adding to
epstats (``third_party/dreamerv3/embodied/run/train_eval.py:67``), so the
stream is not preserved in ``scope/metrics.jsonl`` or wandb. The post-hoc
RHAE CLI (``scripts/compute_rhae.py``) needs it to segment eval rollouts
into per-level AI action counts. This wrapper buffers rewards in memory
per episode and flushes one JSONL line on ``is_last``. Apply it to the
eval env factory only; training rollouts would balloon the file with
policy-noise rewards that are not useful for RHAE.

Output format (one episode per line, matches
``scripts.compute_rhae.load_episodes_from_jsonl``)::

    {"rewards": [r0, r1, r2, ...], "terminal_state": "GAME_OVER"}

where ``r0`` is the initial-obs reward (always 0 per DV3 convention;
``segment_episode_actions_per_level`` already skips it) and ``r1..rN``
are the post-action rewards.

``terminal_state`` is the inner env's terminal ``fd.state`` name -
``"WIN"`` / ``"GAME_OVER"`` for a ``terminated`` episode, ``"NOT_FINISHED"``
for a ``truncated`` one - read from ``info["state"]`` (written by
``ARC3GymEnv._info`` at ``arc3_wm/env.py:214`` and re-exposed by
``ARC3EmbodiedEnv.info``). It records the terminal cause directly so
future eval runs no longer have to infer GAME_OVER-vs-WIN from
``reward == 0``. The key is **omitted** when the inner env exposes no
``info["state"]`` (a non-ARC env or a bare test double), so the record
degrades to the legacy ``{"rewards": [...]}`` shape. The format is
backward-compatible either way: ``compute_rhae`` keys only on
``"rewards"`` and ``ai_actions = len(rewards) - 1`` is unchanged.

The wrapper duck-types ``embodied.core.wrappers.Wrapper`` (same
``__init__(env)`` + attribute-forwarding contract). Avoiding the subclass
keeps this module importable on the laptop without the embodied/JAX
stack, so the unit tests run cleanly.

Concurrency: in append mode, single-line writes are line-atomic on POSIX
and small enough (~10 KB max) to be so on Windows in practice. Phase 4
uses ``eval_envs=1``, so there is one wrapper instance and no contention.
For parallel eval envs, either give each worker its own sink path or add
an explicit file lock around ``_flush``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["EvalRewardSink"]


class EvalRewardSink:
    """Wraps an embodied.Env to record per-step rewards to a JSONL sink.

    On every step the wrapper forwards the action to the inner env,
    appends the returned ``reward`` to an in-memory episode buffer,
    and flushes one JSON line when the inner env returns
    ``is_last=True``. The buffer resets on ``is_first=True``.

    Intended exclusively for the eval-env factory in
    ``embodied.run.train_eval``. Training rollouts must NOT go through
    this wrapper (would record policy-noise rewards that are not
    useful for RHAE and would balloon the file).
    """

    def __init__(self, env: Any, sink_path: Path | str) -> None:
        self.env = env
        self._sink_path = Path(sink_path)
        self._rewards: list[float] = []
        self._sink_path.parent.mkdir(parents=True, exist_ok=True)

    # --- embodied.core.wrappers.Wrapper duck-type surface ----------------

    def __len__(self) -> int:
        return len(self.env)

    def __bool__(self) -> bool:
        return bool(self.env)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self.env, name)
        except AttributeError:
            raise ValueError(name)

    # --- the actual sink behaviour ----------------------------------------

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = self.env.step(action)
        # ``is_first`` and ``is_last`` are dict keys on embodied transitions.
        # On a reset step both may be set on the first obs only;
        # subsequent steps update them per episode boundary.
        if obs.get("is_first"):
            self._rewards = []
        self._rewards.append(float(obs["reward"]))
        if obs.get("is_last"):
            self._flush(self._read_terminal_state())
        return obs

    def _read_terminal_state(self) -> str | None:
        """Best-effort read of the inner env's terminal ``fd.state`` name.

        ``ARC3GymEnv._info`` writes ``info["state"] = fd.state.name``
        (``arc3_wm/env.py:214``) and ``ARC3EmbodiedEnv`` re-exposes it via
        its ``.info`` property; the read traverses any embodied wrappers
        (``UnifyDtypes`` / ``CheckSpaces``) through their ``__getattr__``.
        At ``is_last`` the inner env's ``.info`` still reflects the
        terminal step - embodied's auto-reset only fires on the *next*
        step - so this is the true terminal cause.

        Every failure mode degrades to ``None`` (no ``.info``, no
        ``"state"`` key, a non-Mapping ``.info``): logging must never
        break an eval run, and ``None`` makes ``_flush`` emit the legacy
        ``{"rewards": [...]}`` shape.
        """
        try:
            state = self.env.info.get("state")
        except (AttributeError, ValueError, TypeError):
            return None
        return str(state) if state is not None else None

    def _flush(self, terminal_state: str | None = None) -> None:
        record: dict[str, Any] = {"rewards": self._rewards}
        if terminal_state is not None:
            record["terminal_state"] = terminal_state
        line = json.dumps(record)
        with self._sink_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._rewards = []
