"""Eval-env wrapper that records per-step rewards to a JSONL sink.

Per the Phase-4 dry-run hand-off (2026-05-12), DV3's eval logfn pops
the per-step rewards stack before adding to epstats
(``third_party/dreamerv3/embodied/run/train_eval.py:67``), so the
stream is NOT preserved in ``scope/metrics.jsonl`` or wandb. Our
post-hoc RHAE CLI (``scripts/compute_rhae.py``) needs that stream to
segment eval rollouts into per-level AI action counts.

The chosen approach (per session sign-off) is option (ii) from the
Step-5 stop-point: an env-side wrapper that buffers rewards in
memory per episode and flushes one JSONL line on ``is_last``. The
wrapper is applied to the EVAL env factory only — training rollouts
would balloon the file with policy-noise rewards that are not useful
for RHAE.

Output format (one episode per line, matches
``scripts.compute_rhae.load_episodes_from_jsonl``)::

    {"rewards": [r0, r1, r2, ...]}

where ``r0`` is the initial-obs reward (always 0 per DV3 convention;
``segment_episode_actions_per_level`` already skips it) and
``r1..rN`` are the post-action rewards.

The wrapper duck-types ``embodied.core.wrappers.Wrapper`` — same
``__init__(env)`` + attribute-forwarding contract. Avoiding the
subclass keeps this module importable on the laptop without the
embodied/JAX stack, so the unit tests run cleanly.

Concurrency: in append mode, single-line writes are line-atomic on
POSIX and large enough for our line sizes (~10 KB max) on Windows in
practice. Phase 4 uses ``eval_envs=1`` so there is one wrapper
instance and no contention. If parallel eval envs are ever turned on,
either (a) give each worker its own sink path, or (b) add an explicit
file lock around ``_flush``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


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
            self._flush()
        return obs

    def _flush(self) -> None:
        line = json.dumps({"rewards": self._rewards})
        with self._sink_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._rewards = []
