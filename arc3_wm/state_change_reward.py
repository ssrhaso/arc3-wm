"""Training-only state-change reward bonus for ARC-AGI-3 (embodied wrapper).

The Phase-4 diagnosis is reward cold start: on most games
the native ``r = delta levels_completed`` never fires under a uniform policy,
so the reward head has no positive example, imagined returns are constant and
the actor stays at its uniform fixed point. This wrapper manufactures a dense
signal for "you affected the world" without touching DreamerV3:

    r' = r + beta * [grid changed] / sqrt(N(grid'))

where ``N(grid')`` counts visits to the exact post-action grid (a hash of the
64x64x3 frame), so repeatedly flickering the same cells earns a vanishing
bonus (count-based novelty gating). ``beta = 0`` disables the wrapper.

Duck-types ``embodied.core.wrappers.Wrapper`` exactly like
``arc3_wm.eval_reward_sink.EvalRewardSink`` so it is importable and testable on
a laptop without JAX. It must wrap the *training* env factory only; the eval
env stays native so RHAE remains a post-hoc measurement of native progress.

The per-step bonus is exposed as ``log/bonus`` (a scalar, so DreamerV3's
episode logging reports its avg/max/sum per episode) and the change flag as
``log/changed``. Both keys are appended to ``obs_space`` so
``embodied.wrappers.CheckSpaces`` accepts them.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

__all__ = ["StateChangeRewardWrapper", "grid_hash"]

OBS_KEY = "image"
BONUS_KEY = "log/bonus"
CHANGED_KEY = "log/changed"


def grid_hash(frame: np.ndarray) -> bytes:
    """Exact-content hash of a frame (dtype and shape included)."""
    arr = np.ascontiguousarray(frame)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.digest()


class StateChangeRewardWrapper:
    """Add a novelty-gated state-change bonus to the native reward."""

    def __init__(self, env: Any, beta: float, gate: str = "novelty") -> None:
        if beta < 0:
            raise ValueError(f"beta must be >= 0; got {beta}")
        if gate not in ("novelty", "binary"):
            raise ValueError(f"gate must be 'novelty' or 'binary'; got {gate!r}")
        self.env = env
        self.beta = float(beta)
        self.gate = gate
        self._prev: bytes | None = None
        self._counts: defaultdict[bytes, int] = defaultdict(int)
        self.total_bonus = 0.0
        self.n_changed = 0
        self.n_steps = 0

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

    @property
    def obs_space(self) -> dict:
        import elements  # deferred: not needed to import this module on a laptop

        space = dict(self.env.obs_space)
        space[BONUS_KEY] = elements.Space(np.float32)
        space[CHANGED_KEY] = elements.Space(np.float32)
        return space

    # --- the shaping ------------------------------------------------------

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = dict(self.env.step(action))
        frame = np.asarray(obs[OBS_KEY])
        h = grid_hash(frame)
        bonus = 0.0
        changed = 0.0
        if bool(obs.get("is_first", False)) or self._prev is None:
            # Episode start: nothing to compare against, no bonus.
            self._prev = h
        else:
            self.n_steps += 1
            if h != self._prev:
                changed = 1.0
                self.n_changed += 1
                self._counts[h] += 1
                if self.gate == "novelty":
                    bonus = self.beta / float(np.sqrt(self._counts[h]))
                else:
                    bonus = self.beta
            self._prev = h
        if bool(obs.get("is_last", False)):
            self._prev = None
        self.total_bonus += bonus
        obs["reward"] = np.float32(float(obs["reward"]) + bonus)
        obs[BONUS_KEY] = np.float32(bonus)
        obs[CHANGED_KEY] = np.float32(changed)
        return obs

    def stats(self) -> dict[str, float]:
        """Run-level diagnostics: change rate, distinct grids, bonus paid."""
        return {
            "steps": float(self.n_steps),
            "changed_frac": (self.n_changed / self.n_steps) if self.n_steps else 0.0,
            "distinct_grids": float(len(self._counts)),
            "total_bonus": float(self.total_bonus),
        }
