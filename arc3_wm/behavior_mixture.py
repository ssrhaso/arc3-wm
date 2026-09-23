"""Type-balanced behavior mixture for the flat 4102-way DreamerV3 actor.

Control arm for the action-factorization result. A ``(type, x, y)`` head does
not only re-express the action: it changes the prior probability of pressing
any button from 5/4102 under the flat head to 5/6 on a game exposing six
action types. To separate that induced type prior from coordinate geometry,
this proxy keeps the flat 4102-way actor untouched and, during *training*
only, replaces each environment's action with probability ``eps(t)`` by a
draw that is uniform over the game's exposed action types and then uniform
within the type (a random cell for ``ACTION6``). ``eps`` anneals linearly
from ``eps0`` to 0 over ``anneal_steps`` environment steps. Evaluation
(``mode='eval'``) is never touched, so RHAE stays a property of the learned
policy.

The proxy sits at the agent boundary: ``embodied.Driver`` records whatever
``policy`` returns as the executed action, so replay and the world model see
the mixed behavior policy, exactly as DreamerV3 sees its own stochastic actor.
Everything else (``train``, ``report``, ``save``, ``load``, ``init_policy``)
is delegated to the wrapped agent. No DreamerV3 code is modified.

Laptop-importable: numpy only.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "TYPE_SLICES",
    "N_ACTIONS",
    "exposed_types_from_mask",
    "sample_type_balanced",
    "TypeBalancedMixturePolicy",
]

N_ACTIONS = 4102
# Flat layout from arc3_wm.action_space: 0-4 ACTION1-5, 5..4100 ACTION6 (64x64 clicks), 4101 ACTION7.
TYPE_SLICES: dict[int, tuple[int, int]] = {
    1: (0, 1), 2: (1, 2), 3: (2, 3), 4: (3, 4), 5: (4, 5),
    6: (5, 4101),
    7: (4101, 4102),
}


def exposed_types_from_mask(mask: np.ndarray) -> list[int]:
    """Action types (1..7) with at least one valid flat index in a 4102-way mask."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape[0] != N_ACTIONS:
        raise ValueError(f"mask must have {N_ACTIONS} entries; got {mask.shape[0]}")
    return [t for t, (lo, hi) in TYPE_SLICES.items() if bool(mask[lo:hi].any())]


def sample_type_balanced(rng: np.random.Generator, types: Sequence[int]) -> int:
    """Uniform over ``types``, then uniform over the flat indices of that type."""
    if not types:
        raise ValueError("no exposed action types to sample from")
    t = int(types[int(rng.integers(0, len(types)))])
    lo, hi = TYPE_SLICES[t]
    return int(rng.integers(lo, hi))


class TypeBalancedMixturePolicy:
    """Wrap a DreamerV3-style agent; mix type-balanced random actions into training."""

    def __init__(
        self,
        agent: Any,
        types: Iterable[int],
        eps0: float = 0.3,
        anneal_steps: int = 200_000,
        seed: int = 0,
        action_key: str = "action",
    ) -> None:
        types = [int(t) for t in types]
        if not types or any(t not in TYPE_SLICES for t in types):
            raise ValueError(f"types must be a non-empty subset of 1..7; got {types}")
        if not 0.0 <= eps0 <= 1.0:
            raise ValueError(f"eps0 must be in [0, 1]; got {eps0}")
        if anneal_steps < 1:
            raise ValueError(f"anneal_steps must be >= 1; got {anneal_steps}")
        self._agent = agent
        self.types = types
        self.eps0 = float(eps0)
        self.anneal_steps = int(anneal_steps)
        self.action_key = action_key
        self._rng = np.random.default_rng(seed)
        self.env_steps = 0
        self.n_redrawn = 0

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._agent, name)

    def eps(self) -> float:
        frac = 1.0 - self.env_steps / self.anneal_steps
        return self.eps0 * max(0.0, frac)

    def policy(self, carry, obs, mode: str = "train"):
        carry, acts, outs = self._agent.policy(carry, obs, mode=mode)
        if mode != "train":
            return carry, acts, outs
        actions = np.array(acts[self.action_key], copy=True)
        n = int(actions.shape[0])
        eps = self.eps()
        self.env_steps += n
        if eps > 0.0:
            redraw = self._rng.random(n) < eps
            for i in np.flatnonzero(redraw):
                actions[i] = sample_type_balanced(self._rng, self.types)
            self.n_redrawn += int(redraw.sum())
        acts = dict(acts)
        acts[self.action_key] = actions.astype(acts[self.action_key].dtype, copy=False)
        return carry, acts, outs

    def stats(self) -> dict[str, float]:
        return {
            "env_steps": float(self.env_steps),
            "n_redrawn": float(self.n_redrawn),
            "eps": self.eps(),
        }
