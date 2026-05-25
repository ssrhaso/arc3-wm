"""Flat 4102-way action space for ARC-AGI-3.

Layout (matches CLAUDE.md §"Action space"):

    0..4    -> ACTION1..ACTION5  (parameter-less)
    5..4100 -> ACTION6 with (x, y) = unravel_index(idx - 5, (64, 64))
    4101    -> ACTION7

ACTION6 is the only "complex" action (carries x, y in [0, 63]).
``GameAction.RESET`` is *not* in the flat space — the wrapper calls
``env.reset()`` directly. Per-step masking comes from
``fd.available_actions`` (a list of int IDs in 1..7); use
:func:`build_mask` to project that to a length-4102 boolean array.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from arcengine import GameAction

GRID = 64
ACTION6_BASE = 5
ACTION6_COUNT = GRID * GRID  # 4096
ACTION7_INDEX = ACTION6_BASE + ACTION6_COUNT  # 4101
N_ACTIONS = ACTION7_INDEX + 1  # 4102


def flat_to_arc(idx: int) -> Tuple[GameAction, Optional[dict]]:
    """Decode a flat index to ``(GameAction, data_or_None)``.

    ``data`` is ``None`` for parameter-less actions and ``{"x": int, "y": int}``
    for ACTION6.
    """
    if not (0 <= idx < N_ACTIONS):
        raise ValueError(f"flat action index {idx} out of range [0, {N_ACTIONS})")
    if idx < ACTION6_BASE:
        # GameAction is an Enum with tuple values internally; value-lookup via
        # GameAction(int) raises. Use the dedicated from_id classmethod.
        return GameAction.from_id(idx + 1), None  # 0..4 -> ACTION1..ACTION5
    if idx == ACTION7_INDEX:
        return GameAction.ACTION7, None
    # ACTION6 grid
    rel = idx - ACTION6_BASE
    y, x = divmod(rel, GRID)  # row-major: idx = ACTION6_BASE + y*GRID + x
    return GameAction.ACTION6, {"x": int(x), "y": int(y)}


def arc_to_flat(action: GameAction, x: Optional[int] = None, y: Optional[int] = None) -> int:
    """Encode ``(GameAction, x, y)`` to a flat index.

    ``x`` and ``y`` are required iff ``action == ACTION6``; both must be in
    ``[0, 63]``. Raises on RESET — RESET is not in the flat space.
    """
    if action == GameAction.RESET:
        raise ValueError("RESET is not in the flat action space; call env.reset() directly")
    if action == GameAction.ACTION6:
        if x is None or y is None:
            raise ValueError("ACTION6 requires x, y")
        if not (0 <= x < GRID and 0 <= y < GRID):
            raise ValueError(f"ACTION6 (x, y) must be in [0, {GRID}); got ({x}, {y})")
        return ACTION6_BASE + y * GRID + x
    if action == GameAction.ACTION7:
        return ACTION7_INDEX
    # ACTION1..ACTION5 -> 0..4
    if 1 <= action.value <= 5:
        return action.value - 1
    raise ValueError(f"unsupported GameAction: {action!r}")


def build_mask(available_actions: Iterable[int]) -> np.ndarray:
    """Return a length-4102 boolean mask of currently-valid flat indices.

    ``available_actions`` is a list of integer action IDs in ``1..7`` (the
    raw form the engine returns on ``FrameDataRaw.available_actions``).
    ACTION6 is treated as a single "click is allowed" flag — when
    ``6 ∈ available``, all 4096 grid cells are unmasked.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    avail = set(int(a) for a in available_actions)
    for aid in (1, 2, 3, 4, 5):
        if aid in avail:
            mask[aid - 1] = True
    if 6 in avail:
        mask[ACTION6_BASE : ACTION6_BASE + ACTION6_COUNT] = True
    if 7 in avail:
        mask[ACTION7_INDEX] = True
    return mask


def logit_bias(mask: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    """Additive bias for actor logits: ``0.0`` where allowed, ``-inf`` where masked.

    This is the canonical realisation of the masking step CLAUDE.md
    describes ("set actor logits to ``-inf`` on unsupported indices before
    sampling"): add the returned array to the policy logits, then sample or
    argmax as usual. Adding ``-inf`` drives the post-softmax probability of
    masked actions to exactly zero while leaving the relative logits of the
    allowed actions untouched.

    ``mask`` is the length-4102 boolean array from :func:`build_mask`. The
    result has the same shape. Keeping bias generation separate from
    :func:`build_mask` lets callers cache the boolean mask (e.g. to log the
    number of valid actions) and derive the additive bias on demand.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (N_ACTIONS,):
        raise ValueError(f"mask must have shape ({N_ACTIONS},); got {mask.shape}")
    bias = np.zeros(N_ACTIONS, dtype=dtype)
    bias[~mask] = -np.inf
    return bias
