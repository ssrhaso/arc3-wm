"""Minimal Gymnasium wrapper over a single ARC-AGI-3 game.

Milestone (1) of the Phase-1 rescoping (see docs/design-decisions.md):
single game (default ``vc33``), OFFLINE mode, no replay loader, no RHAE.

Observation: ``Box(0, 255, (64, 64, 3), uint8)`` — ``frame[-1]`` palette-decoded.
Action:      ``Discrete(4102)`` — flat layout from ``arc3_wm.action_space``.
Reward:      ``levels_completed[t+1] - levels_completed[t]`` (D4).
Episode end: terminated on ``state ∈ {WIN, GAME_OVER}``;
             truncated on ``max_steps`` timeout (default 1000, D9).

Action masking is exposed via ``info["action_mask"]`` after every reset/step.
DreamerV3 doesn't read it directly, but downstream training code can apply
it to the policy logits.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from arcengine import GameAction, GameState

import arc_agi

from . import action_space as A
from .palette import decode_frame

OBS_HW = 64
TERMINAL_STATES = frozenset({GameState.WIN, GameState.GAME_OVER})


class ARC3GymEnv(gym.Env):
    """Single-game Gymnasium env over ``arc_agi.Arcade`` in OFFLINE mode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        game_id: str = "vc33",
        seed: int = 0,
        max_steps: int = 1000,
        arcade: Optional[arc_agi.Arcade] = None,
    ) -> None:
        super().__init__()
        # The arc_agi.Arcade is heavy (creates a scorecard, scans environment_files),
        # so allow callers to share one across vector envs / multi-game runs.
        self._arcade = arcade if arcade is not None else arc_agi.Arcade()
        if self._arcade.operation_mode != arc_agi.OperationMode.OFFLINE:
            raise RuntimeError(
                f"ARC3GymEnv requires OFFLINE mode; arc_agi.Arcade was created "
                f"with operation_mode={self._arcade.operation_mode}. Set "
                f"OPERATION_MODE=offline in .env."
            )
        self._game_id = game_id
        self._seed = int(seed)
        self._max_steps = int(max_steps)

        wrapper = self._arcade.make(game_id, seed=self._seed)
        if wrapper is None:
            raise RuntimeError(
                f"arc_agi.Arcade.make({game_id!r}) returned None; "
                f"environment_files/{game_id}/ may not be cached."
            )
        self._wrapper = wrapper

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(OBS_HW, OBS_HW, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(A.N_ACTIONS)

        self._steps = 0
        self._prev_levels_completed = 0
        self._last_available: list[int] = []

    # --- Gymnasium interface ----------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None and seed != self._seed:
            # arc_agi seed is set at make() time; rebuild the wrapper to honour it.
            self._seed = int(seed)
            wrapper = self._arcade.make(self._game_id, seed=self._seed)
            if wrapper is None:
                raise RuntimeError(
                    f"arc_agi.Arcade.make({self._game_id!r}, seed={self._seed}) returned None"
                )
            self._wrapper = wrapper
        super().reset(seed=seed)
        fd = self._wrapper.reset()
        if fd is None:
            raise RuntimeError(f"{self._game_id} reset() returned None")
        self._steps = 0
        self._prev_levels_completed = int(fd.levels_completed)
        return self._obs(fd), self._info(fd)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        action = int(action)
        arc_action, data = A.flat_to_arc(action)
        fd = self._wrapper.step(arc_action, data=data)
        if fd is None:
            raise RuntimeError(
                f"{self._game_id} step({arc_action.name}) returned None"
            )

        levels = int(fd.levels_completed)
        reward = float(levels - self._prev_levels_completed)
        self._prev_levels_completed = levels

        self._steps += 1
        terminated = fd.state in TERMINAL_STATES
        truncated = (not terminated) and self._steps >= self._max_steps

        return self._obs(fd), reward, terminated, truncated, self._info(fd)

    def close(self) -> None:
        # arc_agi has no explicit close; drop our reference for GC.
        self._wrapper = None  # type: ignore[assignment]

    # --- internals --------------------------------------------------------

    def _obs(self, fd) -> np.ndarray:
        # D2: take the last animation layer, palette-decode to (H, W, 3) uint8.
        if not fd.frame:
            # Engine should never return empty; treat as black frame to avoid
            # propagating None into DreamerV3.
            return np.zeros((OBS_HW, OBS_HW, 3), dtype=np.uint8)
        layer = np.asarray(fd.frame[-1])
        if layer.shape != (OBS_HW, OBS_HW):
            raise RuntimeError(
                f"unexpected frame layer shape {layer.shape}; expected ({OBS_HW}, {OBS_HW})"
            )
        return decode_frame(layer)

    def _info(self, fd) -> dict[str, Any]:
        avail = list(int(a) for a in (fd.available_actions or ()))
        self._last_available = avail
        return {
            "available_actions": avail,
            "action_mask": A.build_mask(avail),
            "levels_completed": int(fd.levels_completed),
            "win_levels": int(fd.win_levels),
            "state": fd.state.name if hasattr(fd.state, "name") else str(fd.state),
            "guid": fd.guid,
            "steps": self._steps,
        }
