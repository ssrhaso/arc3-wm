"""Bridge ``arc3_wm.env.ARC3GymEnv`` to DreamerV3's ``embodied.Env`` interface.

Per design-decisions.md D12, we bypass ``dreamerv3/main.py``; this module
is the env half of the bridge. ``scripts/launch_pergame.py`` (a future
deliverable) imports ``ARC3EmbodiedEnv`` and builds the rest of the
training loop (agent, replay, driver).

Key translations vs. ``embodied/envs/from_gym.py``:

- Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)`` ->
  embodied step-dict with ``is_terminal = terminated`` (NOT on truncated)
  and ``is_last = terminated OR truncated``.
- Single-array obs -> ``{"image": ndarray, "log/action_mask": ndarray, ...}``.
  The ``log/`` prefix is a documented embodied convention for keys that
  the agent should not consume - see ``embodied/core/base.py``.
- Action space -> ``{"action": Discrete(4102), "reset": bool}``, matching
  what ``FromGym`` produces.

Action masking is exposed but **not enforced** (D11). The agent samples
from the full discrete space; ``arc_agi`` no-ops dead actions silently.
"""
from __future__ import annotations

from typing import Any, Optional

import elements
import numpy as np

import arc_agi

from .action_space import N_ACTIONS
from .env import OBS_HW, ARC3GymEnv

# We duck-type the embodied.Env interface rather than subclassing it.
# embodied.core.base.Env is a stub (only NotImplementedError raisers); the
# rest of embodied (Driver, Replay, wrappers) uses duck-typing on
# obs_space / act_space / step / close. Subclassing would require
# `from embodied.core.base import Env`, which triggers
# `embodied/__init__.py` -> `import portal` and the JAX-flavoured
# transitive deps - not installable on the laptop. The real `embodied`
# package on Vast.ai accepts duck-typed envs unchanged.

OBS_KEY = "image"
ACT_KEY = "action"


class ARC3EmbodiedEnv:
    """``embodied.Env`` adapter over a single-game ``ARC3GymEnv``."""

    def __init__(
        self,
        game_id: str = "vc33",
        seed: int = 0,
        max_steps: int = 1000,
        arcade: Optional[arc_agi.Arcade] = None,
    ) -> None:
        self._gym = ARC3GymEnv(
            game_id=game_id, seed=seed, max_steps=max_steps, arcade=arcade
        )
        # `_done=True` forces the next step() call to reset, regardless of
        # what `action['reset']` says - matches FromGym's bootstrap logic.
        self._done = True
        self._info: dict[str, Any] = {}

    # --- embodied.Env interface ------------------------------------------

    @property
    def obs_space(self) -> dict[str, elements.Space]:
        return {
            OBS_KEY: elements.Space(np.uint8, (OBS_HW, OBS_HW, 3), 0, 255),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self) -> dict[str, elements.Space]:
        return {
            ACT_KEY: elements.Space(np.int32, (), 0, N_ACTIONS),
            "reset": elements.Space(bool),
        }

    @property
    def info(self) -> dict[str, Any]:
        return self._info

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        # Driver-initiated reset OR auto-reset after a previous terminal/truncated.
        if action.get("reset") or self._done:
            self._done = False
            obs, info = self._gym.reset()
            self._info = info
            return self._pack(obs, info, reward=0.0, is_first=True, is_last=False, is_terminal=False)

        a = int(action[ACT_KEY])
        obs, reward, terminated, truncated, info = self._gym.step(a)
        self._info = info
        self._done = bool(terminated or truncated)
        return self._pack(
            obs,
            info,
            reward=float(reward),
            is_first=False,
            is_last=bool(terminated or truncated),
            is_terminal=bool(terminated),  # truncated is NOT terminal
        )

    def close(self) -> None:
        self._gym.close()

    # --- internals --------------------------------------------------------

    def _pack(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        *,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> dict[str, Any]:
        return {
            OBS_KEY: np.asarray(obs, dtype=np.uint8),
            "reward": np.float32(reward),
            "is_first": np.bool_(is_first),
            "is_last": np.bool_(is_last),
            "is_terminal": np.bool_(is_terminal),
        }
