"""arc3_wm - a world-model RL substrate for ARC-AGI-3.

This package makes a single ARC-AGI-3 game usable as a standard RL
environment and supplies the offline-data and metric plumbing for
model-based RL on it. The two adapters are the durable contribution:

- :class:`~arc3_wm.env.ARC3GymEnv` - a stock ``gymnasium.Env``
  (pure-Python, no JAX). Registered Gymnasium ids ``ARC3/<game>-v0``
  are available after ``import arc3_wm`` (see
  :mod:`arc3_wm.registration`).
- :class:`~arc3_wm.embodied_env.ARC3EmbodiedEnv` - the same game behind
  DreamerV3's ``embodied.Env`` interface. Exposed *lazily*: importing it
  pulls the JAX-side ``elements`` dependency, so it is resolved only on
  first attribute access to keep the laptop/Gymnasium path import-clean.

The flat 4102-way action space and its masking helpers are re-exported
from :mod:`arc3_wm.action_space`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .action_space import (
    N_ACTIONS,
    arc_to_flat,
    build_mask,
    describe_action,
    flat_to_arc,
    logit_bias,
)
from .env import ARC3GymEnv
from .registration import PUBLIC_GAMES, register_envs

__version__ = "0.2.0"

# Register the Gymnasium ids on import - the standard convention for a
# packaged gym environment (cf. how `ale_py` / `gymnasium.envs` self-register).
# Idempotent and side-effect-free beyond the registry mutation.
register_envs()

__all__ = [
    "ARC3GymEnv",
    "ARC3EmbodiedEnv",
    "N_ACTIONS",
    "flat_to_arc",
    "arc_to_flat",
    "build_mask",
    "describe_action",
    "logit_bias",
    "PUBLIC_GAMES",
    "register_envs",
    "__version__",
]

if TYPE_CHECKING:  # for type-checkers only; not imported at runtime
    from .embodied_env import ARC3EmbodiedEnv


def __getattr__(name: str) -> Any:
    """PEP 562 lazy export of ``ARC3EmbodiedEnv``.

    Kept out of the eager import path because :mod:`arc3_wm.embodied_env`
    imports ``elements`` (DreamerV3 / JAX side), which is not installable
    on the laptop-only Gymnasium path.
    """
    if name == "ARC3EmbodiedEnv":
        from .embodied_env import ARC3EmbodiedEnv

        return ARC3EmbodiedEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
