"""Gymnasium registration for the 25 public ARC-AGI-3 games.

A formalised gym environment is discoverable through the standard
``gymnasium.make`` entry point, not only via a direct class import. This
module registers one id per public game::

    import arc3_wm                       # registers on import
    import gymnasium as gym
    env = gym.make("ARC3/vc33-v0")       # -> ARC3GymEnv(game_id="vc33")

Design notes:

- **No ``max_episode_steps``.** :class:`~arc3_wm.env.ARC3GymEnv` already
  truncates internally at its ``max_steps`` (D9). Letting Gymnasium wrap
  a second ``TimeLimit`` would double-truncate and desync ``info``.
- **``disable_env_checker=True``.** The passive env checker would step a
  real game at ``make`` time, which needs cached ``environment_files/``;
  the wrapper has its own contract tests (``tests/test_wrapper_spec.py``).
- **Idempotent.** ``register_envs()`` is safe to call repeatedly and is
  invoked once from ``arc3_wm/__init__.py``; already-present ids are
  skipped rather than re-registered.

``PUBLIC_GAMES`` is the canonical 25-id set (matches
``data/human_baselines.json`` keys). It is hard-coded here, not derived
from that fixture, so registration works in a fresh clone with no data.
"""
from __future__ import annotations

__all__ = ["PUBLIC_GAMES", "ENV_ID_FMT", "env_id", "register_envs"]

# The 25 public-demo ARC-AGI-3 games (sorted). Stable, documented set.
PUBLIC_GAMES: tuple[str, ...] = (
    "ar25", "bp35", "cd82", "cn04", "dc22",
    "ft09", "g50t", "ka59", "lf52", "lp85",
    "ls20", "m0r0", "r11l", "re86", "s5i5",
    "sb26", "sc25", "sk48", "sp80", "su15",
    "tn36", "tr87", "tu93", "vc33", "wa30",
)

ENV_ID_FMT = "ARC3/{game}-v0"
_ENTRY_POINT = "arc3_wm.env:ARC3GymEnv"


def env_id(game: str) -> str:
    """Return the registered Gymnasium id for ``game`` (e.g. ``ARC3/vc33-v0``)."""
    return ENV_ID_FMT.format(game=game)


def register_envs() -> tuple[str, ...]:
    """Register ``ARC3/<game>-v0`` for every game in :data:`PUBLIC_GAMES`.

    Idempotent: ids already in ``gymnasium.registry`` are left untouched.
    Returns the full tuple of ids this package owns (registered or
    already-present), in :data:`PUBLIC_GAMES` order.
    """
    import gymnasium as gym

    ids: list[str] = []
    for game in PUBLIC_GAMES:
        eid = env_id(game)
        ids.append(eid)
        if eid in gym.registry:
            continue
        gym.register(
            id=eid,
            entry_point=_ENTRY_POINT,
            kwargs={"game_id": game},
            disable_env_checker=True,
        )
    return tuple(ids)
