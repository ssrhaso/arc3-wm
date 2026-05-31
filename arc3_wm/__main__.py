"""``python -m arc3_wm`` - a one-command install sanity check.

Imports the package (which self-registers the Gymnasium ids), then prints
the version, the flat action-space size, and every registered
``ARC3/<game>-v0`` id. It touches no ``environment_files/`` and no network,
so it answers "is my arc3_wm install wired up correctly?" without needing
cached games or the JAX side.

    $ python -m arc3_wm
    arc3_wm 0.1.0
    flat action space: 4102 actions
    25 registered Gymnasium ids:
      ARC3/ar25-v0
      ...
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import __version__
from .action_space import N_ACTIONS
from .registration import PUBLIC_GAMES, env_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print package info. Returns a process exit code (0 on success)."""
    _ = argv  # no flags yet; kept for a stable, testable signature
    lines = [
        f"arc3_wm {__version__}",
        f"flat action space: {N_ACTIONS} actions",
        f"{len(PUBLIC_GAMES)} registered Gymnasium ids:",
        *(f"  {env_id(game)}" for game in PUBLIC_GAMES),
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main(sys.argv[1:]))
