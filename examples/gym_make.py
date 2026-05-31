"""Plug-and-play demo: drive ARC-AGI-3 through the *registered* gym id.

This is the "formalised gym env" proof: no direct class import, no
`arc3_wm` symbol referenced in the agent at all - just the standard
`gymnasium.make("ARC3/<game>-v0")` entry point that `import arc3_wm`
registers. Anything that resolves a Gymnasium id (sb3, cleanrl, your
own loop) reaches ARC-AGI-3 this way.

Contrast with `random_agent.py`, which constructs `ARC3GymEnv`
directly. Both paths are supported; this one is the discoverable one.

Prerequisites (one-time, see docs/using-the-wrapper.md):
  - `pip install -e .`
  - `ARC_API_KEY` exported, then `python scripts/cache_env_files.py`
  - a `.env` with OPERATION_MODE=offline, ENVIRONMENTS_DIR=environment_files

Usage:
  python examples/gym_make.py --game vc33 --episodes 3
  python examples/gym_make.py --list
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="vc33", help="ARC-AGI-3 game id (default: vc33)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--list",
        action="store_true",
        help="Print every registered ARC3/<game>-v0 id and exit.",
    )
    args = p.parse_args(argv)

    # Imported here so --help works without the deps installed/configured.
    try:
        import gymnasium as gym
        import numpy as np

        # The import that performs registration. No other arc3_wm symbol
        # is used below - the registered id is the whole integration point.
        import arc3_wm
        from arc3_wm.registration import env_id
    except ImportError as e:
        print(f"Missing dependency: {e}. Run `pip install -e .`", file=sys.stderr)
        return 2

    if args.list:
        for g in arc3_wm.PUBLIC_GAMES:
            print(env_id(g))
        return 0

    eid = env_id(args.game)
    if eid not in gym.registry:
        print(f"{eid!r} is not registered (unknown game {args.game!r})", file=sys.stderr)
        return 1

    try:
        env = gym.make(eid, seed=args.seed, max_steps=args.max_steps)
    except Exception as e:  # arc3_wm raises clear RuntimeErrors for setup misses
        print(f"Setup error: {e}", file=sys.stderr)
        print(
            "See docs/using-the-wrapper.md Section Prerequisites "
            "(cache_env_files.py + .env).",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(args.seed)
    print(
        f"id={eid} obs={env.observation_space.shape} "
        f"actions={env.action_space.n}"
    )

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        while True:
            valid = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(valid)) if valid.size else 0
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        print(
            f"  episode {ep}: steps={steps} reward={total_reward:.0f} "
            f"levels_completed={info['levels_completed']}/{info['win_levels']} "
            f"state={info['state']} "
            f"end={'terminated' if terminated else 'truncated'}"
        )

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
