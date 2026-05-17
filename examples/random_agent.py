"""Plug-and-play demo: a random agent on ARC-AGI-3 via the Gymnasium seam.

This is the minimal proof that ARC-AGI-3 is a standard `gymnasium.Env`
through `arc3_wm` — no JAX, no DreamerV3, no GPU, ~10 lines of agent.
Anything that drives a Gym env drives this.

Prerequisites (one-time, see docs/using-the-wrapper.md):
  - `pip install -e .`
  - `ARC_API_KEY` exported, then `python scripts/cache_env_files.py`
  - a `.env` with OPERATION_MODE=offline, ENVIRONMENTS_DIR=environment_files

Usage:
  python examples/random_agent.py --game vc33 --episodes 3
  python examples/random_agent.py --game vc33 --episodes 3 --mask
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="vc33", help="ARC-AGI-3 game id (default: vc33)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mask",
        action="store_true",
        help="Sample only from info['action_mask'] (recommended for real agents).",
    )
    args = p.parse_args(argv)

    # Imported here so --help works without arc_agi installed/configured.
    try:
        import arc_agi
        from arc3_wm.env import ARC3GymEnv
    except ImportError as e:
        print(f"Missing dependency: {e}. Run `pip install -e .`", file=sys.stderr)
        return 2

    try:
        arcade = arc_agi.Arcade()
        env = ARC3GymEnv(
            game_id=args.game,
            seed=args.seed,
            max_steps=args.max_steps,
            arcade=arcade,
        )
    except RuntimeError as e:
        # ARC3GymEnv raises clear errors for the two common setup misses:
        # non-OFFLINE mode and uncached environment_files/<game>/.
        print(f"Setup error: {e}", file=sys.stderr)
        print(
            "See docs/using-the-wrapper.md § Prerequisites "
            "(cache_env_files.py + .env).",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(args.seed)
    print(
        f"game={args.game} obs={env.observation_space.shape} "
        f"actions={env.action_space.n} mask={'on' if args.mask else 'off'}"
    )

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        while True:
            if args.mask:
                valid = np.flatnonzero(info["action_mask"])
                action = int(rng.choice(valid)) if valid.size else 0
            else:
                action = int(rng.integers(env.action_space.n))
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
