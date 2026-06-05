"""Download ARC-AGI-3 env files into environment_files/ for OFFLINE use.

OFFLINE mode requires environment_files/<game>/metadata.json present locally.
arc_agi.Arcade.make() in NORMAL mode is the only way to fetch them; after this
runs, OFFLINE-mode make() succeeds for the cached games. Idempotent: arc_agi
only downloads games not already present in environment_files/.

By default it caches the Phase-4 game set (pilot trio + expansion trio). Pass
game ids to cache a specific subset, or --all for every one of the 25 public
games:

    python scripts/cache_env_files.py                # default Phase-4 set
    python scripts/cache_env_files.py vc33 sb26      # just these two
    python scripts/cache_env_files.py --all          # all 25 public games

NOTE on Arcade priority logic: env var takes precedence over constructor arg
*unless* constructor arg is non-NORMAL. Since we want NORMAL, we set
OPERATION_MODE=normal in os.environ BEFORE importing arc_agi.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["OPERATION_MODE"] = "normal"

# Import only after env override - arc_agi.base.load_dotenv runs at import time
# but with override=False, so our os.environ takes precedence over .env.
import arc_agi  # noqa: E402

from arc3_wm.registration import PUBLIC_GAMES  # noqa: E402

# Default set: the pilot trio (vc33, tu93, cd82) plus the Phase-4 expansion
# trio (sb26, tn36, ls20, lf52), so OFFLINE-mode make() succeeds for every
# Phase-4 game in one pass.
DEFAULT_GAMES = ["vc33", "tu93", "cd82", "sb26", "tn36", "ls20", "lf52"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "games",
        nargs="*",
        help="game ids to cache (default: the Phase-4 set)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="cache all 25 public games",
    )
    return p.parse_args(argv)


def resolve_games(args: argparse.Namespace) -> list[str]:
    """Return the game ids to cache, or raise ValueError on bad input."""
    if args.all and args.games:
        raise ValueError("pass either game ids or --all, not both")
    if args.all:
        return list(PUBLIC_GAMES)
    if args.games:
        unknown = [g for g in args.games if g not in PUBLIC_GAMES]
        if unknown:
            raise ValueError(
                f"unknown game id(s) {unknown}; expected from the 25 public "
                f"games (arc3_wm.PUBLIC_GAMES)"
            )
        return list(args.games)
    return list(DEFAULT_GAMES)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        games = resolve_games(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not os.environ.get("ARC_API_KEY"):
        print("ARC_API_KEY missing (needed for NORMAL-mode download).", file=sys.stderr)
        return 2

    arc = arc_agi.Arcade()
    print(f"operation_mode = {arc.operation_mode}")
    if arc.operation_mode != arc_agi.OperationMode.NORMAL:
        print("Refusing to run: must be NORMAL to download.", file=sys.stderr)
        return 2

    failures = []
    for gid in games:
        print(f"\n--- caching {gid} ---")
        env = arc.make(gid)
        if env is None:
            print(f"FAIL: arc.make({gid!r}) returned None")
            failures.append(gid)
            continue
        print(f"OK: {gid} -> wrapper type={type(env).__name__}")

    print()
    if failures:
        print(f"{len(failures)} failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
