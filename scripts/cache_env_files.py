"""One-shot: download env files for the pilot games into environment_files/.

OFFLINE mode requires environment_files/<game>/metadata.json present locally.
arc_agi.Arcade._download_game (NORMAL mode) is the only way to fetch them.
After this runs, OFFLINE-mode make() will succeed for the cached games.

NOTE on Arcade priority logic: env var takes precedence over constructor arg
*unless* constructor arg is non-NORMAL. Since we want NORMAL, we have to set
OPERATION_MODE=normal in os.environ BEFORE importing arc_agi.
"""
from __future__ import annotations

import os
import sys

os.environ["OPERATION_MODE"] = "normal"

# Import only after env override — arc_agi.base.load_dotenv runs at import time
# but with override=False, so our os.environ takes precedence over .env.
import arc_agi  # noqa: E402

GAMES = ["vc33", "tu93", "cd82", "sb26"]


def main() -> int:
    if not os.environ.get("ARC_API_KEY"):
        print("ARC_API_KEY missing (needed for NORMAL-mode download).", file=sys.stderr)
        return 2

    arc = arc_agi.Arcade()
    print(f"operation_mode = {arc.operation_mode}")
    if arc.operation_mode != arc_agi.OperationMode.NORMAL:
        print("Refusing to run: must be NORMAL to download.", file=sys.stderr)
        return 2

    failures = []
    for gid in GAMES:
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
    sys.exit(main())
