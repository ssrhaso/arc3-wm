#!/usr/bin/env python
"""Preprocess the human-replay corpus into per-game npz tensor caches.

Usage:
    python scripts/trm_preprocess_replays.py \
        --replays-root data/replays/public_games-dataset \
        --out data/trm_cache [game_id ...]

With no game ids, processes every game directory found under the root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3_wm.trm.data import preprocess_game  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("games", nargs="*", help="game ids (default: all found)")
    args = parser.parse_args(argv)

    games = args.games or sorted(
        p.name for p in args.replays_root.iterdir() if p.is_dir()
    )
    written = 0
    for game in games:
        out = preprocess_game(args.replays_root, game, args.out)
        if out is None:
            print(f"{game}: no replays found", file=sys.stderr)
            continue
        import numpy as np

        data = np.load(out)
        print(
            f"{game}: {len(data['grids'])} frames, "
            f"{len(data['episode_starts'])} episodes -> {out}"
        )
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
