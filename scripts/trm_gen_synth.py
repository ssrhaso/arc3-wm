#!/usr/bin/env python
"""Generate the synthetic-game corpus (SDG) as npz caches.

Usage:
    python scripts/trm_gen_synth.py --out data/synth_cache \
        [--games 200] [--episodes 4] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3_wm.trm.synthetic import generate_corpus  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = generate_corpus(args.out, args.games, args.episodes, args.seed)
    import numpy as np

    total = sum(len(np.load(p)["grids"]) for p in paths)
    print(f"wrote {len(paths)} synthetic games, {total} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
