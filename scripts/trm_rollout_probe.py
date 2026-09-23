#!/usr/bin/env python
"""Open-loop rollout fidelity probe for a TRM world model.

Replays held-out human action sequences through the model open-loop
(predictions fed back as inputs) and reports per-cell accuracy at several
horizons against the copy-last-frame baseline - the same probe the
DreamerV3 study used, where the RSSM never beat copy at horizon 8.

Usage:
    python scripts/trm_rollout_probe.py --data data/trm_cache --game vc33 \
        --ckpt checkpoints/trm_wm/vc33/best.pt [--horizons 2 4 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def val_windows(cache, horizon: int, max_windows: int, rng) -> list[int]:
    """Start indices t of val transitions with horizon consecutive steps."""
    valid = set(cache.transition_idx.tolist())
    starts = [
        t for t in cache.transition_idx
        if all((t + k) in valid for k in range(horizon))
    ]
    if len(starts) > max_windows:
        starts = list(rng.choice(starts, size=max_windows, replace=False))
    return starts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--predict-steps", type=int, default=None,
                        help="supervision steps per rollout step (default: halt cap)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    from arc3_wm.trm.data import _GameCache, train_val_split_episodes
    from arc3_wm.trm.training import load_wm, resolve_device

    npz = args.data / f"{args.game}.npz"
    _, va = train_val_split_episodes(npz, args.val_fraction, args.seed)
    cache = _GameCache(npz, episodes=va)
    device = resolve_device(args.device)
    model = load_wm(args.ckpt, device=device)
    rng = np.random.default_rng(args.seed)

    report = {"game": args.game, "ckpt": str(args.ckpt), "horizons": {}}
    for horizon in args.horizons:
        starts = val_windows(cache, horizon, args.max_windows, rng)
        if not starts:
            report["horizons"][horizon] = None
            continue
        grids0 = torch.from_numpy(
            np.stack([cache.grids[t] for t in starts]).astype(np.int64)
        ).to(device)
        actions = torch.from_numpy(
            np.stack(
                [[int(cache.actions[t + 1 + k]) for k in range(horizon)] for t in starts]
            )
        ).to(device)
        with torch.no_grad():
            frames = model.rollout(grids0, actions, max_steps=args.predict_steps)
        finals = frames[:, -1].cpu().numpy()
        targets = np.stack([cache.grids[t + horizon] for t in starts])
        model_acc = float((finals == targets).mean())
        copy_acc = float(
            (np.stack([cache.grids[t] for t in starts]) == targets).mean()
        )
        report["horizons"][horizon] = {
            "windows": len(starts),
            "model_cell_acc": round(model_acc, 4),
            "copy_cell_acc": round(copy_acc, 4),
            "beats_copy": model_acc > copy_acc,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
