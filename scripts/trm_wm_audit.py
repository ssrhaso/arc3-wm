#!/usr/bin/env python
"""Overfitting audit for a trained TRM world model.

Recomputes held-out val metrics twice: on the full val split, and restricted
to *novel* transitions whose (state, action) input never appears in the
training split. A large gap between the two indicates memorisation; parity
indicates within-game generalisation. Also reports the train/val overlap
rate and the best-epoch position from metrics.jsonl (an early best epoch
plus late val decline is the overfit signature).

Usage:
    python scripts/trm_wm_audit.py --data data/trm_cache --game vc33 \
        --ckpt checkpoints/trm_wm/vc33/best.pt [--seed 0] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def novel_val_indices(cache_train, cache_val) -> tuple[np.ndarray, float]:
    """Val transition indices whose (grid, action) input is not in train."""

    def input_key(cache, t) -> bytes:
        return cache.grids[t].tobytes() + int(cache.actions[t + 1]).to_bytes(
            2, "little", signed=True
        )

    train_inputs = {input_key(cache_train, t) for t in cache_train.transition_idx}
    novel = [
        t for t in cache_val.transition_idx
        if input_key(cache_val, t) not in train_inputs
    ]
    overlap = 1.0 - len(novel) / max(len(cache_val.transition_idx), 1)
    return np.asarray(novel, dtype=np.int64), overlap


def best_epoch_report(metrics_path: Path) -> dict:
    """Val-curve shape: best epoch, final epoch, and late-decline flag."""
    if not metrics_path.exists():
        return {}
    vals = []
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if "val" in rec and "exact_match" in rec["val"]:
            vals.append((rec["epoch"], rec["val"]["exact_match"]))
    if not vals:
        return {}
    best_epoch, best = max(vals, key=lambda x: x[1])
    final_epoch, final = vals[-1]
    return {
        "val_points": len(vals),
        "best_epoch": best_epoch,
        "best_exact": best,
        "final_epoch": final_epoch,
        "final_exact": final,
        "late_decline": bool(final < 0.9 * best and final_epoch > best_epoch),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    import torch

    from arc3_wm.trm.data import _GameCache, train_val_split_episodes
    from arc3_wm.trm.training import evaluate_wm, load_wm, resolve_device

    npz = args.data / f"{args.game}.npz"
    tr, va = train_val_split_episodes(npz, args.val_fraction, args.seed)
    cache_train = _GameCache(npz, episodes=tr)
    cache_val = _GameCache(npz, episodes=va)
    novel_idx, overlap = novel_val_indices(cache_train, cache_val)

    device = resolve_device(args.device)
    model = load_wm(args.ckpt, device=device)

    class _Subset:
        """WMTransitionDataset-shaped view over explicit cache indices."""

        def __init__(self, cache, idx):
            self._cache = cache
            self._idx = idx

        def __len__(self):
            return len(self._idx)

        def __getitem__(self, i):
            c, t = self._cache, int(self._idx[i])
            reward = float(c.levels[t + 1] - c.levels[t] > 0)
            return {
                "grid": torch.from_numpy(c.grids[t].astype(np.int64)),
                "action": torch.tensor(int(c.actions[t + 1]), dtype=torch.long),
                "next_grid": torch.from_numpy(c.grids[t + 1].astype(np.int64)),
                "reward": torch.tensor(reward),
                "state": torch.tensor(int(c.states[t + 1]), dtype=torch.long),
            }

    full = evaluate_wm(
        model, _Subset(cache_val, cache_val.transition_idx),
        batch_size=args.batch_size, max_batches=10**6,
    )
    novel = evaluate_wm(
        model, _Subset(cache_val, novel_idx),
        batch_size=args.batch_size, max_batches=10**6,
    )
    report = {
        "game": args.game,
        "ckpt": str(args.ckpt),
        "input_overlap_train_val": round(overlap, 4),
        "val_full": full,
        "val_novel_only": novel,
        "curve": best_epoch_report(args.ckpt.parent / "metrics.jsonl"),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
