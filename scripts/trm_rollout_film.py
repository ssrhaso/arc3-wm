#!/usr/bin/env python
"""Qualitative rollout filmstrips: what the world model actually imagines.

For a held-out human action sequence, renders ground-truth frames (top
row) against the model's open-loop imagination (bottom row, predictions
fed back) at horizon H - the qualitative companion to the rollout probe,
answering the draft's request for imagined-rollout images.

Usage:
    python scripts/trm_rollout_film.py --data <cache> --game vc33 \
        --ckpt <wm>/best.pt --out figures/film_vc33.png [--horizon 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def render_strip(frames_true, frames_pred, scale: int = 3) -> np.ndarray:
    """Two rows of palette grids -> one RGB image array (uint8)."""
    from arc3_wm.palette import decode_frame

    pad = 2
    h = 64 * scale
    cols = len(frames_true)
    width = cols * (h + pad) - pad
    height = 2 * (h + pad) - pad
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for row, frames in enumerate((frames_true, frames_pred)):
        for col, grid in enumerate(frames):
            rgb = decode_frame(np.asarray(grid, dtype=np.int16))
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
            y0 = row * (h + pad)
            x0 = col * (h + pad)
            canvas[y0 : y0 + h, x0 : x0 + h] = rgb
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=None,
                        help="explicit start index; default: the val window "
                             "with the most ground-truth change")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    from arc3_wm.trm.data import _GameCache, train_val_split_episodes
    from arc3_wm.trm.training import load_wm, resolve_device
    from trm_rollout_probe import val_windows

    npz = args.data / f"{args.game}.npz"
    _, va = train_val_split_episodes(npz, args.val_fraction, args.seed)
    cache = _GameCache(npz, episodes=va)
    device = resolve_device(args.device)
    model = load_wm(args.ckpt, device=device)
    rng = np.random.default_rng(args.seed)

    starts = val_windows(cache, args.horizon, 512, rng)
    if not starts:
        print(f"no val window of horizon {args.horizon}", file=sys.stderr)
        return 1
    if args.window is not None:
        t0 = args.window
    else:
        # Pick the visually busiest window (most cells changing).
        def churn(t):
            return int(
                sum((cache.grids[t + k] != cache.grids[t + k + 1]).sum()
                    for k in range(args.horizon))
            )
        t0 = max(starts, key=churn)

    actions = torch.tensor(
        [[int(cache.actions[t0 + 1 + k]) for k in range(args.horizon)]]
    ).to(device)
    grid0 = torch.from_numpy(cache.grids[t0].astype(np.int64))[None].to(device)
    with torch.no_grad():
        pred = model.rollout(grid0, actions).cpu().numpy()[0]

    frames_true = [cache.grids[t0 + 1 + k] for k in range(args.horizon)]
    frames_pred = [pred[k] for k in range(args.horizon)]
    canvas = render_strip(frames_true, frames_pred)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(canvas).save(args.out)
    except ImportError:
        np.save(args.out.with_suffix(".npy"), canvas)
        print("Pillow unavailable; saved raw array instead", file=sys.stderr)
    acc = [float((frames_pred[k] == frames_true[k]).mean())
           for k in range(args.horizon)]
    print(f"{args.game}: window t0={t0}, per-step cell acc "
          + " ".join(f"{a:.3f}" for a in acc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
