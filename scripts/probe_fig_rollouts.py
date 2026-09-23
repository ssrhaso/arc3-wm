"""Qualitative imagined-rollout strip for the paper (CPU, matplotlib).

Reads ONLY the stage-2 rollout predictions already on disk - never re-runs a
forward pass - so the figure can be iterated freely:

    pred/{game}_human_rollout.npz  (probe_predict) -> fig_rollouts_{game}.png

Row 1 is ground truth x_{t+1..t+8}; row 2 is the RSSM's open-loop decode under
the identical real action sequence; the copy-last-frame image x_t sits at the
left. Both rows are quantised to the 16-colour palette before display, which is
what ``probe_score`` scores on, so the picture and the numbers agree.

Window choice is the *median* of ground-truth churn, not the maximum. The
busiest window flatters the model: it tracks enough there to blur the point,
whereas a representative window shows the imagined board locking onto a wrong
configuration at h=1 and holding it while the truth moves.

Usage::

    python scripts/probe_fig_rollouts.py --game cd82 \\
        --pred-dir results/dynamics_probe/pred \\
        --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3_wm.dynamics_probe import quantize_to_palette  # noqa: E402
from arc3_wm.palette import decode_frame  # noqa: E402


def select_window(q_true: np.ndarray, q_ctx: np.ndarray, mode: str = "median") -> int:
    """Index of the window whose ground truth churns a representative amount.

    Churn is the number of cells that differ from the preceding frame, summed
    over the horizon, with the context frame standing in as the predecessor of
    the first future frame. ``median`` picks the middle of that distribution;
    ``max`` reproduces the busiest-window default of ``trm_rollout_film``.
    """
    prev = np.concatenate([q_ctx[:, None], q_true[:, :-1]], axis=1)
    churn = (q_true != prev).sum(axis=(1, 2, 3))
    if mode == "max":
        return int(churn.argmax())
    if mode != "median":
        raise ValueError(f"unknown window mode: {mode!r}")
    return int(np.argsort(churn)[len(churn) // 2])


def build_figure(q_true, q_pred, q_ctx, *, horizon: int, scale: float = 0.72):
    """Two labelled rows of ``horizon`` frames with the copy baseline at left.

    Column 0 carries the row labels and holds no image; column 1 is the
    copy-last-frame baseline, shown once in the upper row; the remaining
    ``horizon`` columns pair ground truth above the open-loop decode.
    """
    ncols = horizon + 2
    title_in = 0.30
    label_w = 0.62  # width ratio of the label column, in cell units
    height = scale * 2 + title_in
    fig, axes = plt.subplots(
        2, ncols, figsize=(scale * (horizon + 1) + scale * label_w, height),
        gridspec_kw={"wspace": 0.06, "hspace": 0.06,
                     "width_ratios": [label_w] + [1.0] * (ncols - 1)},
    )
    # Near-full bleed: default margins would leave the square images floating
    # in slack and pull the two rows apart.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0,
                        top=1.0 - title_in / height)

    for r in range(2):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.75")

    for r, text in enumerate(("ground truth", "RSSM open-loop")):
        axes[r, 0].axis("off")
        axes[r, 0].text(0.94, 0.5, text, fontsize=6.5, ha="right", va="center",
                        transform=axes[r, 0].transAxes)

    axes[0, 1].imshow(decode_frame(q_ctx), interpolation="nearest")
    axes[0, 1].set_title("copy $x_t$", fontsize=6.5, pad=2)
    axes[1, 1].axis("off")

    for k in range(horizon):
        axes[0, k + 2].imshow(decode_frame(q_true[k]), interpolation="nearest")
        axes[1, k + 2].imshow(decode_frame(q_pred[k]), interpolation="nearest")
        axes[0, k + 2].set_title(f"$h{{=}}{k + 1}$", fontsize=6.5, pad=2)
    return fig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="cd82")
    p.add_argument("--pred-dir", default="results/dynamics_probe/pred")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--window", type=int, default=None,
                   help="explicit window index; default: median ground-truth churn")
    p.add_argument("--window-mode", choices=("median", "max"), default="median")
    p.add_argument("--dpi", type=int, default=400)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.pred_dir) / f"{args.game}_human_rollout.npz"
    if not src.exists():
        print(f"[probe_fig_rollouts] missing {src}", file=sys.stderr)
        return 1
    z = np.load(src, allow_pickle=True)
    horizon = int(z["horizon"])
    q_true = quantize_to_palette(z["rb_true"])
    q_pred = quantize_to_palette(z["rb_pred"])
    q_ctx = quantize_to_palette(z["rb_context"])

    i = args.window if args.window is not None else select_window(
        q_true, q_ctx, args.window_mode)
    ep, start = int(z["rb_ep_id"][i]), int(z["rb_start"][i])

    fig = build_figure(q_true[i], q_pred[i], q_ctx[i], horizon=horizon)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"fig_rollouts_{args.game}.png"
    fig.savefig(dest, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    acc = [float((q_pred[i, k] == q_true[i, k]).mean()) for k in range(horizon)]
    copy = [float((q_ctx[i] == q_true[i, k]).mean()) for k in range(horizon)]
    print(f"[probe_fig_rollouts] {args.game} window={i} ep_id={ep} start={start}")
    print("  model acc: " + " ".join(f"{a:.3f}" for a in acc))
    print("  copy  acc: " + " ".join(f"{a:.3f}" for a in copy))
    print(f"[probe_fig_rollouts] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
