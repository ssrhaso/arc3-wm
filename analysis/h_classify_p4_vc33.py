"""H1/H2/H3 classifier for the Phase-4 vc33 dry-run success-burst collapse.

Inputs: scratch/p4-vc33-dryrun/metrics.jsonl (B2 mirror of qeohyn7i).
Outputs:
  figures/p4_vc33_failure_mode.png  (200 DPI, 2-panel)
  figures/p4_vc33_failure_mode.svg
  Prints a 3-line H1/H2/H3 verdict to stdout.

Plot window: env-step 200k-280k.
Burst window (from Q1): env-step 232,679 - 245,151 (13 of 17 train clears).
Pre-burst stats: env-step 100k-225k (wider for stable mean/std on losses
  that log every ~9k env-steps).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


METRICS = Path("scratch/p4-vc33-dryrun/metrics.jsonl")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

LOSS_KEYS = ("train/loss/dyn", "train/loss/rep", "train/loss/image")
EXPLORE_KEY = "train/rand/action"
EP_SCORE_KEY = "episode/score"
STEP_KEY = "step"

PLOT_LO, PLOT_HI = 200_000, 280_000
BURST_LO, BURST_HI = 232_000, 246_000   # widened by ~3k each side around the 13-clear cluster
PREBURST_LO, PREBURST_HI = 100_000, 225_000


def load_series(path: Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in (*LOSS_KEYS, EXPLORE_KEY, EP_SCORE_KEY)}
    with path.open() as f:
        for raw in f:
            d = json.loads(raw)
            step = d.get(STEP_KEY)
            if step is None:
                continue
            for k in series:
                if k in d and d[k] is not None:
                    series[k].append((int(step), float(d[k])))
    for k in series:
        series[k].sort()
    return series


def in_window(rows: list[tuple[int, float]], lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [(s, v) for s, v in rows if lo <= s <= hi]
    if not rows:
        return np.array([]), np.array([])
    xs, ys = zip(*rows)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def rolling_episode_rate(score_rows: list[tuple[int, float]], window_eps: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling success rate over a window of `window_eps` consecutive episodes,
    indexed by the env-step at which each episode ended. Each row in metrics.jsonl
    with episode/score corresponds to one ended episode."""
    if not score_rows:
        return np.array([]), np.array([])
    xs, ys = zip(*score_rows)
    xs = np.asarray(xs, dtype=float)
    ys = (np.asarray(ys, dtype=float) > 0).astype(float)
    n = len(ys)
    if n < window_eps:
        return xs, np.cumsum(ys) / np.arange(1, n + 1)
    csum = np.concatenate([[0.0], np.cumsum(ys)])
    rate = (csum[window_eps:] - csum[:-window_eps]) / window_eps
    # align each rolling value to the env-step of the LAST episode in the window
    rate_xs = xs[window_eps - 1 :]
    return rate_xs, rate


def classify(series: dict[str, list[tuple[int, float]]]) -> tuple[str, list[str]]:
    notes: list[str] = []

    # Cut 1: loss bumps > 2sigma above pre-burst mean during burst window
    loss_bumps: dict[str, dict[str, float]] = {}
    for k in LOSS_KEYS:
        x_pre, y_pre = in_window(series[k], PREBURST_LO, PREBURST_HI)
        x_burst, y_burst = in_window(series[k], BURST_LO, BURST_HI)
        if len(y_pre) < 3 or len(y_burst) == 0:
            notes.append(f"  - {k}: insufficient data (pre={len(y_pre)}, burst={len(y_burst)})")
            continue
        mu, sigma = float(np.mean(y_pre)), float(np.std(y_pre, ddof=1))
        peak = float(np.max(y_burst))
        z = (peak - mu) / sigma if sigma > 0 else 0.0
        bump = z > 2.0
        loss_bumps[k] = {"mu": mu, "sigma": sigma, "peak": peak, "z": z, "bump": bump}
        notes.append(f"  - {k}: pre mu={mu:.4g} sigma={sigma:.3g}; burst peak={peak:.4g} (z={z:+.2f}) -> {'BUMP' if bump else 'flat'}")

    any_bump = any(b["bump"] for b in loss_bumps.values())

    # Cut 2: env-step-binned success count across 200k-280k.
    # Why not 5k-EP rolling: at vc33 mean ep-length ~52 env-steps, 5k episodes
    # spans ~260k env-steps -> wider than the entire plot window, so any rolling
    # 5k-EP rate is dominated by window-convolution shape, not signal shape.
    # The honest signal is "successes per env-step bin".
    score_rows_in = [(s, v) for s, v in series[EP_SCORE_KEY] if PLOT_LO <= s <= PLOT_HI]
    bin_width = 5000
    bin_edges = np.arange(PLOT_LO, PLOT_HI + bin_width, bin_width)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_eps = np.zeros(len(bin_edges) - 1)
    bin_clears = np.zeros(len(bin_edges) - 1)
    for s, v in score_rows_in:
        idx = int((s - PLOT_LO) // bin_width)
        if 0 <= idx < len(bin_eps):
            bin_eps[idx] += 1
            if v > 0:
                bin_clears[idx] += 1
    bin_rate = np.where(bin_eps > 0, bin_clears / bin_eps, 0.0)

    if bin_rate.max() > 0:
        peak_idx = int(np.argmax(bin_rate))
        peak_val = float(bin_rate[peak_idx])
        peak_step = float(bin_centres[peak_idx])
        thresh = 0.10 * peak_val
        # rise: first bin with rate >= thresh, up to and including peak
        rise_idxs = np.where((bin_rate >= thresh) & (np.arange(len(bin_rate)) <= peak_idx))[0]
        rise_start = float(bin_centres[rise_idxs[0]]) if len(rise_idxs) else peak_step
        rise_time = peak_step - rise_start + bin_width  # add one bin-width so a single-bin spike has nonzero rise
        # decay: first bin after peak with rate <= thresh
        post = np.where((np.arange(len(bin_rate)) > peak_idx) & (bin_rate <= thresh))[0]
        decay_end = float(bin_centres[post[0]]) if len(post) else float(bin_centres[-1])
        decay_time = decay_end - peak_step
        notes.append(f"  - env-step-binned rate (5k-step bins): peak={peak_val:.4f} ({int(bin_clears[peak_idx])}/{int(bin_eps[peak_idx])} eps) at step={peak_step:.0f}")
        notes.append(f"  - rise (first bin >= 10% peak -> peak bin, inclusive) = {rise_time:.0f} env-steps")
        notes.append(f"  - decay (peak bin -> first bin <= 10% peak) = {decay_time:.0f} env-steps")
        bins_above_thresh = int(np.sum(bin_rate >= thresh))
        notes.append(f"  - bins with rate >= 10% peak: {bins_above_thresh} out of {len(bin_rate)}")
    else:
        peak_val = rise_time = decay_time = peak_step = 0.0
        notes.append("  - env-step-binned rate: no successes in plot window")

    # Cut 3: rand/action 5k-step buckets, does crater coincide with success collapse?
    er_x, er_y = in_window(series[EXPLORE_KEY], PLOT_LO, PLOT_HI)
    bucket = 5000
    edges = np.arange(PLOT_LO, PLOT_HI + bucket, bucket)
    rand_bucket_x: list[float] = []
    rand_bucket_y: list[float] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (er_x >= lo) & (er_x < hi)
        if m.any():
            rand_bucket_x.append((lo + hi) / 2)
            rand_bucket_y.append(float(np.mean(er_y[m])))
    rand_bucket_x_a = np.asarray(rand_bucket_x)
    rand_bucket_y_a = np.asarray(rand_bucket_y)

    # Find biggest single-bucket drop in rand/action and its location
    rand_drop_step = 0.0
    rand_drop = 0.0
    if len(rand_bucket_y_a) >= 2:
        diffs = np.diff(rand_bucket_y_a)
        worst = int(np.argmin(diffs))
        rand_drop = float(diffs[worst])  # negative for drop
        rand_drop_step = float(rand_bucket_x_a[worst + 1])
        notes.append(f"  - rand/action biggest drop: {rand_drop:+.3f} at step ~{rand_drop_step:.0f}")

    # Verdict (thresholds in env-step space on the binned signal)
    if any_bump:
        verdict = "H2"
        why = "loss bump >2sigma during burst window -> WM/critic contamination during the policy's brief latch-on."
    else:
        # "spike" = peak is single-binned (one 5k bin) AND rate concentrates in that bin
        single_bin_peak = (rise_time <= bin_width * 1.5) and (decay_time <= bin_width * 1.5)
        ramp_shape = rise_time >= 20000 and decay_time >= 5000
        if single_bin_peak:
            verdict = "H2"
            why = (
                f"no loss bump AND the success rate is concentrated in a single 5k-step bin "
                f"(rise={rise_time:.0f}, decay={decay_time:.0f}) -> spike: brief, unstable critic latch."
            )
        elif ramp_shape:
            verdict = "H1"
            why = "no loss bump and rate-shape is a ramp (>=20k-step rise + >=5k-step decay) -> classic catastrophic forgetting from entropy decay."
        else:
            # plateau or ambiguous: invoke cut 3
            collapse_step = peak_step + decay_time
            crater_coincides = (
                abs(rand_drop_step - collapse_step) < 7500 and rand_drop < -0.05
            )
            if crater_coincides:
                verdict = "H3"
                why = (
                    f"no loss bump; rate plateau collapsed at ~{collapse_step:.0f}, "
                    f"rand/action cratered {rand_drop:+.3f} at ~{rand_drop_step:.0f} "
                    f"(within 7.5k env-steps) -> entropy schedule killed exploration before policy was robust."
                )
            else:
                verdict = "AMBIGUOUS"
                why = (
                    f"no loss bump, rate shape neither spike nor ramp (rise={rise_time:.0f}, decay={decay_time:.0f}); "
                    f"rand/action drop ({rand_drop:+.3f} at {rand_drop_step:.0f}) does not align with collapse."
                )

    return verdict, notes + [f"VERDICT: {verdict} - {why}"]


def main() -> int:
    if not METRICS.exists():
        print(f"missing {METRICS}", file=sys.stderr)
        return 2
    series = load_series(METRICS)

    # Pre-burst means for loss normalization
    pre_means = {}
    for k in LOSS_KEYS:
        _, y = in_window(series[k], PREBURST_LO, PREBURST_HI)
        pre_means[k] = float(np.mean(y)) if len(y) else 1.0

    # ---- Figure
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: normalized losses
    colors = {"train/loss/dyn": "#1f77b4", "train/loss/rep": "#2ca02c", "train/loss/image": "#d62728"}
    for k in LOSS_KEYS:
        x, y = in_window(series[k], PLOT_LO, PLOT_HI)
        if len(x) == 0:
            continue
        ynorm = y / pre_means[k]
        ax_l.plot(x / 1000, ynorm, "o-", color=colors[k], label=k.split("/")[-1] + f" (preburst mu={pre_means[k]:.3g})", markersize=5)
    ax_l.axvspan(BURST_LO / 1000, BURST_HI / 1000, color="gold", alpha=0.18, label="burst window")
    ax_l.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax_l.set_xlabel("env-step / 1k")
    ax_l.set_ylabel("loss / pre-burst mean")
    ax_l.set_title("WM losses, 200k-280k\n(normalized to pre-burst mean; horizontal=1.0=baseline)")
    ax_l.legend(loc="upper left", fontsize=8)
    ax_l.grid(alpha=0.3)

    # Right panel: env-step-binned rate (bars, honest signal) + 1k-EP rolling
    # (line, the agent's earlier resolution) + 5k-EP rolling (faint, what the
    # user requested; included as evidence that 5k-EP is window-dominated)
    # on left axis; rand/action on right axis.
    score_rows_in = [(s, v) for s, v in series[EP_SCORE_KEY] if PLOT_LO <= s <= PLOT_HI]
    bin_width = 5000
    bin_edges = np.arange(PLOT_LO, PLOT_HI + bin_width, bin_width)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_eps = np.zeros(len(bin_edges) - 1)
    bin_clears = np.zeros(len(bin_edges) - 1)
    for s, v in score_rows_in:
        idx = int((s - PLOT_LO) // bin_width)
        if 0 <= idx < len(bin_eps):
            bin_eps[idx] += 1
            if v > 0:
                bin_clears[idx] += 1
    bin_rate_plot = np.where(bin_eps > 0, bin_clears / bin_eps, 0.0)
    ax_r.bar(bin_centres / 1000, bin_rate_plot, width=(bin_width / 1000) * 0.9,
             color="#9467bd", alpha=0.55, label="success rate, 5k-step bins (honest)")

    for window, color, alpha, lw in [(1000, "#1f77b4", 0.95, 1.3), (5000, "#7fbf7f", 0.55, 1.0)]:
        rx, ry = rolling_episode_rate(series[EP_SCORE_KEY], window_eps=window)
        mask = (rx >= PLOT_LO) & (rx <= PLOT_HI)
        ax_r.plot(rx[mask] / 1000, ry[mask], color=color, linewidth=lw, alpha=alpha,
                  label=f"rolling-{window // 1000}k-ep rate")

    ax_r.axvspan(BURST_LO / 1000, BURST_HI / 1000, color="gold", alpha=0.18)
    ax_r.set_xlabel("env-step / 1k")
    ax_r.set_ylabel("success rate", color="#444444")
    ax_r.tick_params(axis="y", labelcolor="#444444")
    ax_r.grid(alpha=0.3)
    ax_r.set_title("Success rate (3 views) + exploration, 200k-280k")

    ax_r2 = ax_r.twinx()
    er_x, er_y = in_window(series[EXPLORE_KEY], PLOT_LO, PLOT_HI)
    ax_r2.plot(er_x / 1000, er_y, color="#ff7f0e", linewidth=1.2, alpha=0.85, label="train/rand/action")
    ax_r2.set_ylabel("train/rand/action", color="#ff7f0e")
    ax_r2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_r2.set_ylim(0, 1.02)

    # combined legend
    lines1, labels1 = ax_r.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle("Phase-4 vc33 dry-run: failure-mode classification (env-step 200k-280k)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = FIG_DIR / "p4_vc33_failure_mode.png"
    svg = FIG_DIR / "p4_vc33_failure_mode.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)

    verdict, notes = classify(series)
    print("=" * 70)
    print("FAILURE-MODE CLASSIFIER: Phase-4 vc33 dry-run")
    print("=" * 70)
    for n in notes:
        print(n)
    print("=" * 70)
    print(f"FIGURE: {png}, {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
