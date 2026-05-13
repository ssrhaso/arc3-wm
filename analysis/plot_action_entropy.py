"""Plot train/ent/action over the vc33 dry-run (qeohyn7i / commit 7d0d17a).

Decisive on the "is the model acting?" question. Entropy near log(4096) means
near-uniform clicks; entropy near 0 means collapsed to one cell.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rows = []
    with Path("scratch/p4-vc33-dryrun/metrics.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            v = d.get("train/ent/action")
            if v is not None:
                rows.append((int(d["step"]), float(v)))
    rows.sort()
    xs = np.array([s for s, _ in rows])
    ys = np.array([v for _, v in rows])

    ent_max = math.log(4096)
    eff = np.exp(ys)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: entropy in nats with the theoretical max line
    ax_l.plot(xs / 1000, ys, "o-", color="#1f77b4", markersize=4)
    ax_l.axhline(ent_max, color="black", linewidth=0.7, alpha=0.5, label=f"log(4096) = {ent_max:.3f}")
    ax_l.axvspan(232, 246, color="gold", alpha=0.18, label="burst window")
    ax_l.set_xlabel("env-step / 1k")
    ax_l.set_ylabel("train/ent/action  (nats)")
    ax_l.set_title("Actor policy entropy across vc33 dry-run\n(near log(4096) = clicking near-uniformly)")
    ax_l.legend(loc="upper right", fontsize=8)
    ax_l.grid(alpha=0.3)

    # Right: effective action count = exp(entropy)
    ax_r.semilogy(xs / 1000, eff, "o-", color="#d62728", markersize=4)
    ax_r.axhline(4096, color="black", linewidth=0.7, alpha=0.5, label="4096 (uniform)")
    ax_r.axhline(150, color="gray", linewidth=0.5, linestyle="--", alpha=0.5, label="end-of-training ~150 cells")
    ax_r.axvspan(232, 246, color="gold", alpha=0.18)
    ax_r.set_xlabel("env-step / 1k")
    ax_r.set_ylabel("effective action count = exp(entropy)  (log)")
    ax_r.set_title("Effective # of cells the actor samples from")
    ax_r.legend(loc="lower left", fontsize=8)
    ax_r.grid(alpha=0.3, which="both")

    fig.suptitle("vc33 dry-run actor entropy — is the model acting?")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = Path("figures/p4_vc33_action_entropy.png")
    out_svg = Path("figures/p4_vc33_action_entropy.svg")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote {out_png}, {out_svg}")


if __name__ == "__main__":
    main()
