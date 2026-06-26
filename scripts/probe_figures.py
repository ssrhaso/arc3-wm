"""Regenerate all probe figures from saved artifacts (CPU, matplotlib).

Reads ONLY the on-disk results from the score/fit stages - never re-runs a
forward pass - so figures can be iterated freely after one cluster run:

    scored/horizon_curve.csv      (probe_score)   -> fig_horizon_fidelity.png
    scored/probe_fits.json        (probe_fit_probes) -> fig_latent_probe.png
    scored/competence_table.csv   (probe_score)   -> used for the plane
    + per-game RHAE (paper values, override with --rhae-json) -> fig_competence_performance.png

The competence x performance plane is the headline: competence (probe metric) on
x, RHAE on y. The thesis is decorrelation - e.g. cd82 high competence, RHAE 0.

Usage::

    python scripts/probe_figures.py --scored-dir results/dynamics_probe/scored \\
        --outdir results/dynamics_probe/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# Paper RHAE (warm seed 0) - override with --rhae-json {"game": value, ...}.
DEFAULT_RHAE = {"vc33": 0.0548, "sb26": 0.0, "cd82": 0.0,
                "tn36": 0.0, "ls20": 0.0, "lf52": 0.0}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fig_horizon_fidelity(scored: Path, out: Path) -> bool:
    rows = _read_csv(scored / "horizon_curve.csv")
    if not rows:
        return False
    by = defaultdict(list)
    for r in rows:
        if r.get("source") == "human":  # human = the strong-signal source
            by[r["game"]].append(r)
    if not by:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for game, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: int(r["h"]))
        h = [int(r["h"]) for r in rs]
        model = [float(r["model_acc"]) for r in rs]
        copy = [float(r["copy_acc"]) for r in rs]
        line, = ax.plot(h, model, marker="o", label=f"{game} model")
        ax.plot(h, copy, linestyle="--", color=line.get_color(), alpha=0.5)
    ax.set_xlabel("imagination horizon (steps)")
    ax.set_ylabel("cell accuracy")
    ax.set_title("Rollout fidelity vs copy-baseline (dashed) — human source")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig_horizon_fidelity.png", dpi=150)
    plt.close(fig)
    return True


def fig_latent_probe(scored: Path, out: Path, target: str = "level_id") -> bool:
    path = scored / "probe_fits.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    games, bacc, ctrl = [], [], []
    for key, r in sorted(data.items()):
        pr = r.get("probes", {}).get(f"{target}/both")
        if not pr or pr.get("status") != "ok":
            continue
        games.append(r["game"])
        bacc.append(pr["balanced_acc"])
        ctrl.append(pr["perm_balanced_acc_mean"])
    if not games:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(games))
    ax.bar([i - 0.2 for i in x], bacc, width=0.4, label="probe (balanced acc)")
    ax.bar([i + 0.2 for i in x], ctrl, width=0.4, label="permutation control", alpha=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(games)
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"Linear probe: '{target}' decodable from RSSM latents (both h+z)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig_latent_probe_{target}.png", dpi=150)
    plt.close(fig)
    return True


def fig_competence_performance(scored: Path, out: Path, rhae: dict) -> bool:
    """x = competence (latent level_id bacc if available, else rollout acc@H), y = RHAE."""
    comp = {}
    fits = scored / "probe_fits.json"
    if fits.exists():
        for r in json.loads(fits.read_text()).values():
            pr = r.get("probes", {}).get("level_id/both")
            if pr and pr.get("status") == "ok":
                comp[r["game"]] = ("latent level-id bacc", pr["balanced_acc"])
    if not comp:  # fall back to rollout fidelity at the last horizon
        for row in _read_csv(scored / "competence_table.csv"):
            if row.get("source") == "human" and row.get("model_acc_H"):
                comp[row["game"]] = ("rollout acc@H", float(row["model_acc_H"]))
    if not comp:
        return False
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xlabel = next(iter(comp.values()))[0]
    for game, (_lbl, x) in comp.items():
        y = rhae.get(game, 0.0)
        ax.scatter(x, y, s=60)
        ax.annotate(game, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.set_xlabel(f"competence ({xlabel})")
    ax.set_ylabel("performance (RHAE)")
    ax.set_title("Competence × performance — the dissociation")
    fig.tight_layout(); fig.savefig(out / "fig_competence_performance.png", dpi=150)
    plt.close(fig)
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored-dir", default="results/dynamics_probe/scored")
    p.add_argument("--outdir", default="results/dynamics_probe/figures")
    p.add_argument("--rhae-json", default=None, help="override per-game RHAE")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    scored = Path(args.scored_dir)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rhae = dict(DEFAULT_RHAE)
    if args.rhae_json:
        rhae.update(json.loads(Path(args.rhae_json).read_text()))
    made = {
        "horizon_fidelity": fig_horizon_fidelity(scored, out),
        "latent_probe(level_id)": fig_latent_probe(scored, out, "level_id"),
        "latent_probe(transition)": fig_latent_probe(scored, out, "transition"),
        "competence_performance": fig_competence_performance(scored, out, rhae),
    }
    for name, ok in made.items():
        print(f"  {'[made]' if ok else '[skip]'} {name}")
    if not any(made.values()):
        print(f"[probe_figures] no inputs found in {scored} - run score/fit stages first")
        return 1
    print(f"[probe_figures] figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
