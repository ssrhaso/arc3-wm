"""Consolidate the probe artifacts into one paper-ready results CSV.

Merges the latent-probe results (scored/probe_fits.json) with the rollout +
counterfactual scores (scored/competence_table.csv) and per-game RHAE into a
single per-game row. Pure CSV/JSON, no deps.

    python scripts/probe_make_summary.py \\
        --scored-dir results/dynamics_probe/scored \\
        --out results/dynamics_probe/scored/probe_results_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# RHAE @500k, warm seed-0 (the checkpoints probed). From the paper's Table.
RHAE = {"vc33": 0.0548, "sb26": 0.0, "cd82": 0.0, "tn36": 0.0, "ls20": 0.0, "lf52": 0.0}

COLUMNS = [
    "game", "rhae",
    # --- latent probe (competence): balanced acc vs permutation control ---
    "levelid_bacc", "levelid_ctrl", "levelid_chance", "levelid_above",
    "levelid_bacc_deter", "levelid_bacc_stoch",
    "transition_bacc", "transition_ctrl", "transition_above",
    # --- rollout fidelity (human source): model vs copy at horizon H ---
    "rollout_model_accH", "rollout_copy_accH", "rollout_deltaH",
    # --- counterfactual (random source): action sensitivity ---
    "cf_sensitivity", "cf_beats_copy_frac",
]


def _r(x, n=4):
    return round(float(x), n) if x is not None else ""


def build(scored: Path) -> list[dict]:
    fits = json.loads((scored / "probe_fits.json").read_text())
    comp = {}
    with open(scored / "competence_table.csv") as fh:
        for row in csv.DictReader(fh):
            comp[(row["game"], row["source"])] = row

    games = sorted({k.split("/")[0] for k in fits})
    rows = []
    for g in games:
        pr = fits.get(f"{g}/human", {}).get("probes", {})

        def get(key, field="balanced_acc"):
            d = pr.get(key, {})
            return d.get(field) if d.get("status") == "ok" else None

        roll = comp.get((g, "human"), {})
        cf = comp.get((g, "random"), {})
        rows.append({
            "game": g,
            "rhae": _r(RHAE.get(g)),
            "levelid_bacc": _r(get("level_id/both")),
            "levelid_ctrl": _r(get("level_id/both", "perm_balanced_acc_mean")),
            "levelid_chance": _r(get("level_id/both", "chance")),
            "levelid_above": get("level_id/both", "above_control"),
            "levelid_bacc_deter": _r(get("level_id/deter")),
            "levelid_bacc_stoch": _r(get("level_id/stoch")),
            "transition_bacc": _r(get("transition/both")),
            "transition_ctrl": _r(get("transition/both", "perm_balanced_acc_mean")),
            "transition_above": get("transition/both", "above_control"),
            "rollout_model_accH": _r(roll.get("model_acc_H")) if roll.get("model_acc_H") else "",
            "rollout_copy_accH": _r(roll.get("copy_acc_H")) if roll.get("copy_acc_H") else "",
            "rollout_deltaH": _r(roll.get("delta_H")) if roll.get("delta_H") else "",
            "cf_sensitivity": _r(cf.get("sensitivity")) if cf.get("sensitivity") else "",
            "cf_beats_copy_frac": _r(cf.get("beats_copy_frac")) if cf.get("beats_copy_frac") else "",
        })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored-dir", default="results/dynamics_probe/scored")
    p.add_argument("--out", default="results/dynamics_probe/scored/probe_results_summary.csv")
    args = p.parse_args(argv)
    rows = build(Path(args.scored_dir))
    # sort by competence (level-id decodability) descending - the headline order
    rows.sort(key=lambda r: (r["levelid_bacc"] or 0), reverse=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] wrote {args.out}\n")
    # pretty-print to stdout
    wch = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLUMNS}
    print("  ".join(c.ljust(wch[c]) for c in COLUMNS))
    for r in rows:
        print("  ".join(str(r[c]).ljust(wch[c]) for c in COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
