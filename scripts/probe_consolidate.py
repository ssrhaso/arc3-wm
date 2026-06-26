"""Consolidate every probe artifact into two professional CSVs.

Reads:
  scored/probe_fits.json            - latent probe, per-game fine-tuned WM (CIs, 4 targets)
  scored_pretrained/probe_fits.json - latent probe, cross-game PRETRAINED WM (pre-control)
  scored/competence_table.csv       - rollout fidelity + counterfactual (human & random)
  + per-game RHAE (warm seed-0, from the paper Table)

Writes (into --outdir, default scored/):
  probe_master_by_game.csv  - WIDE: one row per game, every headline number
  probe_detail_long.csv     - TIDY/LONG: one row per (game, checkpoint, target, feature)

Both are pure values (no string-formatted CIs) so they're analysis-ready.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RHAE = {"vc33": 0.0548, "sb26": 0.0, "cd82": 0.0, "tn36": 0.0, "ls20": 0.0, "lf52": 0.0}


def _load_fits(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _probe(fits: dict, gk: str, key: str, field: str = "balanced_acc"):
    pr = fits.get(gk, {}).get("probes", {}).get(key, {})
    return pr.get(field) if pr.get("status") == "ok" else None


def _comp(comp: dict, game: str, source: str, field: str):
    row = comp.get((game, source), {})
    v = row.get(field, "")
    return float(v) if v not in ("", None) else None


# ---------------------------------------------------------------------------

LONG_COLS = ["game", "checkpoint", "target", "feature", "balanced_acc",
             "ci_lo", "ci_hi", "ctrl_mean", "ctrl_ci_hi", "chance",
             "above_control", "n_classes", "n_folds"]


def build_long(ft: dict, pre: dict) -> list[dict]:
    rows = []
    for ckpt_name, fits in (("finetuned", ft), ("pretrained", pre)):
        for gk in sorted(fits):
            game = fits[gk].get("game", gk.split("/")[0])
            for key, pr in fits[gk].get("probes", {}).items():
                if pr.get("status") != "ok":
                    continue
                target, feature = key.split("/")
                rows.append({
                    "game": game, "checkpoint": ckpt_name,
                    "target": target, "feature": feature,
                    "balanced_acc": pr.get("balanced_acc"),
                    "ci_lo": pr.get("ci_lo"), "ci_hi": pr.get("ci_hi"),
                    "ctrl_mean": pr.get("ctrl_mean"), "ctrl_ci_hi": pr.get("ctrl_ci_hi"),
                    "chance": pr.get("chance"), "above_control": pr.get("above_control"),
                    "n_classes": pr.get("n_classes"), "n_folds": pr.get("n_folds"),
                })
    return rows


MASTER_COLS = [
    "game", "rhae", "latent_null",
    # latent competence, fine-tuned per-game WM (level identity)
    "ft_levelid_bacc", "ft_levelid_ci_lo", "ft_levelid_ci_hi",
    "ft_levelid_ctrl_ci_hi", "ft_levelid_above",
    "ft_levelid_deter", "ft_levelid_stoch",
    # level identity, pretrained cross-game WM (pre-control)
    "pre_levelid_bacc", "pre_levelid_above",
    # transition events
    "ft_transition_bacc", "ft_transition_ci_lo", "ft_transition_ci_hi", "ft_transition_above",
    "pre_transition_bacc",
    # extra targets (competence beyond level appearance)
    "ft_rewardimm_bacc", "ft_rewardimm_above",
    "ft_timereset_bacc", "ft_timereset_above",
    # rollout fidelity (human source)
    "rollout_model_accH", "rollout_copy_accH", "rollout_deltaH",
    # counterfactual action-sensitivity
    "cf_human_sensitivity", "cf_human_beats_copy", "cf_human_rank0",
    "cf_random_sensitivity", "cf_random_beats_copy",
]


def build_master(ft: dict, pre: dict, comp: dict) -> list[dict]:
    games = sorted({gk.split("/")[0] for gk in ft})
    rows = []
    for g in games:
        gk = f"{g}/human"
        rows.append({
            "game": g, "rhae": RHAE.get(g), "latent_null": ft.get(gk, {}).get("null"),
            "ft_levelid_bacc": _probe(ft, gk, "level_id/both"),
            "ft_levelid_ci_lo": _probe(ft, gk, "level_id/both", "ci_lo"),
            "ft_levelid_ci_hi": _probe(ft, gk, "level_id/both", "ci_hi"),
            "ft_levelid_ctrl_ci_hi": _probe(ft, gk, "level_id/both", "ctrl_ci_hi"),
            "ft_levelid_above": _probe(ft, gk, "level_id/both", "above_control"),
            "ft_levelid_deter": _probe(ft, gk, "level_id/deter"),
            "ft_levelid_stoch": _probe(ft, gk, "level_id/stoch"),
            "pre_levelid_bacc": _probe(pre, gk, "level_id/both"),
            "pre_levelid_above": _probe(pre, gk, "level_id/both", "above_control"),
            "ft_transition_bacc": _probe(ft, gk, "transition/both"),
            "ft_transition_ci_lo": _probe(ft, gk, "transition/both", "ci_lo"),
            "ft_transition_ci_hi": _probe(ft, gk, "transition/both", "ci_hi"),
            "ft_transition_above": _probe(ft, gk, "transition/both", "above_control"),
            "pre_transition_bacc": _probe(pre, gk, "transition/both"),
            "ft_rewardimm_bacc": _probe(ft, gk, "reward_imminence/both"),
            "ft_rewardimm_above": _probe(ft, gk, "reward_imminence/both", "above_control"),
            "ft_timereset_bacc": _probe(ft, gk, "time_since_reset/both"),
            "ft_timereset_above": _probe(ft, gk, "time_since_reset/both", "above_control"),
            "rollout_model_accH": _comp(comp, g, "human", "model_acc_H"),
            "rollout_copy_accH": _comp(comp, g, "human", "copy_acc_H"),
            "rollout_deltaH": _comp(comp, g, "human", "delta_H"),
            "cf_human_sensitivity": _comp(comp, g, "human", "sensitivity"),
            "cf_human_beats_copy": _comp(comp, g, "human", "beats_copy_frac"),
            "cf_human_rank0": _comp(comp, g, "human", "rank0_frac"),
            "cf_random_sensitivity": _comp(comp, g, "random", "sensitivity"),
            "cf_random_beats_copy": _comp(comp, g, "random", "beats_copy_frac"),
        })
    rows.sort(key=lambda r: (r["ft_levelid_bacc"] or 0), reverse=True)
    return rows


def _write(path: Path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {path}  ({len(rows)} rows)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored-dir", default="results/dynamics_probe/scored")
    p.add_argument("--pretrained-dir", default="results/dynamics_probe/scored_pretrained")
    p.add_argument("--outdir", default="results/dynamics_probe/scored")
    args = p.parse_args(argv)
    scored = Path(args.scored_dir)
    ft = _load_fits(scored / "probe_fits.json")
    pre = _load_fits(Path(args.pretrained_dir) / "probe_fits.json")
    comp = {}
    ctab = scored / "competence_table.csv"
    if ctab.exists():
        with open(ctab) as fh:
            for row in csv.DictReader(fh):
                comp[(row["game"], row["source"])] = row
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    _write(out / "probe_master_by_game.csv", MASTER_COLS, build_master(ft, pre, comp))
    _write(out / "probe_detail_long.csv", LONG_COLS, build_long(ft, pre))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
