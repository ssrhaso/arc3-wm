"""Stage 3 of the dynamics-competence probe: score predictions -> tables.

CPU-only, JAX-free. Reads the predicted-frame npz files written by the JAX
``predict`` stage (or the synthetic generator) and emits the competence
artifacts: a per-(game, source) table, a per-horizon fidelity curve (for the
paper figure), and a JSON dump. See ``arc3_wm.dynamics_probe`` for the metrics.

Expected inputs in ``--pred-dir`` (any subset):
    <game>_<source>_rollout.npz   (Probe B - multi-step fidelity)
    <game>_<source>_cf.npz        (Probe A - one-step action-sensitivity)

Outputs in ``--outdir``:
    competence_table.csv   one row per (game, source)
    horizon_curve.csv      (game, source, h, model_acc, copy_acc, changed, mse)
    probe_summary.json     everything, machine-readable

Usage::

    python scripts/probe_score.py --pred-dir results/dynamics_probe/pred \\
        --outdir results/dynamics_probe/scored
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.dynamics_probe import score_counterfactual, score_rollout  # noqa: E402


def _scalar(npz, key, default=None):
    return npz[key].item() if key in npz else default


def score_rollout_file(path: Path) -> dict:
    d = np.load(path)
    r = score_rollout(d["rb_pred"], d["rb_true"], d["rb_context"])
    return {
        "game": str(d["game"]), "source": str(d["source"]),
        "horizon": int(_scalar(d, "horizon", r["model_acc"].shape[0])),
        "n_windows": r["n"],
        "model_acc": r["model_acc"].tolist(),
        "copy_acc": r["copy_acc"].tolist(),
        "changed": r["changed"].tolist(),
        "mse": r["mse"].tolist(),
    }


def score_cf_file(path: Path) -> dict:
    d = np.load(path)
    n = int(d["cf_pred"].shape[0])
    out = {"game": str(d["game"]), "source": str(d["source"]), "n_specs": n}
    if n == 0:
        out.update(sensitivity=None, taken_acc=None, copy_acc=None,
                   beats_copy_frac=None, rank0_frac=None)
        return out
    s = score_counterfactual(
        d["cf_pred"], d["cf_true_next"], d["cf_taken_idx"], d["cf_context"])
    out.update(s)
    return out


def build_tables(pred_dir: Path) -> tuple[list[dict], list[dict], dict]:
    rollouts = {}
    for p in sorted(pred_dir.glob("*_rollout.npz")):
        r = score_rollout_file(p)
        rollouts[(r["game"], r["source"])] = r
    cfs = {}
    for p in sorted(pred_dir.glob("*_cf.npz")):
        c = score_cf_file(p)
        cfs[(c["game"], c["source"])] = c

    keys = sorted(set(rollouts) | set(cfs))
    table_rows, curve_rows = [], []
    for (game, source) in keys:
        r = rollouts.get((game, source))
        c = cfs.get((game, source))
        row = {"game": game, "source": source}
        if r:
            ma, ca = r["model_acc"], r["copy_acc"]
            row.update({
                "n_windows": r["n_windows"], "horizon": r["horizon"],
                "model_acc_1": round(ma[0], 4), "copy_acc_1": round(ca[0], 4),
                "delta_1": round(ma[0] - ca[0], 4),
                "model_acc_H": round(ma[-1], 4), "copy_acc_H": round(ca[-1], 4),
                "delta_H": round(ma[-1] - ca[-1], 4),
                "changed_H": round(r["changed"][-1], 4),
            })
            for h in range(len(ma)):
                curve_rows.append({
                    "game": game, "source": source, "h": h + 1,
                    "model_acc": round(ma[h], 6), "copy_acc": round(ca[h], 6),
                    "changed": round(r["changed"][h], 6), "mse": round(r["mse"][h], 4),
                })
        if c:
            row.update({
                "cf_n": c["n_specs"],
                "sensitivity": None if c["sensitivity"] is None else round(c["sensitivity"], 4),
                "cf_taken_acc": None if c["taken_acc"] is None else round(c["taken_acc"], 4),
                "cf_copy_acc": None if c["copy_acc"] is None else round(c["copy_acc"], 4),
                "beats_copy_frac": None if c["beats_copy_frac"] is None else round(c["beats_copy_frac"], 4),
                "rank0_frac": None if c["rank0_frac"] is None else round(c["rank0_frac"], 4),
            })
        table_rows.append(row)

    summary = {
        "rollout": {f"{g}/{s}": r for (g, s), r in rollouts.items()},
        "counterfactual": {f"{g}/{s}": c for (g, s), c in cfs.items()},
    }
    return table_rows, curve_rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list({k: None for row in rows for k in row})  # union, order-preserving
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", default="results/dynamics_probe/pred")
    p.add_argument("--outdir", default="results/dynamics_probe/scored")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = Path(args.pred_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_rows, curve_rows, summary = build_tables(pred_dir)
    if not table_rows:
        print(f"[probe_score] no *_rollout.npz / *_cf.npz found in {pred_dir}")
        return 1
    _write_csv(outdir / "competence_table.csv", table_rows)
    _write_csv(outdir / "horizon_curve.csv", curve_rows)
    (outdir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[probe_score] wrote {len(table_rows)} rows -> {outdir/'competence_table.csv'}")
    for row in table_rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
