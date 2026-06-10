#!/usr/bin/env python3
"""Read-only audit extractor for the ls20 / lf52 Phase-4 runs (8 runs:
{ls20, lf52} x {warm, cold} x {seed 0, seed 1}).

Purpose: characterise *why* the ls20/lf52 policies are stuck, from logged
evidence alone, to harden the random-policy / controller-bottleneck diagnosis
before deciding whether a masked re-run is needed. NO training, NO writes to B2.

Data source (W&B project arc3-wm-sprint is NOT reachable under the available
credentials -- the given entity has 0 projects -- so this falls back to B2):

  Bucket: arc-agi-3-replays-hasaan
  warm: phase4-proper/p4-{game}-s{seed}-warm-98de390/{metrics,eval_episodes}.jsonl
  cold: phase4-fromscratch/p4-fromscratch-{game}-s{seed}-a06c02f/{metrics,eval_episodes}.jsonl

This script reads the already-downloaded copies in
``scratch/ls20_lf52_audit/`` (24 files: metrics + eval + launch.log per run).
Re-download with ``b2 file download`` if the scratch dir is absent.

Run:  python analysis/ls20_lf52_audit.py
"""
from __future__ import annotations
import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "scratch" / "ls20_lf52_audit"
LN4102 = math.log(4102)
ACTION_SPACE = 4102
VALID = {"ls20": 4}  # ls20 valid set size (4 directional). lf52 NOT in logs.

RUNS = [(arm, g, s)
        for arm in ("warm", "cold")
        for g in ("ls20", "lf52")
        for s in (0, 1)]


def series(path: Path, key: str):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if key in d and d[key] is not None:
                out.append((d.get("step"), d[key]))
    return out


def vals(path, key):
    return [v for _, v in series(path, key)]


def eval_episodes(path: Path):
    eps = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eps.append(json.loads(line)["rewards"])
    return eps


def main():
    if not SCRATCH.exists():
        raise SystemExit(f"scratch dir absent: {SCRATCH} -- re-download from B2 first.")
    for arm, g, s in RUNS:
        rid = f"{arm}_{g}_s{s}"
        m = SCRATCH / f"{rid}_metrics.jsonl"
        e = SCRATCH / f"{rid}_eval.jsonl"
        rand = vals(m, "train/rand/action")
        ent = vals(m, "train/ent/action")
        esc = vals(m, "episode/score")
        rew = vals(m, "train/rew")
        val = vals(m, "train/val")
        L = vals(m, "episode/length")
        img = vals(m, "train/loss/image")
        dyn = vals(m, "train/loss/dyn")
        eps = eval_episodes(e)
        lvl = [sum(r) for r in eps]
        acts = [len(r) - 1 for r in eps]
        print(f"== {rid} ==")
        print(f"  rand/action  : {min(rand):.4f}..{max(rand):.4f}  (1.0 = fully uniform)")
        print(f"  ent/action   : {min(ent):.5f}..{max(ent):.5f}  ln(4102)={LN4102:.5f}  gap_last={LN4102-ent[-1]:+.1e}")
        print(f"  episode/score: nonzero {sum(1 for v in esc if v)}/{len(esc)}  max={max(esc):.3f}")
        print(f"  train/rew    : {min(rew):+.2e}..{max(rew):+.2e}   train/val: {min(val):+.2e}..{max(val):+.2e}")
        print(f"  episode/len  : mean {st.mean(L):.1f} med {st.median(L):.0f} min {min(L)} max {max(L)}")
        print(f"  loss/image   : {img[0]:.1f}->{img[-1]:.2f}   loss/dyn: {dyn[0]:.1f}->{dyn[-1]:.2f}")
        print(f"  eval         : {len(eps)} eps  levels_sum={sum(lvl):.0f} any>0={any(l>0 for l in lvl)}"
              f"  ai_actions mean/min/max={st.mean(acts):.0f}/{min(acts)}/{max(acts)}")
        if g in VALID:
            v = VALID[g]
            print(f"  dilution(analytic, GIVEN rand=1.0): valid {v}/{ACTION_SPACE} "
                  f"= {v/ACTION_SPACE*100:.4f}%  dead:valid = {(ACTION_SPACE-v)/v:.1f}:1")
        else:
            print(f"  dilution: lf52 valid-set size NOT in logs -- cannot quantify")


if __name__ == "__main__":
    main()
