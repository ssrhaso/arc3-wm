#!/usr/bin/env python3
"""Read-only W&B probe for the ls20/lf52 gap-closing atoms (2026-06-09).

Companion to ``analysis/ls20_lf52_audit.py`` (which reads the B2 snapshot).
This script locates the sweep in W&B, inventories one ls20 run, and analyses
the board-render GIFs that B2 lacked, to test the three atoms the B2 audit
could not close: (a) action distribution, (b) board/pixel change, (c)
episode-termination reason.

The sweep lives at::

    hasofocus-university-of-the-west-of-england/arc3-wm-sprint   (24 runs, finished)

NOTE this is reachable only under the *hasofocus* W&B account key, not the
*haso* account whose key was cached in ~/_netrc during the first audit (that
account sees 0 projects under hasofocus-*). Export the hasofocus key:

    export WANDB_API_KEY=...      # hasofocus key
    python analysis/ls20_lf52_wandb_probe.py

8 ls20/lf52 run ids:
  warm: ls20 2329m1sj / lwncvl0i   lf52 ttjocji0 / hr6y701f
  cold: ls20 xkqetscm / h5s5q7lt   lf52 dqqb8jtm / 1mnwbryd

Read-only: no writes to W&B or B2, no training.
"""
from __future__ import annotations
import os
import sys

PROJECT = "hasofocus-university-of-the-west-of-england/arc3-wm-sprint"
LS20_WARM_S0 = "2329m1sj"
RUN_IDS = {
    "warm_ls20_s0": "2329m1sj", "warm_ls20_s1": "lwncvl0i",
    "warm_lf52_s0": "ttjocji0", "warm_lf52_s1": "hr6y701f",
    "cold_ls20_s0": "xkqetscm", "cold_ls20_s1": "h5s5q7lt",
    "cold_lf52_s0": "dqqb8jtm", "cold_lf52_s1": "1mnwbryd",
}


def main():
    import wandb
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        sys.exit("set WANDB_API_KEY to the hasofocus account key first")
    api = wandb.Api(api_key=key, timeout=30)

    r = api.run(f"{PROJECT}/{LS20_WARM_S0}")
    print(f"== inventory: {r.name} ({r.id}) ==")
    print(f"  summary keys: {len(r.summary.keys())}")
    print(f"  history cols: {len(r.history(samples=1, pandas=True).columns)}")
    print(f"  logged_artifacts: {[a.name for a in r.logged_artifacts()]}")
    fs = [f.name for f in r.files()]
    print(f"  files: {len(fs)}  policy_gifs={sum('policy_image' in f for f in fs)}"
          f"  openloop_gifs={sum('openloop/image' in f for f in fs)}"
          f"  action_artifact={any('action' in f.lower() and f.endswith(('.json','.parquet','.table.json')) for f in fs)}")

    print("\n== media presence across all 8 runs (action_artifact should be False everywhere) ==")
    for nm, rid in RUN_IDS.items():
        rr = api.run(f"{PROJECT}/{rid}")
        ff = [f.name for f in rr.files()]
        print(f"  {nm:14} id={rid}: policy={sum('policy_image' in f for f in ff):2}"
              f" openloop={sum('openloop/image' in f for f in ff):2}"
              f" action_artifact={any('action' in f.lower() and f.endswith(('.json','.parquet')) for f in ff)}")

    print("\nTo analyse board change, download a few GIFs and run PIL ImageSequence over\n"
          "the frames (see analysis/ls20_lf52_audit.md 2026-06-09 section for the result).")


if __name__ == "__main__":
    main()
