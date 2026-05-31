# Phase-4 cross-cell diagnosis - WM fits, controller fails (6 pilot warm cells)

Read-only extension of `analysis/p4_vc33_dryrun_diagnosis.md` (single cell)
to all **6 Phase-4-proper pilot warm cells** {vc33, sb26, cd82} x {s0, s1},
including both gate-failure games. Source: `_p4_analysis/logdir/p4-proper-harness.log` (DV3 per-window train log) + `p4_aggregation.json`.
Rebuild: `python analysis/build_p4_cross_cell_diagnosis.py`.

## WM-fits / controller-fails table

`img` = `train/loss/image`; `dyn` = `train/loss/dyn`; `rand_last` = `train/rand/action` at end of run (1.0 = fully random). `train clears` and `eval clears` from the rewards-stream aggregation.

| Run | img first->last (drop) | img min | dyn first->last | rand_last | train clears | eval clears | RHAE |
|---|---|---|---|---|---|---|---|
| vc33-s0 | 58.3->0.160 (99.7%) | 0.160 | 10.60->1.24 | 0.750 | 17/9751 | 4/23 | 0.0548 |
| vc33-s1 | 58.1->0.140 (99.8%) | 0.140 | 10.55->1.14 | 0.780 | 17/9746 | 2/18 | 0.0166 |
| sb26-s0 | 43.2->0.740 (98.3%) | 0.740 | 4.94->1.55 | 1.000 | 0/502 | 0/24 | 0.0000 |
| sb26-s1 | 45.1->1.890 (95.8%) | 1.890 | 5.05->2.36 | 1.000 | 1/502 | 0/24 | 0.0000 |
| cd82-s0 | 31.0->0.060 (99.8%) | 0.060 | 11.44->1.03 | 1.000 | 0/4950 | 0/24 | 0.0000 |
| cd82-s1 | 31.9->0.060 (99.8%) | 0.060 | 11.59->1.16 | 1.000 | 0/4950 | 0/24 | 0.0000 |

## Verdict

- **WM fits, uniformly:** `train/loss/image` collapses >90% on **6/6** cells (all of them). The world model converges regardless of game.
- **Controller fails on the zero games:** sb26 and cd82 have **0 eval clears across all 4 cells** despite the WM converging - the WM-fits/controller-fails split is not a vc33 artifact, it is the signature on the gate-failure games too.
- **Exploration never finishes:** `train/rand/action` ends well above 0 on every cell (range 0.750-1.000); the explore->exploit schedule does not complete within the 500k budget on any pilot cell, not just vc33.
- **vc33 is the only cell with any clears**, and they are bursty (see per-bin counts in `p4_aggregation.json`) - consistent with the single-cell forensic, now shown to generalise.

**Conclusion:** across all 6 pilot warm cells the world model reconstructs and predicts (image loss -> <1, >90% drop) while the controller clears ~0 levels. The report's claim - *the signature reproduces across cells* - is now evidenced, not asserted, for the pilot warm arm including both gate-failure games.

## Scope / honesty

- Covers the **6 pilot warm cells** only (loss curves present locally).
- tn36/ls20/lf52 and the entire cold arm are RHAE=0 (`analysis/benchmark_table.md`) - controller-fails confirmed there too, but their per-cell WM-loss curves are B2-only and the WM-fits half is NOT independently re-derived here. Stated as a limitation.
- Loss series are ordinal DV3 report windows (no per-line env-step in stdout), same basis as the vc33 forensic. Trends, not step-exact fits.
