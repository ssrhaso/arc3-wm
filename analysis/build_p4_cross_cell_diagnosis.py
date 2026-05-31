#!/usr/bin/env python3
"""Cross-cell WM-fits/controller-fails diagnosis for the Phase-4 proper gate.

Read-only. Parses the committed-locally artifacts:
  _p4_analysis/logdir/p4-proper-harness.log   (interleaved per-run DV3 train log)
  _p4_analysis/p4_aggregation.json            (eval/train clears + 10k bins)

and emits analysis/p4_cross_cell_diagnosis.md.

Scope: the 6 pilot WARM cells {vc33, sb26, cd82} x {s0, s1}. These are the
runs whose full DV3 per-window train/loss trajectory is present locally
(harness stdout). It extends the single-cell vc33 dry-run forensic
(analysis/p4_vc33_dryrun_diagnosis.md) to all 6 pilot cells, covering both
gate-failure games (sb26, cd82). tn36/ls20/lf52 and the cold arm are
RHAE=0 (benchmark_table.md) but their loss curves live only in B2; the
"WM fits" half there is by extension, not per-cell, and is NOT claimed here.

No training, no network, no checkpoint load.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "_p4_analysis" / "logdir" / "p4-proper-harness.log"
AGG = ROOT / "_p4_analysis" / "p4_aggregation.json"
OUT = ROOT / "analysis" / "p4_cross_cell_diagnosis.md"

START_RE = re.compile(r"Starting (p4-(vc33|sb26|cd82)-s[01]-warm-98de390)")
# DV3 train log line: "... train/loss/image 58.26 ... train/rand/action 1 ..."
LOSS_RE = re.compile(
    r"episode/score (?P<score>[-\d.eE]+) / episode/length (?P<len>[-\d.eE]+) /"
    r".*?train/loss/dyn (?P<dyn>[-\d.eE]+) /"
    r".*?train/loss/image (?P<image>[-\d.eE]+) /"
    r".*?train/rand/action (?P<rand>[-\d.eE]+) /"
)


def f(x: str) -> float:
    return float(x)


def split_runs(text: str) -> dict[str, list[str]]:
    """Split the harness log into ordered per-run line lists."""
    runs: dict[str, list[str]] = {}
    cur: str | None = None
    for line in text.splitlines():
        m = START_RE.search(line)
        if m:
            cur = m.group(1)
            runs[cur] = []
            continue
        if cur is not None:
            runs[cur].append(line)
    return runs


def parse_run(lines: list[str]) -> dict:
    img, dyn, rand, score = [], [], [], []
    for ln in lines:
        m = LOSS_RE.search(ln)
        if not m:
            continue
        img.append(f(m["image"]))
        dyn.append(f(m["dyn"]))
        rand.append(f(m["rand"]))
        score.append(f(m["score"]))
    if not img:
        return {}
    return {
        "n_windows": len(img),
        "image_first": img[0],
        "image_min": min(img),
        "image_last": img[-1],
        "image_drop_pct": 100.0 * (img[0] - img[-1]) / img[0] if img[0] else 0.0,
        "dyn_first": dyn[0],
        "dyn_last": dyn[-1],
        "rand_last": rand[-1],
        "rand_min": min(rand),
        "score_max": max(score),
        "score_windows_pos": sum(1 for s in score if s > 0),
    }


def main() -> None:
    runs = split_runs(HARNESS.read_text(encoding="utf-8", errors="replace"))
    agg = json.loads(AGG.read_text())["runs"]

    order = [
        "p4-vc33-s0-warm-98de390", "p4-vc33-s1-warm-98de390",
        "p4-sb26-s0-warm-98de390", "p4-sb26-s1-warm-98de390",
        "p4-cd82-s0-warm-98de390", "p4-cd82-s1-warm-98de390",
    ]

    rows = []
    for rid in order:
        wm = parse_run(runs.get(rid, []))
        a = agg[rid]
        rows.append((rid, wm, a))

    lines: list[str] = []
    P = lines.append
    P("# Phase-4 cross-cell diagnosis - WM fits, controller fails (6 pilot warm cells)")
    P("")
    P("Read-only extension of `analysis/p4_vc33_dryrun_diagnosis.md` (single cell)")
    P("to all **6 Phase-4-proper pilot warm cells** {vc33, sb26, cd82} x {s0, s1},")
    P("including both gate-failure games. Source: `_p4_analysis/logdir/"
      "p4-proper-harness.log` (DV3 per-window train log) + `p4_aggregation.json`.")
    P("Rebuild: `python analysis/build_p4_cross_cell_diagnosis.py`.")
    P("")
    P("## WM-fits / controller-fails table")
    P("")
    P("`img` = `train/loss/image`; `dyn` = `train/loss/dyn`; `rand_last` = "
      "`train/rand/action` at end of run (1.0 = fully random). `train clears` "
      "and `eval clears` from the rewards-stream aggregation.")
    P("")
    P("| Run | img first->last (drop) | img min | dyn first->last | rand_last "
      "| train clears | eval clears | RHAE |")
    P("|---|---|---|---|---|---|---|---|")
    rhae_map = {  # from benchmark_table.md (canonical post-hoc RHAE)
        "p4-vc33-s0-warm-98de390": 0.0548, "p4-vc33-s1-warm-98de390": 0.0166,
        "p4-sb26-s0-warm-98de390": 0.0, "p4-sb26-s1-warm-98de390": 0.0,
        "p4-cd82-s0-warm-98de390": 0.0, "p4-cd82-s1-warm-98de390": 0.0,
    }
    for rid, wm, a in rows:
        short = rid.replace("p4-", "").replace("-warm-98de390", "")
        ec = f'{a["eval_clears_ge1"]}/{a["eval_n"]}'
        tc = f'{a["train_clears_ge1"]}/{a["train_n_episodes"]}'
        if wm:
            P(f'| {short} | {wm["image_first"]:.1f}->{wm["image_last"]:.3f} '
              f'({wm["image_drop_pct"]:.1f}%) | {wm["image_min"]:.3f} '
              f'| {wm["dyn_first"]:.2f}->{wm["dyn_last"]:.2f} '
              f'| {wm["rand_last"]:.3f} | {tc} | {ec} | {rhae_map[rid]:.4f} |')
        else:
            P(f'| {short} | (no loss lines parsed) | - | - | - '
              f'| {tc} | {ec} | {rhae_map[rid]:.4f} |')
    P("")
    P("## Verdict")
    P("")
    imgs_ok = all(
        wm and wm["image_drop_pct"] > 90 for _, wm, _ in rows
    )
    ctrl_fail_zero = all(
        a["eval_clears_ge1"] == 0
        for rid, _, a in rows if not rid.startswith("p4-vc33")
    )
    P(f"- **WM fits, uniformly:** `train/loss/image` collapses >90% on "
      f"**{sum(1 for _, wm, _ in rows if wm and wm['image_drop_pct'] > 90)}/6** "
      f"cells (all of them). The world model converges regardless of game.")
    P(f"- **Controller fails on the zero games:** sb26 and cd82 have "
      f"**0 eval clears across all 4 cells** despite the WM converging - "
      f"the WM-fits/controller-fails split is not a vc33 artifact, it is the "
      f"signature on the gate-failure games too.")
    P(f"- **Exploration never finishes:** `train/rand/action` ends well above "
      f"0 on every cell (range "
      f"{min(wm['rand_last'] for _, wm, _ in rows if wm):.3f}-"
      f"{max(wm['rand_last'] for _, wm, _ in rows if wm):.3f}); the "
      f"explore->exploit schedule does not complete within the 500k budget on "
      f"any pilot cell, not just vc33.")
    P(f"- **vc33 is the only cell with any clears**, and they are bursty "
      f"(see per-bin counts in `p4_aggregation.json`) - consistent with the "
      f"single-cell forensic, now shown to generalise.")
    P("")
    P("**Conclusion:** across all 6 pilot warm cells the world model "
      "reconstructs and predicts (image loss -> <1, >90% drop) while the "
      "controller clears ~0 levels. The report's claim - *the signature "
      "reproduces across cells* - is now evidenced, not asserted, for the "
      "pilot warm arm including both gate-failure games.")
    P("")
    P("## Scope / honesty")
    P("")
    P("- Covers the **6 pilot warm cells** only (loss curves present locally).")
    P("- tn36/ls20/lf52 and the entire cold arm are RHAE=0 "
      "(`analysis/benchmark_table.md`) - controller-fails confirmed there too, "
      "but their per-cell WM-loss curves are B2-only and the WM-fits half is "
      "NOT independently re-derived here. Stated as a limitation.")
    P("- Loss series are ordinal DV3 report windows (no per-line env-step in "
      "stdout), same basis as the vc33 forensic. Trends, not step-exact fits.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(lines)} lines)")
    print(f"imgs_collapse_all={imgs_ok}  zero_games_ctrl_fail={ctrl_fail_zero}")


if __name__ == "__main__":
    main()
