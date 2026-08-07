#!/usr/bin/env python
"""Aggregate a TRM sweep directory into result tables.

Walks ``<sweep>/eval/<game>_s<seed>_<composition>/`` directories, recomputes
RHAE from each ``eval_episodes.jsonl`` (via the reference implementation),
collects WM/BC validation metrics from the training run dirs, and writes
``results.json`` plus a markdown summary to stdout.

Usage:
    python scripts/trm_build_results.py --sweep /scratch/$USER/arc3-trm/sweep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_rhae import aggregate_eval_episodes, load_episodes_from_jsonl  # noqa: E402

from arc3_wm.rhae import RHAEAggregator  # noqa: E402

EVAL_DIR_RE = re.compile(r"^(?P<game>[a-z0-9]+)_s(?P<seed>\d+)_(?P<comp>[a-z]+)$")


def load_baselines(path: Path) -> dict:
    return json.loads(path.read_text())


def collect(sweep: Path, baselines: dict) -> dict:
    agg = RHAEAggregator(human_baselines=baselines)
    rows = []
    for eval_dir in sorted((sweep / "eval").glob("*")):
        m = EVAL_DIR_RE.match(eval_dir.name)
        episodes_file = eval_dir / "eval_episodes.jsonl"
        if not m or not episodes_file.exists():
            continue
        game, seed, comp = m["game"], int(m["seed"]), m["comp"]
        rewards = load_episodes_from_jsonl(episodes_file)
        per_level = aggregate_eval_episodes(rewards)
        result = agg(game_id=game, ai_actions_per_level=per_level)
        summary_file = eval_dir / "summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        rows.append(
            {
                "game": game,
                "seed": seed,
                "composition": comp,
                "episodes": len(rewards),
                "rhae": result[f"eval/rhae/per_game/{game}"],
                "levels_completed": result[f"eval/rhae/levels_completed/{game}"],
                "level_clears_total": summary.get("level_clears"),
                "wins": summary.get("wins"),
                "total_actions": summary.get("total_actions"),
            }
        )
    metrics = {}
    for kind in ("wm", "bc"):
        for run_dir in sorted((sweep / kind).glob("*")):
            best = run_dir / "best.pt"
            run_json = run_dir / "run.json"
            if not run_json.exists():
                continue
            last_val = None
            metrics_file = run_dir / "metrics.jsonl"
            if metrics_file.exists():
                for line in metrics_file.read_text().splitlines():
                    rec = json.loads(line)
                    if "val" in rec:
                        last_val = rec["val"] | {"step": rec.get("step")}
            metrics[f"{kind}/{run_dir.name}"] = {
                "has_best": best.exists(),
                "last_val": last_val,
            }
    return {"rhae": rows, "training": metrics}


def to_markdown(results: dict) -> str:
    rows = results["rhae"]
    games = sorted({r["game"] for r in rows})
    comps = ["random", "bc", "wm", "hybrid"]
    lines = ["| game | seed | " + " | ".join(comps) + " |",
             "|---|---|" + "---|" * len(comps)]
    for game in games:
        for seed in sorted({r["seed"] for r in rows if r["game"] == game}):
            cells = []
            for comp in comps:
                match = [
                    r for r in rows
                    if r["game"] == game and r["seed"] == seed and r["composition"] == comp
                ]
                cells.append(f"{match[0]['rhae']:.4f}" if match else "-")
            lines.append(f"| {game} | {seed} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument(
        "--baselines", type=Path,
        default=Path(__file__).resolve().parent.parent / "data/human_baselines.json",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = collect(args.sweep, load_baselines(args.baselines))
    out = args.out or (args.sweep / "results.json")
    out.write_text(json.dumps(results, indent=2))
    print(to_markdown(results))
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
