"""Spec for scripts/trm_build_results.py - sweep aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import trm_build_results as B  # noqa: E402

BASELINES = {
    "vc33": {"baselines": {"1": 20, "2": 40}, "total_levels": 3},
    "ls20": {"baselines": {"1": 10}, "total_levels": 2},
}


def _write_eval(dirpath: Path, episodes: list[list[float]], summary: dict) -> None:
    dirpath.mkdir(parents=True)
    with open(dirpath / "eval_episodes.jsonl", "w") as fh:
        for rewards in episodes:
            fh.write(json.dumps({"rewards": rewards, "terminal_state": None}) + "\n")
    (dirpath / "summary.json").write_text(json.dumps(summary))


def test_collect_and_markdown(tmp_path):
    sweep = tmp_path / "sweep"
    # vc33 hybrid: clears level 1 in 10 actions (rewards[0] is the initial
    # obs step) -> level score min((20/10)^2, 1.15) = 1.15, weighted 1/(1+2+3).
    _write_eval(
        sweep / "eval" / "vc33_s0_hybrid",
        [[0.0] + [0.0] * 9 + [1.0]],
        {"level_clears": 1, "wins": 0, "total_actions": 10},
    )
    _write_eval(
        sweep / "eval" / "vc33_s0_random",
        [[0.0, 0.0, 0.0]],
        {"level_clears": 0, "wins": 0, "total_actions": 2},
    )
    # A malformed directory name is skipped.
    (sweep / "eval" / "notamatch").mkdir(parents=True)

    results = B.collect(sweep, BASELINES)
    rows = {(r["game"], r["seed"], r["composition"]): r for r in results["rhae"]}
    assert rows[("vc33", 0, "random")]["rhae"] == 0.0
    hybrid = rows[("vc33", 0, "hybrid")]
    assert hybrid["levels_completed"] == 1
    assert hybrid["rhae"] > 0.15  # 1.15 * 1/6 = 0.1917
    md = B.to_markdown(results)
    assert "| vc33 | 0 |" in md
    assert "0.0000" in md and "0.19" in md


def test_main_writes_results_json(tmp_path):
    sweep = tmp_path / "sweep"
    _write_eval(
        sweep / "eval" / "ls20_s1_bc",
        [[0.0, 0.0]],
        {"level_clears": 0, "wins": 0, "total_actions": 1},
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps(BASELINES))
    rc = B.main(["--sweep", str(sweep), "--baselines", str(baselines)])
    assert rc == 0
    data = json.loads((sweep / "results.json").read_text())
    assert data["rhae"][0]["game"] == "ls20"
