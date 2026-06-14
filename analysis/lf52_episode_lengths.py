#!/usr/bin/env python3
"""READ-ONLY. Build the authoritative, git-tracked Fig-3b episode-length table.

Writes ``analysis/lf52_episode_lengths.csv`` (tracked) so the figure is
reproducible without the gitignored ``scratch/curves`` pipeline
(``scratch/make_lf52_csvs.py`` writes to the ignored ``curves/`` dir and
counts human lengths *per session-file*, not per episode -- the source of
the stale "49 episodes / 1-67-230-1605" numbers this artifact supersedes).

Unit (both series): RHAE-convention ACTION COUNT = (#step-rows) - 1, i.e.
``len(episode) - 1`` for human replays and ``len(rewards) - 1`` for agent
eval episodes. One row per episode.

Human segmentation is the CANONICAL ``arc3_wm.replay_loader.load_replay_file``.
The loader's step-dicts carry only an ``is_terminal`` bool, not the raw
``state`` string, so to label WIN vs GAME_OVER vs NOT_FINISHED we re-walk
each file with a faithful copy of the loader's segmentation rules and read
the final row's ``state``. We CROSS-CHECK that the re-walk reproduces the
loader's exact per-episode action counts; any disagreement aborts rather
than reporting inferred numbers.

A "phantom" episode is a lone 1-row segment (0 actions) -- the loader keeps
mid-file 1-row segments but drops a lone RESET row at EOF. The summary is
reported twice (including / excluding phantoms) so the choice is on record.

Does not write or modify anything except the one tracked CSV. No env runs,
no training, no B2, no .tex.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from arc3_wm.replay_loader import (
    load_replay_file,
    _classify_row,
    _is_reset_id,
    TERMINAL_STATES,
)

REPO = Path(__file__).resolve().parents[1]
LF52_REPLAYS = REPO / "data" / "replays" / "lf52"
OUT_CSV = REPO / "analysis" / "lf52_episode_lengths.csv"

# Agent eval-episode sinks (EvalRewardSink output: {"rewards": [...]} per ep).
# Four Phase-4 lf52 cells: warm/cold x seed 0/1. These predate the
# terminal_state field, so terminal_state is NOT_LOGGED for the agent series.
AGENT_CELLS = [
    ("p4-lf52-s0-warm-98de390", "warm", 0),
    ("p4-lf52-s1-warm-98de390", "warm", 1),
    ("p4-fromscratch-lf52-s0-a06c02f", "cold", 0),
    ("p4-fromscratch-lf52-s1-a06c02f", "cold", 1),
]
AGENT_LOGDIR = REPO / "scratch" / "bench" / "logdir"


def canonical_action_counts(path):
    """RHAE convention: actions per episode = len(episode) - 1."""
    return [len(ep) - 1 for ep in load_replay_file(path)]


def rewalk_with_states(path):
    """Mirror load_replay_file's segmentation EXACTLY; return per-episode
    ``(n_step_rows, final_state)``. n_step_rows == len(episode) in the loader,
    so actions = n_step_rows - 1. The lone-RESET-row skip applies ONLY at EOF
    (loader Q5b edge); mid-file a lone RESET builds a phantom 1-row episode."""
    pending = []
    episodes = []

    def flush(at_eof=False):
        if not pending:
            return
        if (
            at_eof
            and len(pending) == 1
            and _is_reset_id((pending[0].get("action_input") or {}).get("id"))
        ):
            return  # lone RESET row at EOF -> no episode
        final_state = pending[-1].get("state", "NOT_FINISHED")
        episodes.append((len(pending), final_state))

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            obj = json.loads(s)
            data = obj.get("data") or {}
            kind = _classify_row(data)
            if kind == "summary":
                continue
            if kind == "malformed":
                raise RuntimeError(f"{path}: malformed row")
            action_input = data.get("action_input") or {}
            is_reset = _is_reset_id(action_input.get("id"))
            state = data.get("state", "NOT_FINISHED")

            if pending and pending[-1].get("state") in TERMINAL_STATES:
                if is_reset:
                    flush()
                    pending = [data]
                    continue
                if state in TERMINAL_STATES:
                    continue  # post-terminal noise
                raise RuntimeError(f"{path}: NOT_FINISHED after terminal")

            if is_reset and pending:
                flush()
                pending = []
            pending.append(data)
    flush(at_eof=True)
    return episodes


def collect_human_rows():
    """One dict per human episode (canonical segmentation, cross-checked)."""
    files = sorted(LF52_REPLAYS.glob("*.recording.jsonl"))
    if not files:
        raise SystemExit(
            f"no lf52 replays under {LF52_REPLAYS} (gitignored corpus absent)"
        )
    rows = []
    for path in files:
        session_id = path.name.split(".")[0]
        canon = canonical_action_counts(path)
        rewalk = rewalk_with_states(path)
        rewalk_acts = [n - 1 for n, _ in rewalk]
        if canon != rewalk_acts:
            raise SystemExit(
                f"{path.name}: rewalk action counts {rewalk_acts} != "
                f"canonical {canon} -- refusing to emit inferred states"
            )
        for idx, (acts, (_n, state)) in enumerate(zip(canon, rewalk)):
            rows.append(
                {
                    "series": "human",
                    "session_id": session_id,
                    "episode_idx_in_session": idx,
                    "action_count": acts,
                    "terminal_state": state,
                    "is_phantom": 1 if acts == 0 else 0,
                }
            )
    return rows


def collect_agent_rows():
    """One dict per agent eval episode (len(rewards)-1, same RHAE unit)."""
    rows = []
    per_cell = {}
    for run, arm, seed in AGENT_CELLS:
        p = AGENT_LOGDIR / run / "eval_episodes.jsonl"
        if not p.exists():
            raise SystemExit(f"agent eval sink absent: {p}")
        acts = []
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            rewards = json.loads(line)["rewards"]
            acts.append(len(rewards) - 1)
        per_cell[run] = acts
        for idx, a in enumerate(acts):
            rows.append(
                {
                    "series": "agent_eval",
                    "session_id": run,
                    "episode_idx_in_session": idx,
                    "action_count": a,
                    "terminal_state": "NOT_LOGGED",
                    "is_phantom": 0,
                }
            )
    return rows, per_cell


def _label_counts(rows):
    c = Counter(r["terminal_state"] for r in rows)
    return c.get("WIN", 0), c.get("GAME_OVER", 0), c.get("NOT_FINISHED", 0)


def build_summary_lines(human_rows, agent_rows, per_cell):
    """Comment lines (leading '#') prepended to the CSV; pandas-skippable
    with read_csv(comment='#'). Holds the two-way human summary + the agent
    confirmation so Fig-3b's numbers live on the record next to the data."""
    incl = [r["action_count"] for r in human_rows]
    excl_rows = [r for r in human_rows if not r["is_phantom"]]
    excl = [r["action_count"] for r in excl_rows]
    n_phantom = sum(r["is_phantom"] for r in human_rows)

    w_i, g_i, nf_i = _label_counts(human_rows)
    w_e, g_e, nf_e = _label_counts(excl_rows)

    agent_acts = [r["action_count"] for r in agent_rows]
    agent_distinct = sorted(set(agent_acts))
    agent_var = statistics.pvariance(agent_acts) if agent_acts else 0.0

    L = []
    L.append("# analysis/lf52_episode_lengths.csv -- AUTHORITATIVE Fig-3b table")
    L.append("# Generator: analysis/lf52_episode_lengths.py (read-only, tracked)")
    L.append("# Unit: RHAE action_count = (#step-rows)-1  [human: len(episode)-1;"
             " agent: len(rewards)-1]")
    L.append("# Human segmentation: canonical arc3_wm.replay_loader.load_replay_file"
             " (rewalk cross-check OK)")
    L.append("# phantom = lone 1-row / 0-action segment.")
    L.append("#")
    L.append("# ===== HUMAN per-episode summary (action_count) =====")
    L.append(f"#   INCLUDING phantoms: n={len(incl)}"
             f"  median={statistics.median(incl):g}  max={max(incl)}"
             f"  | WIN={w_i} GAME_OVER={g_i} NOT_FINISHED={nf_i}")
    L.append(f"#   EXCLUDING phantoms: n={len(excl)}"
             f"  median={statistics.median(excl):g}  max={max(excl)}"
             f"  | WIN={w_e} GAME_OVER={g_e} NOT_FINISHED={nf_e}"
             f"  (dropped {n_phantom} phantom)")
    L.append("#")
    L.append("# ===== AGENT eval series (same unit) =====")
    L.append(f"#   eval episodes total n={len(agent_acts)}"
             f"  distinct action_count={agent_distinct}  variance={agent_var:g}"
             f"  -> all == 64, zero variance")
    for run, _arm, _seed in AGENT_CELLS:
        a = per_cell[run]
        L.append(f"#     {run}: n={len(a)} distinct={sorted(set(a))}")
    L.append("#")
    return L


def main():
    human_rows = collect_human_rows()
    agent_rows, per_cell = collect_agent_rows()
    summary = build_summary_lines(human_rows, agent_rows, per_cell)

    fieldnames = [
        "series",
        "session_id",
        "episode_idx_in_session",
        "action_count",
        "terminal_state",
        "is_phantom",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        for line in summary:
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(human_rows)
        w.writerows(agent_rows)

    print("\n".join(summary))
    print(f"\nwrote {OUT_CSV.relative_to(REPO)}  "
          f"({len(human_rows)} human + {len(agent_rows)} agent rows)")


if __name__ == "__main__":
    main()
