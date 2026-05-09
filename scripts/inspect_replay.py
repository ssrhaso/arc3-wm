"""Quick replay-file schema inspector. Phase 0 doc helper.

Usage:
    .venv/Scripts/python.exe scripts/inspect_replay.py <path>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(path: str) -> int:
    p = Path(path)
    n_lines = 0
    top_keys = Counter()
    data_keys = Counter()
    states = Counter()
    actions = Counter()
    levels = Counter()
    full_resets = 0
    first_line = None
    last_line = None

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)
            if first_line is None:
                first_line = obj
            last_line = obj
            top_keys.update(obj.keys())
            d = obj.get("data", {}) or {}
            data_keys.update(d.keys())
            if "state" in d:
                states[d["state"]] += 1
            ai = d.get("action_input")
            if isinstance(ai, dict):
                actions[ai.get("id", "?")] += 1
            if "levels_completed" in d:
                levels[d["levels_completed"]] += 1
            if d.get("full_reset"):
                full_resets += 1

    print(f"file: {p}")
    print(f"size: {p.stat().st_size:,} bytes")
    print(f"lines: {n_lines}")
    print(f"top-level keys: {dict(top_keys)}")
    print(f"data keys: {dict(data_keys)}")
    print(f"state counts: {dict(states)}")
    print(f"action counts: {dict(actions)}")
    print(f"levels_completed values: {dict(levels)}")
    print(f"full_reset events: {full_resets}")
    if first_line:
        d0 = first_line.get("data", {}) or {}
        print(
            f"first line: game_id={d0.get('game_id')} "
            f"state={d0.get('state')} "
            f"available_actions={d0.get('available_actions')} "
            f"frame_layers={len(d0.get('frame', []))} "
            f"frame_shape={(len(d0.get('frame', [[]])[0]), len(d0.get('frame', [[[]]])[0][0])) if d0.get('frame') else None}"
        )
    if last_line:
        dz = last_line.get("data", {}) or {}
        print(
            f"last  line: state={dz.get('state')} levels_completed={dz.get('levels_completed')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
