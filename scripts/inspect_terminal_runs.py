"""Diagnostic: count consecutive runs of terminal-state rows in a replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path_arg: str) -> int:
    p = Path(path_arg)
    runs = []
    current = None
    prev_levels = None
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            data = obj.get("data") or {}
            if "state" not in data or "action_input" not in data:
                continue
            state = data.get("state")
            ai = data.get("action_input") or {}
            action_id = ai.get("id")
            levels = int(data.get("levels_completed", -1))
            if state in {"WIN", "GAME_OVER"}:
                if current is None:
                    current = {"start": line_no, "state": state, "len": 0,
                               "actions": [], "levels": []}
                current["len"] += 1
                current["actions"].append(action_id)
                current["levels"].append(levels)
            else:
                if current is not None:
                    runs.append(current)
                    current = None
            prev_levels = levels
    if current is not None:
        runs.append(current)
    print(f"file: {p}")
    print(f"terminal-state runs: {len(runs)}")
    for r in runs:
        print(
            f"  start_line={r['start']} state={r['state']} len={r['len']}"
            f" levels_seen={sorted(set(r['levels']))}"
            f" first_actions={r['actions'][:5]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
