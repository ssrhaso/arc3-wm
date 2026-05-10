"""Survey all staged replays for levels_completed decreases."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def scan(path: Path):
    prev_levels = None
    prev_state = None
    prev_action_id = None
    drops = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            data = obj.get("data") or {}
            if "levels_completed" not in data or "action_input" not in data:
                continue
            levels = int(data["levels_completed"])
            ai = data.get("action_input") or {}
            action_id = ai.get("id")
            state = data.get("state")
            if prev_levels is not None and levels < prev_levels:
                drops.append(
                    {
                        "line": line_no,
                        "from": prev_levels,
                        "to": levels,
                        "prev_state": prev_state,
                        "curr_state": state,
                        "prev_action": prev_action_id,
                        "curr_action": action_id,
                    }
                )
            prev_levels = levels
            prev_state = state
            prev_action_id = action_id
    return drops


def main(root_arg: str) -> int:
    root = Path(root_arg)
    files = sorted(root.rglob("*.recording.jsonl"))
    n_files = len(files)
    n_with_drops = 0
    total_drops = 0
    by_game: Counter = Counter()
    drop_contexts: Counter = Counter()
    for p in files:
        drops = scan(p)
        if drops:
            n_with_drops += 1
            total_drops += len(drops)
            game_id = p.parent.name
            by_game[game_id] += len(drops)
            for d in drops:
                key = (
                    f"prev_state={d['prev_state']}, curr_state={d['curr_state']}, "
                    f"prev_action={d['prev_action']}, curr_action={d['curr_action']}"
                )
                drop_contexts[key] += 1
    print(f"files scanned: {n_files}")
    print(f"files with at least one decrease: {n_with_drops}")
    print(f"total decreases: {total_drops}")
    print(f"by game: {dict(by_game)}")
    print(f"contexts (top 10):")
    for ctx, n in drop_contexts.most_common(10):
        print(f"  [{n}x] {ctx}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
