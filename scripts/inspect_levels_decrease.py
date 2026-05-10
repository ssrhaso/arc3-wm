"""One-off diagnostic: find rows where levels_completed decreases.

Used to surface a Phase-1.7 finding — at least one staged replay has a
levels_completed drop, contradicting the replay-format.md claim that the
field is monotonically non-decreasing within a session.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path_arg: str) -> int:
    p = Path(path_arg)
    prev_levels = None
    prev_action_id = None
    prev_state = None
    decreases = []
    with p.open("r", encoding="utf-8") as f:
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
                decreases.append(
                    (line_no, prev_levels, levels,
                     prev_action_id, action_id, prev_state, state)
                )
            prev_levels = levels
            prev_action_id = action_id
            prev_state = state
    print(f"file: {p}")
    print(f"decreases found: {len(decreases)}")
    for tup in decreases[:20]:
        line_no, prev, curr, prev_ai, curr_ai, prev_st, curr_st = tup
        print(
            f"  line {line_no}: levels {prev} -> {curr} (delta={curr - prev})"
            f" prev_action={prev_ai} curr_action={curr_ai}"
            f" prev_state={prev_st} curr_state={curr_st}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
