"""Check: does every terminal-to-non-terminal transition go through an
explicit RESET row? If yes, the implicit-restart rule is unnecessary —
just discard post-terminal rows until the next RESET.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def scan(path: Path):
    """Walk rows; for every terminal->non-terminal transition, capture
    whether an explicit RESET sat between them."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            data = obj.get("data") or {}
            if "state" not in data or "action_input" not in data:
                continue
            ai = data.get("action_input") or {}
            rows.append({
                "line": line_no,
                "state": data["state"],
                "action": ai.get("id"),
                "levels": int(data.get("levels_completed", -1)),
            })

    transitions = []
    in_terminal_block = False
    block_start = None
    for i, r in enumerate(rows):
        is_terminal = r["state"] in {"WIN", "GAME_OVER"}
        if is_terminal and not in_terminal_block:
            in_terminal_block = True
            block_start = i
        elif not is_terminal and in_terminal_block:
            # Exited a terminal block.
            block_rows = rows[block_start:i]
            block_actions = [br["action"] for br in block_rows]
            # Did this row's action_input.id signal RESET?
            this_is_reset = r["action"] in (0, "RESET")
            transitions.append({
                "block_start_line": block_rows[0]["line"],
                "block_end_line": block_rows[-1]["line"],
                "block_len": len(block_rows),
                "first_state": block_rows[0]["state"],
                "next_action": r["action"],
                "next_state": r["state"],
                "next_is_reset": this_is_reset,
            })
            in_terminal_block = False
            block_start = None
    return transitions


def main(root_arg: str) -> int:
    root = Path(root_arg)
    files = sorted(root.rglob("*.recording.jsonl"))
    summary = Counter()
    examples = []
    for p in files:
        for t in scan(p):
            key = (
                f"first_state={t['first_state']}, "
                f"next_action={t['next_action']}, "
                f"next_state={t['next_state']}"
            )
            summary[key] += 1
            if t["next_action"] not in (0, "RESET"):
                examples.append((p.name, t))
    print(f"files scanned: {len(files)}")
    print(f"terminal->non-terminal transitions:")
    for k, v in summary.most_common():
        print(f"  [{v}x] {k}")
    if examples:
        print(f"\n*** non-RESET exits from terminal block ({len(examples)}):")
        for name, t in examples[:10]:
            print(f"  {name}: lines {t['block_start_line']}-{t['block_end_line']} "
                  f"({t['block_len']} rows in terminal block) -> "
                  f"action={t['next_action']} state={t['next_state']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
