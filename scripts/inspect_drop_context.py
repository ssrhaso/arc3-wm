"""Show ±5 rows of context around any levels_completed drop."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path_arg: str) -> int:
    p = Path(path_arg)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            data = obj.get("data") or {}
            if "levels_completed" not in data or "action_input" not in data:
                continue
            ai = data.get("action_input") or {}
            rows.append({
                "line": line_no,
                "state": data.get("state"),
                "levels": int(data["levels_completed"]),
                "action": ai.get("id"),
            })

    drop_indices = []
    for i in range(1, len(rows)):
        if rows[i]["levels"] < rows[i - 1]["levels"]:
            drop_indices.append(i)

    print(f"file: {p}")
    print(f"total step rows: {len(rows)}")
    print(f"drops: {len(drop_indices)}")
    for di in drop_indices:
        lo = max(0, di - 5)
        hi = min(len(rows), di + 6)
        print(f"  context for drop at row index {di} (line {rows[di]['line']}):")
        for j in range(lo, hi):
            r = rows[j]
            marker = " <-- DROP" if j == di else ""
            print(
                f"    [{j}] line={r['line']:4d} state={r['state']:14s} "
                f"levels={r['levels']} action={r['action']}{marker}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
