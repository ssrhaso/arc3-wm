"""Fetch ARC-AGI-3 documentation pages into docs/arc-agi-3/.

Phase 0 deliverable. Run from repo root:
    .venv/Scripts/python.exe scripts/fetch_docs.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PAGES = [
    "methodology.md",
    "game-schema.md",
    "available-games.md",
    "local-vs-online.md",
    "actions.md",
    "recordings.md",
    "rate_limits.md",
    "vocabulary.md",
    "scorecards.md",
    "toolkit/overview.md",
    "toolkit/arc_agi.md",
    "toolkit/list-actions.md",
    "toolkit/list-games.md",
    "toolkit/get-scorecard.md",
    "toolkit/submit-action.md",
    "toolkit/environment_wrapper.md",
]

BASE = "https://docs.arcprize.org/"
OUT = Path(__file__).resolve().parents[1] / "docs" / "arc-agi-3"


def fetch(rel: str) -> tuple[str, bytes]:
    url = BASE + rel
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return url, resp.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for rel in PAGES:
        out_path = OUT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            url, data = fetch(rel)
        except Exception as e:
            print(f"FAIL {rel}: {e}", file=sys.stderr)
            failures.append(rel)
            continue
        out_path.write_bytes(data)
        print(f"OK   {rel}  ({len(data)} bytes) -> {out_path}")
    if failures:
        print(f"\n{len(failures)} failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
