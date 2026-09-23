"""End-to-end smoke: parse every staged replay, report aggregate stats.

Surfaces:
- Total files / episodes / steps.
- Per-game episode + step counts.
- Stats counter (post-terminal noise rows discarded).
- Any ReplayParseError (file:line + reason).
- Multi-level transitions (reward > 1.0): observational, not a failure.
- Files with full_reset=True warnings.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

from arc3_wm.replay_loader import (
    ReplayParseError,
    load_replay_file,
)


def main(root_arg: str) -> int:
    root = Path(root_arg)
    files = sorted(root.rglob("*.recording.jsonl"))
    n_files = len(files)
    n_episodes = 0
    n_steps = 0
    n_multi_level = 0
    n_files_with_warnings = 0
    by_game_eps: Counter = Counter()
    by_game_steps: Counter = Counter()
    errors: list = []
    stats: dict = {}

    for p in files:
        game_id = p.parent.name
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                eps = list(load_replay_file(p, stats=stats))
            except ReplayParseError as e:
                errors.append((p, str(e)))
                continue
            if any(issubclass(w.category, UserWarning) for w in caught):
                n_files_with_warnings += 1
        for ep in eps:
            n_episodes += 1
            by_game_eps[game_id] += 1
            for s in ep:
                n_steps += 1
                by_game_steps[game_id] += 1
                if float(s["reward"]) > 1.0:
                    n_multi_level += 1

    print(f"files: {n_files}")
    print(f"episodes: {n_episodes}")
    print(f"steps: {n_steps}")
    print(f"multi-level transitions (reward > 1.0): {n_multi_level}")
    print(f"files emitting UserWarning: {n_files_with_warnings}")
    print(f"post-terminal noise rows discarded: {stats.get('noise_rows_discarded', 0)}")
    print()
    print("per-game (episodes, steps):")
    for g in sorted(set(list(by_game_eps.keys()) + list(by_game_steps.keys()))):
        print(f"  {g:6s}  eps={by_game_eps[g]:4d}  steps={by_game_steps[g]:6d}")
    if errors:
        print()
        print(f"*** {len(errors)} files raised ReplayParseError:")
        for p, msg in errors[:20]:
            print(f"  {p.name}")
            print(f"    {msg}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
