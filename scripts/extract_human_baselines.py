"""Build the per-game / per-level human-baseline action-count fixture.

CLI:

    python scripts/extract_human_baselines.py \\
        --replays-root data/replays \\
        --out data/human_baselines.json

Derives ``data/human_baselines.json`` from the 340 ``.recording.jsonl``
files under ``data/replays/{game_id}/``. The fixture is consumed by
``arc3_wm.rhae.RHAEAggregator`` and the post-hoc CLI
``scripts/compute_rhae.py``.

Per Decision A (this session), ``total_levels`` is read from the
JSONL ``win_levels`` field (sourced from ``arc_agi.FrameData.win_levels``
per ``arc3_wm/env.py:148``; documented at ``docs/replay-format.md``).
It is per-game-constant - the extractor asserts every row of every
session for a game agrees on ``win_levels`` and raises loudly on
mismatch (cheap insurance against replay/build drift).

Per Decision B (this session, mirrors Notion "RHAE - reference >
Coverage threshold"), the ``baselines`` map only contains levels with
``n >= min_completers`` (default 2). Levels with a single completer are
statistically indefensible as a median and are silently dropped from
``baselines``; ``total_levels`` still reflects the full game so
``RHAEAggregator`` can skip uncovered levels from both numerator AND
denominator.

Output fixture shape (per Decision A)::

    {
      "vc33": {
        "total_levels": 7,
        "baselines": {"1": 47, "2": 132, ...}
      },
      ...
    }

Per D5 (prior session) this reuses ``arc3_wm.replay_loader.load_replay_file``
for cn04-safe episode segmentation. Per D1 the upper-median rule is
``sorted(values)[len(values) // 2]`` 0-indexed - methodology.md's
"upper of two middle entries". Per-session aggregation: each
``.recording.jsonl`` contributes one entry per level it cleared, the
MIN across the session's episodes (best attempt). Each session
contributes at most one entry per level to the per-game upper-median
pool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping

# Add repo root to sys.path so ``from arc3_wm.replay_loader import ...``
# resolves when this file is run as a script (not as a module).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.replay_loader import load_replay_file  # noqa: E402


def upper_median(values: List[int]) -> int:
    """methodology.md / D1: ``sorted(values)[len(values) // 2]`` 0-indexed.

    For odd n this is the true middle entry; for even n it's the upper
    of the two middle entries (e.g. n=2 -> larger; n=10 -> index 5).
    For n=1 the rule degenerates to the sole value. For n=0 raises -
    no median is defined.
    """
    if not values:
        raise ValueError("upper_median: empty values; no median defined")
    return sorted(values)[len(values) // 2]


def count_actions_per_level(
    episode: List[Mapping[str, Any]],
) -> dict[int, int]:
    """Per-episode 1-indexed-level -> action-count for COMPLETED levels.

    Method: walk the episode's step-dict list and maintain the cumulative
    reward (which equals ``levels_completed`` at the time of obs[i],
    since the loader emits ``reward[i] = levels_completed[i] -
    levels_completed[i-1]`` and ``cum_reward`` resets to 0 with the
    first step). The player's CURRENT level at step ``i`` is
    ``cum_reward + 1``. The action at the sentinel last step
    (``is_last=True``) is a loader-padding placeholder - not a real
    action - so it must NOT contribute to any level's count.

    After the walk, ``cum_reward`` equals the total number of levels
    cleared in this episode. Only levels with index <= that count are
    "completed" and appear in the output; partial counts for the level
    the player died on are filtered out.
    """
    if not episode:
        return {}
    counts: dict[int, int] = {}
    cum_reward = 0
    n = len(episode)
    for i, step in enumerate(episode):
        cum_reward += int(step["reward"])
        level = cum_reward + 1  # 1-indexed current level
        if i == n - 1:
            break  # sentinel last step; its action is not real
        counts[level] = counts.get(level, 0) + 1
    completed_max = cum_reward
    return {k: v for k, v in counts.items() if k <= completed_max}


def extract_per_session_baselines(
    episodes: Iterable[List[Mapping[str, Any]]],
) -> dict[int, int]:
    """Per-session per-level MIN: one entry per level the player cleared.

    For each episode in the session, compute per-level action counts via
    ``count_actions_per_level``. If the player cleared the same level in
    multiple episodes (mid-session retry), keep the MIN. Levels the
    player never cleared are absent from the returned dict.
    """
    per_level: dict[int, int] = {}
    for ep in episodes:
        ep_counts = count_actions_per_level(list(ep))
        for level, count in ep_counts.items():
            existing = per_level.get(level)
            if existing is None or count < existing:
                per_level[level] = count
    return per_level


def read_win_levels(path: Path) -> int:
    """Return the JSONL's ``win_levels`` (per-game-constant total level count).

    Filters out ``win_levels == 0`` rows: those are cn04-style
    post-terminal-noise rows where the engine resets both
    ``levels_completed`` and ``win_levels`` to 0 (verified across the
    340-replay dataset; the only ``wl=0`` rows are 12 cn04 noise rows
    that ``load_replay_file`` itself discards). Every real game has at
    least one level, so ``win_levels=0`` is not a legitimate game total.

    Raises if no row carries a non-zero ``win_levels`` (empty/malformed
    file) or if non-zero rows disagree (engine recorded inconsistent
    game state - surface, don't guess).
    """
    path = Path(path)
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                # Malformed line; load_replay_file will surface it on
                # its own walk. We only care about extracting win_levels.
                continue
            data = obj.get("data") or {}
            wl = data.get("win_levels")
            if wl is None:
                continue
            wl = int(wl)
            if wl == 0:
                continue  # cn04 post-terminal noise; not a real total
            seen.add(wl)
    if not seen:
        raise RuntimeError(
            f"{path}: no row carried 'win_levels'; cannot derive total_levels"
        )
    if len(seen) > 1:
        raise RuntimeError(
            f"{path}: win_levels not constant within session: {sorted(seen)}"
        )
    return seen.pop()


def extract_baselines(
    replays_root: Path, *, min_completers: int = 2
) -> dict[str, dict]:
    """Walk ``replays_root``, emit per-game ``{total_levels, baselines}``.

    Iterates ``replays_root/{game_id}/*.recording.jsonl`` deterministically
    (sorted by full path). For each file:

    1. Read ``win_levels`` from the JSONL; assert per-game-constant
       across all sessions of the same game (raises on mismatch - cheap
       insurance against replay/build drift).
    2. Parse episodes via ``load_replay_file``; collapse to per-level
       action counts via ``extract_per_session_baselines``.

    Per-game output::

        {
          "total_levels": int,   # JSONL win_levels (Decision A)
          "baselines": {         # only levels with n >= min_completers (Decision B)
            "1": int, "2": int, ...
          }
        }

    Levels with fewer than ``min_completers`` entries are silently
    dropped from ``baselines`` (statistically indefensible as a
    median). ``total_levels`` still reflects the full game count;
    downstream ``RHAEAggregator`` skips uncovered levels from both
    numerator and denominator.
    """
    if min_completers < 1:
        raise ValueError(
            f"min_completers must be >= 1; got {min_completers}"
        )
    replays_root = Path(replays_root)
    per_game_pools: dict[str, dict[int, list[int]]] = {}
    total_levels_per_game: dict[str, int] = {}
    for path in sorted(replays_root.rglob("*.recording.jsonl")):
        game_id = path.parent.name
        wl = read_win_levels(path)
        existing = total_levels_per_game.get(game_id)
        if existing is None:
            total_levels_per_game[game_id] = wl
        elif existing != wl:
            raise RuntimeError(
                f"{game_id}: win_levels mismatch across sessions "
                f"(existing={existing}, found={wl} in {path.name}); "
                f"replays may have been recorded against different game builds"
            )
        episodes = list(load_replay_file(path))
        session_levels = extract_per_session_baselines(episodes)
        for level, count in session_levels.items():
            per_game_pools.setdefault(game_id, {}).setdefault(
                level, []
            ).append(count)

    out: dict[str, dict] = {}
    for game_id in sorted(total_levels_per_game):
        levels = per_game_pools.get(game_id, {})
        baselines: dict[str, int] = {}
        for level in sorted(levels):
            pool = levels[level]
            if len(pool) >= min_completers:
                baselines[str(level)] = upper_median(pool)
        out[game_id] = {
            "total_levels": total_levels_per_game[game_id],
            "baselines": baselines,
        }
    return out


def _summary(baselines: dict[str, dict]) -> str:
    """One-screen Step-2 sign-off summary: per game emit total_levels,
    n_covered (levels with a baseline after n>=2), n_uncovered (the
    rest), and the per-covered-level action-count range."""
    lines = []
    for game_id in sorted(baselines):
        entry = baselines[game_id]
        total_levels = entry["total_levels"]
        levels = entry["baselines"]
        n_covered = len(levels)
        n_uncovered = total_levels - n_covered
        if levels:
            counts = list(levels.values())
            rng = f"[{min(counts)}..{max(counts)}]"
            covered_keys = ",".join(sorted(levels, key=int))
        else:
            rng = "n/a"
            covered_keys = "<none>"
        lines.append(
            f"  {game_id}: total_levels={total_levels}, n_covered={n_covered} "
            f"({covered_keys}), n_uncovered={n_uncovered}, range={rng}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-game per-level upper-median human action-count "
            "baselines from the 340-replay public-demo dataset."
        )
    )
    parser.add_argument(
        "--replays-root",
        type=Path,
        default=Path("data/replays"),
        help="Root directory containing {game_id}/{guid}.recording.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/human_baselines.json"),
        help="Output JSON fixture path",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a per-game summary to stdout after extraction",
    )
    args = parser.parse_args(argv)

    baselines = extract_baselines(args.replays_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(baselines, f, sort_keys=True, indent=2)
    n_covered_total = sum(
        len(v["baselines"]) for v in baselines.values()
    )
    n_levels_total = sum(v["total_levels"] for v in baselines.values())
    print(
        f"Wrote {len(baselines)} games to {args.out} - "
        f"{n_covered_total}/{n_levels_total} covered levels "
        f"({n_covered_total / max(1, n_levels_total):.0%} RHAE coverage)"
    )
    if args.print_summary:
        print(_summary(baselines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
