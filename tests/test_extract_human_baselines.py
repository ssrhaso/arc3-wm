"""Red-skeleton tests for ``scripts/extract_human_baselines.py``.

The extractor builds the per-game / per-level human-baseline action-count
fixture downstream of ``arc3_wm.rhae.RHAEAggregator``. Per D5 it reuses
``arc3_wm.replay_loader.load_replay_file`` for cn04-safe episode
segmentation rather than re-parsing raw JSONLs.

Per D1 the upper-median rule is methodology.md's "upper of the two middle
entries" - ``sorted(values)[len(values) // 2]`` 0-indexed. (The
colloquial "3rd-place for ~10 testers" framing is wrong and not used.)

Action-counting semantic note (surfaced, not silently chosen): every
non-RESET row in a JSONL is a state-changing engine submission per
methodology.md Section "What Counts as an Action"; the loader already discards
RESET rows (they're episode boundaries) and pads each episode's last
step with a sentinel action that's masked by ``is_last``. So
"state-changing action count for level k" = count of non-sentinel steps
in the episode whose pre-step cumulative reward equals ``k - 1``
(equivalently: count of step-dicts whose post-step cumulative reward is
``k``, since the level-up reward fires WITH the post-level-up obs in
DV3 convention). Whether a no-op-looking action (e.g. ACTION1 pressed
when nothing visibly changes) should count is not resolvable from the
recorded data; this implementation counts every recorded engine
submission, consistent with the only signal we have.

Per-session aggregation: each ``.recording.jsonl`` is one session per
``docs/replay-format.md`` ("Each .recording.jsonl is one human play
session (one guid)"). Per (session, level): take MIN action count
across episodes within the session that cleared that level - captures
the player's best-known attempt at the level, consistent with
methodology.md's "by fewest actions" framing. Each session contributes
at most one entry per level to the upper-median pool.

Output shape per task brief:

    {
      "vc33": {"1": 23, "2": 47, ...},
      "tu93": {"1": 18, ...},
      ...
    }

- JSON-serializable; outer keys are game_id strings; inner keys are
1-indexed level numbers as strings; values are positive ints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from scripts.extract_human_baselines import (
    count_actions_per_level,
    extract_baselines,
    extract_per_session_baselines,
    read_win_levels,
    upper_median,
)


# ---------------------------------------------------------------------------
# Step-dict synthesis helpers - match arc3_wm.replay_loader output schema
# ---------------------------------------------------------------------------


def _make_step(
    *,
    reward: float,
    is_first: bool = False,
    is_last: bool = False,
    is_terminal: bool = False,
    action: int = 1,
) -> dict[str, Any]:
    """Build a single step-dict matching ``load_replay_file`` output.

    Image is a zero placeholder - count_actions_per_level only inspects
    ``reward`` and ``is_last``.
    """
    return {
        "image": np.zeros((64, 64, 3), dtype=np.uint8),
        "action": np.int32(action),
        "reward": np.float32(reward),
        "is_first": np.bool_(is_first),
        "is_last": np.bool_(is_last),
        "is_terminal": np.bool_(is_terminal),
    }


def _episode_with_level_ups(
    *, action_counts: list[int], cleared: list[bool]
) -> List[dict[str, Any]]:
    """Synthesize an episode where the player takes ``action_counts[k]``
    actions while on level ``k+1``. If ``cleared[k]`` is True, a level-up
    reward fires at the boundary into level ``k+2`` (i.e. reward=1 on the
    first step of the NEXT level segment).

    Constraints:
    - ``len(action_counts) == len(cleared)``.
    - The episode is closed with one extra sentinel step (the loader's
      n-th step with ``is_last=True`` and ``action=0`` placeholder).
    - If the player did not clear the final level, that level's segment
      ends with a terminal step (``is_terminal=True``) before the sentinel.
    """
    assert len(action_counts) == len(cleared)
    if not action_counts:
        return [_make_step(reward=0.0, is_first=True, is_last=True)]

    steps: List[dict[str, Any]] = []
    for k, count in enumerate(action_counts):
        for j in range(count):
            # First step of level 1 carries is_first; first step of every
            # subsequent level carries reward=1 (level-up reflected at this
            # obs in DV3 convention).
            is_first = (k == 0 and j == 0)
            reward = 1.0 if (k > 0 and j == 0) else 0.0
            steps.append(_make_step(reward=reward, is_first=is_first))
        if not cleared[k]:
            # Player died on level k+1: mark the last real step on this
            # level as terminal.
            steps[-1]["is_terminal"] = np.bool_(True)
            break
    # Sentinel last step. If the final level WAS cleared, the level-up
    # reward fires here; the sentinel itself doesn't carry real action.
    final_cleared = cleared[-1] if len(cleared) == len(action_counts) else False
    sentinel_reward = 1.0 if final_cleared else 0.0
    steps.append(
        _make_step(
            reward=sentinel_reward,
            is_last=True,
            action=0,  # LAST_STEP_SENTINEL per replay_loader.py
        )
    )
    return steps


# ===========================================================================
# upper_median - methodology.md "upper of two middle entries" (D1)
# ===========================================================================


def test_upper_median_n10_returns_index_5():
    """n=10 -> sorted[5] (0-indexed). Documents D1 outcome for the
    n=~10 testers case used across the 25 ARC-AGI-3 games."""
    # sorted ascending: [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]; idx 5 = 20
    values = [28, 14, 22, 10, 18, 24, 12, 26, 16, 20]
    assert upper_median(values) == 20


def test_upper_median_n9_returns_index_4():
    """n=9 (odd) -> middle element = sorted[4]."""
    values = [50, 10, 30, 20, 40, 70, 60, 90, 80]
    assert upper_median(values) == 50  # sorted[4] of [10,20,30,40,50,60,70,80,90]


def test_upper_median_n2_returns_larger():
    """n=2 (even) -> "upper of two middle entries" = the larger value."""
    assert upper_median([10, 20]) == 20
    assert upper_median([20, 10]) == 20  # input order doesn't matter


def test_upper_median_n1_returns_sole_value():
    """n=1: ``sorted[1 // 2] = sorted[0]`` = the sole value. Documented
    edge case - the upper-median rule degenerates cleanly to "take what
    you have". Real Phase-4 implication: a level cleared by only one
    tester in the public-demo dataset still gets a baseline (with a
    single-completer caveat). Surfaces during Step-3 fixture review
    so we can decide whether to trust it on a per-level basis."""
    assert upper_median([42]) == 42


def test_upper_median_n4_returns_index_2():
    """methodology.md illustrative example: 4 players -> "third place is
    the baseline". 0-indexed that's sorted[2] (= 3rd-from-fewest).
    Sanity-check the rule matches methodology.md's n=4 case."""
    assert upper_median([10, 20, 30, 40]) == 30


def test_upper_median_n5_returns_index_2():
    """methodology.md illustrative example: 5 players -> "third place".
    For odd n, sorted[n//2] is the true middle."""
    assert upper_median([10, 20, 30, 40, 50]) == 30


def test_upper_median_empty_raises():
    """n=0: no median possible. Caller is responsible for filtering empty
    level pools before calling."""
    with pytest.raises(ValueError, match="empty|no values"):
        upper_median([])


def test_upper_median_handles_duplicates():
    """Ties don't change the rule - sort-then-index works directly."""
    assert upper_median([5, 5, 5, 5]) == 5
    assert upper_median([10, 10, 20, 20]) == 20  # sorted[2] = 20


def test_upper_median_does_not_mutate_input():
    """Caller's list must remain unsorted after the call."""
    values = [3, 1, 2]
    _ = upper_median(values)
    assert values == [3, 1, 2]


# ===========================================================================
# count_actions_per_level - per-episode level segmentation
# ===========================================================================


def test_count_actions_empty_episode():
    """Defensive: an empty episode (would come from a zero-row file)
    yields no completed levels."""
    assert count_actions_per_level([]) == {}


def test_count_actions_single_step_sentinel_only():
    """A 1-step episode is just the sentinel - no real action taken,
    no level cleared. Returns empty dict."""
    episode = [_make_step(reward=0.0, is_first=True, is_last=True)]
    assert count_actions_per_level(episode) == {}


def test_count_actions_no_levels_cleared():
    """Player took 5 actions and died on level 1 without clearing it.
    No completed levels -> empty dict. (Distinguishes from level-1 entry
    with action_count=5: only CLEARED levels contribute.)"""
    episode = _episode_with_level_ups(action_counts=[5], cleared=[False])
    assert count_actions_per_level(episode) == {}


def test_count_actions_single_level_cleared():
    """Player took 10 actions to clear level 1, episode ended (no level 2
    attempt). Output: {1: 10}."""
    episode = _episode_with_level_ups(action_counts=[10], cleared=[True])
    assert count_actions_per_level(episode) == {1: 10}


def test_count_actions_multi_level_cleared_then_died():
    """Player cleared levels 1 and 2 (15 + 25 actions), then died on
    level 3 after 5 actions. Output: {1: 15, 2: 25}. Level 3's partial
    count is NOT in the output (only CLEARED levels)."""
    episode = _episode_with_level_ups(
        action_counts=[15, 25, 5], cleared=[True, True, False]
    )
    assert count_actions_per_level(episode) == {1: 15, 2: 25}


def test_count_actions_all_levels_cleared():
    """Player cleared all three levels. Last level's reward fires on the
    sentinel step. Output: {1: 10, 2: 15, 3: 20}."""
    episode = _episode_with_level_ups(
        action_counts=[10, 15, 20], cleared=[True, True, True]
    )
    assert count_actions_per_level(episode) == {1: 10, 2: 15, 3: 20}


def test_count_actions_does_not_count_sentinel_action():
    """The last step's action is a sentinel (placeholder) - must NOT
    contribute to any level's count. Regression guard."""
    # 5-step episode where the player cleared exactly one level. The level-up
    # appears WITH the sentinel step's reward; that sentinel itself adds 0.
    episode = _episode_with_level_ups(action_counts=[5], cleared=[True])
    # Without the sentinel: 5 actions for level 1.
    # If we erroneously counted the sentinel: 6.
    assert count_actions_per_level(episode) == {1: 5}


# ===========================================================================
# extract_per_session_baselines - per-file MIN across episodes per level
# ===========================================================================


def test_per_session_single_episode():
    """One session, one episode clearing level 1 in 12 actions.
    Session's contribution: {1: 12}."""
    ep = _episode_with_level_ups(action_counts=[12], cleared=[True])
    assert extract_per_session_baselines([ep]) == {1: 12}


def test_per_session_min_across_retries():
    """Player retried within the session and got a faster clear on the
    second attempt (25 -> 18 actions). The session's level-1 entry is
    the MIN (18), not the first (25). Justification: methodology.md's
    "by fewest actions" framing favors the player's best attempt."""
    ep1 = _episode_with_level_ups(action_counts=[25], cleared=[True])
    ep2 = _episode_with_level_ups(action_counts=[18], cleared=[True])
    assert extract_per_session_baselines([ep1, ep2]) == {1: 18}


def test_per_session_partial_clears():
    """Player's first attempt cleared level 1 (40 actions) and died on
    level 2; second attempt cleared levels 1+2 (35 + 50). Session's
    contribution: {1: min(40, 35) = 35, 2: 50}. Level 2 has only one
    successful clear so no min between attempts."""
    ep1 = _episode_with_level_ups(action_counts=[40, 5], cleared=[True, False])
    ep2 = _episode_with_level_ups(
        action_counts=[35, 50], cleared=[True, True]
    )
    assert extract_per_session_baselines([ep1, ep2]) == {1: 35, 2: 50}


def test_per_session_no_clears():
    """Player died on level 1 in every attempt - no level contribution
    from this session. Returns empty dict; the session does not
    participate in any per-level pool downstream."""
    ep1 = _episode_with_level_ups(action_counts=[5], cleared=[False])
    ep2 = _episode_with_level_ups(action_counts=[7], cleared=[False])
    assert extract_per_session_baselines([ep1, ep2]) == {}


# ===========================================================================
# extract_baselines - full extraction against a synthetic replays dir
# ===========================================================================


def _write_replay(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a synthetic .recording.jsonl. The extractor delegates parsing
    to load_replay_file so the rows must conform to the JSONL schema in
    docs/replay-format.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _synth_row(
    *,
    action_id: int,
    levels_completed: int,
    state: str = "NOT_FINISHED",
    win_levels: int = 7,
    x: int | None = None,
    y: int | None = None,
) -> dict[str, Any]:
    """One synthetic JSONL row - minimal fields for load_replay_file.

    The frame is a (1, 64, 64) palette-int grid of all zeros (decoded
    by ``decode_frame`` to a black RGB image).
    """
    frame = [[[0] * 64 for _ in range(64)]]
    action_data: dict[str, Any] = {"game_id": "synth"}
    if x is not None:
        action_data["x"] = x
    if y is not None:
        action_data["y"] = y
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "data": {
            "game_id": "synth-0000",
            "frame": frame,
            "state": state,
            "action_input": {
                "id": action_id,
                "data": action_data,
                "reasoning": None,
            },
            "guid": "synth-guid",
            "full_reset": False,
            "available_actions": [1, 2, 3, 4, 5, 6, 7],
            "levels_completed": levels_completed,
            "win_levels": win_levels,
        },
    }


def _synth_session(
    *, action_counts: list[int], cleared: list[bool]
) -> list[dict[str, Any]]:
    """Build a synthetic .recording.jsonl row list mirroring the spec of
    ``_episode_with_level_ups``: ``action_counts[k]`` actions on level
    k+1, plus a level-up if ``cleared[k]``.

    Row layout (DV3 convention (B) per replay_loader):
    - Row 0 is the RESET row (action_input.id == 0), pre-game frame.
    - Row i (i >= 1) is the result of the (i-1)-th real action; its
      ``action_input`` is the (i-1)-th action chosen and its
      ``levels_completed`` reflects the post-action world.
    """
    rows: list[dict[str, Any]] = [_synth_row(action_id=0, levels_completed=0)]
    levels_done = 0
    for k, count in enumerate(action_counts):
        for j in range(count):
            # Each action emits one row: the row records the action_input
            # that produced this row's frame.
            cleared_this_step = (
                cleared[k] and j == count - 1
            )  # last action of level segment, AND the level was cleared
            new_levels = levels_done + (1 if cleared_this_step else 0)
            state = "NOT_FINISHED"
            if (not cleared[k]) and j == count - 1:
                state = "GAME_OVER"  # player died on this action
            elif cleared[k] and j == count - 1 and k == len(action_counts) - 1:
                state = "WIN"  # cleared the final level
            rows.append(
                _synth_row(
                    action_id=1,  # ACTION1
                    levels_completed=new_levels,
                    state=state,
                )
            )
            levels_done = new_levels
        if not cleared[k]:
            break
    return rows


def test_extract_baselines_three_sessions_upper_median(tmp_path: Path):
    """3 sessions clearing level 1 in {10, 20, 30} actions. n=3 >= 2,
    upper_median = sorted[1] = 20. total_levels = 7 (synth default
    win_levels). Output: {"synth": {"total_levels": 7,
    "baselines": {"1": 20}}}."""
    for i, count in enumerate([10, 20, 30]):
        rows = _synth_session(action_counts=[count], cleared=[True])
        _write_replay(
            tmp_path / "synth" / f"sess{i}.recording.jsonl", rows
        )
    out = extract_baselines(tmp_path)
    assert out == {
        "synth": {"total_levels": 7, "baselines": {"1": 20}}
    }


def test_extract_baselines_died_on_level3_contributes_only_1_and_2(
    tmp_path: Path,
):
    """Brief spec: 'Players who died on level 3 contribute to baselines
    for levels 1 and 2 only.' Two sessions:
    - A: cleared 1,2 in {10, 20}, died on level 3 after 5.
    - B: cleared 1,2,3 in {30, 40, 100}.
    Level 1 pool: {10, 30} -> n=2 kept, upper_median=30.
    Level 2 pool: {20, 40} -> n=2 kept, upper_median=40.
    Level 3 pool: {100} -> n=1 DROPPED per D-B."""
    rows_a = _synth_session(
        action_counts=[10, 20, 5], cleared=[True, True, False]
    )
    rows_b = _synth_session(
        action_counts=[30, 40, 100], cleared=[True, True, True]
    )
    _write_replay(tmp_path / "synth" / "a.recording.jsonl", rows_a)
    _write_replay(tmp_path / "synth" / "b.recording.jsonl", rows_b)
    out = extract_baselines(tmp_path)
    assert out == {
        "synth": {
            "total_levels": 7,
            "baselines": {"1": 30, "2": 40},
        }
    }


def test_extract_baselines_multiple_games_isolated(tmp_path: Path):
    """Two game subdirs, each with two sessions clearing level 1.
    Each game's pool is independent - no cross-contamination."""
    for i, count in enumerate([10, 20]):
        rows = _synth_session(action_counts=[count], cleared=[True])
        _write_replay(tmp_path / "vc33" / f"p{i}.recording.jsonl", rows)
    for i, count in enumerate([30, 40]):
        rows = _synth_session(action_counts=[count], cleared=[True])
        _write_replay(tmp_path / "tu93" / f"p{i}.recording.jsonl", rows)
    out = extract_baselines(tmp_path)
    assert out == {
        "vc33": {"total_levels": 7, "baselines": {"1": 20}},
        "tu93": {"total_levels": 7, "baselines": {"1": 40}},
    }


def test_extract_baselines_omits_levels_with_zero_completers(tmp_path: Path):
    """Two sessions cleared level 1, no session ever cleared level 2.
    Output: baselines has only {"1": ...}; level 2 is absent (n=0)."""
    rows_a = _synth_session(action_counts=[10, 5], cleared=[True, False])
    rows_b = _synth_session(action_counts=[20, 7], cleared=[True, False])
    _write_replay(tmp_path / "synth" / "a.recording.jsonl", rows_a)
    _write_replay(tmp_path / "synth" / "b.recording.jsonl", rows_b)
    out = extract_baselines(tmp_path)
    assert "2" not in out["synth"]["baselines"]
    assert "1" in out["synth"]["baselines"]
    assert out["synth"]["total_levels"] == 7


def test_extract_baselines_output_is_json_serializable(tmp_path: Path):
    """Output is JSON-serializable with the new shape. total_levels is
    a top-level int; baselines is a sub-dict of {str: int}."""
    for i, count in enumerate([10, 20]):
        rows = _synth_session(action_counts=[count], cleared=[True])
        _write_replay(tmp_path / "synth" / f"p{i}.recording.jsonl", rows)
    out = extract_baselines(tmp_path)
    encoded = json.dumps(out)
    decoded = json.loads(encoded)
    assert decoded == out
    for game_id, entry in decoded.items():
        assert isinstance(game_id, str)
        assert isinstance(entry["total_levels"], int)
        assert entry["total_levels"] > 0
        for lvl, count in entry["baselines"].items():
            assert isinstance(lvl, str)
            assert isinstance(count, int)
            assert count > 0


def test_extract_baselines_is_deterministic(tmp_path: Path):
    """Same input replays -> identical output (no dict-ordering surprises,
    no random tie-breaking)."""
    for i, count in enumerate([10, 20, 30]):
        rows = _synth_session(action_counts=[count], cleared=[True])
        _write_replay(tmp_path / "synth" / f"p{i}.recording.jsonl", rows)
    out1 = extract_baselines(tmp_path)
    out2 = extract_baselines(tmp_path)
    assert out1 == out2
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)


# ===========================================================================
# Decision A - total_levels comes from JSONL win_levels, not baseline coverage
# ===========================================================================


def test_total_levels_from_win_levels_not_baseline_max(tmp_path: Path):
    """The tu93 motivating case (this session): 9 total levels per JSONL
    win_levels, but humans cleared only level 1 in the synthetic replays.
    total_levels MUST be 9 (from win_levels), not 1 (from max baseline).
    The previous max(baseline_keys) inference is the bug Decision A fixes."""
    rows_a = _synth_session(action_counts=[10, 5], cleared=[True, False])
    rows_b = _synth_session(action_counts=[20, 7], cleared=[True, False])
    for r in rows_a + rows_b:
        r["data"]["win_levels"] = 9  # override the _synth_row default of 7
    _write_replay(tmp_path / "tu93" / "a.recording.jsonl", rows_a)
    _write_replay(tmp_path / "tu93" / "b.recording.jsonl", rows_b)
    out = extract_baselines(tmp_path)
    assert out["tu93"]["total_levels"] == 9
    # And the baseline coverage stops at level 1, not at level 9.
    assert sorted(out["tu93"]["baselines"]) == ["1"]


def test_read_win_levels_returns_constant_per_file(tmp_path: Path):
    """Reads the JSONL row-by-row; returns the single seen value when
    every row agrees (which is the documented per-game-constant case)."""
    rows = _synth_session(action_counts=[5], cleared=[True])
    for r in rows:
        r["data"]["win_levels"] = 9
    p = tmp_path / "p.recording.jsonl"
    _write_replay(p, rows)
    assert read_win_levels(p) == 9


def test_read_win_levels_raises_on_intra_file_mismatch(tmp_path: Path):
    """Defensive: if a single session's rows disagree on win_levels,
    the data is broken - raise rather than silently picking one."""
    rows = _synth_session(action_counts=[5], cleared=[True])
    for i, r in enumerate(rows):
        r["data"]["win_levels"] = 7 if i % 2 == 0 else 9
    p = tmp_path / "p.recording.jsonl"
    _write_replay(p, rows)
    with pytest.raises(RuntimeError, match="not constant"):
        read_win_levels(p)


def test_read_win_levels_filters_cn04_noise_rows(tmp_path: Path):
    """cn04 post-terminal noise rows reset both levels_completed AND
    win_levels to 0 (verified: 12 such rows across all 340 replays, all
    in cn04). The reader must ignore them. Without this filter, the
    intra-file mismatch check fires on every cn04 file that has
    post-terminal noise."""
    rows = _synth_session(action_counts=[5], cleared=[True])
    # Append cn04-style noise: wl=0, levels_completed=0, GAME_OVER.
    rows.append(
        _synth_row(
            action_id=1,
            levels_completed=0,
            state="GAME_OVER",
            win_levels=0,
        )
    )
    p = tmp_path / "p.recording.jsonl"
    _write_replay(p, rows)
    # Real win_levels=7 wins; the wl=0 noise row is silently filtered.
    assert read_win_levels(p) == 7


def test_extract_baselines_win_levels_mismatch_across_sessions_raises(
    tmp_path: Path,
):
    """Haso's "cheap insurance" check: two sessions of the SAME game
    that disagree on win_levels means one was recorded against a
    different game build. Raise loudly so we don't silently mix them."""
    rows_a = _synth_session(action_counts=[5], cleared=[True])
    rows_b = _synth_session(action_counts=[6], cleared=[True])
    for r in rows_b:
        r["data"]["win_levels"] = 8  # disagrees with A's default 7
    _write_replay(tmp_path / "synth" / "a.recording.jsonl", rows_a)
    _write_replay(tmp_path / "synth" / "b.recording.jsonl", rows_b)
    with pytest.raises(RuntimeError, match="win_levels mismatch"):
        extract_baselines(tmp_path)


# ===========================================================================
# Decision B - n >= 2 coverage threshold; uncovered levels excluded
# ===========================================================================


def test_n1_level_dropped_from_baselines(tmp_path: Path):
    """A level with exactly 1 completer is silently dropped from
    ``baselines`` (per D-B). total_levels still reflects the full game."""
    rows = _synth_session(action_counts=[42], cleared=[True])
    _write_replay(tmp_path / "synth" / "p.recording.jsonl", rows)
    out = extract_baselines(tmp_path)
    assert out == {
        "synth": {"total_levels": 7, "baselines": {}}
    }


def test_n1_specific_level_dropped_others_kept(tmp_path: Path):
    """Mixed-coverage game: level 1 has n=2 (kept), level 2 has n=1
    (dropped). Confirms the threshold applies per level, not per game."""
    # Both sessions clear level 1. Only session A clears level 2.
    rows_a = _synth_session(
        action_counts=[10, 20], cleared=[True, True]
    )
    rows_b = _synth_session(
        action_counts=[15, 5], cleared=[True, False]
    )
    _write_replay(tmp_path / "synth" / "a.recording.jsonl", rows_a)
    _write_replay(tmp_path / "synth" / "b.recording.jsonl", rows_b)
    out = extract_baselines(tmp_path)
    # Level 1 pool: {10, 15} -> n=2, upper_median=15.
    # Level 2 pool: {20} -> n=1, DROPPED.
    assert out == {
        "synth": {"total_levels": 7, "baselines": {"1": 15}}
    }


def test_all_uncovered_game_emits_empty_baselines(tmp_path: Path):
    """Edge case: a game where every level has n<2 -> baselines is empty
    but total_levels is still recorded. Downstream RHAEAggregator skips
    the game from the total-score mean (tested separately in Step 3)."""
    rows = _synth_session(action_counts=[10, 20], cleared=[True, True])
    _write_replay(tmp_path / "synth" / "only.recording.jsonl", rows)
    out = extract_baselines(tmp_path)
    assert out == {
        "synth": {"total_levels": 7, "baselines": {}}
    }


def test_extract_baselines_min_completers_threshold_configurable(
    tmp_path: Path,
):
    """min_completers defaults to 2 per D-B but is a parameter; bumping
    to 3 drops more levels. Future-proofs for the 'revisit threshold
    before Phase 5' clause in Notion."""
    # 2 sessions clearing level 1: n=2.
    rows_a = _synth_session(action_counts=[10], cleared=[True])
    rows_b = _synth_session(action_counts=[20], cleared=[True])
    _write_replay(tmp_path / "synth" / "a.recording.jsonl", rows_a)
    _write_replay(tmp_path / "synth" / "b.recording.jsonl", rows_b)
    # Default min_completers=2 -> level 1 kept.
    assert extract_baselines(tmp_path)["synth"]["baselines"] == {"1": 20}
    # min_completers=3 -> level 1 dropped.
    assert extract_baselines(tmp_path, min_completers=3)["synth"]["baselines"] == {}


def test_extract_baselines_rejects_min_completers_below_1():
    """min_completers=0 (or negative) is nonsensical: a "median" over
    zero values is undefined, and upper_median already raises on
    empty input. Surface at the public API instead of upper_median."""
    with pytest.raises(ValueError, match="min_completers"):
        extract_baselines(Path("does_not_matter"), min_completers=0)


# ===========================================================================
# cn04-style noise - extractor inherits load_replay_file's noise handling
# ===========================================================================


def test_extract_baselines_handles_trailing_terminal_noise(tmp_path: Path):
    """load_replay_file discards post-terminal noise rows (cn04 quirk).
    The extractor must inherit that behavior - adding trailing GAME_OVER
    rows after a clean level-1-clear must NOT inflate the level-1 count.
    Two sessions so n=2 keeps the level in baselines per D-B."""
    for i in range(2):
        rows = _synth_session(action_counts=[10], cleared=[True])
        rows.append(
            _synth_row(action_id=1, levels_completed=0, state="GAME_OVER")
        )
        rows.append(
            _synth_row(action_id=1, levels_completed=0, state="GAME_OVER")
        )
        _write_replay(tmp_path / "synth" / f"p{i}.recording.jsonl", rows)
    out = extract_baselines(tmp_path)
    # Both sessions: level 1 in 10 actions each. n=2 -> upper_median = 10.
    # The trailing noise rows did not get counted as level-1 actions
    # (would have inflated the count past 10 if not discarded).
    assert out == {
        "synth": {"total_levels": 7, "baselines": {"1": 10}}
    }
