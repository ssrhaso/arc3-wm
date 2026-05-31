"""Load human-replay JSONLs into DreamerV3 buffer step dicts.

See ``docs/replay-format.md`` for the JSONL schema and
``docs/design-decisions.md`` (D4, D5) for reward and id-parsing decisions.

Per-step output schema (matches ``ARC3EmbodiedEnv._pack`` plus ``action``):

    image:       np.uint8  (64, 64, 3)  - frame[-1] palette-decoded
    action:      np.int32  ()           - flat index in [0, 4102)
    reward:      np.float32 ()          - delta levels_completed
    is_first:    np.bool_  ()
    is_last:     np.bool_  ()
    is_terminal: np.bool_  ()

Action alignment is **convention (B)**: ``action[t] = flat(action_input on
row t+1)`` - the action chosen *at* obs[t]. The last step of every
episode uses sentinel ``action = 0`` (masked by ``is_last`` downstream).

Episode boundaries (verified by 39-file empirical scan, 122 transitions):
- A row with ``action_input.id == 0`` (or ``"RESET"``) after line 0 ends
  the previous episode and starts a new one (the RESET row itself is the
  first row of the new episode - its frame is the post-reset obs).
- An episode ends at the **first** terminal-state row (state in {WIN,
  GAME_OVER}). Subsequent rows while pending's last row is terminal are
  **post-terminal bookkeeping noise** that the engine emits between the
  player's death/win and the explicit RESET they hit several frames
  later (this is where the cn04 levels_completed drops live: noise
  rows, not implicit restarts). Those rows are discarded; the noise
  count is exposed via the optional ``stats`` arg for observability.
- Two symmetric tripwires guard the assumption boundaries:
  1. A ``levels_completed`` decrease while pending's last row is
     ``NOT_FINISHED`` raises ``ReplayParseError`` - drops should only
     ever happen post-terminal.
  2. A ``NOT_FINISHED`` row that follows a terminal row with no
     intervening explicit RESET raises ``ReplayParseError`` - the
     39-file survey shows every terminal->non-terminal transition goes
     through an explicit RESET, so anything else is unobserved engine
     behaviour worth surfacing for review.
- EOF closes the current episode.
- A file containing only a RESET row (+ optional summary) yields zero
  episodes - no phantom 1-step episode.

Errors:
- ``ReplayParseError`` is raised for any schema deviation, with the
  file path and 1-indexed line number in the message.
- ``full_reset=True`` triggers a ``UserWarning`` but does not split an
  episode (D5: surface unverified semantics, don't silently re-interpret).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
from arcengine import GameAction

from .action_space import arc_to_flat
from .palette import decode_frame

OBS_HW = 64
RESET_ID_INT = 0
RESET_ID_STR = "RESET"
TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})
LAST_STEP_SENTINEL = 0  # = ACTION1 in flat space; masked by is_last downstream


class ReplayParseError(Exception):
    """JSONL row deviates from the documented schema. Message includes
    the file path and 1-indexed line number (D5)."""


# ---------------------------------------------------------------------------
# Row-level helpers
# ---------------------------------------------------------------------------


def _parse_lines(path: Path) -> Iterator[Tuple[int, dict[str, Any]]]:
    """Yield ``(1-indexed line_no, parsed_obj)`` for non-blank lines."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ReplayParseError(
                    f"{path}:line {line_no}: malformed JSON: {e}"
                ) from e


def _classify_row(data: dict[str, Any]) -> str:
    """Return one of ``"step"``, ``"summary"``, or ``"malformed"``."""
    has_frame = "frame" in data
    has_action_input = "action_input" in data
    if has_frame and has_action_input:
        return "step"
    if not has_frame and not has_action_input:
        return "summary"
    return "malformed"


def _is_reset_id(raw_id: Any) -> bool:
    return bool(raw_id == RESET_ID_INT or raw_id == RESET_ID_STR)


# ---------------------------------------------------------------------------
# Action mapping - convention (B), accepts int or string ids (D5)
# ---------------------------------------------------------------------------


def _resolve_game_action(
    raw_id: Any, *, path: Path, line_no: int
) -> GameAction:
    if isinstance(raw_id, str):
        try:
            return GameAction.from_name(raw_id)
        except ValueError as e:
            raise ReplayParseError(
                f"{path}:line {line_no}: unknown action_input.id name {raw_id!r}"
            ) from e
    if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)):
        # bool is a subclass of int in Python - reject explicitly so a
        # rogue True/False doesn't silently map to ACTION1.
        raise ReplayParseError(
            f"{path}:line {line_no}: action_input.id must be int or string, "
            f"got {type(raw_id).__name__}"
        )
    try:
        return GameAction.from_id(int(raw_id))
    except ValueError as e:
        raise ReplayParseError(
            f"{path}:line {line_no}: unknown action_input.id integer {raw_id}"
        ) from e


def _flat_action_or_none(
    action_input: dict[str, Any], *, path: Path, line_no: int
) -> Optional[int]:
    """Map ``action_input`` to a flat index, or ``None`` if RESET.

    RESET is not in the flat action space - callers substitute the
    last-step sentinel (0) when this returns None.
    """
    raw_id = action_input.get("id")
    if _is_reset_id(raw_id):
        return None
    ga = _resolve_game_action(raw_id, path=path, line_no=line_no)
    if ga == GameAction.RESET:
        return None  # defensive: from_name("reset") would also land here
    data = action_input.get("data") or {}
    if ga == GameAction.ACTION6:
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            raise ReplayParseError(
                f"{path}:line {line_no}: ACTION6 missing x/y in data {data!r}"
            )
        try:
            return arc_to_flat(ga, x=int(x), y=int(y))
        except (ValueError, TypeError) as e:
            raise ReplayParseError(
                f"{path}:line {line_no}: ACTION6 bad x/y: {e}"
            ) from e
    try:
        return arc_to_flat(ga)
    except ValueError as e:
        raise ReplayParseError(
            f"{path}:line {line_no}: cannot map {ga} to flat index: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------


def _decode_image(frame: Any, *, path: Path, line_no: int) -> np.ndarray:
    if not frame:
        # Engine should never produce an empty frame on a step row, but
        # mirror env.py's defensive zero-fill rather than crash.
        return np.zeros((OBS_HW, OBS_HW, 3), dtype=np.uint8)
    layer = np.asarray(frame[-1])
    if layer.ndim != 2 or layer.shape != (OBS_HW, OBS_HW):
        raise ReplayParseError(
            f"{path}:line {line_no}: unexpected frame layer shape {layer.shape}; "
            f"expected ({OBS_HW}, {OBS_HW})"
        )
    try:
        return decode_frame(layer)
    except ValueError as e:
        raise ReplayParseError(
            f"{path}:line {line_no}: palette decode failed: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Episode building
# ---------------------------------------------------------------------------


def _build_episode(
    rows: List[Tuple[int, dict[str, Any]]], *, path: Path
) -> List[dict[str, Any]]:
    """Turn a list of ``(line_no, data)`` step rows into step dicts.

    Action alignment is convention (B): step ``i`` stores the action
    chosen *at* obs[i], i.e. ``flat(action_input on row i+1)``. The last
    step of the episode (no successor row) gets the sentinel.
    """
    if not rows:
        return []
    first_data = rows[0][1]
    prev_levels = int(first_data.get("levels_completed", 0))
    n = len(rows)
    episode: List[dict[str, Any]] = []
    for i, (line_no, data) in enumerate(rows):
        image = _decode_image(data.get("frame"), path=path, line_no=line_no)
        levels = int(data.get("levels_completed", prev_levels))
        reward = float(levels - prev_levels)
        prev_levels = levels

        state = data.get("state", "NOT_FINISHED")
        is_terminal = state in TERMINAL_STATES
        is_first = i == 0
        is_last = i == n - 1

        if i + 1 < n:
            next_line_no, next_data = rows[i + 1]
            next_ai = next_data.get("action_input") or {}
            flat = _flat_action_or_none(
                next_ai, path=path, line_no=next_line_no
            )
            action = LAST_STEP_SENTINEL if flat is None else int(flat)
        else:
            action = LAST_STEP_SENTINEL

        episode.append(
            {
                "image": image,
                "action": np.int32(action),
                "reward": np.float32(reward),
                "is_first": np.bool_(is_first),
                "is_last": np.bool_(is_last),
                "is_terminal": np.bool_(is_terminal),
            }
        )
    return episode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_replay_file(
    path: Path, *, stats: Optional[dict[str, Any]] = None
) -> Iterator[List[dict[str, Any]]]:
    """Yield one list-of-step-dicts per episode in ``path``.

    If ``stats`` is provided, the loader updates its
    ``"noise_rows_discarded"`` counter as it goes - useful for spotting
    files where the post-terminal noise block is unusually long.
    """
    path = Path(path)
    pending: List[Tuple[int, dict[str, Any]]] = []

    def _bump_noise() -> None:
        if stats is not None:
            stats["noise_rows_discarded"] = (
                stats.get("noise_rows_discarded", 0) + 1
            )

    for line_no, obj in _parse_lines(path):
        data = obj.get("data") or {}
        kind = _classify_row(data)
        if kind == "summary":
            continue
        if kind == "malformed":
            raise ReplayParseError(
                f"{path}:line {line_no}: row missing frame or action_input "
                f"(neither summary nor step shape)"
            )

        if data.get("full_reset"):
            warnings.warn(
                f"{path}: full_reset=True on line {line_no}; informational "
                f"only - episode boundaries unaffected",
                UserWarning,
                stacklevel=2,
            )

        action_input = data.get("action_input") or {}
        raw_id = action_input.get("id")
        is_explicit_reset = _is_reset_id(raw_id)
        state = data.get("state", "NOT_FINISHED")

        # Post-terminal mode: the previous episode is already closed at
        # pending[-1] (a terminal row). Three legal next-row shapes:
        # (a) explicit RESET - flush pending, this row starts the new ep.
        # (b) another terminal row - post-terminal noise, discard.
        # (c) NOT_FINISHED row without RESET - symmetric tripwire raise.
        if pending and pending[-1][1].get("state") in TERMINAL_STATES:
            if is_explicit_reset:
                ep = _build_episode(pending, path=path)
                if ep:
                    yield ep
                pending = [(line_no, data)]
                continue
            if state in TERMINAL_STATES:
                _bump_noise()
                continue
            raise ReplayParseError(
                f"{path}:line {line_no}: NOT_FINISHED row follows terminal "
                f"block with no intervening RESET (possible engine "
                f"auto-restart; unobserved in 39-file survey, surfacing "
                f"for review per design-decisions D5)"
            )

        # Not in post-terminal mode. pending[-1] (if any) is NOT_FINISHED,
        # so a levels_completed drop here is the dangerous case.
        if pending and not is_explicit_reset:
            prev_data = pending[-1][1]
            prev_levels = int(prev_data.get("levels_completed", 0))
            curr_levels = int(data.get("levels_completed", prev_levels))
            if curr_levels < prev_levels:
                prev_state = prev_data.get("state", "NOT_FINISHED")
                raise ReplayParseError(
                    f"{path}:line {line_no}: levels_completed dropped "
                    f"{prev_levels}->{curr_levels} with prev_state="
                    f"{prev_state!r} (expected WIN|GAME_OVER); see "
                    f"design-decisions for the post-terminal-noise rule"
                )

        if is_explicit_reset and pending:
            ep = _build_episode(pending, path=path)
            if ep:
                yield ep
            pending = []
        pending.append((line_no, data))

    # End-of-file flush. A lone RESET row at EOF (no actions taken) yields
    # zero episodes - Q5b edge case.
    if pending:
        if len(pending) == 1:
            sole_ai = pending[0][1].get("action_input") or {}
            if _is_reset_id(sole_ai.get("id")):
                return
        ep = _build_episode(pending, path=path)
        if ep:
            yield ep


def load_replays_directory(
    root: Path, *, stats: Optional[dict[str, Any]] = None
) -> Iterator[Tuple[str, List[dict[str, Any]]]]:
    """Walk ``root`` recursively, yield ``(game_id, episode)`` pairs.

    ``game_id`` is the parent-folder name. Iteration order is
    deterministic: files are sorted by full path. If ``stats`` is
    provided, the same dict is threaded into every per-file loader so
    counters accumulate across the walk.
    """
    root = Path(root)
    for p in sorted(root.rglob("*.recording.jsonl")):
        game_id = p.parent.name
        for ep in load_replay_file(p, stats=stats):
            yield game_id, ep
