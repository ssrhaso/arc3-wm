"""Stub for arc3_wm.replay_loader. Tests are written first (test-first
discipline per CLAUDE.md); implementation lands after Haso reviews the
red test run.

See ``tests/test_replay_loader.py`` for the full schema + alignment
contract this module must satisfy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List, Tuple


class ReplayParseError(Exception):
    """Raised when a JSONL row deviates from the documented schema.

    Message must include the file path and 1-indexed line number so the
    bad row is locatable without reproducing the parser internally
    (per design-decisions.md D5).
    """


def load_replay_file(path: Path) -> Iterator[List[dict]]:
    """Yield one list-of-step-dicts per episode in a single JSONL.

    See ``tests/test_replay_loader.py`` for the per-step schema and
    action-alignment contract (convention B).
    """
    raise NotImplementedError("replay_loader.load_replay_file: stub awaiting impl")


def load_replays_directory(root: Path) -> Iterator[Tuple[str, List[dict]]]:
    """Walk ``root`` recursively, yield ``(game_id, episode)`` pairs.

    ``game_id`` is the parent-folder name. Order is deterministic
    (sorted by path) so tests can assert exact counts.
    """
    raise NotImplementedError("replay_loader.load_replays_directory: stub awaiting impl")
