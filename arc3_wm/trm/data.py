"""Replay corpus -> tensors for TRM training.

Parses the human-replay ``*.recording.jsonl`` corpus directly to palette
grids (no RGB round-trip) and keeps fields the DreamerV3-oriented
``replay_loader`` discards: the per-step terminal state (WIN vs GAME_OVER)
and ``available_actions`` (the action mask at decision time). Episode
segmentation follows the same rules as ``arc3_wm.replay_loader`` (RESET rows
split episodes; the first terminal row ends one; trailing summary rows are
skipped) and is cross-validated against it in tests.

Cache layout: one ``<game_id>.npz`` per game with flat per-frame arrays
plus episode start offsets; transition/BC samples are derived on load.

numpy-only at module level; torch is imported inside the Dataset classes so
the preprocessing path stays laptop-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..action_space import GameAction, arc_to_flat, build_mask

STATE_TO_INDEX = {"NOT_FINISHED": 0, "WIN": 1, "GAME_OVER": 2}


def _coerce_action_id(raw) -> int:
    """Accept both serialisations: int value or enum-name string (D5)."""
    if isinstance(raw, bool):
        raise ValueError("bool action id")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        return GameAction[raw].value
    raise ValueError(f"bad action id {raw!r}")


def _row_to_flat_action(action_input: dict) -> int:
    """action_input -> flat 0..4101 index; RESET (id 0) -> -1."""
    aid = _coerce_action_id(action_input["id"])
    if aid == 0:
        return -1
    action = GameAction.from_id(aid)
    data = action_input.get("data") or {}
    if aid == 6:
        return arc_to_flat(action, x=int(data["x"]), y=int(data["y"]))
    return arc_to_flat(action)


def iter_replay_rows(path: Path) -> Iterator[dict]:
    """Yield per-step ``data`` payloads, skipping the trailing summary row."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)["data"]
            if "frame" not in data or "action_input" not in data:
                continue  # session-summary footer
            yield data


def parse_replay_episodes(path: Path) -> list[dict[str, np.ndarray]]:
    """One replay file -> list of episodes of aligned per-frame arrays.

    Per episode: ``grids`` uint8 [T,64,64] (frame[-1], settled state),
    ``actions`` int16 [T] (flat action that PRODUCED frame t; -1 at t=0),
    ``levels`` int16 [T], ``states`` int8 [T] (0/1/2), ``avail`` uint8 [T,7]
    (available action types 1..7 at frame t, i.e. at decision time for the
    next action).
    """
    episodes: list[dict[str, np.ndarray]] = []
    current: list[dict] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 1:
            episodes.append(_stack_episode(current))
        current = []

    terminal_seen = False
    for row in iter_replay_rows(path):
        flat = _row_to_flat_action(row["action_input"])
        if flat == -1 and current:
            flush()
            terminal_seen = False
        if terminal_seen:
            continue  # post-terminal bookkeeping noise
        frame = row["frame"]
        if not frame:
            continue
        layer = np.asarray(frame[-1], dtype=np.uint8)
        if layer.shape != (64, 64):
            raise ValueError(f"bad frame shape {layer.shape} in {path.name}")
        state = STATE_TO_INDEX.get(row.get("state", "NOT_FINISHED"), 0)
        avail = np.zeros(7, dtype=np.uint8)
        for a in row.get("available_actions", []):
            if 1 <= int(a) <= 7:
                avail[int(a) - 1] = 1
        current.append(
            {
                "grid": layer,
                "action": flat,
                "level": int(row.get("levels_completed", 0)),
                "state": state,
                "avail": avail,
            }
        )
        if state != 0:
            terminal_seen = True
    flush()
    # Drop degenerate episodes (a lone RESET row has no transition).
    return [ep for ep in episodes if len(ep["grids"]) >= 2]


def _stack_episode(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "grids": np.stack([r["grid"] for r in rows]),
        "actions": np.asarray([r["action"] for r in rows], dtype=np.int16),
        "levels": np.asarray([r["level"] for r in rows], dtype=np.int16),
        "states": np.asarray([r["state"] for r in rows], dtype=np.int8),
        "avail": np.stack([r["avail"] for r in rows]).astype(np.uint8),
    }


def preprocess_game(
    replays_root: Path, game_id: str, out_dir: Path
) -> Optional[Path]:
    """All replays of one game -> ``out_dir/<game_id>.npz``. Returns the
    written path, or None if the game has no replay files."""
    game_dir = replays_root / game_id
    files = sorted(game_dir.glob("*.recording.jsonl")) if game_dir.is_dir() else []
    if not files:
        return None
    episodes: list[dict[str, np.ndarray]] = []
    for path in files:
        episodes.extend(parse_replay_episodes(path))
    if not episodes:
        return None
    starts = np.cumsum([0] + [len(ep["grids"]) for ep in episodes[:-1]]).astype(np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{game_id}.npz"
    np.savez_compressed(
        out,
        grids=np.concatenate([ep["grids"] for ep in episodes]),
        actions=np.concatenate([ep["actions"] for ep in episodes]),
        levels=np.concatenate([ep["levels"] for ep in episodes]),
        states=np.concatenate([ep["states"] for ep in episodes]),
        avail=np.concatenate([ep["avail"] for ep in episodes]),
        episode_starts=starts,
    )
    return out


class _GameCache:
    """Loaded npz + derived transition index.

    ``episodes`` restricts the transition index to those episode ids (the
    train/val split from ``train_val_split_episodes``); None keeps all.
    """

    def __init__(self, path: Path, episodes: Optional[np.ndarray] = None) -> None:
        data = np.load(path)
        self.grids = data["grids"]
        self.actions = data["actions"]
        self.levels = data["levels"]
        self.states = data["states"]
        self.avail = data["avail"]
        starts = data["episode_starts"]
        n = len(self.grids)
        ends = np.append(starts[1:], n)
        # Transition t -> t+1 exists for every t whose successor is in the
        # same episode; actions[t+1] is the action taken at frame t.
        keep = np.ones(n, dtype=bool)
        keep[ends - 1] = False
        if episodes is not None:
            in_split = np.zeros(n, dtype=bool)
            for ep in episodes:
                in_split[starts[ep] : ends[ep]] = True
            keep &= in_split
        self.transition_idx = np.nonzero(keep)[0]
        # Guard: the successor row of a kept index must carry a real action.
        assert (self.actions[self.transition_idx + 1] >= 0).all()


Spec = "Path | tuple[Path, Optional[np.ndarray]]"


def _load_caches(specs) -> list[_GameCache]:
    """Accept paths or (path, episode_ids) pairs (the train/val split)."""
    caches = []
    for spec in specs:
        if isinstance(spec, tuple):
            path, episodes = spec
        else:
            path, episodes = spec, None
        caches.append(_GameCache(Path(path), episodes=episodes))
    return caches


class WMTransitionDataset:
    """(grid, action) -> (next_grid, reward, state) transition samples.

    ``specs``: npz paths, or (path, episode_ids) pairs to restrict to a
    train/val episode split. ``dedup=True`` collapses identical
    (grid, action, next_grid) triples, which shrinks the heavily-static
    corpus and reweights rare transitions upward - the copy-degeneracy
    countermeasure at the data level.
    """

    def __init__(self, specs: list, dedup: bool = True) -> None:
        import torch  # noqa: F401  (deferred; keeps preprocessing torch-free)

        self._caches = _load_caches(specs)
        self._index: list[tuple[int, int]] = []
        seen: set[bytes] = set()
        for ci, cache in enumerate(self._caches):
            for t in cache.transition_idx:
                if dedup:
                    key = (
                        cache.grids[t].tobytes()
                        + int(cache.actions[t + 1]).to_bytes(2, "little", signed=True)
                        + cache.grids[t + 1].tobytes()
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                self._index.append((ci, int(t)))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        import torch

        ci, t = self._index[i]
        c = self._caches[ci]
        reward = float(c.levels[t + 1] - c.levels[t] > 0)
        return {
            "grid": torch.from_numpy(c.grids[t].astype(np.int64)),
            "action": torch.tensor(int(c.actions[t + 1]), dtype=torch.long),
            "next_grid": torch.from_numpy(c.grids[t + 1].astype(np.int64)),
            "reward": torch.tensor(reward),
            "state": torch.tensor(int(c.states[t + 1]), dtype=torch.long),
        }


class BCDataset:
    """(grid, mask) -> human action samples for behaviour cloning.

    ``specs`` as in WMTransitionDataset: paths or (path, episode_ids).
    """

    def __init__(self, specs: list) -> None:
        import torch  # noqa: F401

        self._caches = _load_caches(specs)
        self._index = [
            (ci, int(t))
            for ci, cache in enumerate(self._caches)
            for t in cache.transition_idx
        ]
        self._mask_cache: dict[bytes, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _mask(self, avail_row: np.ndarray) -> np.ndarray:
        key = avail_row.tobytes()
        mask = self._mask_cache.get(key)
        if mask is None:
            types = [i + 1 for i in range(7) if avail_row[i]]
            mask = build_mask(types)
            self._mask_cache[key] = mask
        return mask

    def __getitem__(self, i: int) -> dict:
        import torch

        ci, t = self._index[i]
        c = self._caches[ci]
        return {
            "grid": torch.from_numpy(c.grids[t].astype(np.int64)),
            "action": torch.tensor(int(c.actions[t + 1]), dtype=torch.long),
            "mask": torch.from_numpy(self._mask(c.avail[t]).copy()),
        }


def train_val_split_episodes(
    npz_path: Path, val_fraction: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Episode-grouped split: returns (train_episode_ids, val_episode_ids).

    Splitting frames within an episode would leak near-duplicate frames
    across the split; whole episodes only.
    """
    data = np.load(npz_path)
    n_eps = len(data["episode_starts"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_eps)
    n_val = max(1, int(round(n_eps * val_fraction))) if n_eps > 1 else 0
    return np.sort(order[n_val:]), np.sort(order[:n_val])
