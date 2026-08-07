"""Spec for arc3_wm.trm.data - replay parsing, caching, datasets.

Synthetic-replay tests always run; corpus cross-validation tests skip when
the local replay download (data/replays/public_games-dataset) is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arc3_wm.trm.data import (
    BCDataset,
    STATE_TO_INDEX,
    WMTransitionDataset,
    parse_replay_episodes,
    preprocess_game,
    train_val_split_episodes,
)

CORPUS = Path(__file__).resolve().parent.parent / "data/replays/public_games-dataset"


def _row(action_id, frame_fill=0, state="NOT_FINISHED", levels=0, avail=(1, 2, 3, 4, 5, 6, 7), x=None, y=None):
    data = {"game_id": "test-0"}
    if x is not None:
        data.update(x=x, y=y)
    frame = [[[frame_fill] * 64 for _ in range(64)]]
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "data": {
            "game_id": "test-0",
            "guid": "g",
            "frame": frame,
            "state": state,
            "action_input": {"id": action_id, "data": data, "reasoning": None},
            "available_actions": list(avail),
            "levels_completed": levels,
            "full_reset": False,
            "win_levels": 2,
        },
    }


def _write_replay(path: Path, rows: list[dict], with_summary=True) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        if with_summary:
            fh.write(json.dumps({"timestamp": "t", "data": {"levels_completed": 1, "won": 1, "played": 1, "total_actions": 3, "cards": []}}) + "\n")


@pytest.fixture()
def synthetic_replay(tmp_path):
    rows = [
        _row(0, frame_fill=1),                      # RESET -> frame A
        _row(1, frame_fill=2),                      # ACTION1 -> frame B
        _row(6, frame_fill=3, levels=1, x=7, y=3),  # click -> frame C, level up
        _row("RESET", frame_fill=1),                # mid-session reset (string id)
        _row(2, frame_fill=4, state="WIN", levels=2),
    ]
    game_dir = tmp_path / "testg"
    game_dir.mkdir()
    _write_replay(game_dir / "a.recording.jsonl", rows)
    return tmp_path


def test_parse_episodes_splits_on_reset(synthetic_replay):
    eps = parse_replay_episodes(synthetic_replay / "testg" / "a.recording.jsonl")
    assert len(eps) == 2
    assert len(eps[0]["grids"]) == 3
    assert len(eps[1]["grids"]) == 2
    # Actions: -1 at reset rows; click encodes to 5 + y*64 + x.
    assert eps[0]["actions"].tolist() == [-1, 0, 5 + 3 * 64 + 7]
    assert eps[1]["actions"].tolist() == [-1, 1]
    assert eps[1]["states"].tolist() == [0, STATE_TO_INDEX["WIN"]]
    assert eps[0]["levels"].tolist() == [0, 0, 1]
    # Frames captured from frame[-1].
    assert eps[0]["grids"][0, 0, 0] == 1 and eps[0]["grids"][2, 0, 0] == 3


def test_parse_skips_post_terminal_noise(tmp_path):
    rows = [
        _row(0, frame_fill=1),
        _row(1, frame_fill=2, state="GAME_OVER"),
        _row(2, frame_fill=9),  # post-terminal noise, no reset
    ]
    game_dir = tmp_path / "g2"
    game_dir.mkdir()
    _write_replay(game_dir / "b.recording.jsonl", rows, with_summary=False)
    eps = parse_replay_episodes(game_dir / "b.recording.jsonl")
    assert len(eps) == 1
    assert len(eps[0]["grids"]) == 2
    assert eps[0]["states"].tolist() == [0, STATE_TO_INDEX["GAME_OVER"]]


def test_preprocess_and_datasets(synthetic_replay, tmp_path):
    out = preprocess_game(synthetic_replay, "testg", tmp_path / "cache")
    assert out is not None and out.name == "testg.npz"
    data = np.load(out)
    assert data["grids"].shape == (5, 64, 64)
    assert data["episode_starts"].tolist() == [0, 3]

    wm = WMTransitionDataset([out], dedup=False)
    assert len(wm) == 3  # 2 transitions in ep0, 1 in ep1
    sample = wm[1]
    assert sample["grid"].shape == (64, 64)
    assert sample["reward"].item() == 1.0  # the level-up click
    assert sample["action"].item() == 5 + 3 * 64 + 7
    win_sample = wm[2]
    assert win_sample["state"].item() == STATE_TO_INDEX["WIN"]

    bc = BCDataset([out])
    assert len(bc) == 3
    b0 = bc[0]
    assert b0["mask"].shape == (4102,)
    assert b0["mask"].to(int).sum() > 0


def test_wm_dedup_collapses_identical_transitions(tmp_path):
    rows = [_row(0, frame_fill=1)] + [_row(5, frame_fill=1) for _ in range(10)]
    game_dir = tmp_path / "g3"
    game_dir.mkdir()
    _write_replay(game_dir / "c.recording.jsonl", rows, with_summary=False)
    out = preprocess_game(tmp_path, "g3", tmp_path / "cache")
    assert len(WMTransitionDataset([out], dedup=False)) == 10
    assert len(WMTransitionDataset([out], dedup=True)) == 1


def test_missing_game_returns_none(tmp_path):
    assert preprocess_game(tmp_path, "nope", tmp_path / "cache") is None


def test_train_val_split_is_episode_grouped(synthetic_replay, tmp_path):
    out = preprocess_game(synthetic_replay, "testg", tmp_path / "cache")
    train, val = train_val_split_episodes(out, val_fraction=0.5, seed=0)
    assert set(train.tolist()) | set(val.tolist()) == {0, 1}
    assert not set(train.tolist()) & set(val.tolist())
    wm_train = WMTransitionDataset([out], dedup=False)
    # Episode filter restricts transitions.
    from arc3_wm.trm.data import _GameCache

    cache_all = _GameCache(out)
    cache_val = _GameCache(out, episodes=val)
    assert len(cache_val.transition_idx) < len(cache_all.transition_idx)
    assert len(wm_train) == len(cache_all.transition_idx)


@pytest.mark.skipif(not CORPUS.is_dir(), reason="local replay corpus not downloaded")
def test_cross_validate_against_replay_loader():
    """Same episode boundaries and actions as arc3_wm.replay_loader."""
    from arc3_wm.dynamics_probe import quantize_to_palette
    from arc3_wm.replay_loader import load_replay_file

    path = CORPUS / "vc33" / "7fdb11de-8464-40ae-926a-dcad63592822.recording.jsonl"
    ours = parse_replay_episodes(path)
    theirs = list(load_replay_file(path))
    assert len(ours) == len(theirs)
    for ep_a, ep_b in zip(ours, theirs):
        assert len(ep_a["grids"]) == len(ep_b)
        # replay_loader: action[t] = action chosen AT obs t (from row t+1),
        # sentinel 0 on the last step. Ours: action[t] = action that
        # PRODUCED frame t (-1 on the first row). Alignment check:
        for t in range(len(ep_b) - 1):
            assert int(ep_b[t]["action"]) == int(ep_a["actions"][t + 1])
        # Frames agree through the RGB round-trip.
        for t in (0, len(ep_b) - 1):
            rgb = ep_b[t]["image"]
            assert (quantize_to_palette(rgb) == ep_a["grids"][t]).all()
