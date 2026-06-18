"""Tests for arc3_wm.probe_data segmentation + candidate-action logic.

CPU-only. This is the data prep shared by the JAX predict stage and the local
synthetic generator, so getting the window/spec alignment right here is what
lets the GPU stage be a thin forward pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from arc3_wm.action_space import ACTION6_BASE, ACTION7_INDEX
from arc3_wm.palette import PALETTE_RGB
from arc3_wm.probe_data import (
    candidate_actions_for_types,
    iter_episodes,
    make_counterfactual_specs,
    make_rollout_windows,
)


def _frame(val: int) -> np.ndarray:
    return PALETTE_RGB[np.full((4, 4), val % 16, dtype=np.int16)]


def _npz(episodes, has_avail=True):
    """episodes: list of (frames_vals, actions, avail_rows). Returns npz-like dict."""
    frames, actions, ep_id, avail = [], [], [], []
    for e, (vals, acts, av) in enumerate(episodes):
        for i, v in enumerate(vals):
            frames.append(_frame(v))
            actions.append(acts[i])
            ep_id.append(e)
            avail.append(av[i])
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "actions": np.array(actions, dtype=np.int32),
        "ep_id": np.array(ep_id, dtype=np.int32),
        "avail": np.stack(avail).astype(bool),
        "has_avail": np.array(has_avail),
    }


# --- iter_episodes ---------------------------------------------------------

def test_iter_episodes_splits_by_id():
    npz = _npz([
        ([0, 1, 2], [10, 11, 0], [[True] * 7] * 3),
        ([5, 6], [20, 0], [[False] * 7] * 2),
    ])
    eps = list(iter_episodes(npz))
    assert [e.ep_id for e in eps] == [0, 1]
    assert eps[0].frames.shape[0] == 3 and eps[1].frames.shape[0] == 2
    assert eps[0].actions[0] == 10


# --- make_rollout_windows --------------------------------------------------

def test_rollout_window_alignment():
    # L=6, C=2, H=3 -> one window (stride defaults to H); next would need 2+6>6.
    acts = [100, 101, 102, 103, 104, 0]
    npz = _npz([(list(range(6)), acts, [[True] * 7] * 6)])
    w = make_rollout_windows(npz, context_len=2, horizon=3)
    assert len(w) == 1
    win = w[0]
    assert win.start == 2
    assert win.context_frames.shape[0] == 2
    assert list(win.future_actions) == [102, 103, 104]
    # context_last is frame at index 1; true_future are frames 2,3,4.
    assert np.array_equal(win.context_last, _frame(1))
    assert np.array_equal(win.true_future[0], _frame(2))
    assert win.true_future.shape == (3, 4, 4, 3)


def test_rollout_windows_multiple_with_stride():
    # L=9, C=1, H=2, stride=2 -> windows at c=1,3,5,7 (end+H<=9): 1,3,5,7 -> 7+2=9 ok
    acts = list(range(8)) + [0]
    npz = _npz([(list(range(9)), acts, [[True] * 7] * 9)])
    w = make_rollout_windows(npz, context_len=1, horizon=2, stride=2)
    assert [win.start for win in w] == [1, 3, 5, 7]


def test_rollout_windows_context_is_fixed_length():
    # Regression: context must be a fixed C-frame window (not a growing prefix),
    # else the JAX batch is ragged. C=2, H=2, stride=2, L=9 -> starts 2,4,6.
    acts = list(range(8)) + [0]
    npz = _npz([(list(range(9)), acts, [[True] * 7] * 9)])
    w = make_rollout_windows(npz, context_len=2, horizon=2, stride=2)
    assert [win.start for win in w] == [2, 4, 6]
    for win in w:
        assert win.context_frames.shape[0] == 2  # fixed, not growing
    # window at start=4 has context = frames[2:4], context_last = frame 3.
    assert np.array_equal(w[1].context_frames, np.stack([_frame(2), _frame(3)]))
    assert np.array_equal(w[1].context_last, _frame(3))
    assert np.array_equal(w[1].true_future, np.stack([_frame(4), _frame(5)]))


def test_rollout_windows_skips_short_episodes():
    npz = _npz([([0, 1], [5, 0], [[True] * 7] * 2)])  # too short for C+H=4
    assert make_rollout_windows(npz, context_len=2, horizon=2) == []


def test_rollout_windows_max_cap():
    acts = list(range(20)) + [0]
    npz = _npz([(list(range(21)), acts, [[True] * 7] * 21)])
    w = make_rollout_windows(npz, context_len=1, horizon=1, stride=1, max_windows=3)
    assert len(w) == 3


def test_rollout_windows_validates_args():
    npz = _npz([([0, 1, 2], [1, 2, 0], [[True] * 7] * 3)])
    with pytest.raises(ValueError):
        make_rollout_windows(npz, context_len=0, horizon=1)


# --- candidate_actions_for_types -------------------------------------------

def test_candidates_ls20_directional_only():
    # Only ACTION1..4 available (ls20): 4 candidates, taken first, no click.
    avail = np.array([True, True, True, True, False, False, False])
    rng = np.random.default_rng(0)
    c = candidate_actions_for_types(avail, taken=2, n_click=5, rng=rng)
    assert c[0] == 2  # taken first
    assert set(c.tolist()) == {0, 1, 2, 3}
    assert c.shape[0] == 4


def test_candidates_click_game_samples_cells():
    avail = np.array([False, False, False, False, False, True, False])  # ACTION6 only
    rng = np.random.default_rng(1)
    taken = ACTION6_BASE + 100
    c = candidate_actions_for_types(avail, taken=taken, n_click=8, rng=rng)
    assert c[0] == taken
    assert c.shape[0] >= 8  # taken + up to 8 sampled (dedup may drop the dup)
    assert all(ACTION6_BASE <= a <= ACTION7_INDEX for a in c)


def test_candidates_dedup_preserves_taken_first():
    avail = np.array([True, False, False, False, False, False, True])  # A1, A7
    rng = np.random.default_rng(2)
    c = candidate_actions_for_types(avail, taken=0, n_click=0, rng=rng)  # taken == A1
    assert c[0] == 0
    assert (c == 0).sum() == 1  # taken not duplicated by the A1 entry
    assert ACTION7_INDEX in c


# --- make_counterfactual_specs ---------------------------------------------

def test_counterfactual_requires_avail():
    npz = _npz([([0, 1, 2], [1, 2, 0], [[True] * 7] * 3)], has_avail=False)
    assert make_counterfactual_specs(npz, context_len=1, n_click=4, max_specs=None, seed=0) == []


def test_counterfactual_filters_static_transitions():
    # Frames: 0,0,1 -> step0 static (0->0), step1 changes (0->1). require_change
    # keeps only t=1 (but t must be < L-1=2, so t in {0,1}); t=0 static dropped.
    avail = [[True, True, True, True, False, False, False]] * 3
    npz = _npz([([0, 0, 1], [1, 2, 0], avail)])
    specs = make_counterfactual_specs(
        npz, context_len=1, n_click=4, max_specs=None, seed=0, require_change=True
    )
    assert len(specs) == 1
    assert specs[0].t == 1
    assert np.array_equal(specs[0].true_next, _frame(1))
    assert specs[0].taken_idx == 0
    assert specs[0].candidate_actions[0] == 2  # taken action at step 1


def test_counterfactual_context_grows_with_t():
    avail = [[True, True, True, True, False, False, False]] * 4
    npz = _npz([([0, 1, 2, 3], [1, 2, 3, 0], avail)])
    specs = make_counterfactual_specs(
        npz, context_len=2, n_click=0, max_specs=None, seed=0, require_change=False
    )
    # t in [context_len-1 .. L-2] = [1, 2]; contexts length t+1 = 2, 3.
    assert [s.t for s in specs] == [1, 2]
    assert specs[0].context_frames.shape[0] == 2
    assert specs[1].context_frames.shape[0] == 3
