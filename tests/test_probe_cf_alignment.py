"""Regression test for the counterfactual action-alignment bug.

Probe A (counterfactual action-sensitivity) decodes a one-step prediction under
each candidate action from a shared state. The forward override imagines a single
step whose driving action is ``prevact[:, C] == action[:, C-1]`` -- the *last
context action* slot, not the future slot ``action[:, C]``. The original harness
wrote the candidate into the future slot, so the model never saw it and every
candidate decoded the identical frame (measured ``sensitivity == 0`` on every
game -- an artifact, not an action-blind world model).

These tests pin the contract at the batch level (pure numpy, no JAX), so a
regression that re-breaks the alignment fails on the laptop before any GPU run.
"""
from __future__ import annotations

import numpy as np

from arc3_wm.probe_data import RolloutWindow
from scripts.probe_predict import windows_to_batch

C = 4  # context_len used by the probe pipeline
CAND = 1957  # a candidate action distinct from the real last-context action


def _prevact_at_imagined_step(batch: dict, context_len: int) -> int:
    """Replicate the report override's prevact for the single imagined step.

    ``prevact[:, t] = action[:, t-1]`` (prepend zero); the H=1 imagine consumes
    ``sh(prevact)[:, 0] = prevact[:, C] = action[:, C-1]``.
    """
    action = batch["action"]
    prevact = np.concatenate([np.zeros_like(action[:, :1]), action[:, :-1]], axis=1)
    return int(prevact[0, context_len])


def _make_window(context_actions, future_actions) -> RolloutWindow:
    return RolloutWindow(
        ep_id=0, start=1,
        context_frames=np.zeros((C, 64, 64, 3), np.uint8),
        context_actions=np.asarray(context_actions, np.int32),
        future_actions=np.asarray(future_actions, np.int32),
        true_future=np.zeros((1, 64, 64, 3), np.uint8),
        context_last=np.zeros((64, 64, 3), np.uint8),
    )


def test_fixed_alignment_candidate_drives_imagined_step():
    """The fix: candidate in the last context slot reaches the imagined step."""
    real_last = 99
    w = _make_window([10, 11, 12, real_last][:C - 1] + [CAND], [0])
    batch = windows_to_batch([w], C, 1)
    assert _prevact_at_imagined_step(batch, C) == CAND


def test_buggy_alignment_candidate_is_dropped():
    """Documents the original bug: candidate in the future slot never drives H=1."""
    real_last = 99
    w = _make_window([10, 11, 12, real_last], [CAND])  # candidate in future slot
    batch = windows_to_batch([w], C, 1)
    driver = _prevact_at_imagined_step(batch, C)
    assert driver == real_last  # the REAL action drives it, not the candidate
    assert driver != CAND       # candidate is silently ignored at H=1


def test_distinct_candidates_yield_distinct_drivers_after_fix():
    """Two different candidates must produce two different imagined-step drivers."""
    drivers = set()
    for cand in (0, 1, 2, 3, CAND):
        w = _make_window([10, 11, 12, cand], [0])
        drivers.add(_prevact_at_imagined_step(windows_to_batch([w], C, 1), C))
    assert drivers == {0, 1, 2, 3, CAND}
