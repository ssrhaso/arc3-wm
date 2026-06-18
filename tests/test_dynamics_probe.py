"""Tests for arc3_wm.dynamics_probe metric primitives (Phase-6 probes).

CPU-only, JAX-free: these exercise the frame-comparison maths that the probe
pipeline's score stage relies on. The forward-pass stage that *produces*
predicted frames is tested separately on a GPU box.
"""
from __future__ import annotations

import numpy as np
import pytest

from arc3_wm.dynamics_probe import (
    action_discrimination,
    action_sensitivity,
    cell_accuracy,
    changed_fraction,
    frame_mse,
    quantize_to_palette,
    score_counterfactual,
    score_rollout,
)
from arc3_wm.palette import PALETTE_RGB, decode_frame

H = W = 8


def _frame_from_indices(idx: np.ndarray) -> np.ndarray:
    return decode_frame(idx.astype(np.int16))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- quantize_to_palette ---------------------------------------------------

def test_quantize_exact_palette_roundtrip():
    # Every exact palette colour must quantise back to its own index.
    idx = np.arange(16, dtype=np.int16).reshape(4, 4)
    rgb = decode_frame(idx)
    assert np.array_equal(quantize_to_palette(rgb), idx)


def test_quantize_nearest_for_perturbed_colour():
    # A small perturbation off an exact palette colour still snaps to it.
    idx = np.array([[6, 9], [14, 0]], dtype=np.int16)
    rgb = decode_frame(idx).astype(np.float32) + 3.0  # within nearest-colour basin
    assert np.array_equal(quantize_to_palette(rgb), idx)


def test_quantize_rejects_non_rgb_last_axis():
    with pytest.raises(ValueError):
        quantize_to_palette(np.zeros((4, 4, 2), dtype=np.float32))


def test_quantize_handles_float_continuous_input():
    # Decoder output is continuous; halfway-ish between two greys resolves to one.
    rgb = np.full((1, 1, 3), 0xCB, dtype=np.float32)  # ~ off-white (0xCC)
    assert quantize_to_palette(rgb)[0, 0] == 1


# --- cell_accuracy ---------------------------------------------------------

def test_cell_accuracy_identical_is_one():
    f = _frame_from_indices(_rng(0).integers(0, 16, (H, W)))
    assert cell_accuracy(f, f) == 1.0


def test_cell_accuracy_half_wrong():
    idx = np.zeros((2, 4), dtype=np.int16)  # 8 cells
    true = _frame_from_indices(idx)
    pred_idx = idx.copy()
    pred_idx[0, :] = 8  # flip the 4-cell top row
    pred = _frame_from_indices(pred_idx)
    assert cell_accuracy(pred, true) == pytest.approx(0.5)


def test_cell_accuracy_tolerates_decoder_noise():
    # Continuous prediction near the right colours still scores 1.0 after quantise.
    idx = _rng(1).integers(0, 16, (H, W))
    true = _frame_from_indices(idx)
    pred = true.astype(np.float32) + _rng(2).uniform(-4, 4, true.shape)
    assert cell_accuracy(pred, true) == 1.0


def test_cell_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cell_accuracy(np.zeros((4, 4, 3)), np.zeros((5, 4, 3)))


# --- changed_fraction ------------------------------------------------------

def test_changed_fraction_zero_when_static():
    f = _frame_from_indices(_rng(3).integers(0, 16, (H, W)))
    assert changed_fraction(f, f) == 0.0


def test_changed_fraction_counts_moved_cells():
    idx = np.zeros((4, 4), dtype=np.int16)
    a = _frame_from_indices(idx)
    b_idx = idx.copy()
    b_idx[0, 0] = 8  # 1 of 16 cells differs
    b = _frame_from_indices(b_idx)
    assert changed_fraction(a, b) == pytest.approx(1 / 16)


# --- frame_mse -------------------------------------------------------------

def test_frame_mse_zero_identical():
    f = _frame_from_indices(_rng(4).integers(0, 16, (H, W)))
    assert frame_mse(f, f) == 0.0


def test_frame_mse_positive_when_different():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.full((2, 2, 3), 10, dtype=np.uint8)
    assert frame_mse(a, b) == pytest.approx(100.0)


# --- action_sensitivity ----------------------------------------------------

def test_action_sensitivity_zero_when_action_blind():
    # Same prediction for every action == action-blind == copying.
    one = _frame_from_indices(_rng(5).integers(0, 16, (H, W)))
    preds = np.stack([one, one, one, one])
    assert action_sensitivity(preds) == 0.0


def test_action_sensitivity_full_when_every_cell_depends_on_action():
    base = np.zeros((H, W), dtype=np.int16)
    preds = np.stack([_frame_from_indices(base + k) for k in range(4)])  # all cells differ
    assert action_sensitivity(preds) == 1.0


def test_action_sensitivity_partial():
    base = _rng(6).integers(0, 16, (H, W)).astype(np.int16)
    p0 = _frame_from_indices(base)
    alt = base.copy()
    alt[0, 0] = (alt[0, 0] + 1) % 16  # exactly one cell action-dependent
    p1 = _frame_from_indices(alt)
    assert action_sensitivity(np.stack([p0, p1])) == pytest.approx(1 / (H * W))


def test_action_sensitivity_needs_two_actions():
    one = _frame_from_indices(np.zeros((H, W), dtype=np.int16))
    with pytest.raises(ValueError):
        action_sensitivity(one[None])


def test_action_sensitivity_rejects_bad_shape():
    with pytest.raises(ValueError):
        action_sensitivity(np.zeros((4, 8, 8)))  # missing RGB axis


# --- action_discrimination -------------------------------------------------

def test_action_discrimination_taken_is_best():
    true_idx = _rng(7).integers(0, 16, (H, W)).astype(np.int16)
    true = _frame_from_indices(true_idx)
    taken_pred = true  # taken action predicts truth exactly
    wrong = _frame_from_indices((true_idx + 5) % 16)  # all cells wrong
    preds = np.stack([wrong, taken_pred, wrong])
    out = action_discrimination(preds, true, taken_idx=1)
    assert out["taken_acc"] == 1.0
    assert out["rank"] == 0
    assert out["mean_other_acc"] < 1.0


def test_action_discrimination_beats_copy_flag():
    # context (copy) differs from truth in the cell the taken action fixes.
    ctx_idx = np.zeros((H, W), dtype=np.int16)
    true_idx = ctx_idx.copy()
    true_idx[0, 0] = 8  # the move flips one cell
    true = _frame_from_indices(true_idx)
    taken_pred = true  # model nails it
    other = _frame_from_indices(ctx_idx)  # = copy
    preds = np.stack([taken_pred, other])
    out = action_discrimination(
        preds, true, taken_idx=0, context_rgb=_frame_from_indices(ctx_idx)
    )
    assert out["beats_copy"] is True
    assert out["copy_acc"] == pytest.approx((H * W - 1) / (H * W))


def test_action_discrimination_no_context_returns_none_flags():
    true = _frame_from_indices(np.zeros((H, W), dtype=np.int16))
    preds = np.stack([true, true])
    out = action_discrimination(preds, true, taken_idx=0)
    assert out["copy_acc"] is None and out["beats_copy"] is None


def test_action_discrimination_taken_idx_out_of_range():
    true = _frame_from_indices(np.zeros((H, W), dtype=np.int16))
    preds = np.stack([true, true])
    with pytest.raises(ValueError):
        action_discrimination(preds, true, taken_idx=5)


# --- score_rollout (batched, Stage 3) --------------------------------------

def test_score_rollout_perfect_model_beats_copy_on_moving_board():
    # A board that changes every step: perfect model scores 1.0, copy decays.
    N, Hh = 3, 4
    rng = _rng(10)
    true_idx = np.stack([
        np.stack([rng.integers(0, 16, (H, W)) for _ in range(Hh)]) for _ in range(N)
    ]).astype(np.int16)  # (N, H, H, W)
    true = PALETTE_RGB[true_idx]
    pred = true.copy()  # perfect prediction
    ctx = PALETTE_RGB[rng.integers(0, 16, (N, H, W)).astype(np.int16)]
    out = score_rollout(pred, true, ctx)
    assert out["model_acc"].shape == (Hh,)
    assert np.allclose(out["model_acc"], 1.0)
    assert (out["model_acc"] >= out["copy_acc"]).all()
    assert out["n"] == N
    assert np.allclose(out["mse"], 0.0)


def test_score_rollout_static_board_copy_ties_model():
    # Static board: copy is perfect, changed==0, model (perfect) ties it.
    N, Hh = 2, 3
    base = _rng(11).integers(0, 16, (H, W)).astype(np.int16)
    frame = PALETTE_RGB[base]
    true = np.broadcast_to(frame, (N, Hh, H, W, 3)).copy()
    pred = true.copy()
    ctx = np.broadcast_to(frame, (N, H, W, 3)).copy()
    out = score_rollout(pred, true, ctx)
    assert np.allclose(out["copy_acc"], 1.0)
    assert np.allclose(out["changed"], 0.0)


def test_score_rollout_shape_validation():
    bad = np.zeros((2, 3, H, W, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        score_rollout(bad, np.zeros((2, 3, H, W, 2)), np.zeros((2, H, W, 3)))


# --- score_counterfactual (batched, Stage 3) -------------------------------

def test_score_counterfactual_competent_model():
    # Taken action predicts truth; alternatives differ and are wrong.
    N, A = 4, 3
    rng = _rng(12)
    cf_pred = np.zeros((N, A, H, W, 3), dtype=np.uint8)
    cf_true = np.zeros((N, H, W, 3), dtype=np.uint8)
    ctx = np.zeros((N, H, W, 3), dtype=np.uint8)
    taken = np.zeros(N, dtype=np.int32)
    for i in range(N):
        tnext = rng.integers(0, 16, (H, W)).astype(np.int16)
        cf_true[i] = PALETTE_RGB[tnext]
        ctx[i] = PALETTE_RGB[(tnext + 7) % 16]  # context differs from truth
        cf_pred[i, 0] = PALETTE_RGB[tnext]          # taken: correct
        cf_pred[i, 1] = PALETTE_RGB[(tnext + 3) % 16]  # wrong
        cf_pred[i, 2] = PALETTE_RGB[(tnext + 9) % 16]  # wrong
    out = score_counterfactual(cf_pred, cf_true, taken, ctx)
    assert out["sensitivity"] > 0.0          # predictions depend on action
    assert out["taken_acc"] == pytest.approx(1.0)
    assert out["beats_copy_frac"] == 1.0
    assert out["rank0_frac"] == 1.0
    assert out["n"] == N


def test_score_counterfactual_action_blind_model():
    # Same prediction for every action == sensitivity 0 (copying, not modelling).
    N, A = 3, 4
    rng = _rng(13)
    cf_pred = np.zeros((N, A, H, W, 3), dtype=np.uint8)
    cf_true = np.zeros((N, H, W, 3), dtype=np.uint8)
    ctx = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i in range(N):
        one = PALETTE_RGB[rng.integers(0, 16, (H, W)).astype(np.int16)]
        cf_pred[i, :] = one  # identical across actions
        cf_true[i] = one
        ctx[i] = one
    out = score_counterfactual(cf_pred, cf_true, np.zeros(N, np.int32), ctx)
    assert out["sensitivity"] == 0.0


def test_score_counterfactual_needs_two_actions():
    with pytest.raises(ValueError):
        score_counterfactual(
            np.zeros((2, 1, H, W, 3)), np.zeros((2, H, W, 3)),
            np.zeros(2, np.int32), np.zeros((2, H, W, 3)),
        )
