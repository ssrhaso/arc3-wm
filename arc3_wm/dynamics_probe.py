"""Pure metric functions for the dynamics-competence probes (Phase 6).

These quantify whether a *frozen* DreamerV3 world model's predictions track
reality, independent of the (dead) reward head - the load-bearing evidence for
the "competence without performance" claim. See ``docs`` and the Phase-6 probe
scripts for the surrounding pipeline.

This module is deliberately JAX-free and frame-only: every function takes
already-materialised RGB frames (ground-truth from the offline env / replays,
predicted from the JAX ``predict`` stage) and returns plain Python floats. That
keeps the metrics + their tests runnable on the laptop CPU; only the
forward-pass stage that *produces* the predicted frames needs a GPU box.

Two probes are scored from these primitives:

* **Counterfactual action-sensitivity (one-step).** Given per-action one-step
  predictions from a single real state, does the prediction *depend* on the
  action (``action_sensitivity``), and does the *taken* action's prediction
  match the true next frame better than wrong actions and better than the
  copy-last-frame baseline (``action_discrimination``)? Zero sensitivity ==
  an action-blind model that merely copies the board.

* **Multi-step rollout fidelity.** Per-horizon cell accuracy of an open-loop
  imagined rollout vs. ground truth, compared against copy-last-frame
  (``cell_accuracy`` / ``changed_fraction`` per step). Beating copy across the
  actor's planning horizon is what rules out "low loss is trivial copying on a
  near-static board".

Grids are discrete (16-colour palette), so the natural unit is **per-cell
palette accuracy**: quantise both frames to nearest palette index and compare.
``frame_mse`` is a continuous backup that does not depend on quantisation.
"""
from __future__ import annotations

import numpy as np

from .palette import PALETTE_RGB

__all__ = [
    "quantize_to_palette",
    "cell_accuracy",
    "changed_fraction",
    "frame_mse",
    "action_sensitivity",
    "action_discrimination",
    "score_rollout",
    "score_counterfactual",
]

_PAL_F32 = PALETTE_RGB.astype(np.float32)  # (16, 3)


def quantize_to_palette(rgb: np.ndarray) -> np.ndarray:
    """Map ``(..., 3)`` RGB (uint8 or float) to nearest-palette index ``(...)``.

    The world-model decoder emits continuous RGB; ground-truth frames are exact
    palette colours. Quantising both to the nearest of the 16 fixed palette
    colours (min L2 in RGB space) puts predictions and targets in the same
    discrete space so a per-cell match is well defined. Exact palette colours
    quantise to themselves, so this is loss-free on ground truth.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected last axis 3 (RGB), got shape {arr.shape}")
    # (..., 1, 3) - (16, 3) -> (..., 16) squared distances; argmin over palette.
    d = ((arr[..., None, :] - _PAL_F32) ** 2).sum(-1)
    return d.argmin(-1).astype(np.int16)


def _check_pair(pred_rgb: np.ndarray, true_rgb: np.ndarray) -> None:
    p, t = np.asarray(pred_rgb), np.asarray(true_rgb)
    if p.shape != t.shape:
        raise ValueError(f"frame shape mismatch: {p.shape} vs {t.shape}")
    if p.shape[-1] != 3:
        raise ValueError(f"expected RGB last axis 3, got {p.shape}")


def cell_accuracy(pred_rgb: np.ndarray, true_rgb: np.ndarray) -> float:
    """Fraction of grid cells whose nearest-palette colour matches.

    1.0 == every cell predicted to the correct colour; chance for a 16-colour
    grid is ~1/16 only if colours are uniform, but ARC boards are
    background-dominated so the meaningful comparison is always vs. the copy
    baseline (``cell_accuracy(context, true)``), not vs. chance.
    """
    _check_pair(pred_rgb, true_rgb)
    return float((quantize_to_palette(pred_rgb) == quantize_to_palette(true_rgb)).mean())


def changed_fraction(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """Fraction of cells that differ between two frames (in palette space).

    Characterises how much a transition actually moves the board, so a high
    copy-baseline accuracy can be read correctly: if ``changed_fraction`` is
    tiny the board is near-static and copy is a strong (not impressive)
    baseline; the model only earns credit by beating copy where change is real.
    """
    _check_pair(a_rgb, b_rgb)
    return float((quantize_to_palette(a_rgb) != quantize_to_palette(b_rgb)).mean())


def frame_mse(pred_rgb: np.ndarray, true_rgb: np.ndarray) -> float:
    """Mean squared per-pixel RGB error; quantisation-free continuous backup."""
    _check_pair(pred_rgb, true_rgb)
    return float(
        ((np.asarray(pred_rgb, np.float32) - np.asarray(true_rgb, np.float32)) ** 2).mean()
    )


def action_sensitivity(preds_by_action: np.ndarray) -> float:
    """Fraction of cells whose one-step prediction depends on the action.

    ``preds_by_action`` is ``(A, H, W, 3)``: the decoded one-step prediction
    from a *single* real state under each of ``A`` candidate actions. Returns
    the fraction of cells where the ``A`` action-conditioned predictions are not
    all identical (in palette space). **0.0 == an action-blind world model**
    (every action yields the same next frame - the model is copying, not
    modelling dynamics); >0 means the dynamics head responds to the action.
    Requires ``A >= 2``.
    """
    preds = np.asarray(preds_by_action)
    if preds.ndim != 4 or preds.shape[-1] != 3:
        raise ValueError(f"expected (A, H, W, 3), got {preds.shape}")
    A = preds.shape[0]
    if A < 2:
        raise ValueError(f"need >=2 actions to measure sensitivity, got {A}")
    idx = quantize_to_palette(preds)  # (A, H, W)
    all_agree = (idx == idx[0]).all(axis=0)  # cells where every action agrees
    return float((~all_agree).mean())


def action_discrimination(
    preds_by_action: np.ndarray,
    true_rgb: np.ndarray,
    taken_idx: int,
    context_rgb: np.ndarray | None = None,
) -> dict:
    """Does the *taken* action's prediction best explain the true next frame?

    ``preds_by_action`` ``(A, H, W, 3)`` are one-step predictions for ``A``
    candidate actions from one real state; ``true_rgb`` is the actual next
    frame; ``taken_idx`` indexes the action actually taken. Returns:

    * ``taken_acc`` - cell accuracy of the taken action's prediction,
    * ``mean_other_acc`` - mean cell accuracy of the *other* actions',
    * ``rank`` - number of other actions whose prediction matches truth strictly
      better than the taken one (0 == taken action is the single best
      explanation of what happened),
    * ``copy_acc`` - cell accuracy of copy-last-frame (only if ``context_rgb``
      given), the baseline the taken prediction must beat,
    * ``beats_copy`` - ``taken_acc > copy_acc`` (None without context).

    A competent dynamics head should rank the taken action best (rank 0) and
    beat copy: it predicts *this* action's specific consequence, not a generic
    next board.
    """
    preds = np.asarray(preds_by_action)
    if preds.ndim != 4 or preds.shape[-1] != 3:
        raise ValueError(f"expected (A, H, W, 3), got {preds.shape}")
    A = preds.shape[0]
    if not (0 <= taken_idx < A):
        raise ValueError(f"taken_idx {taken_idx} out of range [0, {A})")
    accs = [cell_accuracy(preds[i], true_rgb) for i in range(A)]
    taken = accs[taken_idx]
    others = [a for i, a in enumerate(accs) if i != taken_idx]
    out = {
        "taken_acc": taken,
        "mean_other_acc": float(np.mean(others)) if others else float("nan"),
        "rank": int(sum(o > taken for o in others)),
    }
    if context_rgb is not None:
        copy_acc = cell_accuracy(context_rgb, true_rgb)
        out["copy_acc"] = copy_acc
        out["beats_copy"] = bool(taken > copy_acc)
    else:
        out["copy_acc"] = None
        out["beats_copy"] = None
    return out


# --- batched scorers (Stage 3 aggregation; consume Stage 2 predicted npz) ---

def score_rollout(
    pred: np.ndarray, true: np.ndarray, context_last: np.ndarray
) -> dict:
    """Per-horizon multi-step rollout fidelity vs. copy-last-frame.

    ``pred``/``true`` are ``(N, H, 64, 64, 3)`` (N rollouts, H imagined steps);
    ``context_last`` is ``(N, 64, 64, 3)`` (last *observed* frame, held constant
    by the copy baseline). Returns per-horizon arrays of length ``H``:

    * ``model_acc`` - cell accuracy of the imagined frame at step k,
    * ``copy_acc`` - cell accuracy of holding ``context_last`` constant,
    * ``changed`` - fraction of cells that actually moved vs. ``context_last``
      (so a high ``copy_acc`` is read correctly: tiny ``changed`` == near-static
      board, copy is strong; the model earns credit only by ``model_acc`` >
      ``copy_acc`` where ``changed`` is real),
    * ``mse`` - mean per-pixel RGB error of the imagined frame,
    * ``n`` - number of rollouts averaged.

    The load-bearing quantity is ``model_acc - copy_acc`` across the horizon:
    positive across the actor's planning length is what refutes "low loss is
    trivial copying".
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    ctx = np.asarray(context_last)
    if pred.shape != true.shape or pred.ndim != 5 or pred.shape[-1] != 3:
        raise ValueError(
            f"pred/true must match as (N, H, 64, 64, 3); got {pred.shape} vs {true.shape}"
        )
    if ctx.shape[0] != pred.shape[0] or ctx.shape[1:] != pred.shape[2:]:
        raise ValueError(
            f"context_last must be (N, 64, 64, 3) matching pred; got {ctx.shape}"
        )
    N, H = pred.shape[:2]
    pi = quantize_to_palette(pred)          # (N, H, 64, 64)
    ti = quantize_to_palette(true)          # (N, H, 64, 64)
    ci = quantize_to_palette(ctx)           # (N, 64, 64)
    ci_b = ci[:, None]                      # (N, 1, 64, 64) broadcast over H
    model_acc = (pi == ti).mean(axis=(0, 2, 3))     # (H,)
    copy_acc = (ci_b == ti).mean(axis=(0, 2, 3))    # (H,)
    changed = (ci_b != ti).mean(axis=(0, 2, 3))     # (H,)
    mse = ((pred.astype(np.float32) - true.astype(np.float32)) ** 2).mean(axis=(0, 2, 3, 4))
    return {
        "model_acc": model_acc,
        "copy_acc": copy_acc,
        "changed": changed,
        "mse": mse,
        "n": int(N),
    }


def score_counterfactual(
    cf_pred: np.ndarray,
    cf_true_next: np.ndarray,
    cf_taken_idx: np.ndarray,
    cf_context: np.ndarray,
) -> dict:
    """Aggregate one-step counterfactual action-sensitivity over many states.

    * ``cf_pred`` ``(N, A, 64, 64, 3)`` - one-step prediction per candidate
      action from each of N real states,
    * ``cf_true_next`` ``(N, 64, 64, 3)`` - actual next frame,
    * ``cf_taken_idx`` ``(N,)`` - column of the action actually taken,
    * ``cf_context`` ``(N, 64, 64, 3)`` - the state frame (copy baseline).

    Returns means over the N states: ``sensitivity`` (fraction of cells whose
    prediction depends on the action; 0 == action-blind), ``taken_acc``,
    ``mean_other_acc``, ``copy_acc``, ``beats_copy_frac`` (share where the taken
    action's prediction beats copy), ``rank0_frac`` (share where the taken
    action is the single best explanation of the true next frame), and ``n``.
    """
    cf_pred = np.asarray(cf_pred)
    cf_true = np.asarray(cf_true_next)
    taken = np.asarray(cf_taken_idx)
    ctx = np.asarray(cf_context)
    if cf_pred.ndim != 5 or cf_pred.shape[-1] != 3:
        raise ValueError(f"cf_pred must be (N, A, 64, 64, 3); got {cf_pred.shape}")
    N, A = cf_pred.shape[:2]
    if A < 2:
        raise ValueError(f"need >=2 candidate actions, got A={A}")
    sens, taken_acc, other_acc, copy_acc, beats, rank0 = [], [], [], [], [], []
    for i in range(N):
        sens.append(action_sensitivity(cf_pred[i]))
        d = action_discrimination(
            cf_pred[i], cf_true[i], int(taken[i]), context_rgb=ctx[i]
        )
        taken_acc.append(d["taken_acc"])
        other_acc.append(d["mean_other_acc"])
        copy_acc.append(d["copy_acc"])
        beats.append(1.0 if d["beats_copy"] else 0.0)
        rank0.append(1.0 if d["rank"] == 0 else 0.0)
    return {
        "sensitivity": float(np.mean(sens)),
        "taken_acc": float(np.mean(taken_acc)),
        "mean_other_acc": float(np.nanmean(other_acc)),
        "copy_acc": float(np.mean(copy_acc)),
        "beats_copy_frac": float(np.mean(beats)),
        "rank0_frac": float(np.mean(rank0)),
        "n": int(N),
    }
