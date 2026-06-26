"""Shared, JAX-free data prep for the dynamics-competence probes.

Both the JAX ``predict`` stage (on a GPU box) and the local synthetic-prediction
generator consume the *same* window/state specs produced here, so the
episode-segmentation and candidate-action logic is written and tested once on
the laptop; the predict stage only adds the model forward pass on top.

Inputs are the flat-transition npz dicts written by
``scripts/probe_collect_holdout.py`` (keys: ``frames, actions, rewards, ep_id,
step, is_last, avail``).

Two spec builders:

* ``make_rollout_windows`` - for **Probe B** (multi-step rollout fidelity): each
  window has a context prefix (frames + actions to ``observe``) and a future of
  ``horizon`` real actions whose true frames the imagined rollout is scored
  against. Only episodes with at least ``context_len + horizon`` steps qualify.
* ``make_counterfactual_specs`` - for **Probe A** (one-step action-sensitivity):
  each spec has a context prefix up to step ``t``, the action actually taken, the
  true next frame, and a candidate action set (taken + sampled alternatives,
  drawn from the per-frame available action *types*). Needs ``avail`` (random
  source); human-source npz carry no availability and are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .action_space import ACTION6_BASE, ACTION6_COUNT, ACTION7_INDEX

__all__ = [
    "Episode",
    "RolloutWindow",
    "CounterfactualSpec",
    "iter_episodes",
    "make_rollout_windows",
    "candidate_actions_for_types",
    "make_counterfactual_specs",
    "build_rollout_prediction_npz",
    "build_counterfactual_prediction_npz",
    "frame_labels",
]


def frame_labels(rewards: np.ndarray, ep_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ``(level_id, transition)`` labels for the latent probe.

    Derived from the reward stream (reward == delta-levels) so they can be
    recomputed from any holdout npz without re-collecting:

    * ``level_id`` (int32) - levels cleared *before* this step (exclusive prefix
      sum of reward within the episode); the frame shows the pre-clear board, so
      this is the level the frame is on. The ordinal level-identity target.
    * ``transition`` (bool) - did a level-clear fire at this step (reward > 0).
      The binary transition-event target.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    ep_id = np.asarray(ep_id)
    transition = rewards > 0
    level_id = np.zeros(len(rewards), dtype=np.int32)
    for e in np.unique(ep_id):
        m = ep_id == e
        csum = np.cumsum(rewards[m])
        level_id[m] = (csum - rewards[m]).astype(np.int32)
    return level_id, transition


@dataclass
class Episode:
    frames: np.ndarray   # (L, 64, 64, 3) uint8
    actions: np.ndarray  # (L,) int32 - action taken AT frames[t] (conv. B)
    avail: np.ndarray    # (L, 7) bool - action-type availability (or all-False)
    ep_id: int


@dataclass
class RolloutWindow:
    ep_id: int
    start: int                  # episode step index of the first future frame
    context_frames: np.ndarray  # (C, 64, 64, 3) uint8
    context_actions: np.ndarray  # (C,) int32 (prevact for observe)
    future_actions: np.ndarray  # (H,) int32 (actions driving the imagined roll)
    true_future: np.ndarray     # (H, 64, 64, 3) uint8
    context_last: np.ndarray    # (64, 64, 3) uint8 (copy-baseline frame)


@dataclass
class CounterfactualSpec:
    ep_id: int
    t: int                       # episode step of the state being branched
    context_frames: np.ndarray   # (t+1, 64, 64, 3) uint8 (obs through step t)
    context_actions: np.ndarray  # (t+1,) int32
    context_last: np.ndarray     # (64, 64, 3) uint8 (= context_frames[-1])
    true_next: np.ndarray        # (64, 64, 3) uint8 (real frame after taken act)
    candidate_actions: np.ndarray  # (A,) int32 (taken + alternatives)
    taken_idx: int               # index of the taken action in candidate_actions
    candidate_meta: dict = field(default_factory=dict)


def iter_episodes(npz: dict) -> Iterator[Episode]:
    """Group a flat-transition npz dict into per-episode arrays (sorted by id)."""
    ep_id = np.asarray(npz["ep_id"])
    frames = np.asarray(npz["frames"])
    actions = np.asarray(npz["actions"])
    avail = (
        np.asarray(npz["avail"]) if "avail" in npz
        else np.zeros((frames.shape[0], 7), dtype=bool)
    )
    for e in np.unique(ep_id):
        m = ep_id == e
        yield Episode(
            frames=frames[m], actions=actions[m].astype(np.int32),
            avail=avail[m].astype(bool), ep_id=int(e),
        )


def make_rollout_windows(
    npz: dict, *, context_len: int, horizon: int, stride: int | None = None,
    max_windows: int | None = None,
) -> list[RolloutWindow]:
    """Build Probe-B windows from every episode long enough to hold one.

    The context is a **fixed** window of ``context_len`` frames immediately
    preceding the rollout ``start`` (so batches are rectangular for the JAX
    forward). ``stride`` defaults to ``horizon`` (non-overlapping futures). An
    episode of length ``L`` yields windows at ``start = C, C+stride, ...`` while
    ``start + horizon <= L``. ``future_actions[k]`` is the action taken at frame
    ``start+k`` and drives the imagined step to frame ``start+k+1``;
    ``context_last`` is the frame just before ``start`` (the copy baseline).
    """
    if context_len < 1 or horizon < 1:
        raise ValueError("context_len and horizon must be >= 1")
    stride = horizon if stride is None else stride
    windows: list[RolloutWindow] = []
    for ep in iter_episodes(npz):
        L = ep.frames.shape[0]
        start = context_len
        while start + horizon <= L:
            windows.append(
                RolloutWindow(
                    ep_id=ep.ep_id, start=start,
                    context_frames=ep.frames[start - context_len:start],
                    context_actions=ep.actions[start - context_len:start],
                    future_actions=ep.actions[start:start + horizon],
                    true_future=ep.frames[start:start + horizon],
                    context_last=ep.frames[start - 1],
                )
            )
            if max_windows and len(windows) >= max_windows:
                return windows
            start += stride
    return windows


def candidate_actions_for_types(
    avail_types: np.ndarray, taken: int, *, n_click: int, rng: np.random.Generator
) -> np.ndarray:
    """Flat candidate-action set from a 7-bool action-type availability vector.

    Includes the parameter-less available types (ACTION1-5 -> 0..4, ACTION7 ->
    4101) and, if ACTION6 is available, ``n_click`` uniformly-sampled click
    cells (flat ``ACTION6_BASE + cell``). The ``taken`` action is always present
    and de-duplicated; the returned array preserves first-seen order so the
    taken action's index is recoverable by the caller.
    """
    cands: list[int] = [int(taken)]
    for t in range(1, 6):  # ACTION1..ACTION5 -> flat 0..4
        if avail_types[t - 1]:
            cands.append(t - 1)
    if avail_types[5]:  # ACTION6 click grid
        cells = rng.choice(ACTION6_COUNT, size=min(n_click, ACTION6_COUNT), replace=False)
        cands.extend(int(ACTION6_BASE + c) for c in cells)
    if avail_types[6]:  # ACTION7 undo
        cands.append(ACTION7_INDEX)
    # De-dup preserving order (taken stays first).
    seen: set[int] = set()
    uniq = [a for a in cands if not (a in seen or seen.add(a))]
    return np.array(uniq, dtype=np.int32)


def make_counterfactual_specs(
    npz: dict, *, context_len: int, n_click: int, max_specs: int | None,
    seed: int, require_change: bool = True, allow_no_avail: bool = False,
) -> list[CounterfactualSpec]:
    """Build Probe-A specs at states with a known available-action set.

    A spec is created at episode steps ``t >= context_len - 1`` that are not the
    episode's last step (so a true next frame exists). When ``require_change``
    (default) only states whose taken action actually moved the board are kept -
    those are the states where action-sensitivity is meaningful (this is the
    "state-changing actions only" filter).

    Availability: the random source carries per-frame ``avail`` so the candidate
    set is the truly-available actions. The human source has no ``avail``; with
    ``allow_no_avail=True`` we fall back to all seven action types as candidates
    (so human counterfactuals are possible, at the cost of including a few
    engine-rejected types). Returns ``[]`` if no avail and ``allow_no_avail`` is
    False.
    """
    has_avail = bool(npz.get("has_avail", np.array(False)))
    if not has_avail and not allow_no_avail:
        return []
    rng = np.random.default_rng(seed)
    all_types = np.ones(7, dtype=bool)
    specs: list[CounterfactualSpec] = []
    for ep in iter_episodes(npz):
        L = ep.frames.shape[0]
        for t in range(context_len - 1, L - 1):
            true_next = ep.frames[t + 1]
            if require_change and bool((true_next == ep.frames[t]).all()):
                continue
            taken = int(ep.actions[t])
            avail_t = ep.avail[t] if has_avail else all_types
            cands = candidate_actions_for_types(
                avail_t, taken, n_click=n_click, rng=rng
            )
            if cands.shape[0] < 2:
                continue
            specs.append(
                CounterfactualSpec(
                    ep_id=ep.ep_id, t=t,
                    context_frames=ep.frames[: t + 1],
                    context_actions=ep.actions[: t + 1],
                    context_last=ep.frames[t],
                    true_next=true_next,
                    candidate_actions=cands,
                    taken_idx=0,  # taken is first by construction
                    candidate_meta={"n_candidates": int(cands.shape[0])},
                )
            )
            if max_specs and len(specs) >= max_specs:
                return specs
    return specs


# --- prediction-npz assemblers (the Stage 2 output contract) ----------------
#
# Both the JAX predict stage and the local synthetic generator call these with
# a ``predict_fn`` callback, so the on-disk schema consumed by the score stage
# is defined in exactly one place.
#
#   rollout npz : rb_pred, rb_true, rb_context (N,H/1,64,64,3 uint8),
#                 rb_ep_id, rb_start (N,) int32, plus game/source/horizon meta.
#   counterfac. : cf_pred (N,A,64,64,3), cf_true_next/cf_context (N,64,64,3),
#                 cf_actions (N,A) int32, cf_taken_idx (N,) int32, plus meta.


def build_rollout_prediction_npz(
    windows: list[RolloutWindow], predict_rollout_fn, *, game: str, source: str,
) -> dict:
    """Assemble the Probe-B prediction npz.

    ``predict_rollout_fn(context_frames, context_actions, future_actions)`` must
    return ``(H, 64, 64, 3)`` predicted frames for the future actions, given the
    observed context. Stage 2 implements this with the frozen WM
    (observe-then-imagine); the synthetic generator fakes it.
    """
    if not windows:
        raise RuntimeError(f"no rollout windows for {game}/{source}")
    H = windows[0].future_actions.shape[0]
    preds, trues, ctxs, ep_ids, starts, acts = [], [], [], [], [], []
    for w in windows:
        if w.future_actions.shape[0] != H:
            raise ValueError("all windows must share one horizon")
        p = np.asarray(predict_rollout_fn(
            w.context_frames, w.context_actions, w.future_actions))
        if p.shape != w.true_future.shape:
            raise ValueError(
                f"predict_rollout_fn returned {p.shape}, expected {w.true_future.shape}")
        preds.append(p.astype(np.uint8))
        trues.append(w.true_future.astype(np.uint8))
        ctxs.append(w.context_last.astype(np.uint8))
        ep_ids.append(w.ep_id)
        starts.append(w.start)
        acts.append(w.future_actions.astype(np.int32))
    return {
        "rb_pred": np.stack(preds), "rb_true": np.stack(trues),
        "rb_context": np.stack(ctxs),
        "rb_ep_id": np.array(ep_ids, np.int32),
        "rb_start": np.array(starts, np.int32),
        "rb_actions": np.stack(acts),
        "game": np.array(game), "source": np.array(source),
        "horizon": np.array(H),
    }


def build_counterfactual_prediction_npz(
    specs: list[CounterfactualSpec], predict_cf_fn, *, target_a: int,
    game: str, source: str,
) -> dict:
    """Assemble the Probe-A prediction npz with a fixed candidate count ``A``.

    Specs with fewer than ``target_a`` candidates are dropped; the rest are
    truncated to ``target_a`` (the taken action stays at index 0) so arrays stay
    rectangular. ``predict_cf_fn(context_frames, context_actions,
    candidate_actions)`` returns ``(A, 64, 64, 3)`` one-step predictions, one per
    candidate action.
    """
    kept = [s for s in specs if s.candidate_actions.shape[0] >= target_a]
    if not kept:
        return {
            "cf_pred": np.zeros((0, target_a, 64, 64, 3), np.uint8),
            "cf_true_next": np.zeros((0, 64, 64, 3), np.uint8),
            "cf_context": np.zeros((0, 64, 64, 3), np.uint8),
            "cf_actions": np.zeros((0, target_a), np.int32),
            "cf_taken_idx": np.zeros((0,), np.int32),
            "game": np.array(game), "source": np.array(source),
            "target_a": np.array(target_a), "n_specs": np.array(0),
        }
    preds, trues, ctxs, actions, taken = [], [], [], [], []
    for s in kept:
        cand = s.candidate_actions[:target_a].astype(np.int32)
        p = np.asarray(predict_cf_fn(s.context_frames, s.context_actions, cand))
        if p.shape != (target_a, 64, 64, 3):
            raise ValueError(
                f"predict_cf_fn returned {p.shape}, expected {(target_a, 64, 64, 3)}")
        preds.append(p.astype(np.uint8))
        trues.append(s.true_next.astype(np.uint8))
        ctxs.append(s.context_last.astype(np.uint8))
        actions.append(cand)
        taken.append(s.taken_idx)  # 0 by construction (taken first, kept first)
    return {
        "cf_pred": np.stack(preds), "cf_true_next": np.stack(trues),
        "cf_context": np.stack(ctxs), "cf_actions": np.stack(actions),
        "cf_taken_idx": np.array(taken, np.int32),
        "game": np.array(game), "source": np.array(source),
        "target_a": np.array(target_a), "n_specs": np.array(len(kept)),
    }
