"""Synthetic stand-in for the JAX predict stage - LOCAL DRY-RUN ONLY.

Produces predicted-frame npz files in the exact contract the GPU ``predict``
stage must emit, but fabricates the predictions from ground truth at a chosen
``--quality``. This lets the whole pipeline (collect -> predict -> score) run
end-to-end on the laptop so the score stage and table/figure outputs are
validated before the real forward pass runs on the GH200.

Qualities (sanity targets for the score stage):
  * ``perfect`` - rollout predicts the true future; counterfactual predicts the
    true next frame for the *taken* action and the held context for the others
    (=> model_acc 1.0, sensitivity = real board-change rate, taken ranks best).
  * ``copy``    - everything predicts the last observed frame held constant
    (=> model_acc == copy_acc, sensitivity 0). The null model.
  * ``noisy``   - true future with a fraction of cells corrupted.

This is NOT part of the result - it is a test fixture for the plumbing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.palette import PALETTE_RGB  # noqa: E402
from arc3_wm.probe_data import (  # noqa: E402
    make_counterfactual_specs,
    make_rollout_windows,
)


def _corrupt(frame: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    n = int(frac * H * W)
    if n:
        ys = rng.integers(0, H, n)
        xs = rng.integers(0, W, n)
        out[ys, xs] = PALETTE_RGB[rng.integers(0, 16, n)]
    return out


def generate(holdout_npz: Path, quality: str, *, context_len: int, horizon: int,
             target_a: int, n_click: int, seed: int, outdir: Path) -> None:
    d = dict(np.load(holdout_npz, allow_pickle=True))
    game, source = str(d["game"]), str(d["source"])
    rng = np.random.default_rng(seed)

    # --- Probe B: rollout (build npz directly; we hold the truth here) ---
    windows = make_rollout_windows(d, context_len=context_len, horizon=horizon)
    if windows:
        # Build by hand to use each window's true_future / context_last.
        preds, trues, ctxs, ep_ids, starts, acts = [], [], [], [], [], []
        for w in windows:
            if quality == "perfect":
                p = w.true_future.copy()
            elif quality == "copy":
                p = np.repeat(w.context_last[None], horizon, axis=0)
            elif quality == "noisy":
                p = np.stack([_corrupt(f, 0.1, rng) for f in w.true_future])
            else:
                raise ValueError(quality)
            preds.append(p.astype(np.uint8)); trues.append(w.true_future.astype(np.uint8))
            ctxs.append(w.context_last.astype(np.uint8))
            ep_ids.append(w.ep_id); starts.append(w.start)
            acts.append(w.future_actions.astype(np.int32))
        rb = {
            "rb_pred": np.stack(preds), "rb_true": np.stack(trues),
            "rb_context": np.stack(ctxs), "rb_ep_id": np.array(ep_ids, np.int32),
            "rb_start": np.array(starts, np.int32), "rb_actions": np.stack(acts),
            "game": np.array(game), "source": np.array(source),
            "horizon": np.array(horizon),
        }
        np.savez_compressed(outdir / f"{game}_{source}_rollout.npz", **rb)
        print(f"[synthetic:{quality}] {game}/{source} rollout: {len(windows)} windows")

    # --- Probe A: counterfactual (needs availability -> random source) ---
    specs = make_counterfactual_specs(
        d, context_len=context_len, n_click=n_click, max_specs=None, seed=seed)
    if specs:
        # Build by hand using each spec's truth.
        kept = [s for s in specs if s.candidate_actions.shape[0] >= target_a]
        preds, trues, ctxs, actions, taken = [], [], [], [], []
        for s in kept:
            cand = s.candidate_actions[:target_a]
            A = cand.shape[0]
            if quality == "perfect":
                p = np.repeat(s.context_last[None], A, axis=0)  # others = "no change"
                p[s.taken_idx] = s.true_next                    # taken = truth
            elif quality == "copy":
                p = np.repeat(s.context_last[None], A, axis=0)  # all = copy
            elif quality == "noisy":
                p = np.repeat(s.context_last[None], A, axis=0)
                p[s.taken_idx] = _corrupt(s.true_next, 0.1, rng)
            else:
                raise ValueError(quality)
            preds.append(p.astype(np.uint8)); trues.append(s.true_next.astype(np.uint8))
            ctxs.append(s.context_last.astype(np.uint8)); actions.append(cand.astype(np.int32))
            taken.append(s.taken_idx)
        if kept:
            cf = {
                "cf_pred": np.stack(preds), "cf_true_next": np.stack(trues),
                "cf_context": np.stack(ctxs), "cf_actions": np.stack(actions),
                "cf_taken_idx": np.array(taken, np.int32),
                "game": np.array(game), "source": np.array(source),
                "target_a": np.array(target_a), "n_specs": np.array(len(kept)),
            }
            np.savez_compressed(outdir / f"{game}_{source}_cf.npz", **cf)
            print(f"[synthetic:{quality}] {game}/{source} cf: {len(kept)} specs (A={target_a})")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--holdout-dir", default="results/dynamics_probe/holdout")
    p.add_argument("--outdir", default="results/dynamics_probe/pred")
    p.add_argument("--quality", choices=["perfect", "copy", "noisy"], default="perfect")
    p.add_argument("--context-len", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--target-a", type=int, default=4)
    p.add_argument("--n-click", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--glob", default="*.npz", help="which holdout npz to process")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.holdout_dir).glob(args.glob))
    if not files:
        print(f"no holdout npz in {args.holdout_dir}/{args.glob}")
        return 1
    for f in files:
        generate(f, args.quality, context_len=args.context_len, horizon=args.horizon,
                 target_a=args.target_a, n_click=args.n_click, seed=args.seed, outdir=outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
