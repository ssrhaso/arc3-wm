"""Step-1 latent probe, part B: fit linear probes on dumped RSSM latents (CPU).

Reads `<game>_<source>_latents.npz` from `probe_dump_latents.py` and asks: is
task structure **linearly readable** from the frozen world model's latent state?
For each latent (`deter` h_t, `stoch` z_t, and both concatenated) and each target
(`level_id`, `transition`), it fits a linear classifier with a **group-wise
(by-episode) train/test split** and a **label-permutation control** for the
chance floor. Above-control balanced accuracy => the latent encodes the variable.

Dependency-free: a ridge (least-squares) linear classifier in numpy, not sklearn
(keeps the probe runnable in the same minimal env as the rest of the pipeline).
A logistic variant could be swapped in; the above-vs-control conclusion is robust
to the choice. Linear by design - decodability shows the info is present AND
linearly accessible (consistent with a control bottleneck); a null is the strong
direction (info absent => bottleneck is representation, not control).

Group split is load-bearing: adjacent frames are near-identical, so a frame-level
random split leaks and inflates accuracy. We split by `ep_id`.

Output `<outdir>/probe_fits.json`: per (game, latent, target) the balanced
accuracy, permuted-control mean/std, chance, and an `above_control` flag.

Usage::

    python scripts/probe_fit_probes.py --latents-dir results/dynamics_probe/latents \\
        --outdir results/dynamics_probe/scored
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def group_split(groups: np.ndarray, test_frac: float, seed: int):
    """Boolean (train, test) masks splitting whole groups (episodes), not frames."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_groups = set(uniq[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    return ~test_mask, test_mask


def _standardize(x_tr: np.ndarray, x_te: np.ndarray):
    mu = x_tr.mean(0)
    sd = x_tr.std(0) + 1e-6
    return (x_tr - mu) / sd, (x_te - mu) / sd


def _ridge_classify(x_tr, y_tr, x_te, classes, lam: float):
    """Closed-form ridge regression to one-hot targets; predict by argmax."""
    Y = (y_tr[:, None] == classes[None, :]).astype(np.float64)
    x_tr1 = np.concatenate([x_tr, np.ones((len(x_tr), 1))], 1)
    x_te1 = np.concatenate([x_te, np.ones((len(x_te), 1))], 1)
    A = x_tr1.T @ x_tr1 + lam * np.eye(x_tr1.shape[1])
    W = np.linalg.solve(A, x_tr1.T @ Y)
    return classes[np.argmax(x_te1 @ W, axis=1)]


def balanced_accuracy(y_true, y_pred, classes) -> float:
    """Mean per-class recall - robust to class imbalance (e.g. rare transitions)."""
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def fit_linear_probe(X, y, groups, *, seed=0, n_perm=5, test_frac=0.3, lam=1.0) -> dict:
    """Fit a ridge linear probe with a group split + permutation control."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    if len(classes) < 2:
        return {"status": "degenerate_single_class", "n_classes": int(len(classes))}
    tr, te = group_split(groups, test_frac, seed)
    if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
        return {"status": "degenerate_split", "n_train": int(tr.sum()), "n_test": int(te.sum())}
    x_tr, x_te = _standardize(X[tr], X[te])
    y_tr, y_te = y[tr], y[te]
    pred = _ridge_classify(x_tr, y_tr, x_te, classes, lam)
    bacc = balanced_accuracy(y_te, pred, classes)
    acc = float((pred == y_te).mean())
    rng = np.random.default_rng(seed + 1)
    perms = []
    for _ in range(n_perm):
        yp = y_tr.copy()
        rng.shuffle(yp)
        if len(np.unique(yp)) < 2:
            continue
        perms.append(balanced_accuracy(y_te, _ridge_classify(x_tr, yp, x_te, classes, lam), classes))
    perm_mean = float(np.mean(perms)) if perms else float("nan")
    perm_std = float(np.std(perms)) if perms else float("nan")
    return {
        "status": "ok",
        "balanced_acc": round(bacc, 4),
        "acc": round(acc, 4),
        "perm_balanced_acc_mean": round(perm_mean, 4),
        "perm_balanced_acc_std": round(perm_std, 4),
        "above_control": bool(bacc > perm_mean + 2 * (perm_std if perm_std == perm_std else 0)),
        "chance": round(1.0 / len(classes), 4),
        "n_classes": int(len(classes)),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }


def fit_probe_cv(X, y, groups, *, n_folds=20, test_frac=0.3, lam=1.0, seed=0,
                 n_perm=1) -> dict:
    """Episode-grouped K-fold probe with bootstrap CIs over the folds.

    Repeats the **group(episode)-wise** split ``n_folds`` times with different
    seeds; each fold fits the ridge probe and a label-permuted control. Reports
    the mean balanced accuracy with a 95% percentile CI over folds, and the
    control's CI. ``above_control`` is the strict CI-separation test: the probe's
    2.5th percentile exceeds the control's 97.5th percentile (a "null" target has
    overlapping CIs).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    if len(classes) < 2:
        return {"status": "degenerate_single_class", "n_classes": int(len(classes))}
    baccs, ctrls = [], []
    for f in range(n_folds):
        tr, te = group_split(groups, test_frac, seed + f)
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        x_tr, x_te = _standardize(X[tr], X[te])
        y_tr, y_te = y[tr], y[te]
        baccs.append(balanced_accuracy(y_te, _ridge_classify(x_tr, y_tr, x_te, classes, lam), classes))
        rng = np.random.default_rng(1000 + seed + f)
        cs = []
        for _ in range(n_perm):
            yp = y_tr.copy()
            rng.shuffle(yp)
            if len(np.unique(yp)) >= 2:
                cs.append(balanced_accuracy(y_te, _ridge_classify(x_tr, yp, x_te, classes, lam), classes))
        if cs:
            ctrls.append(float(np.mean(cs)))
    if len(baccs) < 3:
        return {"status": "too_few_folds", "n_folds": int(len(baccs))}
    baccs, ctrls = np.array(baccs), np.array(ctrls)
    lo, hi = np.percentile(baccs, [2.5, 97.5])
    clo, chi = np.percentile(ctrls, [2.5, 97.5]) if len(ctrls) else (np.nan, np.nan)
    return {
        "status": "ok",
        "balanced_acc": round(float(np.mean(baccs)), 4),
        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
        "ctrl_mean": round(float(np.mean(ctrls)), 4) if len(ctrls) else None,
        "ctrl_ci_hi": round(float(chi), 4) if len(ctrls) else None,
        "above_control": bool(lo > chi) if len(ctrls) else None,
        "chance": round(1.0 / len(classes), 4),
        "n_classes": int(len(classes)), "n_folds": int(len(baccs)),
    }


def derive_extra_targets(npz: dict, *, k_imminence: int = 5, n_time_bins: int = 5) -> dict:
    """Targets derivable from the dumped latents npz (no new forward pass).

    Both require the episode blocks to be contiguous/in-order (they are, from
    probe_dump_latents). These probe whether the latent carries *dynamic/temporal*
    structure, not just the visually-present level number:

    * ``reward_imminence`` - binary: does a level-clear (transition) fire within
      the next ``k_imminence`` steps? Forward-looking, not in the current frame.
    * ``time_since_reset`` - steps since the episode reset, quantile-binned for
      classification. A timing variable absent from any single frame's appearance.
    """
    ep = np.asarray(npz["ep_id"])
    tr = np.asarray(npz["transition"]).astype(int)
    n = len(ep)
    tsr = np.zeros(n, dtype=int)
    imm = np.zeros(n, dtype=int)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        tsr[idx] = np.arange(len(idx))
        t = tr[idx]
        for j in range(len(idx)):
            imm[idx[j]] = int(t[j + 1:j + 1 + k_imminence].any())
    qs = np.unique(np.quantile(tsr, np.linspace(0, 1, n_time_bins + 1)[1:-1]))
    tsr_binned = np.digitize(tsr, qs)
    return {"reward_imminence": imm, "time_since_reset": tsr_binned}


def probe_npz(path: Path, *, seed=0, n_folds=20, n_perm=1) -> dict:
    """All targets x {deter, stoch, both} with grouped-fold bootstrap CIs.

    Split is **episode-grouped** (group_split on ep_id) - confirmed here, never
    frame-level, so adjacent near-identical frames can't leak across train/test.
    """
    d = dict(np.load(path, allow_pickle=True))
    game, source = str(d["game"]), str(d["source"])
    deter, stoch = d["deter"], d["stoch"]
    both = np.concatenate([deter, stoch], axis=1)
    groups = d["ep_id"]
    feats = {"deter": deter, "stoch": stoch, "both": both}
    targets = {
        "level_id": d["level_id"],
        "transition": d["transition"].astype(np.int32),
        **derive_extra_targets(d),
    }
    out = {"game": game, "source": source, "n_frames": int(deter.shape[0]),
           "split": "episode-grouped", "n_folds": n_folds, "probes": {}}
    for tname, y in targets.items():
        for fname, X in feats.items():
            out["probes"][f"{tname}/{fname}"] = fit_probe_cv(
                X, y, groups, n_folds=n_folds, seed=seed, n_perm=n_perm)
    # Null flag: no target's 'both' probe clears control by CI separation.
    both_keys = [f"{t}/both" for t in targets]
    any_above = any(out["probes"][k].get("above_control") for k in both_keys)
    out["null"] = not any_above
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents-dir", default="results/dynamics_probe/latents")
    p.add_argument("--outdir", default="results/dynamics_probe/scored")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=20, help="episode-grouped folds for CIs")
    p.add_argument("--n-perm", type=int, default=1, help="permutation controls per fold")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    files = sorted(Path(args.latents_dir).glob("*_latents.npz"))
    if not files:
        print(f"no *_latents.npz in {args.latents_dir}")
        return 1
    results = {}
    for f in files:
        r = probe_npz(f, seed=args.seed, n_folds=args.n_folds, n_perm=args.n_perm)
        results[f"{r['game']}/{r['source']}"] = r
        print(f"[{r['game']}/{r['source']}] {r['n_frames']} frames, split={r['split']}, "
              f"{r['n_folds']} folds  {'<<< NULL' if r['null'] else ''}")
        for key, pr in r["probes"].items():
            if pr.get("status") == "ok":
                print(f"    {key:24} bacc={pr['balanced_acc']:.3f} "
                      f"[{pr['ci_lo']:.3f},{pr['ci_hi']:.3f}]  ctrl<={pr['ctrl_ci_hi']:.3f} "
                      f"(chance {pr['chance']:.3f})  above={pr['above_control']}")
            else:
                print(f"    {key:24} {pr['status']}")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "probe_fits.json").write_text(json.dumps(results, indent=2))
    print(f"[probe_fit_probes] wrote {outdir/'probe_fits.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
