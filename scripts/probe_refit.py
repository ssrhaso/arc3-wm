"""Re-score the frozen-latent probes with permutation tests and Holm correction.

``scripts/probe_fit_probes.py`` judged a target decodable by comparing percentile
intervals over 20 overlapping 70/30 episode splits against a control permuted
once per fold, with no correction for the number of tests. This script keeps the
same probe (ridge on standardized features, episode-grouped splits) and replaces
the inference: non-overlapping grouped K-fold, a real permutation test per
(game, feature set, target), Holm correction over the whole family, and paired
per-fold differences of latent against the pixel and clock baselines.

Feature sets, all scored on identical folds so the differences are paired:

* ``deter`` / ``stoch`` / ``both`` from the frozen RSSM state.
* ``pixel``  a fixed random Gaussian projection of the raw frame to 512 dims,
  the "could you read this off the picture" control.
* ``clock``  one-hot bins of the step index, the "is this a step counter"
  control.

Targets are the two dumped labels (``level_id``, ``transition``) plus the three
derived in ``probe_fit_probes.derive_extra_targets``: ``reward_imminence``
(a clear within 5 steps), ``reward_proximity`` (within 10), and
``time_since_reset``.

Usage::

    python scripts/probe_refit.py --n-perm 1000 \\
        --latents-dir results/dynamics_probe/latents \\
        --holdout-dir results/dynamics_probe/holdout \\
        --outdir results/dynamics_probe/refit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from arc3_wm.probe_stats import (  # noqa: E402
    RidgeFolds,
    holm,
    paired_fold_difference,
    permutation_test,
)
from probe_fit_probes import derive_extra_targets  # noqa: E402

PIXEL_DIM = 512
CLOCK_BINS = 10


def pixel_features(frames: np.ndarray, dim: int = PIXEL_DIM, seed: int = 0) -> np.ndarray:
    """Random Gaussian projection of flattened frames, the raw-pixel control."""
    flat = frames.reshape(len(frames), -1).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    proj = rng.normal(scale=1.0 / np.sqrt(flat.shape[1]), size=(flat.shape[1], dim)).astype(np.float32)
    return flat @ proj


def clock_features(step: np.ndarray, n_bins: int = CLOCK_BINS) -> np.ndarray:
    """One-hot quantile bins of the within-episode step index."""
    step = np.asarray(step, dtype=float)
    edges = np.unique(np.quantile(step, np.linspace(0, 1, n_bins + 1)[1:-1]))
    idx = np.digitize(step, edges)
    out = np.zeros((len(step), int(idx.max()) + 1), dtype=np.float32)
    out[np.arange(len(step)), idx] = 1.0
    return out


def build_targets(latents: dict) -> dict[str, np.ndarray]:
    """The five probe targets, all derived without a new forward pass."""
    extra5 = derive_extra_targets(latents, k_imminence=5)
    extra10 = derive_extra_targets(latents, k_imminence=10)
    return {
        "level_id": np.asarray(latents["level_id"]).astype(int),
        "transition": np.asarray(latents["transition"]).astype(int),
        "reward_imminence": np.asarray(extra5["reward_imminence"]).astype(int),
        "reward_proximity": np.asarray(extra10["reward_imminence"]).astype(int),
        "time_since_reset": np.asarray(extra5["time_since_reset"]).astype(int),
    }


def build_features(latents: dict, holdout: dict | None, seed: int) -> dict[str, np.ndarray]:
    feats = {
        "deter": np.asarray(latents["deter"]),
        "stoch": np.asarray(latents["stoch"]),
        "both": np.concatenate([np.asarray(latents["deter"]), np.asarray(latents["stoch"])], 1),
    }
    if holdout is not None:
        n = len(feats["both"])
        if len(holdout["frames"]) != n:
            raise ValueError(
                f"holdout has {len(holdout['frames'])} rows, latents {n}: "
                "row alignment is required for the pixel and clock baselines"
            )
        if not np.array_equal(np.asarray(holdout["ep_id"]), np.asarray(latents["ep_id"])):
            raise ValueError("holdout and latents disagree on ep_id ordering")
        feats["pixel"] = pixel_features(np.asarray(holdout["frames"]), seed=seed)
        feats["clock"] = clock_features(np.asarray(holdout["step"]))
    return feats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latents-dir", type=Path, default=Path("results/dynamics_probe/latents"))
    ap.add_argument("--holdout-dir", type=Path, default=Path("results/dynamics_probe/holdout"))
    ap.add_argument("--outdir", type=Path, default=Path("results/dynamics_probe/refit"))
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="permutations for the primary feature sets (both, pixel, clock)")
    ap.add_argument("--n-perm-secondary", type=int, default=200,
                    help="permutations for deter and stoch, reported without Holm membership")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--primary-features", nargs="*", default=["both", "pixel", "clock"])
    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.latents_dir.glob("*_latents.npz"))
    if args.games:
        files = [f for f in files if f.name.split("_")[0] in args.games]
    if not files:
        raise SystemExit(f"no *_latents.npz under {args.latents_dir}")

    results: dict = {}
    pvals_primary: dict[str, float] = {}
    t0 = time.time()
    for path in files:
        game, source = path.name.split("_")[0], path.name.split("_")[1]
        latents = dict(np.load(path, allow_pickle=True))
        hpath = args.holdout_dir / f"{game}_{source}.npz"
        holdout = dict(np.load(hpath, allow_pickle=True)) if hpath.exists() else None
        if holdout is None:
            print(f"[{game}] no holdout at {hpath}: pixel and clock baselines skipped", flush=True)
        feats = build_features(latents, holdout, args.seed)
        targets = build_targets(latents)
        groups = np.asarray(latents["ep_id"])
        print(f"[{game}] n={len(groups)} episodes={len(np.unique(groups))} "
              f"features={list(feats)} ({time.time() - t0:.0f}s)", flush=True)

        game_res: dict = {"n_frames": int(len(groups)), "n_episodes": int(len(np.unique(groups)))}
        fold_scores: dict[tuple[str, str], list[float]] = {}
        for fname, X in feats.items():
            folds = RidgeFolds(X, groups, n_folds=args.n_folds, lam=args.lam, seed=args.seed)
            n_perm = args.n_perm if fname in args.primary_features else args.n_perm_secondary
            for tname, y in targets.items():
                res = permutation_test(folds, y, n_perm=n_perm, seed=args.seed)
                res["primary"] = fname in args.primary_features
                game_res[f"{fname}/{tname}"] = res
                if res.get("status") == "ok":
                    fold_scores[(fname, tname)] = res["fold_scores"]
                    if res["primary"]:
                        pvals_primary[f"{game}/{fname}/{tname}"] = res["p_value"]
            print(f"  {fname}: " + ", ".join(
                f"{t}={game_res[f'{fname}/{t}'].get('balanced_acc', 'NA')}"
                f"(p={game_res[f'{fname}/{t}'].get('p_value', 'NA')})" for t in targets), flush=True)

        for tname in targets:
            for base in ("pixel", "clock"):
                a, b = fold_scores.get(("both", tname)), fold_scores.get((base, tname))
                if a and b and len(a) == len(b):
                    game_res[f"both_minus_{base}/{tname}"] = paired_fold_difference(
                        a, b, seed=args.seed)
        results[game] = game_res

    adjusted = holm(pvals_primary)
    for key, padj in adjusted.items():
        game, fname, tname = key.split("/")
        results[game][f"{fname}/{tname}"]["p_holm"] = round(float(padj), 5)
        results[game][f"{fname}/{tname}"]["decodable"] = bool(padj < 0.05)

    out = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_tests_in_family": len(pvals_primary),
        "holm_alpha": 0.05,
        "results": results,
    }
    (args.outdir / "probe_refit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    rows = ["game,feature,target,balanced_acc,chance,null_mean,p_value,p_holm,decodable"]
    for game, gres in results.items():
        for key, r in gres.items():
            if not isinstance(r, dict) or r.get("status") != "ok" or "minus" in key:
                continue
            fname, tname = key.split("/")
            rows.append(",".join(str(x) for x in [
                game, fname, tname, r["balanced_acc"], r["chance"], r["null_mean"],
                r["p_value"], r.get("p_holm", ""), r.get("decodable", "")]))
    (args.outdir / "probe_refit.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nwrote {args.outdir}/probe_refit.json and .csv "
          f"({len(pvals_primary)} tests in the Holm family, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
