"""Step-1 latent probe, part A: dump frozen-WM RSSM latents (JAX/GPU-or-CPU).

Runs the frozen world model's encoder + RSSM **observe** path over a game's
holdout frames and writes the per-frame model-state - deterministic recurrent
state ``deter`` (h_t) and stochastic categorical latent ``stoch`` (z_t) - to an
``.npz``, paired with the per-frame probe labels (`level_id`, `transition`,
`ep_id`). No imagination, no decode. `scripts/probe_fit_probes.py` (CPU) then
fits linear decoders on these latents.

This complements the rollout probe (`probe_predict.py`): rollout asks "are the
imagined *frames* faithful"; this asks "is *task structure* (which level, did a
transition fire) linearly readable from the latent state" - the stronger,
pixel-free competence claim. Probe `h_t` and `z_t` separately: decodable from
`h_t` but not a single-frame `z_t` => genuine temporal integration.

Reuses the agent build/restore from `probe_predict.py` with `mode="latents"`.
Same env as Stage 2 (see scripts/requirements-probe.txt). Use `--platform cpu`
on a CPU node.

Output `<outdir>/<game>_<source>_latents.npz`:
    deter      (N, Dh)   float32   per-frame recurrent state
    stoch      (N, Dz)   float32   per-frame categorical latent (flattened)
    level_id   (N,)      int32     levels cleared before this frame
    transition (N,)      bool      level-clear fired at this frame
    ep_id      (N,)      int32     episode id (for group-wise probe splits)

Usage::

    python scripts/probe_dump_latents.py --game cd82 --source human \\
        --ckpt ckpt_cd82 --holdout results/dynamics_probe/holdout/cd82_human.npz \\
        --platform cpu --outdir results/dynamics_probe/latents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
for p in (str(_REPO_ROOT), str(_DV3)):
    if p not in sys.path:
        sys.path.insert(0, p)

from arc3_wm.probe_data import frame_labels  # noqa: E402
from scripts.probe_predict import (  # noqa: E402
    IMG_KEY, OBS_HW, make_probe_agent, restore_checkpoint, run_report,
)

# WM param prefixes (encoder/dynamics/decoder/reward/continue) - what a WM-only
# pretrained .pkl carries and all the latents probe needs.
_WM_REGEX = r"^(enc|dyn|dec|rew|con)/"


def restore_from_pkl(agent, pkl_path: Path):
    """Restore WM params from a WM-only pickle (e.g. the Phase-3 pretrained WM).

    The pretrained checkpoint is a single ``.pkl`` of ``{'params','counters'}``
    (agent.save format), not a checkpoint dir. Load only the WM keys (enc/dyn/
    dec/rew/con) by regex; actor/critic stay at fresh init (unused for latents).
    """
    import pickle
    state = pickle.load(open(pkl_path, "rb"))
    agent.load(state, regex=_WM_REGEX)
    return agent


def _episodes(npz: dict):
    """Yield (frames, actions, level_id, transition, ep_id) per episode, in
    flat-npz order, with labels derived from the reward stream."""
    ep_id = np.asarray(npz["ep_id"])
    frames = np.asarray(npz["frames"])
    actions = np.asarray(npz["actions"])
    level_id, transition = frame_labels(npz["rewards"], ep_id)
    for e in np.unique(ep_id):
        m = ep_id == e
        yield frames[m], actions[m].astype(np.int32), level_id[m], transition[m], int(e)


def _build_batch(eps, max_t: int) -> dict:
    """Pad episodes to (B, max_t) standard report batch. Trailing padding never
    affects valid (earlier) frames' latents, so it's safe to drop afterwards."""
    B = len(eps)
    image = np.zeros((B, max_t, OBS_HW, OBS_HW, 3), np.uint8)
    action = np.zeros((B, max_t), np.int32)
    is_first = np.zeros((B, max_t), bool)
    for b, (f, a, _l, _t, _e) in enumerate(eps):
        L = f.shape[0]
        image[b, :L] = f
        action[b, :L] = a
        is_first[b, 0] = True
    return {
        IMG_KEY: image, "action": action,
        "reward": np.zeros((B, max_t), np.float32),
        "is_first": is_first,
        "is_last": np.zeros((B, max_t), bool),
        "is_terminal": np.zeros((B, max_t), bool),
        "consec": np.zeros((B, max_t), np.int32),
        "stepid": np.zeros((B, max_t, 20), np.uint8),
    }


def dump_latents(agent, holdout_npz: Path, *, ep_batch: int | None) -> dict:
    d = dict(np.load(holdout_npz, allow_pickle=True))
    game, source = str(d["game"]), str(d["source"])
    eps = list(_episodes(d))
    if not eps:
        raise RuntimeError(f"no episodes in {holdout_npz}")
    max_t = max(f.shape[0] for f, *_ in eps)
    ep_batch = ep_batch or len(eps)

    out = {k: [] for k in ("deter", "stoch", "level_id", "transition", "ep_id")}
    for s in range(0, len(eps), ep_batch):
        chunk = eps[s:s + ep_batch]
        mets = run_report(agent, _build_batch(chunk, max_t))
        deter = np.asarray(mets["probe/deter"])          # (B, max_t, Dh)
        stoch = np.asarray(mets["probe/stoch"])          # (B, max_t, ...)
        stoch = stoch.reshape(stoch.shape[0], stoch.shape[1], -1)  # flatten z
        for b, (f, _a, lvl, trn, e) in enumerate(chunk):
            L = f.shape[0]
            out["deter"].append(deter[b, :L].astype(np.float32))
            out["stoch"].append(stoch[b, :L].astype(np.float32))
            out["level_id"].append(lvl.astype(np.int32))
            out["transition"].append(trn.astype(bool))
            out["ep_id"].append(np.full(L, e, np.int32))
        print(f"  [{game}/{source}] episodes {s}..{s+len(chunk)} done")
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update(game=np.array(game), source=np.array(source))
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--source", default="human")
    p.add_argument("--ckpt", help="extracted full checkpoint dir (per-game)")
    p.add_argument("--ckpt-pkl", help="WM-only .pkl (e.g. pretrained cross-game WM)")
    p.add_argument("--holdout", required=True)
    p.add_argument("--outdir", default="results/dynamics_probe/latents")
    p.add_argument("--platform", choices=["cpu", "cuda"], default=None)
    p.add_argument("--compute-dtype", default=None,
                   help="float32 on aarch64 CPU (bf16 hits an LLVM codegen bug)")
    p.add_argument("--ep-batch", type=int, default=None,
                   help="episodes per forward (default: all; lower if OOM)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    flags = []
    if args.platform:
        flags += ["--jax.platform", args.platform]
    if args.compute_dtype:
        flags += ["--jax.compute_dtype", args.compute_dtype]
    # context_len is irrelevant in latents mode (no split); pass a small valid int.
    agent, _ = make_probe_agent(args.game, 1, flags, mode="latents")
    if args.ckpt_pkl:
        restore_from_pkl(agent, Path(args.ckpt_pkl))
    elif args.ckpt:
        restore_checkpoint(agent, Path(args.ckpt))
    else:
        print("need --ckpt (dir) or --ckpt-pkl (WM .pkl)"); return 2
    res = dump_latents(agent, Path(args.holdout), ep_batch=args.ep_batch)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{args.game}_{args.source}_latents.npz"
    np.savez_compressed(dest, **res)
    print(f"[dump_latents] wrote {dest} :: {res['deter'].shape[0]} frames, "
          f"deter{res['deter'].shape} stoch{res['stoch'].shape}, "
          f"levels {sorted(set(res['level_id'].tolist()))}, "
          f"transitions {int(res['transition'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
