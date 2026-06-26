"""Stage 1 of the dynamics-competence probe: collect ground-truth episodes.

CPU-only, JAX-free. Produces, per (game, source), a flat-transition ``.npz`` of
real episodes that the JAX ``predict`` stage replays through a frozen world
model and the ``score`` stage compares against. See
``arc3_wm.dynamics_probe`` for the metrics and ``docs`` for the probe rationale.

Two sources (run both; they answer different questions):

* ``random`` - a masked-uniform random policy in the OFFLINE ``ARC3GymEnv``.
  These are *genuinely held out* (fresh samples the WM never trained on) from
  the same behavioural distribution the per-game runs trained under. Needs the
  game's ``environment_files/`` cached (``scripts/cache_env_files.py``).
* ``human`` - the human-replay corpus (``data/replays/<game>/``). Richer,
  goal-directed, board-changing dynamics: the harder fidelity test. These seeded
  the training buffer, so they are *in-distribution* - reported as such; the
  copy-baseline comparison in the score stage is what makes either source
  diagnostic regardless of held-out status.

Output schema (one ``.npz`` per game+source), a flat transition table the
predict stage segments by ``ep_id``:

    frames : (T, 64, 64, 3) uint8   - the observation AT step t
    actions: (T,)           int32   - flat action taken at frames[t]
                                       (convention B; sentinel 0 on the last
                                       step of each episode)
    rewards: (T,)           float32 - delta levels_completed at step t
    ep_id  : (T,)           int32   - episode index within this file
    step   : (T,)           int32   - 0-based position within the episode
    is_last: (T,)           bool    - terminal/truncation marker for the episode
    avail  : (T, 7)         bool    - ACTION1..7 availability (random source);
                                       all-False for human (unknown), see
                                       ``has_avail`` scalar
    game, source, has_avail         - 0-d metadata arrays

Usage::

    python scripts/probe_collect_holdout.py --game vc33 --source both \\
        --n-episodes 40 --outdir results/dynamics_probe/holdout
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# OFFLINE before importing arc_agi (env.py enforces it; .env also sets it).
os.environ.setdefault("OPERATION_MODE", "offline")

import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.replay_loader import load_replay_file  # noqa: E402

OBS_HW = 64
N_ACTION_TYPES = 7  # ACTION1..ACTION7 -> columns 0..6


def _avail_to_mask(available_actions: list[int]) -> np.ndarray:
    """List of GameAction ints (1..7) -> 7-bool type-availability vector."""
    m = np.zeros(N_ACTION_TYPES, dtype=bool)
    for a in available_actions:
        if 1 <= int(a) <= N_ACTION_TYPES:
            m[int(a) - 1] = True
    return m


def _pack(records: list[dict[str, Any]], game: str, source: str, has_avail: bool) -> dict:
    """Stack a list of per-step dicts into the flat-transition npz schema.

    Adds two per-frame labels used by the latent-probing stage
    (``scripts/probe_dump_latents.py`` + ``probe_fit_probes.py``):

    * ``level_id``   - levels cleared *before* this step (exclusive prefix sum of
      reward within the episode). The frame shows the pre-clear board, so the
      label is the level the frame belongs to. The ordinal "which level am I on"
      probe target. (Always 0 on games a random policy never clears.)
    * ``transition`` - bool, did a level-clear fire at this step (reward > 0).
      The binary "transition event" probe target.
    """
    if not records:
        raise RuntimeError(f"no transitions collected for {game}/{source}")
    from arc3_wm.probe_data import frame_labels

    frames = np.stack([r["frame"] for r in records]).astype(np.uint8)
    rewards = np.array([r["reward"] for r in records], dtype=np.float32)
    ep_id = np.array([r["ep_id"] for r in records], dtype=np.int32)
    level_id, transition = frame_labels(rewards, ep_id)
    return {
        "frames": frames,
        "actions": np.array([r["action"] for r in records], dtype=np.int32),
        "rewards": rewards,
        "ep_id": ep_id,
        "step": np.array([r["step"] for r in records], dtype=np.int32),
        "is_last": np.array([r["is_last"] for r in records], dtype=bool),
        "avail": np.stack([r["avail"] for r in records]).astype(bool),
        "level_id": level_id,
        "transition": transition,
        "game": np.array(game),
        "source": np.array(source),
        "has_avail": np.array(has_avail),
    }


def collect_random(
    game: str,
    *,
    n_episodes: int,
    max_steps: int,
    env_seed: int,
    action_seed: int,
    max_transitions: int | None,
) -> dict:
    """Masked-uniform random rollouts in the OFFLINE env -> npz dict."""
    from arc3_wm.env import ARC3GymEnv  # local import: needs env-files cached

    env = ARC3GymEnv(game, seed=env_seed, max_steps=max_steps)
    rng = np.random.default_rng(action_seed)
    records: list[dict[str, Any]] = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        step = 0
        while True:
            mask = np.asarray(info["action_mask"], dtype=bool)
            valid = np.flatnonzero(mask)
            # Empty valid set should not happen; fall back to ACTION5 no-op.
            a = int(rng.choice(valid)) if valid.size else 4
            avail = _avail_to_mask(info["available_actions"])
            nobs, reward, term, trunc, info = env.step(a)
            records.append(
                {
                    "frame": obs, "action": a, "reward": float(reward),
                    "ep_id": ep, "step": step, "is_last": bool(term or trunc),
                    "avail": avail,
                }
            )
            obs = nobs
            step += 1
            if term or trunc:
                # Record the terminal frame too (sentinel action), so the
                # predict stage has the full observed sequence to score against.
                records.append(
                    {
                        "frame": obs, "action": 0, "reward": 0.0,
                        "ep_id": ep, "step": step, "is_last": True,
                        "avail": _avail_to_mask(info["available_actions"]),
                    }
                )
                break
            if max_transitions and len(records) >= max_transitions:
                break
        if max_transitions and len(records) >= max_transitions:
            break
    env.close()
    return _pack(records, game, "random", has_avail=True)


def collect_human(
    game: str,
    *,
    replays_root: Path,
    max_episodes: int | None,
    max_transitions: int | None,
) -> dict:
    """Human-replay episodes for one game -> npz dict (no avail info)."""
    game_dir = Path(replays_root) / game
    files = sorted(game_dir.rglob("*.recording.jsonl"))
    if not files:
        raise FileNotFoundError(f"no replays under {game_dir}")
    records: list[dict[str, Any]] = []
    ep_id = 0
    no_avail = np.zeros(N_ACTION_TYPES, dtype=bool)
    for path in files:
        for episode in load_replay_file(path):
            n = len(episode)
            for i, sd in enumerate(episode):
                records.append(
                    {
                        "frame": np.asarray(sd["image"]),
                        "action": int(sd["action"]),
                        "reward": float(sd["reward"]),
                        "ep_id": ep_id,
                        "step": i,
                        "is_last": bool(sd["is_last"]) or (i == n - 1),
                        "avail": no_avail,
                    }
                )
            ep_id += 1
            if max_episodes and ep_id >= max_episodes:
                break
            if max_transitions and len(records) >= max_transitions:
                break
        if (max_episodes and ep_id >= max_episodes) or (
            max_transitions and len(records) >= max_transitions
        ):
            break
    return _pack(records, game, "human", has_avail=False)


def _summary(d: dict) -> str:
    n_eps = int(d["ep_id"].max()) + 1 if d["ep_id"].size else 0
    changed = float((d["frames"][1:] != d["frames"][:-1]).any(axis=(1, 2, 3)).mean()) if d["frames"].shape[0] > 1 else 0.0
    return (
        f"{str(d['game'])}/{str(d['source'])}: {d['frames'].shape[0]} transitions, "
        f"{n_eps} episodes, board-change rate {changed:.2f}, "
        f"reward>0 steps {int((d['rewards'] > 0).sum())}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--source", choices=["random", "human", "both"], default="both")
    p.add_argument("--n-episodes", type=int, default=40, help="random rollouts")
    p.add_argument("--max-steps", type=int, default=1000, help="random episode cap")
    p.add_argument("--max-human-episodes", type=int, default=None)
    p.add_argument("--max-transitions", type=int, default=None,
                   help="hard cap per source (board-static games can be long)")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--action-seed", type=int, default=0)
    p.add_argument("--replays-root", default="data/replays")
    p.add_argument("--outdir", default="results/dynamics_probe/holdout")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sources = ["random", "human"] if args.source == "both" else [args.source]
    for source in sources:
        if source == "random":
            d = collect_random(
                args.game, n_episodes=args.n_episodes, max_steps=args.max_steps,
                env_seed=args.env_seed, action_seed=args.action_seed,
                max_transitions=args.max_transitions,
            )
        else:
            d = collect_human(
                args.game, replays_root=Path(args.replays_root),
                max_episodes=args.max_human_episodes,
                max_transitions=args.max_transitions,
            )
        out = outdir / f"{args.game}_{source}.npz"
        np.savez_compressed(out, **d)
        print(f"[wrote] {out}  ::  {_summary(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
