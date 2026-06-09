"""Random-policy offline RHAE eval on a single ARC-AGI-3 game.

Purpose (Pre-submission, 2026-06-04): produce a *logged* random-policy RHAE
so the paper can state whether the trained per-game RHAE is above random.
For vc33 the trained warm seed-0 number is 0.0548 (= 1.15 x 1 / 21; only
level 1 ever cleared, in <=12 actions so the per-level score capped).

This is the cheapest possible path: a uniform-random agent in the REAL
OFFLINE arc_agi env. No world model, no DreamerV3 policy. It reuses the
exact eval substrate the trained run used so the comparison is honest:

  * env: ``arc3_wm.embodied_env.ARC3EmbodiedEnv`` - the same embodied
    adapter ``embodied.run.train_eval`` drove for the trained runs.
  * reward capture: ``arc3_wm.eval_reward_sink.EvalRewardSink`` - same
    ``{logdir}/eval_episodes.jsonl`` artifact, same format.
  * scoring: ``scripts.compute_rhae`` -> ``arc3_wm.rhae.RHAEAggregator``,
    against ``data/human_baselines.json`` (D-A/D-B, n>=2 covered subset).
    NOT a second hand-rolled RHAE path.

Two arms (session sign-off 2026-06-04, "run both"):

  unmasked : uniform over all 4102 flat actions. Matches the trained vc33
             eval, which applied NO masking (D11: the actor's
             ``train/ent/action`` sat at log(4102)=8.31923 exactly). This is
             the apples-to-apples comparator for the trained per-game RHAE.
  masked   : uniform over the currently-valid flat indices
             (``action_space.build_mask`` from ``fd.available_actions``).
             Satisfies the literal brief + the mask-respecting test. For
             vc33 the only live action type is ACTION6 (the 64x64 click), so
             masking removes just the 6 dead indices - the two arms are
             near-identical here by construction.

Eval protocol mirrors ``scripts/launch_phase4_proper.sh``: OFFLINE arc_agi,
single game, ``max_steps=1000`` (env default), natural WIN/GAME_OVER
termination (no per-level budget exists in the pipeline - the trained eval
episodes were short because the agent hit GAME_OVER, not a budget). Episode
variation comes from the action RNG, exactly as the trained run's variation
came from the stochastic actor (fixed construction seed).

CLI::

    python scripts/eval_random_rhae.py --game-id vc33 --n-episodes 300 \\
        --arms both --outdir results/random-eval

Writes ``{outdir}/{game}-{arm}/eval_episodes.jsonl`` per arm and a combined
``{outdir}/{game}/random_rhae_summary.json`` results artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Repo root on sys.path so ``scripts.compute_rhae`` and ``arc3_wm`` resolve
# when this file is run directly as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc3_wm.action_space import N_ACTIONS  # noqa: E402

# Trained reference: vc33 warm seed-0 Phase-4-proper (run group 98de390),
# per_game RHAE = 1.15 * 1 / sum(1..6) = 0.0547619... (the draft's 0.0548).
# Recomputed from the committed eval_episodes.jsonl via scripts/compute_rhae.py.
TRAINED_REF_RHAE = {"vc33": 0.05476190476190477}


def sample_action(rng: np.random.Generator, mask: np.ndarray, masked: bool) -> int:
    """Sample one flat action index.

    masked=True  -> uniform over the live indices (``mask`` True). Raises
                    ValueError on an all-dead mask (an env that reports no
                    available actions is a bug to surface, not to paper over).
    masked=False -> uniform over the full ``[0, N_ACTIONS)`` space, ignoring
                    ``mask`` (matches the trained unmasked actor; arc_agi
                    no-ops any dead index downstream).
    """
    if masked:
        valid = np.flatnonzero(np.asarray(mask, dtype=bool))
        if valid.size == 0:
            raise ValueError("empty action mask: env reports no available actions")
        return int(valid[int(rng.integers(valid.size))])
    return int(rng.integers(N_ACTIONS))


def run_random_eval(
    *,
    game_id: str,
    n_episodes: int,
    max_steps: int,
    masked: bool,
    env_seed: int,
    action_seed: int,
    sink_path: Path | str,
    arcade: Optional[Any] = None,
) -> dict:
    """Drive a uniform-random agent for ``n_episodes`` and write the per-episode
    reward streams to ``sink_path`` via ``EvalRewardSink``.

    Returns a diagnostics dict (episode/step counts, mask-violation count,
    seeds, sink path). The sink file is truncated first so re-runs and the
    two arms never mix into one artifact.
    """
    from arc3_wm.embodied_env import ARC3EmbodiedEnv
    from arc3_wm.eval_reward_sink import EvalRewardSink

    sink_path = Path(sink_path)
    sink_path.parent.mkdir(parents=True, exist_ok=True)
    if sink_path.exists():
        sink_path.unlink()

    env = ARC3EmbodiedEnv(game_id=game_id, seed=env_seed, max_steps=max_steps, arcade=arcade)
    sink = EvalRewardSink(env, sink_path)
    rng = np.random.default_rng(action_seed)

    total_steps = 0
    invalid_action_count = 0
    episode_lengths: list[int] = []

    try:
        for _ in range(n_episodes):
            # Driver-style reset: ARC3EmbodiedEnv emits is_first; the sink
            # buffers the initial-obs reward (0.0) and starts a fresh episode.
            sink.step({"action": 0, "reset": True})
            steps = 0
            while True:
                mask = sink.env.info["action_mask"]
                a = sample_action(rng, mask, masked)
                if not bool(mask[a]):
                    invalid_action_count += 1
                tx = sink.step({"action": int(a), "reset": False})
                steps += 1
                total_steps += 1
                if bool(tx["is_last"]):
                    break
            episode_lengths.append(steps)
    finally:
        env.close()

    return {
        "game_id": game_id,
        "masked": masked,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "env_seed": env_seed,
        "action_seed": action_seed,
        "total_steps": total_steps,
        # In masked mode this is the mask-respect guarantee (must be 0). In
        # unmasked mode it counts how often the uniform/4102 sampler landed on
        # a dead index that arc_agi no-op'd - informational only.
        "invalid_action_count": invalid_action_count,
        "episode_len_min": min(episode_lengths) if episode_lengths else 0,
        "episode_len_max": max(episode_lengths) if episode_lengths else 0,
        "episode_len_mean": (sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0,
        "sink_path": str(sink_path),
    }


def score_episodes(
    *,
    sink_path: Path | str,
    game_id: str,
    baselines_path: Path | str,
    step: Optional[int] = None,
) -> tuple[dict, int]:
    """Score an ``eval_episodes.jsonl`` through the SAME post-hoc path that
    produced the trained numbers. Returns ``(metrics, n_episodes)``."""
    from scripts.compute_rhae import compute_rhae, load_episodes_from_jsonl

    episodes = load_episodes_from_jsonl(Path(sink_path))
    baselines = json.loads(Path(baselines_path).read_text(encoding="utf-8"))
    metrics = compute_rhae(
        episodes_rewards=episodes, game_id=game_id, baselines=baselines
    )
    return metrics, len(episodes)


def _ratio(trained: Optional[float], random_score: float):
    """trained / random, JSON-safe.

    Returns None when there is no trained ref; the string ``"inf"`` when
    random is exactly zero but trained is positive (trained strictly above a
    zero floor - the headline result for vc33); otherwise the float ratio.
    A string (not ``float('inf')``) keeps the results artifact valid JSON.
    """
    if trained is None:
        return None
    if random_score == 0.0:
        return "inf" if trained > 0 else 0.0
    return trained / random_score


def _arm_result(
    *,
    game_id: str,
    masked: bool,
    n_episodes: int,
    max_steps: int,
    env_seed: int,
    action_seed: int,
    baselines_path: Path,
    outdir: Path,
    arcade: Optional[Any],
    step: Optional[int],
) -> dict:
    arm = "masked" if masked else "unmasked"
    sink_path = outdir / f"{game_id}-{arm}" / "eval_episodes.jsonl"
    diag = run_random_eval(
        game_id=game_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        masked=masked,
        env_seed=env_seed,
        action_seed=action_seed,
        sink_path=sink_path,
        arcade=arcade,
    )
    metrics, n_eps = score_episodes(
        sink_path=sink_path, game_id=game_id, baselines_path=baselines_path, step=step
    )
    per_game = float(metrics[f"eval/rhae/per_game/{game_id}"])
    levels_completed = int(metrics[f"eval/rhae/levels_completed/{game_id}"])
    level_scores = {
        int(k.rsplit("/", 1)[1]): float(v)
        for k, v in metrics.items()
        if k.startswith(f"eval/rhae/level_scores/{game_id}/")
    }
    trained = TRAINED_REF_RHAE.get(game_id)
    return {
        "arm": arm,
        "random_per_game_rhae": per_game,
        "levels_completed": levels_completed,
        "level_scores": level_scores,
        "n_eval_episodes": n_eps,
        "trained_ref_rhae": trained,
        "trained_over_random_ratio": _ratio(trained, per_game),
        "diagnostics": diag,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Uniform-random offline RHAE eval (masked and/or unmasked) on one "
            "ARC-AGI-3 game, scored through scripts.compute_rhae."
        )
    )
    parser.add_argument("--game-id", default="vc33")
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--arms",
        default="both",
        choices=["both", "masked", "unmasked"],
        help="Which sampling arm(s) to run (default: both).",
    )
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--baselines", type=Path, default=Path("data/human_baselines.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results/random-eval"))
    parser.add_argument(
        "--step",
        type=int,
        default=500000,
        help="Trained-run step the comparison is against (label only).",
    )
    args = parser.parse_args(argv)

    if not args.baselines.exists():
        raise FileNotFoundError(f"baselines file not found: {args.baselines}")

    import arc_agi

    arcade = arc_agi.Arcade()  # one OFFLINE arcade shared across arms

    arms = ["masked", "unmasked"] if args.arms == "both" else [args.arms]
    game_dir = args.outdir / args.game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    arm_results = []
    for arm in arms:
        res = _arm_result(
            game_id=args.game_id,
            masked=(arm == "masked"),
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            env_seed=args.env_seed,
            action_seed=args.action_seed,
            baselines_path=args.baselines,
            outdir=args.outdir,
            arcade=arcade,
            step=args.step,
        )
        arm_results.append(res)

    summary = {
        "game_id": args.game_id,
        "step": args.step,
        "trained_ref_rhae": TRAINED_REF_RHAE.get(args.game_id),
        "config": {
            "n_episodes": args.n_episodes,
            "max_steps": args.max_steps,
            "env_seed": args.env_seed,
            "action_seed": args.action_seed,
            "baselines": str(args.baselines),
        },
        "arms": arm_results,
    }
    summary_path = game_dir / "random_rhae_summary.json"
    # allow_nan=False guarantees the artifact is valid JSON (no Infinity/NaN
    # tokens); _ratio already stringifies the inf case.
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"=== random-policy RHAE: {args.game_id} (trained ref = "
          f"{TRAINED_REF_RHAE.get(args.game_id)}) ===")
    for res in arm_results:
        ratio = res["trained_over_random_ratio"]
        ratio_s = "n/a" if ratio is None else ("inf" if ratio == "inf" else f"{ratio:.2f}x")
        print(
            f"  {res['arm']:>8}: random_rhae={res['random_per_game_rhae']:.6f} "
            f"levels_cleared={res['levels_completed']} "
            f"n_eps={res['n_eval_episodes']} "
            f"trained/random={ratio_s} "
            f"(invalid_actions={res['diagnostics']['invalid_action_count']}, "
            f"ep_len mean={res['diagnostics']['episode_len_mean']:.1f})"
        )
    print(f"  artifact: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
