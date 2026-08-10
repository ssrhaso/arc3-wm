#!/usr/bin/env python
"""Evaluate a TRM agent composition online and write RHAE-ready episodes.

Writes ``eval_episodes.jsonl`` in the EvalRewardSink schema, so the
existing ``scripts/compute_rhae.py`` consumes the output unchanged.

Compositions (mirroring AgentConfig):
    --bc-ckpt only               -> BC policy agent
    --wm-ckpt only               -> model-based novelty planner
    --bc-ckpt and --wm-ckpt      -> hybrid
    neither                      -> masked random baseline

Example:
    python scripts/trm_eval_agent.py --game vc33 --episodes 50 \
        --bc-ckpt checkpoints/trm_bc/vc33/best.pt \
        --wm-ckpt checkpoints/trm_wm/vc33/best.pt \
        --out results/trm/vc33-hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--agent", choices=["score", "graph"], default="score",
                   help="score = weighted-component agent; graph = state-graph scientist loop")
    p.add_argument("--max-click-objects", type=int, default=24)
    p.add_argument("--wm-noop-prune", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bc-ckpt", type=Path, default=None)
    p.add_argument("--wm-ckpt", type=Path, nargs="+", default=None,
                   help="one checkpoint, or several for an ensemble")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--w-bc", type=float, default=1.0)
    p.add_argument("--w-novelty", type=float, default=1.0)
    p.add_argument("--w-reward", type=float, default=10.0)
    p.add_argument("--w-change", type=float, default=0.5)
    p.add_argument("--w-disagree", type=float, default=0.0)
    p.add_argument("--max-click-candidates", type=int, default=64)
    p.add_argument("--wm-predict-steps", type=int, default=2)
    p.add_argument("--bc-score", choices=["logp", "prob"], default="logp")
    p.add_argument("--plan-depth", type=int, default=1)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument(
        "--resume", action="store_true",
        help="append to an existing eval_episodes.jsonl instead of restarting",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import arc_agi

    from arc3_wm.env import ARC3GymEnv
    from arc3_wm.trm.agents import TRMAgent, run_episode
    from arc3_wm.trm.config import AgentConfig, GraphAgentConfig
    from arc3_wm.trm.graph_agent import GraphAgent, run_graph_episode
    from arc3_wm.trm.training import load_policy, load_wm, resolve_device

    device = resolve_device(args.device)
    policy = wm = None
    if args.bc_ckpt:
        policy = load_policy(args.bc_ckpt, device=device, use_ema=not args.no_ema)
    if args.wm_ckpt:
        models = [load_wm(c, device=device, use_ema=not args.no_ema) for c in args.wm_ckpt]
        wm = models if len(models) > 1 else models[0]

    if args.agent == "graph":
        graph_cfg = GraphAgentConfig(
            max_click_objects=args.max_click_objects,
            wm_noop_prune=args.wm_noop_prune,
            wm_predict_steps=args.wm_predict_steps,
            w_bc=args.w_bc if policy is not None else 0.0,
            seed=args.seed,
        )
        agent = GraphAgent(graph_cfg, world_model=wm, policy=policy, device=device)
        episode_fn = lambda env, agent_, max_actions: run_graph_episode(
            env, agent_, max_actions=max_actions
        )
        return _run_eval(args, agent, episode_fn)

    agent_cfg = AgentConfig(
        use_bc=policy is not None,
        use_wm=wm is not None,
        w_bc=args.w_bc,
        w_novelty=args.w_novelty,
        w_reward=args.w_reward,
        w_change=args.w_change,
        w_disagree=args.w_disagree,
        epsilon=args.epsilon,
        max_click_candidates=args.max_click_candidates,
        wm_predict_steps=args.wm_predict_steps,
        bc_score=args.bc_score,
        plan_depth=args.plan_depth,
        beam_width=args.beam_width,
        seed=args.seed,
    )
    agent = TRMAgent(agent_cfg, policy=policy, world_model=wm, device=device)
    episode_fn = lambda env, agent_, max_actions: run_episode(
        env, agent_, max_actions=max_actions
    )
    return _run_eval(args, agent, episode_fn)


def _run_eval(args, agent, episode_fn) -> int:
    import arc_agi

    from arc3_wm.env import ARC3GymEnv

    arcade = arc_agi.Arcade()
    env = ARC3GymEnv(game_id=args.game, arcade=arcade, seed=args.seed,
                     max_steps=args.max_actions)

    args.out.mkdir(parents=True, exist_ok=True)
    sink = args.out / "eval_episodes.jsonl"
    done_episodes = 0
    if args.resume and sink.exists():
        done_episodes = sum(1 for _ in open(sink))
    summary = {
        "game": args.game,
        "episodes": 0,
        "level_clears": 0,
        "wins": 0,
        "total_actions": 0,
        "resumed_at": done_episodes or None,
        "config": {
            "agent": vars(args) | {"out": str(args.out),
                                   "bc_ckpt": str(args.bc_ckpt),
                                   "wm_ckpt": str(args.wm_ckpt)},
        },
    }
    start = time.time()
    mode = "a" if (args.resume and done_episodes) else "w"
    with open(sink, mode) as fh:
        for ep in range(done_episodes, args.episodes):
            record = episode_fn(env, agent, args.max_actions)
            # compute_rhae expects the DV3 stream shape: rewards[0] is the
            # initial-obs step (no action taken) and is skipped there, so
            # prepend it; every later entry is one action's reward.
            fh.write(json.dumps({"rewards": [0.0] + record["rewards"],
                                 "terminal_state": record["terminal_state"]}) + "\n")
            fh.flush()
            summary["episodes"] += 1
            summary["level_clears"] += int(sum(record["rewards"]))
            summary["wins"] += int(record["terminal_state"] == "WIN")
            summary["total_actions"] += record["steps"]
            if (ep + 1) % 5 == 0:
                elapsed = time.time() - start
                print(
                    f"ep {ep + 1}/{args.episodes}: clears={summary['level_clears']} "
                    f"wins={summary['wins']} actions={summary['total_actions']} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
