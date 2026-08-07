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
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bc-ckpt", type=Path, default=None)
    p.add_argument("--wm-ckpt", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--w-bc", type=float, default=1.0)
    p.add_argument("--w-novelty", type=float, default=1.0)
    p.add_argument("--w-reward", type=float, default=10.0)
    p.add_argument("--w-change", type=float, default=0.5)
    p.add_argument("--max-click-candidates", type=int, default=64)
    p.add_argument("--no-ema", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import arc_agi

    from arc3_wm.env import ARC3GymEnv
    from arc3_wm.trm.agents import TRMAgent, run_episode
    from arc3_wm.trm.config import AgentConfig
    from arc3_wm.trm.training import load_policy, load_wm, resolve_device

    device = resolve_device(args.device)
    policy = wm = None
    if args.bc_ckpt:
        policy = load_policy(args.bc_ckpt, device=device, use_ema=not args.no_ema)
    if args.wm_ckpt:
        wm = load_wm(args.wm_ckpt, device=device, use_ema=not args.no_ema)

    agent_cfg = AgentConfig(
        use_bc=policy is not None,
        use_wm=wm is not None,
        w_bc=args.w_bc,
        w_novelty=args.w_novelty,
        w_reward=args.w_reward,
        w_change=args.w_change,
        epsilon=args.epsilon,
        max_click_candidates=args.max_click_candidates,
        seed=args.seed,
    )
    agent = TRMAgent(agent_cfg, policy=policy, world_model=wm, device=device)

    arcade = arc_agi.Arcade()
    env = ARC3GymEnv(game_id=args.game, arcade=arcade, seed=args.seed,
                     max_steps=args.max_actions)

    args.out.mkdir(parents=True, exist_ok=True)
    sink = args.out / "eval_episodes.jsonl"
    summary = {
        "game": args.game,
        "episodes": 0,
        "level_clears": 0,
        "wins": 0,
        "total_actions": 0,
        "config": {
            "agent": vars(args) | {"out": str(args.out),
                                   "bc_ckpt": str(args.bc_ckpt),
                                   "wm_ckpt": str(args.wm_ckpt)},
        },
    }
    start = time.time()
    with open(sink, "w") as fh:
        for ep in range(args.episodes):
            record = run_episode(env, agent, max_actions=args.max_actions)
            fh.write(json.dumps({"rewards": record["rewards"],
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
