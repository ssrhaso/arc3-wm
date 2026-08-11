#!/usr/bin/env python
"""Reproduce one eval episode and render it as a filmstrip.

Replays episodes 0..N with the exact seeding of ``trm_eval_agent.py`` (env
seeded once, RNG state carried across episodes), captures the target
episode's frames, and verifies the reproduced reward vector against the
recorded ``eval_episodes.jsonl`` line before drawing anything.

Example:
    python scripts/trm_episode_film.py --game vc33 --episode 1 \
        --bc-ckpt checkpoints/trm_bc/vc33/best.pt \
        --record results/trm/vc33-bc/eval_episodes.jsonl \
        --out film_vc33_ep1.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def action_name(idx: int) -> str:
    from arc3_wm.action_space import ACTION6_BASE, ACTION7_INDEX, GRID

    if idx < ACTION6_BASE:
        return f"ACTION{idx + 1}"
    if idx == ACTION7_INDEX:
        return "ACTION7"
    rel = idx - ACTION6_BASE
    return f"CLICK({rel // GRID},{rel % GRID})"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--episode", type=int, default=0,
                   help="0-based episode index to capture")
    p.add_argument("--bc-ckpt", type=Path, default=None)
    p.add_argument("--wm-ckpt", type=Path, nargs="+", default=None)
    p.add_argument("--record", type=Path, default=None,
                   help="eval_episodes.jsonl to verify the replay against")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--panels", type=int, default=10)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    import arc_agi
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from arc3_wm.dynamics_probe import quantize_to_palette
    from arc3_wm.env import ARC3GymEnv
    from arc3_wm.trm.agents import TRMAgent
    from arc3_wm.trm.config import AgentConfig
    from arc3_wm.trm.training import load_policy, load_wm, resolve_device

    device = resolve_device(args.device)
    policy = load_policy(args.bc_ckpt, device=device) if args.bc_ckpt else None
    wm = None
    if args.wm_ckpt:
        models = [load_wm(c, device=device) for c in args.wm_ckpt]
        wm = models if len(models) > 1 else models[0]
    cfg = AgentConfig(use_bc=policy is not None, use_wm=wm is not None,
                      epsilon=args.epsilon, seed=args.seed)
    agent = TRMAgent(cfg, policy=policy, world_model=wm, device=device)

    arcade = arc_agi.Arcade()
    env = ARC3GymEnv(game_id=args.game, arcade=arcade, seed=args.seed,
                     max_steps=args.max_actions)

    trace = None
    for ep in range(args.episode + 1):
        obs, info = env.reset()
        agent.reset()
        capture = ep == args.episode
        frames = [np.asarray(obs, dtype=np.uint8).copy()] if capture else None
        actions: list[int] = []
        rewards: list[float] = []
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < args.max_actions:
            grid = np.asarray(quantize_to_palette(obs), dtype=np.uint8)
            action = agent.act(grid, np.asarray(info["action_mask"], dtype=bool))
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            steps += 1
            if capture:
                actions.append(int(action))
                frames.append(np.asarray(obs, dtype=np.uint8).copy())
        if capture:
            trace = {"frames": frames, "actions": actions, "rewards": rewards,
                     "terminal": info.get("state")}

    verified = None
    if args.record:
        lines = args.record.read_text().splitlines()
        rec = json.loads(lines[args.episode])
        verified = rec["rewards"][1:] == trace["rewards"] \
            and rec["terminal_state"] == trace["terminal"]
        print(f"replay matches recorded episode: {verified} "
              f"({len(trace['rewards'])} actions, "
              f"clears at {[i + 1 for i, r in enumerate(trace['rewards']) if r > 0]})")

    n = len(trace["actions"])
    clears = [i for i, r in enumerate(trace["rewards"]) if r > 0]
    keep = {0, n - 1} | set(clears)
    for c in clears:  # frame just before each clear shows the setup
        keep.add(max(0, c - 1))
    even = np.linspace(0, n - 1, args.panels).astype(int)
    for s in even:
        if len(keep) >= args.panels:
            break
        keep.add(int(s))
    steps_shown = sorted(keep)[: args.panels]

    fig, axes = plt.subplots(1, len(steps_shown) + 1,
                             figsize=(1.9 * (len(steps_shown) + 1), 2.6))
    axes[0].imshow(trace["frames"][0], interpolation="nearest")
    axes[0].set_title("start", fontsize=8)
    for ax, s in zip(axes[1:], steps_shown):
        ax.imshow(trace["frames"][s + 1], interpolation="nearest")
        label = f"t={s + 1}\n{action_name(trace['actions'][s])}"
        if trace["rewards"][s] > 0:
            label += "\nLEVEL CLEAR"
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:green")
                spine.set_linewidth(3)
        ax.set_title(label, fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    tag = "VERIFIED vs recorded eval" if verified else "unverified replay"
    fig.suptitle(
        f"{args.game} eval episode {args.episode}: {n} actions, "
        f"{len(clears)} level clear(s), terminal={trace['terminal']}  [{tag}]",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0 if verified in (True, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
