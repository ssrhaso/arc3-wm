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


def _svg_frame(rgb: np.ndarray, cell: float) -> str:
    """One 64x64 RGB frame -> vector rects, merging horizontal runs and
    identical consecutive rows."""
    h, w, _ = rgb.shape
    rows = [rgb[y].tobytes() for y in range(h)]
    parts = []
    y = 0
    while y < h:
        y2 = y
        while y2 + 1 < h and rows[y2 + 1] == rows[y]:
            y2 += 1
        x = 0
        while x < w:
            c = rgb[y, x]
            x2 = x
            while x2 + 1 < w and (rgb[y, x2 + 1] == c).all():
                x2 += 1
            parts.append(
                f'<rect x="{x * cell:g}" y="{y * cell:g}" '
                f'width="{(x2 - x + 1) * cell:g}" '
                f'height="{(y2 - y + 1) * cell:g}" '
                f'fill="#{c[0]:02x}{c[1]:02x}{c[2]:02x}"/>'
            )
            x = x2 + 1
        y = y2 + 1
    return "".join(parts)


def render_svg(trace: dict, steps_shown: list[int], title: str,
               out: Path, cell: float = 2.0) -> None:
    side = 64 * cell
    pad, label_h, top = 10, 30, 26
    panels = [(-1, trace["frames"][0])] + [
        (s, trace["frames"][s + 1]) for s in steps_shown
    ]
    width = pad + len(panels) * (side + pad)
    height = top + side + label_h + pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:g} {height:g}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<text x="{width / 2:g}" y="16" font-size="11" '
        f'text-anchor="middle">{title}</text>',
    ]
    n_clears = 0
    for i, (s, frame) in enumerate(panels):
        ox = pad + i * (side + pad)
        parts.append(f'<g transform="translate({ox:g},{top})">')
        parts.append(_svg_frame(np.asarray(frame), cell))
        if s < 0:
            label, edge = "start", "#999"
        else:
            label = f"t={s + 1} {action_name(trace['actions'][s])}"
            edge = "#999"
            if trace["rewards"][s] > 0:
                n_clears += 1
                label += f" · LEVEL {n_clears} CLEAR"
                edge = "#2a9d2a"
            elif s == len(trace["actions"]) - 1 \
                    and trace["terminal"] == "GAME_OVER":
                label += " · GAME_OVER"
                edge = "#cc2222"
        parts.append(
            f'<rect x="0" y="0" width="{side:g}" height="{side:g}" '
            f'fill="none" stroke="{edge}" '
            f'stroke-width="{3 if edge != "#999" else 1}"/>'
        )
        for j, chunk in enumerate(label.split(" · ")):
            parts.append(
                f'<text x="{side / 2:g}" y="{side + 12 + 11 * j:g}" '
                f'font-size="8.5" text-anchor="middle" '
                f'fill="{edge if j else "#333"}">{chunk}</text>'
            )
        parts.append("</g>")
    parts.append("</svg>")
    out.write_text("".join(parts))


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

    tag = "VERIFIED vs recorded eval" if verified else "unverified replay"
    title = (f"{args.game} eval episode {args.episode}: {n} actions, "
             f"{len(clears)} level clear(s), terminal={trace['terminal']}  [{tag}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".svg":
        render_svg(trace, steps_shown, title, args.out)
        print(f"wrote {args.out}")
        return 0 if verified in (True, None) else 1

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
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0 if verified in (True, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
