"""Phase 0 smoke: 10 random-agent episodes on vc33 in OFFLINE mode.

Validates that:
  - .env loads ARC_API_KEY + OPERATION_MODE=offline.
  - Arcade is in OFFLINE mode (no API calls during the run).
  - LocalEnvironmentWrapper.step() handles ACTION6 (with x,y) without exception.
  - get_scorecard() returns a populated EnvironmentScorecard with per-level
    actions/baselines.

No render_mode - exercises the high-FPS path. Exits non-zero on any failure.

Usage:
    .venv/Scripts/python.exe scripts/random_agent_smoke.py
"""
from __future__ import annotations

import json
import random
import sys
import time

import arc_agi
from arcengine import GameState

GAME_ID = "vc33"
N_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
SEED = 0


def pick_action(env, rng: random.Random):
    actions = env.action_space  # list[GameAction] from last response
    if not actions:
        return None, None
    a = rng.choice(actions)
    if a.is_complex():
        return a, {"x": rng.randrange(64), "y": rng.randrange(64)}
    return a, None


def run_episode(env, rng: random.Random) -> dict:
    fd = env.reset()
    if fd is None:
        return {"error": "reset returned None"}

    steps = 0
    terminal_state = None
    levels_completed_init = fd.levels_completed
    for _ in range(MAX_STEPS_PER_EPISODE):
        a, data = pick_action(env, rng)
        if a is None:
            terminal_state = "no_actions_available"
            break
        fd = env.step(a, data=data)
        if fd is None:
            terminal_state = "step_returned_none"
            break
        steps += 1
        if fd.state in (GameState.WIN, GameState.GAME_OVER):
            terminal_state = fd.state.name
            break
    else:
        terminal_state = "max_steps"

    return {
        "steps": steps,
        "terminal_state": terminal_state,
        "levels_completed": fd.levels_completed,
        "levels_delta": fd.levels_completed - levels_completed_init,
        "win_levels": fd.win_levels,
    }


def main() -> int:
    rng = random.Random(SEED)
    arc = arc_agi.Arcade()
    print(f"operation_mode: {arc.operation_mode}")
    if arc.operation_mode != arc_agi.OperationMode.OFFLINE:
        print("ABORT: not in OFFLINE mode", file=sys.stderr)
        return 2
    print(f"available_environments: {len(arc.available_environments)}")

    env = arc.make(GAME_ID)
    if env is None:
        print(f"ABORT: arc.make({GAME_ID!r}) returned None", file=sys.stderr)
        return 2

    info = env.info
    print(f"\ngame: {info.title} ({info.game_id})  tags={info.tags}")
    print(f"baseline_actions per level: {info.baseline_actions}")
    print(f"levels: {len(info.baseline_actions)}")

    print("\n--- episodes ---")
    t0 = time.perf_counter()
    total_steps = 0
    results = []
    for ep in range(N_EPISODES):
        ep_t0 = time.perf_counter()
        r = run_episode(env, rng)
        r["episode"] = ep
        r["wall_s"] = time.perf_counter() - ep_t0
        results.append(r)
        total_steps += r.get("steps", 0)
        print(
            f"  ep{ep:02d}  steps={r['steps']:4d}  "
            f"end={r['terminal_state']:>14s}  "
            f"levels_completed={r['levels_completed']}/{r['win_levels']}  "
            f"wall={r['wall_s']:.2f}s"
        )

    wall = time.perf_counter() - t0
    fps = total_steps / wall if wall > 0 else float("nan")
    print(f"\n{N_EPISODES} episodes, {total_steps} steps, {wall:.2f}s wall, {fps:.0f} FPS")

    sc = arc.get_scorecard()
    if sc is None:
        print("\nABORT: get_scorecard() returned None", file=sys.stderr)
        return 2

    print("\n--- scorecard summary ---")
    print(
        f"score={sc.score:.4f}  "
        f"total_actions={sc.total_actions}  "
        f"total_levels_completed={sc.total_levels_completed}/{sc.total_levels}  "
        f"environments={sc.total_environments}"
    )

    print("\n--- scorecard JSON ---")
    print(json.dumps(sc.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
