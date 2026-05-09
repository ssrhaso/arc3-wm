"""End-to-end smoke: 100 random-agent episodes on vc33 via ARC3GymEnv.

Phase-1 milestone (1) gate. Asserts:
- 100 episodes complete with no exceptions.
- Every step's chosen action lies in info["action_mask"].
- The toolkit scorecard is populated (per-level baselines / actions present).
- Total step throughput is at least 200 FPS on this laptop (safety floor; the
  Phase-0 smoke clocked ~921 FPS, so 200 is generous).
"""
from __future__ import annotations

import random
import time

import numpy as np
import pytest

from arc3_wm.action_space import N_ACTIONS
from arc3_wm.env import ARC3GymEnv
import arc_agi


N_EPISODES = 100
SEED = 0
MAX_STEPS_PER_EPISODE = 200  # vc33 random clicks GAME_OVER ~50 steps; 200 is plenty.


def _sample_masked(mask: np.ndarray, rng: random.Random) -> int:
    idxs = np.flatnonzero(mask)
    assert idxs.size > 0, "no available actions — mask is empty"
    return int(rng.choice(idxs.tolist()))


@pytest.mark.timeout(120)
def test_random_agent_100_episodes_vc33():
    arcade = arc_agi.Arcade()
    assert arcade.operation_mode == arc_agi.OperationMode.OFFLINE
    env = ARC3GymEnv(
        game_id="vc33", seed=SEED, max_steps=MAX_STEPS_PER_EPISODE, arcade=arcade
    )
    rng = random.Random(SEED)

    total_steps = 0
    mask_violations = 0
    t0 = time.perf_counter()
    try:
        for _ in range(N_EPISODES):
            _, info = env.reset()
            done = False
            while not done:
                mask = info["action_mask"]
                a = _sample_masked(mask, rng)
                # Sanity: chosen action must be live.
                if not mask[a]:
                    mask_violations += 1
                _, _, term, trunc, info = env.step(a)
                total_steps += 1
                done = term or trunc
    finally:
        env.close()

    wall = time.perf_counter() - t0
    fps = total_steps / wall if wall > 0 else float("inf")

    assert mask_violations == 0
    assert total_steps >= N_EPISODES, "every episode should run at least 1 step"
    # Loose FPS floor — protects against pathological regressions.
    assert fps > 200, f"FPS regression: {fps:.0f} (Phase 0 baseline ~921)"

    # Scorecard sanity.
    sc = arcade.get_scorecard()
    assert sc is not None
    dump = sc.model_dump()
    assert dump["total_actions"] >= total_steps
    # Per-level baselines should be the vc33 vector from EnvironmentInfo.
    runs = dump["environments"][0]["runs"]
    assert any(r["level_baseline_actions"] == [7, 18, 44, 61, 131, 34, 152] for r in runs)
