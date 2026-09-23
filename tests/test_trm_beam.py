"""Spec for depth-2 beam planning and bc_score modes in TRMAgent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.action_space import build_mask  # noqa: E402
from arc3_wm.trm.agents import TRMAgent  # noqa: E402
from arc3_wm.trm.config import AgentConfig  # noqa: E402


class TwoStepStubWM:
    """Deterministic toy dynamics over frame fill values.

    States are uniform grids identified by their fill value. Action 0:
    0 -> 1 -> 1 (dead end: state 1 loops). Action 1: 0 -> 2 -> 3.
    All rewards/state heads absent, so scoring is novelty-only.
    """

    TRANSITIONS = {
        (0, 0): 1, (1, 0): 1, (2, 0): 2, (3, 0): 3,
        (0, 1): 2, (1, 1): 1, (2, 1): 3, (3, 1): 3,
    }

    def predict(self, grids, actions, max_steps=None):
        import torch as t

        b = grids.shape[0]
        logits = t.zeros(b, 64, 64, 16)
        for i in range(b):
            fill = int(grids[i, 0, 0])
            nxt = self.TRANSITIONS[(fill, int(actions[i]))]
            logits[i, :, :, nxt] = 10.0

        class Out:
            next_logits = logits
            reward_logit = None
            state_logits = None
            change_logits = None
            carry = None

        return Out()


def _agent(depth):
    cfg = AgentConfig(
        use_bc=False, use_wm=True, epsilon=0.0, w_change=0.0,
        plan_depth=depth, beam_width=2, second_step_candidates=4,
        plan_discount=0.5, seed=0,
    )
    return TRMAgent(cfg, world_model=TwoStepStubWM())


def test_depth2_sees_past_equal_first_step():
    # From state 0: action 0 -> state 1 (dead end), action 1 -> state 2
    # (leads on to 3). Make both one-step successors equally stale, so
    # depth-1 ties (and with seed 0 may pick either), while depth-2 must
    # strictly prefer action 1 whose second step reaches fresh state 3.
    grid = np.zeros((64, 64), dtype=np.uint8)
    mask = build_mask([1, 2])
    a1 = _agent(1)
    a2 = _agent(2)
    for agent in (a1, a2):
        for fill in (1, 2):
            g = np.full((64, 64), fill, dtype=np.uint8)
            for _ in range(3):
                agent.memory.observe(g)
    assert a2.act(grid, mask) == 1
    # Depth-1 has no signal to break the tie deterministically toward 1;
    # the test only pins that depth-2 does.


def test_depth2_respects_beam_and_runs():
    agent = _agent(2)
    grid = np.zeros((64, 64), dtype=np.uint8)
    mask = build_mask([1, 2])
    action = agent.act(grid, mask)
    assert action in (0, 1)


def test_bc_score_prob_bounds_component():
    class StubPolicy:
        def act(self, g, mask=None, temperature=0.0):
            import torch as t

            logits = t.full((1, 4102), -30.0)
            logits[0, 0] = 30.0

            class Out:
                flat_logits = logits

            return t.tensor([0]), Out()

    cfg = AgentConfig(use_bc=True, use_wm=False, epsilon=0.0,
                      bc_score="prob", w_bc=1.0, seed=0)
    agent = TRMAgent(cfg, policy=StubPolicy())
    mask = build_mask([1, 2])
    assert agent.act(np.zeros((64, 64), dtype=np.uint8), mask) == 0


def test_plan_depth_validation():
    with pytest.raises(ValueError):
        AgentConfig(plan_depth=3)
    with pytest.raises(ValueError):
        AgentConfig(bc_score="logits")
