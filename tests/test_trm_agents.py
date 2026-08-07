"""Spec for arc3_wm.trm.agents - candidate pruning, novelty, composition."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.action_space import ACTION6_BASE, ACTION7_INDEX, build_mask  # noqa: E402
from arc3_wm.trm.agents import (  # noqa: E402
    NoveltyMemory,
    TRMAgent,
    candidate_actions,
    salient_click_cells,
)
from arc3_wm.trm.config import (  # noqa: E402
    AgentConfig,
    PolicyConfig,
    TokenizerConfig,
    TRMCoreConfig,
    WorldModelConfig,
)
from arc3_wm.trm.policy import TRMPolicy  # noqa: E402
from arc3_wm.trm.world_model import TRMWorldModel  # noqa: E402

TINY_CORE = TRMCoreConfig(d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1, halt_max_steps=1)
TINY_TOK = TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4)


def rng():
    return np.random.default_rng(0)


def test_salient_cells_prioritise_changes_then_foreground():
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[5, 5] = 3  # foreground
    prev = grid.copy()
    prev[2, 2] = 1  # cell (2,2) changed since prev
    cells = salient_click_cells(grid, prev, max_cells=4, rng=rng())
    assert cells[0] == 2 * 64 + 2
    assert 5 * 64 + 5 in cells.tolist()


def test_candidate_actions_respect_mask():
    mask = build_mask([1, 3])  # only ACTION1 and ACTION3
    grid = np.zeros((64, 64), dtype=np.uint8)
    cands = candidate_actions(mask, grid, None, 8, rng())
    assert set(cands.tolist()) == {0, 2}


def test_candidate_actions_include_pruned_clicks():
    mask = build_mask([6])
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[1, 1] = 5
    cands = candidate_actions(mask, grid, None, 4, rng())
    assert len(cands) <= 4
    assert all(ACTION6_BASE <= a < ACTION7_INDEX for a in cands)
    assert ACTION6_BASE + 1 * 64 + 1 in cands.tolist()


def test_novelty_decays_with_visits():
    mem = NoveltyMemory(power=0.5)
    g = np.zeros((64, 64), dtype=np.uint8)
    assert mem.novelty(g) == 1.0
    mem.observe(g)
    first = mem.novelty(g)
    mem.observe(g)
    assert mem.novelty(g) < first < 1.0


def _mini_models():
    torch.manual_seed(0)
    pol = TRMPolicy(PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK))
    wm = TRMWorldModel(WorldModelConfig(core=TINY_CORE, tokenizer=TINY_TOK))
    return pol, wm


def test_agent_component_requirements():
    with pytest.raises(ValueError, match="policy"):
        TRMAgent(AgentConfig(use_bc=True, use_wm=False))
    with pytest.raises(ValueError, match="world_model"):
        TRMAgent(AgentConfig(use_bc=False, use_wm=True))


@pytest.mark.parametrize(
    "use_bc,use_wm",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_agent_compositions_produce_masked_actions(use_bc, use_wm):
    pol, wm = _mini_models()
    cfg = AgentConfig(
        use_bc=use_bc, use_wm=use_wm, epsilon=0.0, max_click_candidates=8, seed=1
    )
    agent = TRMAgent(
        cfg,
        policy=pol if use_bc else None,
        world_model=wm if use_wm else None,
    )
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[3, 3] = 2
    mask = build_mask([1, 2, 6])
    for _ in range(3):
        action = agent.act(grid, mask)
        assert mask[action]


def test_agent_novelty_prefers_state_changing_action():
    # World model stub: action 0 keeps the grid, action 1 changes it.
    class StubWM:
        def predict(self, grids, actions, max_steps=None):
            import torch as t

            b = grids.shape[0]
            logits = t.zeros(b, 64, 64, 16)
            for i, a in enumerate(actions.tolist()):
                target = grids[i] if a == 0 else (grids[i] + 1) % 16
                logits[i].scatter_(-1, target[..., None], 10.0)

            class Out:
                next_logits = logits
                reward_logit = None
                state_logits = None
                change_logits = None
                carry = None

            return Out()

    cfg = AgentConfig(use_bc=False, use_wm=True, epsilon=0.0, w_change=0.0, seed=0)
    agent = TRMAgent(cfg, world_model=StubWM())
    grid = np.zeros((64, 64), dtype=np.uint8)
    mask = build_mask([1, 2])  # actions 0 and 1
    # Visit the current grid a few times: staying put must look stale.
    for _ in range(3):
        agent.memory.observe(grid)
    assert agent.act(grid, mask) == 1


def test_agent_epsilon_one_is_masked_random():
    cfg = AgentConfig(use_bc=False, use_wm=False, epsilon=1.0, seed=0)
    agent = TRMAgent(cfg)
    mask = build_mask([2])
    actions = {agent.act(np.zeros((64, 64), dtype=np.uint8), mask) for _ in range(5)}
    assert actions == {1}
