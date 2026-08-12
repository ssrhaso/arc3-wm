"""Spec for arc3_wm.trm.agents - candidate pruning, novelty, composition."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arc3_wm.action_space import (  # noqa: E402
    ACTION6_BASE,
    ACTION7_INDEX,
    N_ACTIONS,
    build_mask,
)
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


def test_bc_temperature_samples_full_distribution():
    torch.manual_seed(0)
    pol = TRMPolicy(PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK))
    cfg = AgentConfig(use_bc=True, use_wm=False, epsilon=0.0,
                      bc_temperature=1.0, seed=3)
    agent = TRMAgent(cfg, policy=pol)
    grid = np.zeros((64, 64), dtype=np.uint8)
    mask = np.ones(N_ACTIONS, dtype=bool)
    a1 = [agent.act(grid, mask) for _ in range(8)]
    assert all(0 <= a < N_ACTIONS for a in a1)
    assert len(set(a1)) > 1  # sampling, not argmax
    agent2 = TRMAgent(cfg, policy=pol)
    a2 = [agent2.act(grid, mask) for _ in range(8)]
    assert a1 == a2  # seeded generator -> reproducible


def test_act_output_fields_frozen_consistently():
    # With a random halt head some samples halt earlier than others; every
    # returned field must come from the same (first-halt) step, so
    # reassembling flat from type+click must reproduce it exactly.
    torch.manual_seed(0)
    from arc3_wm.trm.config import TRMCoreConfig, TokenizerConfig
    core = TRMCoreConfig(d_model=32, n_heads=4, n_layers=1, l_cycles=1,
                         h_cycles=1, n_supervision=3, halt_max_steps=3)
    pol = TRMPolicy(PolicyConfig(core=core, tokenizer=TINY_TOK))
    torch.nn.init.normal_(pol.core.q_head.weight, std=1.0)
    grid = torch.randint(0, 16, (6, 64, 64))
    _, out = pol.act(grid, temperature=0.0)
    from arc3_wm.trm.tokenizer import assemble_flat_logits
    rebuilt = assemble_flat_logits(out.type_logits, out.click_logits)
    assert torch.equal(rebuilt, out.flat_logits)


def test_stablemax_mask_bias_leak_is_negligible():
    from arc3_wm.trm.core import stablemax_cross_entropy
    from arc3_wm.trm.policy import MASK_BIAS
    logits = torch.full((1, N_ACTIONS), float(MASK_BIAS))
    logits[0, :6] = -100.0  # adversarially weak valid actions
    x = logits.float()
    log_s = torch.where(x >= 0, torch.log1p(x.clamp(min=0)),
                        -torch.log1p((-x).clamp(min=0)))
    p = torch.softmax(log_s, dim=-1)
    assert p[0, 6:].sum().item() < 1e-3  # leaked mass to 4096 masked actions


def test_plan_policy_shapes_loss_and_act():
    torch.manual_seed(0)
    cfg = PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK, plan_length=4)
    pol = TRMPolicy(cfg)
    grid = torch.randint(0, 16, (3, 64, 64))
    out = pol(grid)
    assert out.plan_logits.shape == (3, 4, N_ACTIONS)
    assert torch.equal(out.flat_logits, out.plan_logits[:, 0])
    action = torch.randint(0, N_ACTIONS, (3, 4))
    valid = torch.ones(3, 4)
    valid[1, 2:] = 0.0
    parts = pol.loss(out, action, plan_valid=valid)
    assert "bc" in parts and "step0_accuracy" in parts
    parts["bc"].backward()  # trains end to end
    a, out2 = pol.act(grid, temperature=0.0)
    assert a.shape == (3,)
    assert torch.equal(a, out2.flat_logits.argmax(-1))  # MPC executes step 0


def test_plan_step0_weight_interpolates_to_bc():
    torch.manual_seed(0)
    base = dict(core=TINY_CORE, tokenizer=TINY_TOK, plan_length=4)
    pol = TRMPolicy(PolicyConfig(**base, plan_step0_weight=1.0))
    grid = torch.randint(0, 16, (3, 64, 64))
    out = pol(grid)
    action = torch.randint(0, N_ACTIONS, (3, 4))
    parts_w1 = pol.loss(out, action, plan_valid=torch.ones(3, 4))
    nll0 = pol.loss(out, action[:, 0])  # plain BC loss on slot 0's logits?
    # w0=1.0: plan bc-loss must equal CE on slot 0 alone.
    from arc3_wm.trm.core import stablemax_cross_entropy
    ref = stablemax_cross_entropy(out.plan_logits[:, 0], action[:, 0]).mean()  # official: per-element fp64
    assert torch.allclose(parts_w1["bc"], ref, atol=1e-6)
