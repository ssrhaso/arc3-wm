"""Spec for arc3_wm.trm.world_model and arc3_wm.trm.policy."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from arc3_wm.trm.config import (  # noqa: E402
    PolicyConfig,
    TokenizerConfig,
    TRMCoreConfig,
    WorldModelConfig,
)
from arc3_wm.trm.core import count_parameters  # noqa: E402
from arc3_wm.trm.policy import TRMPolicy  # noqa: E402
from arc3_wm.trm.world_model import STATES, TRMWorldModel  # noqa: E402

TINY_CORE = TRMCoreConfig(d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=2, halt_max_steps=3)
TINY_TOK = TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4)
WM_CFG = WorldModelConfig(core=TINY_CORE, tokenizer=TINY_TOK, changed_cell_weight=5.0)
POL_CFG = PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK)


def grids(b=2):
    torch.manual_seed(0)
    return torch.randint(0, 16, (b, 64, 64))


def test_wm_forward_shapes():
    torch.manual_seed(0)
    wm = TRMWorldModel(WM_CFG)
    out = wm(grids(), torch.tensor([0, 5]))
    assert out.next_logits.shape == (2, 64, 64, 16)
    assert out.change_logits.shape == (2, 64, 64)
    assert out.reward_logit.shape == (2,)
    assert out.state_logits.shape == (2, len(STATES))
    assert out.q_halt.shape == (2,)


def test_wm_rejects_rgb_input():
    wm = TRMWorldModel(WM_CFG)
    with pytest.raises(ValueError, match="palette"):
        wm(torch.full((1, 64, 64), 200), torch.tensor([0]))


def test_wm_action_conditioning_changes_prediction():
    torch.manual_seed(0)
    wm = TRMWorldModel(WM_CFG)
    g = grids(1)
    out_a = wm(g, torch.tensor([0]))
    out_b = wm(g, torch.tensor([1]))
    assert not torch.allclose(out_a.next_logits, out_b.next_logits)


def test_wm_loss_parts_and_backward():
    torch.manual_seed(0)
    wm = TRMWorldModel(WM_CFG)
    g = grids()
    nxt = g.clone()
    nxt[:, 0, 0] = (nxt[:, 0, 0] + 1) % 16
    out = wm(g, torch.tensor([0, 5]))
    parts = wm.loss(
        out, nxt,
        reward=torch.tensor([0.0, 1.0]),
        state=torch.tensor([0, 1]),
        prev_grid=g,
    )
    assert set(parts) == {"grid", "change", "reward", "state", "halt", "exact_match",
                          "q_halt_accuracy"}
    total = parts["grid"] + parts["change"] + parts["reward"] + parts["state"] + parts["halt"]
    total.backward()
    grads = [p.grad for p in wm.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_wm_supervision_steps_refine_with_carry():
    torch.manual_seed(0)
    wm = TRMWorldModel(WM_CFG)
    g = grids(1)
    a = torch.tensor([3])
    x = wm.embed(g, a)
    out1 = wm(g, a, x=x)
    out2 = wm(g, a, carry=out1.carry, x=x)
    assert not torch.allclose(out1.next_logits, out2.next_logits)


def test_wm_predict_and_rollout_shapes():
    wm = TRMWorldModel(WM_CFG)
    g = grids(1)
    out = wm.predict(g, torch.tensor([0]))
    assert out.next_logits.shape == (1, 64, 64, 16)
    frames = wm.rollout(g, torch.tensor([[0, 1, 2]]), max_steps=1)
    assert frames.shape == (1, 3, 64, 64)
    assert frames.max() < 16 and frames.min() >= 0


def test_wm_optional_heads_off():
    cfg = WorldModelConfig(
        core=TINY_CORE, tokenizer=TINY_TOK,
        reward_head=False, state_head=False, change_head=False,
    )
    wm = TRMWorldModel(cfg)
    out = wm(grids(1), torch.tensor([0]))
    assert out.change_logits is None and out.reward_logit is None and out.state_logits is None
    parts = wm.loss(out, grids(1))
    assert set(parts) == {"grid", "halt", "exact_match", "q_halt_accuracy"}


def test_policy_forward_and_mask():
    torch.manual_seed(0)
    pol = TRMPolicy(POL_CFG)
    g = grids()
    mask = torch.zeros(2, 4102, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 4101] = True
    with torch.no_grad():
        out = pol(g, mask=mask)
    assert out.flat_logits.shape == (2, 4102)
    probs = torch.softmax(out.flat_logits, dim=-1)
    # Masked-out actions carry ~zero probability.
    assert probs[:, 1:4101].sum() < 1e-6
    assert probs[:, 0].sum() + probs[:, 4101].sum() == pytest.approx(2.0, abs=1e-5)


def test_policy_bc_loss_and_halt_target():
    torch.manual_seed(0)
    pol = TRMPolicy(PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK, value_head=True))
    out = pol(grids())
    parts = pol.loss(out, torch.tensor([0, 5]), value_target=torch.tensor([0.0, 1.0]))
    assert set(parts) == {"bc", "value", "halt", "accuracy", "q_halt_accuracy"}
    (parts["bc"] + parts["value"] + parts["halt"]).backward()


def test_policy_value_head_off_by_default():
    # The value head has no training target or consumer; default off.
    pol = TRMPolicy(PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK))
    assert pol.value_head is None
    parts = pol.loss(pol(grids()), torch.tensor([0, 5]))
    assert "value" not in parts


def test_policy_act_respects_mask_and_temperature():
    torch.manual_seed(0)
    pol = TRMPolicy(POL_CFG)
    g = grids(1)
    mask = torch.zeros(1, 4102, dtype=torch.bool)
    mask[:, 2] = True
    action, _ = pol.act(g, mask=mask, temperature=1.0)
    assert action.item() == 2
    action_greedy, _ = pol.act(g, mask=mask, temperature=0.0)
    assert action_greedy.item() == 2


def test_paper_scale_param_counts():
    # Default (paper-scale) configs: each model must stay under ~12M params
    # (DreamerV3 size12m, the baseline's budget) and above 4M (real capacity).
    wm = TRMWorldModel(WorldModelConfig())
    pol = TRMPolicy(PolicyConfig())
    for n in (count_parameters(wm), count_parameters(pol)):
        assert 4e6 < n < 12e6, n


def test_wm_halt_token_slot_feeds_q_halt():
    # Position 0 is a learned halt/summary slot (the official q head reads a
    # learned slot, not a content token): perturbing it must move q_halt.
    torch.manual_seed(0)
    wm = TRMWorldModel(WM_CFG)
    assert wm.seq_len == TINY_TOK.n_tokens + 2  # halt + patches + action
    g, a = grids(1), torch.tensor([0])
    torch.nn.init.normal_(wm.core.q_head.weight, std=0.5)
    with torch.no_grad():
        base = wm(g, a).q_halt.clone()
        wm.halt_token.add_(1.0)
        moved = wm(g, a).q_halt
    assert not torch.allclose(base, moved)


def test_policy_halt_token_slot_feeds_q_halt():
    torch.manual_seed(0)
    pol = TRMPolicy(POL_CFG)
    assert pol.seq_len == TINY_TOK.n_tokens + 1  # halt + patches
    g = grids(1)
    torch.nn.init.normal_(pol.core.q_head.weight, std=0.5)
    with torch.no_grad():
        base = pol(g).q_halt.clone()
        pol.halt_token.add_(1.0)
        moved = pol(g).q_halt
    assert not torch.allclose(base, moved)
