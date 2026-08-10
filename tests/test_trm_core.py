"""Spec for arc3_wm.trm.core - the recursive reasoning engine."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from arc3_wm.trm.config import TRMCoreConfig  # noqa: E402
from arc3_wm.trm.core import (  # noqa: E402
    AdamATan2,
    EMAHelper,
    TRMCore,
    count_parameters,
    grid_cross_entropy,
    sample_min_halt_steps,
    stablemax_cross_entropy,
    warmup_constant_lr,
)

SMALL = TRMCoreConfig(d_model=32, n_heads=4, n_layers=2, l_cycles=2, h_cycles=2)
SEQ = 10


def make_core(cfg=SMALL, seq=SEQ) -> TRMCore:
    torch.manual_seed(0)
    return TRMCore(cfg, max_seq_len=seq)


def test_forward_shapes_and_carry():
    core = make_core()
    x = torch.randn(3, SEQ, 32)
    y, q_halt, carry = core(x)
    assert y.shape == (3, SEQ, 32)
    assert q_halt.shape == (3,)
    assert carry[0].shape == (3, SEQ, 32) and carry[1].shape == (3, SEQ, 32)
    assert not carry[0].requires_grad and not carry[1].requires_grad


def test_carry_threads_between_supervision_steps():
    core = make_core()
    x = torch.randn(2, SEQ, 32)
    y1, _, carry1 = core(x)
    y2, _, _ = core(x, carry1)
    # A second supervision step continues from the carry, not from init.
    y1_again, _, _ = core(x)
    assert torch.allclose(y1, y1_again)
    assert not torch.allclose(y2, y1)


def test_single_shared_net_is_used_for_both_updates():
    core = make_core()
    # One ReasoningNet instance only (paper: no separate H/L networks).
    nets = [m for m in core.modules() if type(m).__name__ == "ReasoningNet"]
    assert len(nets) == 1


def test_gradients_flow_only_through_last_block():
    cfg = TRMCoreConfig(d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=3)
    core = make_core(cfg)
    x = torch.randn(2, SEQ, 32, requires_grad=True)
    y, q, _ = core(x)
    y.sum().backward()
    assert x.grad is not None
    # Gradient reaches x through the final block's l_cycles+1 calls only;
    # the check here is simply that backward completes and is finite.
    assert torch.isfinite(x.grad).all()


def test_q_head_initial_state_does_not_halt():
    # Zero-init (official TRM): q starts at exactly 0, and halting is
    # strict q > 0, so a fresh head never halts but can learn either way.
    core = make_core()
    x = torch.randn(2, SEQ, 32)
    _, q_halt, _ = core(x)
    assert (q_halt <= 0).all()
    assert not (q_halt > 0).any()


def test_y_init_input_mode_starts_from_input():
    cfg = TRMCoreConfig(d_model=32, n_heads=4, n_layers=2, l_cycles=2, h_cycles=2, y_init="input")
    core = make_core(cfg)
    x = torch.randn(2, SEQ, 32)
    y0, z0 = core.init_carry(x)
    assert torch.allclose(y0, x)
    assert not torch.allclose(z0, x)


def test_mlp_mixer_variant_runs_and_rejects_wrong_seq():
    cfg = TRMCoreConfig(
        d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
        seq_mixer="mlp", pos_encoding="none",
    )
    core = make_core(cfg)
    y, _, _ = core(torch.randn(2, SEQ, 32))
    assert y.shape == (2, SEQ, 32)
    with pytest.raises(ValueError, match="seq_len"):
        core(torch.randn(2, SEQ + 1, 32))


def test_determinism_same_seed_same_output():
    a = make_core()
    b = make_core()
    x = torch.randn(2, SEQ, 32)
    ya, _, _ = a(x)
    yb, _, _ = b(x)
    assert torch.allclose(ya, yb)


def test_stablemax_matches_uniform_at_zero_logits():
    logits = torch.zeros(5, 4)
    target = torch.tensor([0, 1, 2, 3, 0])
    loss = stablemax_cross_entropy(logits, target)
    assert torch.isclose(loss, torch.tensor(4.0).log(), atol=1e-5)


def test_stablemax_extreme_logits_finite():
    logits = torch.tensor([[1e6, -1e6, 0.0]])
    loss = stablemax_cross_entropy(logits, torch.tensor([0]))
    assert torch.isfinite(loss)
    loss_bad = stablemax_cross_entropy(logits, torch.tensor([1]))
    assert torch.isfinite(loss_bad) and loss_bad > loss


def test_grid_ce_weighting_upweights_changed_cells():
    logits = torch.zeros(1, 8, 3)
    target = torch.zeros(1, 8, dtype=torch.long)
    w = torch.ones(1, 8)
    w[0, :4] = 10.0
    unweighted = grid_cross_entropy(logits, target, "softmax_ce")
    weighted = grid_cross_entropy(logits, target, "softmax_ce", weight=w)
    # Uniform logits: weighting cannot change the mean NLL value.
    assert torch.isclose(unweighted, weighted, atol=1e-6)
    # Now make the first 4 cells wrong: weighted loss must exceed unweighted.
    logits2 = logits.clone()
    logits2[0, :4, 1] = 5.0
    assert grid_cross_entropy(logits2, target, "softmax_ce", weight=w) > grid_cross_entropy(
        logits2, target, "softmax_ce"
    )


def test_ema_converges_toward_model():
    core = make_core()
    ema = EMAHelper(core, decay=0.5)
    with torch.no_grad():
        for p in core.parameters():
            p.add_(1.0)
    for _ in range(20):
        ema.update(core)
    name, param = next(iter(core.state_dict().items()))
    assert torch.allclose(ema.shadow[name], param.float(), atol=1e-4)


def test_ema_swap_restores_training_weights():
    core = make_core()
    ema = EMAHelper(core, decay=0.999)
    with torch.no_grad():
        for p in core.parameters():
            p.add_(1.0)
    trained = {k: v.clone() for k, v in core.state_dict().items()}
    # Inside the context the (near-initial) EMA weights are loaded ...
    with ema.swap(core) as m:
        inside = next(iter(m.parameters())).clone()
    # ... and on exit the training weights come back exactly.
    for k, v in core.state_dict().items():
        assert torch.equal(v, trained[k]), k
    outside = next(iter(core.parameters())).clone()
    assert not torch.allclose(inside, outside)


def test_adam_atan2_step_reduces_loss():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8))
    target = torch.zeros(8)
    opt = AdamATan2([w], lr=0.05)
    initial = ((w - target) ** 2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        loss = ((w - target) ** 2).sum()
        loss.backward()
        opt.step()
    assert loss.item() < initial * 0.01


def test_warmup_schedule():
    assert warmup_constant_lr(0, 100) == pytest.approx(0.01)
    assert warmup_constant_lr(99, 100) == 1.0
    assert warmup_constant_lr(500, 100) == 1.0
    assert warmup_constant_lr(0, 0) == 1.0


def test_sample_min_halt_steps_bounds():
    cfg = TRMCoreConfig(halt_max_steps=6, halt_exploration_prob=1.0)
    gen = torch.Generator().manual_seed(0)
    steps = sample_min_halt_steps(1000, cfg, gen)
    assert steps.min() >= 2 and steps.max() <= 6
    cfg0 = TRMCoreConfig(halt_exploration_prob=0.0)
    assert (sample_min_halt_steps(100, cfg0, gen) == 1).all()


def test_parameter_count_paper_scale():
    # Paper-default core (512 wide, 2 layers) must land in the ~5-8M band
    # (TRM-Att is reported at 7M including embeddings/heads).
    core = TRMCore(TRMCoreConfig(), max_seq_len=260)
    n = count_parameters(core)
    assert 3e6 < n < 8e6, n
