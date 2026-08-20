"""Spec for PolicyConfig.action_head: the cell/xy/flat representation
ablation (paper Track-B tie-in - supervised deconfound of the factored-
action mechanism)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from arc3_wm.action_space import ACTION6_BASE, ACTION6_COUNT, GRID, N_ACTIONS  # noqa: E402
from arc3_wm.trm import config as C  # noqa: E402
from arc3_wm.trm.core import EMAHelper  # noqa: E402
from arc3_wm.trm.policy import TRMPolicy  # noqa: E402
from arc3_wm.trm.training import load_policy, save_checkpoint  # noqa: E402

TINY_CORE = C.TRMCoreConfig(
    d_model=32, n_heads=4, n_layers=1, l_cycles=1, h_cycles=1,
    n_supervision=2, halt_max_steps=2,
)
TINY_TOK = C.TokenizerConfig(d_model=32, patch_size=16, cell_embed_dim=4)


def _cfg(head: str, plan_length: int = 1) -> C.PolicyConfig:
    return C.PolicyConfig(core=TINY_CORE, tokenizer=TINY_TOK,
                          plan_length=plan_length, action_head=head)


def _grid(b: int = 2) -> torch.Tensor:
    g = torch.randint(0, 16, (b, GRID, GRID))
    return g


@pytest.mark.parametrize("head", ["cell", "xy", "flat", "flatsh"])
@pytest.mark.parametrize("plan_length", [1, 3])
def test_forward_shapes(head, plan_length):
    pol = TRMPolicy(_cfg(head, plan_length))
    out = pol(_grid())
    assert out.flat_logits.shape == (2, N_ACTIONS)
    assert out.type_logits.shape == (2, C.N_ACTION_TYPES)
    assert out.click_logits.shape == (2, ACTION6_COUNT)
    if plan_length > 1:
        assert out.plan_logits.shape == (2, plan_length, N_ACTIONS)
    else:
        assert out.plan_logits is None


def test_xy_click_block_is_additive():
    """xy head: click logit at (y, x) must decompose as f(y) + g(x)."""
    pol = TRMPolicy(_cfg("xy"))
    out = pol(_grid(1))
    block = out.click_logits[0].view(GRID, GRID)
    # For an additive block, block[y,x] - block[y,0] - block[0,x] + block[0,0] == 0
    resid = block - block[:, :1] - block[:1, :] + block[:1, :1]
    assert torch.allclose(resid, torch.zeros_like(resid), atol=1e-5)


def test_cell_click_block_is_not_additive():
    """The spatial cell head is strictly more expressive than xy."""
    torch.manual_seed(0)
    pol = TRMPolicy(_cfg("cell"))
    out = pol(_grid(1))
    block = out.click_logits[0].view(GRID, GRID)
    resid = block - block[:, :1] - block[:1, :] + block[:1, :1]
    assert resid.abs().max() > 1e-4


@pytest.mark.parametrize("head", ["flat", "flatsh"])
def test_flat_head_is_the_distribution(head):
    """flat heads: flat_logits come straight from the linear, no assembly."""
    pol = TRMPolicy(_cfg(head))
    out = pol(_grid(1))
    # click view is an exact slice of the flat logits
    assert torch.equal(
        out.click_logits[0],
        out.flat_logits[0, ACTION6_BASE:ACTION6_BASE + ACTION6_COUNT],
    )


@pytest.mark.parametrize("head", ["cell", "xy", "flat", "flatsh"])
def test_loss_backward_reaches_head(head):
    pol = TRMPolicy(_cfg(head))
    out = pol(_grid())
    action = torch.tensor([0, ACTION6_BASE + 65])
    parts = pol.loss(out, action)
    (parts["bc"] + parts["halt"]).backward()
    params = [p for n, p in pol.named_parameters()
              if any(t in n for t in ("flat_head", "x_head", "y_head",
                                      "click_head", "type_head", "slot_"))]
    assert params and all(p.grad is not None for p in params)


@pytest.mark.parametrize("head", ["cell", "xy", "flat", "flatsh"])
def test_checkpoint_roundtrip_preserves_head(head, tmp_path):
    cfg = _cfg(head)
    pol = TRMPolicy(cfg)
    ema = EMAHelper(mu=0.5)
    ema.register(pol)
    opt = torch.optim.SGD(pol.parameters(), lr=0.1)
    path = tmp_path / "best.pt"
    save_checkpoint(path, pol, ema, opt, C.to_dict(cfg), step=1)
    loaded = load_policy(path)
    assert loaded.cfg.action_head == head
    g = _grid(1)
    with torch.no_grad():
        a, _ = loaded.act(g, temperature=0.0)
    assert 0 <= int(a[0]) < N_ACTIONS


def test_legacy_config_defaults_to_cell():
    """Old checkpoints (no action_head key) must load as the cell head."""
    d = C.to_dict(_cfg("cell"))
    d.pop("action_head")
    cfg = C.from_dict(C.PolicyConfig, d)
    assert cfg.action_head == "cell"
    pol = TRMPolicy(cfg)
    assert hasattr(pol, "click_head")


def test_unknown_head_raises():
    with pytest.raises(ValueError, match="action_head"):
        TRMPolicy(_cfg("banana"))


def test_flatsh_slots_differ_and_param_matched():
    """flatsh: slots produce different distributions (nonlinear conditioning
    works) and the head stays small at K=32."""
    torch.manual_seed(0)
    pol = TRMPolicy(_cfg("flatsh", plan_length=32))
    out = pol(_grid(1))
    assert out.plan_logits.shape == (1, 32, N_ACTIONS)
    assert (out.plan_logits[0, 0] - out.plan_logits[0, 1]).abs().max() > 1e-6
    n = sum(q.numel() for q in pol.parameters())
    n_flat = sum(q.numel() for q in TRMPolicy(_cfg("flat", plan_length=32)).parameters())
    assert n < n_flat / 5  # shared head kills the K-scaling blowup
