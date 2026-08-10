"""The TRM recursive reasoning core.

Faithful to "Less is More: Recursive Reasoning with Tiny Networks"
(arXiv:2510.04871) and the official repo:

- one tiny shared network (2 post-norm RMSNorm blocks, SwiGLU, non-causal
  attention or token-mixing MLP) used for both the z-update and the y-update;
- states: y (answer, decodable) and z (latent reasoning), initialised from
  fixed non-trainable buffers;
- one recursion block = ``l_cycles`` z-updates (injection y + x) then one
  y-update (injection z); ``h_cycles`` blocks per supervision step, all but
  the last under ``torch.no_grad()``, the last back-propagated in full (no
  1-step/IFT approximation);
- a single binary halt head read at token 0 (ACT; BCE against
  "answer currently correct", supervised by the trainer);
- EMA of weights for evaluation.

The deep-supervision outer loop lives in the trainer (explicit loop over
``n_supervision`` steps with detached carry), not here - see config.py for
why this deviates from the official carry-across-batches implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import TRMCoreConfig

Carry = tuple[torch.Tensor, torch.Tensor]  # (y, z), both [B, S, D]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float) -> None:
        super().__init__()
        hidden = _round_to(int(dim * expansion * 2 / 3), 64)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _round_to(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


class RotaryEmbedding(nn.Module):
    """Standard 1D RoPE over the flattened token sequence."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, Dh]
        seq = x.shape[-2]
        cos = self.cos[:seq].to(x.dtype)
        sin = self.sin[:seq].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, cfg: TRMCoreConfig, max_seq_len: int) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope: Optional[RotaryEmbedding] = None
        if cfg.pos_encoding == "rope":
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q = self.rope.rotate(q)
            k = self.rope.rotate(k)
        y = F.scaled_dot_product_attention(q, k, v)  # non-causal
        return self.out(y.transpose(1, 2).reshape(b, s, d))


class TokenMixMLP(nn.Module):
    """Attention-free sequence mixer (TRM-MLP variant): MLP over tokens."""

    def __init__(self, cfg: TRMCoreConfig, max_seq_len: int) -> None:
        super().__init__()
        self.seq_len = max_seq_len
        self.mix = nn.Sequential(
            nn.Linear(max_seq_len, max_seq_len, bias=False),
            nn.SiLU(),
            nn.Linear(max_seq_len, max_seq_len, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(f"TokenMixMLP built for seq_len={self.seq_len}, got {x.shape[1]}")
        return self.mix(x.transpose(1, 2)).transpose(1, 2)


class TRMBlock(nn.Module):
    """Post-norm block: x = norm(x + mixer(x)); x = norm(x + swiglu(x))."""

    def __init__(self, cfg: TRMCoreConfig, max_seq_len: int) -> None:
        super().__init__()
        if cfg.seq_mixer == "attention":
            self.mixer: nn.Module = SelfAttention(cfg, max_seq_len)
        else:
            self.mixer = TokenMixMLP(cfg, max_seq_len)
        self.mlp = SwiGLU(cfg.d_model, cfg.expansion)
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.mixer(x))
        x = self.norm2(x + self.mlp(x))
        return x


class ReasoningNet(nn.Module):
    """The single shared network: additive injection, then n_layers blocks."""

    def __init__(self, cfg: TRMCoreConfig, max_seq_len: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TRMBlock(cfg, max_seq_len) for _ in range(cfg.n_layers))

    def forward(self, state: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        x = state + injection
        for layer in self.layers:
            x = layer(x)
        return x


class TRMCore(nn.Module):
    """One supervision step of TRM recursion over pre-embedded tokens.

    ``forward(x, carry)`` runs ``h_cycles`` recursion blocks (the last with
    gradients) and returns ``(y, q_halt, new_carry)``; the caller decodes y,
    computes losses, and threads the detached carry into the next
    supervision step.
    """

    def __init__(self, cfg: TRMCoreConfig, max_seq_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = ReasoningNet(cfg, max_seq_len)
        # Fixed (non-trainable) initial states, following the paper.
        gen = torch.Generator().manual_seed(0)
        self.register_buffer("y_init", torch.randn(cfg.d_model, generator=gen))
        self.register_buffer("z_init", torch.randn(cfg.d_model, generator=gen))
        self.q_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, cfg.halt_bias_init)

    def init_carry(self, x: torch.Tensor) -> Carry:
        b, s, d = x.shape
        if self.cfg.y_init == "input":
            y = x.detach().clone()
        else:
            y = self.y_init.expand(b, s, d).clone()
        z = self.z_init.expand(b, s, d).clone()
        return y, z

    def _block(self, y: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Carry:
        for _ in range(self.cfg.l_cycles):
            z = self.net(z, y + x)
        y = self.net(y, z)
        return y, z

    def forward(
        self, x: torch.Tensor, carry: Optional[Carry] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Carry]:
        if carry is None:
            carry = self.init_carry(x)
        y, z = carry
        if self.cfg.h_cycles > 1:
            with torch.no_grad():
                for _ in range(self.cfg.h_cycles - 1):
                    y, z = self._block(y, z, x)
        y = y.detach()
        z = z.detach()
        # Final block with full backprop through all l_cycles + 1 net calls.
        y, z = self._block(y, z, x)
        q_halt = self.q_head(y[:, 0]).squeeze(-1)
        return y, q_halt, (y.detach(), z.detach())


class EMAHelper:
    """Exponential moving average of parameters, for evaluation only.

    Shadows the full ``state_dict`` (buffers included) in fp32 on the same
    device. ``swap()`` is a context manager that loads the EMA weights for
    eval and restores the training weights on exit.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(v)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def swap(self, model: nn.Module):
        return _EMASwap(self, model)


class _EMASwap:
    def __init__(self, ema: EMAHelper, model: nn.Module) -> None:
        self.ema = ema
        self.model = model
        self._backup: dict = {}

    def __enter__(self):
        self._backup = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        cast = {k: v.to(dtype=b.dtype) for (k, v), b in zip(self.ema.shadow.items(), self._backup.values())}
        self.model.load_state_dict(cast)
        return self.model

    def __exit__(self, *exc):
        self.model.load_state_dict(self._backup)
        self._backup = {}
        return False


def stablemax_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """StableMax cross-entropy (Prieto et al. 2025), as used by TRM.

    s(x) = x + 1 for x >= 0, 1 / (1 - x) otherwise; p_i = s(x_i) / sum s(x_j).
    Computed in fp32 in log space. ``logits`` [..., C], ``target`` [...].
    """
    x = logits.float()
    log_s = torch.where(x >= 0, torch.log1p(x.clamp(min=0)), -torch.log1p((-x).clamp(min=0)))
    log_p = log_s - torch.logsumexp(log_s, dim=-1, keepdim=True)
    nll = -log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"bad reduction {reduction!r}")


def grid_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    kind: str = "stablemax_ce",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-cell CE with optional per-cell weights. logits [..., C], target [...]."""
    if kind == "stablemax_ce":
        nll = stablemax_cross_entropy(logits, target, reduction="none")
    elif kind == "softmax_ce":
        nll = F.cross_entropy(
            logits.flatten(0, -2).float(), target.flatten(), reduction="none"
        ).view(target.shape)
    else:
        raise ValueError(f"bad loss kind {kind!r}")
    if weight is not None:
        return (nll * weight).sum() / weight.sum().clamp(min=1e-8)
    return nll.mean()


class AdamATan2(torch.optim.Optimizer):
    """Adam variant replacing eps with atan2 (Everett et al. 2024), as in TRM.

    update = a * atan2(m_hat, b * sqrt(v_hat)); decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        a: float = 1.2732395447351628,  # 4 / pi
        b: float = 1.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                t = state["step"]
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = group["a"] * torch.atan2(m_hat, group["b"] * v_hat.sqrt())
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss


def warmup_constant_lr(step: int, warmup_steps: int) -> float:
    """TRM's schedule: linear warmup then constant (lr_min_ratio = 1)."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_min_halt_steps(
    batch: int,
    cfg: TRMCoreConfig,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-sample minimum halt step (ACT exploration, paper section 4).

    With prob ``halt_exploration_prob`` a sample must ponder at least
    U{2..halt_max_steps} supervision steps before its halt signal is obeyed.
    """
    explore = torch.rand(batch, generator=generator) < cfg.halt_exploration_prob
    lo, hi = 2, max(2, cfg.halt_max_steps)
    rand_steps = torch.randint(lo, hi + 1, (batch,), generator=generator)
    return torch.where(explore, rand_steps, torch.ones(batch, dtype=torch.long))


