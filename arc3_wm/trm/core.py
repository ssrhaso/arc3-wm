"""The TRM recursive reasoning core, built on the official code.

The recursion-critical pieces are the official TinyRecursiveModels
implementation, linked from the pinned third_party clone (see
``_official.py``): ``ACTV1Block`` (post-norm parameter-free RMSNorm with
non-causal attention, or the official ``mlp_t`` sequence-mixing SwiGLU for
``seq_mixer="mlp"``), the shared ``ReasoningModule``, ``CastedLinear``
initialisation, RoPE, fp64 ``stablemax_cross_entropy``, the EMA helper, and
the trunc-normal std-1 init states. Only the driver is ours:

- states y (answer, decodable) and z (latent reasoning), carried detached
  between supervision steps; one recursion block = ``l_cycles`` z-updates
  (injection y + x) then one y-update (injection z); ``h_cycles`` blocks per
  supervision step, all but the last under ``torch.no_grad()``, the last
  back-propagated in full (no 1-step/IFT approximation) - line-for-line the
  official Inner forward;
- the halt head is the official 2-logit ``CastedLinear`` (zero weight, bias
  ``halt_bias_init``; official fills -5), of which only the halt logit is
  read at token 0 (``no_ACT_continue=True`` semantics);
- deep supervision itself lives in the trainer (explicit loop over
  ``n_supervision`` steps with detached carry), not the official
  carry-across-batches wrapper - see config.py for the documented deviation
  list.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ._official import (
    CastedLinear,
    OfficialEMAHelper,
    RotaryEmbedding,
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    stablemax_cross_entropy,
    trunc_normal_init_,
)
from .config import TRMCoreConfig

# Re-exported under the historical name; the EMA helper IS the official one.
EMAHelper = OfficialEMAHelper

Carry = tuple[torch.Tensor, torch.Tensor]  # (y, z), both [B, S, D]


def _block_config(cfg: TRMCoreConfig, max_seq_len: int) -> "TinyRecursiveReasoningModel_ACTV1Config":
    """The official pydantic config, filled from our TRMCoreConfig.

    vocab/batch/puzzle fields are unused by the block itself (they belong to
    the official Inner shell we deliberately do not adopt).
    """
    return TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=max_seq_len,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        num_puzzle_identifiers=1,
        vocab_size=2,
        H_cycles=cfg.h_cycles,
        L_cycles=cfg.l_cycles,
        H_layers=0,
        L_layers=cfg.n_layers,
        hidden_size=cfg.d_model,
        expansion=cfg.expansion,
        num_heads=cfg.n_heads,
        pos_encodings=cfg.pos_encoding,
        rms_norm_eps=cfg.norm_eps,
        halt_max_steps=cfg.halt_max_steps,
        halt_exploration_prob=cfg.halt_exploration_prob,
        forward_dtype="float32",
        mlp_t=(cfg.seq_mixer == "mlp"),
    )


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
        self.max_seq_len = max_seq_len
        ocfg = _block_config(cfg, max_seq_len)
        self.net = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(ocfg) for _ in range(cfg.n_layers)]
        )
        self.rotary: Optional[RotaryEmbedding] = None
        if cfg.pos_encoding == "rope":
            self.rotary = RotaryEmbedding(
                dim=cfg.d_model // cfg.n_heads,
                max_position_embeddings=max_seq_len,
                base=10000.0,
            )
        # Fixed (non-trainable) initial states, as in the official repo
        # (trunc-normal std-1 buffers drawn from the global RNG, which the
        # train scripts seed).
        self.register_buffer("y_init", trunc_normal_init_(torch.empty(cfg.d_model), std=1.0))
        self.register_buffer("z_init", trunc_normal_init_(torch.empty(cfg.d_model), std=1.0))
        self.q_head = CastedLinear(cfg.d_model, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(cfg.halt_bias_init)

    def init_carry(self, x: torch.Tensor) -> Carry:
        b, s, d = x.shape
        if self.cfg.y_init == "input":
            y = x.detach().clone()
        else:
            y = self.y_init.expand(b, s, d).clone()
        z = self.z_init.expand(b, s, d).clone()
        return y, z

    def _block(
        self, y: torch.Tensor, z: torch.Tensor, x: torch.Tensor, cos_sin
    ) -> Carry:
        for _ in range(self.cfg.l_cycles):
            z = self.net(z, y + x, cos_sin=cos_sin)
        y = self.net(y, z, cos_sin=cos_sin)
        return y, z

    def forward(
        self, x: torch.Tensor, carry: Optional[Carry] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Carry]:
        if self.cfg.seq_mixer == "mlp" and x.shape[1] != self.max_seq_len:
            # The official mlp_t block builds a sequence-length-sized SwiGLU;
            # fail loudly instead of hitting a raw shape error there.
            raise ValueError(
                f"mlp mixer built for seq_len={self.max_seq_len}, got {x.shape[1]}"
            )
        if carry is None:
            carry = self.init_carry(x)
        y, z = carry
        cos_sin = self.rotary() if self.rotary is not None else None
        if self.cfg.h_cycles > 1:
            with torch.no_grad():
                for _ in range(self.cfg.h_cycles - 1):
                    y, z = self._block(y, z, x, cos_sin)
        y = y.detach()
        z = z.detach()
        # Final block with full backprop through all l_cycles + 1 net calls.
        y, z = self._block(y, z, x, cos_sin)
        q_halt = self.q_head(y[:, 0]).to(torch.float32)[..., 0]  # halt logit only
        return y, q_halt, (y.detach(), z.detach())


def grid_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    kind: str = "stablemax_ce",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-cell CE with optional per-cell weights. logits [..., C], target [...].

    The stablemax path is the official fp64 implementation (no reduction -
    per-element NLL), optionally weighted here.
    """
    if kind == "stablemax_ce":
        nll = stablemax_cross_entropy(logits, target.long())
    elif kind == "softmax_ce":
        nll = F.cross_entropy(
            logits.flatten(0, -2).float(), target.flatten().long(), reduction="none"
        ).view(target.shape)
    else:
        raise ValueError(f"bad loss kind {kind!r}")
    if weight is not None:
        return (nll * weight).sum() / weight.sum().clamp(min=1e-8)
    return nll.mean()


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
