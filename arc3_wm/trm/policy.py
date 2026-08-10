"""TRM as a factored policy over the flat 4102-way action space.

The answer state y decodes to 7 action-type logits plus 4096 click-cell
logits, assembled into flat 4102-way logits (a proper joint distribution;
see tokenizer.assemble_flat_logits). The per-game action mask is applied as
an additive bias, so unavailable actions carry exactly zero probability -
the substrate exposes the mask and this policy enforces it, unlike the
stock-DreamerV3 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import N_ACTION_TYPES, PolicyConfig
from .core import Carry, TRMCore, stablemax_cross_entropy
from .tokenizer import (
    CellHead,
    GridTokenizer,
    assemble_flat_logits,
    click_logits_from_cells,
    pool_tokens,
    rgb_free_grid_check,
)

MASK_BIAS = -1e9  # additive bias for unavailable actions (finite, CE-safe)


@dataclass
class PolicyOutput:
    flat_logits: torch.Tensor  # [B, 4102], mask applied if given
    type_logits: torch.Tensor  # [B, 7]
    click_logits: torch.Tensor  # [B, 4096]
    value: Optional[torch.Tensor]  # [B]
    q_halt: torch.Tensor  # [B]
    carry: Carry


class TRMPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = GridTokenizer(cfg.tokenizer)
        self.seq_len = cfg.tokenizer.n_tokens
        self.core = TRMCore(cfg.core, max_seq_len=self.seq_len)
        d = cfg.core.d_model
        self.type_head = nn.Linear(d, N_ACTION_TYPES)
        self.click_head = CellHead(cfg.tokenizer)
        self.value_head = nn.Linear(d, 1) if cfg.value_head else None

    def embed(self, grid: torch.Tensor) -> torch.Tensor:
        rgb_free_grid_check(grid)
        return self.tokenizer(grid)

    def forward(
        self,
        grid: torch.Tensor,
        carry: Optional[Carry] = None,
        mask: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """One supervision step. ``mask`` is the (B, 4102) bool availability
        mask from ``info["action_mask"]``."""
        if x is None:
            x = self.embed(grid)
        y, q_halt, carry_out = self.core(x, carry)
        pooled = pool_tokens(y)
        type_logits = self.type_head(pooled)
        click_logits = click_logits_from_cells(self.click_head(y))
        flat = assemble_flat_logits(type_logits, click_logits)
        if mask is not None:
            flat = flat + torch.where(
                mask.bool(), torch.zeros_like(flat), torch.full_like(flat, MASK_BIAS)
            )
        value = self.value_head(pooled).squeeze(-1) if self.value_head else None
        return PolicyOutput(
            flat_logits=flat,
            type_logits=type_logits,
            click_logits=click_logits,
            value=value,
            q_halt=q_halt,
            carry=carry_out,
        )

    def loss(
        self,
        out: PolicyOutput,
        action: torch.Tensor,
        value_target: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Behaviour-cloning losses against the human action.
        ``sample_mask`` (float [B], 1 = active) excludes ACT-halted samples
        (per-sample deep supervision)."""
        parts: dict[str, torch.Tensor] = {}
        sm = (
            sample_mask.float()
            if sample_mask is not None
            else torch.ones(action.shape[0], device=action.device)
        )
        denom = sm.sum().clamp(min=1.0)

        def masked_mean(per_sample: torch.Tensor) -> torch.Tensor:
            return (per_sample * sm).sum() / denom

        if self.cfg.loss == "stablemax_ce":
            nll = stablemax_cross_entropy(out.flat_logits, action.long(), reduction="none")
        else:
            nll = F.cross_entropy(
                out.flat_logits.float(), action.long(), reduction="none"
            )
        parts["bc"] = masked_mean(nll)
        if out.value is not None and value_target is not None:
            parts["value"] = masked_mean(
                F.mse_loss(out.value, value_target.float(), reduction="none")
            )
        with torch.no_grad():
            correct = out.flat_logits.argmax(-1) == action.long()
        parts["halt"] = masked_mean(
            F.binary_cross_entropy_with_logits(
                out.q_halt, correct.float(), reduction="none"
            )
        )
        parts["accuracy"] = masked_mean(correct.float()).detach()
        return parts

    @torch.no_grad()
    def act(
        self,
        grid: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, PolicyOutput]:
        """Inference: recurse until halt, then sample (or argmax at T=0)."""
        steps = max_steps or self.cfg.core.halt_max_steps
        x = self.embed(grid)
        carry: Optional[Carry] = None
        out: Optional[PolicyOutput] = None
        done: Optional[torch.Tensor] = None
        frozen_logits: Optional[torch.Tensor] = None
        for _ in range(steps):
            out = self.forward(grid, carry, mask=mask, x=x)
            carry = out.carry
            halted = out.q_halt > 0
            if done is None:
                done = torch.zeros_like(halted)
            if frozen_logits is None:
                frozen_logits = out.flat_logits.clone()
            else:
                frozen_logits[~done] = out.flat_logits[~done]
            done = done | halted
            if bool(done.all()):
                break
        assert out is not None and frozen_logits is not None
        out = PolicyOutput(
            flat_logits=frozen_logits,
            type_logits=out.type_logits,
            click_logits=out.click_logits,
            value=out.value,
            q_halt=out.q_halt,
            carry=out.carry,
        )
        if temperature <= 0:
            return out.flat_logits.argmax(-1), out
        probs = torch.softmax(out.flat_logits / temperature, dim=-1)
        action = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return action, out
