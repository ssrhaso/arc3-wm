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
    flat_logits: torch.Tensor  # [B, 4102] (plan step 0), mask applied if given
    type_logits: torch.Tensor  # [B, 7] (plan step 0)
    click_logits: torch.Tensor  # [B, 4096] (plan step 0)
    value: Optional[torch.Tensor]  # [B]
    q_halt: torch.Tensor  # [B]
    carry: Carry
    plan_logits: Optional[torch.Tensor] = None  # [B, K, 4102] when K > 1


class TRMPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = GridTokenizer(cfg.tokenizer)
        # Learned halt/summary slot at sequence position 0 (see world_model).
        self.halt_token = nn.Parameter(torch.zeros(cfg.core.d_model))
        self.seq_len = cfg.tokenizer.n_tokens + 1  # + halt token
        self.core = TRMCore(cfg.core, max_seq_len=self.seq_len)
        d = cfg.core.d_model
        self.plan_length = cfg.plan_length
        self.type_head = nn.Linear(d, N_ACTION_TYPES * self.plan_length)
        self.click_head = CellHead(cfg.tokenizer, n_maps=self.plan_length)
        self.value_head = nn.Linear(d, 1) if cfg.value_head else None

    def embed(self, grid: torch.Tensor) -> torch.Tensor:
        rgb_free_grid_check(grid)
        tokens = self.tokenizer(grid)
        halt = self.halt_token.view(1, 1, -1).expand(tokens.shape[0], 1, -1)
        return torch.cat([halt, tokens], dim=1)

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
        body = y[:, 1:]  # position 0 is the learned halt/summary slot
        pooled = pool_tokens(body)
        k = self.plan_length
        b = grid.shape[0]
        type_k = self.type_head(pooled).view(b, k, -1)  # [B, K, 7]
        maps = self.click_head(body)  # [B, 64, 64] or [B, K, 64, 64]
        click_k = maps.reshape(b, k, -1)  # [B, K, 4096]
        flat_k = assemble_flat_logits(
            type_k.reshape(b * k, -1), click_k.reshape(b * k, -1)
        ).view(b, k, -1)
        # The availability mask is only known for the current frame, so it
        # biases plan step 0 alone; later plan steps stay unmasked.
        if mask is not None:
            step0 = flat_k[:, 0] + torch.where(
                mask.bool(),
                torch.zeros_like(flat_k[:, 0]),
                torch.full_like(flat_k[:, 0], MASK_BIAS),
            )
            flat_k = torch.cat([step0[:, None], flat_k[:, 1:]], dim=1)
        value = self.value_head(pooled).squeeze(-1) if self.value_head else None
        return PolicyOutput(
            flat_logits=flat_k[:, 0],
            type_logits=type_k[:, 0],
            click_logits=click_k[:, 0],
            value=value,
            q_halt=q_halt,
            carry=carry_out,
            plan_logits=flat_k if k > 1 else None,
        )

    def loss(
        self,
        out: PolicyOutput,
        action: torch.Tensor,
        value_target: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
        plan_valid: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Behaviour-cloning losses against the human action(s).

        ``action`` is [B] for plain BC, or [B, K] for the plan refiner with
        ``plan_valid`` (float [B, K], 1 = step exists before episode end).
        The halt target is full-(plan-)correctness either way.
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

        def ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if self.cfg.loss == "stablemax_ce":
                # Official fp64 implementation; per-element NLL, no reduction.
                return stablemax_cross_entropy(logits, target.long())
            return F.cross_entropy(logits.float(), target.long(), reduction="none")

        if action.dim() == 2:
            b, k = action.shape
            logits_k = out.plan_logits
            assert logits_k is not None, "plan action given to a K=1 policy"
            pv = (
                plan_valid.float()
                if plan_valid is not None
                else torch.ones(b, k, device=action.device)
            )
            nll_k = ce(logits_k.reshape(b * k, -1), action.reshape(b * k)).view(b, k)
            w0 = self.cfg.plan_step0_weight
            if w0 > 0:
                tail_pv = pv.clone()
                tail_pv[:, 0] = 0.0
                tail_n = tail_pv.sum(-1)
                tail_mean = (nll_k * tail_pv).sum(-1) / tail_n.clamp(min=1.0)
                nll = w0 * nll_k[:, 0] + (1.0 - w0) * torch.where(
                    tail_n > 0, tail_mean, nll_k[:, 0]
                )
            else:
                nll = (nll_k * pv).sum(-1) / pv.sum(-1).clamp(min=1.0)
            with torch.no_grad():
                step_hit = logits_k.argmax(-1) == action.long()
                correct = ((step_hit | (pv < 0.5)).all(-1))
                parts["step0_accuracy"] = masked_mean(
                    step_hit[:, 0].float()
                ).detach()
        else:
            nll = ce(out.flat_logits, action)
            with torch.no_grad():
                correct = out.flat_logits.argmax(-1) == action.long()
        parts["bc"] = masked_mean(nll)
        if out.value is not None and value_target is not None:
            parts["value"] = masked_mean(
                F.mse_loss(out.value, value_target.float(), reduction="none")
            )
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
        early_halt: bool = False,
    ) -> tuple[torch.Tensor, PolicyOutput]:
        """Inference. Default is the official/NVARC protocol: recurse for all
        ``max_steps`` supervision steps (default ``core.halt_max_steps``) and
        decode the LAST step's output, then sample (or argmax at T=0).
        ``early_halt=True`` restores per-sample freezing at the first halting
        step."""
        steps = max_steps or self.cfg.core.halt_max_steps
        x = self.embed(grid)
        carry: Optional[Carry] = None
        out: Optional[PolicyOutput] = None
        if not early_halt:
            for _ in range(steps):
                out = self.forward(grid, carry, mask=mask, x=x)
                carry = out.carry
        else:
            done: Optional[torch.Tensor] = None
            frozen: dict[str, torch.Tensor] = {}
            for _ in range(steps):
                out = self.forward(grid, carry, mask=mask, x=x)
                carry = out.carry
                halted = out.q_halt > 0
                if done is None:
                    done = torch.zeros_like(halted)
                # Freeze every output field at each sample's first halting
                # step, so the returned PolicyOutput is internally consistent.
                for name in ("flat_logits", "type_logits", "click_logits",
                             "value", "q_halt", "plan_logits"):
                    value = getattr(out, name)
                    if value is None:
                        continue
                    if name not in frozen:
                        frozen[name] = value.clone()
                    else:
                        idx = ~done
                        frozen[name][idx] = value[idx]
                done = done | halted
                if bool(done.all()):
                    break
            assert out is not None and frozen
            out = PolicyOutput(
                flat_logits=frozen["flat_logits"],
                type_logits=frozen["type_logits"],
                click_logits=frozen["click_logits"],
                value=frozen.get("value"),
                q_halt=frozen["q_halt"],
                carry=out.carry,
                plan_logits=frozen.get("plan_logits"),
            )
        assert out is not None
        if temperature <= 0:
            return out.flat_logits.argmax(-1), out
        probs = torch.softmax(out.flat_logits / temperature, dim=-1)
        action = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        return action, out
