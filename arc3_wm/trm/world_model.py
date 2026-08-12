"""TRM as a discrete next-frame world model.

Given the current palette grid and one flat action, the model recursively
refines a prediction of the next grid (per-cell 16-way logits) plus reward
(level clear), terminal state, and a per-cell change mask. The answer state
y of the TRM core *is* the prediction; deep supervision refines it.

Training targets the failure mode measured for DreamerV3 on this benchmark
(open-loop rollouts never beating copy-last-frame): the cross-entropy
up-weights cells that change between frames (``changed_cell_weight``), and
the change head gives the planner a calibrated "does this action do
anything" signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import WorldModelConfig
from .core import Carry, TRMCore, grid_cross_entropy
from .tokenizer import (
    ActionEncoder,
    CellHead,
    GridHead,
    GridTokenizer,
    pool_tokens,
    rgb_free_grid_check,
)

# Terminal-state vocabulary (index into state head logits).
STATES = ("NOT_FINISHED", "WIN", "GAME_OVER")


@dataclass
class WMOutput:
    next_logits: torch.Tensor  # [B, 64, 64, 16]
    change_logits: Optional[torch.Tensor]  # [B, 64, 64]
    reward_logit: Optional[torch.Tensor]  # [B]
    state_logits: Optional[torch.Tensor]  # [B, 3]
    q_halt: torch.Tensor  # [B]
    carry: Carry


class TRMWorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = GridTokenizer(cfg.tokenizer)
        self.action_encoder = ActionEncoder(cfg.tokenizer)
        # Learned halt/summary slot at sequence position 0: the official q
        # head reads a learned slot (the puzzle-emb token), not a content
        # token. Zero-init mirrors the official puzzle-emb init_std=0.
        self.halt_token = nn.Parameter(torch.zeros(cfg.core.d_model))
        self.seq_len = cfg.tokenizer.n_tokens + 2  # + halt token + action token
        self.core = TRMCore(cfg.core, max_seq_len=self.seq_len)
        self.grid_head = GridHead(cfg.tokenizer)
        self.change_head = CellHead(cfg.tokenizer) if cfg.change_head else None
        d = cfg.core.d_model
        self.reward_head = nn.Linear(d, 1) if cfg.reward_head else None
        self.state_head = nn.Linear(d, len(STATES)) if cfg.state_head else None

    def embed(self, grid: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        rgb_free_grid_check(grid)
        tokens = self.tokenizer(grid)
        act = self.action_encoder(action).unsqueeze(1)
        halt = self.halt_token.view(1, 1, -1).expand(tokens.shape[0], 1, -1)
        return torch.cat([halt, tokens, act], dim=1)

    def forward(
        self,
        grid: torch.Tensor,
        action: torch.Tensor,
        carry: Optional[Carry] = None,
        x: Optional[torch.Tensor] = None,
    ) -> WMOutput:
        """One supervision step. Pass ``x`` to reuse a precomputed embedding
        across supervision steps (the input embedding does not change)."""
        if x is None:
            x = self.embed(grid, action)
        y, q_halt, carry_out = self.core(x, carry)
        grid_tokens = y[:, 1 : self.cfg.tokenizer.n_tokens + 1]  # skip the halt slot
        pooled = pool_tokens(grid_tokens)
        return WMOutput(
            next_logits=self.grid_head(grid_tokens),
            change_logits=self.change_head(grid_tokens) if self.change_head else None,
            reward_logit=self.reward_head(pooled).squeeze(-1) if self.reward_head else None,
            state_logits=self.state_head(pooled) if self.state_head else None,
            q_halt=q_halt,
            carry=carry_out,
        )

    def loss(
        self,
        out: WMOutput,
        next_grid: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        prev_grid: Optional[torch.Tensor] = None,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Per-step losses. ``prev_grid`` enables changed-cell weighting and
        the change-mask target; ``reward`` is binary (level cleared);
        ``state`` indexes STATES. ``sample_mask`` (float [B], 1 = active)
        excludes ACT-halted samples from every term - per-sample deep
        supervision, matching the official carry-slot semantics."""
        cfg = self.cfg
        parts: dict[str, torch.Tensor] = {}
        b = next_grid.shape[0]
        sm = (
            sample_mask.float()
            if sample_mask is not None
            else torch.ones(b, device=next_grid.device)
        )
        denom = sm.sum().clamp(min=1.0)

        def masked_mean(per_sample: torch.Tensor) -> torch.Tensor:
            return (per_sample * sm).sum() / denom

        changed = None
        weight = sm[:, None, None].expand(-1, 64, 64).clone()
        if prev_grid is not None:
            changed = (next_grid != prev_grid).float()
            weight = weight * (1.0 + (cfg.changed_cell_weight - 1.0) * changed)
        parts["grid"] = grid_cross_entropy(
            out.next_logits, next_grid.long(), cfg.loss, weight=weight
        )
        if out.change_logits is not None and changed is not None:
            parts["change"] = masked_mean(
                F.binary_cross_entropy_with_logits(
                    out.change_logits, changed, reduction="none"
                ).mean(dim=(1, 2))
            )
        if out.reward_logit is not None and reward is not None:
            parts["reward"] = masked_mean(
                F.binary_cross_entropy_with_logits(
                    out.reward_logit, reward.float(), reduction="none"
                )
            )
        if out.state_logits is not None and state is not None:
            parts["state"] = masked_mean(
                F.cross_entropy(out.state_logits.float(), state.long(), reduction="none")
            )
        # ACT halt target: the decoded next frame is exactly right.
        with torch.no_grad():
            exact = (out.next_logits.argmax(-1) == next_grid.long()).flatten(1).all(-1)
            q_acc = ((out.q_halt > 0) == exact).float()
        parts["halt"] = masked_mean(
            F.binary_cross_entropy_with_logits(
                out.q_halt, exact.float(), reduction="none"
            )
        )
        # Official metric: how often the halt decision matches actual
        # correctness (detached; never enters the loss).
        parts["q_halt_accuracy"] = masked_mean(q_acc).detach()
        parts["exact_match"] = masked_mean(exact.float()).detach()
        return parts

    @torch.no_grad()
    def predict(
        self,
        grid: torch.Tensor,
        action: torch.Tensor,
        max_steps: Optional[int] = None,
        early_halt: bool = False,
    ) -> WMOutput:
        """Inference. Default is the official/NVARC protocol: at eval the ACT
        halt signal never fires, the model runs all ``max_steps`` supervision
        steps (default ``core.halt_max_steps``), and the returned output is
        the LAST step's. ``early_halt=True`` restores the previous behaviour:
        each sample's output frozen at its first halting step, stopping when
        every sample has halted."""
        steps = max_steps or self.cfg.core.halt_max_steps
        x = self.embed(grid, action)
        carry: Optional[Carry] = None
        out: Optional[WMOutput] = None
        if not early_halt:
            for _ in range(steps):
                out = self.forward(grid, action, carry, x=x)
                carry = out.carry
            assert out is not None
            return out
        done: Optional[torch.Tensor] = None
        frozen: dict[str, torch.Tensor] = {}
        for _ in range(steps):
            out = self.forward(grid, action, carry, x=x)
            carry = out.carry
            halted = out.q_halt > 0
            if done is None:
                done = torch.zeros_like(halted)
            newly = halted & ~done
            for name in ("next_logits", "change_logits", "reward_logit",
                         "state_logits", "q_halt"):
                value = getattr(out, name)
                if value is None:
                    continue
                if name not in frozen:
                    frozen[name] = value.clone()
                else:
                    idx = ~done  # keep updating only not-yet-frozen rows
                    frozen[name][idx] = value[idx]
            done = done | newly
            if bool(done.all()):
                break
        assert out is not None
        return WMOutput(
            next_logits=frozen["next_logits"],
            change_logits=frozen.get("change_logits"),
            reward_logit=frozen.get("reward_logit"),
            state_logits=frozen.get("state_logits"),
            q_halt=frozen["q_halt"],
            carry=out.carry,
        )

    @torch.no_grad()
    def rollout(
        self,
        grid: torch.Tensor,
        actions: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Open-loop rollout: feed predictions back in. ``actions`` [B, H].
        Returns predicted grids [B, H, 64, 64]."""
        frames = []
        current = grid
        for t in range(actions.shape[1]):
            out = self.predict(current, actions[:, t], max_steps=max_steps)
            current = out.next_logits.argmax(-1)
            frames.append(current)
        return torch.stack(frames, dim=1)
