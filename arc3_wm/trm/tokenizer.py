"""Grid/action tokenisation components.

ARC-AGI-3 frames are 64x64 palette-index grids (16 colours). The tokenizer
maps a grid to a sequence of patch tokens (each patch embeds its cells'
colours), the action encoder maps a flat action index to one token, and the
grid head decodes patch tokens back to per-cell colour logits.

All components speak palette indices, never RGB; use
``arc3_wm.dynamics_probe.quantize_to_palette`` to recover indices from the
env's RGB observations (exact inverse of the palette decode).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..action_space import ACTION6_BASE, ACTION6_COUNT, ACTION7_INDEX, GRID, N_ACTIONS
from .config import (
    ACTION6_TYPE_INDEX,
    N_ACTION_TYPES,
    PALETTE_SIZE,
    TokenizerConfig,
)


def flat_action_components(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat index [B] -> (type [B] in 0..6, x [B], y [B]; -1 x/y off ACTION6)."""
    if ((action < 0) | (action >= N_ACTIONS)).any():
        raise ValueError("action index out of range")
    is_click = (action >= ACTION6_BASE) & (action < ACTION6_BASE + ACTION6_COUNT)
    is_a7 = action == ACTION7_INDEX
    a_type = torch.where(
        is_click,
        torch.full_like(action, ACTION6_TYPE_INDEX),
        torch.where(is_a7, torch.full_like(action, N_ACTION_TYPES - 1), action),
    )
    rel = (action - ACTION6_BASE).clamp(min=0)
    x = torch.where(is_click, rel % GRID, torch.full_like(action, -1))
    y = torch.where(is_click, rel // GRID, torch.full_like(action, -1))
    return a_type, x, y


class GridTokenizer(nn.Module):
    """(B, 64, 64) int grid -> (B, n_tokens, d_model) patch tokens."""

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cell_embed = nn.Embedding(PALETTE_SIZE, cfg.cell_embed_dim)
        self.project = nn.Linear(cfg.cells_per_patch * cfg.cell_embed_dim, cfg.d_model)
        if cfg.learned_pos:
            self.pos = nn.Parameter(torch.zeros(cfg.n_tokens, cfg.d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.dim() != 3 or grid.shape[-2:] != (GRID, GRID):
            raise ValueError(f"expected (B, {GRID}, {GRID}) grid, got {tuple(grid.shape)}")
        b = grid.shape[0]
        p = self.cfg.patch_size
        t = self.cfg.tokens_per_side
        emb = self.cell_embed(grid.long())  # [B, 64, 64, E]
        # -> [B, t, t, p, p, E] -> [B, t*t, p*p*E]
        emb = emb.view(b, t, p, t, p, -1).permute(0, 1, 3, 2, 4, 5)
        emb = emb.reshape(b, t * t, p * p * emb.shape[-1])
        tokens = self.project(emb)
        if self.pos is not None:
            tokens = tokens + self.pos
        return tokens


class ActionEncoder(nn.Module):
    """Flat action index -> one d_model token (type + factorised x/y coords)."""

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.type_embed = nn.Embedding(N_ACTION_TYPES, d)
        # Index 0 is reserved for "no coordinate" (non-click actions); the 64
        # real coordinates map to 1..64.
        self.x_embed = nn.Embedding(GRID + 1, d // 2)
        self.y_embed = nn.Embedding(GRID + 1, d // 2)
        self.project = nn.Linear(d + d, d)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        a_type, x, y = flat_action_components(action)
        coords = torch.cat([self.x_embed(x.long() + 1), self.y_embed(y.long() + 1)], dim=-1)
        token = torch.cat([self.type_embed(a_type.long()), coords], dim=-1)
        return self.project(token)


class GridHead(nn.Module):
    """Patch tokens -> per-cell colour logits (B, 64, 64, 16)."""

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decode = nn.Linear(cfg.d_model, cfg.cells_per_patch * PALETTE_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        if n != self.cfg.n_tokens:
            raise ValueError(f"expected {self.cfg.n_tokens} tokens, got {n}")
        p = self.cfg.patch_size
        t = self.cfg.tokens_per_side
        logits = self.decode(tokens)  # [B, n, p*p*16]
        logits = logits.view(b, t, t, p, p, PALETTE_SIZE).permute(0, 1, 3, 2, 4, 5)
        return logits.reshape(b, GRID, GRID, PALETTE_SIZE)


class CellHead(nn.Module):
    """Patch tokens -> one scalar logit per cell (B, 64, 64).

    Used for the world model's change mask and the policy's click head.
    """

    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decode = nn.Linear(cfg.d_model, cfg.cells_per_patch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        if n != self.cfg.n_tokens:
            raise ValueError(f"expected {self.cfg.n_tokens} tokens, got {n}")
        p = self.cfg.patch_size
        t = self.cfg.tokens_per_side
        logits = self.decode(tokens)  # [B, n, p*p]
        logits = logits.view(b, t, t, p, p).permute(0, 1, 3, 2, 4)
        return logits.reshape(b, GRID, GRID)


def rgb_free_grid_check(grid: torch.Tensor) -> None:
    """Tripwire: values must be palette indices, not RGB bytes."""
    if grid.max() >= PALETTE_SIZE:
        raise ValueError(
            "grid values exceed the 16-colour palette; did you pass RGB? "
            "Use arc3_wm.dynamics_probe.quantize_to_palette first."
        )


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool token sequence -> [B, d]; shared by scalar heads."""
    return tokens.mean(dim=1)


def click_logits_from_cells(cell_logits: torch.Tensor) -> torch.Tensor:
    """(B, 64, 64) cell logits -> (B, 4096) flat click logits (row-major y, x)."""
    return cell_logits.reshape(cell_logits.shape[0], -1)


def assemble_flat_logits(
    type_logits: torch.Tensor, click_logits: torch.Tensor
) -> torch.Tensor:
    """Factored heads -> flat 4102-way logits.

    flat[0..4] = type[0..4]; flat[5 + y*64 + x] = type[ACTION6] +
    log-uniform-normalised click; flat[4101] = type[6]. The click block is
    log-softmaxed before adding the type logit so the factorisation is a
    proper joint distribution: p(a6, cell) = p(a6) * p(cell | a6).
    """
    if type_logits.shape[-1] != N_ACTION_TYPES:
        raise ValueError("type_logits must have 7 entries")
    if click_logits.shape[-1] != ACTION6_COUNT:
        raise ValueError("click_logits must have 4096 entries")
    click_log_p = F.log_softmax(click_logits.float(), dim=-1)
    a6 = type_logits[..., ACTION6_TYPE_INDEX : ACTION6_TYPE_INDEX + 1].float()
    flat = torch.cat(
        [
            type_logits[..., :ACTION6_TYPE_INDEX].float(),
            a6 + click_log_p,
            type_logits[..., N_ACTION_TYPES - 1 :].float(),
        ],
        dim=-1,
    )
    if flat.shape[-1] != N_ACTIONS:
        raise AssertionError("flat logits assembly produced wrong width")
    return flat
