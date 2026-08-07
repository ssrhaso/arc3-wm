"""Composable online agents for the ARC3 Gym env.

``TRMAgent`` scores candidate actions by summing whichever component
signals are enabled (AgentConfig):

    score(a) = w_bc * log pi_BC(a | s)                   [policy component]
             + w_reward * P(level clear | WM(s, a))      [world model]
             + w_novelty * novelty(predicted next state) [world model]
             + w_change * P(frame changes | WM(s, a))    [world model]
             - w_reward * P(GAME_OVER | WM(s, a))        [world model]

Novelty is count-based over exact grid hashes (the environment is
deterministic and discrete, where hash-frontier exploration is the strong
known baseline). The per-game action mask is always enforced. ACTION6
candidates are pruned to salient cells (non-background + recently changed)
capped at ``max_click_candidates``.

Degenerate compositions: use_bc only -> BC agent; use_wm only -> model-based
novelty planner; neither -> masked random agent.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..action_space import ACTION6_BASE, ACTION6_COUNT, ACTION7_INDEX, GRID, N_ACTIONS
from .config import AgentConfig


def salient_click_cells(
    grid: np.ndarray,
    prev_grid: Optional[np.ndarray],
    max_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Candidate (y, x) cells for ACTION6, most-salient first.

    Priority: cells that changed since the previous frame, then cells not of
    the background colour (the modal colour), then a uniform grid subsample.
    Returns flat cell indices (y * 64 + x), at most ``max_cells``.
    """
    changed = (
        np.nonzero((grid != prev_grid).ravel())[0]
        if prev_grid is not None
        else np.array([], dtype=np.int64)
    )
    background = np.bincount(grid.ravel(), minlength=16).argmax()
    non_bg = np.nonzero((grid != background).ravel())[0]
    stride = np.arange(0, GRID * GRID, 8 * GRID + 8)  # sparse fallback lattice
    ordered: list[int] = []
    seen: set[int] = set()
    for pool in (changed, rng.permutation(non_bg), stride):
        for cell in pool:
            c = int(cell)
            if c not in seen:
                seen.add(c)
                ordered.append(c)
            if len(ordered) >= max_cells:
                return np.asarray(ordered, dtype=np.int64)
    return np.asarray(ordered, dtype=np.int64)


def candidate_actions(
    mask: np.ndarray,
    grid: np.ndarray,
    prev_grid: Optional[np.ndarray],
    max_click_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Masked flat action candidates: all simple actions + pruned clicks."""
    if mask.shape != (N_ACTIONS,):
        raise ValueError("mask must be the flat 4102 availability mask")
    simple = [i for i in (0, 1, 2, 3, 4, ACTION7_INDEX) if mask[i]]
    actions = list(simple)
    if mask[ACTION6_BASE : ACTION6_BASE + ACTION6_COUNT].any():
        cells = salient_click_cells(grid, prev_grid, max_click_candidates, rng)
        actions.extend(int(ACTION6_BASE + c) for c in cells if mask[ACTION6_BASE + c])
    if not actions:  # degenerate mask; fall back to whatever is available
        actions = np.nonzero(mask)[0].tolist()
    return np.asarray(actions, dtype=np.int64)


class NoveltyMemory:
    """Exact-hash visit counts over grids; novelty = 1 / count^power."""

    def __init__(self, power: float = 0.5) -> None:
        self.power = power
        self.counts: Counter[bytes] = Counter()

    def observe(self, grid: np.ndarray) -> None:
        self.counts[grid.astype(np.uint8).tobytes()] += 1

    def novelty(self, grid: np.ndarray) -> float:
        count = self.counts[grid.astype(np.uint8).tobytes()]
        if count == 0:
            return 1.0
        return float(1.0 / (count + 1) ** self.power)


class TRMAgent:
    """Composition of TRM policy and/or TRM world model for online play.

    ``policy`` and ``world_model`` are torch modules (or None, per config);
    the agent itself is numpy-facing: ``act(obs_grid, mask)`` -> flat index.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        policy=None,
        world_model=None,
        device: str = "cpu",
    ) -> None:
        if cfg.use_bc and policy is None:
            raise ValueError("use_bc requires a policy")
        if cfg.use_wm and world_model is None:
            raise ValueError("use_wm requires a world_model")
        self.cfg = cfg
        self.policy = policy
        self.world_model = world_model
        self.device = device
        self.memory = NoveltyMemory(cfg.novelty_count_power)
        self.rng = np.random.default_rng(cfg.seed)
        self._prev_grid: Optional[np.ndarray] = None

    def reset(self) -> None:
        """New episode: clear the frame-delta context (novelty persists;
        revisiting a start state should not look novel)."""
        self._prev_grid = None

    def act(self, grid: np.ndarray, mask: np.ndarray) -> int:
        import torch

        cfg = self.cfg
        self.memory.observe(grid)
        cands = candidate_actions(
            mask, grid, self._prev_grid, cfg.max_click_candidates, self.rng
        )
        if self.rng.random() < cfg.epsilon:
            choice = int(self.rng.choice(cands))
            self._prev_grid = grid.copy()
            return choice

        scores = np.zeros(len(cands), dtype=np.float64)
        with torch.no_grad():
            if cfg.use_bc:
                g = torch.from_numpy(grid.astype(np.int64))[None].to(self.device)
                m = torch.from_numpy(mask.copy())[None].to(self.device)
                _, out = self.policy.act(g, mask=m, temperature=0.0)
                log_p = torch.log_softmax(out.flat_logits[0].float(), dim=-1)
                scores += cfg.w_bc * log_p.cpu().numpy()[cands]
            if cfg.use_wm:
                g = torch.from_numpy(grid.astype(np.int64))[None].to(self.device)
                batch = g.expand(len(cands), -1, -1)
                acts = torch.from_numpy(cands).to(self.device)
                out = self.world_model.predict(
                    batch, acts, max_steps=cfg.wm_predict_steps
                )
                pred = out.next_logits.argmax(-1).cpu().numpy()
                if out.reward_logit is not None:
                    scores += cfg.w_reward * torch.sigmoid(out.reward_logit).cpu().numpy()
                if out.state_logits is not None:
                    p_state = torch.softmax(out.state_logits.float(), dim=-1).cpu().numpy()
                    scores -= cfg.w_reward * p_state[:, 2]  # avoid predicted GAME_OVER
                if out.change_logits is not None:
                    p_change = torch.sigmoid(out.change_logits.float()).mean((1, 2))
                    scores += cfg.w_change * p_change.cpu().numpy()
                scores += cfg.w_novelty * np.asarray(
                    [self.memory.novelty(pred[i]) for i in range(len(cands))]
                )

        best = np.flatnonzero(scores == scores.max())
        choice = int(cands[self.rng.choice(best)])
        self._prev_grid = grid.copy()
        return choice


def run_episode(
    env,
    agent: TRMAgent,
    max_actions: Optional[int] = None,
) -> dict:
    """Play one episode; returns the EvalRewardSink-compatible record
    {"rewards": [...], "terminal_state": ...} plus diagnostics."""
    from ..dynamics_probe import quantize_to_palette

    obs, info = env.reset()
    agent.reset()
    rewards: list[float] = []
    steps = 0
    limit = max_actions or 10**9
    terminated = truncated = False
    while not (terminated or truncated) and steps < limit:
        grid = np.asarray(quantize_to_palette(obs), dtype=np.uint8)
        action = agent.act(grid, np.asarray(info["action_mask"], dtype=bool))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        steps += 1
    return {
        "rewards": rewards,
        "terminal_state": info.get("state"),
        "levels_completed": int(info.get("levels_completed", 0)),
        "steps": steps,
    }
