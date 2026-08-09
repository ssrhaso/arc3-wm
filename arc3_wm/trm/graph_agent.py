"""The scientist-loop agent: explore systematically, then execute.

Per step, in priority order:
1. EXECUTE: if a known rewarding edge is reachable through the graph,
   follow the shortest known path to it (replayable because dynamics are
   deterministic; on any observed mismatch the path is dropped).
2. PROBE: if the current node has untested candidate actions, try one
   (WM-scored when a world model is attached, else first-in-order; edges
   the WM confidently predicts as no-ops are deferred to last).
3. NAVIGATE: BFS through known edges to the nearest node that still has
   untested candidates and replay that path.
4. Fallback: uniform over candidates (disconnected frontier).

The graph persists across episodes; ``reset()`` only clears the pending
path. A later episode that starts in a known state with a discovered goal
goes straight to EXECUTE - the explore-then-execute meta-policy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import GraphAgentConfig
from .graph import StateGraph, graph_candidates, node_key


class GraphAgent:
    def __init__(
        self,
        cfg: GraphAgentConfig,
        world_model=None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.world_model = world_model
        self.device = device
        self.graph = StateGraph()
        self.rng = np.random.default_rng(cfg.seed)
        self._path: list[int] = []
        self._expected: Optional[tuple] = None  # node the path expects us at
        self._prev: Optional[tuple] = None  # (key, action) awaiting outcome
        self.stats = {"probe": 0, "navigate": 0, "execute": 0, "fallback": 0,
                      "mismatch": 0, "pruned": 0}

    def reset(self) -> None:
        self._path = []
        self._expected = None
        self._prev = None

    def observe_transition(
        self,
        prev_grid: np.ndarray,
        action: int,
        grid: np.ndarray,
        reward: float,
        terminal_state: Optional[str],
        prev_level: int,
        level: int,
    ) -> None:
        src = node_key(prev_level, prev_grid)
        dst = node_key(level, grid)
        self.graph.record(src, action, dst, reward, terminal_state)

    def act(self, grid: np.ndarray, mask: np.ndarray, level: int = 0) -> int:
        key = node_key(level, grid)
        self.graph.ensure_node(
            key, graph_candidates(grid, mask, self.cfg.max_click_objects)
        )

        # Replay path verification: are we where the plan expects?
        if self._path and self._expected is not None and key != self._expected:
            self.stats["mismatch"] += 1
            self._path = []
            self._expected = None

        if not self._path:
            goal_path = self.graph.path_to_reward(key)
            if goal_path:
                self._path = goal_path
                self._expected = key
                self.stats["execute"] += 1

        if not self._path:
            untested = self.graph.untested(key)
            if untested:
                action = self._choose_probe(key, grid, untested)
                if action is not None:
                    self.stats["probe"] += 1
                    return action
            frontier_path = self.graph.path_to_frontier(key)
            if frontier_path:
                self._path = frontier_path
                self._expected = key
                self.stats["navigate"] += 1

        if self._path:
            action = self._path.pop(0)
            self._expected = self.graph.successor(self._expected, action)
            if not self._path:
                self._expected = None
            return int(action)

        # Disconnected frontier or fully exhausted node: uniform fallback.
        self.stats["fallback"] += 1
        cands = self.graph.candidates.get(key) or [int(np.flatnonzero(mask)[0])]
        return int(self.rng.choice(cands))

    def _choose_probe(
        self, key, grid: np.ndarray, untested: list[int]
    ) -> Optional[int]:
        """Pick an untested edge; with a WM, prefer predicted-effectful
        edges and defer confident no-ops."""
        if self.world_model is None or self.cfg.wm_noop_prune <= 0:
            return int(untested[0])
        import torch

        with torch.no_grad():
            g = torch.from_numpy(grid.astype(np.int64))[None].to(self.device)
            batch = g.expand(len(untested), -1, -1)
            acts = torch.tensor(untested, device=self.device)
            out = self.world_model.predict(
                batch, acts, max_steps=self.cfg.wm_predict_steps
            )
            pred = out.next_logits.argmax(-1)
            same = (pred == g).flatten(1).all(-1).cpu().numpy()
            p_change = (
                torch.sigmoid(out.change_logits.float()).amax(dim=(1, 2)).cpu().numpy()
                if out.change_logits is not None
                else np.ones(len(untested))
            )
        deferred = self.graph.deferred.setdefault(key, [])
        keep = []
        for i, action in enumerate(untested):
            if same[i] and p_change[i] < self.cfg.wm_noop_prune and action not in deferred:
                deferred.append(action)
                self.stats["pruned"] += 1
            else:
                keep.append(action)
        if keep:
            return int(keep[0])
        # Everything deferred: test deferred edges after all (model may be wrong).
        return int(deferred.pop(0)) if deferred else None


def run_graph_episode(env, agent: GraphAgent, max_actions: int = 1000) -> dict:
    """Play one episode with transition feedback; EvalRewardSink-shaped."""
    from ..dynamics_probe import quantize_to_palette

    obs, info = env.reset()
    agent.reset()
    rewards: list[float] = []
    steps = 0
    terminated = truncated = False
    grid = np.asarray(quantize_to_palette(obs), dtype=np.uint8)
    level = int(info.get("levels_completed", 0))
    while not (terminated or truncated) and steps < max_actions:
        action = agent.act(grid, np.asarray(info["action_mask"], dtype=bool), level)
        obs, reward, terminated, truncated, info = env.step(action)
        new_grid = np.asarray(quantize_to_palette(obs), dtype=np.uint8)
        new_level = int(info.get("levels_completed", 0))
        agent.observe_transition(
            grid, action, new_grid, float(reward), info.get("state"),
            level, new_level,
        )
        grid, level = new_grid, new_level
        rewards.append(float(reward))
        steps += 1
    return {
        "rewards": rewards,
        "terminal_state": info.get("state"),
        "levels_completed": int(info.get("levels_completed", 0)),
        "steps": steps,
        "agent_stats": dict(agent.stats),
    }
