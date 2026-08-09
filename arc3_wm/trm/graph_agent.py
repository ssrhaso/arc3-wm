"""The scientist-loop agent: explore systematically, then execute.

Per step, in priority order:
1. EXECUTE: if a known rewarding edge is reachable through the graph,
   follow the shortest known path to it (replayable because dynamics are
   deterministic; on any observed mismatch the plan is dropped).
2. PROBE: if the current node has untested candidate actions, try the most
   promising one (global action-type effectiveness prior; with a world
   model also predicted reward/novelty/change, deferring confident no-ops).
3. NAVIGATE: BFS through known edges to the nearest node that still has
   untested candidates and replay that path.
4. Fallback: uniform over candidates (disconnected frontier).

Clock calibration: many games draw a time/energy indicator that mutates
every step, which makes every raw frame unique and defeats graph reuse.
After the first two (action-divergent) episodes, cells whose value
trajectories are identical across both episodes despite different action
sequences are pure functions of time; they are masked out of the node hash
and the graph is rebuilt from the transition log. The graph and all
statistics persist across episodes - RHAE's min-over-episodes scoring
rewards explore-then-execute.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..action_space import ACTION6_BASE, ACTION6_COUNT, ACTION7_INDEX
from .config import GraphAgentConfig
from .graph import NodeKey, StateGraph, graph_candidates, node_key

CLOCK_SENTINEL = 255  # masked cells take this value inside hashed grids


def _action_type(action: int) -> int:
    """Flat action -> type index 0..6 (clicks collapse to type 5)."""
    if ACTION6_BASE <= action < ACTION6_BASE + ACTION6_COUNT:
        return 5
    return 6 if action == ACTION7_INDEX else int(action)


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
        self._expected: Optional[NodeKey] = None
        # Transition log (frames interned by bytes) for graph rebuilds.
        self._frames: dict[bytes, np.ndarray] = {}
        self._log: list[tuple] = []  # (lvl, src_b, action, lvl2, dst_b, r, term)
        self._cands: dict[tuple[int, bytes], list[int]] = {}
        # Clock calibration state. Two complementary detectors:
        # time-identical cells (same trajectory across action-divergent
        # episodes) and irreversible cells (progress/energy bars advance
        # monotonically and never revert, unlike board cells an avatar
        # passes over). A rebuilt graph whose masked hashing creates
        # deterministic conflicts rolls the mask back - self-checking.
        self._episode_seqs: list[list[bytes]] = [[]]
        self.clock_mask: Optional[np.ndarray] = None
        self._seen_bits = np.zeros((64, 64), dtype=np.uint16)
        self._change_count = np.zeros((64, 64), dtype=np.int64)
        self._revert_count = np.zeros((64, 64), dtype=np.int64)
        # Global per-action-type effectiveness [tried, changed].
        self.type_stats = np.zeros((7, 2), dtype=np.int64)
        self.stats = {"probe": 0, "navigate": 0, "execute": 0, "fallback": 0,
                      "mismatch": 0, "pruned": 0, "clock_cells": 0}

    # ---- hashing with clock mask ----

    def _key(self, level: int, grid: np.ndarray) -> NodeKey:
        if self.clock_mask is not None:
            grid = np.where(self.clock_mask, CLOCK_SENTINEL, grid)
        return node_key(level, grid)

    def _intern(self, grid: np.ndarray) -> bytes:
        raw = grid.astype(np.uint8).tobytes()
        if raw not in self._frames:
            self._frames[raw] = grid.astype(np.uint8).copy()
        return raw

    # ---- episode lifecycle ----

    def reset(self) -> None:
        self._path = []
        self._expected = None
        self._seen_bits[:] = 0
        if self._episode_seqs[-1]:
            self._episode_seqs.append([])
        # Recalibrate at each episode boundary until frozen (8 episodes in).
        if len(self._episode_seqs) in range(3, 9):
            self._calibrate_clock()

    def _calibrate_clock(self) -> None:
        ep1, ep2 = self._episode_seqs[0], self._episode_seqs[1]
        n = min(len(ep1), len(ep2))
        if n < 10:
            return
        a = np.stack([self._frames[b] for b in ep1[:n]])
        b = np.stack([self._frames[b] for b in ep2[:n]])
        time_mask = (a == b).all(axis=0) & (a != a[0]).any(axis=0)
        irrev_mask = (self._change_count >= 4) & (self._revert_count == 0)
        base = time_mask | irrev_mask
        # UI-strip expansion: indicators live in bands (an energy bar's
        # cells can individually revert via pickups, but a few of its cells
        # betray the band). Rows/columns with several flagged cells are
        # masked whole; the conflict check below guards against overreach.
        expand = np.zeros((64, 64), dtype=bool)
        expand[base.sum(axis=1) >= 3, :] = True
        expand[:, base.sum(axis=0) >= 3] = True
        old = self.clock_mask
        applied = None
        for mask in (base | expand, base, time_mask):
            if not mask.any():
                continue
            if old is not None and np.array_equal(mask, old):
                applied = old
                break
            self.clock_mask = mask
            self._rebuild_graph()
            if self._conflict_rate() <= 0.10:
                applied = mask
                break
        if applied is None:
            self.clock_mask = old if old is not None else np.zeros((64, 64), dtype=bool)
            self._rebuild_graph()
        self.stats["clock_cells"] = int(self.clock_mask.sum())

    def _conflict_rate(self) -> float:
        """Fraction of logged transitions whose (node, action) maps to more
        than one masked successor - determinism violations from aliasing.

        Terminal transitions are excluded: masking a consumable (energy)
        indicator aliases exactly the death step, and the runtime path
        mismatch handler already recovers from that."""
        seen: dict[tuple, tuple] = {}
        conflicts = 0
        total = 0
        for lvl, src_b, action, lvl2, dst_b, _r, term in self._log:
            if term is not None and term != "NOT_FINISHED":
                continue
            src = self._key(lvl, self._frames[src_b])
            dst = self._key(lvl2, self._frames[dst_b])
            total += 1
            prev = seen.setdefault((src, action), dst)
            conflicts += int(prev != dst)
        return conflicts / max(total, 1)

    def _rebuild_graph(self) -> None:
        graph = StateGraph()
        merged: dict[NodeKey, list[int]] = {}
        for (lvl, src_b), cands in self._cands.items():
            key = self._key(lvl, self._frames[src_b])
            merged[key] = list(dict.fromkeys(merged.get(key, []) + cands))
        for key, cands in merged.items():
            graph.ensure_node(key, cands)
        for lvl, src_b, action, lvl2, dst_b, reward, term in self._log:
            src = self._key(lvl, self._frames[src_b])
            dst = self._key(lvl2, self._frames[dst_b])
            graph.record(src, action, dst, reward, term)
        self.graph = graph

    # ---- observation ----

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
        src_b = self._intern(prev_grid)
        dst_b = self._intern(grid)
        self._log.append(
            (prev_level, src_b, action, level, dst_b, float(reward), terminal_state)
        )
        # Irreversibility statistics (progress bars never revert).
        changed_cells = prev_grid != grid
        new_bits = (1 << grid.astype(np.uint16))
        reverted = changed_cells & ((self._seen_bits & new_bits) != 0)
        self._change_count += changed_cells
        self._revert_count += reverted
        self._seen_bits |= (1 << prev_grid.astype(np.uint16))
        self._seen_bits |= new_bits
        src = self._key(prev_level, prev_grid)
        dst = self._key(level, grid)
        self.graph.record(src, action, dst, reward, terminal_state)
        a_type = _action_type(action)
        self.type_stats[a_type, 0] += 1
        self.type_stats[a_type, 1] += int(src != dst)

    # ---- acting ----

    def act(self, grid: np.ndarray, mask: np.ndarray, level: int = 0) -> int:
        raw_b = self._intern(grid)
        self._episode_seqs[-1].append(raw_b)
        key = self._key(level, grid)
        if key not in self.graph.candidates:
            cands = graph_candidates(grid, mask, self.cfg.max_click_objects)
            self.graph.ensure_node(key, cands)
            self._cands[(level, raw_b)] = cands
        elif (level, raw_b) not in self._cands:
            self._cands[(level, raw_b)] = list(self.graph.candidates[key])

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

        self.stats["fallback"] += 1
        cands = self.graph.candidates.get(key) or [int(np.flatnonzero(mask)[0])]
        return int(self.rng.choice(cands))

    def _type_priors(self, actions: list[int]) -> np.ndarray:
        """Optimistic change-rate prior per action from global type stats."""
        priors = np.empty(len(actions))
        for i, action in enumerate(actions):
            tried, changed = self.type_stats[_action_type(action)]
            priors[i] = 1.0 if tried < 8 else (changed + 1) / (tried + 1)
        return priors

    def _choose_probe(
        self, key: NodeKey, grid: np.ndarray, untested: list[int]
    ) -> Optional[int]:
        if self.world_model is None or self.cfg.wm_noop_prune <= 0:
            priors = self._type_priors(untested)
            return int(untested[int(priors.argmax())])
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
            p_reward = (
                torch.sigmoid(out.reward_logit.float()).cpu().numpy()
                if out.reward_logit is not None
                else np.zeros(len(untested))
            )
            pred_np = pred.cpu().numpy().astype(np.uint8)
        level = key[0]
        novel = np.array(
            [float(self._key(level, pred_np[i]) not in self.graph.edges)
             for i in range(len(untested))]
        )
        deferred = self.graph.deferred.setdefault(key, [])
        keep: list[int] = []
        scores: list[float] = []
        priors = self._type_priors(untested)
        for i, action in enumerate(untested):
            if same[i] and p_change[i] < self.cfg.wm_noop_prune and action not in deferred:
                deferred.append(action)
                self.stats["pruned"] += 1
            else:
                keep.append(action)
                scores.append(
                    10.0 * p_reward[i] + novel[i] + p_change[i] + priors[i]
                )
        if keep:
            return int(keep[int(np.argmax(scores))])
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
