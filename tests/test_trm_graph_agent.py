"""Spec for the state graph and the scientist-loop agent."""

from __future__ import annotations

import numpy as np
import pytest

from arc3_wm.action_space import build_mask
from arc3_wm.palette import decode_frame
from arc3_wm.trm.config import GraphAgentConfig
from arc3_wm.trm.graph import (
    StateGraph,
    graph_candidates,
    node_key,
    objects_from_grid,
)
from arc3_wm.trm.graph_agent import GraphAgent, run_graph_episode


def test_objects_from_grid_centroids_exclude_background():
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[10:14, 10:14] = 3  # 4x4 block at (12, 12)-ish
    grid[40, 40] = 5  # single cell
    cells = objects_from_grid(grid, max_objects=8)
    assert 40 * 64 + 40 in cells  # smallest first
    assert cells[0] == 40 * 64 + 40
    assert any(c // 64 in (11, 12) and c % 64 in (11, 12) for c in cells)
    assert len(cells) == 2  # background not an object


def test_graph_candidates_respect_mask():
    grid = np.zeros((64, 64), dtype=np.uint8)
    grid[5, 5] = 7
    cands = graph_candidates(grid, build_mask([1, 2, 6]), max_click_objects=4)
    assert 0 in cands and 1 in cands
    assert any(c >= 5 for c in cands)  # the object click
    cands_no_click = graph_candidates(grid, build_mask([1]), max_click_objects=4)
    assert cands_no_click == [0]


def test_state_graph_bfs_and_terminal_avoidance():
    g = StateGraph()
    a, b, c, d = (0, b"a"), (0, b"b"), (0, b"c"), (0, b"d")
    for n in (a, b, c):
        g.ensure_node(n, [0, 1])
    g.ensure_node(d, [0])
    g.record(a, 0, b, 0.0, None)
    g.record(b, 0, c, 0.0, None)
    g.record(b, 1, d, 0.0, "GAME_OVER")  # terminal edge: never traversed
    g.record(c, 1, c, 1.0, None)  # rewarding self-loop edge at c
    assert g.untested(a) == [1]
    # b has tried both actions; the nearest frontier is c (action 0 untested).
    assert g.path_to_frontier(b) == [0]
    assert g.path_to_reward(a) == [0, 0, 1]
    # d is only reachable through a terminal edge: BFS must not use it.
    assert g.bfs(a, lambda n: n == d) is None


class ChainEnv:
    """S0 -1-> S1 -1-> S2 -1-> WIN; action 0 is a no-op everywhere."""

    def __init__(self):
        self.state = 0
        self.steps = 0

    def _obs(self):
        grid = np.full((64, 64), self.state + 1, dtype=np.uint8)
        info = {
            "action_mask": build_mask([1, 2]),
            "levels_completed": 0,
            "state": "NOT_FINISHED",
        }
        return decode_frame(grid), info

    def reset(self, **kw):
        self.state = 0
        self.steps = 0
        return self._obs()

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        if action == 1:
            if self.state < 2:
                self.state += 1
            else:
                reward = 1.0
                terminated = True
        obs, info = self._obs()
        if terminated:
            info = info | {"state": "WIN", "levels_completed": 1}
        return obs, reward, terminated, self.steps >= 50, info


def test_graph_agent_explores_then_executes_optimally():
    agent = GraphAgent(GraphAgentConfig(seed=0))
    env = ChainEnv()
    ep1 = run_graph_episode(env, agent, max_actions=50)
    assert sum(ep1["rewards"]) == 1.0  # systematic probing finds the win
    explore_len = ep1["steps"]
    ep2 = run_graph_episode(env, agent, max_actions=50)
    assert sum(ep2["rewards"]) == 1.0
    assert ep2["steps"] == 3  # shortest path executed exactly
    assert ep2["steps"] <= explore_len
    assert agent.stats["execute"] >= 1


def test_graph_agent_path_mismatch_recovers():
    agent = GraphAgent(GraphAgentConfig(seed=0))
    env = ChainEnv()
    run_graph_episode(env, agent, max_actions=50)
    # Corrupt the graph: pretend the reward path goes through a wrong node.
    agent.reset()
    obs, info = env.reset()
    from arc3_wm.dynamics_probe import quantize_to_palette

    grid = np.asarray(quantize_to_palette(obs), dtype=np.uint8)
    first = agent.act(grid, np.asarray(info["action_mask"], dtype=bool), 0)
    assert first == 1  # heads for the goal
    # Feed a mismatching observation: agent should drop the plan, not crash.
    wrong = np.full((64, 64), 9, dtype=np.uint8)
    agent.observe_transition(grid, first, wrong, 0.0, None, 0, 0)
    action = agent.act(wrong, np.asarray(info["action_mask"], dtype=bool), 0)
    assert isinstance(action, int)
    assert agent.stats["mismatch"] >= 1


def test_node_key_separates_levels():
    grid = np.zeros((64, 64), dtype=np.uint8)
    assert node_key(0, grid) != node_key(1, grid)


class TickingChainEnv(ChainEnv):
    """ChainEnv plus a progress bar: cell (63, t) flips each step - every
    raw frame is unique, defeating naive hashing."""

    def _obs(self):
        obs, info = super()._obs()
        from arc3_wm.dynamics_probe import quantize_to_palette

        grid = quantize_to_palette(obs).astype(np.uint8)
        grid[63, :] = 0  # static UI strip, independent of game state
        for t in range(min(self.steps, 63)):
            grid[63, t] = 9
        return decode_frame(grid), info


def test_clock_calibration_recovers_graph_reuse():
    agent = GraphAgent(GraphAgentConfig(seed=0))
    env = TickingChainEnv()
    r1 = run_graph_episode(env, agent, max_actions=50)
    r2 = run_graph_episode(env, agent, max_actions=50)
    # Third reset triggers calibration from the two divergent episodes.
    r3 = run_graph_episode(env, agent, max_actions=50)
    assert agent.clock_mask is not None and agent.clock_mask.any()
    assert agent.stats["clock_cells"] > 0
    # Avatar/state cells (row 0-62 fill) must NOT be masked.
    assert not agent.clock_mask[:63, :].any()
    # With the clock masked, the goal path replays exactly.
    r4 = run_graph_episode(env, agent, max_actions=50)
    assert sum(r4["rewards"]) == 1.0
    assert r4["steps"] == 3
