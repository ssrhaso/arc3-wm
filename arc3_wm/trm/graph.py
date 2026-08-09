"""State-graph components for the scientist-loop agent.

ARC-AGI-3 dynamics are single-frame deterministic to 99%+ (measured on the
replay corpus), so the visited state space is an exactly replayable graph:
nodes are (level, grid) states, edges are tried actions. That buys two
things a greedy novelty walker cannot do: navigate to the nearest untested
edge via known shortest paths (no wasted exploration actions), and replay
the shortest known path to a discovered level-clear edge (one clean
execution episode after the exploration episodes - the meta-policy RHAE's
min-over-episodes scoring rewards).

Also here: connected-component object extraction, which shrinks the
ACTION6 candidate set from 4096 cells to object centroids.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np

from ..action_space import ACTION6_BASE, ACTION6_COUNT, ACTION7_INDEX, GRID, N_ACTIONS

NodeKey = tuple[int, bytes]  # (levels_completed, grid bytes)


def node_key(level: int, grid: np.ndarray) -> NodeKey:
    return (int(level), grid.astype(np.uint8).tobytes())


def objects_from_grid(grid: np.ndarray, max_objects: int = 24) -> list[int]:
    """Connected components (4-conn, same colour, non-background) -> flat
    click cell indices (y * 64 + x) at component centroids, smallest
    components first (buttons and pieces before backdrops)."""
    background = int(np.bincount(grid.ravel(), minlength=16).argmax())
    labels = np.full(grid.shape, -1, dtype=np.int32)
    components: list[tuple[int, int, int]] = []  # (area, cy, cx)
    next_label = 0
    for sy in range(GRID):
        for sx in range(GRID):
            if labels[sy, sx] != -1 or grid[sy, sx] == background:
                continue
            colour = grid[sy, sx]
            stack = [(sy, sx)]
            labels[sy, sx] = next_label
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < GRID and 0 <= nx < GRID
                        and labels[ny, nx] == -1 and grid[ny, nx] == colour
                    ):
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            cy, cx = int(round(sum(ys) / len(ys))), int(round(sum(xs) / len(xs)))
            if grid[cy, cx] == background:  # centroid off-object (L-shapes)
                cy, cx = cells[len(cells) // 2]
            components.append((len(cells), cy, cx))
            next_label += 1
    components.sort()
    return [cy * GRID + cx for _, cy, cx in components[:max_objects]]


def graph_candidates(
    grid: np.ndarray, mask: np.ndarray, max_click_objects: int
) -> list[int]:
    """Masked candidate actions for a node: simple actions + object clicks."""
    cands = [a for a in (0, 1, 2, 3, 4, ACTION7_INDEX) if mask[a]]
    if mask[ACTION6_BASE : ACTION6_BASE + ACTION6_COUNT].any():
        for cell in objects_from_grid(grid, max_click_objects):
            idx = ACTION6_BASE + cell
            if mask[idx]:
                cands.append(int(idx))
    return cands


class StateGraph:
    """Deterministic transition graph over visited states."""

    def __init__(self) -> None:
        self.candidates: dict[NodeKey, list[int]] = {}
        self.edges: dict[NodeKey, dict[int, NodeKey]] = {}
        self.terminal: dict[tuple[NodeKey, int], str] = {}  # WIN / GAME_OVER
        self.reward_edges: set[tuple[NodeKey, int]] = set()
        self.deferred: dict[NodeKey, list[int]] = {}  # WM-predicted no-ops

    def ensure_node(self, key: NodeKey, candidates: Iterable[int]) -> None:
        if key not in self.candidates:
            self.candidates[key] = list(candidates)
            self.edges[key] = {}

    def record(
        self,
        src: NodeKey,
        action: int,
        dst: NodeKey,
        reward: float,
        terminal_state: Optional[str],
    ) -> None:
        self.edges.setdefault(src, {})[action] = dst
        if reward > 0:
            self.reward_edges.add((src, action))
        if terminal_state in ("WIN", "GAME_OVER"):
            self.terminal[(src, action)] = terminal_state

    def untested(self, key: NodeKey) -> list[int]:
        tried = self.edges.get(key, {})
        return [a for a in self.candidates.get(key, []) if a not in tried]

    def _traversable(self, src: NodeKey, action: int) -> bool:
        # Terminal edges end the episode; never walk through them.
        return (src, action) not in self.terminal

    def bfs(self, start: NodeKey, is_target) -> Optional[list[int]]:
        """Shortest action path through KNOWN edges from ``start`` to the
        first node where ``is_target(node)`` holds. Returns None if
        unreachable; [] if start is already a target."""
        if is_target(start):
            return []
        seen = {start}
        queue: deque[tuple[NodeKey, list[int]]] = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            for action, nxt in self.edges.get(node, {}).items():
                if not self._traversable(node, action) or nxt in seen:
                    continue
                if is_target(nxt):
                    return path + [action]
                seen.add(nxt)
                queue.append((nxt, path + [action]))
        return None

    def path_to_frontier(self, start: NodeKey) -> Optional[list[int]]:
        return self.bfs(start, lambda n: len(self.untested(n)) > 0)

    def path_to_reward(self, start: NodeKey) -> Optional[list[int]]:
        """Shortest path to (and through) a known rewarding edge."""
        sources = {src for src, _ in self.reward_edges}
        if not sources:
            return None
        path = self.bfs(start, lambda n: n in sources)
        if path is None:
            return None
        node = start
        for action in path:
            node = self.edges[node][action]
        reward_action = next(a for s, a in self.reward_edges if s == node)
        return path + [reward_action]

    def successor(self, src: NodeKey, action: int) -> Optional[NodeKey]:
        return self.edges.get(src, {}).get(action)
