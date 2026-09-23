"""Procedural synthetic games for world-model pretraining (the SDG move).

The decisive ingredient of the ARC-AGI-2 reference solution was a corpus of
synthetic tasks that turned a memorising TRM into a cross-task prior. The
interactive analogue: sample small grid *games* from parametrised rule
families, roll policies in them, and emit transition corpora in exactly the
npz cache format the training pipeline already consumes - so a
synthetic-pretrained WM drops into every existing script unchanged.

Rule families (each instance samples colours, sizes, layouts):
- maze:    avatar block, walls, goal cell; directional moves w/ collision;
           optional energy-bar UI strip (teaches clock patterns)
- pusher:  sokoban-style block pushing onto a target
- toggle:  clicking an object cycles its colour; match a target palette
- sweeper: clicking an object removes it; clear the board
- faller:  a click drops a block down its column (gravity + stacking)

Episodes mix random behaviour with scripted goal-reaching so reward and
terminal transitions occur (the reward head needs positives). Data is
written per-game as ``synth###.npz`` via the standard cache fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..action_space import ACTION6_BASE, GRID

FAMILIES = ("maze", "pusher", "toggle", "sweeper", "faller")


def _blank(bg: int) -> np.ndarray:
    return np.full((GRID, GRID), bg, dtype=np.uint8)


def _paint_block(grid, y, x, size, colour) -> None:
    grid[y : y + size, x : x + size] = colour


class _Game:
    """Minimal interactive interface shared by all families."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.level = 0
        self.steps = 0

    def obs(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> tuple[float, Optional[str]]:
        raise NotImplementedError

    def scripted_action(self) -> Optional[int]:
        """An action moving toward the goal, when the family knows one."""
        return None

    avail: tuple[int, ...] = (1, 2, 3, 4)


class MazeGame(_Game):
    avail = (1, 2, 3, 4)

    def __init__(self, rng):
        super().__init__(rng)
        self.bg = int(rng.integers(0, 16))
        colours = [c for c in range(16) if c != self.bg]
        self.wall_c, self.avatar_c, self.goal_c = rng.choice(colours, 3, replace=False)
        self.size = int(rng.integers(2, 5))
        self.energy_ui = bool(rng.random() < 0.5)
        self.energy = int(rng.integers(60, 120))
        self.cells = GRID // self.size
        self.walls = rng.random((self.cells, self.cells)) < 0.22
        self.walls[0, 0] = False
        free = np.argwhere(~self.walls)
        self.pos = tuple(free[0])
        self.goal = tuple(free[rng.integers(1, len(free))])
        self.walls[self.goal] = False

    def obs(self):
        g = _blank(self.bg)
        s = self.size
        for (y, x) in np.argwhere(self.walls):
            _paint_block(g, y * s, x * s, s, self.wall_c)
        _paint_block(g, self.goal[0] * s, self.goal[1] * s, s, self.goal_c)
        _paint_block(g, self.pos[0] * s, self.pos[1] * s, s, self.avatar_c)
        if self.energy_ui:
            g[63, :] = self.bg
            g[63, : max(0, self.energy // 2)] = self.wall_c
        return g

    def step(self, action):
        dy, dx = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if 0 <= ny < self.cells and 0 <= nx < self.cells and not self.walls[ny, nx]:
            self.pos = (ny, nx)
            self.energy -= 1
        if self.pos == self.goal:
            self.level += 1
            return 1.0, "WIN"
        if self.energy <= 0:
            return 0.0, "GAME_OVER"
        return 0.0, None

    def scripted_action(self):
        # Greedy step toward the goal through free cells.
        best, best_d = None, None
        for a, (dy, dx) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
            ny, nx = self.pos[0] + dy, self.pos[1] + dx
            if 0 <= ny < self.cells and 0 <= nx < self.cells and not self.walls[ny, nx]:
                d = abs(ny - self.goal[0]) + abs(nx - self.goal[1])
                if best_d is None or d < best_d:
                    best, best_d = a, d
        return best


class PusherGame(MazeGame):
    def __init__(self, rng):
        super().__init__(rng)
        free = [tuple(c) for c in np.argwhere(~self.walls)
                if tuple(c) not in (self.pos, self.goal)]
        self.box = free[rng.integers(0, len(free))]
        self.box_c = (self.wall_c + 5) % 16
        if self.box_c == self.bg:
            self.box_c = (self.box_c + 1) % 16

    def obs(self):
        g = super().obs()
        _paint_block(g, self.box[0] * self.size, self.box[1] * self.size,
                     self.size, self.box_c)
        return g

    def step(self, action):
        dy, dx = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if not (0 <= ny < self.cells and 0 <= nx < self.cells) or self.walls[ny, nx]:
            return 0.0, None
        if (ny, nx) == self.box:
            by, bx = ny + dy, nx + dx
            if not (0 <= by < self.cells and 0 <= bx < self.cells) or self.walls[by, bx]:
                return 0.0, None
            self.box = (by, bx)
        self.pos = (ny, nx)
        if self.box == self.goal:
            self.level += 1
            return 1.0, "WIN"
        return 0.0, None

    def scripted_action(self):
        return int(self.rng.integers(0, 4))


class ToggleGame(_Game):
    avail = (6,)

    def __init__(self, rng):
        super().__init__(rng)
        self.bg = int(rng.integers(0, 16))
        self.n = int(rng.integers(3, 7))
        self.palette = [c for c in range(16) if c != self.bg]
        self.target = int(rng.choice(self.palette))
        self.spots = []
        self.colours = []
        for _ in range(self.n):
            y, x = int(rng.integers(2, 56)), int(rng.integers(2, 56))
            self.spots.append((y, x))
            self.colours.append(int(rng.choice(self.palette)))

    def obs(self):
        g = _blank(self.bg)
        for (y, x), c in zip(self.spots, self.colours):
            _paint_block(g, y, x, 4, c)
        return g

    def _hit(self, cy, cx):
        for i, (y, x) in enumerate(self.spots):
            if y <= cy < y + 4 and x <= cx < x + 4:
                return i
        return None

    def step(self, action):
        if action >= ACTION6_BASE:
            rel = action - ACTION6_BASE
            i = self._hit(rel // GRID, rel % GRID)
            if i is not None:
                idx = self.palette.index(self.colours[i])
                self.colours[i] = self.palette[(idx + 1) % len(self.palette)]
        if all(c == self.target for c in self.colours):
            self.level += 1
            return 1.0, "WIN"
        return 0.0, None

    def scripted_action(self):
        for i, c in enumerate(self.colours):
            if c != self.target:
                y, x = self.spots[i]
                return ACTION6_BASE + (y + 1) * GRID + (x + 1)
        return None


class SweeperGame(ToggleGame):
    def step(self, action):
        if action >= ACTION6_BASE:
            rel = action - ACTION6_BASE
            i = self._hit(rel // GRID, rel % GRID)
            if i is not None:
                self.spots.pop(i)
                self.colours.pop(i)
        if not self.spots:
            self.level += 1
            return 1.0, "WIN"
        return 0.0, None

    def scripted_action(self):
        if self.spots:
            y, x = self.spots[0]
            return ACTION6_BASE + (y + 1) * GRID + (x + 1)
        return None


class FallerGame(_Game):
    avail = (6,)

    def __init__(self, rng):
        super().__init__(rng)
        self.bg = int(rng.integers(0, 16))
        self.block_c = (self.bg + 1 + int(rng.integers(0, 14))) % 16
        if self.block_c == self.bg:
            self.block_c = (self.block_c + 1) % 16
        self.cols = 8
        self.heights = np.zeros(self.cols, dtype=int)
        self.target = int(rng.integers(3, 6))

    def obs(self):
        g = _blank(self.bg)
        w = GRID // self.cols
        for c in range(self.cols):
            for h in range(self.heights[c]):
                _paint_block(g, GRID - (h + 1) * w, c * w, w, self.block_c)
        return g

    def step(self, action):
        if action >= ACTION6_BASE:
            col = ((action - ACTION6_BASE) % GRID) // (GRID // self.cols)
            if self.heights[col] < self.cols:
                self.heights[col] += 1
        if (self.heights >= self.target).all():
            self.level += 1
            return 1.0, "WIN"
        return 0.0, None

    def scripted_action(self):
        col = int(np.argmin(self.heights))
        w = GRID // self.cols
        return ACTION6_BASE + 2 * GRID + (col * w + 1)


_CLASSES = {"maze": MazeGame, "pusher": PusherGame, "toggle": ToggleGame,
            "sweeper": SweeperGame, "faller": FallerGame}


def make_game(family: str, rng: np.random.Generator) -> _Game:
    return _CLASSES[family](rng)


def rollout_game(
    game: _Game,
    rng: np.random.Generator,
    max_steps: int = 120,
    scripted_prob: float = 0.35,
) -> dict[str, np.ndarray]:
    """One episode -> cache-format arrays (grids/actions/levels/states/avail)."""
    grids = [game.obs()]
    actions = [-1]
    levels = [game.level]
    states = [0]
    avail_row = np.zeros(7, dtype=np.uint8)
    for a in game.avail:
        avail_row[a - 1] = 1
    for _ in range(max_steps):
        scripted = game.scripted_action() if rng.random() < scripted_prob else None
        if scripted is None:
            if 6 in game.avail:
                cell = int(rng.integers(0, GRID * GRID))
                action = ACTION6_BASE + cell
            else:
                action = int(rng.integers(0, 4))
        else:
            action = int(scripted)
        reward, term = game.step(action)
        grids.append(game.obs())
        actions.append(action)
        levels.append(game.level)
        states.append({None: 0, "WIN": 1, "GAME_OVER": 2}[term])
        if term is not None:
            break
    return {
        "grids": np.stack(grids),
        "actions": np.asarray(actions, dtype=np.int16),
        "levels": np.asarray(levels, dtype=np.int16),
        "states": np.asarray(states, dtype=np.int8),
        "avail": np.tile(avail_row, (len(grids), 1)),
    }


def generate_corpus(
    out_dir: Path,
    n_games: int = 200,
    episodes_per_game: int = 4,
    seed: int = 0,
) -> list[Path]:
    """Sample games round-robin over families; one npz per game."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_games):
        family = FAMILIES[i % len(FAMILIES)]
        episodes = []
        for _ in range(episodes_per_game):
            game = make_game(family, rng)
            episodes.append(rollout_game(game, rng))
        starts = np.cumsum(
            [0] + [len(ep["grids"]) for ep in episodes[:-1]]
        ).astype(np.int64)
        path = out_dir / f"synth{i:03d}.npz"
        np.savez_compressed(
            path,
            grids=np.concatenate([ep["grids"] for ep in episodes]),
            actions=np.concatenate([ep["actions"] for ep in episodes]),
            levels=np.concatenate([ep["levels"] for ep in episodes]),
            states=np.concatenate([ep["states"] for ep in episodes]),
            avail=np.concatenate([ep["avail"] for ep in episodes]),
            episode_starts=starts,
        )
        paths.append(path)
    return paths
