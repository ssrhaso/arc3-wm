"""Test-time training: fine-tune the world model on the agent's own
experience during evaluation.

The NVARC recipe (ARC Prize 2025 winner) applied to the interactive
setting: no demonstrations of the target game are needed - the graph
agent's transition log is the training stream. Between episodes the WM
takes a burst of deep-supervision steps on fresh transitions (recent data
mixed with a replay of older ones), using the NVARC test-time
hyperparameters (lr 1e-4, warmup 200).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .core import EMAHelper
from .training import TrainConfig, deep_supervision_batch, make_optimizer

STATE_INDEX = {"NOT_FINISHED": 0, None: 0, "WIN": 1, "GAME_OVER": 2}


class OnlineFineTuner:
    """Holds the optimizer state across updates; call ``update`` with the
    agent's cumulative transition log after each episode."""

    def __init__(
        self,
        model,
        lr: float = 1e-4,
        warmup_steps: int = 200,
        batch_size: int = 32,
        ema_decay: float = 0.999,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.cfg = TrainConfig(
            lr=lr, warmup_steps=warmup_steps, batch_size=batch_size,
            bf16=False, seed=seed,
        )
        self.optimizer = make_optimizer(model, self.cfg)
        self.ema = EMAHelper(model, decay=ema_decay)
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self.seen = 0  # transitions already consumed at least once

    def _batches(self, log, frames, n_steps: int):
        """Sample batches mixing fresh transitions with replay."""
        n = len(log)
        fresh = list(range(self.seen, n))
        device = next(self.model.parameters()).device
        for _ in range(n_steps):
            replay = self.rng.integers(0, n, size=self.cfg.batch_size // 2).tolist()
            idx = (fresh + replay)[: self.cfg.batch_size] if fresh else replay
            take = [log[i] for i in idx]
            batch = {
                "grid": torch.stack(
                    [torch.from_numpy(frames[t[1]].astype(np.int64)) for t in take]
                ).to(device),
                "action": torch.tensor([t[2] for t in take], dtype=torch.long,
                                       device=device),
                "next_grid": torch.stack(
                    [torch.from_numpy(frames[t[4]].astype(np.int64)) for t in take]
                ).to(device),
                "reward": torch.tensor([float(t[5] > 0) for t in take],
                                       device=device),
                "state": torch.tensor(
                    [STATE_INDEX.get(t[6], 0) for t in take], dtype=torch.long,
                    device=device,
                ),
            }
            yield batch

    def update(self, log: list, frames: dict, max_steps: int = 50) -> dict:
        """Fine-tune on the transition log. ``log`` rows are the graph
        agent's (lvl, src_bytes, action, lvl2, dst_bytes, reward, terminal)
        tuples; ``frames`` maps bytes -> grid arrays. Returns stats."""
        n = len(log)
        fresh = n - self.seen
        if n < self.cfg.batch_size // 2 or fresh == 0:
            return {"steps": self.global_step, "fresh": fresh, "total": n}
        n_batches = int(np.clip(fresh // 8 + 1, 1, max_steps))
        self.model.train()
        agg = {}
        for batch in self._batches(log, frames, n_batches):
            parts, consumed = deep_supervision_batch(
                self.model, batch, self.optimizer, self.ema, self.cfg,
                self.global_step, mode="wm",
            )
            self.global_step += consumed
            for k, v in parts.items():
                agg[k] = v
        self.model.eval()
        self.seen = n
        return {"steps": self.global_step, "fresh": fresh, "total": n, **{
            k: round(float(v), 4) for k, v in agg.items()
        }}
