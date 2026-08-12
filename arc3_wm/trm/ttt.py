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


class PolicySelfImitation:
    """Test-time self-imitation for the policy (the NVARC TTFT analogue).

    NVARC fine-tunes the answer producer on the test task's demonstration
    pairs; a new interactive game has no demonstrations, so the analogue
    is the agent's own SUCCESSFUL level segments: every stretch of play
    that ends in a level clear is a self-earned demo of a complete level
    solution. The policy (plain BC or plan refiner) is fine-tuned on those
    segments with the NVARC TTFT hyperparameters (lr 1e-4, warmup 200,
    same objective as pretraining, EMA); exploratory/failed play is never
    imitated.
    """

    def __init__(
        self,
        policy,
        lr: float = 1e-4,
        warmup_steps: int = 200,
        batch_size: int = 32,
        ema_decay: float = 0.999,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.cfg = TrainConfig(
            lr=lr, warmup_steps=warmup_steps, batch_size=batch_size,
            bf16=False, seed=seed,
        )
        self.optimizer = make_optimizer(policy, self.cfg)
        self.ema = EMAHelper(policy, decay=ema_decay)
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        # Each segment: (grids [T,64,64], actions [T], masks [T,4102]),
        # ending at (and including) the clearing action.
        self.segments: list[tuple] = []
        self._pending_new = 0

    def ingest(self, record: dict) -> int:
        """Extract successful level segments from a recorded episode."""
        rewards = record.get("rewards", [])
        grids, actions = record.get("grids"), record.get("actions")
        if not grids:
            return 0
        start = 0
        added = 0
        for j, r in enumerate(rewards):
            if r > 0:
                seg = slice(start, j + 1)
                self.segments.append(
                    (np.stack(grids[seg]), np.asarray(actions[seg]),
                     np.stack(record["masks"][seg]))
                )
                added += 1
                start = j + 1
        self._pending_new += added
        return added

    def _sample_batch(self, device):
        import torch

        k = getattr(self.policy, "plan_length", 1)
        idx = self.rng.integers(0, len(self.segments), size=self.cfg.batch_size)
        g, a, m, pv = [], [], [], []
        for si in idx:
            grids, actions, masks = self.segments[si]
            t = int(self.rng.integers(0, len(actions)))
            g.append(grids[t])
            m.append(masks[t])
            if k == 1:
                a.append(int(actions[t]))
            else:
                lab = np.zeros(k, dtype=np.int64)
                val = np.zeros(k, dtype=np.float32)
                tail = actions[t : t + k]
                lab[: len(tail)] = tail
                val[: len(tail)] = 1.0
                a.append(lab)
                pv.append(val)
        batch = {
            "grid": torch.from_numpy(np.stack(g).astype(np.int64)).to(device),
            "action": torch.as_tensor(np.asarray(a), dtype=torch.long).to(device),
            "mask": torch.from_numpy(np.stack(m)).to(device),
        }
        if k > 1:
            batch["plan_valid"] = torch.from_numpy(np.stack(pv)).to(device)
        return batch

    def update(self, max_steps: int = 50) -> dict:
        """One fine-tuning burst over the accumulated success segments."""
        if not self.segments or self._pending_new == 0:
            return {"steps": self.global_step, "segments": len(self.segments)}
        n_batches = int(np.clip(self._pending_new * 4, 1, max_steps))
        device = next(self.policy.parameters()).device
        self.policy.train()
        agg: dict = {}
        for _ in range(n_batches):
            parts, consumed = deep_supervision_batch(
                self.policy, self._sample_batch(device), self.optimizer,
                self.ema, self.cfg, self.global_step, mode="bc",
            )
            self.global_step += consumed
            agg = parts
        self.policy.eval()
        self._pending_new = 0
        return {"steps": self.global_step, "segments": len(self.segments), **{
            k: round(float(v), 4) for k, v in agg.items()
        }}
