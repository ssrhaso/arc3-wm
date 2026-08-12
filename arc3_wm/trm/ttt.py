"""Test-time tuning: fine-tune on the agent's own experience during evaluation.

The reference ARC-AGI-2 recipe's ingredients carried over: per-task reset
(the eval process reloads the checkpoint per game), the test-time
hyperparameters (lr 1e-4, warmup 200, batch 128), the test-time recursion
schedule (H_cycles=4, L_cycles=4, halt/supervision cap 10, applied only
during ``update``), and EMA-weight inference (the agent acts on the EMA
copy - ``trm_eval_agent`` syncs it after every burst via the official
helper's ``ema()``).

What does NOT carry over: the reference recipe fine-tunes the answer
producer on the test task's demonstration pairs (ground truth) with
128-way augmentation voting. A new interactive game has no demonstrations,
so the WM tunes on the agent's own transition log (fresh transitions mixed
with replay) and the policy analogue is self-imitation on its own
successful level segments. Augmentation stays off (ARC-AGI-3 dynamics are
not equivariant).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ._official import OfficialEMAHelper
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
        batch_size: int = 128,  # reference test-time value
        ema_decay: float = 0.999,
        seed: int = 0,
        # Reference test-time recursion schedule (eval-time overrides),
        # applied only during update(); None keeps the training schedule.
        ttt_h_cycles: Optional[int] = 4,
        ttt_l_cycles: Optional[int] = 4,
        ttt_n_supervision: Optional[int] = 10,
        ttt_halt_max: Optional[int] = 10,
    ) -> None:
        self.model = model
        self.cfg = TrainConfig(
            lr=lr, warmup_steps=warmup_steps, batch_size=batch_size,
            bf16=False, seed=seed,
        )
        self.optimizer = make_optimizer(model, self.cfg)
        self.ema = OfficialEMAHelper(mu=ema_decay)
        self.ema.register(model)
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator().manual_seed(seed)
        self.ttt_schedule = (ttt_h_cycles, ttt_l_cycles, ttt_n_supervision, ttt_halt_max)
        self.global_step = 0
        self.seen = 0  # transitions already consumed at least once

    def _batches(self, log, frames, n_steps: int):
        """Sample batches mixing fresh transitions with replay.

        Each batch is at most half fresh transitions (drawn without
        replacement across the burst) and the rest replay of older ones,
        shuffled together. (Previously fresh rows could crowd replay out
        entirely and were fed in trajectory order.)
        """
        n = len(log)
        fresh = list(range(self.seen, n))
        self.rng.shuffle(fresh)
        cursor = 0
        device = next(self.model.parameters()).device
        for _ in range(n_steps):
            take_fresh = min(self.cfg.batch_size // 2, len(fresh) - cursor)
            fresh_idx = fresh[cursor : cursor + take_fresh]
            cursor += take_fresh
            # Replay = already-consumed rows (the whole log once nothing is
            # older yet); fresh rows only enter via the capped fresh draw.
            pool_hi = self.seen if self.seen > 0 else n
            replay = self.rng.integers(
                0, pool_hi, size=self.cfg.batch_size - len(fresh_idx)
            ).tolist()
            idx = fresh_idx + replay
            self.rng.shuffle(idx)
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
        core = self.model.cfg.core
        saved = (core.h_cycles, core.l_cycles, core.n_supervision, core.halt_max_steps)
        if self.ttt_schedule[0] is not None:
            (core.h_cycles, core.l_cycles,
             core.n_supervision, core.halt_max_steps) = self.ttt_schedule
        agg = {}
        try:
            for batch in self._batches(log, frames, n_batches):
                parts, consumed = deep_supervision_batch(
                    self.model, batch, self.optimizer, self.ema, self.cfg,
                    self.global_step, mode="wm", generator=self.torch_gen,
                )
                self.global_step += consumed
                for k, v in parts.items():
                    agg[k] = v
        finally:
            (core.h_cycles, core.l_cycles,
             core.n_supervision, core.halt_max_steps) = saved
        self.model.eval()
        self.seen = n
        return {"steps": self.global_step, "fresh": fresh, "total": n, **{
            k: round(float(v), 4) for k, v in agg.items()
        }}


class PolicySelfImitation:
    """Test-time self-imitation for the policy.

    The reference recipe fine-tunes the answer producer on the test task's
    demonstration pairs; a new interactive game has no demonstrations, so
    the analogue is the agent's own SUCCESSFUL level segments: every stretch
    of play that ends in a level clear is a self-earned demo of a complete
    level solution. The policy (plain BC or plan refiner) is fine-tuned on
    those segments with the reference test-time settings (lr 1e-4, warmup
    200, batch 128, same objective as pretraining, EMA, test-time recursion
    schedule). Exploratory/failed play is never imitated: only segments
    ending in a clear are used, and within them only greedy (non-epsilon)
    steps become training inputs - epsilon-random moves inside a lucky
    segment are not demonstrations worth cloning.
    """

    def __init__(
        self,
        policy,
        lr: float = 1e-4,
        warmup_steps: int = 200,
        batch_size: int = 128,  # reference test-time value
        ema_decay: float = 0.999,
        seed: int = 0,
        ttt_h_cycles: Optional[int] = 4,
        ttt_l_cycles: Optional[int] = 4,
        ttt_n_supervision: Optional[int] = 10,
        ttt_halt_max: Optional[int] = 10,
    ) -> None:
        self.policy = policy
        self.cfg = TrainConfig(
            lr=lr, warmup_steps=warmup_steps, batch_size=batch_size,
            bf16=False, seed=seed,
        )
        self.optimizer = make_optimizer(policy, self.cfg)
        self.ema = OfficialEMAHelper(mu=ema_decay)
        self.ema.register(policy)
        self.rng = np.random.default_rng(seed)
        self.torch_gen = torch.Generator().manual_seed(seed)
        self.ttt_schedule = (ttt_h_cycles, ttt_l_cycles, ttt_n_supervision, ttt_halt_max)
        self.global_step = 0
        # Each segment: (grids [T,64,64], actions [T], masks [T,4102],
        # greedy [T]), ending at (and including) the clearing action.
        self.segments: list[tuple] = []
        self._pending_new = 0

    def ingest(self, record: dict) -> int:
        """Extract successful level segments from a recorded episode."""
        rewards = record.get("rewards", [])
        grids, actions = record.get("grids"), record.get("actions")
        if not grids:
            return 0
        greedy = record.get("greedy")  # per-step flags from run_episode
        start = 0
        added = 0
        for j, r in enumerate(rewards):
            if r > 0:
                seg = slice(start, j + 1)
                if greedy is not None:
                    g = np.asarray(greedy[seg], dtype=bool)
                else:
                    g = np.ones(j + 1 - start, dtype=bool)
                self.segments.append(
                    (np.stack(grids[seg]), np.asarray(actions[seg]),
                     np.stack(record["masks"][seg]), g)
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
            grids, actions, masks, greedy = self.segments[si]
            # Only greedy steps become inputs (their slot-0 label is the
            # action the agent chose deliberately); the plan tail stays as
            # recorded, epsilon steps included - dropping them would break
            # the tail's time continuity.
            cand = np.nonzero(greedy)[0]
            pool = cand if len(cand) else np.arange(len(actions))
            t = int(pool[self.rng.integers(0, len(pool))])
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
        core = self.policy.cfg.core
        saved = (core.h_cycles, core.l_cycles, core.n_supervision, core.halt_max_steps)
        if self.ttt_schedule[0] is not None:
            (core.h_cycles, core.l_cycles,
             core.n_supervision, core.halt_max_steps) = self.ttt_schedule
        agg: dict = {}
        try:
            for _ in range(n_batches):
                parts, consumed = deep_supervision_batch(
                    self.policy, self._sample_batch(device), self.optimizer,
                    self.ema, self.cfg, self.global_step, mode="bc",
                    generator=self.torch_gen,
                )
                self.global_step += consumed
                agg = parts
        finally:
            (core.h_cycles, core.l_cycles,
             core.n_supervision, core.halt_max_steps) = saved
        self.policy.eval()
        self._pending_new = 0
        return {"steps": self.global_step, "segments": len(self.segments), **{
            k: round(float(v), 4) for k, v in agg.items()
        }}
