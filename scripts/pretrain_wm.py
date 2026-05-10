"""Stub for scripts/pretrain_wm.py — Phase 3 cross-game WM pretraining.

Tests are written first (test-first discipline per CLAUDE.md). The full
implementation lands after Haso reviews the red test run; this stub
exists so ``tests/test_pretrain_wm.py`` collects cleanly and fails with
``NotImplementedError`` against each public symbol rather than a single
``ModuleNotFoundError`` collection error.

Public surface (matches ``tests/test_pretrain_wm.py`` and the design
brief in the chat that opened this file):

- ``build_argparser`` / ``parse_args`` — same shape as
  ``scripts/launch_pergame.py`` but with ``--replays-root`` instead of
  ``--task``. Default ``--configs`` ladder is ``size12m arc3 pretrain``.
- ``load_merged_configs`` — re-merges ``dreamerv3/configs.yaml`` +
  ``configs/arc3.yaml``; the latter must define a ``pretrain`` block
  (Phase 3 deliverable, not yet on disk).
- ``build_config(args, leftover)`` — ``elements.Config`` builder. Sets
  ``script="pretrain_wm"`` so a stray ``embodied.run.train`` invocation
  trips on an unknown script (defensive belt-and-braces; the actual
  Phase-3 gate is in the run loop).
- ``populate_buffer_from_replays(replay, root, *, stats=None)`` —
  iterates ``arc3_wm.replay_loader.load_replays_directory`` and calls
  ``replay.add(step)`` for every step dict. Returns the total transition
  count. ``stats["per_game_counts"]`` exposes the per-game distribution
  for the Phase-3 even-distribution gate; ``stats["noise_rows_discarded"]``
  is threaded through from the loader.
- ``make_wm_only_agent(config)`` — constructs an
  ``arc3_wm.wm_only_agent.WMOnlyAgent`` (subclass of
  ``dreamerv3.agent.Agent``) that exposes a ``wm_train`` entry point
  branching BEFORE ``self.imagine(...)`` so imagination rollouts +
  actor/critic loss computation are skipped entirely (saves ~2h on the
  6h budget). The regular ``train`` method is inherited unchanged but
  never called by the pretrain loop. Phase-3 gate row 2: verified by
  spying that ``self.opt.step`` count is 0 and ``self.wm_opt.step``
  count is > 0 across the loop.
- ``pretrain_wm_loop(agent, replay, logger, args)`` — sibling of
  ``embodied.run.train``. Custom run loop that calls ``agent.wm_train``
  on samples from ``replay``, never invokes a Driver, never calls
  ``agent.policy``, writes checkpoints under ``logdir/ckpt`` and
  ``cp.load_or_save()`` on entry so resume from preemption works.
- ``RHAEHeldOutHook(holdout, every_n_steps)`` — callable that returns
  a metrics dict every ``every_n_steps`` steps and ``None`` otherwise.
  Uses the agent's reward / continue heads to predict per-step
  level-up probability on a held-out replay subset; emits keys under
  the ``rhae/`` prefix.
- ``main(argv)`` — CLI entry. Imports the heavy DV3 / JAX stack lazily
  inside helpers so the module is importable on a laptop.

See ``docs/phase-checklists.md`` §"Phase 3" for the gate matrix and
``docs/design-decisions.md`` D11/D12 for why we own the launcher
instead of forking dreamerv3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# Match the launcher's path discipline: the heavy stack is on
# third_party/dreamerv3, but never imported at module top.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
if _DV3.is_dir() and str(_DV3) not in sys.path:
    sys.path.insert(0, str(_DV3))


_STUB_MSG = (
    "scripts.pretrain_wm: stub awaiting impl — see "
    "tests/test_pretrain_wm.py for the contract"
)


def build_argparser() -> argparse.ArgumentParser:
    raise NotImplementedError(_STUB_MSG)


def parse_args(argv: Optional[Sequence[str]] = None):
    raise NotImplementedError(_STUB_MSG)


def load_merged_configs() -> dict:
    raise NotImplementedError(_STUB_MSG)


def build_config(args, leftover):
    raise NotImplementedError(_STUB_MSG)


def populate_buffer_from_replays(
    replay: Any,
    root: Path,
    *,
    stats: Optional[dict[str, Any]] = None,
) -> int:
    raise NotImplementedError(_STUB_MSG)


def make_wm_only_agent(config):
    raise NotImplementedError(_STUB_MSG)


def pretrain_wm_loop(*, agent, replay, logger, args) -> None:
    raise NotImplementedError(_STUB_MSG)


class RHAEHeldOutHook:
    """Periodic held-out-replay hook. See module docstring."""

    def __init__(self, *, holdout, every_n_steps: int) -> None:
        raise NotImplementedError(_STUB_MSG)

    def __call__(self, *, step: int, agent) -> Optional[dict[str, Any]]:
        raise NotImplementedError(_STUB_MSG)


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise NotImplementedError(_STUB_MSG)
