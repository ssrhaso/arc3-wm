"""scripts/pretrain_wm.py — Phase 3 cross-game WM pretraining.

Sibling of ``scripts/launch_pergame.py``. The launcher trains
end-to-end on a single arc3_<game> task; this script pretrains the
World Model only, on a buffer pre-populated with all 340 human
replays. The actor and critic stay at their initial weights — they're
trained per-game in Phase 4 starting from this WM checkpoint.

Phase-3 contract (see ``docs/phase-checklists.md``):

- All 340 replays load into ``embodied.replay`` (laptop tests use a
  tiny synthetic buffer; Vast runs the full set).
- WM-only updates enforced by ``WMOnlyAgent`` overriding
  ``Agent.loss`` + ``Agent.train`` and rebuilding ``self.opt`` over
  [dyn, enc, dec, rew, con] only — pol/val never see grads. The
  pretrain loop calls the inherited ``agent.train`` (which now
  IS the WM-only path).
- All four WM losses (recon, dyn, rew, con) trend down over an epoch.
- Checkpoint cadence ≥30 min; resume from preemption verified.
- RHAE held-out hook spikes near actual level-up boundaries.

Public surface — see ``tests/test_pretrain_wm.py`` for the binding
contract on each entry point. Heavy DV3 / JAX deps stay lazy (laptop
importability matches ``scripts/launch_pergame.py``).
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

ARC3_CONFIG_PATH = _REPO_ROOT / "configs" / "arc3.yaml"
DREAMERV3_CONFIG_PATH = _DV3 / "dreamerv3" / "configs.yaml"

DEFAULT_CONFIGS_LADDER = ("size12m", "arc3", "pretrain")
"""Layered on top of dreamerv3's ``defaults`` block. Order matters —
``pretrain`` is rightmost so its overrides win."""


_STUB_MSG = (
    "scripts.pretrain_wm: stub awaiting impl — see "
    "tests/test_pretrain_wm.py for the contract"
)


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Top-level CLI flags. Anything else is forwarded to elements.Flags
    via parse_known_args, same as scripts/launch_pergame.py."""
    p = argparse.ArgumentParser(
        prog="pretrain_wm.py",
        description="DreamerV3 cross-game World-Model pretraining on ARC-AGI-3 replays.",
    )
    p.add_argument(
        "--logdir",
        required=True,
        help="Run directory. Re-using an existing logdir resumes from the last checkpoint.",
    )
    p.add_argument(
        "--replays-root",
        required=True,
        help="Root directory containing per-game replay JSONLs "
             "(e.g. data/replays/). Walked recursively for *.recording.jsonl.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS_LADDER),
        help=f"Config blocks to layer on top of defaults "
             f"(default: {' '.join(DEFAULT_CONFIGS_LADDER)}).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    """Return (named_args, leftover_args). Leftovers go to elements.Flags."""
    parser = build_argparser()
    return parser.parse_known_args(argv)


# ----------------------------------------------------------------------
# Config resolution — mirrors scripts/launch_pergame.py
# ----------------------------------------------------------------------


def load_merged_configs() -> dict:
    """Read dreamerv3/configs.yaml + configs/arc3.yaml and return a single dict.

    The arc3.yaml file defines the ``arc3`` and ``pretrain`` blocks;
    block-name collisions with dreamerv3 raise. This function does NOT
    inject env-suite defaults — pretrain has no env, so the launcher's
    ``env.arc3`` injection is unnecessary here.
    """
    import ruamel.yaml as yaml

    parser = yaml.YAML(typ="safe")
    base = parser.load(DREAMERV3_CONFIG_PATH.read_text(encoding="utf-8"))
    arc3 = parser.load(ARC3_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    if "defaults" not in base:
        raise RuntimeError(
            f"{DREAMERV3_CONFIG_PATH} missing 'defaults' block — dreamerv3 changed?"
        )

    merged = dict(base)
    for name, block in arc3.items():
        if name in merged:
            raise RuntimeError(
                f"config block name collision: arc3.yaml redefines {name!r} from dreamerv3"
            )
        merged[name] = block
    return merged


def build_config(args: argparse.Namespace, leftover: Sequence[str]):
    """Build an elements.Config from argparse + leftover key=value flags.

    Phase-3 belt-and-braces: the merged config is expected to set
    ``script=pretrain_wm`` (via the ``pretrain`` block in arc3.yaml).
    A stray ``embodied.run.train`` invocation against this config
    would trip on the unknown script name.
    """
    import elements

    merged = load_merged_configs()
    config = elements.Config(merged["defaults"])
    for name in args.configs:
        if name == "defaults":
            continue
        if name not in merged:
            raise ValueError(
                f"unknown config block {name!r} (available: {sorted(merged)})"
            )
        config = config.update(merged[name])

    config = config.update(
        logdir=args.logdir,
        seed=args.seed,
    )

    if leftover:
        config = elements.Flags(config).parse(list(leftover))

    if "{timestamp}" in config.logdir:
        config = config.update(logdir=config.logdir.format(timestamp=elements.timestamp()))
    return config


# ----------------------------------------------------------------------
# Below: stubs for the remaining steps. Land in subsequent commits.
# ----------------------------------------------------------------------


def populate_buffer_from_replays(
    replay: Any,
    root: Path,
    *,
    stats: Optional[dict[str, Any]] = None,
) -> int:
    """Pre-populate ``replay`` from every JSONL under ``root``.

    Walks ``root`` via ``arc3_wm.replay_loader.load_replays_directory``
    and calls ``replay.add(step)`` for every step dict in every episode.
    Returns the total transition count.

    ``stats`` is updated in place (if provided):

    - ``per_game_counts``: ``dict[str, int]`` — transitions added per
      ``game_id`` (parent folder name). Phase-3 gate row 1 expects
      this distribution to be roughly even.
    - ``noise_rows_discarded``: int — threaded through from the loader's
      post-terminal-noise rule (Phase 1.7).

    Each replay-loader episode is added to the buffer as a single
    "worker" stream (DreamerV3's online buffer interprets episodes
    via the is_first / is_last flags already in the step dict, so
    worker numbering is purely for chunking; we use a monotonic
    counter so chunks are well-isolated).
    """
    from arc3_wm.replay_loader import load_replays_directory

    if stats is not None:
        stats.setdefault("per_game_counts", {})
        stats.setdefault("noise_rows_discarded", 0)

    n_total = 0
    worker = 0
    for game_id, episode in load_replays_directory(Path(root), stats=stats):
        for step in episode:
            replay.add(step, worker=worker)
        if stats is not None:
            counts = stats["per_game_counts"]
            counts[game_id] = counts.get(game_id, 0) + len(episode)
        n_total += len(episode)
        worker += 1
    return n_total


def make_wm_only_agent(config):
    """Build a ``WMOnlyAgent`` from the merged config.

    Phase 3 has no env; obs / act spaces are derived from the
    ``arc3_wm.embodied_env`` contract that the replay loader writes
    against. Mirrors the filtering pattern in ``dreamerv3.main.make_agent``
    (exclude the ``reset`` action key — pretrain has no driver).
    """
    import elements
    import numpy as np

    from arc3_wm.action_space import N_ACTIONS
    from arc3_wm.embodied_env import OBS_HW
    from arc3_wm.wm_only_agent import WMOnlyAgent

    obs_space = {
        "image": elements.Space(np.uint8, (OBS_HW, OBS_HW, 3), 0, 255),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act_space = {
        "action": elements.Space(np.int32, (), 0, N_ACTIONS),
    }

    return WMOnlyAgent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )


def _save_checkpoint(ckpt_dir: Path, agent: Any) -> None:
    """Atomic-rename pickle save of ``agent.save()`` under ``ckpt_dir``.

    Side-stepping ``elements.Checkpoint`` because its ``_cleanup`` step
    interacts badly with ``elements.Path.name`` on Windows — that
    attribute returns the full path instead of the basename, so the
    "exclude `latest` from cleanup candidates" filter fails and the
    just-created timestamp folder gets deleted on every save. This
    helper keeps the same contract (agent.save() → bytes → disk) but
    uses stdlib pickle + pathlib + os.replace for atomicity.
    """
    import pickle

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / "latest.pkl"
    tmp = ckpt_dir / "latest.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(agent.save(), f)
    import os as _os
    _os.replace(str(tmp), str(target))


def _load_checkpoint_if_exists(ckpt_dir: Path, agent: Any) -> bool:
    """Restore ``agent`` from ``ckpt_dir/latest.pkl`` if present.

    Returns True if a checkpoint was loaded; False if the directory has
    no checkpoint yet (initial run).
    """
    import pickle

    target = ckpt_dir / "latest.pkl"
    if not target.exists():
        return False
    with target.open("rb") as f:
        agent.load(pickle.load(f))
    return True


def pretrain_wm_loop(*, agent, replay, logger, args) -> None:
    """Custom run loop — sibling of ``embodied.run.train``, WM-only.

    Phase-3 contract (verified by tests/test_pretrain_wm.py):

    - Calls ``agent.train(carry, batch)``. With ``WMOnlyAgent`` plumbed
      in (option-(A): override ``Agent.loss`` + ``Agent.train`` rather
      than add a parallel ``wm_train``), this IS the WM-only path:
      pol/val are not in ``self.modules``, no imagination runs, no
      slow-critic update fires.
    - Never invokes ``agent.policy`` — no env, no Driver, no rollouts.
    - The single rebuilt ``self.opt`` (over [dyn, enc, dec, rew, con])
      is the only optimizer. Pol/val params still exist on the model
      but receive no gradients.
    - Writes checkpoints under ``logdir/ckpt/latest.pkl`` at
      ``args.save_every`` cadence; loads at entry for resume.
    - Logger receives the WM-loss dict every ``args.log_every`` ticks.
    - RHAE held-out hook is wired in step 5b — for now, only the loop
      mechanics are exercised.

    The loop is intentionally simple: each iteration does
    ``int(args.train_ratio)`` ``train`` updates and bumps an integer
    step counter by 1. Termination: ``step >= args.steps``.
    """
    import elements

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = logdir / "ckpt"

    # Cadence clocks. elements.when.Clock(every=0) returns True every call,
    # which is what the cadence test (save_every=0) relies on.
    should_log = elements.when.Clock(args.log_every)
    should_save = elements.when.Clock(args.save_every)

    # Buffer warm-up gate — should be a no-op on a pre-populated buffer.
    min_buffer = max(1, args.batch_size * args.batch_length)

    carry = agent.init_train(args.batch_size)

    # Resume from preemption: load existing checkpoint, otherwise lay
    # down an initial snapshot so the ckpt dir exists for downstream
    # cadence detection.
    if not _load_checkpoint_if_exists(ckpt_dir, agent):
        _save_checkpoint(ckpt_dir, agent)

    # Stream — yield batches from replay.sample. agent.stream() wraps
    # with internal preprocessing (Prefetch on Vast; pass-through on the
    # mock-backed laptop tests).
    def _replay_generator():
        while True:
            yield replay.sample(args.batch_size, "train")

    stream = iter(agent.stream(_replay_generator()))

    train_updates_per_iter = max(1, int(args.train_ratio))
    step = 0
    last_metrics: dict = {}

    while step < args.steps:
        if len(replay) < min_buffer:
            # Should never happen on the Phase-3 path (buffer is
            # pre-populated to ~180k transitions) but guard so tiny test
            # buffers don't deadlock.
            break

        for _ in range(train_updates_per_iter):
            batch = next(stream)
            carry, _outs, last_metrics = agent.train(carry, batch)

        step += 1

        if should_save(step):
            _save_checkpoint(ckpt_dir, agent)
        if should_log(step):
            try:
                logger.add(last_metrics, prefix="train")
            except Exception:  # noqa: BLE001 — mock-friendly: laptop tests pass MagicMock
                pass

    # Final checkpoint — guarantees a usable artifact even if save_every
    # never fired during a short run.
    _save_checkpoint(ckpt_dir, agent)


class RHAEHeldOutHook:
    """Periodic held-out-replay hook. See module docstring."""

    def __init__(self, *, holdout, every_n_steps: int) -> None:
        raise NotImplementedError(_STUB_MSG)

    def __call__(self, *, step: int, agent) -> Optional[dict[str, Any]]:
        raise NotImplementedError(_STUB_MSG)


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise NotImplementedError(_STUB_MSG)
