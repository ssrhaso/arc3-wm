"""scripts/pretrain_wm.py — Phase 3 cross-game WM pretraining.

Sibling of ``scripts/launch_pergame.py``. The launcher trains
end-to-end on a single arc3_<game> task; this script pretrains the
World Model only, on a buffer pre-populated with all 340 human
replays. The actor and critic stay at their initial weights — they're
trained per-game in Phase 4 starting from this WM checkpoint.

Phase-3 contract (see ``docs/phase-checklists.md``):

- All 340 replays load into ``embodied.replay`` (laptop tests use a
  tiny synthetic buffer; Vast runs the full set).
- WM-only updates verified by code inspection — ``WMOnlyAgent.wm_train``
  uses a separate optimizer over [enc, dyn, dec, rew, con] only.
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
