"""Phase 3 pretrain smoke — Vast-only.

Gates step 5b. Validates that the WM-only path actually trains: a real
``WMOnlyAgent`` (subclass of dreamerv3.agent.Agent) on a tiny synthetic
buffer for ~1-2k steps, emitting ``train/loss/{dyn,rew,con,image}`` to
a JSONL trail Haso can eyeball before the RHAEHeldOutHook wires in.

This is NOT a CPU/laptop script. dreamerv3 ships JAX-CUDA12 only;
attempting to run on CPU either fails at import or trains so slowly
that the smoke produces no useful curves. Structural correctness
(custom run loop calls wm_train, never train; spy on opt/wm_opt;
loss-tree shape) is already covered by ``tests/test_pretrain_wm.py``
on the laptop with a mock agent.

What this script intentionally does NOT do:

- Pre-populate from the real 340-replay set. The smoke uses ~50
  fabricated step dicts so JAX compilation dominates wall-clock and
  curves stabilise within ~minutes. The full-buffer run is the actual
  Phase-3 pretrain (scripts/pretrain_wm.py main).
- Wire in the RHAE held-out hook. That's step 5b — the gate this
  smoke informs.
- Save anything outside ``--logdir/scope/*.jsonl`` (and the
  per-step-cadence checkpoints under ``--logdir/ckpt/``). The smoke
  is throwaway; clean up the logdir afterwards.

Launch (Vast, after pulling repo + B2 replays):

    python scripts/pretrain_wm_smoke.py \\
        --logdir ~/logdir/pretrain-smoke-{timestamp} \\
        --steps 1500

Eyeball loss curves with the dreamerv3-shipped Scope viewer:

    python -m scope.viewer --basedir ~/logdir

Or tail JSONL directly:

    tail -f ~/logdir/pretrain-smoke-*/scope/metrics.jsonl

Pass criteria (informal, qualitative):
- All four WM losses (loss/dyn, loss/rew, loss/con, loss/image)
  emit non-NaN values from step ~10 onward.
- loss/image trends down monotonically (or near-so) over the run.
- loss/dyn / loss/rew / loss/con do not explode.
- No assertion failures, no JAX shape mismatches, no replay-buffer
  drain warnings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Match the discipline of scripts/launch_pergame.py + scripts/pretrain_wm.py:
# heavy stack on third_party/dreamerv3, never imported at module top.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
if _DV3.is_dir() and str(_DV3) not in sys.path:
    sys.path.insert(0, str(_DV3))


N_SYNTHETIC_TRANSITIONS = 2000
"""Just past the warmup gate. The pretrain loop guards on
``len(replay) >= batch_size * batch_length`` (1024 with size12m
defaults), and embodied's Replay counts sample-able batch_length
windows rather than raw steps — so an undersized buffer reports
len()==0 and the loop breaks before any wm_train fires. 2000 gives
~30 windows' worth, still small enough that JAX compilation
dominates wall-clock."""


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pretrain_wm_smoke.py",
        description="Vast-only smoke for Phase-3 WM-only pretraining.",
    )
    p.add_argument("--logdir", required=True)
    p.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Outer-loop step count (default: 1500).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def _fabricate_buffer(replay, n_transitions: int, batch_length: int) -> int:
    """Lay down ``n_transitions`` synthetic step dicts in ``replay``.

    The buffer schema mirrors what arc3_wm.replay_loader emits — the
    real loader is faster to bypass for the smoke, since the structural
    correctness of replay loading is already covered by
    ``tests/test_replay_loader.py``.
    """
    import numpy as np

    from arc3_wm.action_space import N_ACTIONS
    from arc3_wm.embodied_env import OBS_HW

    rng = np.random.default_rng(0)
    # One fake "episode": is_first only on step 0; is_last + is_terminal
    # on the very last step. Reward is 0 except a single +1 spike at
    # mid-episode so the reward head sees signal.
    for i in range(n_transitions):
        step = {
            "image": rng.integers(0, 256, (OBS_HW, OBS_HW, 3), dtype=np.uint8),
            "action": np.int32(rng.integers(0, N_ACTIONS)),
            "reward": np.float32(1.0 if i == n_transitions // 2 else 0.0),
            "is_first": np.bool_(i == 0),
            "is_last": np.bool_(i == n_transitions - 1),
            "is_terminal": np.bool_(i == n_transitions - 1),
        }
        replay.add(step, worker=0)
    return n_transitions


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    # Lazy heavy imports — laptop-importable for argparse-only checks.
    import elements
    import embodied

    import scripts.pretrain_wm as P
    from dreamerv3.main import make_logger, make_replay, make_stream  # noqa: F401

    # Assemble a complete merged config in the same shape pretrain_wm.main
    # would (size12m + arc3 + pretrain), then override --run.steps so the
    # smoke actually terminates.
    parsed_args, leftover = P.parse_args([
        "--logdir", args.logdir,
        # Smoke doesn't read replays — point at a non-existent path the
        # populate_buffer code path never runs against.
        "--replays-root", "/dev/null",
        "--seed", str(args.seed),
        "--run.steps", str(args.steps),
        # Force checkpoint cadence to fire near end of run, not every
        # 30 min — the smoke is < 30 min total.
        "--run.save_every", "60",
        "--run.log_every", "5",
    ])
    config = P.build_config(parsed_args, leftover)

    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(str(logdir / "config.yaml"))
    print(f"Smoke logdir: {logdir}")
    print(f"Steps:       {args.steps}")
    print(f"Buffer:      {N_SYNTHETIC_TRANSITIONS} synthetic transitions")

    # Build the real replay buffer (same factory dreamerv3.main uses) so
    # the smoke exercises the production sample / stream code path.
    replay = make_replay(config, "replay")
    n = _fabricate_buffer(replay, N_SYNTHETIC_TRANSITIONS, config.batch_length)
    print(f"Pre-populated buffer: {n} transitions, len(replay)={len(replay)}")

    # The real WMOnlyAgent — subclass of dreamerv3.agent.Agent.
    agent = P.make_wm_only_agent(config)
    print("WMOnlyAgent constructed; entering pretrain_wm_loop.")

    logger = make_logger(config)
    run_args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    P.pretrain_wm_loop(agent=agent, replay=replay, logger=logger, args=run_args)
    logger.close()
    print(f"Smoke complete. Loss JSONL under: {logdir}/scope/")


if __name__ == "__main__":
    main()
