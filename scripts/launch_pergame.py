"""DreamerV3 launcher with arc3_<game> task support.

Per design-decisions.md D12, this bypasses ``dreamerv3/main.py`` rather than
forking it. The launcher reuses dreamerv3's ``configs.yaml``, agent, replay,
stream, and logger machinery; only ``make_env`` is overridden to dispatch
``arc3_<game>`` tasks to ``ARC3EmbodiedEnv``.

CLI:
    python scripts/launch_pergame.py \\
        --logdir ~/logdir/{timestamp} \\
        --configs size12m arc3 \\
        --task arc3_vc33 \\
        --seed 0

``--task crafter_reward`` is supported for milestone (2) (Crafter sanity);
it falls through to the existing ``embodied.envs.crafter:Crafter``.

Resume from preemption: re-run with the same ``--logdir``. ``embodied.run.train``
auto-loads the latest checkpoint when ``logdir`` already contains one.

W&B: enabled iff ``WANDB_PROJECT`` is set in the environment. Otherwise
JSONL + Scope outputs only (laptop-friendly default).

Imports of the heavy DreamerV3 / JAX / portal stack are deferred to inside
helper functions so this module is importable on a laptop without JAX
(useful for argparse tests). The actual ``embodied.run.train(...)`` call
happens only inside ``main()`` and only when ``--script train`` is set
(default). Tests never invoke ``main`` end-to-end on the laptop.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# Make ``import embodied`` / ``import dreamerv3`` resolve our pinned source.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
if _DV3.is_dir() and str(_DV3) not in sys.path:
    sys.path.insert(0, str(_DV3))

ARC3_CONFIG_PATH = _REPO_ROOT / "configs" / "arc3.yaml"
DREAMERV3_CONFIG_PATH = _DV3 / "dreamerv3" / "configs.yaml"


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Top-level CLI flags. Anything else is forwarded to elements.Flags."""
    p = argparse.ArgumentParser(
        prog="launch_pergame.py",
        description="DreamerV3 launcher with arc3_<game> task support.",
    )
    p.add_argument(
        "--logdir",
        required=True,
        help="Run directory. Re-using an existing logdir resumes from the last checkpoint.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["defaults"],
        help="Config blocks to layer on top of defaults (e.g. 'size12m arc3').",
    )
    p.add_argument(
        "--task",
        required=True,
        help="Task identifier. arc3_<game> dispatches to ARC3EmbodiedEnv; "
             "anything else falls through to dreamerv3 (e.g. 'crafter_reward').",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--script",
        default="train",
        choices=["train", "train_eval", "eval_only"],
        help="Which embodied.run script to call (default: train).",
    )
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    """Return (named_args, leftover_args). Leftovers go to elements.Flags."""
    parser = build_argparser()
    return parser.parse_known_args(argv)


# ----------------------------------------------------------------------
# Config resolution
# ----------------------------------------------------------------------

DEFAULT_ARC3_ENV = {"max_steps": 1000, "use_seed": True}
"""Per-suite defaults injected into ``defaults.env.arc3`` so ``elements.Config``
can later honour ``--env.arc3.max_steps`` overrides. ``elements.Config.update``
is strict about new keys; introducing ``env.arc3.*`` purely via the named
``arc3:`` block fails. Inject at the defaults level to keep everything
typed consistently with how dreamerv3 handles atari/crafter/dmc/etc."""


def load_merged_configs() -> dict:
    """Read dreamerv3/configs.yaml + configs/arc3.yaml and return a single dict.

    Injects an ``env.arc3`` sub-block into defaults so per-suite overrides are
    legal under ``elements.Config``'s strict-update semantics.
    """
    import ruamel.yaml as yaml

    parser = yaml.YAML(typ="safe")
    base = parser.load(DREAMERV3_CONFIG_PATH.read_text(encoding="utf-8"))
    arc3 = parser.load(ARC3_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    if "defaults" not in base:
        raise RuntimeError(
            f"{DREAMERV3_CONFIG_PATH} missing 'defaults' block — dreamerv3 changed?"
        )

    # Inject per-suite defaults into the dreamerv3 'defaults' block.
    defaults_env = base["defaults"].setdefault("env", {})
    defaults_env.setdefault("arc3", {}).update(DEFAULT_ARC3_ENV)

    merged = dict(base)
    for name, block in arc3.items():
        if name in merged:
            raise RuntimeError(
                f"config block name collision: arc3.yaml redefines {name!r} from dreamerv3"
            )
        merged[name] = block
    return merged


def build_config(args: argparse.Namespace, leftover: Sequence[str]):
    """Build an elements.Config from argparse + leftover key=value flags."""
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
        task=args.task,
        seed=args.seed,
        script=args.script,
    )

    if leftover:
        config = elements.Flags(config).parse(list(leftover))

    if "{timestamp}" in config.logdir:
        config = config.update(logdir=config.logdir.format(timestamp=elements.timestamp()))
    return config


# ----------------------------------------------------------------------
# make_env / make_agent / make_replay / make_stream / make_logger
# ----------------------------------------------------------------------

def make_env(config, index: int, **overrides):
    """Build an embodied.Env. arc3_<game> tasks dispatch to ARC3EmbodiedEnv."""
    if str(config.task).startswith("arc3_"):
        from arc3_wm.embodied_env import ARC3EmbodiedEnv

        game_id = config.task.split("_", 1)[1]
        arc3_cfg = config.env.get("arc3", {})
        max_steps = int(arc3_cfg.get("max_steps", 1000))
        use_seed = bool(arc3_cfg.get("use_seed", True))
        seed = (int(config.seed) + int(index)) if use_seed else 0

        env = ARC3EmbodiedEnv(game_id=game_id, seed=seed, max_steps=max_steps)
        return _wrap_env(env, config)

    # Fall through to dreamerv3's builtin dispatch.
    from dreamerv3.main import make_env as _dv3_make_env

    return _dv3_make_env(config, index, **overrides)


def _wrap_env(env, config):
    """Mirror dreamerv3.main.wrap_env (action normalisation + dtype unification)."""
    import embodied

    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_agent(config):
    import elements
    import embodied

    env = make_env(config, 0)
    notlog = lambda k: not k.startswith("log/")  # noqa: E731 — matches dreamerv3 style
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()

    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)

    from dreamerv3.agent import Agent

    return Agent(
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


def make_replay(config, folder: str = "replay", mode: str = "train"):
    from dreamerv3.main import make_replay as _f

    return _f(config, folder, mode)


def make_stream(config, replay, mode: str):
    from dreamerv3.main import make_stream as _f

    return _f(config, replay, mode)


def make_logger(config):
    """Auto-add 'wandb' output when WANDB_PROJECT is set in env."""
    if os.environ.get("WANDB_PROJECT") and "wandb" not in config.logger.outputs:
        config = config.update(logger={"outputs": list(config.logger.outputs) + ["wandb"]})
    from dreamerv3.main import make_logger as _f

    return _f(config)


# ----------------------------------------------------------------------
# Vast-only safety check (laptop = silent)
# ----------------------------------------------------------------------

def vast_only_isinstance_check(env) -> None:
    """If embodied.core.base is importable, assert env satisfies embodied.Env.

    Catches a future upstream tightening of the duck-type contract our
    ARC3EmbodiedEnv relies on. Silent no-op on the laptop where embodied's
    transitive deps (portal, JAX) aren't installed.
    """
    try:
        from embodied.core.base import Env as _EmbodiedEnv
    except Exception:  # noqa: BLE001 — laptop import path is intentionally tolerant
        return
    has_iface = hasattr(env, "obs_space") and hasattr(env, "act_space") and hasattr(env, "step")
    assert isinstance(env, _EmbodiedEnv) or has_iface, (
        "env does not satisfy embodied.Env (neither isinstance nor duck-type)"
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    from functools import partial as bind

    import elements
    import embodied

    args, leftover = parse_args(argv)
    config = build_config(args, leftover)

    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(str(logdir / "config.yaml"))

    print(f"Replica: {config.replica} / {config.replicas}")
    print(f"Logdir:  {logdir}")
    print(f"Task:    {config.task}")
    print(f"Script:  {config.script}")

    # Structural check on the env we'll be training in.
    sample_env = make_env(config, 0)
    try:
        vast_only_isinstance_check(sample_env)
    finally:
        sample_env.close()

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

    if config.script == "train":
        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            run_args,
        )
    elif config.script == "train_eval":
        embodied.run.train_eval(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "eval_replay", "eval"),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            run_args,
        )
    elif config.script == "eval_only":
        embodied.run.eval_only(
            bind(make_agent, config),
            bind(make_env, config),
            bind(make_logger, config),
            run_args,
        )
    else:
        raise NotImplementedError(f"--script {config.script}")


if __name__ == "__main__":
    main()
