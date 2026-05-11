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
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# Make ``import embodied`` / ``import dreamerv3`` resolve our pinned source.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
if _DV3.is_dir() and str(_DV3) not in sys.path:
    sys.path.insert(0, str(_DV3))

ARC3_CONFIG_PATH = _REPO_ROOT / "configs" / "arc3.yaml"
DREAMERV3_CONFIG_PATH = _DV3 / "dreamerv3" / "configs.yaml"

# --- Phase-4 warm-start constants (see docs/phase4-warmstart-notes.md) ----

WM_REGEX = r'^(?:dyn|enc|dec|rew|con)/'
"""Anchored prefix-match for the 5 World-Model modules. Loaded keys
filtered to these prefixes; opt/state/... and any future pol/val keys
are excluded. Validated against Phase-3 v1 ckpt key layout."""

WM_KEY_COUNT = 68
"""Number of param keys (not elements) the Phase-3 v1 ckpt has under
the WM prefixes. Used as a fail-loud invariant in seed_wm_from_ckpt."""

WM_PARAM_COUNT = 9_898_179
"""Sum of param elements across the 68 WM keys. Same fail-loud purpose."""


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
    p.add_argument(
        "--init-from-ckpt",
        default="",
        dest="init_from_ckpt",
        help=(
            "Optional Phase-3 WM checkpoint to seed the Phase-4 agent. "
            "Accepts a local path, b2://bucket/key URL, or https:// URL. "
            "Module params under (dyn|enc|dec|rew|con) are loaded; "
            "optimizer state and counters are reset. See "
            "docs/phase4-warmstart-notes.md."
        ),
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
# Warm-start helpers (Phase-4 --init-from-ckpt path)
# ----------------------------------------------------------------------


def _is_remote_url(path: str) -> bool:
    """True iff ``path`` looks like a remote URL we know how to fetch."""
    return path.startswith("b2://") or path.startswith("https://") or path.startswith("http://")


def _download_to_cache(url: str, cache_dir: Path) -> Path:
    """Fetch ``url`` to ``cache_dir`` and return the local file path.

    Shells out to ``b2 file download`` for b2:// URLs (auth via the
    ``b2`` CLI's configured account) and ``curl`` for http(s)://. The
    target filename is the URL's basename; existing files are reused
    (idempotent — re-running the launcher doesn't re-download).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1] or "ckpt.pkl"
    target = cache_dir / name
    if target.exists():
        return target

    if url.startswith("b2://"):
        # b2://bucket/key... → b2 file download b2://bucket/key... <target>
        cmd = ["b2", "file", "download", url, str(target)]
    else:
        cmd = ["curl", "-fSL", url, "-o", str(target)]
    subprocess.run(cmd, check=True)
    return target


def _resolve_init_ckpt_path(path: str, cache_dir: Optional[Path] = None) -> Path:
    """Resolve ``path`` (local path or remote URL) to a local file path.

    For remote URLs the file is downloaded into ``cache_dir``; the
    default cache dir is ``<repo>/checkpoints/_init_cache/``.
    """
    if not path:
        raise ValueError("init-from-ckpt path must be non-empty")
    if _is_remote_url(path):
        cache_dir = cache_dir if cache_dir is not None else _REPO_ROOT / "checkpoints" / "_init_cache"
        return _download_to_cache(path, cache_dir)
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"--init-from-ckpt local path does not exist or is not a file: {p}"
        )
    return p


def _check_no_double_load(init_from_ckpt: str, from_checkpoint: str) -> None:
    """Reject the combination of --init-from-ckpt and --run.from_checkpoint.

    These are two warm-start mechanisms with different semantics
    (regex-filtered WM-only vs full agent-state resume). Setting both
    is almost always a bug.
    """
    if init_from_ckpt and from_checkpoint:
        raise ValueError(
            "--init-from-ckpt is incompatible with --run.from_checkpoint; "
            "set only one. See docs/phase4-warmstart-notes.md."
        )


def seed_wm_from_ckpt(agent: Any, ckpt_path: Path) -> dict[str, Any]:
    """Restore Phase-3 WM module weights into ``agent``, reset counters.

    Steps:

    1. ``pickle.load(open(ckpt_path, 'rb'))`` — yields ``{'params': ..., 'counters': ...}``.
    2. Validate top-level shape (raises if either key missing).
    3. Mutate ``state['counters']`` to all-zero (Phase-4 starts fresh).
    4. Validate the WM regex matches exactly ``WM_KEY_COUNT`` keys and
       ``WM_PARAM_COUNT`` param elements (fail-loud on regex drift /
       checkpoint format change).
    5. Call ``agent.load(state, regex=WM_REGEX)``.

    Returns a diagnostics dict (matched_keys, matched_params,
    counter_values_before_reset) for caller-side logging.
    """
    import pickle
    import re

    import numpy as np

    state = pickle.load(open(ckpt_path, "rb"))

    if "params" not in state or "counters" not in state:
        raise ValueError(
            f"unexpected checkpoint shape at {ckpt_path}: "
            f"keys={sorted(state) if isinstance(state, dict) else type(state).__name__}; "
            f"expected top-level dict with 'params' and 'counters'."
        )
    for ckey in ("updates", "batches", "actions"):
        if ckey not in state["counters"]:
            raise ValueError(
                f"missing counter {ckey!r} in checkpoint at {ckpt_path}: "
                f"counters={state['counters']}"
            )

    original_counters = dict(state["counters"])
    state["counters"] = {"updates": 0, "batches": 0, "actions": 0}

    params = state["params"]
    matched = {k: v for k, v in params.items() if re.match(WM_REGEX, k)}
    matched_keys = len(matched)
    matched_params = sum(
        int(np.prod(getattr(v, "shape", (1,)))) if hasattr(v, "shape") else 1
        for v in matched.values()
    )

    if matched_keys == 0:
        raise ValueError(
            f"WM regex {WM_REGEX!r} matched zero keys in checkpoint at {ckpt_path}; "
            f"ckpt top-level prefixes: {sorted({k.split('/')[0] for k in params})}. "
            f"Probable cause: checkpoint key schema changed; update WM_REGEX or "
            f"investigate the save path."
        )
    if matched_keys != WM_KEY_COUNT:
        raise ValueError(
            f"WM regex matched {matched_keys} keys; expected exactly {WM_KEY_COUNT}. "
            f"Indicates silent partial load or schema drift. "
            f"See docs/phase4-warmstart-notes.md for the expected layout."
        )
    if matched_params != WM_PARAM_COUNT:
        raise ValueError(
            f"WM matched-key params sum to {matched_params:,}; expected exactly "
            f"{WM_PARAM_COUNT:,}. Shape mismatch in ckpt or stale constants."
        )

    agent.load(state, regex=WM_REGEX)

    # Live-agent counter assertion — verifies agent.load() actually
    # applied the reset (defends against an agent.load() refactor that
    # silently re-derives counters from elsewhere). Reads through the
    # real Counter objects' .value attribute; tests must set these to
    # mirror real agent.load semantics on the mock.
    live_counters = {
        "updates": int(agent.n_updates.value),
        "batches": int(agent.n_batches.value),
        "actions": int(agent.n_actions.value),
    }
    if any(v != 0 for v in live_counters.values()):
        raise RuntimeError(
            f"post-load counter reset failed: live counters = {live_counters}; "
            f"expected all zero. agent.load() did not honour the reset state "
            f"OR the Counter API changed."
        )

    return {
        "matched_keys": matched_keys,
        "matched_params": matched_params,
        "counter_values_before_reset": original_counters,
        "live_counters_after_load": live_counters,
        "ckpt_path": str(ckpt_path),
    }


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

    # Reject the dual-warm-start footgun before any heavy construction.
    _check_no_double_load(args.init_from_ckpt, str(config.run.from_checkpoint))

    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(str(logdir / "config.yaml"))

    print(f"Replica: {config.replica} / {config.replicas}")
    print(f"Logdir:  {logdir}")
    print(f"Task:    {config.task}")
    print(f"Script:  {config.script}")
    if args.init_from_ckpt:
        print(f"Init-from-ckpt: {args.init_from_ckpt}")

    # Structural check on the env we'll be training in.
    sample_env = make_env(config, 0)
    try:
        vast_only_isinstance_check(sample_env)
    finally:
        sample_env.close()

    # Resolve the init-ckpt path up-front so download failures surface
    # before we boot JAX. Then wrap make_agent so the closure handed to
    # embodied.run.train applies the warm-start post-construction.
    init_ckpt_path: Optional[Path] = None
    if args.init_from_ckpt:
        init_ckpt_path = _resolve_init_ckpt_path(args.init_from_ckpt)
        print(f"Init-ckpt resolved to: {init_ckpt_path}")

    def make_agent_with_seed():
        agent = make_agent(config)
        if init_ckpt_path is not None:
            diag = seed_wm_from_ckpt(agent, init_ckpt_path)
            print(
                f"WM seeded: matched_keys={diag['matched_keys']} "
                f"matched_params={diag['matched_params']:,} "
                f"counters_before_reset={diag['counter_values_before_reset']} "
                f"live_counters_after_load={diag['live_counters_after_load']}"
            )
        return agent

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
            make_agent_with_seed,
            bind(make_replay, config, "replay"),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            run_args,
        )
    elif config.script == "train_eval":
        embodied.run.train_eval(
            make_agent_with_seed,
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
            make_agent_with_seed,
            bind(make_env, config),
            bind(make_logger, config),
            run_args,
        )
    else:
        raise NotImplementedError(f"--script {config.script}")


if __name__ == "__main__":
    main()
