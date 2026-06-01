"""Dry-run wiring test for scripts/launch_pergame.py.

Builds env + agent + replay + driver objects without invoking any training.
Skips on hosts without the DreamerV3 stack (laptop). On Vast, this is the
last laptop-style sanity check before the first ``embodied.run.train(...)``
call.

Note on skipping vs xfail: we use ``pytest.importorskip`` so:
  - laptop with no JAX: test SKIPS (suite stays green).
  - Vast with full deps: test RUNS and must pass (catches wiring errors
    before any GPU is allocated).

The launcher's ``main()`` is NOT invoked here. We call the make_*
helpers directly and let object construction surface ImportError /
shape errors early.
"""
from __future__ import annotations

import pytest

# Skip the entire module on hosts where the DreamerV3 deps aren't installable.
pytest.importorskip("jax", reason="dreamerv3 dry-run requires JAX")
pytest.importorskip("portal", reason="dreamerv3 dry-run requires portal")
pytest.importorskip("embodied", reason="dreamerv3 stack not on sys.path")
pytest.importorskip("dreamerv3", reason="dreamerv3 not importable")

import scripts.launch_pergame as L  # noqa: E402


@pytest.fixture
def arc3_config(tmp_path):
    """A complete config object for arc3_vc33, logdir under tmp_path."""
    args, leftover = L.parse_args(
        [
            "--logdir", str(tmp_path / "run"),
            "--task", "arc3_vc33",
            "--configs", "size12m", "arc3",
            "--seed", "0",
        ]
    )
    return L.build_config(args, leftover)


def test_make_env_arc3_dispatch(arc3_config):
    env = L.make_env(arc3_config, 0)
    try:
        assert hasattr(env, "obs_space")
        assert hasattr(env, "act_space")
        assert "image" in env.obs_space
        assert "action" in env.act_space
    finally:
        env.close()


def test_make_replay(arc3_config):
    replay = L.make_replay(arc3_config, "replay")
    assert replay is not None


def test_make_stream(arc3_config):
    replay = L.make_replay(arc3_config, "replay")
    stream = L.make_stream(arc3_config, replay, "train")
    assert stream is not None


def test_make_logger_no_wandb_without_env_var(arc3_config, monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    logger = L.make_logger(arc3_config)
    assert logger is not None


def test_make_logger_adds_wandb_when_env_var_set(arc3_config, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "arc3-wm-sprint")
    # We don't fully construct a WandB session here (would need a wandb
    # account). Just verify the config is mutated to request wandb output.
    if "wandb" in arc3_config.logger.outputs:
        pytest.skip("wandb already in default outputs; can't observe the mutation")
    captured = {}
    import dreamerv3.main as _main
    real = _main.make_logger
    def _capture(cfg):
        captured["outputs"] = list(cfg.logger.outputs)
        return real(cfg)
    monkeypatch.setattr(_main, "make_logger", _capture)
    L.make_logger(arc3_config)
    assert "wandb" in captured["outputs"]


def test_make_agent_random(arc3_config):
    """RandomAgent path doesn't need JAX param init; cheap dry-run wiring check."""
    cfg = arc3_config.update(random_agent=True)
    agent = L.make_agent(cfg)
    assert agent is not None


def test_vast_only_isinstance_check_passes_real_env(arc3_config):
    """Our duck-typed ARC3EmbodiedEnv must satisfy the embodied.Env contract."""
    env = L.make_env(arc3_config, 0)
    try:
        L.vast_only_isinstance_check(env)  # raises if not duck-compat
    finally:
        env.close()


def test_make_env_crafter_passthrough(tmp_path):
    """`crafter_reward` falls through to embodied's existing wrapper."""
    args, leftover = L.parse_args(
        [
            "--logdir", str(tmp_path / "run"),
            "--task", "crafter_reward",
            "--configs", "crafter", "size12m",
        ]
    )
    config = L.build_config(args, leftover)
    env = L.make_env(config, 0)
    try:
        assert hasattr(env, "obs_space")
        assert "image" in env.obs_space
    finally:
        env.close()
