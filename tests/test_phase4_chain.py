"""Phase-4 wired-chain integration test.

Exercises every laptop-verifiable interface in the warm-start →
eval-sink → post-hoc RHAE chain against REAL artifacts:

  - real Phase-3 pkl at checkpoints/pretrained-wm/v1/latest.pkl
  - real ARC3EmbodiedEnv on a cached game in environment_files/
  - real EvalRewardSink writing real reward streams
  - real scripts/compute_rhae.py invoked as a subprocess
  - real data/human_baselines.json

Parameterized over PILOT_GAMES. The list grows per commit: vc33 first,
then sb26, then cd82. Each commit lands one game independently so any
per-game regression has a clean blame target.

What this test does NOT cover: the embodied.run.train_eval driver
itself binding the make_env_eval_with_sink closure and the
make_agent_with_seed closure. That binding needs JAX which isn't on
the laptop. tests/test_launcher_dry_run.py covers it on Vast/5070s.

Skips:
  - Phase-3 pkl absent (fresh clone): test_seed_* skips.
  - environment_files/<game>/ absent: that game's parametrize entries
    skip individually.

Why this exists: derisks the wired chain BEFORE Phase-4 launches. If
this goes red, the corresponding bug would otherwise only surface
at GPU-spend time. This is what the Vast dry-run was buying us minus
the cost.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from arc3_wm import action_space as A
from arc3_wm.embodied_env import ARC3EmbodiedEnv
from arc3_wm.eval_reward_sink import EvalRewardSink

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PHASE3_CKPT = _REPO_ROOT / "checkpoints" / "pretrained-wm" / "v1" / "latest.pkl"
_HUMAN_BASELINES = _REPO_ROOT / "data" / "human_baselines.json"
_ENV_FILES = _REPO_ROOT / "environment_files"

# Pilot composition (post-2026-05-12 swap): vc33, sb26, cd82.
PILOT_GAMES = ["vc33", "sb26", "cd82"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_with_realistic_load() -> mock.MagicMock:
    """Mock agent whose load() zeros counters live, matching real DV3
    agent semantics. Mirrors tests/test_launcher_warmstart.py's helper
    so the laptop chain test doesn't depend on JAX."""
    agent = mock.MagicMock()
    agent.n_updates.value = -1
    agent.n_batches.value = -1
    agent.n_actions.value = -1

    def fake_load(state, regex=None):
        agent.n_updates.value = int(state["counters"]["updates"])
        agent.n_batches.value = int(state["counters"]["updates"])
        agent.n_actions.value = int(state["counters"]["actions"])

    agent.load.side_effect = fake_load
    return agent


def _drive_eval_episodes(
    sink: EvalRewardSink,
    inner_env: ARC3EmbodiedEnv,
    n_steps: int,
    rng: np.random.Generator,
) -> None:
    """Step the sink-wrapped env n_steps times with a mask-aware random
    policy. Auto-resets after is_last. max_steps on the inner env
    guarantees at least one truncation within 200 steps."""
    obs = sink.step({"action": 0, "reset": True})
    for _ in range(n_steps):
        if obs.get("is_last"):
            obs = sink.step({"action": 0, "reset": True})
            continue
        mask = inner_env.info["action_mask"]
        valid = np.flatnonzero(mask)
        assert valid.size > 0, "mask collapsed mid-episode"
        choice = int(rng.choice(valid))
        obs = sink.step({"action": choice, "reset": False})


# ---------------------------------------------------------------------------
# Ckpt resolve + seed (game-agnostic; needs Phase-3 pkl only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PHASE3_CKPT.exists(),
    reason="Phase-3 pkl not local (clone may be fresh)",
)
def test_resolve_local_ckpt_path_idempotent():
    """A local path resolves to itself, no download invoked."""
    import scripts.launch_pergame as L

    resolved = L._resolve_init_ckpt_path(str(_PHASE3_CKPT))
    assert resolved == _PHASE3_CKPT


@pytest.mark.skipif(
    not _PHASE3_CKPT.exists(),
    reason="Phase-3 pkl not local",
)
def test_seed_wm_from_real_phase3_ckpt():
    """seed_wm_from_ckpt against the real pkl + mock agent. Verifies
    every fail-loud invariant in the helper passes against the
    production-shape state dict."""
    import scripts.launch_pergame as L

    agent = _agent_with_realistic_load()
    diag = L.seed_wm_from_ckpt(agent, _PHASE3_CKPT)
    assert diag["matched_keys"] == L.WM_KEY_COUNT == 68
    assert diag["matched_params"] == L.WM_PARAM_COUNT == 9_898_179
    assert diag["counter_values_before_reset"] == {
        "updates": 192_000,
        "batches": 192_001,
        "actions": 0,
    }
    assert diag["live_counters_after_load"] == {
        "updates": 0,
        "batches": 0,
        "actions": 0,
    }
    agent.load.assert_called_once()
    _, kwargs = agent.load.call_args
    assert kwargs.get("regex") == L.WM_REGEX


# ---------------------------------------------------------------------------
# Per-game env + mask + sink + RHAE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game_id", PILOT_GAMES)
def test_env_files_cached(game_id):
    """environment_files/<game_id>/ must be cached locally.

    If this fails, add game_id to scripts/cache_env_files.py's GAMES
    list and run it (requires ARC_API_KEY + NORMAL mode briefly).
    """
    assert (_ENV_FILES / game_id).is_dir(), (
        f"environment_files/{game_id}/ not cached"
    )


@pytest.mark.parametrize("game_id", PILOT_GAMES)
def test_action_mask_non_degenerate_at_reset(game_id):
    """At reset, the env's action_mask is a length-N_ACTIONS bool array
    with at least one valid action. Load-bearing for the launcher's
    random-action sampling at run start."""
    if not (_ENV_FILES / game_id).is_dir():
        pytest.skip(f"environment_files/{game_id}/ not cached")

    env = ARC3EmbodiedEnv(game_id=game_id, seed=0, max_steps=200)
    try:
        obs = env.step({"action": 0, "reset": True})
        assert bool(obs["is_first"])
        mask = env.info["action_mask"]
        assert mask.shape == (A.N_ACTIONS,), (
            f"mask shape {mask.shape}, expected ({A.N_ACTIONS},)"
        )
        assert mask.dtype == np.bool_, f"mask dtype {mask.dtype}, expected bool"
        assert int(mask.sum()) > 0, (
            f"{game_id} reset mask has zero valid actions; "
            f"available_actions={env.info.get('available_actions')}"
        )
    finally:
        env.close()


@pytest.mark.parametrize("game_id", PILOT_GAMES)
def test_eval_sink_writes_real_rewards_jsonl(game_id, tmp_path):
    """Real env wrapped in EvalRewardSink, stepped 200 times with a
    mask-aware random policy. At least one JSONL line must land with
    the schema compute_rhae.load_episodes_from_jsonl expects."""
    if not (_ENV_FILES / game_id).is_dir():
        pytest.skip(f"environment_files/{game_id}/ not cached")

    sink_path = tmp_path / "eval_episodes.jsonl"
    env = ARC3EmbodiedEnv(game_id=game_id, seed=0, max_steps=200)
    sink = EvalRewardSink(env, sink_path=sink_path)
    rng = np.random.default_rng(0)
    try:
        _drive_eval_episodes(sink, env, n_steps=200, rng=rng)
    finally:
        env.close()

    assert sink_path.exists(), "sink JSONL not written"
    lines = sink_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1, (
        f"no eval-episode lines flushed in 200 steps "
        f"(max_steps=200 should force at least one truncation)"
    )
    for i, raw in enumerate(lines, start=1):
        parsed = json.loads(raw)
        assert "rewards" in parsed, f"line {i} missing 'rewards' key"
        assert isinstance(parsed["rewards"], list), f"line {i} 'rewards' not a list"
        assert all(isinstance(r, (int, float)) for r in parsed["rewards"]), (
            f"line {i} contains non-numeric rewards"
        )


@pytest.mark.parametrize("game_id", PILOT_GAMES)
def test_compute_rhae_cli_consumes_real_sink_output(game_id, tmp_path):
    """End-to-end: real env → real sink → real compute_rhae CLI.

    A random-action policy on 200 steps almost certainly clears zero
    levels. The assertion is "CLI ran cleanly and emitted a numeric
    non-NaN per_game_rhae" — proves the format contract holds end-to-
    end. Future training runs that DO clear levels still satisfy this
    test; it only asserts numeric-ness.
    """
    if not (_ENV_FILES / game_id).is_dir():
        pytest.skip(f"environment_files/{game_id}/ not cached")
    if not _HUMAN_BASELINES.exists():
        pytest.skip("data/human_baselines.json not present")

    sink_path = tmp_path / "eval_episodes.jsonl"
    env = ARC3EmbodiedEnv(game_id=game_id, seed=0, max_steps=200)
    sink = EvalRewardSink(env, sink_path=sink_path)
    rng = np.random.default_rng(0)
    try:
        _drive_eval_episodes(sink, env, n_steps=200, rng=rng)
    finally:
        env.close()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/compute_rhae.py",
            "--episodes-file", str(sink_path),
            "--game-id", game_id,
            "--baselines", str(_HUMAN_BASELINES),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"compute_rhae exited non-zero.\nstdout={proc.stdout!r}\n"
        f"stderr={proc.stderr!r}"
    )
    stdout = proc.stdout
    assert game_id in stdout, f"game_id absent from stdout: {stdout!r}"
    m = re.search(r"per_game_rhae=([-+]?\d*\.?\d+)", stdout)
    assert m is not None, f"per_game_rhae not parseable from: {stdout!r}"
    value = float(m.group(1))
    assert not np.isnan(value), f"per_game_rhae is NaN: {stdout!r}"
