"""Tests for ``scripts/pretrain_wm.py`` — Phase 3 cross-game WM pretraining.

This is the red-skeleton companion file. The script does not exist yet;
collection will fail at the top-level ``import scripts.pretrain_wm`` and
the suite drops in red. This is the documented Phase-1.7 / Phase-1.8 / now
Phase-3 pattern: tests first, impl second, concern-ordered commits.

Phase 3 gate (from ``docs/phase-checklists.md``):

- All 340 replays load into ``embodied.replay`` (laptop tests use a tiny
  synthetic buffer; the real-replay sweep is Vast-only).
- WM-only updates verified by code inspection (here: by mock-recording
  the agent's training entry points; the pretrain loop must call a
  WM-only entry, never the regular ``agent.train`` that includes
  imagination + actor + critic loss terms).
- All four WM losses (recon, dynamics, reward, continue) are wired
  through to the optimizer.
- Checkpoint cadence — ≥30 min in production; here we test the cadence
  *mechanism* fires on a dialled-down interval.
- Resume verified — round-trip checkpoint write / read.
- RHAE logging hook fires every N steps on a held-out replay.

Heavy DreamerV3 / JAX deps are NOT imported here. The pretrain script is
expected to mirror ``scripts/launch_pergame.py``'s lazy-import discipline
(D12: own launcher, never fork dreamerv3). Tests use mock agent / mock
optimizer / mock replay objects so the laptop suite runs offline.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

import numpy as np
import pytest


# Importing the not-yet-written module is the red signal. Once the impl
# lands, this import succeeds and the suite below runs.
import scripts.pretrain_wm as P  # noqa: E402


# ---------------------------------------------------------------------------
# Fabricated-replay helpers — mirror tests/test_replay_loader.py
# ---------------------------------------------------------------------------

OBS_HW = 64
ZERO_LAYER = [[0] * OBS_HW for _ in range(OBS_HW)]
ZERO_FRAME = [ZERO_LAYER]


def _row(
    action_id: Any,
    *,
    state: str = "NOT_FINISHED",
    levels_completed: int = 0,
    action_data: Optional[dict] = None,
    game_id: str = "test-game-abc",
    win_levels: int = 7,
) -> dict:
    if action_data is None:
        action_data = {"game_id": game_id}
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "data": {
            "game_id": game_id,
            "frame": ZERO_FRAME,
            "state": state,
            "action_input": {"id": action_id, "data": action_data, "reasoning": None},
            "guid": "fab-guid-0001",
            "full_reset": False,
            "available_actions": [1, 2, 3, 4, 5, 6, 7],
            "levels_completed": levels_completed,
            "win_levels": win_levels,
        },
    }


def _summary_row(*, levels_completed: int = 0, total_actions: int = 0) -> dict:
    return {
        "timestamp": "2026-01-01T00:01:00+00:00",
        "data": {
            "levels_completed": levels_completed,
            "won": False,
            "played": True,
            "total_actions": total_actions,
            "cards": [],
        },
    }


def _write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _make_synthetic_replay_tree(root: Path) -> int:
    """Build a 2-game / 2-file synthetic replay tree under ``root``.

    Each file: RESET + 3 step rows + WIN. Replay loader yields one episode
    of 4 step dicts per file → 16 step dicts total across 2 games × 2 files.
    Returns the expected total step-dict count.
    """
    rows = [
        _row(0),  # RESET
        _row(1, levels_completed=0),
        _row(3, levels_completed=0),
        _row(5, levels_completed=1, state="WIN"),
    ]
    for game_id in ("alpha", "beta"):
        for i in range(2):
            _write_jsonl(root / game_id / f"file{i}.recording.jsonl", rows)
    return 2 * 2 * 4  # 2 games × 2 files × 4 steps/episode


# ---------------------------------------------------------------------------
# Mock objects — stand-ins for embodied.replay / dreamerv3.Agent
# ---------------------------------------------------------------------------


class FakeReplay:
    """Records ``add`` calls; supports ``__len__``; samples are mock dicts."""

    def __init__(self) -> None:
        self.added: List[dict] = []
        self._sample_idx = 0

    def add(self, step: dict, worker: int = 0) -> None:
        # Drop log/* per the embodied convention.
        clean = {k: v for k, v in step.items() if not k.startswith("log/")}
        self.added.append(clean)

    def __len__(self) -> int:
        return len(self.added)

    def stats(self) -> dict:
        return {"items": len(self.added), "inserts": len(self.added)}

    def sample(self, batch: int, mode: str = "train") -> dict:
        # Tiny mock batch — shape mirrors embodied stream output but with
        # B=batch, T=1 so dims line up for any downstream einops.
        self._sample_idx += 1
        return {
            "image": np.zeros((batch, 1, OBS_HW, OBS_HW, 3), dtype=np.uint8),
            "action": np.zeros((batch, 1), dtype=np.int32),
            "reward": np.zeros((batch, 1), dtype=np.float32),
            "is_first": np.zeros((batch, 1), dtype=bool),
            "is_last": np.zeros((batch, 1), dtype=bool),
            "is_terminal": np.zeros((batch, 1), dtype=bool),
        }


class _Spy:
    """Records ``__call__`` invocations. Stand-in for actor/critic
    grad-application calls (``self.pol.update`` / ``self.val.update``
    in the real Agent code path)."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, *args, **kwargs) -> None:
        self.call_count += 1


class RecordingAgent:
    """Mock agent that records which training entry points were called
    and what the WM-only ``train`` path touched.

    Phase 3 gates after option-(A) landing:

    1. ``agent.train`` IS the WM-only path (no separate ``wm_train``
       method). The pretrain loop calls ``agent.train`` — the contract
       no longer hangs on a method-name distinction.
    2. ``agent.policy`` is never called (no env rollouts in pretrain).
    3. The actor/critic update paths (``self.pol.update`` /
       ``self.val.update`` spies) are not invoked. Property-based: even
       if someone wires the full upstream loss back in, the override on
       WMOnlyAgent.train must not fire those bookkeeping calls.
    4. The 4 LOSS TERMS exposed by train are ``{recon, dyn, rew, con}``.
       (No actor/critic keys.)
    5. The 5 MODULES updated are ``{enc, dyn, dec, rew, con}``. (No
       pol/val.) Module count != loss-term count is intentional.
    """

    def __init__(self) -> None:
        self.train_calls = 0
        self.wm_modules_updated: List[str] = []
        self.actor_critic_modules_updated: List[str] = []
        self.last_loss_keys: set[str] = set()
        self.report_calls = 0
        self._carry = object()
        # Spies on the actor/critic grad-application paths. The real
        # upstream Agent.train calls self.slowval.update() unconditionally;
        # the WM-only override must skip that. We expose pol.update and
        # val.update as the property-based gate — both must stay at 0.
        self.pol = type("_Mod", (), {"update": _Spy()})()
        self.val = type("_Mod", (), {"update": _Spy()})()
        self.slowval = type("_Mod", (), {"update": _Spy()})()

    def init_train(self, batch_size: int):
        return self._carry

    def init_report(self, batch_size: int):
        return self._carry

    def init_policy(self, *args, **kwargs):
        return self._carry

    def stream(self, source):
        # Pass-through — pretrain loop iterates the source directly.
        return iter(source)

    def train(self, carry, batch):
        # WM-only path — 4 loss terms, 5 modules. Mirrors the override
        # on WMOnlyAgent.train: no slowval.update, no actor/critic
        # bookkeeping.
        self.train_calls += 1
        self.wm_modules_updated.extend(["enc", "dyn", "dec", "rew", "con"])
        # 4 loss terms: recon (one per image obs key — here a single
        # 'image'), dyn, rew, con. NO actor/critic keys.
        loss_dict = {
            "loss/image": 0.5,   # recon (per-key)
            "loss/dyn": 0.5,
            "loss/rew": 0.5,
            "loss/con": 0.5,
        }
        self.last_loss_keys = set(loss_dict)
        return carry, {}, loss_dict

    def report(self, carry, batch):
        self.report_calls += 1
        return carry, {"report/anything": 0.0}

    def policy(self, *args, **kwargs):  # pragma: no cover — pretrain has no policy rollouts
        raise AssertionError(
            "pretrain WM loop must not invoke agent.policy — no env interaction"
        )

    def save(self):
        return {"agent_state": "ok"}

    def load(self, state):
        self._loaded = state


# ---------------------------------------------------------------------------
# (1) Module surface — public API contract
# ---------------------------------------------------------------------------


def test_module_imports_without_jax():
    """Heavy deps stay lazy. Module-level import must succeed on a laptop
    with no JAX / portal / dreamerv3 — same discipline as launch_pergame."""
    # Drop cached state and re-import via spec to catch top-level leaks.
    for m in [k for k in list(sys.modules) if k.startswith(("jax", "portal", "dreamerv3", "embodied"))]:
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location(
        "pretrain_wm_isolation",
        Path(__file__).resolve().parents[1] / "scripts" / "pretrain_wm.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for forbidden in ("jax", "portal", "dreamerv3", "embodied"):
        assert forbidden not in sys.modules, (
            f"top-level import of {forbidden!r} leaked into pretrain_wm"
        )


def test_public_surface_present():
    for name in (
        "build_argparser",
        "parse_args",
        "build_config",
        "populate_buffer_from_replays",
        "make_wm_only_agent",
        "pretrain_wm_loop",
        "RHAEHeldOutHook",
        "main",
    ):
        assert hasattr(P, name), f"pretrain_wm missing public symbol {name!r}"


# ---------------------------------------------------------------------------
# (2) Argparse — required flags + leftover passthrough
# ---------------------------------------------------------------------------


def test_argparser_required_flags():
    p = P.build_argparser()
    with pytest.raises(SystemExit):
        p.parse_args([])  # both --logdir and --replays-root required
    with pytest.raises(SystemExit):
        p.parse_args(["--logdir", "/tmp/x"])  # missing --replays-root


def test_argparser_happy_path(tmp_path):
    args, leftover = P.parse_args(
        [
            "--logdir", str(tmp_path / "run"),
            "--replays-root", str(tmp_path / "replays"),
            "--seed", "7",
        ]
    )
    assert args.logdir == str(tmp_path / "run")
    assert args.replays_root == str(tmp_path / "replays")
    assert args.seed == 7
    assert leftover == []


def test_argparser_passthrough_leftover(tmp_path):
    """elements-style key=value flags must survive parse_known_args, same
    as scripts/launch_pergame.py."""
    args, leftover = P.parse_args(
        [
            "--logdir", str(tmp_path / "r"),
            "--replays-root", str(tmp_path / "rep"),
            "--run.steps", "100000",
            "--batch_size", "8",
        ]
    )
    assert "--run.steps" in leftover and "100000" in leftover
    assert "--batch_size" in leftover and "8" in leftover


def test_argparser_default_configs_includes_size12m_and_arc3(tmp_path):
    """Default --configs ladder must layer the WM-pretrain block on top of
    size12m + arc3 so per-suite knobs (env.arc3.max_steps etc.) survive
    even though pretrain has no env."""
    args, _ = P.parse_args(
        [
            "--logdir", str(tmp_path / "r"),
            "--replays-root", str(tmp_path / "rep"),
        ]
    )
    # Order matters — 'pretrain' overrides come last.
    assert args.configs[-1] == "pretrain", (
        f"pretrain must be the rightmost config block; got {args.configs}"
    )
    for needed in ("size12m", "arc3"):
        assert needed in args.configs, (
            f"default --configs must include {needed!r}; got {args.configs}"
        )


# ---------------------------------------------------------------------------
# (3) Buffer pre-population — load_replays_directory → replay.add
# ---------------------------------------------------------------------------


def test_populate_buffer_synthetic_count_and_keys(tmp_path):
    """Pre-populating a fake buffer from a synthetic replay tree:
    transition count matches the replay-loader output exactly, and each
    step dict carries the embodied buffer schema (image/action/reward/
    is_first/is_last/is_terminal)."""
    root = tmp_path / "replays"
    expected = _make_synthetic_replay_tree(root)
    replay = FakeReplay()
    n = P.populate_buffer_from_replays(replay, root)
    assert n == expected, f"reported count {n} != expected {expected}"
    assert len(replay) == expected
    expected_keys = {"image", "action", "reward", "is_first", "is_last", "is_terminal"}
    for step in replay.added:
        assert set(step.keys()) >= expected_keys, (
            f"step missing keys; have {set(step.keys())}, need {expected_keys}"
        )


def test_populate_buffer_dtypes_match_replay_loader(tmp_path):
    """Whatever ``arc3_wm.replay_loader`` yields per-step lands in the
    buffer unchanged — uint8 image, int32 action, float32 reward, bool
    flags. Catches a future regression where pretrain_wm cast on the way
    in and silently shifted the WM input distribution."""
    root = tmp_path / "replays"
    _make_synthetic_replay_tree(root)
    replay = FakeReplay()
    P.populate_buffer_from_replays(replay, root)
    s = replay.added[0]
    assert s["image"].dtype == np.uint8
    assert s["image"].shape == (OBS_HW, OBS_HW, 3)
    assert s["action"].dtype == np.int32
    assert s["reward"].dtype == np.float32
    for k in ("is_first", "is_last", "is_terminal"):
        assert s[k].dtype == np.bool_


def test_populate_buffer_per_game_distribution(tmp_path):
    """Per-game distribution must be roughly even — Phase 3 gate row 1
    explicitly calls this out. With 2 games × 2 files each, exactly half
    the transitions come from each game.

    The fake replay's worker-id channel is what we'd use in production
    (one worker per game); for the synthetic case we record game_id via
    the optional ``stats`` dict the helper supports."""
    root = tmp_path / "replays"
    _make_synthetic_replay_tree(root)
    replay = FakeReplay()
    stats: dict = {}
    P.populate_buffer_from_replays(replay, root, stats=stats)
    assert "per_game_counts" in stats, (
        "populate_buffer_from_replays should expose per-game step counts "
        "via stats dict so the Phase-3 distribution gate is checkable"
    )
    counts = stats["per_game_counts"]
    assert set(counts) == {"alpha", "beta"}, f"unexpected games: {set(counts)}"
    assert counts["alpha"] == counts["beta"], (
        f"per-game distribution is uneven on synthetic 2x2 tree: {counts}"
    )


def test_populate_buffer_skips_post_terminal_noise(tmp_path):
    """Replay loader's post-terminal-noise rule (Phase 1.7) must flow
    through unchanged. A terminal row followed by another terminal row
    is noise — those rows do not become buffer transitions."""
    rows = [
        _row(0),
        _row(1, levels_completed=0),
        _row(2, levels_completed=1, state="WIN"),
        _row(2, levels_completed=1, state="WIN"),  # noise
        _row(2, levels_completed=1, state="WIN"),  # noise
    ]
    root = tmp_path / "replays" / "noise-game"
    _write_jsonl(root / "noisy.recording.jsonl", rows)
    replay = FakeReplay()
    stats: dict = {}
    n = P.populate_buffer_from_replays(replay, tmp_path / "replays", stats=stats)
    # Episode = 3 step dicts (RESET + step + terminal). The two trailing
    # noise rows are discarded; stats record them.
    assert n == 3, f"expected 3 transitions after noise drop, got {n}"
    assert stats.get("noise_rows_discarded", 0) == 2


# ---------------------------------------------------------------------------
# (4) Custom run loop — WM-only, no actor/critic, no env rollouts
# ---------------------------------------------------------------------------


def test_pretrain_loop_calls_train(tmp_path):
    """The Phase-3 gate after option-(A) landing: ``agent.train`` IS the
    WM-only path. The pretrain loop calls it; correctness is enforced by
    the property-based assertions below (actor/critic spies stay at 0,
    only WM modules get updates, loss dict has only WM keys)."""
    agent = RecordingAgent()
    replay, args = _seed_replay_and_args(tmp_path)

    P.pretrain_wm_loop(agent=agent, replay=replay, logger=mock.MagicMock(), args=args)

    assert agent.train_calls > 0, "agent.train was never called"


def test_pretrain_loop_does_not_invoke_policy(tmp_path):
    """No env rollouts during pretrain — ``agent.policy`` must never run.
    The mock raises AssertionError if it does, so this also tests a future
    regression where someone adds a Driver to the pretrain loop."""
    agent = RecordingAgent()
    replay, args = _seed_replay_and_args(tmp_path, n_seed=4, steps=2)
    # If policy is touched, RecordingAgent.policy raises — catches regressions.
    P.pretrain_wm_loop(agent=agent, replay=replay, logger=mock.MagicMock(), args=args)


def _seed_replay_and_args(tmp_path, *, n_seed: int = 8, steps: int = 4):
    """Common scaffolding: a replay seeded so trainfn's len-check passes,
    plus a MagicMock ``args`` tuned for fast tests (cadence fires every
    step). Returns (replay, args)."""
    replay = FakeReplay()
    for _ in range(n_seed):
        replay.add({
            "image": np.zeros((OBS_HW, OBS_HW, 3), np.uint8),
            "action": np.int32(0),
            "reward": np.float32(0.0),
            "is_first": np.bool_(False),
            "is_last": np.bool_(False),
            "is_terminal": np.bool_(False),
        })
    args = mock.MagicMock(
        steps=steps, batch_size=2, batch_length=1, train_ratio=1,
        log_every=10, save_every=10, report_every=1_000_000,
        logdir=str(tmp_path / "run"),
    )
    return replay, args


def test_pretrain_loop_actor_critic_paths_not_invoked(tmp_path):
    """SHARPENED GATE (option-(A) property contract): the actor / critic
    grad-application paths must not fire during pretrain. Concretely,
    ``self.pol.update``, ``self.val.update``, and ``self.slowval.update``
    are spied on — all must stay at zero invocations across the loop.

    This replaces the old "self.opt vs self.wm_opt distinct optimizers"
    test — there is now exactly one optimizer (over WM modules only),
    so the gate can no longer be expressed by counting steps on a
    second optimizer instance."""
    agent = RecordingAgent()
    replay, args = _seed_replay_and_args(tmp_path)

    P.pretrain_wm_loop(agent=agent, replay=replay, logger=mock.MagicMock(), args=args)

    assert agent.pol.update.call_count == 0, (
        f"agent.pol.update fired {agent.pol.update.call_count}× — actor "
        "grad-application must not run on the WM-only path"
    )
    assert agent.val.update.call_count == 0, (
        f"agent.val.update fired {agent.val.update.call_count}× — critic "
        "grad-application must not run on the WM-only path"
    )
    assert agent.slowval.update.call_count == 0, (
        f"agent.slowval.update fired {agent.slowval.update.call_count}× — "
        "slow-critic bookkeeping is upstream Agent.train's last line and "
        "must be skipped by WMOnlyAgent.train"
    )
    assert agent.actor_critic_modules_updated == [], (
        f"pol/val modules received gradient updates: "
        f"{agent.actor_critic_modules_updated}"
    )


def test_pretrain_loop_emits_only_four_wm_loss_terms(tmp_path):
    """SHARPENED GATE (chat sharpening — reconciliation): exactly the 4
    WM LOSS TERMS appear in the loss dict — recon (per-image-key, here
    single ``image``), ``dyn``, ``rew``, ``con``. None of upstream
    ``loss()``'s actor/critic/replay-value keys leak through.

    Distinct from the 5-modules test: the contract is "no actor/critic
    LOSS COMPUTED", not "exactly 4 floats returned"."""
    agent = RecordingAgent()
    replay, args = _seed_replay_and_args(tmp_path)

    P.pretrain_wm_loop(agent=agent, replay=replay, logger=mock.MagicMock(), args=args)

    keys = agent.last_loss_keys
    forbidden = {
        "loss/policy", "loss/actor", "loss/value", "loss/critic",
        "loss/repval", "loss/imag",
    }
    assert keys & forbidden == set(), (
        f"loss dict contains forbidden actor/critic/imag keys: "
        f"{keys & forbidden}"
    )
    # Positive assertion: each of the 4 WM term families is present.
    families = {"image", "dyn", "rew", "con"}  # 'image' = per-key recon
    seen = {k.split("/", 1)[1] for k in keys if "/" in k}
    missing = families - seen
    assert not missing, f"WM loss dict missing terms: {missing} (have {seen})"


def test_pretrain_loop_five_wm_modules_receive_gradients(tmp_path):
    """SHARPENED GATE (chat sharpening — reconciliation): the 5 MODULES
    that receive gradient updates are exactly ``{enc, dyn, dec, rew,
    con}``. Module count = 5 (the recorded modules) is deliberately
    distinct from loss-term count = 4 (the loss-key families)."""
    agent = RecordingAgent()
    replay, args = _seed_replay_and_args(tmp_path)

    P.pretrain_wm_loop(agent=agent, replay=replay, logger=mock.MagicMock(), args=args)

    updated = set(agent.wm_modules_updated)
    expected = {"enc", "dyn", "dec", "rew", "con"}
    assert updated == expected, (
        f"WM modules updated = {updated}; expected exactly {expected} "
        f"(5 modules, distinct from the 4 loss terms)"
    )
    assert "pol" not in updated and "val" not in updated, (
        f"actor/critic module leaked into WM update set: {updated}"
    )


# ---------------------------------------------------------------------------
# (5) Checkpoint cadence + resume round-trip
# ---------------------------------------------------------------------------


def test_pretrain_checkpoint_written_at_cadence(tmp_path):
    """Checkpoint mechanism fires at the configured cadence. Phase-3
    requirement: ≥30 min in production; here we set save_every=0 (fire
    every step) so the test runs in ms."""
    agent = RecordingAgent()
    replay, args = _seed_replay_and_args(tmp_path, n_seed=4, steps=3)
    logdir = Path(args.logdir)
    args.save_every = 0  # fire every step

    P.pretrain_wm_loop(agent=agent, replay=replay, logger=mock.MagicMock(), args=args)
    # By convention, dreamerv3's elements.Checkpoint writes under <logdir>/ckpt.
    assert (logdir / "ckpt").exists(), (
        f"no checkpoint directory under {logdir}; cadence mechanism didn't fire"
    )


def test_pretrain_resume_picks_up_from_existing_logdir(tmp_path):
    """Re-running with the same --logdir loads the prior checkpoint.
    Identical convention to ``embodied.run.train``: the loop calls
    ``cp.load_or_save()`` on entry, so a second invocation must call
    ``agent.load`` rather than starting fresh.
    """
    replay, args = _seed_replay_and_args(tmp_path, n_seed=4, steps=2)
    args.logdir = str(tmp_path / "run-resume")
    args.save_every = 0
    # First run — writes a checkpoint.
    P.pretrain_wm_loop(agent=RecordingAgent(), replay=replay, logger=mock.MagicMock(), args=args)

    # Second run — same logdir → must trigger a load on the new agent.
    agent2 = RecordingAgent()
    P.pretrain_wm_loop(agent=agent2, replay=replay, logger=mock.MagicMock(), args=args)
    assert hasattr(agent2, "_loaded"), (
        "second invocation against an existing logdir didn't call agent.load — "
        "Phase 3 resume gate fails"
    )


# ---------------------------------------------------------------------------
# (6) RHAE held-out logging hook
# ---------------------------------------------------------------------------


def test_rhae_hook_fires_every_n_steps():
    """The held-out RHAE hook fires exactly every ``every_n_steps``, no
    more, no less. Phase 3 gate row 7: predicted level-up probability
    spikes near actual level-up boundaries on a held-out replay — the
    hook is what produces that figure."""
    held_out = [{"image": np.zeros((OBS_HW, OBS_HW, 3), np.uint8)} for _ in range(3)]
    hook = P.RHAEHeldOutHook(holdout=held_out, every_n_steps=5)
    fires = []
    for step in range(13):
        # Hook returns metrics (or None) when it fires; record steps where it ran.
        out = hook(step=step, agent=mock.MagicMock())
        if out is not None:
            fires.append(step)
    # First fire at step 0, then every 5 steps: 0, 5, 10.
    assert fires == [0, 5, 10], f"unexpected fire schedule: {fires}"


def test_rhae_hook_emits_level_up_probability_metric():
    """When the hook fires it returns a metrics dict containing a
    ``rhae/level_up_prob`` key (the Phase-3 deliverable). The actual
    value comes from the agent's reward / continue head; here we just
    pin the contract so the impl doesn't drift to a different key."""
    held_out = [{"image": np.zeros((OBS_HW, OBS_HW, 3), np.uint8)}]
    hook = P.RHAEHeldOutHook(holdout=held_out, every_n_steps=1)
    metrics = hook(step=0, agent=mock.MagicMock())
    assert metrics is not None
    assert any(k.startswith("rhae/") for k in metrics), (
        f"RHAE hook output should expose rhae/* metrics; got {set(metrics)}"
    )


def test_rhae_hook_does_not_fire_off_schedule():
    """Sanity check that step-not-a-multiple-of-N truly returns None,
    not an empty dict — the loop uses ``if metrics is not None``."""
    held_out = [{"image": np.zeros((OBS_HW, OBS_HW, 3), np.uint8)}]
    hook = P.RHAEHeldOutHook(holdout=held_out, every_n_steps=10)
    assert hook(step=1, agent=mock.MagicMock()) is None
    assert hook(step=9, agent=mock.MagicMock()) is None


# ---------------------------------------------------------------------------
# (7) Config — pretrain block layered cleanly on top of size12m + arc3
# ---------------------------------------------------------------------------


def test_pretrain_config_block_present():
    """``configs/arc3.yaml`` (or wherever pretrain_wm pulls from) defines
    a ``pretrain`` block. Same collision discipline as launch_pergame —
    the merged dict must carry the block name."""
    merged = P.load_merged_configs()
    assert "pretrain" in merged, (
        "pretrain block missing from merged configs — add to configs/arc3.yaml"
    )


def test_pretrain_config_disables_imagination_loss():
    """Defensive belt-and-braces: WMOnlyAgent's overridden train is the
    only WM-only enforcement at the agent level, but the merged config
    still sets ``script=pretrain_wm`` so a stray invocation of
    ``embodied.run.train`` against the same config would trip on an
    unknown script name rather than silently train actor+critic."""
    args, leftover = P.parse_args(
        [
            "--logdir", "/tmp/x",
            "--replays-root", "/tmp/y",
        ]
    )
    config = P.build_config(args, leftover)
    # Marker: the pretrain block should set ``run.script = pretrain_wm``
    # (sibling of train/train_eval/eval_only) so a stray invocation of
    # embodied.run.train would trip on an unknown script.
    assert getattr(config, "script", None) == "pretrain_wm", (
        f"pretrain config must set script=pretrain_wm; got {getattr(config, 'script', None)!r}"
    )
