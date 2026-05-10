"""Tests for ``arc3_wm.wm_only_agent.WMOnlyAgent`` — Phase 3 WM-only train path.

Surface + source-inspection tests run on the laptop; the runtime-
construction tests skip cleanly via ``pytest.importorskip`` and run on
Vast where the full DreamerV3 stack is installed.

Contract (option-(A): override ``loss`` and ``train`` rather than add a
parallel ``wm_train`` path; needed because ``embodied.jax.Agent.__new__``
hardcodes ``super().__new__(Agent)`` and JIT-wires ``self.model.train`` —
not ``self.model.wm_train`` — at construction time):

1. Subclass relation: ``WMOnlyAgent`` inherits from
   ``dreamerv3.agent.Agent``. ``third_party/dreamerv3/`` is not
   modified.
2. ``WMOnlyAgent.loss`` overrides upstream ``Agent.loss`` and BRANCHES
   BEFORE ``self.imagine(...)`` — imagination is skipped entirely, not
   just dropped from the returned loss tree.
3. The 4 LOSS TERMS exposed by the override are
   ``{recon-key(s), dyn, rew, con}`` (each image obs key gets its own
   recon term; the contract is "no actor/critic loss", not "exactly
   four float entries"). The 5 MODULES that receive gradients are
   ``{enc, dyn, dec, rew, con}``. Module count and loss-term count are
   deliberately distinct.
4. ``self.opt`` is rebuilt in ``WMOnlyAgent.__init__`` over WM modules
   only — pol/val params still exist on the model but are not in the
   optimizer's trainable set. There is NO ``wm_opt``; the WM-only
   contract lives on the override of the inherited ``opt``.
5. ``WMOnlyAgent.train`` overrides upstream ``Agent.train`` minimally:
   identical body except ``self.slowval.update()`` is removed (slow-
   critic bookkeeping is irrelevant on the WM-only path).

Note on lazy subclass: WMOnlyAgent inherits from
``dreamerv3.agent.Agent``, which imports JAX at module load. The impl
uses a lazy-factory pattern (subclass declared inside a function that
the package re-exports) so importing ``arc3_wm.wm_only_agent`` on a
laptop doesn't pull JAX. The laptop-runnable tests below assume this
discipline.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest


# ===========================================================================
# Laptop-runnable: surface + source inspection
# ===========================================================================


def test_module_imports_without_jax():
    """``arc3_wm.wm_only_agent`` must import on a laptop without JAX,
    portal, or dreamerv3 — same discipline as scripts/pretrain_wm.py
    and scripts/launch_pergame.py."""
    for m in [k for k in list(sys.modules)
              if k.startswith(("jax", "portal", "dreamerv3", "embodied"))]:
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location(
        "wm_only_agent_isolation",
        Path(__file__).resolve().parents[1] / "arc3_wm" / "wm_only_agent.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for forbidden in ("jax", "portal", "dreamerv3", "embodied"):
        assert forbidden not in sys.modules, (
            f"top-level import of {forbidden!r} leaked into wm_only_agent"
        )


def test_public_surface_present():
    """Public symbols pinned by the option-(A) contract.

    ``loss`` and ``train`` are overridden on WMOnlyAgent — the inherited
    methods are dreamerv3.agent.Agent's full DreamerV3 paths. The
    overrides must be defined directly on WMOnlyAgent (not just inherited),
    so we check ``vars(cls)`` rather than ``hasattr`` — the latter is
    satisfied by inheritance and would silently regress.

    No ``wm_loss`` / ``wm_train`` method exists. The previous design's
    parallel-method approach is incompatible with embodied.jax.Agent's
    __new__ pattern (see arc3_wm/wm_only_agent.py docstring)."""
    import arc3_wm.wm_only_agent as W
    assert hasattr(W, "WMOnlyAgent"), "missing WMOnlyAgent class"
    cls = W.WMOnlyAgent
    for name in ("loss", "train"):
        assert name in vars(cls), (
            f"WMOnlyAgent must override {name!r} directly (not just inherit) — "
            f"the embodied.jax.Agent JIT pipeline binds self.model.{name} "
            f"at construction time, so a missing override means the upstream "
            f"full-DreamerV3 path runs instead of the WM-only path"
        )
    for absent in ("wm_loss", "wm_train"):
        assert absent not in vars(cls), (
            f"WMOnlyAgent must NOT define {absent!r} — option-(A) contract "
            f"says override the inherited method, do not add a parallel one"
        )


def test_loss_source_skips_imagination():
    """The override BRANCHES before ``self.imagine(...)``. Test checks
    the method source contains no call to imagination / replay-value
    paths from upstream ``loss()`` — burn-in to catch a future
    regression where someone pastes the full upstream body back in.

    Runs on laptop (no JAX needed; just ``inspect.getsource``)."""
    import arc3_wm.wm_only_agent as W
    src = inspect.getsource(W.WMOnlyAgent.loss)
    # Strip the docstring before the forbidden-token scan — the
    # docstring legitimately mentions tokens like `self.dyn.imagine`
    # in backticks while explaining what's intentionally absent.
    body_start = src.find('"""', src.find('"""') + 3) + 3
    body = src[body_start:]
    forbidden = (
        "self.imagine(",       # call expression, not a docstring mention
        "self.dyn.imagine(",
        "imag_loss(",
        "repl_loss(",
    )
    for token in forbidden:
        assert token not in body, (
            f"loss source contains forbidden CALL {token!r} — "
            f"imagination path should be skipped entirely, not just "
            f"have its loss terms dropped from the returned dict"
        )


def test_train_source_skips_slowval_update():
    """The ``train`` override is upstream's body MINUS
    ``self.slowval.update()``. That call is the last line of upstream's
    train (agent.py:137-154) and is the slow-critic bookkeeping that has
    no business firing on the WM-only path."""
    import arc3_wm.wm_only_agent as W
    src = inspect.getsource(W.WMOnlyAgent.train)
    body_start = src.find('"""', src.find('"""') + 3) + 3
    body = src[body_start:]
    assert "self.slowval.update(" not in body, (
        "train override must skip self.slowval.update() — that's the "
        "single material difference between upstream's full train and "
        "the WM-only override"
    )


def test_third_party_dreamerv3_untouched():
    """CLAUDE.md anti-goal + D12: ``third_party/dreamerv3/`` is sacred.
    No file under it should be edited as part of the Phase-3 work."""
    repo_root = Path(__file__).resolve().parents[1]
    dv3 = repo_root / "third_party" / "dreamerv3"
    assert dv3.is_dir(), f"{dv3} missing — repo layout broken"
    # Cheap structural assertion: the upstream main.py and agent.py paths
    # exist and are not no-op shims (anyone replacing them with our own
    # versions would shrink the file dramatically).
    main_py = dv3 / "dreamerv3" / "main.py"
    agent_py = dv3 / "dreamerv3" / "agent.py"
    assert main_py.exists() and main_py.stat().st_size > 5_000, (
        f"{main_py} missing or unexpectedly small — has it been edited?"
    )
    assert agent_py.exists() and agent_py.stat().st_size > 5_000, (
        f"{agent_py} missing or unexpectedly small — has it been edited?"
    )


# ===========================================================================
# Vast-only: needs JAX + dreamerv3 to instantiate the agent.
#
# Module-level ``pytest.importorskip`` would skip the laptop-runnable
# tests above too — not what we want. The ``_require_jax_dv3`` helper
# scopes the skip to the individual tests below.
# ===========================================================================


def _require_jax_dv3():
    pytest.importorskip("jax", reason="WMOnlyAgent runtime tests require JAX")
    pytest.importorskip(
        "dreamerv3",
        reason="WMOnlyAgent inherits from dreamerv3.agent.Agent",
    )


def test_wm_only_agent_subclass_of_dreamerv3_agent():
    _require_jax_dv3()
    """``WMOnlyAgent`` IS a ``dreamerv3.agent.Agent``. The pretrain loop
    swaps in ``WMOnlyAgent`` wherever vanilla Agent would otherwise
    construct."""
    import dreamerv3.agent as dv3_agent
    from arc3_wm.wm_only_agent import WMOnlyAgent
    assert issubclass(WMOnlyAgent, dv3_agent.Agent), (
        "WMOnlyAgent must inherit from dreamerv3.agent.Agent — not duplicate it"
    )


def _build_minimal_wm_only_agent():
    """Construct a ``WMOnlyAgent`` against a tiny obs/act space using the
    ``size12m`` config block. Vast-only (requires JAX); CPU is fine.

    Mirrors ``scripts.pretrain_wm.make_wm_only_agent`` minus the
    full-config ladder — the size12m block alone is sufficient for the
    structural assertions below, and avoids loading the full arc3.yaml
    pretrain block (which the script's main path needs but the tests
    don't).
    """
    import elements
    import numpy as np
    import scripts.pretrain_wm as P
    from arc3_wm.action_space import N_ACTIONS
    from arc3_wm.embodied_env import OBS_HW
    from arc3_wm.wm_only_agent import WMOnlyAgent

    merged = P.load_merged_configs()
    config = elements.Config(merged["defaults"])
    config = config.update(merged["size12m"])
    config = config.update(merged["arc3"])
    config = config.update(merged["pretrain"])
    config = config.update(logdir="/tmp/wm-only-agent-test", seed=0)

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
        obs_space, act_space,
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


def test_modules_are_five_wm_modules_only():
    """``self.modules`` (the optimizer's trainable set) is exactly
    [dyn, enc, dec, rew, con] — upstream's order from ``Agent.modules``
    (agent.py:74-75) minus pol+val. Pol and val instances still exist on
    the agent (``agent.model.pol``, ``agent.model.val``) but are absent
    from ``self.modules`` so the optimizer can't apply grads to them."""
    _require_jax_dv3()
    agent = _build_minimal_wm_only_agent()
    names = [getattr(m, "name", type(m).__name__) for m in agent.model.modules]
    assert names == ["dyn", "enc", "dec", "rew", "con"], (
        f"WMOnlyAgent.modules expected order [dyn, enc, dec, rew, con]; "
        f"got {names}"
    )
    # Pol/val instances must still exist on the model — we don't delete
    # them, we just exclude them from the optimizer.
    assert hasattr(agent.model, "pol"), "agent.model.pol disappeared"
    assert hasattr(agent.model, "val"), "agent.model.val disappeared"


def test_opt_covers_wm_modules_only():
    """The single ``self.opt`` (rebuilt in WMOnlyAgent.__init__) is over
    the 5 WM modules only. There is no second optimizer; the WM-only
    contract lives on the rebuilt inherited ``opt``."""
    _require_jax_dv3()
    agent = _build_minimal_wm_only_agent()
    # The optimizer's modules attribute is what gets gradients applied.
    opt_modules = getattr(agent.model.opt, "modules", None)
    assert opt_modules is not None, (
        "agent.model.opt.modules attribute missing — embodied.jax.Optimizer "
        "API surface changed?"
    )
    names = [getattr(m, "name", type(m).__name__) for m in opt_modules]
    assert set(names) == {"dyn", "enc", "dec", "rew", "con"}, (
        f"opt covers {set(names)}; expected exactly the 5 WM modules"
    )
    assert "pol" not in names and "val" not in names, (
        f"actor/critic leaked into the optimizer's trainable set: {names}"
    )
    # And there's no parallel wm_opt — option-(A) contract.
    assert not hasattr(agent.model, "wm_opt"), (
        "WMOnlyAgent must NOT define a separate wm_opt — option-(A) "
        "rebuilds the inherited self.opt over WM modules instead"
    )


def test_loss_emits_no_actor_or_critic_keys():
    """Loss-tree shape: the dict returned by the ``loss`` override must
    not contain any of the imagination / actor / critic / replay-value
    loss keys that upstream ``loss()`` produces.

    The override returns ``(loss_value, (carry, entries, outs, metrics))``
    where ``outs['losses']`` is the per-term dict. We construct a minimal
    batch and call ``loss`` directly — no JIT, no optimizer step, just
    the loss surface.

    NB: this test bypasses the JIT pipeline by calling ``model.init_train``
    + ``model.loss`` eagerly. embodied.jax.Agent's outer wrapper installs
    a strict GSPMD sharding setup at __init__ that requires every
    ``jnp.zeros`` (e.g. inside upstream rssm.initial) to flow through
    explicit ``jax.device_put`` with the right sharding — which the JIT
    pipeline does for us, but eager Python calls don't. Wrapping the
    direct calls in ``jax.transfer_guard("allow")`` relaxes the guard
    for the duration of this contract check; production training is
    unaffected.
    """
    _require_jax_dv3()
    import jax
    import numpy as np
    from arc3_wm.action_space import N_ACTIONS
    from arc3_wm.embodied_env import OBS_HW

    agent = _build_minimal_wm_only_agent()
    model = agent.model

    # Tiny synthetic batch — B=2, T=4 to keep dyn happy without burning time.
    B, T = 2, 4
    is_first = np.zeros((B, T), dtype=bool)
    is_first[:, 0] = True
    is_last = np.zeros((B, T), dtype=bool)
    is_last[:, -1] = True
    is_terminal = np.zeros((B, T), dtype=bool)
    is_terminal[:, -1] = True
    obs = {
        "image": np.zeros((B, T, OBS_HW, OBS_HW, 3), dtype=np.uint8),
        "reward": np.zeros((B, T), dtype=np.float32),
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }
    prevact = {"action": np.zeros((B, T), dtype=np.int32)}

    with jax.transfer_guard("allow"):
        carry = model.init_train(B)
        _loss_value, (_carry, _entries, outs, _metrics) = model.loss(
            carry, obs, prevact, training=False)

    losses = outs.get("losses", {})
    forbidden = {
        "policy", "actor", "value", "critic", "repval",
        "imag_value", "imag_actor", "imag_advantage",
    }
    bad = set(losses) & forbidden
    assert not bad, (
        f"loss override returned forbidden actor/critic/imag keys: {bad} "
        f"(full set: {set(losses)})"
    )
    # Positive: each WM term family present.
    families = {"dyn", "rew", "con", "image"}  # 'image' = per-key recon
    assert families.issubset(set(losses)), (
        f"loss override missing WM terms: {families - set(losses)} "
        f"(have {set(losses)})"
    )
