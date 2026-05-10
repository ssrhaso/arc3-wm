"""Tests for ``arc3_wm.wm_only_agent.WMOnlyAgent`` — Phase 3 WM-only train path.

Red skeleton. Surface + source-inspection tests run on the laptop; the
runtime-construction tests skip cleanly via ``pytest.importorskip`` and
run on Vast where the full DreamerV3 stack is installed.

Contract (sharpenings from chat):

1. Subclass relation: ``WMOnlyAgent`` inherits from
   ``dreamerv3.agent.Agent``. Override is a trivial diff against
   upstream ``agent.py:137-216``. ``third_party/dreamerv3/`` is not
   modified.
2. ``wm_loss`` branches BEFORE ``self.imagine(...)`` — imagination is
   skipped entirely, not just dropped from the returned loss tree.
3. The 4 LOSS TERMS exposed by ``wm_loss`` are
   ``{recon-key(s), dyn, rew, con}`` (note: each image obs key gets its
   own recon term; the contract is "no actor/critic loss", not
   "exactly four float entries"). The 5 MODULES that receive gradients
   are ``{enc, dyn, dec, rew, con}``. Module count and loss-term count
   are deliberately distinct.
4. ``wm_opt`` is a NEW ``embodied.jax.Optimizer`` over ``wm_modules``
   (5 modules) only. The inherited ``self.opt`` (7 modules) is
   untouched but never called from this code path.

Note on lazy subclass: WMOnlyAgent inherits from
``dreamerv3.agent.Agent``, which imports JAX at module load. The impl
is expected to use a lazy-factory pattern (subclass declared inside a
function that the package re-exports) so importing
``arc3_wm.wm_only_agent`` on a laptop doesn't pull JAX. The
laptop-runnable tests below assume this discipline.
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
    """Public symbols pinned by the chat contract."""
    import arc3_wm.wm_only_agent as W
    assert hasattr(W, "WMOnlyAgent"), "missing WMOnlyAgent class"
    cls = W.WMOnlyAgent
    for name in ("wm_loss", "wm_train"):
        assert callable(getattr(cls, name, None)), (
            f"WMOnlyAgent missing method {name!r}"
        )


def test_wm_loss_source_skips_imagination():
    """The override BRANCHES before ``self.imagine(...)``. Test checks
    the method source contains no call to imagination / replay-value
    paths from upstream ``loss()`` — burn-in to catch a future
    regression where someone pastes the full upstream body back in.

    Runs on laptop (no JAX needed; just ``inspect.getsource``)."""
    import arc3_wm.wm_only_agent as W
    src = inspect.getsource(W.WMOnlyAgent.wm_loss)
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
            f"wm_loss source contains forbidden CALL {token!r} — "
            f"imagination path should be skipped entirely, not just "
            f"have its loss terms dropped from the returned dict"
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


def test_wm_modules_are_five_named_modules():
    """``wm_modules`` is exactly [enc, dyn, dec, rew, con]. Order is the
    upstream order from ``Agent.modules`` minus pol+val (lines 74-75 of
    upstream agent.py)."""
    _require_jax_dv3()
    pytest.skip("requires a built WMOnlyAgent instance; covered by Phase-3 dry run")


def test_wm_opt_distinct_from_inherited_opt():
    """``self.wm_opt`` is a separate optimizer instance from inherited
    ``self.opt``. Spy-on-step tests in test_pretrain_wm.py rely on
    these being independent objects."""
    _require_jax_dv3()
    pytest.skip("requires a built WMOnlyAgent instance; covered by Phase-3 dry run")


def test_wm_loss_emits_no_actor_or_critic_keys():
    """Loss-tree shape (Vast-only): the dict returned by ``wm_loss``
    must not contain any of the imagination / actor / critic / replay-
    value loss keys that upstream ``loss()`` produces."""
    _require_jax_dv3()
    pytest.skip("requires a built WMOnlyAgent instance; covered by Phase-3 dry run")
