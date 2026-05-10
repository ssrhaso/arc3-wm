"""Stub for arc3_wm.wm_only_agent — WM-only DreamerV3 Agent subclass.

Tests-first per CLAUDE.md. Implementation lands after the red commit
matrix is reviewed. See ``tests/test_wm_only_agent.py`` for the
contract this module must satisfy.

Why a subclass and not a fork:
- ``third_party/dreamerv3/`` is sacred (CLAUDE.md anti-goal: "Do not
  refactor DreamerV3 internals"; D12 reaffirms "DreamerV3 source is
  unmodified").
- ``WMOnlyAgent`` lives in this package and inherits from upstream
  ``dreamerv3.agent.Agent``. Override surface is small (~30-40 lines):
  one method to compute a WM-only loss, one to drive a WM-only train
  step, plus an extra optimizer.

Why skip ``imagine()`` entirely (not just drop actor/critic from the
returned loss tree):
- Imagination rollouts cost roughly the same compute as a WM update
  (16-step horizon, full RSSM unroll). On the Phase-3 6h budget
  that's ~2h of pure waste if we let it run.
- Cleanest cut: branch before ``self.imagine(...)`` in the WM-only
  ``loss()`` override. The override is a near-verbatim copy of upstream
  ``agent.py:156-186`` (the World-model block), then return.

Public surface (matches ``tests/test_wm_only_agent.py`` and the
sharpening notes from chat):

- ``WMOnlyAgent`` — subclass of ``dreamerv3.agent.Agent``.
- ``WMOnlyAgent.wm_modules`` — list of [enc, dyn, dec, rew, con]; 5
  modules. The 4 LOSS TERMS are {recon (per-image-key), dyn, rew, con};
  module count and loss-term count are deliberately distinct.
- ``WMOnlyAgent.wm_opt`` — ``embodied.jax.Optimizer`` over
  ``wm_modules`` only. Inherited ``self.opt`` (full optimizer over
  all 7 modules) is left intact but never called by the pretrain loop.
- ``WMOnlyAgent.wm_loss(carry, obs, prevact, training)`` — copies the
  upstream WM block, returns ``(loss, (carry, entries, outs, metrics))``
  with imagination + replay-value-loss code paths skipped.
- ``WMOnlyAgent.wm_train(carry, data)`` — sibling of ``Agent.train``
  that calls ``self.wm_opt(self.wm_loss, ...)`` instead of
  ``self.opt(self.loss, ...)``. Does NOT call ``self.slowval.update()``
  (no critic to track).

Phase-3 gate hookup: pretrain loop calls ONLY ``wm_train``. The
inherited ``train`` method is left unmodified but never invoked. The
spy-on-optimizer test asserts ``self.opt.step`` is never called and
``self.wm_opt.step`` fires every loop iteration.
"""
from __future__ import annotations

# Heavy DV3 / JAX deps stay lazy — laptop importability matches
# scripts/launch_pergame.py + scripts/pretrain_wm.py.

_STUB_MSG = (
    "arc3_wm.wm_only_agent: stub awaiting impl — see "
    "tests/test_wm_only_agent.py for the contract"
)


class WMOnlyAgent:
    """Subclass of ``dreamerv3.agent.Agent`` exposing a WM-only train path.

    Stub. The real class binds to ``dreamerv3.agent.Agent`` lazily inside
    ``__init_subclass_at_runtime__`` (or equivalent — exact mechanism is
    an impl detail). The contract is fixed: ``isinstance(WMOnlyAgent(...),
    dreamerv3.agent.Agent)`` must hold once the heavy stack is importable.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_STUB_MSG)

    def wm_loss(self, carry, obs, prevact, training):
        raise NotImplementedError(_STUB_MSG)

    def wm_train(self, carry, data):
        raise NotImplementedError(_STUB_MSG)
