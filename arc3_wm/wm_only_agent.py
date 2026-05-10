"""arc3_wm.wm_only_agent — WM-only DreamerV3 Agent subclass.

Subclass of upstream ``dreamerv3.agent.Agent`` that overrides ``loss``
and ``train`` so the inherited DreamerV3 pipeline runs the WM-only
path instead of the full imagination + actor + critic path.
``third_party/dreamerv3/`` stays untouched (CLAUDE.md anti-goal +
D12: "DreamerV3 source is unmodified — env registered via embodied/,
no fork").

Why an override of ``loss``/``train`` rather than a parallel
``wm_train`` method: ``embodied.jax.Agent.__new__`` (third_party/
dreamerv3/embodied/jax/agent.py:38-48) hardcodes
``super().__new__(Agent)`` for the outer wrapper and JIT-compiles
``transform.apply(nj.pure(self.model.train), ...)`` at construction.
The outer wrapper exposes only ``train``/``policy``/``report``; any
new method on the model class is unreachable via the outer's JIT
pipeline. So the WM-only path has to live on a method the outer
already knows about — ``train`` (which calls ``loss``).

Lazy parent (PEP 562 ``__getattr__``):
- Module load itself imports nothing from ``dreamerv3`` / ``embodied`` /
  ``jax``. ``tests/test_wm_only_agent.py::test_module_imports_without_jax``
  asserts none of those modules appear in ``sys.modules`` after this
  module loads on a laptop.
- ``WMOnlyAgent`` is materialised on first attribute access via the
  ``_build_wm_only_agent_class()`` factory. On Vast (full DV3 stack
  installed) the factory inherits from ``dreamerv3.agent.Agent``. On a
  laptop the dreamerv3 import raises ``ImportError`` (chex / jax are
  missing); the factory falls back to ``object`` so the class is still
  importable but cannot be instantiated. ``inspect.getsource(...)``
  works in both cases — the methods are defined on the class regardless
  of which parent is in effect.

Phase-3 contract — see ``tests/test_wm_only_agent.py`` and
``tests/test_pretrain_wm.py`` concern-group #4 for the binding tests:

- ``WMOnlyAgent.loss`` returns ``(loss, (carry, entries, outs,
  metrics))`` where ``outs['losses']`` contains only
  ``{recon-key(s), dyn, rew, con}``. No imagination, no replay-value,
  no policy/value losses.
- ``WMOnlyAgent.train`` is upstream ``Agent.train`` minus
  ``self.slowval.update()`` — that's the only material delta.
- ``self.modules`` and ``self.opt`` are rebuilt in ``__init__`` over
  ``[dyn, enc, dec, rew, con]`` only. Pol and val instances still exist
  on the model (they're constructed by the parent ``__init__``) but
  are absent from the optimizer's trainable set.
- 4 LOSS TERMS (recon-per-image-key, dyn, rew, con) vs 5 MODULES
  receiving gradients. Module count and loss-term count are
  deliberately distinct.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# third_party/dreamerv3 path — used only on first WMOnlyAgent access.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
if _DV3.is_dir() and str(_DV3) not in sys.path:
    sys.path.insert(0, str(_DV3))

# Module-level cache for the lazily-constructed class.
_wm_only_agent_class: Any = None


def _build_wm_only_agent_class():
    """Construct ``WMOnlyAgent`` with the right parent class.

    On Vast (full DV3 stack): inherits from ``dreamerv3.agent.Agent``.
    On laptop (no JAX): ``ImportError`` is caught and the class falls
    back to ``object`` parent — importable for source inspection but
    not instantiable.
    """
    try:
        from dreamerv3.agent import Agent as _Parent
        _laptop_fallback = False
    except ImportError:  # noqa: BLE001 — laptop path is intentionally tolerant
        _Parent = object
        _laptop_fallback = True

    class WMOnlyAgent(_Parent):
        """WM-only DreamerV3 Agent. See module docstring."""

        # Marker attribute the public-surface test reads to confirm the
        # class came from the factory, not a dummy stub.
        _WM_ONLY = True

        def __init__(self, *args, **kwargs):
            if _laptop_fallback:
                raise RuntimeError(
                    "WMOnlyAgent cannot be instantiated without the dreamerv3 "
                    "JAX stack. Run on Vast or in an env with JAX + chex + "
                    "ninjax + optax installed."
                )
            super().__init__(*args, **kwargs)
            # Restrict the optimizer to WM modules only. Pol and val
            # instances still exist as self.pol / self.val (they were
            # constructed by parent __init__); we just keep them out of
            # the trainable set so no gradients are applied to them.
            import embodied
            self.modules = [self.dyn, self.enc, self.dec, self.rew, self.con]
            self.opt = embodied.jax.Optimizer(
                self.modules,
                self._make_opt(**self.config.opt),
                summary_depth=1,
                name='opt',
            )

        def loss(self, carry, obs, prevact, training):
            """Verbatim copy of upstream ``Agent.loss`` (agent.py:156-186) —
            the World-model block. Branches BEFORE the imagination block at
            agent.py:188 so neither ``self.dyn.imagine`` nor ``imag_loss``
            nor ``repl_loss`` runs.
            """
            import jax
            import jax.numpy as jnp
            f32 = jnp.float32
            sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
            # NB: matches upstream agent.py:21 isimage definition.
            isimage = lambda s: s.dtype == 'uint8' and len(s.shape) == 3

            enc_carry, dyn_carry, dec_carry = carry
            reset = obs['is_first']
            B, T = reset.shape
            losses = {}
            metrics = {}

            # World model — same flow as upstream agent.py:163-186.
            enc_carry, enc_entries, tokens = self.enc(
                enc_carry, obs, reset, training)
            dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
                dyn_carry, tokens, prevact, reset, training)
            losses.update(los)
            metrics.update(mets)
            dec_carry, dec_entries, recons = self.dec(
                dec_carry, repfeat, reset, training)
            inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
            losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
            con = f32(~obs['is_terminal'])
            if self.config.contdisc:
                con *= 1 - 1 / self.config.horizon
            losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
            for key, recon in recons.items():
                space, value = self.obs_space[key], obs[key]
                assert value.dtype == space.dtype, (key, space, value.dtype)
                target = f32(value) / 255 if isimage(space) else value
                losses[key] = recon.loss(sg(target))

            # END of WM block. Imagination + replay-value-loss are skipped
            # entirely — that's the entire point of this override.

            metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
            # Sum only the WM scales; pol/val scales are silently absent
            # because their loss keys are absent from `losses`.
            loss = sum(v.mean() * self.scales[k] for k, v in losses.items())

            carry = (enc_carry, dyn_carry, dec_carry)
            entries = (enc_entries, dyn_entries, dec_entries)
            outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
            return loss, (carry, entries, outs, metrics)

        def train(self, carry, data):
            """Mirror of upstream ``Agent.train`` (agent.py:137-154) MINUS
            ``self.slowval.update()``. That single line is the slow-critic
            bookkeeping; on the WM-only path it must not fire (val isn't
            in self.modules and isn't being trained).
            """
            import elements
            carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
            metrics, (carry, entries, outs, mets) = self.opt(
                self.loss, carry, obs, prevact, training=True, has_aux=True)
            metrics.update(mets)
            # NB: deliberately omit the slow-critic update from upstream
            # train (last line of agent.py:137-154) — that's the
            # pol/val-side bookkeeping we want to skip.
            outs = {}
            if self.config.replay_context:
                updates = elements.tree.flatdict(dict(
                    stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
                B, T = obs['is_first'].shape
                assert all(x.shape[:2] == (B, T) for x in updates.values()), (
                    (B, T), {k: v.shape for k, v in updates.items()})
                outs['replay'] = updates
            carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
            return carry, outs, metrics

    return WMOnlyAgent


def __getattr__(name: str):
    """PEP 562 lazy attribute resolution. ``WMOnlyAgent`` materialises
    on first access; module load itself touches no DV3 / JAX imports."""
    global _wm_only_agent_class
    if name == "WMOnlyAgent":
        if _wm_only_agent_class is None:
            _wm_only_agent_class = _build_wm_only_agent_class()
        return _wm_only_agent_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
