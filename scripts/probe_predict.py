"""Stage 2 of the dynamics-competence probe: frozen-WM forward pass (JAX/GPU).

RUNS ON A JAX BOX ONLY (GH200 / Vast) - imports dreamerv3 + JAX. The laptop
cannot run this; Stages 1 (collect) and 3 (score) are CPU-only and already
validated against the synthetic generator, so this script only has to fill the
same predicted-frame npz contract with REAL world-model predictions.

What it does (Probe B - multi-step rollout fidelity):
  1. Rebuild the per-game agent exactly as the run did (size12m + arc3 config),
     using explicit obs/act spaces so NO env-files are needed here.
  2. Restore the frozen per-game checkpoint (elements.Checkpoint).
  3. For each held-out rollout window: encode the C-frame context, observe it
     into the RSSM posterior, imagine H steps forward UNDER THE REAL ACTION
     SEQUENCE (open-loop, no learned policy, no peeking at future frames),
     decode each imagined latent to a frame.
  4. Write ``<game>_<source>_rollout.npz`` in the Stage-3 contract.

This mirrors the open-loop block of ``dreamerv3.agent.WorldModel.report``
(third_party/dreamerv3/dreamerv3/agent.py:273-294): observe(firsthalf) ->
imagine(secondhalf prevact) -> decode -> ``recons[key].pred()*255``. We subclass
the Agent and override ``report`` so the agent's own JIT pipeline runs our
forward (the WMOnlyAgent pattern; new methods are unreachable through the outer
JIT, overrides are not).

Counterfactual (Probe A) is a documented follow-on (see PROBE_A_TODO) - it needs
per-action 1-step imagines from a shared observed state; add once B is verified.

Usage (on the GH200, after `pip install` of the dreamerv3 JAX stack)::

    python scripts/probe_predict.py --game cd82 --source human \\
        --ckpt /path/to/extracted/ckpt_dir \\
        --holdout results/dynamics_probe/holdout/cd82_human.npz \\
        --context-len 4 --horizon 8 \\
        --outdir results/dynamics_probe/pred

    # shape/JIT shakeout without real data or ckpt restore:
    python scripts/probe_predict.py --game cd82 --self-test --context-len 4 --horizon 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DV3 = _REPO_ROOT / "third_party" / "dreamerv3"
for p in (str(_REPO_ROOT), str(_DV3)):
    if p not in sys.path:
        sys.path.insert(0, p)

# OFFLINE is harmless here (we never touch arc_agi), but keep parity.
os.environ.setdefault("OPERATION_MODE", "offline")

from arc3_wm.action_space import N_ACTIONS  # noqa: E402
from arc3_wm.probe_data import make_rollout_windows  # noqa: E402

OBS_HW = 64
IMG_KEY = "image"


# ---------------------------------------------------------------------------
# Agent construction (explicit spaces; no env-files needed)
# ---------------------------------------------------------------------------

def build_spaces():
    import elements
    obs_space = {
        IMG_KEY: elements.Space(np.uint8, (OBS_HW, OBS_HW, 3), 0, 255),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act_space = {"action": elements.Space(np.int32, (), 0, N_ACTIONS)}
    return obs_space, act_space


def _make_probe_agent_class(context_len: int):
    """Agent subclass whose ``report`` runs the open-loop rollout (split at C).

    ``context_len`` is baked in as a class attribute so it is a Python-static
    value at trace time (the split index cannot be a traced array).
    """
    from dreamerv3.agent import Agent

    class ProbeAgent(Agent):
        _PROBE_CONTEXT_LEN = int(context_len)

        def report(self, carry, data):
            # Override the MODEL's report. `self` is the model (see Agent.__new__).
            import jax
            import jax.numpy as jnp

            C = self._PROBE_CONTEXT_LEN
            obs = {k: data[k] for k in self.obs_space}
            reset = obs["is_first"]                      # (B, T)
            B, T = reset.shape
            H = T - C

            # prevact[:, t] = action that led INTO frame t (= data action at t-1),
            # matching dreamerv3.agent._apply_replay_context's `prepend`.
            prevact = {
                k: jnp.concatenate(
                    [jnp.zeros_like(data[k][:, :1]), data[k][:, :-1]], 1)
                for k in self.act_space
            }

            enc_carry, dyn_carry, dec_carry, _ = carry

            enc_carry, _, tokens = self.enc(enc_carry, obs, reset, training=False)
            fh = lambda x: jax.tree.map(lambda v: v[:, :C], x)
            sh = lambda x: jax.tree.map(lambda v: v[:, C:], x)

            # Observe the context into the posterior, then imagine H steps under
            # the real action sequence (prior rollout; never sees future frames).
            dyn_carry, _, _obsfeat = self.dyn.observe(
                dyn_carry, fh(tokens), fh(prevact), fh(reset), training=False)
            _, imgfeat, _ = self.dyn.imagine(
                dyn_carry, sh(prevact), length=H, training=False)

            dec_carry, _, imgrecons = self.dec(
                dec_carry, imgfeat, jnp.zeros((B, H), bool), training=False)

            key = self.dec.imgkeys[0]
            pred = jnp.clip(imgrecons[key].pred() * 255, 0, 255).astype(jnp.uint8)
            true = obs[key][:, C:]
            context_last = obs[key][:, C - 1]

            metrics = {
                "probe/pred": pred,            # (B, H, 64, 64, 3) uint8
                "probe/true": true,            # (B, H, 64, 64, 3) uint8
                "probe/context_last": context_last,  # (B, 64, 64, 3) uint8
            }
            carry = (enc_carry, dyn_carry, dec_carry,
                     {k: data[k][:, -1] for k in self.act_space})
            return carry, metrics

    return ProbeAgent


def build_config(game: str, context_len: int, extra_flags: list[str]):
    """Reuse the launcher's config resolution (size12m + arc3 blocks)."""
    from scripts.launch_pergame import build_config as _bc, parse_args as _pa
    argv = [
        "--logdir", "/tmp/probe_logdir",
        "--task", f"arc3_{game}",
        "--configs", "size12m", "arc3",
        # Trace report lazily so the baked-in C is read at first call, and keep
        # the run single-device/simple for a one-off forward pass.
        "--jax.precompile", "False",
        *extra_flags,
    ]
    args, leftover = _pa(argv)
    return _bc(args, leftover)


def make_probe_agent(game: str, context_len: int, extra_flags: list[str]):
    import elements
    from scripts.launch_pergame import build_config as _  # ensure importable

    config = build_config(game, context_len, extra_flags)
    obs_space, act_space = build_spaces()
    AgentCls = _make_probe_agent_class(context_len)
    agent = AgentCls(
        obs_space, act_space,
        elements.Config(
            **config.agent, logdir=config.logdir, seed=config.seed,
            jax=config.jax, batch_size=config.batch_size,
            batch_length=config.batch_length, replay_context=config.replay_context,
            report_length=config.report_length, replica=config.replica,
            replicas=config.replicas,
        ),
    )
    return agent, config


def restore_checkpoint(agent, ckpt_dir: Path):
    """Restore a full per-game checkpoint directory (elements.Checkpoint)."""
    import elements
    cp = elements.Checkpoint(directory=str(ckpt_dir))
    cp.agent = agent
    cp.load(keys=["agent"])
    return agent


# ---------------------------------------------------------------------------
# Batch assembly + forward
# ---------------------------------------------------------------------------

def windows_to_batch(windows, context_len: int, horizon: int):
    """Stack rollout windows into the standard (B, T) report data batch.

    image[:, :C]  = context frames; image[:, C:] = true future (the model never
    encodes these - they ride along only so report can return aligned truth;
    the open-loop imagine uses actions, not these frames).
    action[:, t]  = action taken AT frame t (convention B).
    """
    T = context_len + horizon
    B = len(windows)
    image = np.zeros((B, T, OBS_HW, OBS_HW, 3), np.uint8)
    action = np.zeros((B, T), np.int32)
    for i, w in enumerate(windows):
        image[i, :context_len] = w.context_frames
        image[i, context_len:] = w.true_future
        action[i, :context_len] = w.context_actions
        # action at the last context frame .. second-to-last future frame drive
        # the imagined steps; the final slot is unused (no frame after it).
        action[i, context_len:context_len + horizon] = w.future_actions
    is_first = np.zeros((B, T), bool)
    is_first[:, 0] = True
    return {
        IMG_KEY: image,
        "action": action,
        "reward": np.zeros((B, T), np.float32),
        "is_first": is_first,
        "is_last": np.zeros((B, T), bool),
        "is_terminal": np.zeros((B, T), bool),
        # ext_space (unused by our report, but required by the key assertion):
        "consec": np.zeros((B, T), np.int32),
        "stepid": np.zeros((B, T, 20), np.uint8),
    }


def run_forward(agent, batch: dict):
    """Call the agent's report path on one batch; return numpy probe outputs."""
    import jax
    B = batch[IMG_KEY].shape[0]
    carry = agent.init_report(B)
    seed = agent._seeds(0, agent.train_mirrored)
    data = {**batch, "seed": seed}
    data = jax.tree.map(lambda x: x, data)  # leave on host; outer device_puts
    _, mets = agent.report(carry, data)
    pred = np.asarray(mets["probe/pred"])
    true = np.asarray(mets["probe/true"])
    ctx = np.asarray(mets["probe/context_last"])
    return pred, true, ctx


def predict_rollout(agent, holdout_npz: Path, *, context_len, horizon,
                    max_windows, batch_size):
    d = dict(np.load(holdout_npz, allow_pickle=True))
    game, source = str(d["game"]), str(d["source"])
    windows = make_rollout_windows(
        d, context_len=context_len, horizon=horizon, max_windows=max_windows)
    if not windows:
        raise RuntimeError(f"no rollout windows for {game}/{source}")
    preds, trues, ctxs, ep_ids, starts, acts = [], [], [], [], [], []
    for s in range(0, len(windows), batch_size):
        chunk = windows[s:s + batch_size]
        batch = windows_to_batch(chunk, context_len, horizon)
        p, t, c = run_forward(agent, batch)
        preds.append(p); trues.append(t); ctxs.append(c)
        ep_ids.extend(w.ep_id for w in chunk)
        starts.extend(w.start for w in chunk)
        acts.extend(w.future_actions.astype(np.int32) for w in chunk)
        print(f"  [{game}/{source}] windows {s}..{s+len(chunk)} done")
    return {
        "rb_pred": np.concatenate(preds), "rb_true": np.concatenate(trues),
        "rb_context": np.concatenate(ctxs),
        "rb_ep_id": np.array(ep_ids, np.int32),
        "rb_start": np.array(starts, np.int32),
        "rb_actions": np.stack(acts),
        "game": np.array(game), "source": np.array(source),
        "horizon": np.array(horizon),
    }


def self_test(game, context_len, horizon):
    """Build agent + run one random batch to shake out shapes/JIT (no ckpt)."""
    agent, _ = make_probe_agent(game, context_len, [])
    B, T = 2, context_len + horizon
    batch = {
        IMG_KEY: np.random.randint(0, 256, (B, T, OBS_HW, OBS_HW, 3), np.uint8),
        "action": np.random.randint(0, N_ACTIONS, (B, T), np.int32),
        "reward": np.zeros((B, T), np.float32),
        "is_first": np.zeros((B, T), bool),
        "is_last": np.zeros((B, T), bool),
        "is_terminal": np.zeros((B, T), bool),
        "consec": np.zeros((B, T), np.int32),
        "stepid": np.zeros((B, T, 20), np.uint8),
    }
    batch["is_first"][:, 0] = True
    pred, true, ctx = run_forward(agent, batch)
    print(f"[self-test] pred {pred.shape} true {true.shape} ctx {ctx.shape}")
    assert pred.shape == (B, horizon, OBS_HW, OBS_HW, 3), pred.shape
    print("[self-test] OK - shapes line up; ready for ckpt + real data")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--source", default="human")
    p.add_argument("--ckpt", help="extracted checkpoint directory")
    p.add_argument("--holdout", help="Stage-1 holdout npz")
    p.add_argument("--outdir", default="results/dynamics_probe/pred")
    p.add_argument("--context-len", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("flags", nargs="*", help="extra config key=value overrides")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(args.game, args.context_len, args.horizon)
        return 0
    if not (args.ckpt and args.holdout):
        print("need --ckpt and --holdout (or --self-test)")
        return 2
    agent, _ = make_probe_agent(args.game, args.context_len, list(args.flags))
    restore_checkpoint(agent, Path(args.ckpt))
    out = predict_rollout(
        agent, Path(args.holdout), context_len=args.context_len,
        horizon=args.horizon, max_windows=args.max_windows,
        batch_size=args.batch_size)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{args.game}_{args.source}_rollout.npz"
    np.savez_compressed(dest, **out)
    print(f"[probe_predict] wrote {dest} :: {out['rb_pred'].shape[0]} windows")
    return 0


# PROBE_A_TODO (counterfactual / action-sensitivity), to add once B is verified:
#   - observe a fixed C-frame context per state -> dyn_carry
#   - for each candidate action a in the spec: self.dyn.imagine(dyn_carry,
#     {'action': a[:, None]}, length=1) -> decode -> per-action 1-step frame
#   - assemble with arc3_wm.probe_data.build_counterfactual_prediction_npz
#   This reuses the same agent/override; only the batch + imagine loop differ.

if __name__ == "__main__":
    raise SystemExit(main())
