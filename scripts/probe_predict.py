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


def _make_probe_agent_class(context_len: int, mode: str = "rollout"):
    """Agent subclass whose ``report`` runs a frozen-WM probe forward.

    ``mode``:
      * ``rollout`` - observe a C-frame context, imagine H steps under the real
        actions, decode -> predicted frames (Probe B).
      * ``latents`` - observe the FULL sequence, dump the RSSM model-state
        (deterministic ``deter`` + stochastic ``stoch``) per frame, no decode
        (step-1 latent probe input).

    ``context_len`` and ``mode`` are baked in as class attributes so they are
    Python-static at trace time (the split index / branch cannot be traced).
    """
    from dreamerv3.agent import Agent

    class ProbeAgent(Agent):
        _PROBE_CONTEXT_LEN = int(context_len)
        _PROBE_MODE = str(mode)

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

            if self._PROBE_MODE == "latents":
                # Observe the FULL sequence into the RSSM posterior; dump the
                # model-state (deterministic h + stochastic z) per frame. No
                # imagine, no decode - this is the step-1 latent-probe input.
                enc_carry, _, tokens = self.enc(enc_carry, obs, reset, training=False)
                dyn_carry, _, feat = self.dyn.observe(
                    dyn_carry, tokens, prevact, reset, training=False)
                metrics = {}
                if isinstance(feat, dict):
                    for fk in ("deter", "stoch"):
                        if fk in feat:
                            metrics[f"probe/{fk}"] = feat[fk]
                if not metrics:  # fallback if feat isn't a {deter,stoch} dict
                    metrics["probe/feat"] = self.feat2tensor(feat)
                carry = (enc_carry, dyn_carry, dec_carry,
                         {k: data[k][:, -1] for k in self.act_space})
                return carry, metrics

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
        *extra_flags,
    ]
    args, leftover = _pa(argv)
    return _bc(args, leftover)


def make_probe_agent(game: str, context_len: int, extra_flags: list[str],
                     mode: str = "rollout"):
    import elements
    from scripts.launch_pergame import build_config as _  # ensure importable

    config = build_config(game, context_len, extra_flags)
    obs_space, act_space = build_spaces()
    AgentCls = _make_probe_agent_class(context_len, mode)
    # Disable AOT precompile: it bakes report for (batch_size, report_length);
    # our probe calls report with many different (B, T) shapes, so we want the
    # plain jit that retraces per shape instead. precompile is a JAX Options
    # field absent from configs.yaml, so inject it into the jax sub-config.
    jax_cfg = {**dict(config.jax), "precompile": False}
    agent = AgentCls(
        obs_space, act_space,
        elements.Config(
            **config.agent, logdir=config.logdir, seed=config.seed,
            jax=jax_cfg, batch_size=config.batch_size,
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


def run_report(agent, batch: dict) -> dict:
    """Call the agent's report path on one batch; return the raw metrics dict.

    Shared by all probe modes - the per-mode ``report`` override decides which
    ``probe/*`` arrays come back (rollout: pred/true/context_last; latents:
    deter/stoch). Returned arrays are numpy (the outer report device_gets them).

    The outer report asserts ``data.keys() == self.spaces.keys()`` (obs + act +
    ext). ext_space includes replay-context entries (``dyn/deter``, ``dyn/stoch``)
    our batch omits; our override does its own observe and ignores them, so we
    fill any missing space key with zeros of the declared shape/dtype.
    """
    from embodied.jax import internal
    B, T = batch[IMG_KEY].shape[:2]
    data = dict(batch)
    for k, sp in agent.spaces.items():
        if k not in data:
            data[k] = np.zeros((B, T, *tuple(sp.shape)), sp.dtype)
    # Place inputs on the train sharding before the jit (mirrors stream()):
    # report's _report expects sharded data/carry, not host numpy.
    data = internal.device_put(data, agent.train_sharded)
    data["seed"] = agent._seeds(0, agent.train_mirrored)
    carry = agent.init_report(B)
    _, mets = agent.report(carry, data)
    return mets


def run_forward(agent, batch: dict):
    """Rollout-mode convenience: return (pred, true, context_last) numpy arrays."""
    mets = run_report(agent, batch)
    return (np.asarray(mets["probe/pred"]), np.asarray(mets["probe/true"]),
            np.asarray(mets["probe/context_last"]))


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


def predict_counterfactual(agent, holdout_npz: Path, *, context_len, target_a,
                           n_click, max_specs, seed, batch_size):
    """Probe A: per-action one-step predictions, via the rollout forward at H=1.

    For each branch state we observe the last ``context_len`` frames, then for
    each of ``target_a`` candidate actions run a 1-step open-loop imagine+decode
    (the exact rollout path with horizon=1) - so no new JAX code, just one
    rollout batch per action slot. Needs per-frame availability => random source.
    The scorer (`score_counterfactual`) reads the resulting cf npz.
    """
    from arc3_wm.probe_data import RolloutWindow, make_counterfactual_specs

    d = dict(np.load(holdout_npz, allow_pickle=True))
    game, source = str(d["game"]), str(d["source"])
    specs = make_counterfactual_specs(
        d, context_len=context_len, n_click=n_click, max_specs=max_specs, seed=seed,
        require_change=True, allow_no_avail=True)  # state-changing filter; human-source OK
    kept = [s for s in specs if s.candidate_actions.shape[0] >= target_a]
    base = {"game": np.array(game), "source": np.array(source),
            "target_a": np.array(target_a), "n_specs": np.array(len(kept))}
    if not kept:
        print(f"  [{game}/{source}] no counterfactual specs (needs avail/random source)")
        return {
            "cf_pred": np.zeros((0, target_a, OBS_HW, OBS_HW, 3), np.uint8),
            "cf_true_next": np.zeros((0, OBS_HW, OBS_HW, 3), np.uint8),
            "cf_context": np.zeros((0, OBS_HW, OBS_HW, 3), np.uint8),
            "cf_actions": np.zeros((0, target_a), np.int32),
            "cf_taken_idx": np.zeros((0,), np.int32), **base,
        }
    N = len(kept)
    cf_pred = np.zeros((N, target_a, OBS_HW, OBS_HW, 3), np.uint8)
    for j in range(target_a):
        # The candidate action must drive the SINGLE imagined step. In the report
        # override the imagined frame at offset 0 (the only one at H=1) is decoded
        # from prevact[:, C] = action[:, C-1] -- i.e. the *last context action*
        # slot, NOT the future slot action[:, C]. Putting the candidate in
        # future_actions (action[:, C]) leaves it as prevact[:, C+1], a horizon-2
        # driver H=1 never consumes, so every candidate decoded the identical frame
        # under the REAL last action (the action-blindness was this off-by-one, not
        # the model). Fix: overwrite the last context action with the candidate and
        # leave the (unused) future slot a dummy.
        ctx_act = [s.context_actions[-context_len:].copy() for s in kept]
        for ca, s in zip(ctx_act, kept):
            ca[-1] = s.candidate_actions[j]
        windows = [
            RolloutWindow(
                ep_id=s.ep_id, start=s.t + 1,
                context_frames=s.context_frames[-context_len:],
                context_actions=ca,
                future_actions=np.zeros(1, np.int32),  # unused at H=1 (see above)
                true_future=s.true_next[None], context_last=s.context_last)
            for ca, s in zip(ctx_act, kept)
        ]
        for b0 in range(0, N, batch_size):
            chunk = windows[b0:b0 + batch_size]
            p, _, _ = run_forward(agent, windows_to_batch(chunk, context_len, 1))
            cf_pred[b0:b0 + len(chunk), j] = p[:, 0]
        print(f"  [{game}/{source}] cf action-slot {j+1}/{target_a} done")
    return {
        "cf_pred": cf_pred,
        "cf_true_next": np.stack([s.true_next for s in kept]).astype(np.uint8),
        "cf_context": np.stack([s.context_last for s in kept]).astype(np.uint8),
        "cf_actions": np.stack([s.candidate_actions[:target_a] for s in kept]).astype(np.int32),
        "cf_taken_idx": np.array([s.taken_idx for s in kept], np.int32), **base,
    }


def self_test(game, context_len, horizon, extra_flags=None):
    """Build agent + run one random batch to shake out shapes/JIT (no ckpt)."""
    agent, _ = make_probe_agent(game, context_len, extra_flags or [])
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
    p.add_argument("--mode", choices=["rollout", "counterfactual"], default="rollout",
                   help="rollout = Probe B (multi-step fidelity); counterfactual "
                        "= Probe A (per-action 1-step; needs random source).")
    p.add_argument("--context-len", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-a", type=int, default=4, help="counterfactual: candidate actions")
    p.add_argument("--n-click", type=int, default=8, help="counterfactual: sampled ACTION6 cells")
    p.add_argument("--max-specs", type=int, default=None, help="counterfactual: cap states")
    p.add_argument("--seed", type=int, default=0, help="counterfactual candidate sampling seed")
    p.add_argument("--platform", choices=["cpu", "cuda"], default=None,
                   help="override jax.platform (default: config's cuda). Use cpu "
                        "for login-node validation without a GPU.")
    p.add_argument("--compute-dtype", default=None,
                   help="override jax.compute_dtype. Use float32 on aarch64 CPU: "
                        "the config default bfloat16 hits an LLVM AArch64 codegen "
                        "bug (Cannot select nxv4bf16).")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("flags", nargs="*", help="extra config key=value overrides")
    return p.parse_args(argv)


def _resolve_flags(args) -> list[str]:
    flags = list(args.flags)
    if args.platform:
        flags = ["--jax.platform", args.platform] + flags
    if getattr(args, "compute_dtype", None):
        flags = ["--jax.compute_dtype", args.compute_dtype] + flags
    return flags


def main(argv=None) -> int:
    args = parse_args(argv)
    flags = _resolve_flags(args)
    if args.self_test:
        self_test(args.game, args.context_len, args.horizon, flags)
        return 0
    if not (args.ckpt and args.holdout):
        print("need --ckpt and --holdout (or --self-test)")
        return 2
    agent, _ = make_probe_agent(args.game, args.context_len, flags)
    restore_checkpoint(agent, Path(args.ckpt))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "counterfactual":
        out = predict_counterfactual(
            agent, Path(args.holdout), context_len=args.context_len,
            target_a=args.target_a, n_click=args.n_click, max_specs=args.max_specs,
            seed=args.seed, batch_size=args.batch_size)
        dest = outdir / f"{args.game}_{args.source}_cf.npz"
        np.savez_compressed(dest, **out)
        print(f"[probe_predict] wrote {dest} :: {out['cf_pred'].shape[0]} states")
    else:
        out = predict_rollout(
            agent, Path(args.holdout), context_len=args.context_len,
            horizon=args.horizon, max_windows=args.max_windows,
            batch_size=args.batch_size)
        dest = outdir / f"{args.game}_{args.source}_rollout.npz"
        np.savez_compressed(dest, **out)
        print(f"[probe_predict] wrote {dest} :: {out['rb_pred'].shape[0]} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
