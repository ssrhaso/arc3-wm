# HANDOFF: World Model Selection for Sparse-Reward ARC-AGI-3

## Purpose

Decide which mechanism, if any, addresses the sparse-reward failure that keeps DreamerV3 at RHAE=0 on most Phase-4 games, and whether TWISTER (Burchi, AC-CPC plus Transformer SSM) is the right tool for it.

## TL;DR

1. The Phase-4 failure is reward cold-start (no positive examples in the buffer), not weak world-model representation. cd82 has the tightest world model in the sweep and still scores 0.
2. TWISTER improves representation quality, which is the axis we have already shown to be orthogonal to escape. It is therefore likely aimed one axis away from the actual wall.
3. Run one cheap probe (reward linear-decodability) before committing to any port. It decides whether TWISTER's lever is even relevant.
4. If the probe confirms scarcity, the on-mechanism fix is state-change reward shaping, applied env-side with no DreamerV3 fork.
5. Explore Plan2Explore (Dreamerv3), mini experiments and probe

## Status at a glance

| Item                       | State                                                  |
| -------------------------- | ------------------------------------------------------ |
| Reachability (real env)    | Probed. Random policy gets 0/300 clears on vc33.       |
| Reachability (imagination) | Probed. Imagined return identically 0 on 5 dead games. |
| Reward linear-decodability | Not yet probed. This is the decision gate.             |
| State-change shaping       | Designed, not implemented.                             |
| TWISTER port               | Not started. Gated on the probe.                       |

## Evidence to date

Source: 24-run wandb analysis (2026-05-25), sharpened 2026-06-17, plus the counterfactual dynamics probe (2026-06-19).

**Real-env reachability.** `scripts/eval_random_rhae.py` ran a uniform-random agent through the real env: vc33 scored 0/300 L1 clears (an independent 832-episode varying-seed run also scored 0). On the 5 dead games (cd82, sb26, tn36, ls20, lf52) the trained actor stays at rand/action about 1.000 for the full 500k steps with episode score identically 0, so those runs behave as random-policy runs and never reach reward online. vc33 is the only game that scores, via 17 chance clears while partially committed.

**Imagination reachability.** On all 5 dead games the imagined return is identically 0 (ret_min equals ret_max equals 0, reward-head loss about 0). The reward head fits by predicting all-zero because the buffer contains no clears, so there are no positive examples for any head to learn from.

**World-model fit is orthogonal to escape.** cd82 has the tightest world model in the sweep (image loss 0.065 versus vc33 0.16), the most action-sensitive dynamics, and pixel-robust latent level structure, yet still scores RHAE=0. A better world model is exactly what cd82 already has, and it changed nothing.

**Dynamics competence.** The counterfactual probe (after the alignment-bug fix, commit ad1ee4b) shows the world model is weakly action-conditional but not action-correct: the prediction moves with the action, but the taken action's true consequence is not singled out and rarely beats copy-last-frame. Defensible claim: a weak, partial world model the controller never exploits.

**Conclusion.** The failure is reward cold-start and hard exploration, not representation capacity.

## Decision gate: reward linear-decodability probe

The one test not yet run, and the one that decides whether TWISTER is relevant.

**Question.** If handed a real positive, can the current latent even represent the reward?

**Method.**

1. Freeze the Phase-3 (or a Phase-4) world-model checkpoint.
2. Take held-out human level-up transitions (available via the `data/human_baselines.json` pipeline and `arc3_wm.replay_loader`).
3. Encode to RSSM latents. Fit a linear probe from latent to level-up label.
4. Report decodability (accuracy or AUC) versus a shuffled-label control.

**Interpretation.**

- Decodable: the representation is adequate; reward learning fails only because positives are scarce or never generated. TWISTER will not help. Proceed to shaping or exploration.
- Not decodable even from human positives: a genuine representation gap the reward head cannot bridge. TWISTER's AC-CPC lever is on-target and worth the port.

**Cost.** About one day. Reuses the probe harness in `arc3_wm/dynamics_probe.py` as a template. No training and no new infrastructure.

**Prior.** Given cd82 (tightest fit, still 0), the expected outcome is "decodable", which points away from TWISTER.

## Fallback intervention: state-change reward shaping

The on-mechanism fix if the probe confirms scarcity. This is the StochasticGoose signal named in the CLAUDE.md fallback ladder.

**Idea.** Native reward is delta-levels and fires almost never. A meaningful ARC-3 action changes the grid; a no-op or rejected action does not. Use grid change as a dense proxy for "you affected the world":

```
r_shaped = r_native + beta * change_signal(obs_t, obs_prev)
```

**change_signal options, cheapest to safest.**

- Binary: 1 if obs changed else 0.
- Magnitude: fraction of cells changed (Hamming distance over 4096), normalized.
- Novelty-gated: reward only unseen states (hash the grid, reward inversely to visit count). This is count-based exploration and resists flicker-farming.

**Why it targets the diagnosed mechanism.** The actor-collapse analysis states the falsifiable prediction directly: injecting one positive reward example (chance clear, Plan2Explore intrinsic, or state-change shaping) breaks the entropy-only symmetry on exactly the zero-return games, leaving world-model fit unchanged. Dense change-reward manufactures positives, which seed the reward head, which yields return variance, critic value, and a real actor gradient.

**How to bolt on without forking DreamerV3.** Apply it env-side as an `embodied` wrapper, the same pattern as `arc3_wm/eval_reward_sink.py` (which duck-types `embodied.core.wrappers.Wrapper`). DreamerV3 stays untouched and simply sees a modified reward channel.

```python
class StateChangeRewardWrapper(Wrapper):   # duck-types embodied Wrapper, like EvalRewardSink
    def __init__(self, env, beta=0.01, mode="hamming", novelty_gate=True): ...
    def step(self, action):
        obs = self.env.step(action)
        delta = self._change(obs["image"], self._prev)
        obs["reward"] = obs["reward"] + self.beta * delta
        self._prev = obs["image"]
        if obs["is_last"]:
            self._prev = None
        return obs
```

Wrap the training env factory only. Keep the eval env native so RHAE stays honest (RHAE is post-hoc on native progress). beta, mode, and novelty_gate are config knobs.

**Risks and options.**

- Reward hacking: raw delta-reward lets the agent farm flicker. The novelty-gated variant removes this and is the honest default.
- Potential-based shaping (Ng 1999), `r' = r + gamma * Phi(s') - Phi(s)`, provably preserves the optimal policy, which is cleaner for the paper but weaker at cold-start ignition. Non-potential change-reward is stronger for ignition but changes the optimum. Novelty-gating is the middle ground.
- Paper framing: sparse native reward for eval and RHAE; dense interaction bonus during training only.

## TWISTER assessment

TWISTER (Burchi, github.com/burchim/TWISTER) is a standalone PyTorch codebase built on DreamerV3. Core method is action-conditioned Contrastive Predictive Coding (AC-CPC) inside a Transformer state-space model, evaluated on Atari 100k and DMC.

**Why "sparse-reward Atari" does not transfer directly.** The sparse games TWISTER wins (Freeway, Frostbite, Hero, Private Eye) are reward-rare-but-reachable: a stochastic actor reaches reward occasionally, and a sharper world model exploits it. The ARC-3 dead games are reward-unreachable-by-random, closer to hard exploration. No Atari 100k method, TWISTER included, manufactures the first reward.

**Integration cost.** TWISTER is a separate engine in a separate framework, not a plugin for the JAX danijar/dreamerv3 the sprint is committed to. Two routes:

- Route A: adopt the PyTorch engine wholesale. Rebuilds the env adapter, replay loader, RHAE eval plumbing, and cross-game pretraining against a new codebase, and discards the Phase-3 checkpoint (a JAX RSSM cannot load into a PyTorch TSSM). This is a reference-implementation switch (Haso-owned) plus a second world-model arm (out of scope).
- Route B: port only the AC-CPC auxiliary loss (and optionally the Transformer SSM) into the existing JAX DreamerV3. Keeps all plumbing and the Phase-3 checkpoint. AC-CPC alone is a few days; adding the Transformer SSM is 1 to 2 weeks plus retuning. If TWISTER is pursued mid-sprint at all, this is the only acceptable route.

**Recommendation.** Do not start TWISTER before the decision-gate probe. If the probe says "decodable" (expected), TWISTER is aimed one axis away from the wall and belongs in the follow-up paper, where the AC-CPC to "not action-correct" link is a strong, measured motivation. If the probe says "not decodable", TWISTER's lever becomes relevant and Route B is the path.

## Recommended sequence

1. Run the reward linear-decodability probe (about one day). Decision gate for everything below.
2. If scarcity is confirmed: prototype the novelty-gated StateChangeRewardWrapper on the 5 dead games. Cheapest path from RHAE=0 to RHAE greater than 0, in-framework, keeps the Phase-3 checkpoint and Gymnasium wrapper.
3. If shaping ignites reward on the dead games, evaluate whether it generalizes across the sweep before deciding on exploration methods (Plan2Explore).
4. Hold TWISTER as the follow-up-paper arm unless the probe returns "not decodable", in which case escalate Route B as a sprint candidate.

## Open decisions (Haso owns)

1. Reward shaping is a deliberate deviation from "env native rewards, unmodified" and a DreamerV3 modification (CLAUDE.md decision 5). Needs sign-off before implementation.
2. TWISTER is a second world-model arm and, via Route A, a reference-implementation switch (CLAUDE.md out-of-scope plus decisions 4 and 5). Needs sign-off.
3. Whether the decision-gate probe uses the Phase-3 checkpoint or a Phase-4 per-game checkpoint.

## Files for the intern

Paths are repo-relative. Read CLAUDE.md first; it is the authoritative spec (gitignored by design).

### Start here

- `arc3_wm/env.py`: Gymnasium env over arc_agi; source of the RGB grid any change signal reads.
- `arc3_wm/embodied_env.py`: ARC3EmbodiedEnv, the DreamerV3-facing env.
- `arc3_wm/action_space.py`: flat 4102 action space and per-game masking.
- `arc3_wm/registration.py`: task registration for arc3_<game>.
- `docs/using-the-wrapper.md`: how the wrapper is meant to be driven.
- `docs/design-decisions.md`: numbered design decisions, including D12 (launcher bypasses dreamerv3/main.py).
