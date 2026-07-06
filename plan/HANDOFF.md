# HANDOFF: World Model Selection for Sparse-Reward ARC-AGI-3

## Purpose

Decide which mechanism, if any, addresses the sparse-reward failure that keeps DreamerV3 at RHAE=0 on most Phase-4 games, and whether TWISTER (Burchi, AC-CPC plus Transformer SSM) is the right tool for it.

## TL;DR

1. The Phase-4 failure is reward cold-start (no positive examples in the buffer), not weak world-model representation. cd82 has the tightest world model in the sweep and still scores 0.
2. TWISTER improves representation quality, which is the axis we have already shown to be orthogonal to escape. It is therefore likely aimed one axis away from the actual wall.
3. Run one cheap probe (reward linear-decodability) before committing to any port. It decides whether TWISTER's lever is even relevant.
4. If the probe confirms scarcity, the on-mechanism fix is state-change reward shaping, applied env-side with no DreamerV3 fork.

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
