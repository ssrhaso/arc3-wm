# Phase-4 vc33 dry-run forensic diagnosis

- **Run id (wandb)**: `qeohyn7i` in `hasofocus-university-of-the-west-of-england/arc3-wm-sprint`
- **Commit**: `7d0d17a`
- **Date**: 2026-05-13
- **Branch**: `diag/p4-vc33-dryrun`
- **Hardware**: Vast.ai A100 SXM4 40 GB (preemptible)
- **Run shape**: 500k env-steps, `--script train_eval`, warm-started from `checkpoints/pretrained-wm/v1/latest.pkl`
- **Headline metric**: RHAE = **0.0079** on vc33 (level 1 cleared once in eval; gate "RHAE > 0" passes for this single game)

Figure: `figures/p4_vc33_diagnosis.png` (and `.svg`). Notebook: `analysis/p4_vc33_dryrun_diagnosis.ipynb`.

---

## Q1 — Success timing + trajectory

First train-side success at env-step **68,438** (episode 1,328 of 9,675). 17 train episodes cleared level 1 in total; **13 of those 17 happen in a 2k-env-step burst between env-step 232,679 and 235,151** (episodes 4,546–4,592). Two stragglers at 225,753 and 269,520. **Zero successes** in the final ~230k env-steps. Rolling 1k-episode success rate peaks at **0.015 around env-step 243,989** and returns to **0.0** by end-of-training. Last-50k linear slope on the rolling rate ≈ 0 (R² undefined — flat-zero).

**Bottom line**: not monotonic. Bursty discovery followed by collapse. The policy briefly latched onto a working ACTION6 coordinate, then drifted off it.

## Q2 — Level depth across 18 eval episodes (re-framed; see Caveats)

Eval reward sums (= # level-ups per episode) over the 18 `eval_episodes.jsonl` entries: 16 zeros, 2 ones, 0 twos+. **2/18 cleared level 1; 0/18 reached level 2; deepest single eval episode = 1 level-up**.

**Bottom line**: the agent never reached level 2 in either train or eval. There is no per-level depth signal in the dry-run — the gate question for the 5070 cluster cannot lean on "is it climbing the level ladder" because so far it isn't even sticking on level 1.

## Q3 — WM loss-curve shape at end of training

Last-50k linear fit (n=6 train log rows in the window; DV3 logs train losses every ~9k env-steps):

- `train/loss/image`: slope = `-2.75e-07/step`, R² = 0.45, last value 0.136 → **-20.3% per 100k env-steps** (still mildly decreasing but noisy).
- `train/loss/dyn`: slope = `-3.93e-07/step`, R² = 0.27, last value 1.120 → **-3.5% per 100k env-steps** (essentially flat).

**Bottom line**: WM has converged on what it can model from this data; further env-steps will not buy meaningful additional WM accuracy. Image-recon shows a faint downward drift, dynamics is flat. The model is not the bottleneck — its imagination is already as good as it's going to get on this run.

## Q4 — Exploration → exploitation (proxy; see Caveats)

`train/rand/action` (DV3's fraction-random rate) starts at 1.000 (fully random), leaves the ≥0.98 initial plateau at **env-step 235,376** — *exactly when the success burst ends* — drops below 0.5 at **env-step 315,472**, and ends the run at **0.603** with a min of 0.337. The agent is still ~60% random at 500k env-steps.

**Bottom line**: the exploration→exploitation schedule never finishes within 500k env-steps. The agent found a winning click during the 235k burst but then got pulled back into random behaviour as it didn't yet have enough non-random gradient steps to consolidate. Per-action entropy is not in DV3's default logs; flag.

## Q5 — Action mask transitions across levels

Across **all step rows in all 10 vc33 human replays** (5,448 step rows total), the `available_actions` field is invariantly `[6]` (ACTION6 only). ACTION1–5 and ACTION7 (undo) are masked on every step of every level, including the WIN row. Effective flat action space = 4096 (the 64×64 coord grid).

**Bottom line**: the action mask is NOT a confound. The agent is searching a 4096-action space identically at level 1, 2, 3, … 7. Whatever's failing in Q1/Q2 is not "the policy doesn't know that ACTION7 became available".

## Q6 — Phase-3 pretrain bias audit (vc33 subset)

Distribution of `max(levels_completed)` across vc33's 10 human replays: {0: 1, 2: 1, 3: 1, 6: 1, 7: 6}. Six of ten replays cleared all 7 levels. `win_levels = 7`. Step-row counts per level seen during pretraining: L0=138, L1=207, L2=887, L3=682, L4=996, L5=301, L6=1,319, L7=6 — heavy tail on the *deeper* levels (L4–L6), not on L1.

**Bottom line**: pretrain was NOT L1-skewed. The WM has seen plenty of L2+ vc33 transitions. Level-stratified re-sampling would not change what the WM knows.

---

## Recommendations

### (a) Extend budget beyond 500k? **No.**

Q3 says the WM has plateaued (`train/loss/dyn` at -3.5%/100k, R²=0.27 — within noise). Q1 says successes died at env-step 235k and the rolling rate has been zero for the final 230k env-steps. Adding env-steps without changing the actor-learning regime would burn 5070-hours rolling out a policy that's not consolidating. Phase-4 proper at 500k × 2 seeds is the right wall-clock budget; the **policy schedule** is what needs to change.

### (b) Change `train_ratio` (default 32) or action entropy schedule? **Yes, surface to Haso.**

Q4 is the smoking gun. `train/rand/action` ending at 0.603 means the policy is doing ~60% random clicks at env-step 500k. With `train_ratio=32`, the actor gets ~one gradient update per 32 env-steps; over 500k env-steps that's ~15k actor updates, which DV3 typically expects to be enough on Crafter-class envs but appears insufficient here when the action space is 4096-wide. Two plausible interventions:

1. **Raise `train_ratio` to 64–128** for the per-game pilots, so the actor catches up with the WM while keeping wall-clock cost the same (each env-step now triggers more imagination rollouts; WM is frozen-ish anyway per Q3 so the extra training-compute spend goes to the actor).
2. **Cut the action-entropy floor faster** (DV3's `actor.ent.scale` schedule) so exploration decays on a timescale that matches the per-game 500k budget rather than DV3's default million-step budget.

Both are deviations from stock DV3 hyperparameters and live under §"Decisions Haso owns" item 2 (and item 5 for the entropy schedule, which borders on "DreamerV3 modification"). **Recommend asking Haso, with `train_ratio=64` as the cheaper / more defensible first move** since it doesn't touch the loss formulation.

### (c) Re-pretrain WM with level-stratified replay sampling? **No.**

Q6 + Q5 together rule this out. The pretrain corpus already covers L1–L7 well (only 207 step rows on L1; ~3,200 on L2+); the agent's failure mode is not "WM doesn't know what level 2 looks like" — it's "actor never reaches level 2 to test the WM there". And Q5 shows the action space is constant across levels, so the actor doesn't even have a new affordance to learn for L2+. Re-pretrain would burn another ~6 GPU-h on Vast.ai for no expected gain.

---

## Caveats

- **Q2 re-framed.** Original brief said "17 successful training episodes — pull from `eval_episodes.jsonl`". But `EvalRewardSink` only wraps the EVAL env (see `arc3_wm/eval_reward_sink.py` and the `--script train_eval` branch in `scripts/launch_pergame.py`). Train successes only have the binary aggregate `episode/score` in `metrics.jsonl` — no per-step rewards stream. So Q2 is answered for the 18 eval episodes, which is the per-level depth signal that's actually in the artifact.
- **Q4 partially unanswerable.** DV3 logs `train/rand/action` (fraction-random rate) but not per-step actions and not a per-action histogram, so true policy entropy cannot be computed from the logged data. `train/rand/action` is the closest proxy.
- **Q5 from documentation+replays, not from `arc_agi` runtime.** The brief allowed falling back to docs if OFFLINE init required network. Since every vc33 replay carries the `available_actions` field on every step row, that's a stronger signal than a single live query — and confirmed invariant across all 5,448 step rows in the 10 replays.
- **Q3 noise.** Only 6 train log rows fall in the last 50k env-steps window (DV3 logs every ~9k env-steps at this `train_ratio`). R² of 0.45 / 0.27 reflects that, not a misclassification. The qualitative conclusion ("WM has plateaued") is robust to the noise.
- **No new training runs were performed.** Read-only analysis of B2-mirrored artifacts.

## Artifacts

- `analysis/p4_vc33_dryrun_diagnosis.ipynb` — re-runnable notebook (assumes `scratch/p4-vc33-dryrun/{metrics,eval_episodes}.jsonl` + `scratch/p4-vc33-dryrun/data/replays/vc33/` present)
- `analysis/build_p4_vc33_diagnosis.py` — script that rebuilds the notebook + figure from one source of truth
- `figures/p4_vc33_diagnosis.png` (200 DPI) and `figures/p4_vc33_diagnosis.svg` — 6-panel figure
- `scratch/p4-vc33-dryrun/` — B2-mirrored artifacts (gitignored)
