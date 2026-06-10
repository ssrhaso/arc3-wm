# ls20 / lf52 stuck-policy audit (read-only, logs only)

**Scope.** 8 runs: `{ls20, lf52} × {warm, cold} × {seed 0, seed 1}`, 500k env-steps each.
**Goal.** Characterise *why* these two policies are stuck, from logged evidence alone, to
harden the random-policy / controller-bottleneck diagnosis before deciding whether the
masked ls20/lf52 re-run is still required. **No training, no paper edits, no B2 writes.**

## Source of truth

W&B was tried first and is **not usable** for this sweep under the available credentials:
- `wandb.Api().runs("hasofocus-university-of-the-west-of-england/arc3-wm-sprint")` →
  `ValueError: Could not find project arc3-wm-sprint`.
- `api.projects("hasofocus-university-of-the-west-of-england")` → **`[]`** (0 projects).
- The only project visible to these credentials is `haso-university-of-the-west-of-england/nnet`
  (133 runs — the ECHELON project, unrelated).

→ Fell back to **B2 bucket `arc-agi-3-replays-hasaan`** (read-only app key, confirmed via
`b2 account get`). Run artifacts:

| arm | B2 prefix |
|-----|-----------|
| warm | `phase4-proper/p4-{game}-s{seed}-warm-98de390/` |
| cold | `phase4-fromscratch/p4-fromscratch-{game}-s{seed}-a06c02f/` |

The 24 small files (`metrics.jsonl`, `eval_episodes.jsonl`, `launch.log` × 8 runs) were
downloaded to `scratch/ls20_lf52_audit/`. Each run also has a `ckpt-final.tar.gz` (not pulled).
Reproduce numbers with `python analysis/ls20_lf52_audit.py`.

---

## 1. Metric-key inventory (Step 1)

Confirmed by enumerating **every** distinct key across a full `metrics.jsonl`
(`phase4-proper/p4-ls20-s0-warm-98de390/metrics.jsonl`, 557 lines, step 0 → 496,496).
**61 keys.** Grouped:

- **Episode (per finished episode):** `episode/score`, `episode/length`.
- **Reward/return rate:** `epstats/reward_rate`, `train/rew`, `train/ret`, `train/ret_max`,
  `train/ret_min`, `train/ret_rate`.
- **Policy / exploration:** `train/ent/action` (policy entropy), `train/rand/action`
  (random-action fraction), `train/loss/policy`, `train/adv`, `train/adv_mag`, `train/adv_std`.
- **Critic / value:** `train/val`, `train/slowval`, `train/tar`, `train/loss/value`,
  `train/loss/repval`, `train/weight`.
- **World model:** `train/loss/image` (recon), `train/loss/dyn`, `train/loss/rep`,
  `train/loss/con`, `train/loss/rew`, `train/dyn_ent`, `train/rep_ent`, `train/con`, `train/rew`.
- **Optimiser:** `train/opt/{grad_norm,grad_rms,loss,param_count,param_rms,update_rms,updates}`.
- **Replay:** `replay/{chunks,inserts,items,ram_gb,replay_ratio,samples,streams,updates}`.
- **System:** `fps/policy`, `fps/train`, `usage/nvsmi/*`, `usage/psutil/*`, `step`.

`eval_episodes.jsonl` schema (the `EvalRewardSink` output) — **only one key per line:**
`{"rewards": [float, ...]}`. One line per eval episode; list length = steps + 1 (reset frame).

### Diagnostic atoms that are NOT logged anywhere

These were searched for in keys, in `launch.log`, and as separate artifacts — **absent**:

- **No raw action stream and no action histogram.** Nothing records which action *index* or
  *type* (ACTION1–7 / ACTION6 click-grid) was sampled at any step. Step 2(a) as literally
  posed (empirical per-type distribution; measured valid-set hit fraction) is **not directly
  computable from logs**. See §3 for what we can derive instead.
- **No action-mask vector.** `launch.log` logs only `Metrics filtered by: 'score|length|fps|
  ratio|train/loss/|train/rand/'` and the filtered scalar lines — it never prints the per-game
  valid-action set. The "ls20 valid set = 4 directional" is external knowledge, not in logs.
- **No board-state-change / pixel-diff signal.** There is no inter-frame-diff metric.
  `train/loss/image` is WM *reconstruction* error (decreases as the WM learns the board), **not**
  a measure of whether the board changed under the agent's actions.
- **No explicit `terminated` / `truncated` / `terminated_by_budget` flag** in either file.
  Termination cause is *inferred* from constant episode length (see §2c), not logged directly.

---

## 2. Per-run extracted tables (Step 2 a–f)

Every cell below is from `scratch/ls20_lf52_audit/{run}_{metrics,eval}.jsonl`
(= B2 `phase4-proper/...` warm or `phase4-fromscratch/...` cold). `ln(4102) = 8.31923`.

### (a) Policy — entropy & random fraction  (keys `train/ent/action`, `train/rand/action`)

| run | `rand/action` min..max | `ent/action` min..max | gap to ln(4102), last |
|-----|------------------------|------------------------|------------------------|
| warm_ls20_s0 | 1.0000..1.0000 | 8.31912..8.31923 | +1.4e-06 |
| warm_ls20_s1 | 1.0000..1.0000 | 8.31902..8.31923 | +2.2e-05 |
| warm_lf52_s0 | 1.0000..1.0000 | 8.31921..8.31923 | +8.1e-07 |
| warm_lf52_s1 | 1.0000..1.0000 | 8.31922..8.31923 | +8.1e-07 |
| cold_ls20_s0 | 1.0000..1.0000 | 8.31922..8.31923 | +7.9e-07 |
| cold_ls20_s1 | 1.0000..1.0000 | 8.31922..8.31923 | +8.1e-07 |
| cold_lf52_s0 | 1.0000..1.0000 | 8.31922..8.31923 | +8.1e-07 |
| cold_lf52_s1 | 1.0000..1.0000 | 8.31922..8.31923 | +8.1e-07 |

`rand/action` is **exactly 1.0 at every logged step** (min == max) on all 8 runs, and entropy
sits at the maximum of a uniform over 4102 categories to ≤2e-5, **from step 0 to 500k**. The
policy is uniform over the **full 4102-index action space** the entire run — it never commits.

**Dilution (analytic, valid *because* `rand/action = 1.0` ⇒ uniform sampling):**
for ls20 (valid = 4), expected fraction of sampled actions landing in the valid set =
`4 / 4102 = 0.0975 %`; dead:valid = `4098 : 4 = 1024.5 : 1`. This is the **expected** rate under
the logged-uniform policy — **not** an empirically measured count (no action stream). For **lf52
the valid-set size is not in the logs**, so its dilution ratio cannot be quantified at all.

### (b) Reward / return time-series  (keys `episode/score`, `train/rew`, `train/ret`)

| run | `episode/score` nonzero / total | max | `train/rew` range | `train/ret` range |
|-----|------|-----|-------------------|-------------------|
| warm_ls20_s0 | 0 / 502 | 0.000 | +1.8e-11 .. +6.9e-04 | +7.5e-10 .. +5.6e-03 |
| warm_ls20_s1 | 0 / 502 | 0.000 | +6.6e-12 .. +6.4e-04 | +2.2e-10 .. +5.2e-03 |
| warm_lf52_s0 | 0 / 7399 | 0.000 | +1.2e-13 .. +5.6e-05 | +4.8e-12 .. +4.9e-04 |
| warm_lf52_s1 | 0 / 7399 | 0.000 | +4.2e-13 .. +7.5e-05 | +1.2e-11 .. +6.5e-04 |
| cold_ls20_s0 | 0 / 502 | 0.000 | **0.0 .. 0.0** | **0.0 .. 0.0** |
| cold_ls20_s1 | 0 / 502 | 0.000 | **0.0 .. 0.0** | **0.0 .. 0.0** |
| cold_lf52_s0 | 0 / 7399 | 0.000 | **0.0 .. 0.0** | **0.0 .. 0.0** |
| cold_lf52_s1 | 0 / 7399 | 0.000 | **0.0 .. 0.0** | **0.0 .. 0.0** |

Reward is **structurally zero**. `episode/score` (real env return per finished episode) is 0 for
**every** episode across all runs — 502 episodes/run on ls20, 7399/run on lf52, **no nonzero step**.
The warm `train/rew`/`train/ret` "ranges" top out at ~1e-4 because the warm-started WM *reward
head* emits a vanishingly small non-zero prediction; the **cold** runs are **exactly 0.0** — a
clean confirmation that the actual reward signal never fired. (`train/loss/rew` → 1e-7…1e-11,
i.e. the reward head correctly converges to "always 0".) Mechanism confirmed: reward = Δlevels,
never triggered → **no reward gradient ever existed**.

### (c) Episode-length distribution  (key `episode/length`; eval from `len(rewards)-1`)

| run | train ep/len mean / med / min / max | eval ai_actions mean / min / max |
|-----|-------------------------------------|----------------------------------|
| warm_ls20_s0 | 1001 / 1001 / 1001 / 1001 | 1000 / 1000 / 1000 |
| warm_ls20_s1 | 1001 / 1001 / 1001 / 1001 | 1000 / 1000 / 1000 |
| cold_ls20_s0 | 1001 / 1001 / 1001 / 1001 | 1000 / 1000 / 1000 |
| cold_ls20_s1 | 1001 / 1001 / 1001 / 1001 | 1000 / 1000 / 1000 |
| warm_lf52_s0 | 67.6 / 65 / 65 / 186 | 64 / 64 / 64 |
| warm_lf52_s1 | 67.6 / 65 / 65 / 236 | 64 / 64 / 64 |
| cold_lf52_s0 | 67.6 / 65 / 65 / 187 | 64 / 64 / 64 |
| cold_lf52_s1 | 67.6 / 65 / 65 / 236 | 64 / 64 / 64 |

- **ls20 is horizon-capped at 1000 steps** — every train *and* eval episode is exactly the full
  budget, zero variance. Comparable to the sb26 reference point (~1000). Episodes are
  **truncation-bound, never self-terminating** (inferred from the constant length; no explicit
  flag is logged).
- **lf52 self-terminates at ~64 steps** (eval: exactly 64 every episode; train median 65, tail to
  236). This sits between the vc33 (~51) and cd82 (~101) reference points. The deterministic
  64-step eval cut-off under a uniform-random policy indicates a **fixed short horizon for lf52**,
  not agent-death (death would vary with the random actions). Again inferred, not a logged flag.

### (d) Critic / value estimates  (keys `train/val`, `train/slowval`, `train/tar`)

| run | `train/val` range | `train/slowval` range |
|-----|-------------------|-----------------------|
| warm_ls20_s0 | +2.5e-09 .. +2.33e-05 | +2.4e-09 .. +2.32e-05 |
| warm_ls20_s1 | +1.2e-09 .. +2.15e-05 | +1.1e-09 .. +2.20e-05 |
| warm_lf52_s0 | −2.74e-06 .. +7.1e-12 | −2.52e-06 .. +5.3e-12 |
| warm_lf52_s1 | −4.14e-06 .. +1.1e-11 | −4.34e-06 .. +1.2e-11 |
| cold_ls20_s0 | **0.0 .. 0.0** | **0.0 .. 0.0** |
| cold_ls20_s1 | **0.0 .. 0.0** | **0.0 .. 0.0** |
| cold_lf52_s0 | **0.0 .. 0.0** | **0.0 .. 0.0** |
| cold_lf52_s1 | **0.0 .. 0.0** | **0.0 .. 0.0** |

The critic **never predicts non-zero return**: cold runs are flat exactly 0.0; warm runs stay
within ±4e-6 (numerical residue from the pretrained reward head). No `return_ema` key is logged;
`train/val`/`train/slowval`/`train/tar` are the value series and all are flat-zero. Consistent
with (b): with zero reward there is nothing for the critic to regress.

### (e) Board-state-change / pixel-diff signal

**Not logged.** No such metric exists in any of the 8 runs (see §1). For reference only, the WM
losses *are* healthy (recon `train/loss/image` → 0.27–0.39; dynamics `train/loss/dyn` → ~1.0 in
every run; cold runs start at recon 666–1828 vs warm 41–62, then converge to the same floor) —
i.e. the **world model fits**; the failure is downstream in the controller. But recon error is
**not** a board-*change* signal and cannot stand in for the "the board never changes" claim.

### (f) Eval atoms  (`eval_episodes.jsonl`)

| run | eval episodes | levels_completed (Σ rewards) | ai_actions/ep | terminated_by_budget |
|-----|---------------|------------------------------|---------------|----------------------|
| warm_ls20_s0 | 24 | 0 | 1000 | yes¹ |
| warm_ls20_s1 | 24 | 0 | 1000 | yes¹ |
| cold_ls20_s0 | 24 | 0 | 1000 | yes¹ |
| cold_ls20_s1 | 24 | 0 | 1000 | yes¹ |
| warm_lf52_s0 | 28 | 0 | 64 | (fixed 64-step horizon)¹ |
| warm_lf52_s1 | 26 | 0 | 64 | (fixed 64-step horizon)¹ |
| cold_lf52_s0 | 26 | 0 | 64 | (fixed 64-step horizon)¹ |
| cold_lf52_s1 | 25 | 0 | 64 | (fixed 64-step horizon)¹ |

`levels_completed = Σ rewards = 0` for every eval episode (no reward is ever non-zero in any eval
episode of any run). `ai_actions = len(rewards) − 1`. ¹**`terminated_by_budget` is inferred** from
the constant per-episode length — there is no logged termination flag (§1).

---

## 3. What this lets us claim now vs what we still cannot claim from logs alone

### Now supportable from logged evidence (all 8 runs, both seeds, both arms, identical):

1. **The policy is exactly uniform-random over the full 4102-action space for the entire 500k
   steps.** `train/ent/action ≈ ln(4102)` to ≤2e-5 and `train/rand/action = 1.0` constant from
   step 0. This is *stronger* than the original entropy-only claim: entropy pinned at **ln(4102),
   not ln(4)**, is direct logged proof that the effective action distribution spans all 4102
   indices — i.e. **the mask is not collapsing the space to the valid set** in these runs. The
   ~1024:1 dilution for ls20 is then the *expected* valid-hit rate under that logged-uniform
   policy (4/4102 = 0.0975%).
2. **There is no reward signal and no value signal anywhere.** `episode/score = 0` across all
   502 (ls20) / 7399 (lf52) train episodes and all eval episodes; cold-arm `train/rew`, `train/ret`,
   `train/val`, `train/slowval` are **exactly 0.0**. The sparse-reward → no-gradient mechanism is
   shown, not asserted.
3. **The bottleneck is the controller, not the world model.** Recon and dynamics losses converge
   normally (recon → ~0.3, dyn → ~1.0) in every run; warm and cold collapse to the same zero
   outcome. Pretraining changes only the *initial* WM loss, not the result.
4. **Both games run to a fixed horizon with zero progress** (ls20 truncated at 1000, lf52 at 64),
   stable across seeds and arms.

### Still NOT claimable from logs alone:

1. **The empirical per-action-type distribution / measured valid-set hit fraction.** No action
   stream or histogram was logged, so the 4/4102 dilution is only the *expectation* under the
   logged uniform policy — not a counted observation. Step 2(a) as literally requested cannot be
   answered from data; only derived.
2. **lf52's dilution ratio.** Its valid-set size is not in the logs at all.
3. **"The board never changes."** No board-state-change / pixel-diff atom exists. We can show
   *reward* never fires and the policy is random; we **cannot** show from logs that the pixels are
   static. `train/loss/image` is reconstruction error and does not measure board change.
4. **Causes of the uniform policy** beyond "no reward gradient." Entropy = ln(4102) proves the
   mask is *inactive in effect*, but logs don't record the mask vector, so we cannot distinguish
   "mask never plumbed into the actor" from "mask present but irrelevant because no reward gradient
   ever differentiated actions." Both are consistent with the logged numbers.

## 4. Diagnostic atoms missing from the logs (would require a re-run to obtain)

- **Per-step action index / type stream** (or an action histogram) → to *measure* the valid-set
  hit fraction rather than derive its expectation.
- **The per-game action-mask vector at reset** → to confirm whether masking was plumbed into the
  actor logits at all.
- **An inter-frame board-change / pixel-diff scalar** → the only way to substantiate "the board
  never changes."
- **Explicit `terminated`/`truncated` flags per episode** → currently inferred from length.
- **Post-mask policy entropy** (entropy over the valid set only) → to confirm that masking would
  actually collapse the effective space to ln(|valid|).

---

## Decision surfaced for Haso

**The random-policy / controller-bottleneck diagnosis is now empirically supportable from the
existing logs and does NOT need a re-run to stand up** — entropy ≡ ln(4102) + rand-fraction ≡ 1.0
+ exactly-zero reward/critic across all 8 runs (both seeds, warm and cold) is a complete,
reproducible mechanistic story.

**However, two specific framings still require the masked ls20/lf52 re-run:**

- **The causal masking claim** — "*adding masking would unblock ls20/lf52*." Existing logs prove
  the mask is *inactive* (entropy = ln(4102), not ln(4)) and quantify the dilution only *in
  expectation*; they cannot show that activating the mask fixes anything. Note the zero-reward
  finding (§2b) is a real confound: a policy uniform over 4 valid actions still has to actually
  reach a level-up to generate gradient, so masking alone may not suffice. The masked re-run is
  the experiment that decides this.
- **The "board never changes" claim** — needs a board-change / pixel-diff logger; no existing
  signal supports it.

**Recommendation:** if the paper text restricts itself to *"on ls20/lf52 the learned policy
remains uniform-random over the full action space and receives no reward signal, so neither the
world model's fit nor pretraining transfers into control"*, it is fully backed by logged data —
**ship it, no re-run.** If you want to keep the explicit *masking-would-help* claim or the
*board-never-changes* claim in Table 3 / Fig 2c, run the **masked ls20/lf52 re-run with an
action-stream + pixel-diff logger** — that is the only way those two atoms enter evidence.

---

# 2026-06-09 — W&B follow-up: do the three gap atoms live in W&B as artifacts?

The B2 audit above flagged three atoms absent from the B2 jsonl snapshot: (a) raw action
stream / action-type histogram, (b) board/pixel-change signal, (c) explicit termination flags.
A refreshed W&B key was supplied to check whether W&B holds them as artifacts/media that never
landed in B2. Read-only; reproduce with `analysis/ls20_lf52_wandb_probe.py`.

## Step 0 — the sweep IS in W&B (it was a wrong-account problem, not a sync problem)

`hasofocus-university-of-the-west-of-england/arc3-wm-sprint` — **24 runs, all `finished`.** The
first audit failed because the cached `~/_netrc` key belonged to a **different account**
(`haso-university-of-the-west-of-england`, default entity, sees `nnet` only and **0 projects**
under `hasofocus-*`). Under the new *hasofocus* key, `api.default_entity =
hasofocus-university-of-the-west-of-england` and the project resolves. Not under `haso`,
`hasofocus`, or `haso-university-of-the-west-of-england` (all 0 projects).

The 8 ls20/lf52 run ids (entity/project as above):

| run | id | run | id |
|-----|----|----|----|
| warm ls20 s0 | `2329m1sj` | cold ls20 s0 | `xkqetscm` |
| warm ls20 s1 | `lwncvl0i` | cold ls20 s1 | `h5s5q7lt` |
| warm lf52 s0 | `ttjocji0` | cold lf52 s0 | `dqqb8jtm` |
| warm lf52 s1 | `hr6y701f` | cold lf52 s1 | `1mnwbryd` |

## Step 1 — inventory of one ls20 run (`2329m1sj` = B2 `warm_ls20_s0`)

- **History: 70 columns; summary: 71 keys.** The *scalar* set is identical to B2's
  `metrics.jsonl` (61 keys). The only keys W&B has beyond B2 are **media / strings**:
  `epstats/policy_image`, `eval/openloop/image`, `report/openloop/image` (rendered GIFs),
  plus `eval/params/summary`, `report/params/summary`, `usage/nvsmi/output`, `timer` (strings).
- **The only `*action*` columns are `train/ent/action` and `train/rand/action`** — the same two
  scalars B2 already had. **No raw action stream, no action histogram, no `wandb.Table`.**
- **Logged artifacts: 1** — `run-2329m1sj-history:v0` (type `wandb-history`, the standard scalar
  history table; no action/pixel payload). `used_artifacts`: 0.
- **Files: 53.** `config.yaml`, `output.log`, `requirements.txt`, `wandb-metadata.json`,
  `wandb-summary.json`, and **47 board GIFs**: 36 `media/videos/epstats/policy_image_*.gif`
  (real game board during policy rollouts) + 11 `media/videos/{eval,report}/openloop/image_*.gif`
  (WM open-loop imagination panels). **All 8 runs** carry these (ls20: 36 policy / 11 openloop;
  lf52: 72–73 policy / 13 openloop) and **none carry an action artifact** (verified per-run).

## Step 2 — gap-closing extraction (W&B-only data; PIL frame analysis of the GIFs)

### (a) Action distribution — NOT in W&B either

No action stream / histogram / table exists in any of the 8 runs (`action_artifact=False` for
all). The measured valid-set hit fraction **cannot be counted** from W&B; the ~1024:1 dilution
stays an *expectation*. **Indirect** corroboration from the board media (not a direct count):
- **ls20:** 17 of the 36 logged `policy_image` GIFs are **byte-identical single static frames**
  (same content hash `c07a3ae2…`, T=1) — i.e. in ~47% of logging windows the captured board never
  moved at all. Consistent with board-moving (directional) actions being rare.
- **lf52:** policy GIFs run long (T=65→520) and show a **near-constant ~1-cell change every step**
  (the white cursor relocating — consistent with ACTION6 click-spam dominating the uniform draw)
  with only sparse larger changes (29–258 cells). Again: lots of cursor motion, little board change.

These are *qualitatively consistent* with the dilution story but do **not** substitute for an
action-type count. **Verdict: still needs a re-run** for a measured action distribution.

### (b) Board / pixel change — RESOLVED by media; it REFUTES the literal "board never changes"

No numeric pixel-diff scalar exists, **but the policy-rollout and open-loop GIFs (absent from B2)
let us answer the binary question directly.** PIL `ImageSequence` over the 64×64 frames:

| evidence (run `2329m1sj` unless noted) | finding |
|---|---|
| `policy_image_16016` (ls20), 3 frames | board **changes**: an orange/blue block translates 1 cell; **52/4096 (1.27%) pixels** change per step; cursor + walls + progress bar static |
| `policy_image_450992` (ls20, late), 2 frames | same 1-cell block move (1.27%) — behaviour unchanged at 451k steps |
| `policy_image_0_c07a3ae2…` ×17 (ls20) | **byte-identical static board** — episodes where no board-moving action ever landed |
| `eval/openloop/image_*` (ls20), 33 frames, 196×408 | 3-row panel (ground-truth / WM reconstruction / error); ground-truth board barely moves; WM reconstructs the static layout |
| lf52 `policy_image_*` | ~1 cell/step (cursor) + sparse 29–258-cell bursts (real board changes) |

**Visual confirmation:** ls20 is a sliding-piece puzzle — a fixed white "+" cursor and a movable
orange/blue block inside grey walls, with a yellow progress bar. Directional actions slide the
block by one cell; everything else is static; no level-up ever occurs (reward ≡ 0, §2b/§2f).

So **"the board never changes" is FALSE as literally stated** — the board *does* change (a piece
slides on ls20; the cursor moves and the board occasionally changes on lf52), just **rarely**,
because board-moving actions are ~0.1% of the uniform draw and progress never completes. The
*defensible* reworded claim — *"the policy issues a board-changing action on a vanishing fraction
of steps; ~half of ls20 rollout windows are fully static and no level-up ever occurs"* — **is now
supported from logged media.** A quantified per-step Δpixel **time-series** is still not a logged
scalar (would need a re-run) but is **not required** to answer the binary claim. **Verdict:
now-supported-from-logs (binary); the media contradicts the strong wording — reword it.**

### (c) Termination-reason flag — NOT in W&B either

No `terminated` / `truncated` / termination-reason key in summary or history; only `episode/length`
(already in B2). lf52's exact-64 *eval* endings remain **inferred from constant eval episode
length**, not a logged flag. **Verdict: still needs a re-run** (or env-wrapper code inspection;
from logs alone it is absent).

### Cross-check (W&B vs B2, one run) — exact agreement

`2329m1sj` vs B2 `warm_ls20_s0`: `train/ent/action` 8.31912–8.31923 ≈ ln(4102) ✓;
`episode/score` 0/502 ✓; `train/rew` 1.8e-11–6.9e-4 ✓. **One nuance:** W&B raw
`train/rand/action` is **0.999987–1.0** (not exactly 1.0); B2's "1.0000" was 4-dp rounding. No
substantive mismatch — the policy is uniform to within <2e-5 of the maximum.

## Decision surfaced for Haso — per atom, did W&B close the gap?

| atom | in W&B? | verdict |
|------|---------|---------|
| **(a) action distribution / histogram** | **No** (no stream, no histogram, no table — all 8 runs) | **STILL NEEDS A RE-RUN** for a measured valid-set fraction. Board media only corroborate the dilution indirectly. |
| **(b) board / pixel change** | **Partly — board GIFs, no scalar** | **NOW SUPPORTED FROM LOGS (binary).** Media show the board *does* change but rarely → the literal "board never changes" is **refuted**; reword to "changes are rare / no level-up." A quantified Δpixel curve would still need a re-run but isn't needed for the claim. |
| **(c) termination-reason flag** | **No** | **STILL NEEDS A RE-RUN** (or wrapper code inspection). lf52's 64-step ending stays inferred from length. |

**Bottom line:** W&B added the board *videos* (which B2 lacked) and they are decisive for atom
(b) — but they also show the strong "board never changes" wording is wrong and should be softened.
W&B did **not** contain the action-level or termination-reason data, so atoms (a) and (c) still
require the masked re-run (with an action-stream logger + explicit termination flags) if the paper
wants to make the measured-dilution or termination claims rather than the
random-policy/no-reward/board-rarely-changes claims that existing logs already support.
