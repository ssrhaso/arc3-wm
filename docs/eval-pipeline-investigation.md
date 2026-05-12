# Eval pipeline investigation — Part A findings

> Investigation only; no impl. Three empirical questions answered with
> citations, then trade-offs surfaced for Haso to decide.

## Post-investigation decisions (D1–D5)

Haso resolved the open questions and trade-offs surfaced below. The
decisions override any conflicting recommendation in the body of this
doc; the body is preserved as the historical Part A record.

- **D1** — Upper-median rule is methodology.md's "upper of the two
  middle entries" (index `n//2` 0-indexed of the ascending-sorted
  completer action counts). The "3rd-place" colloquial framing is wrong
  for `n ≠ {4, 5}` and is not used anywhere downstream. methodology.md's
  "third place" wording is correct only as the n=4/n=5 illustration of
  the upper-median rule.
- **D2** — Post-hoc only. No in-loop RHAE hook, no bespoke eval loop.
  Option (II) wins: compute RHAE from `eval/episode/*` keys that stock
  `embodied.run.train_eval` already emits, via a post-hoc
  `scripts/compute_rhae.py`. `arc3_wm/rhae.py` exposes a pure
  `RHAEAggregator` class (no scheduler, no step callback).
- **D3** — The vc33 500k dry-run uses `--script train_eval` so the
  `eval/episode/*` series exists for post-hoc analysis.
- **D4** — Full 500k at $6 stop-and-surface budget; do not descope to
  250k to absorb eval overhead.
- **D5** — Baseline extractor reuses
  `arc3_wm.replay_loader.load_replay_file` for cn04-safe episode
  segmentation rather than reparsing raw JSONLs.

The remainder of this doc is the Part A walk-through that produced
the open questions D1–D5 then resolved.

## Q1 — Per-game human baselines: where do they live?

### Finding: not published; **must derive from the 340 local replays.**

Evidence walked in order:

1. **`docs/arc-agi-3/methodology.md`** is the canonical RHAE methodology
   doc. It specifies the algorithm (upper-median first-time-player action
   count per level) but contains **no baseline data** — only formula and
   prose. See [methodology.md:26-36](arc-agi-3/methodology.md#L26-L36).
2. **`docs/arc-agi-3/scorecards.md`** and
   [docs/arc-agi-3/toolkit/get-scorecard.md](arc-agi-3/toolkit/get-scorecard.md)
   document `Arcade.get_scorecard()` → `EnvironmentScorecard` with
   `.score`, `.games`. **This is the agent's aggregated score**, not
   human baselines — see [scorecards.md:19-26](arc-agi-3/scorecards.md#L19-L26).
   Phase 0's load-bearing question "does `get_scorecard()` return RHAE
   directly?" is resolved here as: yes for the AGENT's RHAE post-run,
   but NOT for the human baseline inputs that go into it.
3. **`docs/arc-agi-3/toolkit/arc_agi.md`** lists every public method on
   `Arcade`: `make`, `get_environments`, `create_scorecard`,
   `get_scorecard`, `close_scorecard`. **No `get_baselines` / `list_baselines` /
   `get_methodology` API.** See [arc_agi.md:118-242](arc-agi-3/toolkit/arc_agi.md#L118-L242).
4. **ARC-AGI-3 paper (arXiv:2603.24621)** — not cached locally under
   `docs/arc-agi-3/`. Notion's "RHAE — reference" §Sources lists
   `https://arcprize.org/blog/arc-agi-3-human-dataset` as the baseline-construction
   reference. That blog (URL only, not fetched) is the methodology
   reference, not a published numeric data source.
5. **`docs/replay-format.md`** confirms the human-replay JSONL schema:
   every per-step row carries `levels_completed: int` (monotone
   non-decreasing) and `win_levels: int` (total levels in the game).
   See [replay-format.md:55-89](replay-format.md#L55-L89) and the
   `levels_completed range: 0..7` aggregate stat for the 1557-line ar25
   sample on [replay-format.md:167-180](replay-format.md#L167-L180).
6. **Local `data/replays/`** has all 25 game subdirs (340 JSONLs
   total — 10 / 11 / 12 / 13 / 14 per game, except `lp85=54`,
   `sc25=15`). vc33 = 10 replays; ~10 completers per level is well
   within the upper-median rule's domain. (Note: Notion's "take
   3rd-place" colloquial framing is the wrong colloquialism for n=10
   per D1; the correct rule is methodology.md's upper of two middle
   entries — for n=10, index 5 of the ascending-sorted list.)
7. **Sample inspection of vc33 replay** (first 3 rows of
   `data/replays/vc33/1469cb95-…recording.jsonl`) shows
   `levels_completed=0`, `win_levels=7`, `action_input.id ∈ {0, 6}`
   present and parseable as documented.

### Baseline derivation sketch

Per game, for each of the ~10–14 player JSONLs in that subdir:
1. Parse rows in order. Skip RESET rows (`action_input.id == 0`).
2. For each row, the action count for the **current level** =
   number of non-RESET rows seen since `levels_completed` last
   incremented. Increment occurs when `levels_completed[i+1] > levels_completed[i]`.
3. Record `(player_guid, level_index_1_based, action_count)` for
   every level the player completed.
4. Per-game, per-level: sort completers ascending by `action_count`.
   **Upper median** per methodology.md: "upper of the two middle
   entries" ([methodology.md:30](arc-agi-3/methodology.md#L30)). In
   0-indexed terms that's `sorted_completers[n // 2]`:
   - n=1 → index 0 (the sole completer; document + test the edge).
   - n=2 → index 1 (the larger of the two).
   - n=4 → index 2 (matches methodology.md's "third place" example).
   - n=5 → index 2 (matches methodology.md's "third place" example).
   - n=9 → index 4.
   - n=10 → index 5.
   (D1 confirms this rule. Notion's "3rd-place for ~10 testers"
   wording does not match the rule numerically for n=10 and is not
   used anywhere downstream.)

### cn04 quirk

`docs/replay-format.md:27-29` and `arc3_wm/replay_loader.py:27-31`
flag cn04 specifically: post-terminal rows emit `levels_completed`
drops as engine bookkeeping noise, not actual level regressions.
The existing replay loader handles this with a state-machine that
discards post-terminal rows after the first WIN/GAME_OVER. The
baseline-extraction utility must apply the same logic, or
equivalently re-use `load_replay_file` and read level transitions
off the resulting transition stream — that's the safer path
because the cn04 logic is already proven across the 340-replay
test suite.

### Conclusion: option (b)

**Baselines are not published. Derive from the 340 local replays via
a new `scripts/extract_human_baselines.py`** that reuses
`arc3_wm/replay_loader.load_replay_file` for cn04-safe segmentation,
computes per-level action counts per player, then takes the upper
median per (game, level) and dumps to a committed JSON fixture
(`data/human_baselines.json`).

### Resolved by D1

methodology.md wins. The extractor picks index `n // 2` of the
ascending-sorted completer action counts (0-indexed) — the upper of
the two middle entries for even n, the middle entry for odd n.
Notion's colloquial "3rd-place" framing was the wrong extrapolation
of methodology.md's n=4/n=5 illustration to general n and is not
used.

## Q2 — Level-boundary detection during rollouts

### Finding: `r = Δ levels_completed` is **literal and unchanged**, and `levels_completed` is also in `info` — pick (b).

Evidence:

1. **`arc3_wm/env.py:113`** — `reward = float(levels - self._prev_levels_completed)`
   with `self._prev_levels_completed` updated immediately after.
   Exactly the reward shape Notion's userMemory describes.
2. **`arc3_wm/env.py:144-152`** — the Gymnasium `info` dict at every
   reset and step exposes `levels_completed`, `win_levels`,
   `available_actions`, `action_mask`, `state`, `guid`, `steps`.
   So the underlying gym env DOES emit `levels_completed`
   independent of the reward signal.
3. **`tests/test_embodied_env.py:5,41`** pins the embodied step-dict
   shape to `{OBS_KEY, "reward", "is_first", "is_last", "is_terminal"}`.
   No info / levels_completed pass-through to the agent's per-step
   packet today.
4. **`arc3_wm/embodied_env.py:63,86,98`** — the gym `info` is stored
   on `self._info` after every step, but is dropped from the
   embodied step-dict returned by `step()`. The hook can read
   `env.info["levels_completed"]` externally if it holds a reference
   to the env, but cannot read it off the standard DV3 driver
   step-callback `(tran, worker)`.

### Three options for the eval loop

(a) **Diff cumulative reward** in the per-episode callback.
   Final-level-reached = sum of per-step rewards, since `r ∈ {0, +1}`
   and `r=+1` exactly on level-up. Per-level action count =
   `Σ steps where cumulative_reward == k` for level k. Robust to
   any future info-dict change; relies only on `r = Δ levels_completed`
   staying literal.

(b) **Poll `env.info["levels_completed"]`** externally (the eval loop
   holds env handles via the make-env closures). Robust to future
   reward shaping; brittle to any code path that runs eval without
   a direct env reference.

(c) **Add a `log/levels_completed` pass-through** to the embodied
   step-dict in `arc3_wm/embodied_env.py`. Mirrors the
   already-existing `log/action_mask` pattern documented in the
   embodied_env docstring. **Risk:** the `log/action_mask` key was
   removed (commit `73c6d09`) because it crashed DV3's logfn scalar
   assertion. `levels_completed` is a scalar int, so likely safe —
   but verify before committing. This is the most invasive option
   and changes the per-step packet contract.

### Recommendation: (a)

Cumulative-reward diffing is the cleanest, doesn't touch the env
contract, and the `r = Δ levels_completed` invariant is already
test-pinned in `tests/test_embodied_env.py` and the Gymnasium-level
reward construction. If we ever change the reward shape (e.g. add
shaping), we'd update RHAE extraction alongside as part of the
same change — they're conceptually one unit.

## Q3 — Eval cadence

### Finding: DV3 has no `eval_every` knob; eval is gated on `report_every` (wall-clock seconds) and only fires under `--script train_eval`.

Evidence:

1. **`third_party/dreamerv3/dreamerv3/configs.yaml:48-70`** — full
   `run` block defaults. Knobs:
   - `train_ratio: 32.0` (size12m runs use this)
   - `log_every: 120` (wall-clock seconds)
   - `report_every: 300` (wall-clock seconds)
   - `save_every: 900` (wall-clock seconds)
   - `envs: 16` (train rollout envs)
   - `eval_envs: 4` (eval-only envs)
   - `eval_eps: 1` (episodes per eval-env per eval cycle)
   - **No `eval_every`.** Eval cadence = `report_every`.
2. **`third_party/dreamerv3/embodied/run/train.py`** — the
   default-script train loop has `should_report` (line ~27) but it
   only triggers a `report` call against the held-out training-replay
   stream, **not** an eval-driver rollout. There is no `driver_eval`
   in `train.py`. **`--script train` produces zero eval rollouts.**
3. **`third_party/dreamerv3/embodied/run/train_eval.py`** — the
   `--script train_eval` path. Line 131-141 of the train loop:
   ```python
   if should_report(step):
     print('Evaluation')
     driver_eval.reset(agent.init_policy)
     driver_eval(eval_policy, episodes=args.eval_eps)
     logger.add(eval_epstats.result(), prefix='epstats')
     ...
   ```
   Eval fires every `report_every` wall-clock seconds (default 300 s),
   runs `eval_eps` episodes per eval-env (default 4 envs × 1 ep = 4
   eval episodes per cycle).
4. **`scripts/launch_pergame.py:88-96,532-560`** — `--script` arg
   accepts `train|train_eval|eval_only` and dispatches to the
   corresponding `embodied.run.*` entry point. **Default `--script` is
   `train`.** Today's `docs/phase4-dryrun-vc33.md` invocation doesn't
   override the default → **the planned 500k vc33 dry-run does NO
   eval rollouts.**

### Cost estimate for `train_eval` on 500k vc33

- Eval cycle cadence: every `report_every=300 s` wall-clock.
- Wall-clock total ≈ 5 h ≈ 18000 s → ~60 eval cycles.
- Eval episodes per cycle: `eval_envs=4 × eval_eps=1 = 4`.
- Total eval episodes over the run: ~240.
- Per-episode env-step cost: bounded by the `max_steps=1000`
  default in `arc3_wm/env.py:42` (which is also the action-budget
  ceiling per Notion "Action budget" if level-1 baseline ≈ 200,
  but vc33-level-1 budget will be different — needs baselines
  to compute). Use 1000 as the upper bound.
- Total eval env-steps consumed: ≤ 240k. **Up to ~48% overhead on
  top of 500k train steps.** Real overhead lower if episodes
  terminate early (level-1 stall → episode ends at ≤ 1000 steps;
  level-1 clear at human-efficient ~20 actions → much shorter).

### Proposed eval cadence

**Keep `report_every=300 s` (DV3 default).** Don't tune the knob.
The cadence gives ~60 eval points across a 5 h dry-run, ~240 total
eval episodes — plenty of granularity for tracking per-game RHAE
over training, and the overhead is acceptable. If overhead during
Phase 4 proper (3-game × 2-seed on 5070s) becomes a budget concern,
raise `report_every` to 600 s in `configs/arc3.yaml` as a separate
tuning step.

### Integration constraint: stock `train_eval` is closed

`embodied.run.train_eval(...)` is a single self-contained function;
there's no hook point to register a `RHAEAggregator`-driven callback
against `driver_eval`. This means the integration choice is one of:

(I) **Fork `train_eval` into our codebase** — violates CLAUDE.md
   anti-goal "Do not refactor DreamerV3 internals. Use as-is."
(II) **Compute RHAE post-hoc from `eval/episode/*` keys that
   `train_eval` already emits.** DV3's per-eval-episode logger
   call (`train_eval.py:65-66`) emits `eval/episode/score`,
   `eval/episode/length`, `eval/episode/rewards` (the per-step
   reward stack with `agg='stack'`). The stack contains the full
   reward sequence — enough to reconstruct per-level action counts
   given `r = Δ levels_completed`. Post-hoc analysis in
   `analysis/*.ipynb` per Notion "Logging & analysis plan".
   **`eval/rhae/*` keys would NOT appear in wandb live; they appear
   after run completion in the analysis notebook.**
(III) **Custom eval loop in `scripts/launch_pergame.py`** — re-implement
   just the `driver_eval` portion (~30 lines, very close to
   `train_eval.py:73-83` + `131-141`), inject the RHAE hook into
   `driver_eval.on_step(...)`. Some duplication of DV3 code but
   explicit, scoped to our launcher, and the duplicated portion
   is small. Live `eval/rhae/*` keys in wandb. **Closest to the
   `pretrain_wm_loop` pattern we already use.**

### Resolution: D2 chose (II), not (III)

Part A recommended (III) for live wandb panels; Haso overrode in D2.
The bespoke eval loop was deemed not worth the duplicated DV3 code
when the post-hoc analysis path is already adequate for the Phase-4
gate question. The `eval/episode/*` series stock `train_eval` emits
is sufficient to reconstruct per-level action counts in
`scripts/compute_rhae.py`, and the dry-run answers "did the agent
clear vc33 level 1?" from `episode/score` directly (vc33's native
reward fires on level-up), not from in-run RHAE.

`arc3_wm/rhae.py` accordingly exposes a pure `RHAEAggregator` class
with no scheduler or step callback; the post-hoc CLI is the only
caller.

## Cross-cutting: scope changes implied for today's dryrun

Two material consequences if Part B proceeds with the recommendations above:

1. **The 500k vc33 dryrun must use `--script train_eval`, not the default
   `train`.** Otherwise no eval rollouts fire and the hook never runs.
   The runbook at [docs/phase4-dryrun-vc33.md](phase4-dryrun-vc33.md)
   needs a one-line patch to add `--script train_eval`.
2. **The dryrun budget at $3 / 5h wall-clock holds**, but accounting for
   the ~48% eval-overhead upper bound, plan for closer to 7 h
   wall-clock on the same A100 — bringing cost to ~$4 at $0.530/hr,
   nudging the brief's $4 stop-and-surface threshold. **Surface this
   to Haso before launch.** Either accept the budget ceiling moves
   to ~$5 or scale the dryrun to fewer env steps (e.g. 250k).

## Part B scope (post D1–D5)

Per D2 the Part B scope is post-hoc-only:

1. **`scripts/extract_human_baselines.py`** + tests + committed
   `data/human_baselines.json` fixture. Per D5 the extractor reuses
   `arc3_wm.replay_loader.load_replay_file` for cn04-safe segmentation
   rather than reparsing raw JSONLs. Per D1 the upper-median rule is
   methodology.md's "upper of two middle entries" (`sorted[n//2]`
   0-indexed).
2. **`scripts/compute_rhae.py`** post-hoc CLI. Reads `eval/episode/*`
   keys from a wandb run id (or local wandb summary export), segments
   each episode into per-level AI action counts using
   `r = Δ levels_completed` cumulative-reward diffs, feeds them into
   `RHAEAggregator(human_baselines=...)`, prints the three-key family
   per Notion "Logging & analysis plan".
3. Patch `docs/phase4-dryrun-vc33.md` per D3/D4: add
   `--script train_eval`, lift stop-and-surface budget from $4 to $6,
   add a "Post-hoc analysis" section. **No `--rhae-eval` flag** —
   that was the (III) artifact and is dropped.

## Blockers / ambiguities — resolved

- Upper-median definition: **resolved by D1** (methodology.md
  upper-of-two-middle-entries; `sorted[n // 2]`).
- Dry-run budget vs eval overhead: **resolved by D4** (full 500k at
  $6 stop-and-surface budget).
- (III) bespoke eval loop vs (II) post-hoc: **resolved by D2** (post-hoc
  via `scripts/compute_rhae.py`).
