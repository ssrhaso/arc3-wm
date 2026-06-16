# Why lf52 episodes end at exactly 64 steps — termination-mechanism audit

**Date:** 2026-06-09
**Scope:** read-only inspection of wrapper/launcher code + logged replay data. No training, no
episode runs, no paper edits, no push.
**Question:** is lf52's exact-64-step ending (a) a fixed/structural cap independent of the agent's
actions, or (b) something the agent's behaviour triggers (e.g. reliably hitting a lose-state)?

## TL;DR

The 64-step ending is a **`terminated=True` engine event (GAME_OVER), NOT a wrapper truncation** —
a *different* mechanism from the 1000-step truncation that sb26/ls20 hit. lf52 is **not**
structurally horizon-capped: human replays run 1→1605 steps and reach WIN. The exact-64 invariance
under a uniform-random eval policy points to an **engine-enforced fixed per-episode action budget
(~64 moves → GAME_OVER when the level is unsolved)** — i.e. a *fixed cap expressed as termination*,
not a varied agent-reached death. **This is now CONFIRMED from the engine source — see the
2026-06-15 update immediately below**, which supersedes the original "cannot be confirmed from code"
caveat and the §"What cannot be determined from code" section.

## Update 2026-06-15 — RESOLVED by reading the lf52 engine source

The lf52 engine class was re-fetched (`scripts/cache_env_files.py lf52`, NORMAL-mode file pull, no
training run) to `environment_files/lf52/271a04aa/lf52.py` (version `271a04aa`, byte-identical to
the version named in the Phase-4 run logs). The exact GAME_OVER trigger is now read directly from
code:

- **`lf52.py:5771-5772`** — `elif level == 1 and self.…asqvqzpfdi >= 64: self.lose()`. Level 1 has a
  hard **64-action budget**; when the budget counter reaches 64 the engine calls `lose()`
  (`lf52.py:2091-2092` sets the lose flag), which `arcengine` surfaces as `GameState.GAME_OVER` and
  our wrapper maps to `terminated=True` ([env.py:164](../arc3_wm/env.py#L164)).
- **Budget tiers are all multiples of 64**: L1 = 64, L2–5 = `64*5` = 320 (`lf52.py:5779`),
  L6–10 = `64*10` = 640 (`lf52.py:5775`). The current level drives the tier via
  `whtqurkphir = self._current_level_index + 1` (`lf52.py:5866`).
- Because the random policy never clears level 1 (reward ≡ 0), **every episode dies at the level-1
  budget of 64** — that is the exact-64.
- **The "64 = 64×64 grid" guess is dead.** lf52's levels are **8×8** boards (`lf52.py:66-137`); the
  64 is a deliberate move budget, unrelated to the grid.
- **Train-tail mechanism (the old residual), also resolved.** The budget counter `asqvqzpfdi`
  increments **+1** only on directional moves (`tmhxwcojkh`, `lf52.py:5277`) and ACTION6 grid-clicks
  (`dghsidbuet`, `lf52.py:5335`), **+20** on a pending undo (`lf52.py:5805`), and is **not**
  incremented by ACTION5 (`lf52.py:5327-5329`) or special-region ACTION6 clicks (`lf52.py:5832`).
  So under uniform-random sampling, budget-exempt actions desync the env-step count from the
  64-budget count: ~92 % of train episodes die at the modal 64 env-steps, with a smooth tail to 236
  (and **no 2×/3×64 clustering**, since these are small per-episode perturbations, not full-budget
  resets) — exactly the logged distribution. eval's smaller sample (~105 episodes) landed on clean
  64s.

**Net:** the termination is **(A) engine** — a hard-coded per-level action budget (`lose()` →
GAME_OVER), not the grid and not our harness. No instrumented rerun is required; the original
"residual / needs-a-rerun" framing below is closed.

## What the code rules OUT

### 1. It is not a wrapper truncation
The wrapper truncates only at `max_steps`, default **1000**, for every game uniformly:
- `arc3_wm/env.py:50` — `max_steps: int = 1000`
- `arc3_wm/env.py:151` — `truncated = (not terminated) and self._steps >= self._max_steps`
- `arc3_wm/env.py:150` — `terminated = fd.state in TERMINAL_STATES`
- `arc3_wm/env.py:38` — `TERMINAL_STATES = frozenset({GameState.WIN, GameState.GAME_OVER})`

64 ≪ 1000, so the episode ends via `terminated=True` (engine state ∈ {WIN, GAME_OVER}), not via the
wrapper's step-limit. Since lf52's reward (`Δlevels_completed`, `env.py:145-147`) is **0 with zero
clears** in every run (see `analysis/ls20_lf52_audit.md` §2b/§2f), the terminal state is **GAME_OVER**,
not WIN.

### 2. There is no per-game horizon override anywhere in the launch path
- `scripts/launch_pergame.py:122` — `DEFAULT_ARC3_ENV = {"max_steps": 1000, ...}`
- `scripts/launch_pergame.py:362,366` — `max_steps = int(arc3_cfg.get("max_steps", 1000))` → passed
  straight to `ARC3EmbodiedEnv`.
- `scripts/launch_phase4_expansion_{warm,fromscratch}.sh:27` — `GAMES=(tn36 ls20 lf52)`; the only
  length-like flag is `--run.steps "${STEPS}"` (`…warm.sh:78`), which is the **total training-step
  budget**, not the per-episode horizon.

So lf52 inherits the same 1000-step wrapper horizon as sb26 and ls20. The 64 does **not** come from
config or wrapper.

### 3. It is not the global horizon — lf52 is not structurally capped at 64
The same wrapper produces wildly different episode lengths per game, and lf52's own **human replays**
(`data/replays/lf52/*.recording.jsonl`, read-only parse) run far past 64:

| game | human-replay episode length (min / med / mean / max) | terminal states seen | win_levels |
|------|------------------------------------------------------|----------------------|-----------|
| **lf52** † | **0 / 67 / 234 / 1604 actions** (1 / 68 / 235 / 1605 rows) | WIN | 10 |
| vc33 ‡ | 1 / 101 / 126 / 502 | WIN + GAME_OVER | 7 |
| cd82 ‡ | 1 / 47 / 60 / 241 | WIN + GAME_OVER | 6 |

† **lf52 recomputed canonically per-episode** via `arc3_wm.replay_loader.load_replay_file`
(`analysis/lf52_episode_lengths.csv`, generator `analysis/lf52_episode_lengths.py`): **n = 48
episodes** (11 session-files), RHAE `action_count` (`len(episode)-1`) min/median/mean/max =
**0 / 67 / 234 / 1604** (equivalently 1 / 68 / 235 / 1605 in step-rows). Of those 48, **4 reach
WIN, 0 GAME_OVER, 44 NOT_FINISHED**; 3 are phantom 1-row/0-action segments (excluding them: n = 45,
median 77). The earlier figure here — *"1 / 67 / 230 / 1605 over 49 episodes"* — came from the
gitignored `scratch/make_lf52_csvs.py` → `curves/` pipeline, which tallies lengths **per
session-file** and used a non-canonical 49-segment split (mean 230 = 11261 rows / 49); it is
superseded by the CSV. ‡ vc33/cd82 rows are the same legacy per-file estimates, **not** recomputed
under the canonical loader; treat as illustrative only.

**15 of the 48 canonical lf52 episodes exceed 200 actions; the longest is 1604 actions (1605
step-rows).** **No fixed 64-step horizon exists in the game** — if it did, no human episode could
exceed 64. (`max(win_levels)=10` confirms lf52 has 10 levels, so a single episode can legitimately
span many level-budgets.)

## What the logs SHOW about the agent's 64

From `analysis/ls20_lf52_audit.md` §2c/§2f (B2 + W&B, all 4 lf52 cells, both seeds, warm+cold):
- **Eval: exactly 64 ai_actions every episode, zero variance**, all cells.
- **Train: median 65, min 65, but a tail up to 236** (`67.6 / 65 / 65 / 186…236`).
- Policy is uniform-random the entire run (`train/rand/action ≡ 1.0`, `ent ≡ ln(4102)`); reward ≡ 0.

The **exact-64 eval invariance under a uniform-random policy** is the load-bearing observation.
A *varied agent-reached lose-state* (option b in its pure form) would produce **variable** lengths —
random actions would stumble into GAME_OVER at different step counts. Getting **exactly 64 every
single eval episode** instead means the GAME_OVER is triggered by a **fixed counter the engine
enforces** — an action *budget*, not an action *sequence*. That is structurally a **fixed cap**, but
one the engine emits as `terminated=True` (GAME_OVER) rather than the wrapper emitting `truncated`.

**Residual the code cannot reconcile:** a *pure* hard-64 GAME_OVER predicts train max ≈ 65 too, yet
train episodes reach 236 with zero clears. *(Resolved 2026-06-15 — see update above: the budget
counter `asqvqzpfdi` is incremented by directional/ACTION6 moves and undo (+20) but NOT by ACTION5
or special-region clicks, so budget-exempt actions stretch the env-step count past 64 on a minority
of episodes; the tail is real and smooth, with no 2×64 clustering.)*

> ⚠️ **Correction (2026-06-15):** an earlier version of this paragraph speculated that *"DV3's eval
> policy is lower-temperature/more-deterministic than train."* That is **false**: DreamerV3's
> `agent.policy(…, mode=…)` ignores the `mode` argument entirely (`dreamerv3/agent.py:115-135` —
> action is always `sample(policy)`), and the policy RNG seed is a monotonic `n_actions` counter
> (`embodied/jax/agent.py:232-234`), so eval draws fresh-random actions with the **identical**
> sampler as train. eval is not lower-temperature; the exact-64 is the engine budget (above), and
> the train/eval tail difference is the budget-exempt-action mechanism plus eval's small sample.

## What cannot be determined from code (honest limit)

> **SUPERSEDED 2026-06-15.** This section was written before the lf52 engine was fetched. The engine
> is now cached at `environment_files/lf52/271a04aa/lf52.py` and the exact trigger is confirmed (see
> the top-of-doc update). The text below is retained as a record of the pre-fetch state only.

lf52's engine class is **not cached locally** — `environment_files/` holds only `cd82, sb26, tu93,
vc33`; `find` for `lf52*.py` returns nothing. The action-space audit established the same
(`analysis/action_space_audit.md`). Therefore:
- The **exact GAME_OVER condition** (is it literally a per-level ~64-move budget? grid-dimension
  coincidence — the board is 64×64; a hard turn limit?) **cannot be read from code.**
- Confirming the budget value and whether it resets per level **requires the cached lf52 engine file
  or a single live rollout** with explicit `terminated`/state logging. Neither is permitted under
  this task's read-only/no-run constraint.

The candidate "64 = 64×64 grid dimension" is **speculative** — plausible (a turn limit set to grid
width) but unverifiable here; do not put it in the paper as fact.

## Classification

| Question | Answer | Basis |
|----------|--------|-------|
| Truncation or termination? | **Termination** (`terminated=True`, GAME_OVER) | `env.py:150-151` + 64 ≪ 1000 + reward≡0 |
| Same mechanism as sb26/ls20's ending? | **No.** sb26/ls20 = 1000-step wrapper **truncation**; lf52 = engine **GAME_OVER** | `ls20_lf52_audit.md` §2c (ls20=1001 truncation-bound) vs lf52=64 self-terminating |
| Fixed cap or agent-triggered lose-state? | **Fixed cap, expressed as termination** — engine action-budget GAME_OVER, not a varied death | exact-64 eval invariance under uniform-random; humans exceed 64 freely |
| Global to all games? | **No** — lf52-specific engine behaviour; wrapper horizon is 1000 for all | `env.py:50`, launch scripts |
| Exact trigger known? | **No** — engine not cached; needs cached file or live rollout | `find` (no `lf52*.py`); `environment_files/` lists 4 games |

## Decision surfaced for Haso

**lf52 belongs in the same *mechanistic* bucket as sb26/cd82/tn36 (uniform-random policy, no reward
gradient, WM-fits/controller-fails) — that part of the paper's framing is correct and is confirmed
for lf52 by the existing logs.** BUT lf52's zero is **partly a too-short-episode artifact that the
paper must describe differently on the horizon axis**:

- sb26/ls20 fail across a **1000-step** budget (wrapper truncation — the policy genuinely explores
  the full horizon and never commits).
- lf52 gets only a **fixed ~64-action budget per episode before the engine declares GAME_OVER** — a
  **15× shorter** exploration window, and a **different termination mechanism** (engine GAME_OVER vs
  wrapper truncation).

So the current prose — *"the policy stays uniform-random under sparse reward and never commits,"*
applied with the implication that lf52 *freely explores and fails like the others* — is **wrong for
lf52 on one axis**: lf52 does not get to freely explore; it is cut off at ~64 engine-enforced moves.
**Recommendation:** keep lf52 in the shared uniform-random/no-reward-gradient story, but add a
one-line carve-out: *"lf52 additionally terminates via an engine GAME_OVER at a fixed ~64-action
budget rather than the 1000-step truncation horizon, so its episodes afford ~15× less exploration
than ls20/sb26 — a contributing, partly structural cause of its zero distinct from the others."*
Do **not** assert the exact 64-move rule as fact (engine uncached); cite it as inferred, and if the
paper needs the precise trigger, that is the one thing requiring a cached-engine inspection or a
single instrumented live rollout.
