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
not a varied agent-reached death. **The precise GAME_OVER trigger cannot be confirmed from code**
because lf52's `environment_files/` are not cached locally; the classification below is what code +
logged data jointly support, with the residual flagged.

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
| **lf52** | **1 / 67 / 230 / 1605** | WIN (humans solve it) | 10 |
| vc33 | 1 / 101 / 126 / 502 | WIN + GAME_OVER | 7 |
| cd82 | 1 / 47 / 60 / 241 | WIN + GAME_OVER | 6 |

15 of 49 lf52 human episodes exceed 200 steps; the longest is 1605. **No fixed 64-step horizon
exists in the game** — if it did, no human episode could exceed 64. (`max(win_levels)=10` confirms
lf52 has 10 levels, so a single episode can legitimately span many level-budgets.)

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
train episodes reach 236 with zero clears. Candidate explanations (none confirmable without the
engine): the budget may reset on within-level events that don't register as a `Δlevels` clear; or
DV3's eval policy is lower-temperature/more-deterministic than train, pinning eval to the modal 64
while train's higher stochasticity occasionally extends. This is exactly why the precise trigger is
flagged as **not determinable from code** below.

## What cannot be determined from code (honest limit)

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
