# Action-space audit — ground-truth valid action sets (6 pilot/expansion games)

**Date:** 2026-06-09
**Scope:** read-only engine introspection. No training, no wrapper changes, no paper edits, no push.
**Games:** `vc33, sb26, cd82, tn36, ls20, lf52`
**Motivation:** the paper makes per-game claims about valid action sets (esp. ls20 "valid set = 4
directional actions only", and a Method-section "5 games with the 4096-cell ACTION6 grid vs ls20
without"). These were reasoned, never verified against the engine. A frame of ls20 shows a visible
cursor that *might* be clickable — casting doubt on "4 only". This audit establishes ground truth.

## Method / sources

The per-game action mask is built from `fd.available_actions` — the integer action-TYPE IDs (1..7)
the engine exposes at every frame (`arc3_wm/env.py:193`, `arc3_wm/action_space.py:87 build_mask`).
Two independent sources, fully agreeing where both exist:

1. **Live engine** (authoritative) — `arc3_wm.env.ARC3GymEnv(...).reset()` in `OFFLINE` mode,
   reading `info["available_actions"]` and `info["action_mask"].sum()`. Available **only** for the
   3 games whose `environment_files/` are cached locally: `vc33`, `sb26`, `cd82`.
2. **Human replays** (`data/replays/<game>/*.recording.jsonl`) — every logged frame carries the
   `available_actions` array the engine produced for the human player. This covers **all 6 games**
   (including the 3 not cached locally: `tn36`, `ls20`, `lf52`). Scan extracted the RESET-row set,
   the union over all step rows, and whether the set ever varied within an episode.

**Cross-check:** for the 3 cached games the live-engine set and the replay set are byte-identical
(see per-game rows). The replay source is therefore trusted for the 3 uncached games.

**Within-episode stability:** for all 6 games the `available_actions` set is **constant across every
logged frame** (vc33 4536 frames, sb26 2549, cd82 2106, tn36 3874, ls20 7592, lf52 11262) — zero
files showed any intra-episode variation. So "the set at reset" fully characterises each game; no
actions unlock or disappear mid-episode in the logged data.

## Key limitation — the engine does NOT expose cell-level ACTION6 validity at reset

`available_actions` lists action **types** (1..7), not which of the 4096 ACTION6 cells are
meaningful. When type `6` is present, `build_mask` unmasks **all 4096** `(x,y)` cells uniformly
(`arc3_wm/action_space.py:100-101`). The engine accepts any `(x,y)` in `[0,63]²` without rejecting
it; whether a given cell *does anything* is a runtime/state question, not a reset-introspection one.
The optional probe below (live games only) measures this; it cannot be answered for tn36/ls20/lf52
without their `environment_files/`.

## Per-game results

`n_valid` = `info["action_mask"].sum()` = count of unmasked flat indices out of 4102.
Flat layout: `0–4`→ACTION1–5, `5–4100`→ACTION6 `(x,y)` grid (4096 cells), `4101`→ACTION7.

| Game | Valid action TYPES | ACTION1–5 | ACTION6 (4096-cell grid) | ACTION7 (undo) | n_valid / 4102 | Source |
|------|--------------------|-----------|--------------------------|----------------|----------------|--------|
| **vc33** | `{6}` | — | ✅ click grid | — | **4096** | live ≡ replay (10 files) |
| **sb26** | `{5,6,7}` | A5 only | ✅ click grid | ✅ | **4098** | live ≡ replay (12 files) |
| **cd82** | `{1,2,3,4,5,6}` | A1–A5 | ✅ click grid | — | **4101** | live ≡ replay (11 files) |
| **tn36** | `{6}` | — | ✅ click grid | — | **4096** | replay only (14 files) |
| **ls20** | `{1,2,3,4}` | A1–A4 | ❌ **not valid** | — | **4** | replay only (13 files) |
| **lf52** | `{1,2,3,4,6,7}` | A1–A4 | ✅ click grid | ✅ | **4101** | replay only (11 files) |

Notes:
- **vc33 / tn36** are pure click-grid games: ACTION6 only, no parameter-less actions.
- **sb26**: ACTION5 + click + undo. **cd82**: full directional set + ACTION5 + click, no undo.
- **ls20**: the four directional actions ONLY — `6` never appears in any of 7592 logged frames
  across 13 replays. The visible cursor is moved by ACTION1–4 (directional), it is **not** a click
  target. The "may be clickable" doubt is resolved: **not clickable**.
- **lf52**: 4 directional + click + undo. (RESET-row set detected in 9/11 files; the other 2 had no
  `id==0` RESET row at the head, but all 11262 frames across all 11 files show the identical
  `{1,2,3,4,6,7}` set, so the game-constant is unambiguous.)

## Dilution ratio (valid : total out of 4102)

| Game | n_valid | valid : total | total : valid ("1 useful per N") |
|------|---------|---------------|----------------------------------|
| vc33 | 4096 | 0.9985 : 1 | **1.0015 : 1** |
| sb26 | 4098 | 0.9990 : 1 | 1.0010 : 1 |
| cd82 | 4101 | 0.9998 : 1 | 1.0002 : 1 |
| tn36 | 4096 | 0.9985 : 1 | 1.0015 : 1 |
| **ls20** | **4** | 0.000975 : 1 | **≈ 1026 : 1** |
| lf52 | 4101 | 0.9998 : 1 | 1.0002 : 1 |

The paper's **"1024:1"** is the ls20 figure: 4102 / 4 = 1025.5 ≈ 1024. It is **correct for ls20 and
only ls20.** The 5 ACTION6 games have ~1:1 nominal dilution (the mask unmasks the whole click grid),
so a uniform "1024:1 dilution across the action space" claim would be wrong if applied to them.

## OPTIONAL state-change probe (live games only — vc33, sb26, cd82)

One step from a fresh reset per action type; ACTION6 sampled at 12 random cells (seed 0). Records
whether the decoded observation changed. This separates "engine lists it valid" from "it visibly
does something." **Only runnable for the 3 cached games** (needs the live engine).

| Game | Parameter-less probes | ACTION6 random-cell probe |
|------|-----------------------|----------------------------|
| vc33 | — | **12/12 cells changed obs** (dense, whole grid live) |
| sb26 | ACTION5 ✅ changed · ACTION7 ✗ no-op¹ | **2/12 cells changed obs** (sparse — most clicks are no-ops) |
| cd82 | ACTION1–4 ✅, ACTION5 ✅ all changed | **12/12 cells changed obs** (dense) |

¹ ACTION7 = undo. At the initial frame there is nothing to undo, so a no-op here is **expected and
does not indicate invalidity** — the engine still lists `7` as a valid type; a meaningful undo
requires a prior committed action.

**Hidden dilution, not visible from the mask:** sb26's ACTION6 grid is listed as 4096 valid cells
but only ~17% (2/12 sampled) actually move the state — the effective interactive surface is far
smaller than `n_valid` implies, and it is **per-game** (vc33/cd82 dense, sb26 sparse). The reset
mask cannot capture this; it is a runtime property. If the paper's dilution/masking prose leans on
"4096 valid click cells" as if all were meaningful, that over-counts the effective action space for
sparse-grid games like sb26. The cell-level effective count for tn36/ls20(n/a)/lf52 is **not
determinable** without their `environment_files/` — flagged as follow-up if needed.

## Verdict vs the paper's current action-space description

| Paper claim | Ground truth | Match? |
|-------------|--------------|--------|
| ls20 valid set = **4 directional actions only** (4 of 4102) | `{1,2,3,4}`, n_valid=4, no `6`/`7` in 7592 frames | ✅ **CORRECT** |
| ls20 cursor possibly clickable | `6` never valid → cursor is directional, not a click target | ✅ doubt **resolved (not clickable)** |
| **5 games include the 4096-cell ACTION6 grid, ls20 does not** | ACTION6 ∈ vc33, sb26, cd82, tn36, lf52 (5); ∉ ls20 (1) | ✅ **CORRECT (5-vs-1 holds)** |
| "1024:1" dilution | ls20 = 4102/4 ≈ 1026:1 ≈ 1024 | ✅ correct **for ls20**; ⚠️ do **not** generalise to the 5 ACTION6 games (~1:1) |

### Flags to surface
- **No type-level mismatches.** Every action-TYPE claim the paper makes about these 6 games matches
  ground truth: ls20 "4 only" ✅, the 5-vs-1 ACTION6 split ✅, the "1024:1" ls20 dilution ✅.
- ⚠️ **Scope the "1024:1" to ls20.** If the prose implies it across the action space generally, it
  is wrong for the 5 ACTION6 games (nominal dilution ~1:1 there).
- ⚠️ **"4096 valid click cells" ≠ 4096 meaningful cells.** Live probe: sb26 click grid is sparse
  (~2/12), vc33/cd82 dense (12/12). The mask treats all cells equally; the effective interactive
  surface is per-game and smaller for sparse games. Not resolvable for tn36/lf52 from reset alone
  (env files not cached) — follow-up if the paper needs effective-cell counts.

## Reproduction
- Replay scan: iterate `data/replays/<game>/*.recording.jsonl`, read `data.available_actions`.
- Live: `ARC3GymEnv(game_id=g, seed=0).reset()` → `info["available_actions"]`,
  `info["action_mask"].sum()` (OFFLINE mode; `environment_files/<game>/` must be cached).
- Cached locally as of this audit: `vc33, sb26, cd82, tu93`. Not cached: `tn36, ls20, lf52`.
