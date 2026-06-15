# Per-task action-diagnostic API — validated findings

**Date:** 2026-06-15
**Module:** `arc3_wm/action_diagnostics.py` · **CLI:** `scripts/action_diagnostics.py`
**Scope:** read-only engine introspection + one checkpoint inspection. No training,
no DreamerV3 fork, no paper edits, no push.

This is the report-back for the task-id → action-diagnostic interface. It records
what was checked against ground truth (engine source, a real checkpoint, the
logged metric keys) rather than taken from the paper. Everything below was
re-derived this session; the code carries the same facts as tests.

---

## 0. The non-negotiable: (a) structural vs (b) empirical, never conflated

The API returns **(a)** the structural/validity profile populated from the engine,
and a **(b)** `usage_count`/`usage_fraction` slot that is **null unless** a real
per-step action log from an instrumented rollout is supplied. (b) is never
synthesised, inferred, or backed out of aggregate statistics. Each field carries a
per-field `source` tag: `engine` / `engine-default` / `engine-probe` /
`run-measured` / `absent`.

## 1. Where the engine exposes per-task action validity

- **Action-TYPE validity is the only thing exposed.** `FrameData.available_actions`
  (engine `enums.py:140`, surfaced by our wrapper at `arc3_wm/env.py:207`) is a list
  of integer action **type** ids `1..7`. `build_mask` projects it onto the flat 4102
  space (`arc3_wm/action_space.py:101`). An index is `valid_on_task` iff its action
  type is in that set; ACTION6's 4096 cells all share the type-level validity of id 6.
- **Cell-level ACTION6 validity is NOT exposed at the API.** The engine *does* know
  which cells are meaningful — `ARCBaseGame._get_valid_clickable_actions`
  (`arcengine/base_game.py:522`) enumerates the clickable/placeable cells from
  `sys_click`/`sys_place` sprites — but `_get_valid_actions` is documented
  "**for internal use only, the data here is never exposed via the API or to
  Users/Agents**" (`base_game.py:484-486`). So cell-level validity is a runtime
  property, not a reset-introspection one. This matches the prior audit
  (`analysis/action_space_audit.md`).

## 2. valid / no-op / state-changing — three distinct properties, not two

They are genuinely three tiers, and the schema keeps `valid_on_task` and
`is_state_changing` as **independent booleans** so none collapse:

| Tier | `valid_on_task` | `is_state_changing` | Example |
|------|-----------------|---------------------|---------|
| A — masked / unavailable | **False** | False | ls20's ACTION6 grid (id 6 ∉ available) |
| B — valid but inert | **True** | **False** | sb26 sparse click cells; undo with nothing to undo |
| C — valid and state-changing | True | True | vc33 click grid; ls20 ACTION1–4 |

Tier B (valid-but-inert) is representable via `inert_indices` (source
`engine-probe`) — it requires a live state-change probe to populate honestly and is
**never** assumed. Absent a probe, every valid index defaults to state-changing
(`n_state_changing == n_valid`), which is the honest structural answer: at
reset-introspection the engine cannot tell an inert valid cell from a live one.

**Provenance of `is_state_changing`** — the source tag distinguishes the three
cases so the default is never mistaken for engine-confirmed truth: an *invalid*
action's `False` is tagged `engine` (it follows directly from `available_actions`);
a *valid, un-probed* cell's default `True` is tagged **`engine-default`** (a
structural inference, not read from the engine's per-cell clickable introspection);
a *probed inert* cell is tagged `engine-probe`. (`probe_state_change`, the live
engine path that would upgrade `engine-default` → `engine`/`engine-probe` per cell,
is intentionally not built yet — it is tied to a real run.)

### ⚠ Reconciliation flag for the ls20 prose

The task brief describes ls20's click grid as *"valid-but-inert — present, accepted,
but changes nothing"* (Tier B). **The engine disagrees:** id 6 never appears in any
of ls20's 7,605 logged frames across 13 replays (`available_actions_from_replays
("ls20") == [1, 2, 3, 4]`). Per the engine ls20's click grid is **Tier A
(masked/unavailable)**, not Tier B. The engine still *accepts* an out-of-mask
ACTION6 without raising (`perform_action` only rejects in WIN/GAME_OVER), so "present,
accepted, changes nothing" is true *operationally* — but it is **not** a valid action
on the task. This does not change the headline number (§5): ls20 is still 4/4102.
Genuine Tier B examples live elsewhere (sb26 sparse cells, undo-at-reset), and the
schema represents them.

## 3. Budget-cost weights — confirmed, but lf52-specific (NOT global)

The paper's **move=1 / undo=20 / no-op=0** weights are **real and confirmed against
engine source**, but they are **lf52's internal survival-budget counter**
`asqvqzpfdi`, not a universal action cost:

- `environment_files/lf52/271a04aa/lf52.py`: directional move `tmhxwcojkh` → `+1`
  (`:5277`); ACTION6 grid-click `dghsidbuet` → `+1` (`:5335`); undo on commit → `+20`
  (`:5805`); ACTION5 and special-region ACTION6 click → `+0` (`:5327`, `:5832`).
  When the counter hits the per-level budget (L1 = 64) the engine calls `lose()` →
  GAME_OVER (cf. `analysis/lf52_termination.md`).
- **The global action accounting is uniform +1.** `arc_agi/scorecard.py`
  `Card.inc_action_count` (`:720`) / `Scorecard.take_action` (`:844-845`) increment by
  exactly 1 for every action id `1..7`. This is the RHAE action count.

So `budget_cost` is modelled per game: `UNIFORM_BUDGET` (every type → 1, the default
and the RHAE truth) vs `LF52_BUDGET` (the confirmed survival weights). The source tag
records which (`engine:uniform` | `engine:lf52`). The lf52 special-region-click `+0`
is a **cell-level** exemption that the type-level ACTION6 cost (1) cannot express —
flagged, not hidden.

## 4. (b) data absence — confirmed against a real checkpoint

Per-step action streams **do not exist anywhere** in the current pipeline:

- **Checkpoint replay buffers pickle to `None`.** `scratch/gate_check/
  20260517T114009F383352/replay_{train,eval}.pkl` are **4 bytes each → `None`**
  (verified by `pickle.load`). No transitions, no action indices.
- **Only aggregates are logged.** The Phase-4 `metrics.jsonl` (61 distinct keys) has
  exactly two action-related keys: `train/ent/action` (entropy scalar) and
  `train/rand/action` (random-action fraction). There is **no** per-action histogram,
  index stream, or table. These two are precisely the scalars the honest-null contract
  forbids reverse-engineering usage from — and the builder exposes no parameter that
  could carry them.

Conclusion: until an instrumented rollout writes a real `{"actions": [...]}` log,
`usage_count`/`usage_fraction` stay null with source `absent`. That is the only
honest state.

## 5. (a) sanity-check vs published Fig-3 panel-A numbers

Engine-derived, this session, via `from_env` (cached) / `from_replays` (uncached):

| Game | source | `n_valid` | `dilution_ratio` (= 4102 / n_valid) | Fig-3 claim | Match? |
|------|--------|-----------|-------------------------------------|-------------|--------|
| **ls20** | replays | **4** ({1,2,3,4}) | **1025.5** ≈ 1025:1 | "4/4102", "~1024:1" | ✅ number matches (prose flag §2) |
| **lf52** | env | **4101** ({1,2,3,4,6,7}) | 1.0002:1 | budget weights move/undo/no-op | ✅ weights confirmed §3 |
| vc33 | env | 4096 ({6}) | 1.0015:1 | — | ✅ vs audit |
| sb26 | env | 4098 ({5,6,7}) | 1.0010:1 | — | ✅ vs audit |
| cd82 | env | 4101 ({1,2,3,4,5,6}) | 1.0002:1 | — | ✅ vs audit |

No disagreement on the numbers. The one flag is the **mechanism description** of
ls20's click grid (§2), not the count.

## 6. The validated schema — real examples

Top level: `{task_id, n_valid, n_state_changing, dilution_ratio, actions: [...]}`
where the rollups are derived from the rows (never hardcoded). Per action row:

```jsonc
// vc33 — an ACTION6 click-grid game (live engine). Click cell idx 5 is valid.
{"action_index": 5, "action_type": "ACTION6", "valid_on_task": true,
 "is_state_changing": true, "budget_cost": 1, "usage_count": null,
 "usage_fraction": null,
 "sources": {"valid_on_task": "engine", "is_state_changing": "engine-default",
             "budget_cost": "engine:uniform", "usage_count": "absent",
             "usage_fraction": "absent"}}
```

```jsonc
// ls20 — directional-only (replays). ACTION1 valid; the ACTION6 grid is Tier A.
{"action_index": 0, "action_type": "ACTION1", "valid_on_task": true,
 "is_state_changing": true, "budget_cost": 1, "usage_count": null,
 "usage_fraction": null, "sources": {"valid_on_task": "engine", ...}}
{"action_index": 5, "action_type": "ACTION6", "valid_on_task": false,
 "is_state_changing": false, "budget_cost": 1, "usage_count": null,
 "usage_fraction": null, "sources": {"valid_on_task": "engine", ...}}
```

```jsonc
// lf52 — budget model resolved automatically. undo (idx 4101) costs 20.
{"action_index": 4101, "action_type": "ACTION7", "valid_on_task": true,
 "is_state_changing": true, "budget_cost": 20, "usage_count": null,
 "usage_fraction": null, "sources": {"budget_cost": "engine:lf52", ...}}
```

CSV is emitted from the same rows; `sources` flattens to `<field>_source` columns and
null usage renders as an empty cell.

## 7. How the collaborator runs it

```bash
# directional-only game from replays (no cached engine needed)
python scripts/action_diagnostics.py --game-id ls20 --source replays \
    --out-json results/ls20_actions.json --out-csv results/ls20_actions.csv

# cached game from the live engine, with measured usage from an instrumented run
python scripts/action_diagnostics.py --game-id vc33 --source env \
    --action-log path/to/actions.jsonl --out-json results/vc33_actions.json
```

The action log is the pinned format a future instrumented rollout must emit: JSONL,
one `{"actions": [flat_idx, ...]}` object per episode (same family as
`arc3_wm.eval_reward_sink`'s reward log). Until that exists, omit `--action-log` and
the (b) columns are honestly null.

## Reproduction
- Engine validity: `arc3_wm.action_diagnostics.from_env(game)` (OFFLINE, cached
  `environment_files/`) / `from_replays(game)` (any of the 25 games).
- Budget weights: `grep -n asqvqzpfdi environment_files/lf52/271a04aa/lf52.py`.
- (b) absence: `pickle.load(open('scratch/gate_check/.../replay_train.pkl','rb'))` →
  `None`; scan `scratch/p4-vc33-dryrun/metrics.jsonl` keys.
- Tests: `pytest tests/test_action_diagnostics.py tests/test_action_diagnostics_cli.py`.
