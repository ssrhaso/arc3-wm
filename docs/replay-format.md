# Replay JSONL format (Phase 0)

> Schema of the human-replay `*.recording.jsonl` files distributed in the
> "ARC-AGI-3 Human Baseline [Public]" Drive folder, plus the closely-related
> recordings the toolkit writes when `save_recording=True` is passed to
> `arc.make()`. See [docs/harness-analysis.md](harness-analysis.md) for the
> live-API counterpart.

## File layout

After `gdown --folder` from the public-demo Drive folder:

```
data/replays/
  ARC-AGI-3 Human Baseline [Public]/
    arc_agi_3_public_demo_human_testing/
      ar25/
        00589449-cfb4-4b98-a31e-2a344db66f89.recording.jsonl
        ...
      bp35/
      cd82/
      cn04/
      ...
```

One subdirectory per `game_id` (the 25 public-demo IDs from
`docs/arc-agi-3/available-games.md`). Each `.recording.jsonl` is one human
play session (one `guid`).

> **Phase 0 download status (this session):** 39 of 342 files retrieved before
> Google Drive's per-IP "too many accesses" quota started rejecting URLs.
> Files captured cover `ar25` (10), `bp35` (14), `cd82` (11), `cn04` (4).
> A bundled `arc_agi_3_public_demo_human_testing.zip` exists in the Drive
> folder and would short-circuit the per-file quota, but it is also blocked by
> the same quota (verified). See "Open issue: Drive quota" below.

## Per-line schema

Each line is a UTF-8 JSON object:

```json
{
  "timestamp": "ISO-8601 UTC string",
  "data": { ... payload ... }
}
```

The wrapper writes one line per `_set_last_response` call: one for the
initial `reset()` (RESET action), then one per `step()`. There is also one
trailing summary line at end-of-session whose `data` has a different shape
(see "Trailing session-summary line" below).

### `data` payload (per-step rows)

Keys verified across `ar25/00589449…jsonl` (1556 lines) and
`cd82/0c6d47d7…jsonl`:

| Field | Type | Notes |
|---|---|---|
| `game_id` | string | e.g. `"ar25-0c556536"` (game-id + version hash) |
| `guid` | string | per-session UUID, constant across all rows in a file |
| `state` | string | `"NOT_FINISHED"`, `"WIN"`, `"GAME_OVER"`; `arcengine.GameState.name` |
| `levels_completed` | int | monotonically non-decreasing within a session |
| `win_levels` | int | total levels in the game (`info.win_levels`) |
| `available_actions` | list[int] | subset of `[1..7]`, dynamic per-step |
| `full_reset` | bool | `true` only on a hard reset (not seen in sample data — always false) |
| `action_input` | object | `{id, data, reasoning}` of the most-recent action |
| `frame` | list[list[list[int]]] | `frame[layer][row][col]` palette index in `[0..15]`; layer count varies (see harness-analysis.md) |

`action_input` shape:

```json
{
  "id": 5,                              // GameAction value: 0=RESET, 1..7=ACTION1..ACTION7
  "data": {"game_id": "cd82-fb555c5d"}, // simple actions: just game_id; ACTION6: includes "x", "y"
  "reasoning": null                     // dict|null; agents may attach reasoning logs
}
```

Note: in the live `arc_agi.wrapper.EnvironmentWrapper._set_last_response` code
the recorded `action_input.id` is written as the **enum name string** (e.g.
`"ACTION1"`). In the human replays we downloaded, `action_input.id` is the
**integer value** (e.g. `5`). This means **two distinct serialisations
exist** — the replay loader must accept both. Verified in:

- `arcengine.../local_wrapper.py` `_set_last_response` writes `id.name if hasattr(id, "name") else str(id)` → string.
- Sampled human JSONLs ar25/cd82: `id` is integer.

The replay loader (Phase 1) must coerce: `int(id)` if int, otherwise `GameAction[id].value`.

### Trailing session-summary line

The very last line in `ar25/00589449…jsonl` (line 1557 of 1557) has a
different `data` payload — no `frame`, no `action_input`. It looks like a
session footer summarising the run:

```
data keys (last line): {'levels_completed', 'won', 'played', 'total_actions', 'cards'}
```

Replay loaders should detect this row by absence of `frame`/`action_input`
and treat it as metadata, not a transition.

## Action / observation tuple for the buffer

To turn one JSONL into DreamerV3 transition tuples `(o_t, a_t, r_t, done_t,
o_{t+1})`:

1. **State `o_t` (image obs):** apply the same layer-selection + palette-decode
   policy as the live wrapper (default proposal: `frame[-1]` → palette → `(64, 64, 3) uint8`).
2. **Action `a_t`:** map `(action_id, data.x, data.y)` to the flat index 0–4101
   (see CLAUDE.md §"Action space"). For ACTION1..5 / ACTION7, indices 0..4 / 4101.
   For ACTION6, use `5 + 64*y + x`.
3. **Reward `r_t`:** `levels_completed[t+1] - levels_completed[t]` is the simplest
   level-up reward signal. Confirm with Haso whether to additionally use
   terminal +/- bonuses. Methodology says env's "native level-up rewards" —
   that's `Δ levels_completed`.
4. **Done `done_t`:** `state[t+1] in {"WIN", "GAME_OVER"}` OR `t+1 == last_per-step row`.
5. **Episode boundary:** within a single JSONL, `levels_completed` only increases.
   When a player resets after losing or to retry a level, the engine increments
   `resets` (visible in scorecard but not in replay rows directly). The
   `full_reset` field appears reserved for that — but it was `false` on every
   line of the 1557-line ar25 sample, so resets in replays may be marked
   differently. Phase 1 loader should handle: end of file = end of episode.

### Worked example — `ar25/00589449….recording.jsonl`, first 5 rows

Annotated. Frame data abbreviated.

```jsonc
// Line 0 — RESET frame (action_input.id == 0)
{
  "timestamp": "2025-11-10T17:36:18.020120+00:00",
  "data": {
    "game_id": "ar25-0c556536",
    "frame": [ /* list[1] of (64, 64) palette ints; mostly 9 (background), with stripes of 5/4 etc. */ ],
    "state": "NOT_FINISHED",
    "action_input": { "id": 0 /*=RESET*/, "data": {"game_id": "ar25-0c556536"}, "reasoning": null },
    "guid": "00589449-cfb4-4b98-a31e-2a344db66f89",
    "full_reset": false,
    "available_actions": [1, 2, 3, 4, 5, 6, 7],
    "levels_completed": 0,
    "win_levels": ...
  }
}

// Line 1 — first action (ACTION3, no x/y)
{
  "timestamp": "2025-11-10T17:36:19....",
  "data": {
    "game_id": "ar25-0c556536",
    "frame": [/* layers reflecting post-action world */],
    "state": "NOT_FINISHED",
    "action_input": { "id": 3, "data": {"game_id": "ar25-0c556536"}, "reasoning": null },
    "guid": "00589449-cfb4-4b98-a31e-2a344db66f89",
    "full_reset": false,
    "available_actions": [1, 2, 3, 4, 5, 6, 7],
    "levels_completed": 0,
    "win_levels": ...
  }
}

// Lines 2–4 follow the same pattern, with action_input.id varying across {1..6} and
// occasionally id=6 (ACTION6) with data containing {"x": ..., "y": ...}
```

### Aggregate stats (1557-line ar25 sample)

```
size:                    21,212,415 bytes
lines:                   1557 (1556 step rows + 1 trailing summary)
top-level keys:          {timestamp, data}
state counts:            {NOT_FINISHED: 1553, GAME_OVER: 3}
action counts (id):      {0: 7, 1: 230, 2: 300, 3: 424, 4: 447, 5: 14, 6: 134}
                         (RESET=0 appears 7 times: 1 initial + 6 mid-session resets)
levels_completed range:  0..7 (all 7 levels eventually completed)
full_reset events:       0
available_actions:       [1, 2, 3, 4, 5, 6, 7] consistently in this sample
frame layout:            list[1] of (64, 64) on most rows
```

> **WARNING on layer count in replays.** The cd82 replay sample
> (`cd82/0c6d47d7…recording.jsonl`) has multi-layer frames (e.g. line 1 has
> `list[15]` after an ACTION5), matching the live API. Replays preserve the
> animation layers — confirm in Phase 1 that the layer-selection policy
> applied during DreamerV3 buffer construction matches the policy applied
> at training time on the live env. If they diverge, the buffer state
> distribution will not match the runtime distribution.

## Schema variation across games

Sampled three games (ar25, bp35, cd82, cn04) — primary fields identical;
`available_actions` and per-(game, action) frame-layer counts vary per game.
Definitive cross-game schema inspection is gated on completing the Drive
download — see open issue.

## Open issue: Drive quota (Phase 0, must resolve before Phase 1)

Google Drive returns "Cannot retrieve the public link of the file. You may
need to change the permission to 'Anyone with the link', or have had many
accesses." after ~39 files. The bundled
`arc_agi_3_public_demo_human_testing.zip` (file id
`1aJmVxDPEyQ7m-FUVqHXCU_LcJGsnmBuk`) is also rate-limited.

Possible resolutions, ordered by friction:

1. **Wait 24–48 h** for Google's per-IP quota to reset, then re-run
   `gdown --folder --continue` (it skips already-downloaded files).
2. **Make a personal Drive copy.** Sign in to Drive, "Make a copy" of the
   bundled zip into Haso's own Drive, download from there. Bypasses share
   quota.
3. **Authenticated gdown.** `gdown --use-cookies` with a logged-in browser
   session export. Requires Haso's browser cookies.
4. **Mirror.** If Arc Prize publishes the bundle elsewhere (HuggingFace,
   GitHub release, S3), use that instead — currently unknown; flag.
5. **Manual download.** Open the Drive folder in a browser and "Download"
   the zip; manual but reliable.

This is a Phase-0 deliverable failure (count is 39, not 342). All
downstream tasks that depend on the full replay set (Phase 1 `replay_loader`
test that parses all 342, Phase 3 cross-game pretraining) are blocked until
this resolves. Surface to Haso in the session summary.
