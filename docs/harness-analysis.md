# `arc_agi` Harness Analysis (Phase 0)

> Findings from poking at `arc_agi==0.9.8` / `arcengine==0.9.3` in a Python REPL,
> cross-referenced with `docs/arc-agi-3/`. All paths in this doc are relative
> to the project root unless noted.

## TL;DR (load-bearing)

| Question | Answer |
|---|---|
| Does `get_scorecard()` return RHAE? | **Yes**, on a 0–100 scale (per-level cap 115.0, not 1.15). Per-level scores, per-level actions, per-level baselines, per-game weighted score, mean across games — all present. `arc3_wm/rhae.py` not needed for primary scoring. Optional sanity-check helper still useful. |
| Observation shape vs. CLAUDE.md’s `(64, 64, 3) uint8`? | **Wrong shape, and the layer count is variable per action and per game.** Real obs is `list[ndarray((64, 64), int8)]` — palette indices in `0..15`, not RGB. The list length is the number of in-engine animation ticks during the step: vc33 always 1; tu93 ACTION4 → 8 layers; cd82 ACTION5 → 15 layers; everything else 1 layer. Wrapper must (a) decide whether to feed `frame[0]`, `frame[-1]`, or a stack/window, and (b) palette-decode to `(64, 64, 3) uint8` before feeding DreamerV3. **Decisions Haso owns — open question.** |
| `OFFLINE` mode active? | Yes. `OPERATION_MODE=offline` in `.env`, auto-loaded by `arc_agi.base` at import. Verified: `arc.operation_mode == OperationMode.OFFLINE`, `make()` succeeds, no API call made. |
| Action 6 coordinate range? | 64×64 confirmed for vc33/tu93/cd82; `data={"x": int, "y": int}` with both in `[0, 63]`. |
| Unsupported actions? | **Silently no-op'd.** No exception; `state` stays `NOT_FINISHED`; `available_actions` unchanged. Without action masking the policy will waste capacity sending no-ops. |
| `ACTION6` with no `data`? | **Also silent** — does not raise. The engine substitutes a default. Bug surface if the wrapper forgets to attach coords. |
| Episode termination? | Per-game. `state ∈ {NOT_FINISHED, WIN, GAME_OVER, ...}`. `levels_completed` advances within an episode; on GAME_OVER call `env.reset()` to start a new episode (the engine increments `resets` and starts a new run within the same scorecard). |

## Versions

```
arc-agi==0.9.8
arcengine==0.9.3
python==3.12.10  (project venv)
```

`arc_agi` does **not** expose `__version__` at module level — use
`importlib.metadata.version("arc-agi")`. CLAUDE.md’s
`python -c "import arc_agi; print(arc_agi.__version__)"` will `AttributeError`;
use `importlib.metadata.version` instead.

## Operation modes (verified empirically)

`arc_agi.base.Arcade._parse_operation_mode_from_env()` reads `OPERATION_MODE`
case-insensitively and accepts one of `{"normal", "online", "offline", "competition"}`.
`arc_agi.base` runs `load_dotenv(".env")` then `load_dotenv(".env.example")` at
import time, so the file must be at CWD when `arc_agi` is first imported.

| Mode | What `make()` does | Network? | API key? |
|---|---|---|---|
| `OFFLINE` | `_find_local_game()` only — must hit local `environment_files/<game>/<version>/metadata.json` | none | optional |
| `NORMAL` (default) | `_download_game()`: fetch metadata + source from API on first call, cache to `environment_files/`, then run locally | on first `make` per game-version | required |
| `ONLINE` | `_create_remote_wrapper()` — every `step` is an HTTP call | every step | required |
| `COMPETITION` | Same plumbing as ONLINE plus competition gating | every step | required |

**Priority quirk in `Arcade.__init__`:** the env var beats the constructor arg
*unless* the constructor arg is non-`NORMAL`. Equivalent: you cannot force
NORMAL programmatically when env says OFFLINE. To re-cache more games, set
`os.environ["OPERATION_MODE"] = "normal"` *before* importing `arc_agi`. See
[scripts/cache_env_files.py](../scripts/cache_env_files.py) for the pattern.

## OFFLINE prerequisite — `environment_files/`

`OFFLINE` reads `environment_files/<game>/<version>/metadata.json` and dynamically
`exec()`s `<game>.py` from the same dir to load the `ARCBaseGame` subclass.
`arc_agi` and `arcengine` do **not** ship game source. Two ways to populate it:

1. Run `make()` once in `NORMAL` mode for each game (this calls `_download_game`
   and writes both `metadata.json` and `<game>.py`). Done in Phase 0 for
   vc33/tu93/cd82 via `scripts/cache_env_files.py`.
2. Tarball + ship — once cached on the laptop, `tar czf env_files.tar.gz environment_files/`
   and `curl … | tar xz` on each remote instance (Vast.ai). See
   [docs/compute-runbook.md](compute-runbook.md).

`environment_files/` and `recordings/` are gitignored (see `.gitignore`).

## Wrapper API surface

`arc.make(game_id, …)` returns a `LocalEnvironmentWrapper` (OFFLINE/NORMAL) or
`RemoteEnvironmentWrapper` (ONLINE/COMPETITION). Both inherit
`EnvironmentWrapper` and expose:

```
env.reset() -> FrameDataRaw
env.step(action: GameAction, data: dict|None=None,
         reasoning: dict|None=None) -> FrameDataRaw
env.action_space  # property: list[GameAction] from last response (DYNAMIC)
env.observation_space  # property: last FrameDataRaw  (also DYNAMIC, NOT a Gym Space)
env.info  # EnvironmentInfo (game metadata)
```

**Both `action_space` and `observation_space` are misleadingly named** — they
are not Gym `Space` objects but the most recent action list / frame.
The arc3 wrapper must build proper `gymnasium.spaces.Discrete(4102)` and
`gymnasium.spaces.Box(0, 255, (64,64,3), uint8)` and translate.

### `FrameDataRaw` fields (verified by introspection)

```
game_id            str           e.g. "vc33-5430563c"
guid               str           per-episode UUID
state              GameState     {NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER, …}
levels_completed   int           monotone within an episode
win_levels         int           total levels in the game
available_actions  list[int]     subset of [1..7]; CHANGES across steps
full_reset         bool          True only on the reset frame
action_input       ActionInput   last (id, data, reasoning)
frame              list[ndarray]  list[L] of int8 (H, W) palette indices
```

There is **no `reward` or `score` field** on `FrameDataRaw`. The training
reward must be derived (delta `levels_completed`, level-up bonus, terminal
`WIN`/`GAME_OVER`) — exactly what CLAUDE.md anticipated.

### `EnvironmentInfo` fields (load-bearing for RHAE)

```
game_id, title, class_name, tags, level_tags, default_fps,
baseline_actions: list[int]   # human upper-median, per level
local_dir, date_downloaded
```

`baseline_actions` is the level-by-level human baseline. **The toolkit ships it
in `metadata.json`.** No need to derive it from JSONL replays for primary RHAE.

## Action space (verified)

`from arcengine import GameAction` is an `IntEnum`:

```
RESET=0, ACTION1=1, ACTION2=2, ACTION3=3, ACTION4=4,
ACTION5=5, ACTION6=6, ACTION7=7
```

`GameAction.is_complex()` returns True only for `ACTION6`. ACTION6 expects
`data={"x": int, "y": int}`; coordinates are in `[0, 63]` for vc33/tu93/cd82.

### `list_actions` analog

There is no separate `list_actions` method on the wrapper. The currently-available
actions for the next step are exposed two ways:
- `env.action_space` → `list[GameAction]` (decoded names, but rebuilt every call).
- `env.observation_space.available_actions` → `list[int]` (raw IDs from the last frame).

Both come from `_last_response.available_actions`, which the engine updates on
every step. **Available actions can change mid-episode.** Per-step masking,
not per-episode masking, is the correct policy.

### Per-game initial available_actions

Verified after `make()` + initial `reset` (no other steps yet):

| Game  | `available_actions` after reset | Levels | Tags             | baseline_actions per level                        |
|-------|---------------------------------|--------|------------------|---------------------------------------------------|
| vc33  | `[6]`                           | 7      | `['click']`      | `[7, 18, 44, 61, 131, 34, 152]`                   |
| tu93  | `[1, 2, 3, 4]`                  | 9      | `['keyboard_click']` | `[19, 16, 34, 42, 123, 80, 14, 23, 111]`     |
| cd82  | `[1, 2, 3, 4, 5, 6]`            | 6      | `['keyboard_click']` | `[55, 8, 41, 21, 23, 23]`                    |

Notes:
- **None of the three pilot games expose `ACTION7` initially.** It may unlock
  later in an episode (the docs say ACTION7 is always undo *for games that
  support it*). Phase 1 should test for `7 in available_actions` mid-episode
  before treating it as universally absent.
- vc33’s level 1 fails on the first wrong click → GAME_OVER in ~50 actions
  observed during the smoke. Random click policy reaches level 0 only.
- ACTION6 grid: 64×64 documented, observed ranges in valid clicks all `[0,63]`.

### Unsupported-action behaviour (verified)

- `step(GameAction.ACTION7)` on vc33 (ACTION7 not in `available_actions`):
  returns a normal `FrameDataRaw`; `state` unchanged; `available_actions`
  unchanged; the action **does NOT raise**. Whether it counts toward the
  scorecard `actions` total — to verify in Phase 1; current observation is
  that the engine treats unsupported simple actions as no-ops but they still
  appear to advance step count. **Mask before sampling.**
- `step(GameAction.ACTION6)` with `data=None`: also silent — engine substitutes
  a default. The wrapper must enforce explicit `(x, y)` for ACTION6.

## Observations (variable layer count — load-bearing)

Observed across vc33/tu93/cd82:

```
fd.frame is list[np.ndarray]   shape per layer = (64, 64), dtype = int8, values ∈ [0, 15]
len(fd.frame)  varies per (game, action), from 1 to 15
```

The list is a **sequence of intra-step animation frames**: each layer is a
discrete tick of the engine evolving the world while the action is being
applied. Sampled deltas (cd82 ACTION5):

```
layer0 -> layer1: 40 cells differ (of 4096)
... 14 transitions, all 30–40 cells changing ...
first vs last: 50 cells differ
```

Per-(game, action) layer counts (one transition each, from a fresh reset):

| Game | RESET | ACTION1 | ACTION2 | ACTION3 | ACTION4 | ACTION5 | ACTION6 | ACTION7 |
|------|-------|---------|---------|---------|---------|---------|---------|---------|
| vc33 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| tu93 | 1 | 1 | 1 | 1 | **8** | 1 | 1 | 1 |
| cd82 | 1 | 1 | 1 | 1 | 1 | **15** | 1 | 1 |

Implications:

1. The wrapper **cannot assume a fixed observation shape from `fd.frame`**.
2. `fd.frame[-1]` is the post-action settled world state — natural choice
   for a single-frame DreamerV3 obs.
3. `fd.frame[0]` is the immediate-post-action frame; for non-animating
   transitions `frame[0] == frame[-1]`. So `frame[-1]` strictly dominates
   if we only feed one frame.
4. A frame stack (e.g., last K layers, padded to fixed size) preserves the
   animation but adds complexity and changes the encoder input channels.
5. CLAUDE.md’s “stock DreamerV3 + stock RGB” line implies single-frame RGB.
   Default plan: take `frame[-1]`, palette-decode to `(64, 64, 3) uint8`.

**Open question for Haso (Decision §"Action-space change" or new):** which
layer-selection policy? Default proposal is `frame[-1]`, but we should
verify across ALL 25 public-demo games before locking it in (a Phase 1
test).

The arc3 wrapper must therefore:
1. Pick a layer-selection policy (default proposal: take `frame[-1]`).
2. Map palette indices `0..15` to RGB via the documented 16-colour palette.
3. Output `(64, 64, 3) uint8` for DreamerV3’s `image` modality.

**Per CLAUDE.md, encoder is stock DreamerV3 with stock RGB.** Use the canonical
ARC palette mapping (16 colours). Fetch from `arcengine` constants if exposed,
otherwise hardcode in `arc3_wm/env.py` and pin the values with a unit test.
(Phase 1 task: locate canonical palette in arcengine source.)

## Recordings (`save_recording=True`)

Set `save_recording=True` on `arc.make(...)` and the wrapper writes to
`recordings/<scorecard_id>/<game_id>-<guid>.jsonl` — one line per `_set_last_response`.
Schema is (almost) the same as the human-replay JSONLs we downloaded; see
[docs/replay-format.md](replay-format.md).

## Scorecard mechanics

`arc.get_scorecard()` returns an `EnvironmentScorecard` (Pydantic model) with:

```
score                          # float, 0..100, env-mean of per-env scores
total_actions                  # int, sum across all runs
total_levels_completed         # int
total_environments_completed   # int  (== "won all levels" environments)
environments: list[
  id, runs: list[
    guid, score, levels_completed, actions, resets, state, completed,
    level_scores, level_actions, level_baseline_actions, ...
  ],
  score, actions, levels_completed, completed, level_count, resets,
]
tags_scores: list[...]   # aggregated per tag
```

The per-run `score` is computed by `EnvironmentScoreCalculator.to_score()`:

```python
# Pseudo-Python from arc_agi.scorecard:
for level in added_levels:                    # add_level() called per level
    level_score = ((baseline / actions_taken) ** 2) * 100  if completed else 0.0
    level_score = min(level_score, 115.0)
total_score = sum(level_score_i * level_index_i) / sum(level_index_i)
max_score   = sum(level_index_i where score_i > 0) / sum(level_index_i) * 100
score       = min(total_score, max_score)
```

This matches `methodology.md`’s RHAE formula:
- per-level cap 1.15 ⇔ 115.0 (×100 scale).
- per-game weighting by 1-indexed level number.
- max bound by completion fraction (so can’t exceed `(1+2+…+k)/(1+…+L)*100` if you stop after level `k`).
- top-level `score` is the mean of per-game scores ⇒ matches "average of all game scores".

**Conclusion:** `arc3_wm/rhae.py` is **not** required for primary metric reporting.
We may still keep a one-screen helper that recomputes from `(level_baseline_actions,
level_actions, levels_completed)` — useful for (a) per-checkpoint logging
without re-instantiating Arcade and (b) a sanity-check unit test that asserts
toolkit-RHAE matches our reference implementation on a hand-worked example
from `methodology.md`.

**Scale gotcha for the paper / our tests:** toolkit returns 0–100 (cap 115).
methodology states 0–1 (cap 1.15). When citing "RHAE" externally, divide by
100 or document the scale.

## Per-mode behaviour for vc33 / tu93 / cd82

All three load and step in OFFLINE without errors (verified). Initial available
actions per game above. Random-agent smoke (10 episodes on vc33, no `render_mode`)
runs at ~921 FPS on this laptop. See `scripts/random_agent_smoke.py` and the
session summary for full output.

## Rate limits (ONLINE only)

`docs/arc-agi-3/rate_limits.md` says 600 RPM on the ONLINE remote API. OFFLINE
has none. We never run training/eval against ONLINE. Only foreseeable ONLINE
use is the official scorecard-submission run for the paper, post-Phase 5;
treat as a separate one-off task.

## Open follow-ups (Phase 1 will resolve)

1. Verify `frame_layers == 1` for all 25 public-demo games (some games may
   ship multi-layer frames). If multi-layer is ever observed, decide on a
   stack/concat policy and surface to Haso.
2. Find canonical 16-colour palette in `arcengine` (or freeze in
   `arc3_wm/env.py` with an introspection test).
3. Confirm whether `step` on an unsupported action increments the
   scorecard `actions` count (probably yes — strong reason to mask).
4. Determine how `ACTION7` becomes available — does the engine surface it
   only after a state change, or is it never exposed for these games?
   Probe during Phase 1 with longer episodes.
5. Confirm `_last_response.available_actions` is consistent between
   wrapper and engine (we currently rely on that property).
