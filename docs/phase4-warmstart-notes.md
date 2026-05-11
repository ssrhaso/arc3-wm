# Phase-4 warm-start notes

Offline inspection of the Phase-3 checkpoint ahead of building
`--init-from-ckpt`. Source: `checkpoints/pretrained-wm/v1/latest.pkl`
(118.8 MB, identical to `b2://arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl`,
wandb run `t9fqbjt7`).

## Top-level pkl shape

`pickle.load(f)` returns a dict with two keys:

```
{
  'params':   <flat dict: 208 keys>,
  'counters': {'updates': 192000, 'batches': 192001, 'actions': 0},
}
```

This is `dreamerv3.embodied.jax.agent.Agent.save()` output verbatim
(see [embodied/jax/agent.py:340-353](../third_party/dreamerv3/embodied/jax/agent.py#L340-L353)).
The pickle wrapper in
[scripts/pretrain_wm.py:_save_checkpoint](../scripts/pretrain_wm.py#L256-L275)
side-steps `elements.Checkpoint`'s `_cleanup` Windows bug — but the
inner payload is the standard agent state, so `agent.load(data, ...)`
on the Phase-4 side reads it directly.

## Counters

| Field | Phase-3 value | Phase-3 expected | Verdict |
|---|---|---|---|
| `updates`  | 192,000 | 6000 outer × `train_ratio=32` = 192,000 | ✓ |
| `batches`  | 192,001 | ≈ `updates` (one batch ahead, see agent.py:368) | ✓ |
| `actions`  | 0       | Phase 3 had no Driver / env policy | ✓ |

No `2.2e8` garbage (would have indicated the save path was reading the
YAML target instead of the live counter — wasn't).

**Phase-4 reset rule:** before calling `agent.load(state, regex=WM_REGEX)`,
mutate `state['counters'] = {'updates': 0, 'batches': 0, 'actions': 0}`.
Cleaner than post-load mutation through the lock-protected `n_updates.value`
attributes.

## Params — top-level prefix breakdown

| Prefix | Keys | Param-elements | Notes |
|---|---|---|---|
| `opt`  | 140 | 19,796,362 | Optax state for the 5 WM modules. `opt/state/1/0` and `opt/state/2/0` are Adam step counts; `opt/state/3` is the top-level optax step. |
| `dyn`  |  27 |  7,028,224 | RSSM dynamics |
| `dec`  |  19 |  1,272,003 | Decoder (image recon) |
| `rew`  |   5 |    721,407 | Reward head |
| `con`  |   5 |    656,129 | Continue head |
| `enc`  |  12 |    220,416 | Encoder (4 stride-2 convs, size12m) |
| **WM module total** | **68** | **9,898,179** | Matches the spec-quoted figure exactly. |
| **All-incl total**  | 208 | 29,694,541 | |

`pol/` and `val/` prefixes are **absent**.
[WMOnlyAgent.__init__](../arc3_wm/wm_only_agent.py#L97-L108) overrode
`self.modules = [dyn, enc, dec, rew, con]` and re-instantiated `self.opt`
over that list, so `_ckpt_groups` never registered pol/val for save.

This means: **even with `regex=None`, `agent.load()` cannot clobber a
Phase-4 actor/critic** — the keys simply aren't in the loaded dict.
The regex is still load-bearing because of the `opt/` keys (see next
section).

## WM regex — committed value

```python
WM_REGEX = r'^(?:dyn|enc|dec|rew|con)/'
```

Anchored at start (redundant with `re.match` but explicit). Matches
exactly the 68 module-weight keys (9,898,179 params). Excludes:

- all 140 `opt/...` keys → fresh optimizer state in Phase 4
- any future `pol/...` or `val/...` keys (defence in depth)

### Why exclude `opt/...`

Loading Phase 3's `opt/state/...` would carry forward:

- WM-side Adam first/second moments (mildly useful — but encodes the
  cross-game offline data distribution, which differs from Phase 4's
  vc33-only online stream),
- Step counter at 192,000 in `opt/state/1/0` and `opt/state/2/0`. Any
  LR-warmup or schedule keyed to optimizer step would think Phase 4
  is mid-training from step 0. Schedules in DreamerV3 default to
  constant LR, so the impact is small in practice — but the cleaner
  claim is *fresh optimizer*.

Paper claim this supports: "Phase-4 per-game agents are initialised
with World-Model weights from the Phase-3 cross-game pretrain;
actor, critic, and all optimizer state are fresh." One sentence,
no caveats about Adam moments.

### Why NOT a broader regex

A pattern like `r'^(?:dyn|enc|dec|rew|con|opt)/'` would also load
`opt/state/.../con/...` etc. Rejected for the schedule + claim
reasons above.

A pattern like `r'^.*/(?:dyn|enc|dec|rew|con)/.*$'` (the example I
floated in the design-question dialogue) would match **zero** keys
against this flat key layout. Caught by writing the doc; would have
caught it at runtime via the fail-loud check in A4 either way.

## Sanity-check assertions for the launcher

The `--init-from-ckpt` code path must fail loud on:

1. `len(matched_keys) == 0` after applying `WM_REGEX` (signals a
   regex regression or a checkpoint format change).
2. `len(matched_keys) != 68` (matched count drifts from the known
   structure → silent partial load).
3. Sum of matched-key param-elements ≠ `9_898_179`.
4. `'counters' not in state` or missing any of `{updates, batches, actions}`
   (signals a structural change in `agent.save()`).

All four are unit-tested in A4 against synthetic state dicts.

## Re-use elsewhere

`A5` (env_files staging) and `B1`-`B3` (Vast smoke) don't depend on
this doc beyond knowing the regex. Captured here so the value is
greppable and reviewable independent of the launcher diff.
