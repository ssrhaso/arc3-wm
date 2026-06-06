# data/

Only `human_baselines.json` is tracked here. The raw human-replay JSONL
dataset (`data/replays/`) is large and gitignored; see
[../docs/replay-format.md](../docs/replay-format.md) for how to obtain and
stage it.

## `human_baselines.json`

The RHAE baseline fixture: per-game, per-level upper-median action counts
derived from the human replays by
[`scripts/extract_human_baselines.py`](../scripts/extract_human_baselines.py).
Consumed by [`arc3_wm/rhae.py`](../arc3_wm/rhae.py) and
`scripts/compute_rhae.py`.

Shape: one entry per public game.

```json
{
  "ar25": {
    "total_levels": 8,
    "baselines": { "1": 21, "2": 50, "3": 53, "...": "..." }
  }
}
```

- `total_levels` - the game's level count, read from the JSONL
  `win_levels` field (not inferred from the baseline keys).
- `baselines` - level index (1-based, string key) -> upper-median
  first-time-player action count. Levels with fewer than 2 completers
  are dropped as statistically indefensible, so the key set can be
  non-contiguous and smaller than `total_levels`. Current coverage:
  129/183 levels (70.5%). Uncovered levels are skipped from both the
  numerator and denominator of the RHAE game score.

Regenerate with `python scripts/extract_human_baselines.py` once
`data/replays/` is staged.
