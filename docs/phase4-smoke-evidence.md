# Phase 4 warm-start smoke evidence

> Single point-in-time record of the Phase-3 → Phase-4 hand-off bring-up.
> Validates `--init-from-ckpt` end-to-end: WM weights load, counters
> reset, env steps, training fires, losses descend monotonically.
> Cite alongside [docs/smoke-evidence.md](smoke-evidence.md) (milestones 2/3).
> Not a runbook — see [docs/phase4-smoke.md](phase4-smoke.md).

## Run metadata

| Field | Value |
|---|---|
| Date | 2026-05-11 |
| Instance | Vast.ai #36557202 |
| GPU | 1× NVIDIA A100 80GB (spot, ~$0.78/hr) |
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` |
| repo HEAD | `4345cbb` (main) |
| dreamerv3 | `b65cf81` |
| Configs | `size12m arc3` |
| Task | `arc3_vc33` |
| Seed | 0 |
| Phase-3 ckpt | `b2://arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl` (118.8 MB, sha matches local) |
| env_files | pilot-3 tarball (cd82, tu93, vc33) |

## WM seeding — verified end-to-end

```
Loaded pretrained checkpoint with keys: ['con/...', 'dec/...', 'dyn/...',
  'enc/...', 'rew/...']   (68 keys total)
WM seeded: matched_keys=68 matched_params=9,898,179
  counters_before_reset={'updates': 192000, 'batches': 192001, 'actions': 0}
  live_counters_after_load={'updates': 0, 'batches': 0, 'actions': 0}
```

All four fail-loud invariants from [docs/phase4-warmstart-notes.md](phase4-warmstart-notes.md):
matched_keys == 68 ✓, matched_params == 9,898,179 ✓, counters_after == all-zero ✓,
state shape sane ✓. The `--init-from-ckpt` regex `^(?:dyn|enc|dec|rew|con)/`
is correct and the live-counter reset survives `agent.load()`.

Optimizer initialized at 12,593,864 params: 9,898,179 WM (warm) +
2,695,685 pol+val (fresh). No `opt/...` keys carried forward (Adam
moments + step counter all fresh) — confirms the regex excluded them
as designed.

## Loss trajectory — `smoke4` (the green run)

Args: `--run.steps 20000 --run.log_every 5 --run.save_every 30`.
Wall-clock from "Start training loop" to exit: ~50 s. 10 logged
windows at ~1,500–2,000 env-step intervals.

| step | image | dyn=rep | rew | con | value | replay_ratio | fps/train |
|---|---|---|---|---|---|---|---|
| 3,168 | 146.8 | 25.75 | 5.8e-4 | 0.12 | 10.82 | 17.95 | 5,178 |
| 5,008 | 60.51 | 10.63 | 3.6e-5 | 0.13 | 10.67 | 32.22 | 1.2e4 |
| 5,856 | 31.39 |  5.95 | 1.7e-5 | 0.12 | 10.24 | 33.11 | 3,893 |
| 7,664 | 19.24 |  4.46 | 1.3e-5 | 0.10 |  9.01 | 32.21 | 1.1e4 |
| 9,520 | 10.17 |  3.45 | 1.3e-5 | 0.10 |  7.97 | 32.50 | 1.2e4 |
| 11,040 |  6.96 |  3.03 | 9.6e-6 | 0.09 |  7.37 | 32.84 | 9,802 |
| 12,880 |  5.33 |  2.77 | 7.3e-6 | 0.09 |  6.62 | 32.22 | 1.2e4 |
| 14,640 |  4.29 |  2.51 | 5.6e-6 | 0.08 |  5.81 | 32.50 | 1.1e4 |
| 16,528 |  3.63 |  2.45 | 4.9e-6 | 0.08 |  4.84 | 32.50 | 1.2e4 |
| 18,416 |  **3.13** |  **2.34** | 3.9e-6 | 0.07 |  3.75 | 32.50 | 1.2e4 |

Image recon 146.8 → 3.13 (47× drop). RSSM dyn/rep 25.75 → 2.34 (11×).
Critic 10.82 → 3.75. All monotonic. The warm-started WM is correctly
fine-tuning on vc33's distribution; the fresh actor and critic are
learning return structure from imagination rollouts.

`replay_ratio` stabilises at ~32 (the configured `train_ratio`),
confirming the buffer is being sampled at the intended rate after
warm-up.

`check_smoke_green.py` verdict: **GREEN** — all six criteria pass
(a/b WM-seed invariants, c std > 0 on all five WM losses, d no NaN,
e no OOM/Arcade-crash, f max(step) > 0).

[wandb run](https://wandb.ai/hasofocus-university-of-the-west-of-england/arc3-wm-sprint).

## Two false-negatives diagnosed and resolved

### Verdict-1 — "training never fired" (smoke at 17:30:05)

Original smoke per [docs/phase4-smoke.md §A.3](phase4-smoke.md#a3--launch-the-smoke):
`--run.steps 10000 --run.log_every 100 --run.save_every 600`.

JSONL ended up containing only `{step, episode/score, episode/length}` —
no `train/loss/*`, no `replay/*`, no `fps/*`. Initial hypothesis (from
the handoff): `trainfn` early-exiting on `len(replay) < batch_size *
batch_length`.

**Source-confirmed false.**
[`embodied/core/replay.py:55-56`](../third_party/dreamerv3/embodied/core/replay.py#L55-L56):
`__len__` returns `len(self.items)`. Items grow ≈ 1 per env step after
the first `length = consec*batlen + replay_context = 65` steps. With
9,857 env steps logged, `len(replay) ≈ 9,792` — comfortably above the
`16 × 64 = 1024` threshold. Training was firing.

The actual cause: `should_log = LocalClock(args.log_every)` is
**wall-clock seconds** ([`embodied/core/clock.py:111`](../third_party/dreamerv3/embodied/core/clock.py#L111)
uses `time.time()`), and the default `first=False` makes the first
call always return False. The smoke's training-loop wall-clock was
75 s, so `should_log(100)` never fired. The block at
[`embodied/run/train.py:106-114`](../third_party/dreamerv3/embodied/run/train.py#L106-L114)
that flushes `train_agg`, `replay.stats`, `fps`, `usage` and calls
`logger.write()` never executed. Only the per-episode `logger.add`
calls in `logfn` (un-gated) survived to disk.

Three corroborating signals:

1. Wall-clock from `head -1` to `tail -1` of `$LOGDIR.log`: 75 s.
2. `$LOGDIR/ckpt/` contained one subdir timestamped 38 s into the run
   — the initial `cp.load_or_save()` baseline at
   [`train.py:90`](../third_party/dreamerv3/embodied/run/train.py#L90),
   not a periodic save. `should_save(600)` never fired either.
3. The "Agent Step 9_857" line in stdout came from `logger.close()`
   flushing per-episode buffer — printed the `logfn`-written keys
   only, confirming `train_agg.result()` was never consumed.

### Verdict-3 — "WM losses not moving" (smoke at 17:59:04)

Re-run with `--run.steps 5000 --run.log_every 5 --run.save_every 30`.
Analyzer RED on criterion (c): `n=1 std=0` for every WM loss.

The single logged window at step 3,248 had healthy losses
(image 144.19, dyn 25.31, …) — training was firing fine. The
RED was a population-size artefact: env stepped at ~458 fps so 5,000
steps completed in ~11 s of training-loop wall-clock, and
`log_every=5` fired only once before exit.

Fixed by bumping to `--run.steps 20000` (verdict-4 above), which gave
~50 s of training-loop wall-clock and 10 logged windows.

## Footgun fix landed in this commit

`docs/phase4-smoke.md` §A.3 originally specified
`--run.log_every 100 --run.save_every 600` for a 10k-step smoke that
ran in 75 s. Both clocks are wall-clock seconds (not steps), and the
smoke completed before either fired. The runbook is updated to use
`--run.log_every 5 --run.save_every 30` with `--run.steps 20000`,
plus an explicit note that these are seconds-not-steps.

## What this run does NOT prove

- vc33 `RHAE > 0` — that's the Phase-4-proper gate, not a 20k-step
  smoke. Per CLAUDE.md the bar is "≥ 2 of {vc33, tu93, cd82} within
  500k env steps".
- Pre-populated replay buffer behaviour — this smoke trains entirely
  on online experience. The CLAUDE.md Phase-4 design calls for
  pre-populating the per-game buffer with ~10–15 human replays;
  `arc3_wm/replay_loader.py` is still an open Phase-1 follow-up.
- Multi-game (tu93, cd82) — single-game smoke only. Wrapper has been
  exercised on vc33 only on this instance.
- WM forgetting under per-game fine-tune — too short a budget to
  surface. Watch at Phase-4-proper budgets.

## Cost

~5 min wall-clock for verdict-4 (the green run) on top of the
verdict-1/3 diagnostics. Aggregate session A100 spot spend across all
three smokes: ≪ $1. The instance was left up for the diagnostic
sequence; should be torn down once any follow-up forensics on the
verdict-1 logdir is exfiltrated to B2.
