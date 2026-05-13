# Phase 4 dry-run on Vast — cd82 / 500k / seed 0

Single-game single-seed dry-run on Vast A100. Derisks the warm-start +
RHAE-logging pipeline at full Phase-4 scale before the local 5070 cluster
comes online. **Not Phase 4 proper** — that's multi-game, multi-seed,
on local 5070s. This is the same plumbing as
[docs/phase4-smoke.md](phase4-smoke.md), scaled from 20k → 500k env
steps, with Phase-4-grade wandb naming + tags + group conventions.

Where this doc and the smoke doc diverge, **this doc wins for the
dry-run**.

## Pre-flight (laptop, before SSH)

Mirrors [docs/phase4-smoke.md §A.0](phase4-smoke.md), with two
additions per D2 (post-hoc RHAE) and D3 (`--script train_eval`):

- `arc3_wm/rhae.py` exposes a pure `RHAEAggregator` class (post-hoc;
  no in-loop scheduler). Rename committed at `a5315d9`.
- `scripts/compute_rhae.py` consumes per-eval-episode reward streams
  and emits the three `eval/rhae/*` keys post-hoc. Committed at
  `1f6a2d0`.

**Verify before launching:**

```bash
pytest -q tests/test_rhae.py                 # → 50 passed
pytest -q tests/test_compute_rhae.py         # → 19 passed
pytest -q tests/test_extract_human_baselines.py  # → 37 passed
```

Same B2 artifacts as the smoke (Phase-3 pkl + env_files-pilot tarball).
Same secrets (`ARC_API_KEY`, `B2_*`, optional `WANDB_*`).

## Instance

| Knob | Value | Rationale |
|---|---|---|
| GPU | **A100 PCIE 40GB on-demand**, target $0.530/hr | Phase-4-validated path. Spot also OK if cheaper. **No Blackwell** (5060 Ti / 5070 / 5070 Ti / 5080 / 5090) — sm_120 / JAX 0.4.33 compat unvalidated and is precisely the latent risk Phase 4 proper carries; running on Vast Blackwell wouldn't derisk the local cluster either. |
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | Same as smoke. |
| Region | EU (Frankfurt / Amsterdam / Stockholm) | Match B2 EU-Central. |
| Disk | 40 GB persistent | Phase-3 pkl + logdir + checkpoints. |
| Spot | Optional | $0.530/hr is the on-demand floor we've seen; spot can drop further if available. |
| Session budget | ~5–7 h wall-clock, ≤ $6 of remaining ~$11.50 Vast credit | Hard ceiling per D4. The ~48% eval overhead from `--script train_eval` (vs. plain `train`) pushes the wall-clock to ~7 h on A100 at $0.530/hr → ~$3.70 fully consumed; the $6 ceiling absorbs the overhead with margin. If approaching $6 with no completion, **stop and surface**. |

## Bring-up

Identical to [docs/phase4-smoke.md §A.2](phase4-smoke.md). Reuse the
existing 12-step script verbatim. The dry-run does not need replays
(uses online buffer), but stages the Phase-3 pkl and `environment_files-pilot.tar.gz`.

Add **two extra env exports** before the install block:

```bash
export WANDB_PROJECT=arc3-wm-sprint
export WANDB_API_KEY=<from wandb.ai/settings>
export WANDB_TAGS=phase-4-dryrun,arm:warm,game:cd82
export WANDB_RUN_GROUP=dryrun-cd82-warm
export WANDB_NAME=p4-dryrun-cd82-s0-warm-1b381ae
```

`launch_pergame.py:auto_add_wandb_output` (line 436) flips wandb on
automatically when `WANDB_PROJECT` is set. `WANDB_TAGS`,
`WANDB_RUN_GROUP`, `WANDB_NAME` are stock wandb env-var hooks the SDK
honors at `wandb.init()` time; no launcher-side code change needed.

## Launch (background)

Per D3, the dry-run runs under `--script train_eval` so DV3's stock
eval driver produces `eval/episode/score` and `eval/episode/length`
across the run. (RHAE is computed post-hoc per D2; see "Post-hoc
analysis" below.)

```bash
TS=$(date -u +%Y%m%dT%H%M%S)
RUN=p4-dryrun-cd82-s0-warm-1b381ae
LOGDIR=/workspace/logdir/$RUN

nohup python scripts/launch_pergame.py \
  --logdir "$LOGDIR" \
  --configs size12m arc3 \
  --task arc3_cd82 \
  --seed 0 \
  --script train_eval \
  --init-from-ckpt checkpoints/pretrained-wm/v1/latest.pkl \
  --run.steps 500000 \
  --run.log_every 30 \
  --run.save_every 600 \
  > "$LOGDIR.log" 2>&1 &

echo "PID=$!" > "$LOGDIR.pid"
disown
```

Cadence sized for 500k steps:
- `log_every=30` s → ~600 logged windows over ~5 h. Comfortable for
  wandb panels and post-hoc `check_smoke_green.py`.
- `save_every=600` s (10 min) → ~30 checkpoints. Phase-4-proper will
  drop intermediate ckpts at completion per Notion "Checkpoint policy";
  for the dry-run we keep them to enable mid-run inspection.

**Do not `tail -f` the log live — backgrounded means backgrounded.**
SSH back periodically; spot-check via `tail -100 "$LOGDIR.log"`.

## Mid-run health checks (every 30–60 min)

```bash
# 1. Still running?
ps -p $(cat "$LOGDIR.pid")

# 2. Recent loss values descending? (sanity, not a substitute for
#    check_smoke_green.py at exit)
grep -E "Agent Step|train/loss/image" "$LOGDIR.log" | tail -20

# 3. Eval episodes firing? `--script train_eval` writes eval/episode/score
#    every `report_every` seconds (default 300 s). Expect ~60 eval cycles
#    over ~5 h. Live RHAE keys are NOT in wandb — RHAE is post-hoc, see
#    "Post-hoc analysis" below.
grep -E "eval/episode/score" "$LOGDIR/scope/metrics.jsonl" 2>/dev/null | tail -5
```

## RHAE telemetry — post-hoc only (D2)

Per D2 the eval loop runs stock `embodied.run.train_eval` with no
in-loop RHAE hook. The dry-run produces `train/loss/*`,
`eval/episode/score`, `eval/episode/length`, `fps_train`, etc. The
per-game RHAE is computed AFTER the run completes via
`scripts/compute_rhae.py` — see "Post-hoc analysis" below. The
Phase-4-gate question "did the agent clear cd82 level 1?" is also
readable directly from `eval/episode/score` (cd82's native reward
fires on level-up).

## Exit criteria — success

1. Training completed 500k steps clean — no NaN, no OOM, no Arcade crash.
2. `loss/image`, `loss/dyn`, `loss/rep` descend monotonically (same
   shape as Phase-3 Run B; baseline from pretrained ckpt is
   `loss/image=53.66`).
3. `episode/score` non-zero by step 500k (proxy for "level 1 cleared"
   on cd82; native reward fires on level-up).
4. Final checkpoint saved to `$LOGDIR/ckpt/latest.pkl`. Upload to B2:
   ```bash
   b2 file upload arc-agi-3-replays-hasaan "$LOGDIR/ckpt/latest.pkl" \
     "dryruns/p4-cd82-s0-warm-1b381ae/latest.pkl"
   b2 file upload arc-agi-3-replays-hasaan "$LOGDIR.log" \
     "dryruns/p4-cd82-s0-warm-1b381ae/launch.log"
   ```
5. Vast instance torn down. Local-laptop `tee` log preserved if you
   ran with `tee`; otherwise pull from B2 per above.

## Exit criteria — failure (escalate, do not debug solo)

| Signal | Action |
|---|---|
| Training crashes / NaN / OOM in first 50k steps | Stop. Attach full log + last wandb step. Surface. |
| `loss/image` regresses past Phase-3 baseline (53.66) in first 20k steps | Stop. Warm-start may not be seeding correctly. Check `WM seeded:` line in stdout per [phase4-smoke.md §A.3](phase4-smoke.md) signal-3. |
| `episode/score` stuck at 0 through 500k steps | Note, don't stop — this is one of the things the dry-run is testing. Phase 4 proper's gate is "RHAE > 0 on ≥ 2/3 of {vc33,sb26,cd82}"; a cd82 zero here means the warm-start isn't enough to clear level 1 on cd82 alone and the Phase-4 gate is in danger. |
| Cost trending past $6 | Stop. Vast instance teardown takes priority over completion. The $6 ceiling already absorbs the ~48% eval overhead from `--script train_eval`; breaching it means something else is wrong. |

## Teardown

```bash
# 1. Kill the run if still running.
kill $(cat "$LOGDIR.pid") 2>/dev/null

# 2. Upload final ckpt + log to B2 (per exit criterion 4).

# 3. Vast.ai console → destroy instance.

# 4. Verify B2 artifacts:
b2 file ls arc-agi-3-replays-hasaan dryruns/p4-cd82-s0-warm-1b381ae/
```

## Post-hoc analysis — `scripts/compute_rhae.py`

After the dry-run completes, compute per-game RHAE locally from the
eval-episode reward streams produced during `--script train_eval`.

The CLI takes a JSONL of episode reward streams (`{"rewards": [...]}`
per line) plus the per-game human baselines from
`data/human_baselines.json` (generated by
`scripts/extract_human_baselines.py`):

```bash
# Pull the run logdir from B2 if you ran teardown first.
b2 file download arc-agi-3-replays-hasaan \
  dryruns/p4-cd82-s0-warm-1b381ae/scope/metrics.jsonl \
  ./metrics.jsonl
# (Followed by whatever extraction step you use to assemble the
# per-eval-episode reward streams. See "Known gap" below.)

python scripts/compute_rhae.py \
  --episodes-file ./eval_episodes.jsonl \
  --game-id cd82 \
  --baselines data/human_baselines.json \
  --step 500000

# → "cd82 @ 500k env steps: levels_completed=N, per_game_rhae=R.RR"
```

## Hand-off note

When the dry-run completes, the recommendation for the next session
is one of:

- **Failure path** (no level 1 by 500k, or losses don't descend): debug
  order per `CLAUDE.md` §Risks-4: action mapping → reward signal
  correctness → exploration. Do not silently scale to longer training.
