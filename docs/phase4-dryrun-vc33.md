# Phase 4 dry-run on Vast — vc33 / 500k / seed 0

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
export WANDB_TAGS=phase-4-dryrun,arm:warm,game:vc33
export WANDB_RUN_GROUP=dryrun-vc33-warm
export WANDB_NAME=p4-dryrun-vc33-s0-warm-1f6a2d0
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
RUN=p4-dryrun-vc33-s0-warm-1f6a2d0
LOGDIR=/workspace/logdir/$RUN

nohup python scripts/launch_pergame.py \
  --logdir "$LOGDIR" \
  --configs size12m arc3 \
  --task arc3_vc33 \
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
grep -E "eval/episode/score" "$LOGDIR/metrics.jsonl" 2>/dev/null | tail -5
```

## RHAE telemetry — post-hoc only (D2)

Per D2 the eval loop runs stock `embodied.run.train_eval` with no
in-loop RHAE hook. The dry-run produces `train/loss/*`,
`eval/episode/score`, `eval/episode/length`, `fps_train`, etc. The
per-game RHAE is computed AFTER the run completes via
`scripts/compute_rhae.py` — see "Post-hoc analysis" below. The
Phase-4-gate question "did the agent clear vc33 level 1?" is also
readable directly from `eval/episode/score` (vc33's native reward
fires on level-up).

## Exit criteria — success

1. Training completed 500k steps clean — no NaN, no OOM, no Arcade crash.
2. `loss/image`, `loss/dyn`, `loss/rep` descend monotonically (same
   shape as Phase-3 Run B; baseline from pretrained ckpt is
   `loss/image=53.66`).
3. `episode/score` non-zero by step 500k (proxy for "level 1 cleared"
   on vc33; native reward fires on level-up).
4. Final checkpoint saved to `$LOGDIR/ckpt/{TIMESTAMP}F{NANOS}/` with a
   22-byte `$LOGDIR/ckpt/latest` pointer file alongside it. DV3 does
   **NOT** write `latest.pkl` on save (asymmetric with `--init-from-ckpt`
   input which does take a `.pkl` filename). Tar the directory before B2:
   ```bash
   B2_PREFIX=dryruns/p4-vc33-s0-warm-$(git rev-parse --short HEAD)
   CKPT_DIR=$(ls -1 "$LOGDIR/ckpt/" | grep -v '^latest$' | head -1)
   tar czf "$LOGDIR/ckpt-final.tar.gz" -C "$LOGDIR/ckpt" "$CKPT_DIR" latest
   b2 file upload arc-agi-3-replays-hasaan "$LOGDIR/ckpt-final.tar.gz" \
     "$B2_PREFIX/ckpt-final.tar.gz"
   b2 file upload arc-agi-3-replays-hasaan "$LOGDIR.log" \
     "$B2_PREFIX/launch.log"
   b2 file upload arc-agi-3-replays-hasaan "$LOGDIR/metrics.jsonl" \
     "$B2_PREFIX/metrics.jsonl"
   b2 file upload arc-agi-3-replays-hasaan "$LOGDIR/eval_episodes.jsonl" \
     "$B2_PREFIX/eval_episodes.jsonl"
   ```
5. Vast instance torn down. Local-laptop `tee` log preserved if you
   ran with `tee`; otherwise pull from B2 per above.

## Exit criteria — failure (escalate, do not debug solo)

| Signal | Action |
|---|---|
| Training crashes / NaN / OOM in first 50k steps | Stop. Attach full log + last wandb step. Surface. |
| `loss/image` regresses past Phase-3 baseline (53.66) in first 20k steps | Stop. Warm-start may not be seeding correctly. Check `WM seeded:` line in stdout per [phase4-smoke.md §A.3](phase4-smoke.md) signal-3. |
| `episode/score` stuck at 0 through 500k steps | Note, don't stop — this is one of the things the dry-run is testing. Phase 4 proper's gate is "RHAE > 0 on ≥ 2/3 of {vc33,sb26,cd82}"; a vc33 zero here means the warm-start isn't enough to clear level 1 on vc33 alone and the Phase-4 gate is in danger. |
| Cost trending past $6 | Stop. Vast instance teardown takes priority over completion. The $6 ceiling already absorbs the ~48% eval overhead from `--script train_eval`; breaching it means something else is wrong. |

## Teardown

```bash
# 1. Kill the run if still running.
kill $(cat "$LOGDIR.pid") 2>/dev/null

# 2. Upload final ckpt + log to B2 (per exit criterion 4).

# 3. Vast.ai console → destroy instance.

# 4. Verify B2 artifacts (note: `b2 ls`, NOT `b2 file ls` — there is no
#    `ls` subcommand under `b2 file` in the current CLI):
b2 ls --recursive "b2://arc-agi-3-replays-hasaan/$B2_PREFIX/"
```

## Post-hoc analysis — `scripts/compute_rhae.py`

After the dry-run completes, compute per-game RHAE locally from the
eval-episode reward streams produced during `--script train_eval`.

The CLI takes a JSONL of episode reward streams (`{"rewards": [...]}`
per line) plus the per-game human baselines from
`data/human_baselines.json` (generated by
`scripts/extract_human_baselines.py`):

```bash
# Pull the eval-episode reward streams from B2 if you ran teardown first.
b2 file download "b2://arc-agi-3-replays-hasaan/$B2_PREFIX/eval_episodes.jsonl" \
  ./eval_episodes.jsonl

python scripts/compute_rhae.py \
  --episodes-file ./eval_episodes.jsonl \
  --game-id vc33 \
  --baselines data/human_baselines.json \
  --step 500000 \
  --print-metrics

# → "vc33 @ 500k env steps: levels_completed=N, per_game_rhae=R.RR"
```

## Hand-off note

When the dry-run completes, the recommendation for the next session
is one of:

- **Failure path** (no level 1 by 500k, or losses don't descend): debug
  order per `CLAUDE.md` §Risks-4: action mapping → reward signal
  correctness → exploration. Do not silently scale to longer training.
- **Success path** (≥1 level cleared by 500k, losses descend): recompose
  Phase 4 proper on the local 5070 cluster with the gate composition
  `{vc33, sb26, cd82} × 2 seeds × 500k`. Mirror the launcher pattern
  (`--init-from-ckpt`, `--script train_eval`, EvalRewardSink wraps
  automatically). Gate: RHAE > 0 on ≥ 2/3 games.

## Dry-run result (2026-05-13)

Run: vc33, seed 0, 500k env steps, warm-start from
`checkpoints/pretrained-wm/v1/latest.pkl` (Phase-3 Run B). Vast A100
SXM4 40GB, conda env `main` (Python 3.12.13, JAX 0.4.33, DV3
b65cf81a). Launcher commit `7d0d17a`.

**Outcome: PASS.**

| Signal | Value |
|---|---|
| `check_smoke_green.py` verdict | GREEN on all 6 criteria (final metrics.jsonl) |
| Warm-start fingerprint | 68 keys / 9,898,179 params matched Phase-3 ckpt; counters reset to 0/0/0 |
| Steps completed | 499,949 (≈ 500k; clean exit) |
| `train/loss/image` | 53.66 (warm-start) → 0.14 (final step) |
| `train/loss/rep` | std=0.04 on last 50 logged values — descending, not plateaued |
| `episode/score` (train-time) | 9,675 episodes, max=1.0, **17 episodes with score > 0** |
| `eval_episodes.jsonl` | 18 eval episodes captured by `EvalRewardSink` (D-C plumbing) |
| `eval/rhae/levels_completed/vc33` | **1** (vc33 level 1 cleared in eval) |
| `eval/rhae/per_game/vc33` | **0.00786** ≈ 0.01 |
| `fps/policy` | ~295 (vs runbook estimate ~150 on PCIE A100) |
| Wall-clock | ~30 min (vs runbook estimate ~7h on PCIE A100) |
| Cost | ~$0.53 (vs $6 stop-and-surface budget) |
| NaN / OOM / Arcade crash | none |

**Phase-4-proper gate implication.** The dry-run answers "does the
warm-start + native-reward pipeline clear vc33 level 1 in 500k?" with
yes (RHAE > 0). The gate question for Phase 4 proper is the same
threshold (RHAE > 0) but on ≥ 2/3 of `{vc33, sb26, cd82}` and with 2
seeds per game. vc33 is precedented; sb26 and cd82 are unseen at
500k scale.

**B2 artifacts** (`b2://arc-agi-3-replays-hasaan/dryruns/p4-vc33-s0-warm-7d0d17a/`):

- `ckpt-final.tar.gz` (141 MB, contains `{TIMESTAMP}F{NANOS}/` + `latest` pointer)
- `metrics.jsonl` (730 KB)
- `eval_episodes.jsonl` (5.2 KB, 18 lines)
- `launch.log` (59 KB)

**Deviations from spec (logged for record):**

- Python 3.12.13 not 3.11 (Vast image had no `python3.11` apt). JAX
  0.4.33 and DV3 both spec 3.12, so in-spec, but not the validated
  laptop path.
- Used existing conda `main` env, not a fresh `.venv`. Harmless
  warnings about torch 2.11 / setuptools<82 (we don't use torch).
- Repo cloned via git bundle, not GitHub. DV3 cloned separately at
  pinned `b65cf81a`.
