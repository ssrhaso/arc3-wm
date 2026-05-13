# Phase 4 warm-start smoke on Vast

Single-instance, single-seed, single-game (vc33) smoke. Validates the
Phase 3 → Phase 4 hand-off: WM weights load from the Phase-3 pkl into a
fresh per-game DreamerV3 agent, the agent constructs the env without
an Arcade crash, and a few thousand env steps run without NaN / OOM.

**This is a smoke, not the Phase 4 sweep.** No 50k step run, no
overnight, no second seed, no multi-game. Hard ceiling: ~3 h
session, ~$10 Vast spend.

Mirrors the structure of [docs/vast-phase3-launch.md](vast-phase3-launch.md).
Where the Phase-3 doc and this one diverge, this doc wins for Phase 4.

## A.0 — Pre-flight checklist (laptop, before SSH)

- [ ] **Phase A artifacts landed** — `e5414e8` or later on `main`:
  - `scripts/launch_pergame.py` has `--init-from-ckpt`
  - `scripts/check_smoke_green.py` exists
  - `scripts/stage_env_files.sh` exists
  - `tests/test_launcher_warmstart.py` + `tests/test_check_smoke_green.py` all pass on laptop
- [ ] **B2 artifacts staged**:
  - `b2://arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl` (118.8 MB) — Phase-3 ckpt
  - `b2://arc-agi-3-replays-hasaan/env-files/v1/environment_files-pilot.tar.gz` (~20 KB)
- [ ] **URLs reachable** (run on laptop or any clean shell):
  ```bash
  curl -I https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl
  curl -I https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/env-files/v1/environment_files-pilot.tar.gz
  ```
  Both must return `HTTP/1.1 200` with sensible `Content-Length`.
- [ ] **Secrets ready**: `ARC_API_KEY`, `B2_KEY_ID`, `B2_APP_KEY`,
  optionally `WANDB_API_KEY`. Same set as the Phase-3 doc §A.0.
- [ ] **Session budget**: ~3 h wall-clock, ~$10 Vast.

## A.1 — Provision spot instance

Per `docs/compute-runbook.md` §"Provisioning a fresh spot instance",
with these Phase-4-specific overrides:

| Knob | Value | Rationale |
|---|---|---|
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | DV3 needs CUDA 12.x (same as Phase 3) |
| GPU | 1× A100 80GB spot | Plenty of headroom for size12m at 10k steps; cheaper than H100. H100 also fine if A100 unavailable. |
| Region | **EU** (Frankfurt / Amsterdam / Stockholm) | Match B2 EU-Central |
| Disk | 40 GB persistent | No replays needed (smoke uses online buffer). Just env_files (~20 KB) + Phase-3 pkl (118.8 MB) + logdir (~few hundred MB). |
| Spot | **YES** | Per CLAUDE.md, never non-preemptible |
| Interruption budget | 1× preemption tolerable | Smoke is throwaway; restart if it preempts (no resume protocol tonight). |

Cost estimate: A100 spot ~$0.80–1.20/hr × ~45 min total = ~$0.60–0.90.
Well inside the $10 ceiling.

## A.2 — First-run setup on the instance

After SSH, paste env vars (replace placeholders):

```bash
export GITHUB_REPO=https://github.com/ssrhaso/ARC_AGI_3.git
export ARC_API_KEY=<from https://three.arcprize.org>
export PRETRAINED_WM_URL=https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl
export ENV_FILES_TAR_URL=https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/env-files/v1/environment_files-pilot.tar.gz
export WANDB_PROJECT=arc3-wm-sprint                  # optional; launch_pergame.py auto-adds wandb to outputs when set
export WANDB_API_KEY=<from wandb.ai/settings>        # optional
export B2_KEY_ID=<your B2 key id>                    # only needed if uploading logs at teardown
export B2_APP_KEY=<your B2 app key>
```

Then run the install + data-pull. This is `compute-runbook.md` §"First-run
setup script" with three Phase-4-specific deltas (no replays, do
pull env_files + the Phase-3 pkl, no TF needed):

```bash
set -euo pipefail

# 0. System packages.
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 git curl

# 1. Clone.
cd /workspace
[ -d arc3-wm ] || git clone "$GITHUB_REPO" arc3-wm
cd arc3-wm

# 2. Python env.
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel setuptools

# 3. JAX GPU FIRST.
pip install -U "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. DreamerV3 deps.
pip install -U -r third_party/dreamerv3/requirements.txt

# 4b. wandb media deps — DV3's logger logs video panels to wandb,
#     which calls moviepy under the hood. moviepy/imageio are NOT
#     pulled by DV3's requirements.txt and are NOT a hard wandb
#     dep, so a barebones `pip install wandb` crashes on first eval
#     boundary with `wandb.errors.errors.Error: wandb.Video requires
#     moviepy when passing raw data`. Surfaced by the 2026-05-13 vc33
#     smoke on Vast A100 (run aa9788oi) before this line was added.
pip install -U "wandb[media]"

# 5. Project deps + dev extras.
pip install -e .
pip install pytest pytest-xdist pytest-timeout b2

# 6. .env for arc_agi (OFFLINE mode).
cat > .env <<EOF
ARC_API_KEY=$ARC_API_KEY
OPERATION_MODE=offline
EOF
chmod 600 .env

# 7. JAX device check — fail early if CUDA isn't visible.
python -c "import jax; devs = jax.devices(); print('JAX devices:', devs); assert any(d.platform == 'gpu' for d in devs), 'no GPU visible to JAX'"

# 8. Stage env_files (pilot-3 bundle, ~20 KB).
[ -d environment_files ] || curl -L --fail "$ENV_FILES_TAR_URL" | tar xz
ls environment_files/                  # should list cd82, tu93, vc33

# 9. Stage the Phase-3 pkl (118.8 MB).
mkdir -p checkpoints/pretrained-wm/v1
[ -f checkpoints/pretrained-wm/v1/latest.pkl ] || \
  curl -L --fail "$PRETRAINED_WM_URL" -o checkpoints/pretrained-wm/v1/latest.pkl
ls -lh checkpoints/pretrained-wm/v1/latest.pkl     # ~118.8 MB

# 10. B2 CLI auth (optional, for teardown log upload).
if [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ]; then
  b2 account authorize "$B2_KEY_ID" "$B2_APP_KEY"
fi

# 11. Optional: wandb login.
[ -z "${WANDB_API_KEY:-}" ] || wandb login "$WANDB_API_KEY"

# 12. Pytest sanity (Vast, with JAX live). Phase-4-touching tests must be green.
pytest -q tests/test_launcher_warmstart.py tests/test_check_smoke_green.py \
          tests/test_launcher_arg_parsing.py tests/test_launcher_imports.py
```

If any step fails, **stop and diagnose** — do not paper over.

The pytest line above also exercises the real-pkl integration test
(`test_seed_wm_from_ckpt_against_real_v1_pkl`) because the pkl is now
local. That single test is the strongest end-to-end check before
launching JAX.

## A.3 — Launch the smoke

```bash
TS=$(date -u +%Y%m%dT%H%M%S)
RUN=p4-vc33-s0-smoke-$TS
LOGDIR=/workspace/logdir/$RUN

python scripts/launch_pergame.py \
  --logdir "$LOGDIR" \
  --configs size12m arc3 \
  --task arc3_vc33 \
  --seed 0 \
  --init-from-ckpt checkpoints/pretrained-wm/v1/latest.pkl \
  --run.steps 20000 \
  --run.log_every 5 \
  --run.save_every 30 \
  2>&1 | tee "$LOGDIR.log"
```

> **`log_every` and `save_every` are wall-clock seconds, not env steps.**
> Both are `embodied.LocalClock` instances (see
> [`embodied/core/clock.py:97-118`](../third_party/dreamerv3/embodied/core/clock.py#L97-L118)).
> With `first=False` (the default) the very first call always returns
> False, then they fire whenever `time.time() >= prev + every`. The
> block at
> [`embodied/run/train.py:106-114`](../third_party/dreamerv3/embodied/run/train.py#L106-L114)
> that flushes `train/loss/*`, `replay/*`, `fps/*` is gated by
> `should_log`; if the smoke's training-loop wall-clock is under
> `log_every`, **nothing from that block reaches JSONL** even when
> training is firing correctly. Per-episode `episode/score` and
> `episode/length` still flush because they're written from `logfn`
> outside the gate. Earlier defaults of `log_every=100, save_every=600`
> were 30–60× too coarse for this smoke and gave a misleading
> "training never fired" signal (see
> [docs/phase4-smoke-evidence.md](phase4-smoke-evidence.md) verdict-1).
> 5 s / 30 s with 20,000 env steps gives ~10 logged windows on A100 —
> enough for `check_smoke_green.py`'s std-based criterion to discriminate.

Expected wall-clock on A100: ~2 min total (~30–60 s JAX compile +
ARC scorecard init, then ~50 s of training-loop wall-clock at
~370–450 env-steps/sec and `fps_train ≈ 1.2e4`).

Expected stdout signals (first ~2 min, watch live):

1. `Init-ckpt resolved to: checkpoints/pretrained-wm/v1/latest.pkl`
2. `JAX compile output` — long pause normal, ~30-60 s
3. **`WM seeded: matched_keys=68 matched_params=9,898,179 counters_before_reset={'updates': 192000, 'batches': 192001, 'actions': 0} live_counters_after_load={'updates': 0, 'batches': 0, 'actions': 0}`** — the critical line; if it doesn't print or has wrong values, kill the run immediately and check (a)/(b) below.
4. `Start training loop`
5. First `Agent Step N` block in stdout with `train/loss/image`,
   `train/loss/dyn`, `train/loss/rep`, `train/loss/rew`,
   `train/loss/con` populated, within ~10 s of "Start training loop".
   A second block at ~5 s later, and so on every ~5 s. Image loss
   should be visibly descending across blocks (147 → ~3 is what
   verdict-4 saw at 20k steps).

If `WM seeded` line doesn't appear within 90 s after `Init-ckpt resolved`,
the warm-start failed (likely JAX shape mismatch on agent.load). Kill,
collect log, file blocker.

## A.4 — Verdict via `scripts/check_smoke_green.py`

When the run exits (cleanly at 10k steps, or you kill it):

```bash
.venv/bin/python scripts/check_smoke_green.py \
  --stdout "$LOGDIR.log" \
  --jsonl  "$LOGDIR/metrics.jsonl"
```

The verdict line is the first stdout line: `GREEN` or `RED`.
Per-criterion breakdown follows. Exit code 0 iff `GREEN`.

Acceptance criteria (from the Phase-4 smoke spec):

| # | Criterion | PASS condition |
|---|---|---|
| a | WM regex matched | `matched_keys == 68` and `matched_params == 9,898,179` |
| b | Counter reset confirmed | `live_counters_after_load == {0, 0, 0}` |
| c | WM losses moving | std over last 50 logged values > 0 for each of `loss/{image, dyn, rep, rew, con}`. Aliases `loss/reward`, `loss/cont` accepted. |
| d | No NaN/Inf in losses | none of `loss/*` columns contain NaN or Inf |
| e | No OOM / Arcade crash | no `Traceback`, `OOM`, `Segmentation fault`, `arc_agi.Arcade.make(...) returned None`, `Killed`, or `AssertionError` in stdout |
| f | ≥1 env step recorded | `max(step) > 0` in JSONL |

`return_ema` / `score > 0` is **NOT** a criterion tonight — 10k steps
on a fresh actor is too short to demand non-trivial return. Revisit at
50k+ on the 5070 cluster.

## A.5 — What to paste back

When the smoke is done, copy these four items into the next message:

1. **The full output of `check_smoke_green.py`** — verdict line + criterion breakdown.
2. **The last ~50 lines of `$LOGDIR.log`** — captures the tail of training, any late NaN, the final save.
3. **The JSONL path** — `$LOGDIR/scope/metrics.jsonl` (or whatever the launcher logged). Don't paste the full file (it's large); we'll pull it via B2 if we need to.
4. **The wandb run URL** — only if `WANDB_PROJECT` was set. The launcher prints it during init.

That's enough to interpret GREEN or RED in the next round.

## A.6 — Teardown

**Mandatory** — no overnight runs tonight per the session scope change.

```bash
# Optional: upload the log + final ckpt for forensics (whether green or red).
if [ -n "${B2_KEY_ID:-}" ]; then
  b2 file upload arc-agi-3-replays-hasaan "$LOGDIR.log" "smoke-runs/$RUN.log" || true
  [ -f "$LOGDIR/ckpt/latest.pkl" ] && \
    b2 file upload arc-agi-3-replays-hasaan "$LOGDIR/ckpt/latest.pkl" \
        "smoke-runs/$RUN/latest.pkl" || true
fi

# Destroy the instance from the vastai dashboard (or `vastai destroy instance <id>`).
```

`destroy`, not `stop` — stopped instances still cost storage. Phase 4
proper will re-stage from B2; nothing on this instance is load-bearing.

## A.7 — If RED

1. Don't `destroy` until logs are uploaded to B2 (above).
2. Note which criterion(a) failed.
3. Paste the `check_smoke_green.py` output + last 50 lines of log back.
4. We'll write `docs/phase4-blockers.md` with the failure mode +
   first-thing-to-try-next-session, then `destroy`.

Common red modes to triage quickly:

| Symptom | Likely cause | Next step |
|---|---|---|
| `WM seeded` line absent | `seed_wm_from_ckpt` raised before printing | Look at the exception in stdout; likely shape mismatch on `agent.load` (config drift between Phase-3 save and Phase-4 build). |
| `WM seeded` shows wrong `matched_keys` (not 68) | Ckpt schema changed or wrong file pulled | Verify the B2 URL + sha. |
| `WM seeded` ok, but `arc_agi.Arcade.make` raises | env_files tarball didn't extract to the right place | `ls environment_files/vc33/` must show `metadata.json` + `vc33.py`. |
| WM losses NaN within first ~500 steps | Likely warm-start shape mismatch silently corrupted weights, OR the Phase-3 pkl is corrupt | Re-download pkl, verify size = 118,792,224 bytes. |
| OOM | A100 80GB should be plenty for size12m; reduce `--batch_size` if mismatched config layered something larger | Check `config.batch_size` print in stdout. |
