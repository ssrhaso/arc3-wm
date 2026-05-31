# Compute runbook (Phases 2-5)

> Operational playbook for the Vast.ai (Phases 2-3) and local 5070 cluster
> (Phases 4-5) workloads. Mirrors Section "Compute" in `CLAUDE.md`. Update this
> file at every phase boundary.

## Single mantra

> **Always launch with `--logdir <persistent>` so that re-running the same
> command resumes from the last checkpoint.** This is the *only*
> preemption-safety primitive we rely on. Without it, every Vast.ai eviction
> wastes ~30 minutes of training.

## Vast.ai (Phases 2-3)

### Account / billing

- Use spot ("interruptible") pricing only.
- Phase 2 + 3 budget: **$25-50 total**. If a single instance is approaching
  $20 of that, stop and reassess before continuing.
- Prefer H100 80GB SXM/PCIe over A100 80GB - typically ~30% better
  $/throughput on Vast.ai for DreamerV3-sized workloads. A100 40GB is fine
  for `size12m`.

### Provisioning a fresh spot instance

In the Vast.ai console:

1. **Image:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (or the closest
   CUDA 12.x image Vast offers). DreamerV3 needs JAX-CUDA12.
2. **Disk:** >= 80 GB persistent storage volume (replays alone are ~ 30 GB).
3. **Region:** prefer EU/US with the best $/H100. Avoid TOR/RU for
   network reliability.
4. **GPU:** 1x H100 SXM5 80GB (preferred) or 1x A100 80GB.
5. **Spot:** **always** mark as interruptible.
6. **SSH:** add Haso's public key during provisioning.

Keep two snapshots in mind: an `arc3-base` image with deps preinstalled
(saves 5 minutes per instance) and an `arc3-data` mountable volume with the
replay tarball pre-extracted.

### Mount layout

We assume the standard Vast layout:

```
/workspace/                  # 80 GB persistent volume, mounted across reboots
  arc3-wm/                   # checked out from your fork
  data/replays/              # extracted from the tarball
  logdir/                    # DreamerV3 logdir - must persist across preemption
  checkpoints/               # WM checkpoints, every 30 minutes
~/.cache/                    # ephemeral; OK to lose
```

`--logdir /workspace/logdir/<run-name>` for every DreamerV3 launch.

### First-run setup script

Run on a fresh spot instance, after SSH. Five env vars must be exported
before this script runs (laptop -> 1Password / SSH agent / paste in the
remote shell - never commit any of these):

```bash
export GITHUB_REPO=https://github.com/ssrhaso/ARC_AGI_3.git
export ARC_API_KEY=<from https://three.arcprize.org>
export REPLAY_TAR_URL=<B2 public URL - see "Data staging" below>
export ENV_FILES_TAR_URL=<B2 public URL for environment_files.tar.gz>
export WANDB_PROJECT=arc3-wm-sprint   # optional; logger picks up automatically
# export WANDB_API_KEY=<from wandb.ai/settings>   # only if WANDB_PROJECT set
```

Then:

```bash
set -euo pipefail

# 0. System packages - Crafter renders via OpenGL on a fresh Linux image.
#    libgl1 / libglib2.0-0 was historically required; current Hafner-Crafter
#    may not need it, but it's cheap insurance and instances are ephemeral.
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 git

# 1. Clone.
cd /workspace
[ -d arc3-wm ] || git clone "$GITHUB_REPO" arc3-wm
cd arc3-wm
git submodule update --init --recursive 2>/dev/null || true   # we don't use submodules; safe no-op.

# 2. Python env (3.11 minimum for danijar/dreamerv3; 3.12 also works).
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel setuptools

# 3. JAX GPU FIRST - this order is REQUIRED by danijar/dreamerv3.
#    The version pin matches third_party/dreamerv3/requirements.txt.
pip install -U "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. DreamerV3 deps.
pip install -U -r third_party/dreamerv3/requirements.txt

# 5. Crafter (NOT in dreamerv3/requirements.txt).
pip install crafter

# 6. Our project deps + editable install of arc3_wm.
pip install -e .                          # honours pyproject.toml dependencies
pip install pytest pytest-xdist pytest-timeout gdown   # dev extras for the test suite

# 7. Write .env so arc_agi auto-loads OFFLINE mode + the API key.
cat > .env <<EOF
ARC_API_KEY=$ARC_API_KEY
OPERATION_MODE=offline
ENVIRONMENTS_DIR=environment_files
RECORDINGS_DIR=recordings
EOF
chmod 600 .env

# 8. Stage replays from object storage (NEVER re-gdown from Drive on a remote).
mkdir -p data
[ -d data/replays ] || curl -L --fail "$REPLAY_TAR_URL" | tar xz -C data/
N_REPLAYS=$(find data/replays -name '*.jsonl' | wc -l)
[ "$N_REPLAYS" -ge 340 ] || { echo "ERROR: replay count $N_REPLAYS < 340" >&2; exit 1; }

# 9. Stage environment_files (the cached arc_agi game source) so OFFLINE
#    mode finds metadata.json without an API call.
[ -d environment_files ] || curl -L --fail "$ENV_FILES_TAR_URL" | tar xz

# 10. Sanity: full pytest suite (skips the dreamerv3 dry-run on laptop;
#     here it should run because JAX is now installed).
pytest -q
```

If `pytest -q` is green, the instance is ready for Phase-2 launch (see
"DreamerV3 launch (Phase 2 - Crafter sanity)" below).

### Data staging - laptop -> object storage -> instance

Cheap and provider-agnostic. Per CLAUDE.md, **never re-`gdown` from Drive
on a remote** - Drive's per-IP quota is exactly the failure mode we hit on
the laptop in Phase 0 (only 39 of 342 files retrieved).

**Target bucket:** Backblaze B2, single bucket `arc3-wm-data` (~$6/TB-month,
public download free under reasonable usage). Use this exact name in every
command below - the runbook depends on it. Alternatives if B2 access
isn't workable: AWS S3, Cloudflare R2 (free egress), or any HTTPS-served
bucket; substitute the bucket name end-to-end.

#### Two artifacts to stage today

| Artifact                                | Source on laptop                          | Target path in bucket                                         | Consumed by              |
|-----------------------------------------|-------------------------------------------|---------------------------------------------------------------|--------------------------|
| `replays-340.tar.gz`                    | `data/replays/` (340 JSONLs, Phase 3 prereq) | `arc-agi-3-replays-hasaan/replays-340.tar.gz`                | Phase 3 WM pretrain      |
| `environment_files-pilot.tar.gz`        | `environment_files/` (3 pilot games, ~20 KB) | `arc-agi-3-replays-hasaan/env-files/v1/environment_files-pilot.tar.gz` | Phase 4 per-game runs - OFFLINE mode |
| `pretrained-wm/v1/latest.pkl`           | `checkpoints/pretrained-wm/v1/latest.pkl` (Phase-3 v1 ckpt, 118.8 MB) | `arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl`        | Phase 4 `--init-from-ckpt` |

> **Bucket name:** the live bucket is `arc-agi-3-replays-hasaan`
> (provisioned in Phase 0; same one Phase 3 used). Earlier drafts of
> this runbook referenced `arc3-wm-data` as a placeholder - wherever
> that string appears below, treat it as the live name.

> **Pilot-3 vs full-25:** the `v1` env-files bundle covers
> `vc33`/`tu93`/`cd82` only - sufficient for Phase 4 tonight's smoke
> and the 3-game pilot. The full-25-game bundle bumps to `v2` and is a
> prereq for the Phase 5 sweep (see Phase 0 follow-up: rerun
> `scripts/cache_env_files.py` for all 25 games before tarballing).

#### Laptop -> bucket (one-time per artifact)

Prereqs: `pip install b2sdk b2` and `b2 account authorize` once with
your Backblaze application key. Bucket must be public-read (or use
presigned URLs and pin them in the run script).

```bash
# Replays (run AFTER OAuth gdown lands all 342, per D1).
find data/replays -name '*.jsonl' | wc -l     # must equal 342 before tarball.
tar czf replays.tar.gz data/replays
b2 file upload arc3-wm-data replays.tar.gz replays.tar.gz

# Environment files - pilot-3 (vc33, tu93, cd82). Wrapped in
# scripts/stage_env_files.sh; see that script for the canonical pattern.
./scripts/stage_env_files.sh bundle               # writes environment_files-pilot.tar.gz
./scripts/stage_env_files.sh upload v1            # uploads to env-files/v1/...
# (Full-25-game bundle: rerun scripts/cache_env_files.py first, then
#  upload as v2 with a renamed tarball.)

# Capture the public download URLs once and persist them. Each URL is
# stable as long as the bucket stays public-read.
b2 file url arc-agi-3-replays-hasaan/replays-340.tar.gz                          # -> REPLAY_TAR_URL
b2 file url arc-agi-3-replays-hasaan/env-files/v1/environment_files-pilot.tar.gz  # -> ENV_FILES_TAR_URL
```

Direct URLs (Phase-4-tonight values, hardcoded):

- `REPLAY_TAR_URL=https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/replays-340.tar.gz`
- `ENV_FILES_TAR_URL=https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/env-files/v1/environment_files-pilot.tar.gz`
- `PRETRAINED_WM_URL=https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/pretrained-wm/v1/latest.pkl`

#### Instance: pull-and-extract

Already wired into the first-run setup script above; reproduced here for
the preemption-recovery path:

```bash
mkdir -p data
curl -L --fail "$REPLAY_TAR_URL" | tar xz -C data/         # -> data/replays/...
curl -L --fail "$ENV_FILES_TAR_URL" | tar xz               # -> environment_files/...
```

~30 s per artifact on a 1 Gbit/s Vast instance vs ~5 minutes via the
Drive API + the looming quota cliff.

### DreamerV3 launch (Phase 2 - Crafter sanity)

Uses our launcher (per D12) - which delegates to dreamerv3's existing
Crafter wrapper for the `crafter_reward` task.

See [`docs/vast-quickstart.md`](vast-quickstart.md) for the exact
copy-pasteable launch sequence (Crafter then vc33 in one ~3h session).

Resume after preemption: re-run with the **same** `--logdir`. embodied's
training loop auto-loads the latest checkpoint.

### Phase 3 - cross-game WM pretrain

Phase 3 loads all 340 staged replays into a DreamerV3 buffer and trains
the world model only (enc + RSSM + dec + rew + con - the 5 modules in
`WMOnlyAgent.wm_modules`). Actor + critic stay at their initial weights
and are trained per-game in Phase 4.

The entry-point script is `scripts/pretrain_wm.py` (full run) +
`scripts/pretrain_wm_smoke.py` (gating smoke). WM-only enforcement
lives in `arc3_wm/wm_only_agent.py::WMOnlyAgent` (subclass of
`dreamerv3.agent.Agent` per D12; the override branches before
`self.imagine(...)` so the actor/critic optimizer never fires).

#### Phase-3-only extra prereq: TensorFlow for jax.profiler

Phase 3 enables `jax.profiler.start_trace`, which lazily imports
TensorFlow on first use. Crafter (Phase 2) doesn't trip this path,
so a fresh image works there without TF - but the Phase-3 smoke and
full pretrain will crash on the first profile event without it.
Install before launching either:

```bash
pip install tensorflow
```

#### Phase-3 smoke (must pass before the full pretrain)

The smoke runs a real `WMOnlyAgent` against a tiny ~50-transition
synthetic buffer for ~1500 steps. Wall-clock ~minutes on H100; emits
all four WM losses to JSONL so the curves can be eyeballed before
the RHAE held-out hook (step 5b) wires in.

```bash
cd ~/arc3-wm   # the cloned repo, third_party/dreamerv3 already pulled

# Sanity check the WM-only assertion suite first (Vast-only tests
# that skip on the laptop). These exercise the subclass relation,
# wm_modules == 5, wm_opt distinct from inherited opt, and the
# loss-tree shape.
python -m pytest tests/test_wm_only_agent.py -v

# Run the smoke. ~minutes on H100; emits ~scope JSONL under logdir.
python scripts/pretrain_wm_smoke.py \
    --logdir ~/logdir/pretrain-smoke-{timestamp} \
    --steps 1500
```

Eyeball the curves either via Scope viewer or by tailing the JSONL
directly:

```bash
# Scope viewer - open in a browser tab.
python -m scope.viewer --basedir ~/logdir &

# Or tail-and-grep the metrics JSONL.
tail -f ~/logdir/pretrain-smoke-*/scope/metrics.jsonl | \
    grep -oE '"train/loss/(dyn|rew|con|image)":[^,]*'
```

**Pass criteria (informal, qualitative):**

- All four WM losses emit non-NaN values from step ~10 onward.
- `train/loss/image` trends down monotonically (or near-so).
- `train/loss/{dyn,rew,con}` do not explode.
- No assertion failures, no shape mismatches, no buffer-drain warnings.

If the smoke is clean, sign off on step 5b (RHAEHeldOutHook impl) +
the full Phase-3 pretrain. If anything looks wrong, flag it before
proceeding - debug order: action mapping -> reward scale -> buffer
schema (per CLAUDE.md risk #3 + Phase-3 anti-pattern guard).

#### Full Phase-3 pretrain (after smoke sign-off + step 5b)

The full pretrain runs on the real 340-replay buffer, ~5h on H100,
with 30-min checkpoint cadence. Launch sequence (forthcoming once
step 5b lands; placeholder for the runbook).

### Preemption-recovery checklist

- Check `/workspace/logdir/${RUN}/ckpt.jax` mtime - should be < 30 min old.
- Re-run the same `tmux new-session` command. embodied logs
  `Loading checkpoint...` on successful resume.
- If the volume is empty (rare; some Vast volumes don't persist across
  region failover): re-run the first-run setup script (it's idempotent -
  repeat-`apt-get` and repeat-`pip install` are no-ops), then re-launch.
- If `data/replays/` or `environment_files/` is missing post-recovery, the
  setup script's `curl|tar xz` lines re-stage them from B2 - assuming
  `REPLAY_TAR_URL` and `ENV_FILES_TAR_URL` are still exported in the shell.
  Re-export from your password manager if not.

### `~/logdir` sync (Phase 2/3)

Cron (or `tmux`-launched bash loop) every 30 minutes, regardless of
preemption:

```bash
while true; do
  sleep 1800
  rsync -a /workspace/logdir/ /workspace/logdir-snapshot/ || true
  # Optional: also push to B2 to survive volume loss.
  b2 sync /workspace/logdir-snapshot b2://arc3-wm-logs/$(hostname) || true
done
```

### Cost estimate (rough, current Vast prices)

| Phase | Instance | Hours | $/hr | $ |
|---|---|---|---|---|
| 2 - Crafter sanity | 1x H100 80GB spot | 6-12 | ~$1.80 | $11-22 |
| 3 - Cross-game WM pretrain | 1x H100 80GB spot | ~6 | ~$1.80 | ~$11 |
| **Total Phase 2 + 3** | | | | **~$25-45** |

Stays within the CLAUDE.md $25-50 budget.

## Local 5070 cluster (Phases 4-5)

### Topology assumptions

- 9-15 RTX 5070 GPUs, each in a separate machine OR multi-GPU box.
- Each is independently addressable (SSH or local). No NCCL needed -
  per-game runs are fully independent.
- Shared storage: NFS or rsync of `data/replays/` and the Phase 3 WM
  checkpoint.

### Phase 4 (forthcoming) - 3-game pilot

Phase-4 launch will add WM checkpoint warm-start, fresh actor+critic
init, and per-game replay pre-population. Current launcher supports
single-game from-scratch training only. This section will be rewritten
against real code when Phase 4 lands.

### Phase 5 (forthcoming) - full 25-game sweep

Phase 5 reuses the Phase-4 launcher across 25 games x 3 seeds in
parallel. This section will be rewritten when Phase 4 lands and the
sweep-driver shape is known.

### Wall-clock estimate

- Phase 4 pilot: 3 games x 2 seeds x 500k steps x ~10h/M-steps ~ 30 GPU-hours.
  6x 5070 in parallel => ~5 hours wall-clock.
- Phase 5 sweep: 25 games x 3 seeds x 1M steps ~ 750 GPU-hours.
  9x 5070 in parallel => ~83 hours = ~3.5 days wall-clock.

### Failure modes (general - apply to any phase)

- GPU drops out -> flag immediately, do not silently continue with fewer seeds.
- Per-game NaN -> kill, save the divergent ckpt for forensics, requeue with
  a fresh seed only after Haso confirms - divergence may be a real signal.

## Anti-goals (do NOT do)

- Don't run the laptop overnight on Phase 4/5 - local 5070s are the spec.
- Don't pay for non-preemptible Vast.ai instances. Spot only.
- Don't fork DreamerV3. Register the env via `embodied/`, period.
- Don't use ONLINE mode for any training/eval. OFFLINE only. Rate limits
  will eat the run.
- Don't run `gdown --folder` on a remote - use the object-storage tarball
  pattern. We hit Drive's per-IP quota on the laptop in Phase 0; the same
  quota will trip on a fresh Vast IP that's been used by anyone else.

## Open questions for Haso

1. Object storage choice - Backblaze B2 by default; OK to set up?
2. Vast.ai region preference - any constraint?
3. Cluster topology for 5070s - is `/shared/` available, or must we
   `rsync` to each box? Affects whether Phase 4 launch can be a single
   command or a per-host loop.
