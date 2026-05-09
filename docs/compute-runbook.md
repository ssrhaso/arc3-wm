# Compute runbook (Phases 2–5)

> Operational playbook for the Vast.ai (Phases 2–3) and local 5070 cluster
> (Phases 4–5) workloads. Mirrors §"Compute" in `CLAUDE.md`. Update this
> file at every phase boundary.

## Single mantra

> **Always launch with `--logdir <persistent>` so that re-running the same
> command resumes from the last checkpoint.** This is the *only*
> preemption-safety primitive we rely on. Without it, every Vast.ai eviction
> wastes ~30 minutes of training.

## Vast.ai (Phases 2–3)

### Account / billing

- Use spot ("interruptible") pricing only.
- Phase 2 + 3 budget: **$25–50 total**. If a single instance is approaching
  $20 of that, stop and reassess before continuing.
- Prefer H100 80GB SXM/PCIe over A100 80GB — typically ~30% better
  $/throughput on Vast.ai for DreamerV3-sized workloads. A100 40GB is fine
  for `size12m`.

### Provisioning a fresh spot instance

In the Vast.ai console:

1. **Image:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (or the closest
   CUDA 12.x image Vast offers). DreamerV3 needs JAX-CUDA12.
2. **Disk:** ≥ 80 GB persistent storage volume (replays alone are ≈ 30 GB).
3. **Region:** prefer EU/US with the best $/H100. Avoid TOR/RU for
   network reliability.
4. **GPU:** 1× H100 SXM5 80GB (preferred) or 1× A100 80GB.
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
  logdir/                    # DreamerV3 logdir — must persist across preemption
  checkpoints/               # WM checkpoints, every 30 minutes
~/.cache/                    # ephemeral; OK to lose
```

`--logdir /workspace/logdir/<run-name>` for every DreamerV3 launch.

### First-run setup script

Run on a fresh spot instance, after SSH. Five env vars must be exported
before this script runs (laptop → 1Password / SSH agent / paste in the
remote shell — never commit any of these):

```bash
export GITHUB_REPO=https://github.com/ssrhaso/ARC_AGI_3.git
export ARC_API_KEY=<from https://three.arcprize.org>
export REPLAY_TAR_URL=<B2 public URL — see "Data staging" below>
export ENV_FILES_TAR_URL=<B2 public URL for environment_files.tar.gz>
export WANDB_PROJECT=arc3-wm-sprint   # optional; logger picks up automatically
# export WANDB_API_KEY=<from wandb.ai/settings>   # only if WANDB_PROJECT set
```

Then:

```bash
set -euo pipefail

# 0. System packages — Crafter renders via OpenGL on a fresh Linux image.
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

# 3. JAX GPU FIRST — this order is REQUIRED by danijar/dreamerv3.
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
[ "$(find data/replays -name '*.jsonl' | wc -l)" = "342" ] || \
  echo "WARNING: replay count != 342" >&2

# 9. Stage environment_files (the cached arc_agi game source) so OFFLINE
#    mode finds metadata.json without an API call.
[ -d environment_files ] || curl -L --fail "$ENV_FILES_TAR_URL" | tar xz

# 10. Sanity: full pytest suite (skips the dreamerv3 dry-run on laptop;
#     here it should run because JAX is now installed).
pytest -q
```

If `pytest -q` is green, the instance is ready for Phase-2 launch (see
"DreamerV3 launch (Phase 2 — Crafter sanity)" below).

### Data staging — laptop → object storage → instance

Cheap and provider-agnostic. Per CLAUDE.md, **never re-`gdown` from Drive
on a remote** — Drive's per-IP quota is exactly the failure mode we hit on
the laptop in Phase 0 (only 39 of 342 files retrieved).

**Target bucket:** Backblaze B2, single bucket `arc3-wm-data` (~$6/TB-month,
public download free under reasonable usage). Use this exact name in every
command below — the runbook depends on it. Alternatives if B2 access
isn't workable: AWS S3, Cloudflare R2 (free egress), or any HTTPS-served
bucket; substitute the bucket name end-to-end.

#### Two artifacts to stage today

| Artifact                         | Source on laptop                          | Target path in bucket                  | Consumed by              |
|----------------------------------|-------------------------------------------|----------------------------------------|--------------------------|
| `replays.tar.gz`                 | `data/replays/` (342 JSONLs, Phase 3 prereq) | `arc3-wm-data/replays.tar.gz`          | Phase 3 WM pretrain      |
| `environment_files.tar.gz`       | `environment_files/` (cached game source) | `arc3-wm-data/environment_files.tar.gz`| All phases — OFFLINE mode |

A third artifact (`pretrained_wm.tar.gz`) will join the table when Phase
3 produces its first checkpoint.

#### Laptop → bucket (one-time per artifact)

Prereqs: `pip install b2sdk b2` and `b2 account authorize` once with
your Backblaze application key. Bucket must be public-read (or use
presigned URLs and pin them in the run script).

```bash
# Replays (run AFTER OAuth gdown lands all 342, per D1).
find data/replays -name '*.jsonl' | wc -l     # must equal 342 before tarball.
tar czf replays.tar.gz data/replays
b2 file upload arc3-wm-data replays.tar.gz replays.tar.gz

# Environment files (run AFTER scripts/cache_env_files.py is rerun for all 25 games;
# Phase-0 cache only covered vc33/tu93/cd82).
tar czf environment_files.tar.gz environment_files
b2 file upload arc3-wm-data environment_files.tar.gz environment_files.tar.gz

# Capture the public download URLs once and persist them in your password
# manager. Each URL is stable as long as the bucket stays public-read.
b2 file url arc3-wm-data/replays.tar.gz             # → REPLAY_TAR_URL
b2 file url arc3-wm-data/environment_files.tar.gz   # → ENV_FILES_TAR_URL
```

#### Instance: pull-and-extract

Already wired into the first-run setup script above; reproduced here for
the preemption-recovery path:

```bash
mkdir -p data
curl -L --fail "$REPLAY_TAR_URL" | tar xz -C data/         # → data/replays/...
curl -L --fail "$ENV_FILES_TAR_URL" | tar xz               # → environment_files/...
```

~30 s per artifact on a 1 Gbit/s Vast instance vs ~5 minutes via the
Drive API + the looming quota cliff.

### DreamerV3 launch (Phase 2 — Crafter sanity)

Uses our launcher (per D12) — which delegates to dreamerv3's existing
Crafter wrapper for the `crafter_reward` task.

See [`docs/vast-quickstart.md`](vast-quickstart.md) for the exact
copy-pasteable launch sequence (Crafter then vc33 in one ~3h session).

Resume after preemption: re-run with the **same** `--logdir`. embodied's
training loop auto-loads the latest checkpoint.

### Phase 3 (forthcoming) — cross-game WM pretrain

Phase 3 will load all 342 replays into a DreamerV3 buffer and train the
world model only (encoder + RSSM + decoders + reward + continue heads;
actor and critic frozen). The entry point script and its replay-loader
dependency do not yet exist. This section will be rewritten against real
code when Phase 3 lands.

### Preemption-recovery checklist

- Check `/workspace/logdir/${RUN}/ckpt.jax` mtime — should be < 30 min old.
- Re-run the same `tmux new-session` command. embodied logs
  `Loading checkpoint…` on successful resume.
- If the volume is empty (rare; some Vast volumes don't persist across
  region failover): re-run the first-run setup script (it's idempotent —
  repeat-`apt-get` and repeat-`pip install` are no-ops), then re-launch.
- If `data/replays/` or `environment_files/` is missing post-recovery, the
  setup script's `curl|tar xz` lines re-stage them from B2 — assuming
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
| 2 — Crafter sanity | 1× H100 80GB spot | 6–12 | ~$1.80 | $11–22 |
| 3 — Cross-game WM pretrain | 1× H100 80GB spot | ~6 | ~$1.80 | ~$11 |
| **Total Phase 2 + 3** | | | | **~$25–45** |

Stays within the CLAUDE.md $25–50 budget.

## Local 5070 cluster (Phases 4–5)

### Topology assumptions

- 9–15 RTX 5070 GPUs, each in a separate machine OR multi-GPU box.
- Each is independently addressable (SSH or local). No NCCL needed —
  per-game runs are fully independent.
- Shared storage: NFS or rsync of `data/replays/` and the Phase 3 WM
  checkpoint.

### Phase 4 (forthcoming) — 3-game pilot

Phase-4 launch will add WM checkpoint warm-start, fresh actor+critic
init, and per-game replay pre-population. Current launcher supports
single-game from-scratch training only. This section will be rewritten
against real code when Phase 4 lands.

### Phase 5 (forthcoming) — full 25-game sweep

Phase 5 reuses the Phase-4 launcher across 25 games × 3 seeds in
parallel. This section will be rewritten when Phase 4 lands and the
sweep-driver shape is known.

### Wall-clock estimate

- Phase 4 pilot: 3 games × 2 seeds × 500k steps × ~10h/M-steps ≈ 30 GPU-hours.
  6× 5070 in parallel ⇒ ~5 hours wall-clock.
- Phase 5 sweep: 25 games × 3 seeds × 1M steps ≈ 750 GPU-hours.
  9× 5070 in parallel ⇒ ~83 hours = ~3.5 days wall-clock.

### Failure modes (general — apply to any phase)

- GPU drops out → flag immediately, do not silently continue with fewer seeds.
- Per-game NaN → kill, save the divergent ckpt for forensics, requeue with
  a fresh seed only after Haso confirms — divergence may be a real signal.

## Anti-goals (do NOT do)

- Don't run the laptop overnight on Phase 4/5 — local 5070s are the spec.
- Don't pay for non-preemptible Vast.ai instances. Spot only.
- Don't fork DreamerV3. Register the env via `embodied/`, period.
- Don't use ONLINE mode for any training/eval. OFFLINE only. Rate limits
  will eat the run.
- Don't run `gdown --folder` on a remote — use the object-storage tarball
  pattern. We hit Drive's per-IP quota on the laptop in Phase 0; the same
  quota will trip on a fresh Vast IP that's been used by anyone else.

## Open questions for Haso

1. Object storage choice — Backblaze B2 by default; OK to set up?
2. Vast.ai region preference — any constraint?
3. Cluster topology for 5070s — is `/shared/` available, or must we
   `rsync` to each box? Affects whether Phase 4 launch can be a single
   command or a per-host loop.
