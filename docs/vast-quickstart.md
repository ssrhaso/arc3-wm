# Vast.ai quickstart - Milestones (2) + (3) combined sanity

> ~3h wall-clock, ~$5 budget. Goal: prove milestone (2) (Crafter) then
> milestone (3) (vc33) spin up cleanly end-to-end on real GPU. Not full
> Phase 2/3 validation - Phase 4 is the real load-bearing gate.

## Pre-flight checklist (read before clicking "rent")

| Item | Required value | Verified against |
|---|---|---|
| GPU | 1x H100 80GB SXM5 spot ("interruptible") | CLAUDE.md Section "Compute" |
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | dreamerv3 pins `jax[cuda12]==0.4.33` (CUDA 12.x) |
| Disk | >= 80 GB persistent volume | dreamerv3+JAX ~5GB, logdir buffer cap ~5GB, crafter+arc-agi < 1GB, no replays/env_files staging needed (see below) |
| nvidia-smi | works out of the box on PyTorch+CUDA Vast images | confirmed via `pip install`-only path |
| SSH key | Haso's public key uploaded at provisioning time | - |
| Spot bid | competitive at ~$1.80/h H100 | - |

**Replays + env_files: NOT staged via B2 for this run.** B2 is a Phase-3+
concern; for milestones (2)/(3):
- Crafter has no replays / no env_files dependency.
- vc33's env_files (~10 KB total) are fetched on-instance via
  `scripts/cache_env_files.py` (NORMAL mode, ~3 seconds, uses `ARC_API_KEY`).

So you only need three env vars exported in the SSH session before the
setup command:

```bash
export ARC_API_KEY=<from https://three.arcprize.org>
# Optional - turns on W&B logging if set; otherwise JSONL+Scope only.
export WANDB_PROJECT=arc3-wm-sprint
# Required only if WANDB_PROJECT is set.
# export WANDB_API_KEY=<from wandb.ai/settings>
```

## Repo access

`git clone https://github.com/ssrhaso/ARC_AGI_3.git` works only if the
GitHub repo is public. If it's private, use one of:

```bash
# Option A: HTTPS with personal access token (PAT).
export GITHUB_TOKEN=<your PAT>
git clone https://${GITHUB_TOKEN}@github.com/ssrhaso/ARC_AGI_3.git

# Option B: SSH (requires deploy key or your SSH key on the instance).
git clone git@github.com:ssrhaso/ARC_AGI_3.git
```

Verify before spinning the instance: open the repo URL in a browser
without being logged in. If you see the README, it's public.

---

## The three commands

Three pasteable blocks. Edit nothing.

### 1. Setup (~20 min, ~$0.70)

**If repo is private, export `GITHUB_TOKEN` (PAT with `repo` read scope)
before pasting.** Public repo: skip; the clone line works as-is.

```bash
set -euo pipefail
cd /workspace
# Vast PyTorch images run as root; no sudo needed and apt-utils is absent.
[ -d arc3-wm ] || git clone "https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/ssrhaso/ARC_AGI_3.git" arc3-wm
cd arc3-wm

apt-get update -y
apt-get install -y libgl1 libglib2.0-0 git tmux jq

# Python 3.11 OR 3.12 both work; danijar/dreamerv3 requires >= 3.11.
# Vast's vastai/pytorch image ships 3.12 at /usr/bin/python3.12.
PYBIN="$(command -v python3.11 || command -v python3.12)"
"$PYBIN" -m venv .venv
. .venv/bin/activate
pip install -U pip wheel setuptools

# JAX GPU FIRST - required by danijar/dreamerv3.
pip install -U "jax[cuda12]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# third_party/ is gitignored (Phase 0 D5, "huge, fetch fresh") so the
# vendored copies are NOT in the cloned repo. Clone + pin to the same
# commits the laptop tested against.
mkdir -p third_party
[ -d third_party/dreamerv3 ] || git clone https://github.com/danijar/dreamerv3.git third_party/dreamerv3
(cd third_party/dreamerv3 && git checkout b65cf81)
[ -d third_party/ARC-AGI-3-Agents ] || git clone https://github.com/arcprize/ARC-AGI-3-Agents.git third_party/ARC-AGI-3-Agents
(cd third_party/ARC-AGI-3-Agents && git checkout 135f20a)

# DreamerV3 deps + Crafter (not in their requirements.txt) + our project.
# wandb is required because the launcher's wandb-guarded path and one of
# our dry-run tests both reach into the real elements.logger.WandBOutput.
pip install -U -r third_party/dreamerv3/requirements.txt
pip install crafter wandb
pip install -e .
pip install pytest pytest-xdist pytest-timeout gdown

# .env so arc_agi auto-loads OFFLINE mode + the API key.
cat > .env <<EOF
ARC_API_KEY=$ARC_API_KEY
OPERATION_MODE=offline
ENVIRONMENTS_DIR=environment_files
RECORDINGS_DIR=recordings
EOF
chmod 600 .env

# Cache vc33/tu93/cd82 env_files via NORMAL mode (~3s each).
# Uses scripts/cache_env_files.py which sets OPERATION_MODE=normal
# in os.environ BEFORE importing arc_agi to defeat the env-var-wins quirk.
python scripts/cache_env_files.py

# Sanity: GPU detected by JAX (check class name not .platform; JAX
# reports device.platform == 'gpu' for CUDA, the class is CudaDevice).
python -c "import jax; devs=jax.devices(); assert any('Cuda' in type(d).__name__ for d in devs), f'no CUDA device: {devs}'; print('JAX devices:', devs)"

# WANDB_MODE=offline so the wandb-init path in test_launcher_dry_run
# doesn't block on missing API auth. (Set this just for pytest; real
# training runs do NOT set WANDB_PROJECT, so wandb stays disabled.)
WANDB_MODE=offline pytest -q
```

**Pass condition:** the final `pytest -q` line prints a green summary.
On Vast (with JAX installed) the laptop-skipped `test_launcher_dry_run.py`
runs - expect **all green, 0 skipped**. If anything fails, stop and ask;
do not launch milestone (2)/(3).

### 2. Crafter launch (~30-60 min, ~$1-2)

```bash
cd /workspace/arc3-wm && . .venv/bin/activate
export RUN=crafter_$(date +%Y%m%d_%H%M%S)
mkdir -p /workspace/logdir
tmux new-session -d -s "$RUN" \
  "python scripts/launch_pergame.py \
    --logdir /workspace/logdir/${RUN} \
    --configs crafter size12m \
    --task crafter_reward \
    --seed 0 \
    --run.steps 100000 \
    > /workspace/logdir/${RUN}.stdout.log 2>&1"
echo "Launched: $RUN"
echo "Re-attach with: tmux attach -t $RUN"
```

**Pass condition:** see "Monitoring" below. Run 60 minutes max; cap at
100k env steps.

### 3. vc33 launch (~30-60 min, ~$1-2)

Run AFTER Crafter has either finished or you've killed it (`tmux kill-session -t $RUN`).

```bash
cd /workspace/arc3-wm && . .venv/bin/activate
export RUN=arc3_vc33_$(date +%Y%m%d_%H%M%S)
mkdir -p /workspace/logdir
tmux new-session -d -s "$RUN" \
  "python scripts/launch_pergame.py \
    --logdir /workspace/logdir/${RUN} \
    --configs size12m arc3 \
    --task arc3_vc33 \
    --seed 0 \
    --run.steps 100000 \
    > /workspace/logdir/${RUN}.stdout.log 2>&1"
echo "Launched: $RUN"
echo "Re-attach with: tmux attach -t $RUN"
```

**Pass condition:** see "Monitoring" below.

---

## Monitoring

**Heads-up on JAX JIT compile:** the first training step compiles size12m
on H100, which can take 5-10 minutes. During that window `metrics.jsonl`
will be empty and `nvidia-smi` will show GPU memory allocated but ~0%
utilisation. **Don't panic in the first 10 minutes.** First log line is
the signal training has actually started.

### "Is it alive?" - run 5-15 min after launch

```bash
tmux ls; nvidia-smi | head -20; \
  ls -lh /workspace/logdir/${RUN}/ 2>/dev/null; \
  echo "--- tail stdout ---"; tail -20 /workspace/logdir/${RUN}.stdout.log
```

What you want to see:
- `tmux ls` lists `$RUN`.
- `nvidia-smi` shows the python process using the H100 (memory allocated;
  >= 30% utilisation post-compile).
- `logdir/${RUN}/` contains `config.yaml` (always) and after compile
  starts, `metrics.jsonl` (and a `ckpt.jax` once `save_every` ticks).
- stdout has lines like `Start training loop`. NaN / Traceback /
  `OOM` / `Killed` are bad - see Failure-mode playbook below.

### "Are losses descending?" - run after the first ~5-10 minutes of training

```bash
tail -20 /workspace/logdir/${RUN}/metrics.jsonl | \
  jq -c '{step, loss_total: .["train/loss/total"], recon: .["train/loss/recon"], dyn: .["train/loss/dyn"]}' 2>/dev/null
```

Look for: `loss_total` trending down across the last 10 reports.
Crafter additionally has a reward signal - substitute key
`.["episode/score"]` from `scores.jsonl` to see reward climbing from 0.

### "Final score?" - run when the run finishes

```bash
echo "--- last 5 metric reports ---"; tail -5 /workspace/logdir/${RUN}/metrics.jsonl | jq -c .
echo "--- last 5 episode scores ---"; tail -5 /workspace/logdir/${RUN}/scores.jsonl | jq -c .
echo "--- run config ---"; cat /workspace/logdir/${RUN}/config.yaml | head -40
```

For vc33: `episode/score` is the env's delta-levels-completed reward, NOT
RHAE. RHAE is in the toolkit scorecard, which our launcher does not yet
read at end-of-run. RHAE > 0 is bonus, not required at this step budget
(per user spec).

### Tear down

```bash
tmux kill-session -t $RUN  # if still running
# Then stop the Vast instance from the console. Persistent volume
# survives if you keep the instance suspended; destroy if you don't
# care about the logdir.
```

---

## Failure-mode playbook

### Crafter milestone (2)

| Symptom | First debug step |
|---|---|
| **Task resolution error at launch** (within 30s, before JIT compile). stdout has `KeyError` or unknown task. | Crafter task naming may have shifted; check `third_party/dreamerv3/embodied/envs/crafter.py` for the current task list. Try `--task crafter_noreward` as fallback. |
| **NaN in loss before 30k steps.** Trace shows `loss_total = nan`. | (a) Check `jax.devices()` returned the H100, not CPU. (b) Re-launch with `--jax.compute_dtype float32` (default is bfloat16; some H100 cards have flaky bf16 stability under specific JAX/CUDA combos). Don't keep retrying without changing something. |
| **GPU not engaged.** `nvidia-smi` shows 0% util after 15 min, no `Start training loop` in stdout. | `python -c "import jax; print(jax.devices())"` - must show `CudaDevice`. If not, JAX fell back to CPU (env mismatch). Try `pip install -U "jax[cuda12]==0.4.33" --force-reinstall`. The current `jax-cuda12-plugin` resolves `nvidia-cudnn-cu12==9.22.x` (not 8.9 - earlier doc text said 8.9, that's stale); cuDNN 9 + CUDA 12.5 driver works. If pip resolves something older, `pip install -U nvidia-cudnn-cu12 --force-reinstall` then re-launch. |
| **OOM during JIT compile.** stdout has `RESOURCE_EXHAUSTED`. | size12m at default batch should fit on 80GB H100. If it doesn't: `--batch_size 8` (default 16). If still OOM: instance is wrong - confirm 80GB H100, not 40GB. |

### vc33 milestone (3)

| Symptom | First debug step |
|---|---|
| **`RuntimeError: arc_agi.Arcade.make('vc33') returned None`** at launch. | `environment_files/vc33/` is missing. Re-run `python scripts/cache_env_files.py`. Confirm `ARC_API_KEY` is exported and `.env` exists. |
| **`RuntimeError: ARC3GymEnv requires OFFLINE mode`** at launch. | `.env` not auto-loaded - check CWD when launcher runs (must be repo root) and that `.env` is present (`ls -la .env`). |
| **dtype mismatch in replay buffer.** Trace mentions `UnifyDtypes` or "expected uint8 got int8". | Our env's `_pack` casts obs to `uint8`. If this fires, it's a regression in `arc3_wm/embodied_env.py::_pack` - re-run `pytest tests/test_embodied_env.py -v` on the instance and capture the failure. Stop and ask; don't try to patch live on the meter. |
| **`KeyError: 'env.arc3.max_steps'`** at config build. | The `DEFAULT_ARC3_ENV` injection in `scripts/launch_pergame.py:load_merged_configs` regressed (D13). `pytest tests/test_launcher_arg_parsing.py -v` will reproduce. Stop and ask. |
| **`AssertionError: ('log/...', (N,), dtype(...))`** in `embodied/run/train.py::logfn` at first driver step. | dreamerv3's logfn asserts every `log/*` key in the obs is a SCALAR. Fixed in commit 73c6d09 by dropping `log/action_mask` from `arc3_wm/embodied_env.py`. If a future change re-adds a non-scalar `log/*` key, this row will fire - drop the key or pre-reduce to a scalar. |

### Rule of thumb

If the failure isn't in the table above, **stop the run** (`tmux kill-session`), capture
`stdout.log` + `metrics.jsonl` + `config.yaml` from the logdir, and ask
before relaunching. The instance keeps burning ~$1.80/h either way; a
5-minute pause to think is cheaper than a 30-minute reboot loop.

---

## What this run does NOT prove

- No replay loader, no RHAE module, no WM warm-start, no per-game replay
  pre-population. Phase 4 is where those become load-bearing.
- Reaching dreamerv3's Crafter reference reward (~11.7) - that's a 1M-step
  run, not 100k. Sanity passes if reward is *climbing*, not at reference.
- vc33 RHAE > 0 - possible at 100k steps but not required. The actor is
  learning from scratch on a Discrete(4102) action space with no masking.
