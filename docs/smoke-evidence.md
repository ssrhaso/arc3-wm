# Vast smoke evidence — milestones (2) + (3)

> Single point-in-time record of the first real Vast.ai bring-up. Cite
> this in CLAUDE.md "Current state" and in the paper's repro section
> when relevant. Not a runbook — see [vast-quickstart.md](vast-quickstart.md).

## Run metadata

| Field | Value |
|---|---|
| Date | 2026-05-09 |
| Instance | Vast.ai #36397706 (Czechia) |
| GPU | 1× NVIDIA H100 80GB HBM3 |
| Driver | 555.58.02 / CUDA 12.5 |
| Image | Vast `vastai/pytorch` (Python 3.12.13, conda venv `/venv/main` ignored, repo .venv used) |
| Disk | 54 GB overlay (under the doc's 80 GB spec; tight but worked for milestones (2)/(3) — no replays staged) |
| JAX | 0.4.33, jax-cuda12-plugin 0.4.33, jax-cuda12-pjrt 0.4.33 |
| cuDNN | 9.22.0.52 (resolved by pip; doc's earlier 8.9 advice was stale) |
| dreamerv3 | b65cf81 ("Update paper reference") |
| Test suite | 72 passed, 0 skipped on Vast (`WANDB_MODE=offline pytest -q`) |

## Milestone (2) — Crafter sanity

Launched `--configs crafter size12m --task crafter_reward --seed 0
--run.steps 100000`. Killed at step 10,710 once losses descended cleanly
— pass condition is "spins up + losses descending + reward climbing,"
not 100k completion (Crafter reference reward is a 1M-step number).

| Metric | Step 4_180 | Step 7_420 | Step 10_710 |
|---|---|---|---|
| `train/loss/image` | 250.49 | 67.04 | **56.40** |
| `train/loss/dyn` | 5.68 | 3.15 | 3.33 |
| `episode/score` | 2.1 | -0.9 | 4.1 |
| `fps/train` | 1.3e4 | 1.4e4 | 1.4e4 |

Image loss factor-4.5 descent across 6,500 steps; per-episode score
noise is normal Crafter behaviour. No NaN, no OOM, checkpoint saved.

## Milestone (3) — vc33 sanity

Launched `--configs size12m arc3 --task arc3_vc33 --seed 0
--run.steps 100000`. **First attempt crashed** at the first driver
step on `AssertionError: ('log/action_mask', (4102,), dtype('bool'))`
in `embodied/run/train.py::logfn`. Hot-fixed by dropping `log/action_mask`
from the embodied wrapper (commit 73c6d09 — see "Doc / code fixes
landed in this session" below).

Second attempt killed at step 49,232 with clean training:

| Metric | Step 49_232 |
|---|---|
| `train/loss/image` | 178.84 |
| `train/loss/dyn` | 4.43 |
| `train/loss/policy` | -1.9e-3 |
| `episode/score` | 0.0 (expected — vc33 reward is Δ-levels-completed; random policy rarely completes a level early) |
| `fps/policy` | 401.28 (arc_agi is much lighter than Crafter at 27 fps) |
| `fps/train` | 1.2e4 |
| Param count | 12,593,864 (size12m, +2M vs Crafter due to 4102-way policy head) |

Pass condition met. RHAE was not measured at this step budget per
spec.

## Doc / code fixes landed in this session

Issues caught during bring-up, all baked into the canonical setup:

1. **commit 73c6d09** — `log/action_mask` dropped from
   `arc3_wm/embodied_env.py`. dreamerv3's logfn asserts every `log/*`
   key is a scalar; a 4102-bool vector trips it. Per D11 the mask was
   decorative anyway.
2. **commit 22c6de2** — `docs/vast-quickstart.md` patched:
   (a) explicit `git clone third_party/dreamerv3 + ARC-AGI-3-Agents`
   with commit pins (third_party/ is gitignored, was never bundled);
   (b) `pip install wandb`; (c) `WANDB_MODE=offline pytest`; (d) JAX
   device check fixed (`'Cuda' in type(d).__name__` instead of
   `d.platform.lower() == 'cuda'`); (e) python3.11/3.12 fallback;
   (f) drop sudo (Vast images run as root, no apt-utils); (g) PAT
   auth via optional `GITHUB_TOKEN` for private repo clone;
   (h) cuDNN row in failure-mode playbook updated (cuDNN 9.x is
   correct, not 8.9).

## Cost

~80 minutes wall-clock end-to-end, on-demand pricing ~$1.82/h →
**~$2.40 spent** (well within the $5 budget; would be ~$1.20 on
spot/interruptible — toggle was not flipped on this run).

## What this run does NOT prove

Per the doc's standing caveats (still apply):

- Reaching dreamerv3's Crafter reference reward (~11.7) — that's a
  1M-step run, not 10k.
- vc33 RHAE > 0 — possible at 100k but not pursued here.
- Phase 3 readiness — replay loader, RHAE module, WM warm-start, B2
  bucket, OAuth gdown for the remaining 303 replays all still open.

Phase 4 remains the load-bearing gate.
