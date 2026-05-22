# scripts/

Operational scripts behind the ARC-AGI-3 world-model study. Run each
from the repository root.

## Data & environment setup
- `cache_env_files.py` — download OFFLINE game files into `environment_files/` (needs `ARC_API_KEY`); required before OFFLINE `make()` succeeds.
- `stage_env_files.sh` — stage cached env files onto a remote training machine.
- `fetch_docs.py` — fetch the ARC-AGI-3 reference docs into `docs/arc-agi-3/`.
- `extract_human_baselines.py` — build `data/human_baselines.json` (per-game/per-level upper-median action counts) from the 340 human replays.

## Training
- `pretrain_wm.py` — cross-game, world-model-only pretraining on the mixed 340-replay buffer.
- `launch_pergame.py` — per-game DreamerV3 launcher (warm-started or from-scratch, `--script train_eval`).
- `launch_phase4_proper.sh`, `launch_phase4_fromscratch.sh`, `launch_phase4_expansion_warm.sh`, `launch_phase4_expansion_fromscratch.sh` — the exact wrappers reproducing the paired 6-game × 2-seed sweep.

## Metric & results
- `compute_rhae.py` — post-hoc RHAE from an eval-episode reward-stream JSONL plus the baseline fixture.
- `build_benchmark_table.py` — assemble the paired cold/warm 6-game RHAE table into `analysis/benchmark_table.{md,json}`.

## Smoke checks
- `random_agent_smoke.py` — random-agent episodes on vc33 (OFFLINE, high-FPS path).
- `smoke_full_replays.py` — parse every staged replay and report aggregate stats.
- `pretrain_wm_smoke.py` — GPU-only smoke that the WM-only training path actually steps.
