# scripts/

Command-line entry points behind the ARC-AGI-3 world-model study. Run
each from the repository root; every script documents its arguments
under `--help` or in its module docstring.

## Data and environment setup
- `cache_env_files.py` - download OFFLINE game files into `environment_files/` (needs `ARC_API_KEY`); required before OFFLINE `make()` succeeds. Caches the pilot set by default; pass game ids or `--all` for the full 25.
- `fetch_docs.py` - fetch the ARC-AGI-3 reference docs from the upstream site into `docs/arc-agi-3/`.
- `extract_human_baselines.py` - build `data/human_baselines.json` (per-game, per-level upper-median action counts) from the 340 human replays.

## DreamerV3 training
- `pretrain_wm.py` - cross-game, world-model-only pretraining on the mixed 340-replay buffer.
- `launch_pergame.py` - per-game DreamerV3 launcher (warm-started or from scratch, `--script train_eval`).

## Baselines
- `ppo_baseline.py` - PPO on the Gymnasium substrate.
- `eval_random_rhae.py` - random-policy offline RHAE on a single game (the zero-skill reference point).

## Metric and result tables
- `compute_rhae.py` - post-hoc RHAE from an eval-episode reward-stream JSONL plus the baseline fixture.
- `action_diagnostics.py` - structural and validity profile of the 4102-way action space for one game, as JSON or CSV.
- `build_benchmark_table.py` - assemble the paired cold/warm 6-game RHAE table into `analysis/benchmark_table.{md,json}`.
- `build_ladder_table.py` - generate the paper's result tables directly from the run tree.
- `build_iclr_tables.py` - generate the paper's tables from the committed data files under `analysis/`.

## Frozen-model probes
Protocol in [docs/dynamics-competence-probe.md](../docs/dynamics-competence-probe.md).
- `probe_dump_latents.py`, `probe_fit_probes.py` - dump frozen RSSM latents, then fit linear probes on them.
- `probe_refit.py` - re-score the latent probes with permutation tests and Holm correction (writes `analysis/probe_refit.*`).
- `probe_collect_holdout.py`, `probe_predict.py`, `probe_score.py` - the three stages of the dynamics-competence probe: collect ground-truth episodes, run the frozen world model forward, score the predictions.
- `probe_predict_synthetic.py` - synthetic stand-in for the predict stage, for local dry runs without JAX.
- `probe_consolidate.py`, `probe_make_summary.py` - merge the probe artifacts into results CSVs.
- `probe_figures.py`, `probe_fig_rollouts.py` - regenerate the probe figures and the imagined-rollout strip.

## TRM supervised arms
Components in [docs/trm-components.md](../docs/trm-components.md); install with `pip install -e ".[trm]"`.
- `trm_preprocess_replays.py`, `trm_gen_synth.py` - build the per-game replay tensor caches and the synthetic-game corpus.
- `trm_train_bc.py`, `trm_train_wm.py` - train the behavior-cloning policy and the discrete world model.
- `trm_eval_agent.py` - evaluate an agent composition online and write RHAE-ready episodes.
- `trm_build_results.py` - aggregate a sweep directory into result tables.
- `trm_rollout_probe.py`, `trm_wm_audit.py`, `trm_halt_probe.py` - open-loop rollout fidelity, overfitting audit, and halting probe for a trained model.
- `trm_episode_film.py`, `trm_rollout_film.py` - render an evaluation episode or an imagined rollout as a filmstrip.

## Smoke checks
- `random_agent_smoke.py` - random-agent episodes on vc33 (OFFLINE, high-FPS path).
- `smoke_full_replays.py` - parse every staged replay and report aggregate stats.
- `pretrain_wm_smoke.py` - GPU-only check that the world-model-only training path steps.
