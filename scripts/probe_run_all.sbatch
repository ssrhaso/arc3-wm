#!/bin/bash
# Master "do everything" probe sweep (CPU, Isambard workq). Runs all three
# frozen-WM probes across every per-game checkpoint and saves all raw artifacts,
# then fits/scores and regenerates figures. Re-run probe_figures.py alone any
# time to regenerate plots from the saved artifacts without re-touching a node.
#
# Run only AFTER probe_cd82.sbatch validates the pipeline. Submit from repo root:
#   cd ~/arc3-wm && sbatch scripts/probe_run_all.sbatch
#
# aarch64 CPU notes baked in: --platform cpu + --compute-dtype float32 (bf16
# hits an LLVM AArch64 codegen bug); compute node (login ulimit -u kills XLA).
#
#SBATCH --job-name=probe-all
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.ahmad@exeter.ac.uk
cd "$HOME/arc3-wm"
module load cray-python/3.11.7
source .venv-probe/bin/activate
export JAX_PLATFORMS=cpu
CD="--platform cpu --compute-dtype float32"   # aarch64 CPU flags
C=4; H=8
B2="https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/phase4-proper"
GAMES="vc33 sb26 cd82 tn36 ls20 lf52"
RANDOM_GAMES="vc33 cd82 sb26 lf52"   # the four whose random holdout we staged
HOLD="results/dynamics_probe/holdout"
run() { echo "+ $*"; "$@" || echo "!! FAILED: $*"; }   # tolerate per-step failure

echo "=== devices ==="; python -c "import jax; print(jax.devices())" || true

# 0. fetch checkpoints (warm seed 0) not already present
for g in $GAMES; do
  if [ ! -d "ckpt_${g}" ]; then
    echo "=== fetch ckpt ${g} ==="
    curl -fsSL -o "ckpt_${g}.tar.gz" "${B2}/p4-${g}-s0-warm-98de390/ckpt-final.tar.gz" \
      && mkdir -p "ckpt_${g}" && tar xzf "ckpt_${g}.tar.gz" -C "ckpt_${g}" || echo "!! ckpt ${g} fetch failed"
  fi
done

# 1. STEP-1 latent probe: dump RSSM latents (human source = multi-level coverage)
for g in $GAMES; do
  run python scripts/probe_dump_latents.py --game "$g" --source human \
    --ckpt "ckpt_${g}" --holdout "${HOLD}/${g}_human.npz" $CD \
    --outdir results/dynamics_probe/latents
done

# 2. PROBE B rollout fidelity: human (all 6) + random (the 4 staged)
for g in $GAMES; do
  run python scripts/probe_predict.py --game "$g" --source human --mode rollout \
    --ckpt "ckpt_${g}" --holdout "${HOLD}/${g}_human.npz" \
    --context-len $C --horizon $H $CD --outdir results/dynamics_probe/pred
done
for g in $RANDOM_GAMES; do
  run python scripts/probe_predict.py --game "$g" --source random --mode rollout \
    --ckpt "ckpt_${g}" --holdout "${HOLD}/${g}_random.npz" \
    --context-len $C --horizon $H $CD --outdir results/dynamics_probe/pred
done

# 3. PROBE A counterfactual (needs per-frame availability => random source).
#    Writes *_cf.npz into the SAME pred/ dir so one probe_score builds one table.
for g in $RANDOM_GAMES; do
  run python scripts/probe_predict.py --game "$g" --source random --mode counterfactual \
    --ckpt "ckpt_${g}" --holdout "${HOLD}/${g}_random.npz" \
    --context-len $C --target-a 4 --n-click 8 $CD --outdir results/dynamics_probe/pred
done

# 4. SCORE (rollout + counterfactual, one table) and FIT (latent probes) - CPU
run python scripts/probe_score.py --pred-dir results/dynamics_probe/pred \
  --outdir results/dynamics_probe/scored
run python scripts/probe_fit_probes.py --latents-dir results/dynamics_probe/latents \
  --outdir results/dynamics_probe/scored

# 5. FIGURES from the saved artifacts
run python scripts/probe_figures.py --scored-dir results/dynamics_probe/scored \
  --outdir results/dynamics_probe/figures

echo "=== competence table ==="; cat results/dynamics_probe/scored/competence_table.csv 2>/dev/null || true

# 6. BUNDLE everything for one-scp retrieval (raw npz + scored + figures + this log)
echo "=== bundling results ==="
cp "probe-all_${SLURM_JOB_ID}.log" "results/dynamics_probe/run.log" 2>/dev/null || true
tar czf "$HOME/arc3-wm/dynamics-probe-results.tar.gz" -C "$HOME/arc3-wm" results/dynamics_probe
ls -la "$HOME/arc3-wm/dynamics-probe-results.tar.gz"
echo "=== DONE ===  retrieve: scp u6ko.aip2.isambard:~/arc3-wm/dynamics-probe-results.tar.gz ."
