#!/bin/bash
# Follow-up probe analyses (CPU, Isambard workq). Three things:
#   1. Counterfactual on the HUMAN source for the dynamic games, filtered to
#      state-changing actions (the dilution-free version of Probe A).
#   2. Pretrained cross-game WM checkpoint: dump latents + fit level_id/transition
#      -> is task structure present BEFORE per-game control training? (a
#      representation-level instance of the warm/cold dissociation).
#   3. Re-fit the already-dumped per-game latents with episode-grouped K-fold
#      bootstrap CIs + two extra targets (reward-imminence, time-since-reset),
#      then re-score (picks up the human counterfactuals) and rebuild the summary.
#
# Prereqs already on Isambard from the full sweep: results/dynamics_probe/{holdout,
# latents}, ckpt_<game> dirs. Submit from repo root:  sbatch scripts/probe_extra.sbatch
#
#SBATCH --job-name=probe-extra
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.ahmad@exeter.ac.uk
cd "$HOME/arc3-wm"
module load cray-python/3.11.7
source .venv-probe/bin/activate
export JAX_PLATFORMS=cpu
CD="--platform cpu --compute-dtype float32"
H=results/dynamics_probe/holdout
DYN="vc33 cd82 sb26 tn36"   # dynamic games (boards actually move)
B2="https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan"
run(){ echo "+++ $*"; "$@" || echo "!!! FAILED: $*"; }

echo "=== devices ==="; python -c "import jax; print(jax.devices())" || true

# 1. Counterfactual on HUMAN source, state-changing filter (no-avail candidates).
for g in $DYN; do
  run python scripts/probe_predict.py --game "$g" --source human --mode counterfactual \
    --ckpt "ckpt_${g}" --holdout "$H/${g}_human.npz" \
    --context-len 4 --target-a 6 --n-click 8 --max-specs 1500 $CD \
    --outdir results/dynamics_probe/pred
done

# 2. Pretrained cross-game WM: dump latents (WM-only .pkl) + fit.
if [ ! -f pretrained-wm.pkl ]; then
  curl -fsSL -o pretrained-wm.pkl "${B2}/pretrained-wm/v1/latest.pkl" || echo "!! pretrained pull failed"
fi
for g in $DYN; do
  run python scripts/probe_dump_latents.py --game "$g" --source human \
    --ckpt-pkl pretrained-wm.pkl --holdout "$H/${g}_human.npz" $CD \
    --outdir results/dynamics_probe/latents_pretrained
done
run python scripts/probe_fit_probes.py --latents-dir results/dynamics_probe/latents_pretrained \
  --outdir results/dynamics_probe/scored_pretrained

# 3. Re-fit per-game latents with CIs + extra targets; re-score; rebuild summary.
run python scripts/probe_fit_probes.py --latents-dir results/dynamics_probe/latents \
  --outdir results/dynamics_probe/scored
run python scripts/probe_score.py --pred-dir results/dynamics_probe/pred \
  --outdir results/dynamics_probe/scored
run python scripts/probe_make_summary.py --scored-dir results/dynamics_probe/scored \
  --out results/dynamics_probe/scored/probe_results_summary.csv

# 4. Bundle for retrieval.
echo "=== bundling ==="
cp "probe-extra_${SLURM_JOB_ID}.log" results/dynamics_probe/extra_run.log 2>/dev/null || true
tar czf "$HOME/arc3-wm/dynamics-probe-extra.tar.gz" -C "$HOME/arc3-wm" \
  results/dynamics_probe/scored results/dynamics_probe/scored_pretrained \
  results/dynamics_probe/latents_pretrained results/dynamics_probe/pred
ls -la "$HOME/arc3-wm/dynamics-probe-extra.tar.gz"
echo "=== competence summary ==="; cat results/dynamics_probe/scored/probe_results_summary.csv 2>/dev/null
echo "=== pretrained-WM probe ==="; python -c "import json;d=json.load(open('results/dynamics_probe/scored_pretrained/probe_fits.json'));[print(k, {t:d[k]['probes'][t]['balanced_acc'] for t in ('level_id/both','transition/both') if d[k]['probes'].get(t,{}).get('status')=='ok'}) for k in d]" 2>/dev/null
echo "=== DONE ===  retrieve: scp u6ko.aip2.isambard:~/arc3-wm/dynamics-probe-extra.tar.gz ."
