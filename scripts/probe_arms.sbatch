#!/bin/bash
# Robustness / warm-vs-cold REPRESENTATION probe (CPU, Isambard workq).
# The headline latent result used warm seed-0 only. Here we dump latents from the
# COLD (from-scratch) seed-0 and WARM seed-1 checkpoints for the dynamic games and
# fit level_id/transition, to test:
#   (a) the competence dissociation is robust across seed + warm/cold arm;
#   (b) whether warm-starting raises latent DECODABILITY even though it did not
#       raise RHAE (a representation-level instance of the warm/cold null).
# Compare the three scored dirs: scored/ (warm-s0, already have), scored_cold_s0/,
# scored_warm_s1/.
#
# Submit from repo root:  sbatch scripts/probe_arms.sbatch
#
#SBATCH --job-name=probe-arms
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
DYN="vc33 cd82 sb26 tn36"
B2="https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan"
run(){ echo "+++ $*"; "$@" || echo "!!! FAILED: $*"; }

# arm -> (B2 run-dir prefix). cold seed0 = phase4-fromscratch; warm seed1 = phase4-proper.
declare -A RUNDIR=(
  [cold_s0]="phase4-fromscratch/p4-fromscratch-GAME-s0-a06c02f"
  [warm_s1]="phase4-proper/p4-GAME-s1-warm-98de390"
)

for arm in cold_s0 warm_s1; do
  for g in $DYN; do
    ck="ckpt_${g}_${arm}"
    if [ ! -d "$ck" ]; then
      url="${B2}/${RUNDIR[$arm]/GAME/$g}/ckpt-final.tar.gz"
      echo "=== fetch $ck from $url ==="
      curl -fsSL -o "${ck}.tar.gz" "$url" && mkdir -p "$ck" && tar xzf "${ck}.tar.gz" -C "$ck" \
        || { echo "!! fetch failed $ck"; continue; }
    fi
    run python scripts/probe_dump_latents.py --game "$g" --source human \
      --ckpt "$ck" --holdout "$H/${g}_human.npz" $CD \
      --outdir "results/dynamics_probe/latents_${arm}"
  done
  run python scripts/probe_fit_probes.py --latents-dir "results/dynamics_probe/latents_${arm}" \
    --outdir "results/dynamics_probe/scored_${arm}"
done

echo "=== bundling ==="
cp "probe-arms_${SLURM_JOB_ID}.log" results/dynamics_probe/arms_run.log 2>/dev/null || true
tar czf "$HOME/arc3-wm/dynamics-probe-arms.tar.gz" -C "$HOME/arc3-wm" \
  results/dynamics_probe/scored_cold_s0 results/dynamics_probe/scored_warm_s1 \
  results/dynamics_probe/latents_cold_s0 results/dynamics_probe/latents_warm_s1
ls -la "$HOME/arc3-wm/dynamics-probe-arms.tar.gz"
echo "=== warm-s0 vs cold-s0 vs warm-s1: level_id/both bacc ==="
python - <<'PY' 2>/dev/null
import json,glob,os
for arm,sd in [("warm_s0","scored"),("cold_s0","scored_cold_s0"),("warm_s1","scored_warm_s1")]:
    p=f"results/dynamics_probe/{sd}/probe_fits.json"
    if not os.path.exists(p): continue
    d=json.load(open(p))
    row={k.split('/')[0]: d[k]['probes'].get('level_id/both',{}).get('balanced_acc') for k in d}
    print(arm, row)
PY
echo "=== DONE ===  retrieve: scp u6ko.aip2.isambard:~/arc3-wm/dynamics-probe-arms.tar.gz ."
