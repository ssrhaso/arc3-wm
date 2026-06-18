#!/bin/bash
# Full-sweep Stage-2 batch: pull every per-game checkpoint from the (public) B2
# bucket and run the frozen-WM rollout (Probe B) for all six games' HUMAN source
# plus the four games whose RANDOM holdout we staged. One job, one queue wait.
#
# Run only AFTER probe_cd82.sbatch validates the pipeline. Submit from repo root:
#   cd ~/arc3-wm && sbatch scripts/probe_all.sbatch
#
#SBATCH --job-name=probe-all
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
set -e
cd "$HOME/arc3-wm"
module load cray-python/3.11.7
source .venv-probe/bin/activate

B2="https://f003.backblazeb2.com/file/arc-agi-3-replays-hasaan/phase4-proper"
GAMES="vc33 sb26 cd82 tn36 ls20 lf52"
RANDOM_GAMES="vc33 cd82 sb26 lf52"   # the four whose random holdout we staged

echo "=== devices ==="; python -c "import jax; print(jax.devices())"

# Pull each checkpoint (warm seed 0) if not already present.
for g in $GAMES; do
  if [ ! -d "ckpt_${g}" ]; then
    echo "=== fetch ckpt ${g} ==="
    curl -fsSL -o "ckpt_${g}.tar.gz" "${B2}/p4-${g}-s0-warm-98de390/ckpt-final.tar.gz"
    mkdir -p "ckpt_${g}" && tar xzf "ckpt_${g}.tar.gz" -C "ckpt_${g}"
  fi
done

# Human source: all six games (the strong-signal probe set).
for g in $GAMES; do
  echo "=== predict ${g} human ==="
  python scripts/probe_predict.py --game "$g" --source human --ckpt "ckpt_${g}" \
    --holdout "results/dynamics_probe/holdout/${g}_human.npz" \
    --context-len 4 --horizon 8 --outdir results/dynamics_probe/pred
done

# Random source: the four with staged holdout (genuinely held-out samples).
for g in $RANDOM_GAMES; do
  echo "=== predict ${g} random ==="
  python scripts/probe_predict.py --game "$g" --source random --ckpt "ckpt_${g}" \
    --holdout "results/dynamics_probe/holdout/${g}_random.npz" \
    --context-len 4 --horizon 8 --outdir results/dynamics_probe/pred
done

echo "=== score ==="
python scripts/probe_score.py --pred-dir results/dynamics_probe/pred \
  --outdir results/dynamics_probe/scored
echo "=== competence table ==="; cat results/dynamics_probe/scored/competence_table.csv
echo "=== DONE ==="
