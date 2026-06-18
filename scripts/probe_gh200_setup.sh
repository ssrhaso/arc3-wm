#!/usr/bin/env bash
# GH200 setup for the dynamics-competence probe Stage 2 (frozen-WM forward).
#
# Run from the repo root on the GH200 AFTER cloning the repo. Stages 1 (collect)
# and 3 (score) are CPU-only and already done on the laptop; this only sets up
# the GPU forward pass. See docs/dynamics-competence-probe.md.
#
#   bash scripts/probe_gh200_setup.sh [GAME] [SEED]    # default: cd82 s0
#
# IMPORTANT: the GH200 is aarch64 + Hopper, NOT the x86 + A100 of the Vast boxes
# used in Phases 2-4. JAX GPU wheels differ on ARM; the install step below is the
# most likely thing to need adjusting. Verify `python -c "import jax;
# print(jax.devices())"` shows the Hopper GPU before proceeding.

set -euo pipefail
BUCKET="b2://arc-agi-3-replays-hasaan"
GAME="${1:-cd82}"
SEED="${2:-s0}"
RUN="p4-${GAME}-${SEED}-warm-98de390"

echo "== [1/4] JAX + dreamerv3 stack (adjust for aarch64/Hopper if needed) =="
pip install -U -r third_party/dreamerv3/requirements.txt
# ARM + CUDA12: the exact jax wheel may need NVIDIA's index or a local build.
# Start with the standard cuda12 extra; if it fails on aarch64, see notes above.
pip install -U "jax[cuda12]" || {
  echo "!! jax[cuda12] install failed on this box (likely aarch64 wheel gap)."
  echo "   Try: pip install -U jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
  echo "   or NVIDIA's JAX container. Halting so you can resolve JAX first."; exit 1; }
pip install "wandb[media]" moviepy   # parity with training box (avoids a media-dep crash)

echo "== [2/4] probe code + holdout data bundle from B2 (overlay onto repo) =="
b2 file download "${BUCKET}/analysis/dynamics-probe-bundle.tar.gz" bundle.tar.gz
tar xzf bundle.tar.gz   # -> arc3_wm/probe_*, scripts/probe_*, results/dynamics_probe/holdout/*

echo "== [3/4] checkpoint for ${RUN} =="
b2 file download "${BUCKET}/phase4-proper/${RUN}/ckpt-final.tar.gz" "ckpt_${GAME}.tar.gz"
mkdir -p "ckpt_${GAME}" && tar xzf "ckpt_${GAME}.tar.gz" -C "ckpt_${GAME}"
echo "   ckpt dir contents:"; ls "ckpt_${GAME}"   # expect: latest  <TS>/

echo "== [4/4] shape/JIT self-test (no ckpt restore) =="
python scripts/probe_predict.py --game "${GAME}" --self-test --context-len 4 --horizon 8

cat <<EOF

Setup done. Next (the real forward pass):

  python scripts/probe_predict.py --game ${GAME} --source human \\
      --ckpt ckpt_${GAME} \\
      --holdout results/dynamics_probe/holdout/${GAME}_human.npz \\
      --context-len 4 --horizon 8 --outdir results/dynamics_probe/pred

Then score (can also run back on the laptop):

  python scripts/probe_score.py --pred-dir results/dynamics_probe/pred \\
      --outdir results/dynamics_probe/scored
EOF
