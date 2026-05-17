#!/usr/bin/env bash
# Phase 4 EXPANSION from-scratch (cold-arm) launch harness — 3 ADDITIONAL games
# x 2 seeds x 500k env-steps, stock DV3, NO warm-start. Byte-faithful copy of
# scripts/launch_phase4_fromscratch.sh; the ONLY differences are:
#   - GAMES is the expansion trio (tn36 ls20 lf52), not the pilot
#   - SHA7 is PINNED to "a06c02f" (not git-HEAD-derived) so the new run names +
#     B2 prefix exactly match the existing -fromscratch-...-a06c02f artifacts.
#     Deliberate: keeps the expansion in the same cold-arm experiment
#     generation so analysis/build_p4_combined.py aggregates all 6 games with
#     no code change. Comparability requires identical config + suffix.
#
# Run from the repo root on the Vast A100 box. Substrate smoke must pass first.
#
# Usage:
#   bash scripts/launch_phase4_expansion_fromscratch.sh 2>&1 | tee /workspace/logdir/p4-expansion-fromscratch-harness.log
#
# Soft-fail policy: RHAE=0 / no levels cleared is a valid scientific outcome and
# does NOT stop the harness. Hard-fail (non-zero exit, NaN, OOM, CUDA error,
# Arcade crash) DOES stop the harness — see CLAUDE.md Risks-4.

set -eo pipefail

SHA7="a06c02f"   # PINNED to match existing -fromscratch-...-a06c02f paired artifacts
BUCKET="arc-agi-3-replays-hasaan"
BASELINES="data/human_baselines.json"
GAMES=(tn36 ls20 lf52)
SEEDS=(0 1)
STEPS=500000
WALLCLOCK_CEILING_SEC=$((18 * 3600))   # 18h hard ceiling, same as proper

export WANDB_PROJECT="arc3-wm-sprint"
export WANDB_ENTITY="hasofocus-university-of-the-west-of-england"
export WANDB_NOTES="Phase 4 expansion from-scratch (cold, no warm-start), stock DV3, sha=${SHA7}"

start_all=$(date +%s)
SUMMARIES=()

run_one() {
  local game="$1" seed="$2"
  local run_name="p4-fromscratch-${game}-s${seed}-${SHA7}"
  local logdir="/workspace/logdir/${run_name}"
  local b2_prefix="phase4-fromscratch/${run_name}"
  local stdout_log="${logdir}.log"

  # Resume: skip if ckpt-final.tar.gz already exists from a previous attempt.
  if [[ -f "${logdir}/ckpt-final.tar.gz" ]]; then
    echo "[$(date -Iseconds)] SKIP: ${run_name} already complete (ckpt-final.tar.gz present)"
    return 0
  fi

  export WANDB_NAME="${run_name}"
  export WANDB_RUN_GROUP="p4-${game}-cold"
  export WANDB_TAGS="phase-4,arm:cold,game:${game},seed:${seed},from-scratch,substrate:a100,expansion"

  echo "==========================================="
  echo "[$(date -Iseconds)] Starting ${run_name}"
  echo "  logdir    = ${logdir}"
  echo "  b2_prefix = ${b2_prefix}"
  echo "==========================================="

  local start_run end_run wallclock wallclock_min exit_code
  start_run=$(date +%s)

  if (( start_run - start_all > WALLCLOCK_CEILING_SEC )); then
    echo "FATAL: cumulative wallclock exceeded ${WALLCLOCK_CEILING_SEC}s before ${run_name}. Halting."
    exit 2
  fi

  set +e
  python scripts/launch_pergame.py \
    --logdir "${logdir}" \
    --configs size12m arc3 \
    --task "arc3_${game}" \
    --seed "${seed}" \
    --script train_eval \
    --run.steps "${STEPS}" \
    --run.log_every 30 \
    --run.save_every 600 \
    2>&1 | tee "${stdout_log}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  end_run=$(date +%s)
  wallclock=$((end_run - start_run))
  wallclock_min=$((wallclock / 60))

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "FATAL: ${run_name} exited ${exit_code}. Stopping harness."
    exit "${exit_code}"
  fi
  # Surgical fail detection: startup `replay_ratio nan` and `fps/train 0`
  # are benign DV3 artifacts before training updates fire. Flag only:
  #   (a) train losses going NaN in any logged window
  #   (b) explicit error tokens (Traceback, OOM, CUDA error, Arcade crash)
  if grep -qE "train/loss/[A-Za-z]+ nan" "${stdout_log}"; then
    echo "FATAL: ${run_name} train loss went NaN. Stopping harness."
    grep -nE "train/loss/[A-Za-z]+ nan" "${stdout_log}" | head -10
    exit 1
  fi
  if grep -qiE "(Traceback|OOM|out of memory|CUDA error|Arcade crash)" "${stdout_log}"; then
    echo "FATAL: ${run_name} stdout shows Traceback/OOM/CUDA-error/Arcade-crash. Stopping harness."
    grep -niE "(Traceback|OOM|out of memory|CUDA error|Arcade crash)" "${stdout_log}" | head -10
    exit 1
  fi

  # DV3 writes ckpt as a directory + 22-byte 'latest' pointer file; tar both.
  local ckpt_dir
  ckpt_dir=$(ls -1 "${logdir}/ckpt/" | grep -v '^latest$' | head -1 || true)
  if [[ -z "${ckpt_dir}" ]]; then
    echo "FATAL: no checkpoint directory in ${logdir}/ckpt/. Stopping harness."
    exit 1
  fi
  tar czf "${logdir}/ckpt-final.tar.gz" -C "${logdir}/ckpt" "${ckpt_dir}" latest

  b2 file upload "${BUCKET}" "${logdir}/ckpt-final.tar.gz"      "${b2_prefix}/ckpt-final.tar.gz"
  b2 file upload "${BUCKET}" "${stdout_log}"                    "${b2_prefix}/launch.log"
  b2 file upload "${BUCKET}" "${logdir}/metrics.jsonl"          "${b2_prefix}/metrics.jsonl"
  b2 file upload "${BUCKET}" "${logdir}/eval_episodes.jsonl"    "${b2_prefix}/eval_episodes.jsonl"

  local rhae_line
  rhae_line=$(python scripts/compute_rhae.py \
    --episodes-file "${logdir}/eval_episodes.jsonl" \
    --game-id "${game}" \
    --baselines "${BASELINES}" \
    --step "${STEPS}" 2>&1 | tail -1) || rhae_line="(compute_rhae failed)"

  local cum_wallclock cum_min summary
  cum_wallclock=$(( $(date +%s) - start_all ))
  cum_min=$((cum_wallclock / 60))
  summary="${run_name}: ${rhae_line} | wallclock=${wallclock_min}min cum=${cum_min}min"
  echo "[$(date -Iseconds)] ONE-LINER: ${summary}"
  SUMMARIES+=("${summary}")
}

for tool in python b2 tar git; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "FATAL: ${tool} not on PATH"; exit 1; }
done
[[ -f "${BASELINES}" ]] || { echo "FATAL: ${BASELINES} missing"; exit 1; }
[[ -f scripts/launch_pergame.py ]] || { echo "FATAL: scripts/launch_pergame.py missing — wrong cwd?"; exit 1; }

for game in "${GAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${game}" "${seed}"
  done
done

echo ""
echo "============= PHASE 4 EXPANSION FROM-SCRATCH COMPLETE ============="
echo "Cumulative wallclock: $(( ($(date +%s) - start_all) / 60 )) min"
echo ""
for s in "${SUMMARIES[@]}"; do
  echo "  ${s}"
done
echo "============= END ==============================================="
