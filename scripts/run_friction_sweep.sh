#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-isaaclab.sh}"
DATA_ROOT="${VTT_RL_DATA:-${REPO_DIR}/data}/results_phd/friction_tests"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${DATA_ROOT}/run_${TIMESTAMP}"

# Edit these as needed
# 20 values from 0.0 to 1.0 (inclusive), evenly spaced
FRICTIONS=($(awk 'BEGIN{for(i=0;i<20;i++){printf "%.3f%s", i/19, (i<19? " " : "")}}'))
# One seed per mu value (aligned by index)
SEEDS=($(seq 0 19))
NUM_EPISODES=20
NUM_ENVS=1

KZ_CHECKPOINT="${KZ_CHECKPOINT:-}"
KZ_DZ_CHECKPOINT="${KZ_DZ_CHECKPOINT:-}"

OSC_OUT="${OUT_ROOT}/osc_pure"
KZ_OUT="${OUT_ROOT}/kz_only"
KZ_DZ_OUT="${OUT_ROOT}/kz_dz"

mkdir -p "${OSC_OUT}" "${KZ_OUT}" "${KZ_DZ_OUT}"

if [ -z "${KZ_CHECKPOINT}" ] || [ ! -f "${KZ_CHECKPOINT}" ]; then
  echo "ERROR: Set KZ_CHECKPOINT to a valid .pth file" >&2
  exit 1
fi

if [ -z "${KZ_DZ_CHECKPOINT}" ] || [ ! -f "${KZ_DZ_CHECKPOINT}" ]; then
  echo "ERROR: Set KZ_DZ_CHECKPOINT to a valid .pth file" >&2
  exit 1
fi

if [ "${#FRICTIONS[@]}" -ne "${#SEEDS[@]}" ]; then
  echo "ERROR: FRICTIONS and SEEDS must have the same length" >&2
  exit 1
fi

cat <<INFO > "${OUT_ROOT}/run_info.txt"
Friction sweep run
==================
Timestamp: ${TIMESTAMP}
FRICTIONS: ${FRICTIONS[*]}
SEEDS: ${SEEDS[*]}
NUM_EPISODES: ${NUM_EPISODES}
NUM_ENVS: ${NUM_ENVS}

KZ_CHECKPOINT: ${KZ_CHECKPOINT}
KZ_DZ_CHECKPOINT: ${KZ_DZ_CHECKPOINT}
INFO

for idx in "${!FRICTIONS[@]}"; do
  mu="${FRICTIONS[$idx]}"
  seed="${SEEDS[$idx]}"
  echo "=== Friction ${mu} | seed ${seed} ==="
  OSC_DIR="${OSC_OUT}/mu_${mu}/seed_${seed}"
  KZ_DIR="${KZ_OUT}/mu_${mu}/seed_${seed}"
  KZ_DZ_DIR="${KZ_DZ_OUT}/mu_${mu}/seed_${seed}"

  mkdir -p "${OSC_DIR}" "${KZ_DIR}" "${KZ_DZ_DIR}"

  STATIC_FRICTION="${mu}" DYNAMIC_FRICTION="${mu}" \
    "${ISAACLAB}" -p "${REPO_DIR}/scripts/run_osc_pure_episode.py" \
    --num_envs "${NUM_ENVS}" \
    --episodes "${NUM_EPISODES}" \
    --output_dir "${OSC_DIR}" \
    --seed "${seed}" \
    --headless

  STATIC_FRICTION="${mu}" DYNAMIC_FRICTION="${mu}" \
    "${ISAACLAB}" -p "${REPO_DIR}/scripts/extract_trajectories.py" \
    --checkpoint "${KZ_CHECKPOINT}" \
    --num_envs "${NUM_ENVS}" \
    --num_episodes "${NUM_EPISODES}" \
    --output "${KZ_DIR}" \
    --seed "${seed}" \
    --headless

  STATIC_FRICTION="${mu}" DYNAMIC_FRICTION="${mu}" \
    "${ISAACLAB}" -p "${REPO_DIR}/scripts/extract_trajectories.py" \
    --checkpoint "${KZ_DZ_CHECKPOINT}" \
    --num_envs "${NUM_ENVS}" \
    --num_episodes "${NUM_EPISODES}" \
    --output "${KZ_DZ_DIR}" \
    --seed "${seed}" \
    --headless

done

echo "Done. Outputs in: ${OUT_ROOT}"
