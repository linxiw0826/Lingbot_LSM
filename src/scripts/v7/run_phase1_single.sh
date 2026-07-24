#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source src/scripts/v7/phase1_env.sh

: "${CASE_ID:?export CASE_ID}"
: "${EVENT_ID:?export EVENT_ID}"
: "${ARM:?export ARM=off|global|correct_local|wrong_local}"
GPU="${GPU:-${PHASE1_GPUS%%,*}}"
MANIFEST="${PHASE1_MANIFEST_DIR}/${CASE_ID}.json"
if [ -z "${SEED:-}" ]; then
  SEED="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_seeds"][0])' "${MANIFEST}")"
fi

python src/pipeline/v7/phase1/run.py validate --manifest "${MANIFEST}"
OUT="${PHASE1_OUTPUT_ROOT}/phase1/${PHASE1_SHA}/${ARM}/${CASE_ID}/${EVENT_ID}/seed_${SEED}"
ARGS=(
  run --manifest "${MANIFEST}" --cases_root "${CASES_ROOT}"
  --case_id "${CASE_ID}" --event_id "${EVENT_ID}" --arm "${ARM}" --seed "${SEED}"
  --output_dir "${OUT}" --ckpt_dir "${CKPT_DIR}" --device cuda:0
  --num_inference_steps "${PHASE1_STEPS}" --seam_buffer "${PHASE1_SEAM_BUFFER}"
  --commit_sha "${PHASE1_SHA}"
)
if [ -n "${LORA_PATH}" ]; then ARGS+=(--lora_path "${LORA_PATH}"); fi
CUDA_VISIBLE_DEVICES="${GPU}" python src/pipeline/v7/phase1/run.py "${ARGS[@]}"
