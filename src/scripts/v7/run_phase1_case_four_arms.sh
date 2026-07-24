#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source src/scripts/v7/phase1_env.sh
: "${CASE_ID:?export CASE_ID}"
: "${EVENT_ID:?export EVENT_ID}"
MANIFEST="${PHASE1_MANIFEST_DIR}/${CASE_ID}.json"
SEED="${SEED:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_seeds"][0])' "${MANIFEST}")}"
GPU="${GPU:-${PHASE1_GPUS%%,*}}"
for arm in off global correct_local wrong_local; do
  CASE_ID="${CASE_ID}" EVENT_ID="${EVENT_ID}" ARM="${arm}" SEED="${SEED}" GPU="${GPU}" \
    bash src/scripts/v7/run_phase1_single.sh
done
