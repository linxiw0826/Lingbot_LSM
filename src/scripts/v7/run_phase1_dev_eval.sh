#!/usr/bin/env bash
set -euo pipefail

: "${PHASE1_DEV_INPUT_ROOT:?export PHASE1_DEV_INPUT_ROOT}"
: "${CASES_ROOT:?export CASES_ROOT}"
: "${PHASE1_MANIFEST_DIR:?export PHASE1_MANIFEST_DIR}"
: "${CASE_ID:?export CASE_ID}"
: "${EVENT_ID:?export EVENT_ID}"
: "${PHASE1_SHA:?export PHASE1_SHA}"

SEED="${SEED:-42}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DEV_EVAL_ROOT="${DEV_EVAL_ROOT:-${PHASE1_DEV_INPUT_ROOT}/dev_eval}"
DINO_DEVICE="${DINO_DEVICE:-cpu}"

cd "${REPO_ROOT}"
EVALUATOR_SHA="$(git rev-parse HEAD)"
test -z "$(git status --short --untracked-files=no)"

PYTHONPATH=src python src/pipeline/v7/phase1/dev_eval.py \
  --input-root "${PHASE1_DEV_INPUT_ROOT}" \
  --repo-root "${REPO_ROOT}" \
  --cases-root "${CASES_ROOT}" \
  --manifest "${PHASE1_MANIFEST_DIR}/${CASE_ID}.json" \
  --commit-sha "${PHASE1_SHA}" \
  --evaluator-commit-sha "${EVALUATOR_SHA}" \
  --case-id "${CASE_ID}" \
  --event-id "${EVENT_ID}" \
  --seed "${SEED}" \
  --output-dir "${DEV_EVAL_ROOT}" \
  --dino-device "${DINO_DEVICE}"
