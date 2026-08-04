#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
export PHASE1_DEV_INPUT_ROOT="${PHASE1_DEV_INPUT_ROOT:-/mnt/nas/wlx/Memory/outputs/phase1_p0001_lookback_negative_control_40step_20260804}"
export CASES_ROOT="${CASES_ROOT:-/mnt/nas/yukki/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
export PHASE1_MANIFEST_DIR="${PHASE1_MANIFEST_DIR:-${SCRIPT_DIR}/manifests}"
export CASE_ID=Ep000027_p0001_75s_87s_lookback_path
export EVENT_ID=lookback_non_strict_probe
export SEED=42
export PHASE1_SHA="${PHASE1_SHA:-64209fe2cca11820863d883cdc851777ea9568b6}"
export DEV_EVAL_ROOT="${DEV_EVAL_ROOT:-${PHASE1_DEV_INPUT_ROOT}/dev_eval}"
export DINO_DEVICE="${DINO_DEVICE:-cpu}"

echo "DEV_NON_STRICT_NEGATIVE_CONTROL=1"
echo "CORRECT_LOCAL_LABEL_IS_MECHANICAL_NOT_SEMANTIC=1"
echo "NOT_A_THIRD_POSITIVE_CASE=1"
echo "DEV_EVAL_ROOT=${DEV_EVAL_ROOT}"

exec bash "${REPO_ROOT}/src/scripts/v7/run_phase1_dev_eval.sh"
