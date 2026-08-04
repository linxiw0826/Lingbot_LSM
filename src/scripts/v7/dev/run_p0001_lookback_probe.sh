#!/usr/bin/env bash
set -euo pipefail

: "${ARM:?export ARM=off|global|correct_local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CASES_ROOT="${CASES_ROOT:-/mnt/nas/yukki/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
export PHASE1_MANIFEST_DIR="${PHASE1_MANIFEST_DIR:-${SCRIPT_DIR}/manifests}"
export DEV_ROOT="${DEV_ROOT:-/mnt/nas/wlx/Memory/outputs/phase1_p0001_lookback_negative_control_40step_20260804}"
export PHASE1_OUTPUT_ROOT="${PHASE1_OUTPUT_ROOT:-${DEV_ROOT}/runs}"
export CASE_ID=Ep000027_p0001_75s_87s_lookback_path
export EVENT_ID=lookback_non_strict_probe
export SEED=42

echo "DEV_NON_STRICT_NEGATIVE_CONTROL=1"
echo "CORRECT_LOCAL_LABEL_IS_MECHANICAL_NOT_SEMANTIC=1"
exec bash "${SCRIPT_DIR}/run_arch_return_40step.sh"
