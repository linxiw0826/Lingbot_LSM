#!/usr/bin/env bash
# Source this file after exporting server-specific paths. No private path is
# committed. All Phase 1 scripts use this single environment contract.

set -euo pipefail

: "${CASES_ROOT:?export CASES_ROOT (five action-case directories)}"
: "${CKPT_DIR:?export CKPT_DIR (LingBot-World base checkpoint)}"
: "${PHASE1_OUTPUT_ROOT:?export PHASE1_OUTPUT_ROOT}"
: "${STATIC_MASK_ROOT:?export STATIC_MASK_ROOT (case_id/static_mask.npy + mask_provenance.json)}"
: "${PHASE1_GUARDRAIL_CONFIG:?export PHASE1_GUARDRAIL_CONFIG (reviewed before generation; threshold-only JSON)}"

LORA_PATH="${LORA_PATH:-}"
# Optional assertion only. Seeds are authoritative in each event manifest.
# If set, this comma-separated set must equal every manifest evaluation_seeds.
PHASE1_SEEDS="${PHASE1_SEEDS:-}"
PHASE1_STEPS="${PHASE1_STEPS:-40}"
PHASE1_SEAM_BUFFER="${PHASE1_SEAM_BUFFER:-8}"
PHASE1_GPUS="${PHASE1_GPUS:-0}"
PHASE1_MANIFEST_DIR="${PHASE1_MANIFEST_DIR:-src/pipeline/v7/phase1/manifests}"
PHASE1_SHA="${PHASE1_SHA:-$(git rev-parse HEAD)}"

export CASES_ROOT CKPT_DIR PHASE1_OUTPUT_ROOT STATIC_MASK_ROOT LORA_PATH
export PHASE1_SEEDS PHASE1_STEPS PHASE1_SEAM_BUFFER PHASE1_GPUS
export PHASE1_MANIFEST_DIR PHASE1_SHA
export PHASE1_GUARDRAIL_CONFIG
