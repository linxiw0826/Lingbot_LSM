#!/usr/bin/env bash
set -euo pipefail

# One-case/one-seed Phase 1 development run. This is intentionally separate
# from the formal five-case/four-arm runner.
: "${ARM:?export ARM=off|global|correct_local}"
case "${ARM}" in
  off|global|correct_local) ;;
  *) echo "[ERROR] unsupported dev arm: ${ARM}" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}"
PILOT_ROOT="${PILOT_ROOT:-/mnt/nas/wlx/Memory/outputs/phase1_three_arm_pilot_20260803}"
DEV_ROOT="${DEV_ROOT:-/mnt/nas/wlx/Memory/outputs/phase1_arch_return_40step_20260804}"
WHEEL_DIR="${WHEEL_DIR:-/mnt/nas/wlx/wheels}"

export CASES_ROOT="${CASES_ROOT:-/mnt/h20/135/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
export CKPT_DIR="${CKPT_DIR:-/mnt/h20/135/lingbot-models/lingbot-world-base-act}"
export PHASE1_MANIFEST_DIR="${PHASE1_MANIFEST_DIR:-${PILOT_ROOT}/manifests}"
export PHASE1_GUARDRAIL_CONFIG="${PHASE1_GUARDRAIL_CONFIG:-${PILOT_ROOT}/guardrail_thresholds.dev_zero_tolerance.json}"
export STATIC_MASK_ROOT="${STATIC_MASK_ROOT:-${PILOT_ROOT}/static_masks_not_used_for_generation}"
export PHASE1_OUTPUT_ROOT="${PHASE1_OUTPUT_ROOT:-${DEV_ROOT}/runs}"
export PHASE1_GPUS="${PHASE1_GPUS:-0}"
export PHASE1_SEEDS=42,43,44
export PHASE1_STEPS=40
export PHASE1_SEAM_BUFFER="${PHASE1_SEAM_BUFFER:-8}"
export LORA_PATH="${LORA_PATH:-}"

export CASE_ID=Ep000027_p0007_26s_35s_fwd_back_two_windows
export EVENT_ID=arch_return
export SEED=42

cd "${REPO_ROOT}"
test -d .git
test -z "$(git status --short --untracked-files=no)"
export PHASE1_SHA="$(git rev-parse HEAD)"

# DLC images are not guaranteed to include OpenCV. Use only the preregistered
# wheel staged on shared NAS; never resolve or download a package at job time.
if ! python -c 'import cv2' >/dev/null 2>&1; then
  shopt -s nullglob
  opencv_wheels=("${WHEEL_DIR}"/opencv_python_headless-4.11.0.86-*.whl)
  shopt -u nullglob
  if [ "${#opencv_wheels[@]}" -ne 1 ]; then
    echo "[ERROR] cv2 is missing and expected exactly one pinned wheel under ${WHEEL_DIR}; found ${#opencv_wheels[@]}" >&2
    exit 4
  fi
  echo "Installing pinned offline OpenCV wheel: ${opencv_wheels[0]}"
  python -m pip install --no-deps "${opencv_wheels[0]}"
fi

python - <<'PY'
import cv2
import numpy
import torch

print("opencv=", cv2.__version__)
print("numpy=", numpy.__version__)
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is unavailable"
print("RUNTIME_PREFLIGHT=PASS")
PY

MANIFEST="${PHASE1_MANIFEST_DIR}/${CASE_ID}.json"
OUT="${PHASE1_OUTPUT_ROOT}/phase1/${PHASE1_SHA}/${ARM}/${CASE_ID}/${EVENT_ID}/seed_${SEED}"
LOG_DIR="${DEV_ROOT}/logs"
LOG="${LOG_DIR}/${CASE_ID}_${EVENT_ID}_seed${SEED}_${ARM}_steps${PHASE1_STEPS}_${PHASE1_SHA}.log"
mkdir -p "${LOG_DIR}" "${DEV_ROOT}/audit" "${DEV_ROOT}/review"

echo "DEV_ONLY ONE_CASE ONE_SEED THREE_ARMS NO_STATIC_MASK NO_WRONG_LOCAL NOT_PHASE1_GO NOT_PAPER_EVIDENCE"
echo "ARM=${ARM}"
echo "PHASE1_SHA=${PHASE1_SHA}"
echo "EXPECTED_OUT=${OUT}"
echo "LOG=${LOG}"

test -d "${CASES_ROOT}/${CASE_ID}"
test -d "${CKPT_DIR}"
test -f "${MANIFEST}"
test -f "${PHASE1_GUARDRAIL_CONFIG}"
test -d "${STATIC_MASK_ROOT}"

python src/pipeline/v7/phase1/run.py validate --manifest "${MANIFEST}"
nvidia-smi

# Refuse to overwrite a completed run. Remove/move it explicitly if a rerun is intended.
if test -s "${OUT}/long_video.mp4" || test -s "${OUT}/provenance.json"; then
  echo "[ERROR] output already exists: ${OUT}" >&2
  exit 3
fi

set +e
bash src/scripts/v7/run_phase1_single.sh 2>&1 | tee "${LOG}"
run_status=${PIPESTATUS[0]}
set -e
echo "RUN_EXIT_CODE=${run_status}"
test "${run_status}" -eq 0

test -s "${OUT}/long_video.mp4"
test -s "${OUT}/provenance.json"
test -s "${OUT}/run_index_entry.json"

EXPECTED_OUT="${OUT}" python - <<'PY'
import json
import os
from pathlib import Path

import cv2

root = Path(os.environ["EXPECTED_OUT"])
provenance = json.loads((root / "provenance.json").read_text())
cap = cv2.VideoCapture(str(root / "long_video.mp4"))
assert cap.isOpened()
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

assert frames == 405, frames
assert abs(fps - 16.0) < 0.01, fps
assert (width, height) == (832, 464), (width, height)
assert provenance["actual_output_frames"] == 405
assert provenance["arm"] == os.environ["ARM"]
assert provenance["case_id"] == os.environ["CASE_ID"]
assert provenance["event_id"] == os.environ["EVENT_ID"]
assert provenance["seed"] == int(os.environ["SEED"])
assert provenance["commit_sha"] == os.environ["PHASE1_SHA"]
if os.environ["ARM"] in {"global", "correct_local"}:
    assert provenance["cumulative_anchor_frame_uses"] > 0

print("frames=", frames)
print("fps=", fps)
print("resolution=", f"{width}x{height}")
print("anchor_uses=", provenance["cumulative_anchor_frame_uses"])
print("ARCH_RETURN_40STEP_ARM=PASS")
PY
