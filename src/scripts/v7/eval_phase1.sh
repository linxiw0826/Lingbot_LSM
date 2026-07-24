#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source src/scripts/v7/phase1_env.sh

RUN_ROOT="${PHASE1_OUTPUT_ROOT}/phase1/${PHASE1_SHA}"
RUN_INDEX="${RUN_ROOT}/runs_index.json"
if [ ! -f "${RUN_INDEX}" ]; then
  python src/pipeline/v7/phase1/collect.py indexes --root "${RUN_ROOT}" --output "${RUN_INDEX}"
fi

CSV_ARGS=()
for manifest in "${PHASE1_MANIFEST_DIR}"/*.json; do
  case "${manifest}" in *.schema.json) continue ;; esac
  case_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["case_id"])' "${manifest}")"
  case_index="${RUN_ROOT}/runs_${case_id}.json"
  python - "${RUN_INDEX}" "${case_id}" "${case_index}" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
selected = [r for r in rows if r["case_id"] == sys.argv[2]]
if not selected:
    raise SystemExit("no run entries for " + sys.argv[2])
json.dump(selected, open(sys.argv[3], "w"), indent=2)
PY
  out_csv="${RUN_ROOT}/per_frame_${case_id}.csv"
  if [ -n "${RAFT_SCORES_ROOT:-}" ]; then
    raft_csv="${RAFT_SCORES_ROOT}/${case_id}.csv"
    if [ ! -f "${raft_csv}" ]; then
      echo "[ERROR] external RAFT evidence missing: ${raft_csv}" >&2
      exit 2
    fi
  else
    raft_csv="${RUN_ROOT}/raft_${case_id}.csv"
    RAFT_ARGS=(
      --manifest "${manifest}"
      --gt_full "${CASES_ROOT}/${case_id}/ground_truth_full.mp4"
      --static_mask "${STATIC_MASK_ROOT}/${case_id}/static_mask.npy"
      --mask_provenance "${STATIC_MASK_ROOT}/${case_id}/mask_provenance.json"
      --runs_index "${case_index}" --output_csv "${raft_csv}" --device cuda:0
    )
    if [ -n "${RAFT_WEIGHTS_PATH:-}" ]; then
      RAFT_ARGS+=(--weights_path "${RAFT_WEIGHTS_PATH}")
    fi
    CUDA_VISIBLE_DEVICES="${PHASE1_GPUS%%,*}" \
      python src/pipeline/v7/phase1/raft_cli.py "${RAFT_ARGS[@]}"
  fi
  CUDA_VISIBLE_DEVICES="${PHASE1_GPUS%%,*}" python src/pipeline/v7/phase1/eval_cli.py score \
    --manifest "${manifest}" \
    --gt_full "${CASES_ROOT}/${case_id}/ground_truth_full.mp4" \
    --static_mask "${STATIC_MASK_ROOT}/${case_id}/static_mask.npy" \
    --mask_provenance "${STATIC_MASK_ROOT}/${case_id}/mask_provenance.json" \
    --runs_index "${case_index}" --raft_scores_csv "${raft_csv}" \
    --output_csv "${out_csv}" --device cuda:0
  CSV_ARGS+=(--input "${out_csv}")
done
python src/pipeline/v7/phase1/collect.py csv "${CSV_ARGS[@]}" \
  --output "${RUN_ROOT}/per_frame_all.csv"
GUARDRAIL_ARGS=(--guardrail_config "${PHASE1_GUARDRAIL_CONFIG}")
if [ -n "${PHASE1_GUARDRAIL_EVIDENCE_ROOT:-}" ]; then
  for name in seam action_following copy_leakage; do
    evidence="${PHASE1_GUARDRAIL_EVIDENCE_ROOT}/${name}.csv"
    if [ ! -f "${evidence}" ]; then
      echo "[ERROR] external guardrail evidence missing: ${evidence}" >&2
      exit 2
    fi
    GUARDRAIL_ARGS+=(--guardrail_evidence "${name}=${evidence}")
  done
fi
SEED_ARGS=()
if [ -n "${PHASE1_SEEDS}" ]; then SEED_ARGS+=(--seeds "${PHASE1_SEEDS}"); fi
python src/pipeline/v7/phase1/eval_cli.py aggregate \
  --per_frame_csv "${RUN_ROOT}/per_frame_all.csv" \
  --output_json "${RUN_ROOT}/aggregate.json" \
  --output_csv "${RUN_ROOT}/aggregate.csv" \
  --runs_index "${RUN_INDEX}" --seam_buffer "${PHASE1_SEAM_BUFFER}" \
  --manifest_dir "${PHASE1_MANIFEST_DIR}" --min_seeds 3 \
  "${SEED_ARGS[@]}" \
  "${GUARDRAIL_ARGS[@]}"
echo "[OK] Phase 1 aggregate: ${RUN_ROOT}/aggregate.json + aggregate.csv"
