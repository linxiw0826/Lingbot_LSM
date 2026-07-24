#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source src/scripts/v7/phase1_env.sh
bash src/scripts/v7/validate_phase1_manifests.sh

IFS=',' read -r -a GPUS <<< "${PHASE1_GPUS}"
if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "[ERROR] need >=1 GPU" >&2
  exit 2
fi

JOBS_TEXT="$(python - "${PHASE1_MANIFEST_DIR}" "${PHASE1_SEEDS}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
override = [int(x) for x in sys.argv[2].split(",") if x]
expected = None
for path in sorted(root.glob("*.json")):
    if path.name.endswith(".schema.json"):
        continue
    data = json.loads(path.read_text())
    seeds = data["evaluation_seeds"]
    if expected is None:
        expected = set(seeds)
    elif set(seeds) != expected:
        raise SystemExit(
            f"inconsistent manifest evaluation_seeds: expected={sorted(expected)}, "
            f"{path}={sorted(seeds)}")
    if override and set(override) != set(seeds):
        raise SystemExit(
            f"seed override differs from {path}: manifest={seeds}, override={override}")
    for event in data["revisit_events"]:
        for seed in seeds:
            for arm in ("off", "global", "correct_local", "wrong_local"):
                print("\t".join((data["case_id"], event["event_id"], arm, str(seed))))
PY
)"
if [ -z "${JOBS_TEXT}" ]; then
  echo "[ERROR] formal manifests produced zero jobs" >&2
  exit 2
fi
mapfile -t JOBS <<< "${JOBS_TEXT}"

pids=()
declare -A PID_JOB
cleanup_workers() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_workers EXIT INT TERM
for index in "${!JOBS[@]}"; do
  IFS=$'\t' read -r case_id event_id arm seed <<< "${JOBS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  while [ "${#pids[@]}" -ge "${#GPUS[@]}" ]; do
    if ! wait "${pids[0]}"; then
      echo "[ERROR] Phase 1 worker failed: ${PID_JOB[${pids[0]}]}" >&2
      exit 1
    fi
    unset "PID_JOB[${pids[0]}]"
    pids=("${pids[@]:1}")
  done
  CASE_ID="${case_id}" EVENT_ID="${event_id}" ARM="${arm}" SEED="${seed}" GPU="${gpu}" \
    bash src/scripts/v7/run_phase1_single.sh &
  pids+=("$!")
  PID_JOB["$!"]="${case_id}/${event_id}/${arm}/seed_${seed}/gpu_${gpu}"
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] Phase 1 worker failed: ${PID_JOB[${pid}]}" >&2
    exit 1
  fi
  unset "PID_JOB[${pid}]"
done
pids=()
trap - EXIT INT TERM
python src/pipeline/v7/phase1/collect.py indexes \
  --root "${PHASE1_OUTPUT_ROOT}/phase1/${PHASE1_SHA}" \
  --output "${PHASE1_OUTPUT_ROOT}/phase1/${PHASE1_SHA}/runs_index.json"
