#!/usr/bin/env bash
set +e

REPO=${REPO:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}
OUTPUT=${OUTPUT:-/mnt/nas/wlx/Memory/outputs/m2ba_a1_code_preflight_20260814}
mkdir -p "$OUTPUT/logs"

cd "$REPO" || { echo "REPO_NOT_FOUND=$REPO"; exit 0; }

python -m pytest -q \
  src/tests/test_m2ba_a1_training.py \
  2>&1 | tee "$OUTPUT/logs/cpu_tests.log"
CPU_EXIT=${PIPESTATUS[0]}
echo "$CPU_EXIT" > "$OUTPUT/logs/cpu_test_exit_code.txt"
echo "CPU_TEST_EXIT=$CPU_EXIT"

python tools/preflight/m2ba_a1/probe_a1_code.py \
  --repo "$REPO" --output "$OUTPUT" \
  2>&1 | tee "$OUTPUT/logs/code_preflight.log"
PROBE_EXIT=${PIPESTATUS[0]}
echo "$PROBE_EXIT" > "$OUTPUT/logs/probe_exit_code.txt"
echo "PROBE_PROCESS_EXIT=$PROBE_EXIT"
echo "A1_CODE_PREFLIGHT_DIAGNOSTIC_COMPLETE"
