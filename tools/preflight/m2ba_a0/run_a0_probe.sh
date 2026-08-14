#!/usr/bin/env bash

# Intentionally nonfatal: diagnostics are authoritative in JSON and failures
# remain visible without terminating the user's terminal task.
set +e

REPO="${M2BA_REPO:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}"
OUTPUT="${M2BA_A0_OUTPUT:-/mnt/nas/wlx/Memory/outputs/m2ba_a0_20260814}"
mkdir -p "$OUTPUT/logs"

echo "=== A0 CPU TESTS ==="
PYTHONPATH="$REPO/src" python -m pytest -q \
  "$REPO/src/tests/test_m2ba_causal_memory_adapter.py" \
  2>&1 | tee "$OUTPUT/logs/cpu_tests.log"
CPU_EXIT=${PIPESTATUS[0]}
echo "CPU_TEST_EXIT=$CPU_EXIT"

echo "=== A0 GPU PARITY/STATE PROBE ==="
PYTHONPATH="$REPO/src:$REPO" python \
  "$REPO/tools/preflight/m2ba_a0/probe_a0.py" \
  --repo "$REPO" --output "$OUTPUT" --cpu-test-exit "$CPU_EXIT" \
  2>&1 | tee "$OUTPUT/logs/gpu_probe.log"
GPU_EXIT=${PIPESTATUS[0]}
echo "GPU_PROBE_PROCESS_EXIT=$GPU_EXIT"

echo "=== AUTHORITATIVE A0 REPORT ==="
python -m json.tool "$OUTPUT/a0_parity_report.json"
REPORT_EXIT=$?
echo "REPORT_READ_EXIT=$REPORT_EXIT"
echo "M2BA_A0_DIAGNOSTIC_COMPLETE"
exit 0
