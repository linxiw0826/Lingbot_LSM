#!/usr/bin/env bash
# This is a diagnostic runner: failures are reported in output/logs, while the
# shell entry point deliberately returns zero so IDE terminals stay open.
set -uo pipefail
finalize_nonfatal() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    echo "RUN_PROBE_WRAPPER_ERROR=$rc" >&2
  fi
  exit 0
}
trap finalize_nonfatal EXIT
REPO="${REPO:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}"
CASES="${CASES:-/mnt/h20/135/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
CKPT="${CKPT:-/mnt/h20/135/lingbot-models/lingbot-world-base-act}"
PROBE_OUT="${PROBE_OUT:-/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROBE_OUT/logs" "$PROBE_OUT/scripts"
mkdir_rc=$?
if [ "$mkdir_rc" -ne 0 ]; then echo "PROBE_SETUP_MKDIR_ERROR=$mkdir_rc" >&2; fi

cp "$SCRIPT_DIR/probe_fixture_freeze.py" "$PROBE_OUT/scripts/probe_fixture_freeze.py"
copy_probe_rc=$?
if [ "$copy_probe_rc" -ne 0 ]; then echo "PROBE_COPY_PYTHON_ERROR=$copy_probe_rc" >&2; fi
cp "$SCRIPT_DIR/run_probe.sh" "$PROBE_OUT/scripts/run_probe.sh"
copy_runner_rc=$?
if [ "$copy_runner_rc" -ne 0 ]; then echo "PROBE_COPY_RUNNER_ERROR=$copy_runner_rc" >&2; fi

set +e
python "$SCRIPT_DIR/probe_fixture_freeze.py" --repo "$REPO" --cases "$CASES" --ckpt "$CKPT" --output "$PROBE_OUT" 2>&1 | tee "$PROBE_OUT/logs/static_probe.log"
pipeline_status=("${PIPESTATUS[@]}")
probe_python_exit="${pipeline_status[0]:-127}"
probe_tee_exit="${pipeline_status[1]:-127}"
echo "PROBE_PYTHON_EXIT=$probe_python_exit"
if [ "$probe_tee_exit" -ne 0 ]; then echo "PROBE_TEE_EXIT=$probe_tee_exit" >&2; fi
printf '%s\n' "$probe_python_exit" > "$PROBE_OUT/logs/probe_exit_code.txt"
write_exit_rc=$?
if [ "$write_exit_rc" -ne 0 ]; then echo "PROBE_EXIT_CODE_LOG_WRITE_ERROR=$write_exit_rc" >&2; fi

exit 0
