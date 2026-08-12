#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}"
CASES="${CASES:-/mnt/h20/135/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
CKPT="${CKPT:-/mnt/h20/135/lingbot-models/lingbot-world-base-act}"
PROBE_OUT="${PROBE_OUT:-/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PROBE_OUT/logs" "$PROBE_OUT/scripts"
cp "$SCRIPT_DIR/probe_fixture_freeze.py" "$PROBE_OUT/scripts/probe_fixture_freeze.py"
cp "$SCRIPT_DIR/run_probe.sh" "$PROBE_OUT/scripts/run_probe.sh"
python "$SCRIPT_DIR/probe_fixture_freeze.py" --repo "$REPO" --cases "$CASES" --ckpt "$CKPT" --output "$PROBE_OUT" 2>&1 | tee "$PROBE_OUT/logs/static_probe.log"
