#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
MANIFEST_DIR="${PHASE1_MANIFEST_DIR:-src/pipeline/v7/phase1/manifests}"
mapfile -t MANIFESTS < <(find "${MANIFEST_DIR}" -maxdepth 1 -name '*.json' ! -name '*.schema.json' | sort)
if [ "${#MANIFESTS[@]}" -ne 5 ]; then
  echo "[ERROR] expected exactly 5 case manifests, found ${#MANIFESTS[@]}" >&2
  exit 2
fi
EXPECTED=(
  Ep000027_p0001_75s_87s_lookback_path
  Ep000027_p0001_head20s_lookaround
  Ep000027_p0006_93s_105s_boxes_lookback
  Ep000027_p0007_26s_35s_fwd_back_two_windows
  Ep000027_p0007_77s_86s_two_windows_revisit
)
for index in "${!EXPECTED[@]}"; do
  if [ "$(basename "${MANIFESTS[$index]}" .json)" != "${EXPECTED[$index]}" ]; then
    echo "[ERROR] manifest set does not match frozen five cases" >&2
    exit 2
  fi
done
ARGS=()
for manifest in "${MANIFESTS[@]}"; do ARGS+=(--manifest "${manifest}"); done
python src/pipeline/v7/phase1/run.py validate "${ARGS[@]}"
echo "[OK] 5/5 manifests are formal-run ready and human approved."
