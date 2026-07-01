#!/usr/bin/env bash
# run_revisit_metric_sanity.sh — revisit DINO 指标 headroom sanity（零生成、不加载骨干）
# 用法：bash src/scripts/v6/run_revisit_metric_sanity.sh <dataset_dir> <metadata_rel> [out_dir]
set -euo pipefail

# 被 kill 的 run 不在仓库留 pymp-* 孤儿
export TMPDIR=/tmp

DATASET_DIR="${1:?需传 dataset_dir}"
METADATA="${2:?需传 metadata 相对路径（如 metadata_verify_train.csv）}"
OUT_DIR="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  --dataset_dir "$DATASET_DIR"
  --metadata "$METADATA"
  --max_cases 20
  --min_time_gap_sec 5.0
  --hit_dist 40.0 --hit_yaw 30.0
  --device "${DEVICE:-cuda:0}"
)
if [[ -n "$OUT_DIR" ]]; then
  ARGS+=(--out "$OUT_DIR")
fi

python src/pipeline/eval/revisit_metric_sanity.py "${ARGS[@]}"
