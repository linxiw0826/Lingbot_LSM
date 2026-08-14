#!/usr/bin/env bash
set +e

REPO=${REPO:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}
OUTPUT=${OUTPUT:-/mnt/nas/wlx/Memory/outputs/m2ba_a1_exploratory_20260814}

AUTHORIZATION=${A1_AUTHORIZATION_FILE:-}
if [ -z "$AUTHORIZATION" ] || [ ! -s "$AUTHORIZATION" ]; then
  echo "A1_TRAINING_BLOCKED: reviewed authorization artifact is missing"
  echo "No model was loaded and no optimizer step was run."
  exit 0
fi

mkdir -p "$OUTPUT/logs"
cd "$REPO" || { echo "REPO_NOT_FOUND=$REPO"; exit 0; }
python tools/preflight/m2ba_a1/train_exploratory.py \
  --repo "$REPO" --output "$OUTPUT" --authorization "$AUTHORIZATION" \
  2>&1 | tee "$OUTPUT/logs/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "$TRAIN_EXIT" > "$OUTPUT/logs/train_process_exit_code.txt"
echo "A1_EXPLORATORY_PROCESS_EXIT=$TRAIN_EXIT"
echo "A1_EXPLORATORY_DIAGNOSTIC_COMPLETE"
