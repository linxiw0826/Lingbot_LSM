#!/usr/bin/env bash
# ============================================================================
# auto_eval_after_train.sh — 训练结束后自动跑 5-case 推理 + 定量 eval（无人值守）
# ----------------------------------------------------------------------------
# 用途：挂 tmux 后睡觉。脚本轮询等待 train_v6.py 结束 + GPU 0-5 释放，然后自动：
#   1) 选最新 checkpoint（优先 PREFER_CKPT，回退最新 epoch_*，再回退最新 step_*）
#   2) 逐臂跑 5 个 action_path case，每 case 一张卡（0-4），推理 log 每 case 分开
#   3) 跑 eval_action_cases.py 定量打分（DINO/SSIM），eval log 单独
# 全程记 master.log，跑完把 eval 汇总表 append 进 master.log（醒来直接看结论）。
#
# 用法：tmux new -s autoeval; bash src/scripts/v6/auto_eval_after_train.sh
# 服务器前置：export TMPDIR=/tmp（脚本内已设）。
# ============================================================================
set -uo pipefail

# ---- 用户配置（按需改）----
OUT="${OUT:-/home/nvme02/wlx/Memory/outputs}"
CKPT_DIR="${CKPT_DIR:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
INF="${INF:-/home/nvme02/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
TRAIN_RUN="${TRAIN_RUN:-$OUT/v6/train/latentconcat_lora_20260706_095953}"   # 当前训练 run 目录
PREFER_CKPT="${PREFER_CKPT:-epoch_5}"     # 优先用哪个 checkpoint（训完 = epoch_5）
RUN_NAME="${RUN_NAME:-v6_eval_ep027}"     # 推理产出 run 名（infer + eval 共用）
STEPS="${STEPS:-40}"                       # 采样步数
NUM_CLIPS="${NUM_CLIPS:-5}"                # 每 case clip 数（5=25s）
POLL_SEC="${POLL_SEC:-60}"                 # 轮询间隔（秒）
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"       # 卡视为"空闲"的显存上限（MiB）
GRACE_SEC="${GRACE_SEC:-600}"              # 宽限期：这段时间内没见到训练进程 → 视为训练已结束，直接进 eval
GPU_WAIT_MAX="${GPU_WAIT_MAX:-3600}"       # 等 GPU 0-4 释放的最长秒数，超时后仍尝试推理（记警告）
export TMPDIR=/tmp
export OUTPUT_ROOT="$OUT"                   # infer/eval 产出根与本脚本 $OUT 对齐（防解耦）

# 5 个 case（一 case 一卡，共 5 卡：0-4）
CASES=(
  Ep000027_p0001_75s_87s_lookback_path
  Ep000027_p0001_head20s_lookaround
  Ep000027_p0006_93s_105s_boxes_lookback
  Ep000027_p0007_26s_35s_fwd_back_two_windows
  Ep000027_p0007_77s_86s_two_windows_revisit
)

# 推理臂：tag | lora参数 | retrieval。默认 trained + base 两臂；
# 想加记忆消融就取消下面 trained_none 那行注释，并把 EVAL_ARMS 改成 trained,trained_none,base。
ARMS=(
  "trained|--lora_path {LORA}|bank"
  "base||none"
  # "trained_none|--lora_path {LORA}|none"
)
EVAL_ARMS="${EVAL_ARMS:-trained,base}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
LOGDIR="$REPO/logs/auto_eval_ep027/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$MASTER"; }

log "=== auto_eval_after_train 启动 ==="
log "REPO=$REPO  TRAIN_RUN=$TRAIN_RUN  RUN_NAME=$RUN_NAME  LOGDIR=$LOGDIR"

# ---- 1a. 先确认训练在跑（防"训练还没起就误触发"）----
log "确认训练是否在运行（宽限 ${GRACE_SEC}s）..."
_seen=0; _waited=0
while [ "$_waited" -lt "$GRACE_SEC" ]; do
  if pgrep -f "src/pipeline/v6/train_v6.py" >/dev/null 2>&1; then _seen=1; break; fi
  sleep "$POLL_SEC"; _waited=$((_waited + POLL_SEC))
done
if [ "$_seen" -eq 1 ]; then
  log "训练在运行 → 等待其结束..."
  while pgrep -f "src/pipeline/v6/train_v6.py" >/dev/null 2>&1; do sleep "$POLL_SEC"; done
  log "train_v6 已退出。"
else
  log "宽限期内未见训练进程 → 视为训练已结束，直接进 eval。"
fi

# ---- 1b. 等 GPU 0-4（推理用卡）释放，带超时防挂死 ----
log "等待 GPU 0-4 显存 < ${GPU_FREE_MIB}MiB（最长 ${GPU_WAIT_MAX}s）..."
_waited=0
while true; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' -v lim="$GPU_FREE_MIB" '$1<=4 && $2+0>lim {print $1}')
  [ -z "$busy" ] && break
  if [ "$_waited" -ge "$GPU_WAIT_MAX" ]; then
    log "⚠️ 等 GPU 0-4 释放超时，卡 [$busy] 仍占用，继续尝试推理（可能 OOM）。"
    break
  fi
  log "卡 [$busy] 仍占用，继续等..."
  sleep "$POLL_SEC"; _waited=$((_waited + POLL_SEC))
done
log "GPU 0-4 就绪。开始 eval 流程。"

# ---- 2. 选 checkpoint ----
CKPTS="$TRAIN_RUN/checkpoints"
if [ -d "$CKPTS/$PREFER_CKPT" ]; then
  LORA="$CKPTS/$PREFER_CKPT"
else
  LORA="$(ls -d "$CKPTS"/epoch_* 2>/dev/null | sort -V | tail -1)"
  [ -z "$LORA" ] && LORA="$(ls -d "$CKPTS"/step_* 2>/dev/null | sort -V | tail -1)"
fi
if [ -z "${LORA:-}" ] || [ ! -f "$LORA/lora.pth" ]; then
  log "❌ 找不到可用 checkpoint（$CKPTS 下无 epoch_*/step_* 或缺 lora.pth）。退出。"
  exit 1
fi
log "用 checkpoint: $LORA"

# ---- 3. 推理（逐臂；每臂 5 case 各一卡；每 case 一份 log）----
run_arm(){  # $1=tag  $2=lora参数(已展开)  $3=retrieval
  local tag="$1" loraarg="$2" retr="$3"
  log "推理臂 [$tag] retrieval=$retr —— 5 case 并行（卡 0-4）..."
  local i c
  for i in "${!CASES[@]}"; do
    c="${CASES[$i]}"
    CUDA_VISIBLE_DEVICES="$i" python src/pipeline/v6/latentconcat_infer.py \
      --ckpt_dir "$CKPT_DIR" $loraarg \
      --image "$INF/$c/image.jpg" --action_path "$INF/$c" \
      --prompt "$(cat "$INF/$c/prompt.txt")" \
      --num_clips "$NUM_CLIPS" --frame_num 81 \
      --retrieval "$retr" --num_inference_steps "$STEPS" \
      --run_name "$RUN_NAME" --tag "$tag" \
      > "$LOGDIR/infer_${tag}_${c}.log" 2>&1 &
  done
  wait
  log "推理臂 [$tag] 完成。"
}

for spec in "${ARMS[@]}"; do
  IFS='|' read -r a_tag a_lora a_retr <<< "$spec"
  a_lora="${a_lora//\{LORA\}/$LORA}"     # 展开 {LORA} 占位
  run_arm "$a_tag" "$a_lora" "$a_retr"
done

# ---- 4. 定量 eval（DINO/SSIM；单独 log）----
log "定量 eval（eval_action_cases.py）..."
python src/pipeline/v6/eval_action_cases.py \
  --infer_root "$OUT/v6/infer/$RUN_NAME" \
  --cases_root "$INF" \
  --arms "$EVAL_ARMS" \
  --device cuda:0 \
  > "$LOGDIR/eval.log" 2>&1
log "eval 完成。scores.csv → $OUT/v6/infer/$RUN_NAME/action_eval_scores.csv"

# ---- 5. 把 eval 汇总表贴进 master.log（醒来直接看）----
log "===== EVAL 汇总（详见 $LOGDIR/eval.log）====="
tail -40 "$LOGDIR/eval.log" | tee -a "$MASTER"
log "=== 全部完成。日志目录: $LOGDIR ==="
