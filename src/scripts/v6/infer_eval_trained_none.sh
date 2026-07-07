#!/usr/bin/env bash
# ============================================================================
# infer_eval_trained_none.sh — 补跑 trained_none 臂（LoRA 照用、retrieval=none）
# ----------------------------------------------------------------------------
# 目的：隔离"记忆增益"。已有 trained(LoRA+bank) 与 base(无LoRA+无bank) 两臂，
#   本脚本补第三臂 trained_none = LoRA(epoch_5) + retrieval=none（给空记忆），
#   然后三臂一起打分。trained − trained_none = 纯记忆贡献（排除 LoRA 容量）。
#
# 全程独立 log / 独立 CSV，绝不覆盖已有 trained/base 的产物：
#   - 视频 : $OUTROOT/trained_none/<case>/long_video.mp4     （新子目录）
#   - log  : logs/trained_none_ep027/<ts>/{master,infer_*,eval_3arm}.log
#   - CSV  : $OUTROOT/action_eval_scores_3arm.csv            （原 2 臂 CSV 保留）
#
# 用法：tmux new -s tnone; bash src/scripts/v6/infer_eval_trained_none.sh
# 前置：export TMPDIR=/tmp（脚本内已设）；卡 0-4 空闲。
# ============================================================================
set -uo pipefail

# ---- 配置（按需 env 覆盖）----
CKPT="${CKPT:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
LORA="${LORA:-/home/nvme02/wlx/Memory/outputs/v6/train/latentconcat_lora_20260706_095953/checkpoints/epoch_5}"
INF="${INF:-/home/nvme02/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected}"
RUN="${RUN:-v6_eval_ep027}"
OUTROOT_BASE="${OUTPUT_ROOT:-/home/nvme02/wlx/Memory/outputs}"
OUTROOT="$OUTROOT_BASE/v6/infer/$RUN"
STEPS="${STEPS:-40}"
NUM_CLIPS="${NUM_CLIPS:-5}"
export TMPDIR=/tmp
export OUTPUT_ROOT="$OUTROOT_BASE"

CASES=(
  Ep000027_p0001_75s_87s_lookback_path
  Ep000027_p0001_head20s_lookaround
  Ep000027_p0006_93s_105s_boxes_lookback
  Ep000027_p0007_26s_35s_fwd_back_two_windows
  Ep000027_p0007_77s_86s_two_windows_revisit
)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
LOGDIR="$REPO/logs/trained_none_ep027/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$MASTER"; }

log "=== infer_eval_trained_none 启动 ==="
log "REPO=$REPO  RUN=$RUN  LORA=$LORA"
log "OUTROOT=$OUTROOT  LOGDIR=$LOGDIR"

# ---- checkpoint 存在性校验 ----
if [ ! -f "$LORA/lora.pth" ]; then
  log "❌ 找不到 $LORA/lora.pth，退出。"; exit 1
fi

# ---- GPU 0-4 空闲检查（占用则告警但继续）----
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', ' '$1<=4 && $2+0>2000 {print $1}')
[ -n "$busy" ] && log "⚠️ 卡 [$busy] 仍占用，可能 OOM；如需等待请先手动确认。"

# ---- 推理：5 case 并行（卡 0-4），trained_none = LoRA + retrieval=none ----
log "推理臂 [trained_none] retrieval=none —— 5 case 并行（卡 0-4）..."
for i in "${!CASES[@]}"; do
  c="${CASES[$i]}"
  CUDA_VISIBLE_DEVICES="$i" python src/pipeline/v6/latentconcat_infer.py \
    --ckpt_dir "$CKPT" --lora_path "$LORA" \
    --image "$INF/$c/image.jpg" --action_path "$INF/$c" \
    --prompt "$(cat "$INF/$c/prompt.txt")" \
    --num_clips "$NUM_CLIPS" --frame_num 81 \
    --retrieval none --num_inference_steps "$STEPS" \
    --run_name "$RUN" --tag trained_none \
    > "$LOGDIR/infer_trained_none_${c}.log" 2>&1 &
done
wait
log "推理臂 [trained_none] 完成。"

# ---- 三臂 eval → 独立 CSV（不覆盖 action_eval_scores.csv）----
CSV3="$OUTROOT/action_eval_scores_3arm.csv"
log "三臂 eval（trained / trained_none / base）→ $CSV3"
python src/pipeline/v6/eval_action_cases.py \
  --infer_root "$OUTROOT" \
  --cases_root "$INF" \
  --arms trained,trained_none,base \
  --out_csv "$CSV3" \
  --device cuda:0 \
  > "$LOGDIR/eval_3arm.log" 2>&1
log "eval 完成。CSV → $CSV3"

# ---- 汇总贴进 master.log（醒来直接看）----
log "===== 三臂 EVAL 汇总（详见 $LOGDIR/eval_3arm.log）====="
tail -30 "$LOGDIR/eval_3arm.log" | tee -a "$MASTER"
log "=== 全部完成。日志目录: $LOGDIR ==="
