#!/usr/bin/env bash
# ============================================================================
# infer_eval_60s.sh — 60s 长视频 base vs v6 定性 demo（5 episode × 2 臂，5 卡并行）
# ----------------------------------------------------------------------------
# 编排：5 个长 episode，卡 i 负责 EPISODES[i]，卡内先 v6（epoch_5 LoRA+bank）后 base
#   （裸骨干+none）顺序跑（单卡放不下两个 14B 并行 → 同卡串行）。5 卡并行 → 墙钟 ≈ 2×单臂。
#   episode 模式（ep027 不在此数据集，用 v4_dynamic 长 episode；定性 demo 无碍）。--score 开：
#   episode 内若有重访点则顺带算 DINO/SSIM（定量能跑就跑，不判 GO）。
#
# 独立命名，绝不覆盖 25s 的 v6_eval_ep027：
#   - 视频 : $OUTROOT/v6_demo_60s/{v6,base}/<episode_id>/long_video.mp4
#   - log  : logs/demo_60s/<ts>/{master, infer_<arm>_<episode>.log}
#
# 用法：tmux new -s demo60; bash src/scripts/v6/infer_eval_60s.sh
# 前置：export TMPDIR=/tmp（脚本内已设）；卡 0-4 空闲。~11h（12 clip×2 臂顺序，5 卡并行）。
# ============================================================================
set -uo pipefail

# ---- 配置（按需 env 覆盖）----
CKPT="${CKPT:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
LORA="${LORA:-/home/nvme02/wlx/Memory/outputs/v6/train/latentconcat_lora_20260706_095953/checkpoints/epoch_5}"
DS="${DS:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"
METADATA="${METADATA:-metadata_all.csv}"   # 含下面 5 个 episode（grep 确认过）
RUN="${RUN:-v6_demo_60s}"                  # 新 run 名，独立不覆盖 25s
NUM_CLIPS="${NUM_CLIPS:-12}"                # 12×81=972 帧 ≈ 60s（上限 = episode 帧数//81）
STEPS="${STEPS:-40}"
OUTROOT_BASE="${OUTPUT_ROOT:-/home/nvme02/wlx/Memory/outputs}"
OUTROOT="$OUTROOT_BASE/v6/infer/$RUN"
export TMPDIR=/tmp
export OUTPUT_ROOT="$OUTROOT_BASE"

# 5 个长 episode（lister top5：clip 数 41/41/41/38/37）；卡 i ↔ EPISODES[i]
EPISODES=(
  ep302_p09
  ep302_p01
  ep302_p00
  ep44_p03
  ep44_p00
)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
LOGDIR="$REPO/logs/demo_60s/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$MASTER"; }

log "=== infer_eval_60s 启动（5 episode × 2 臂，5 卡并行）==="
log "RUN=$RUN  NUM_CLIPS=$NUM_CLIPS  METADATA=$METADATA"
log "EPISODES=${EPISODES[*]}"
log "OUTROOT=$OUTROOT  LOGDIR=$LOGDIR"

[ -f "$LORA/lora.pth" ] || { log "❌ 缺 $LORA/lora.pth"; exit 1; }
[ -f "$DS/$METADATA" ] || { log "❌ 缺 metadata $DS/$METADATA"; exit 1; }

# ---- GPU 0-4 空闲检查（占用告警不阻断）----
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', ' '$1<=4 && $2+0>2000 {print $1}')
[ -n "$busy" ] && log "⚠️ 卡 [$busy] 仍占用，可能 OOM。"

# ---- 单臂推理（episode 模式，--score 开）----
run_one(){  # $1=card $2=episode $3=tag $4=loraarg $5=retr
  local card="$1" ep="$2" tag="$3" loraarg="$4" retr="$5"
  CUDA_VISIBLE_DEVICES="$card" python src/pipeline/v6/latentconcat_infer.py \
    --ckpt_dir "$CKPT" $loraarg \
    --dataset_dir "$DS" --metadata "$METADATA" --episode_id "$ep" \
    --num_clips "$NUM_CLIPS" --frame_num 81 \
    --retrieval "$retr" --num_inference_steps "$STEPS" \
    --prompt_source data --score \
    --run_name "$RUN" --tag "$tag" \
    > "$LOGDIR/infer_${tag}_${ep}.log" 2>&1
}

# ---- 卡 i 负责 episode i：卡内先 v6 后 base 串行；5 卡并行 ----
for i in "${!EPISODES[@]}"; do
  ep="${EPISODES[$i]}"
  (
    log "卡 $i / episode $ep：先 v6（LoRA+bank）..."
    run_one "$i" "$ep" v6 "--lora_path $LORA" bank
    log "卡 $i / episode $ep：v6 完成 → 再 base（none）..."
    run_one "$i" "$ep" base "" none
    log "卡 $i / episode $ep：两臂完成。"
  ) &
done
wait
log "全部 5 episode × 2 臂 完成。"

# ---- 汇总产物 ----
log "===== 产物（应 10 个 long_video.mp4）====="
find "$OUTROOT" -name long_video.mp4 -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' 2>/dev/null | sort | tee -a "$MASTER"
log "scores（--score 若检出重访点才有）："
find "$OUTROOT" -name "scores.csv" 2>/dev/null | tee -a "$MASTER"
log "=== 全部完成。日志: $LOGDIR ==="
