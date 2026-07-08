#!/usr/bin/env bash
# ============================================================================
# infer_eval_60s.sh — 60s 长视频 base vs v6 定性 demo（episode 模式，可选定量）
# ----------------------------------------------------------------------------
# 目的：跑一条 ~60s（NUM_CLIPS=12）的长视频，对比 base（裸骨干）vs v6（epoch_5 LoRA+bank）。
#   episode 模式：从 v4_dynamic 数据集拼一个 episode 的多 clip 长轨迹自回归推理（ep027 不在此
#   数据集，故用 verify episode；定性 demo 无碍）。--score 开：若 episode 内有重访点则顺带算
#   DINO/SSIM（定量能跑就跑，跑不了纯 demo）。
#
# 全程独立命名，绝不覆盖 25s 的 v6_eval_ep027：
#   - 视频 : $OUTROOT/v6_demo_60s/{v6,base}/<episode_id>/long_video.mp4
#   - CSV  : $OUTROOT/v6_demo_60s/<tag>/.../scores.csv（--score 产出，各臂独立）
#   - log  : logs/demo_60s/<ts>/{master,infer_v6,infer_base}.log
#
# 用法：tmux new -s demo60; EPISODE_ID=<挑的长episode> bash src/scripts/v6/infer_eval_60s.sh
# 前置：export TMPDIR=/tmp（脚本内已设）；卡 0-1 空闲。~5h（12 clip 顺序自回归，不可并行 clip）。
# ============================================================================
set -uo pipefail

# ---- 配置（按需 env 覆盖）----
CKPT="${CKPT:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
LORA="${LORA:-/home/nvme02/wlx/Memory/outputs/v6/train/latentconcat_lora_20260706_095953/checkpoints/epoch_5}"
DS="${DS:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"
METADATA="${METADATA:-metadata_verify_train.csv}"   # 相对 DS 的 CSV
EPISODE_ID="${EPISODE_ID:-first}"                    # 挑的长 episode（≥12 clip）；'first'=CSV 第一个
RUN="${RUN:-v6_demo_60s}"                            # 新 run 名，独立不覆盖 25s
NUM_CLIPS="${NUM_CLIPS:-12}"                          # 12×81=972 帧 ≈ 60s（脚本内实际上限=T//81）
STEPS="${STEPS:-40}"
OUTROOT_BASE="${OUTPUT_ROOT:-/home/nvme02/wlx/Memory/outputs}"
OUTROOT="$OUTROOT_BASE/v6/infer/$RUN"
export TMPDIR=/tmp
export OUTPUT_ROOT="$OUTROOT_BASE"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
LOGDIR="$REPO/logs/demo_60s/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
log(){ echo "$(date '+%F %T') | $*" | tee -a "$MASTER"; }

log "=== infer_eval_60s 启动 ==="
log "RUN=$RUN  EPISODE_ID=$EPISODE_ID  NUM_CLIPS=$NUM_CLIPS  DS=$DS"
log "OUTROOT=$OUTROOT  LOGDIR=$LOGDIR"

[ -f "$LORA/lora.pth" ] || { log "❌ 缺 $LORA/lora.pth"; exit 1; }
[ -f "$DS/$METADATA" ] || { log "❌ 缺 metadata $DS/$METADATA（先 ls \$DS/*.csv 确认名字）"; exit 1; }

# ---- GPU 0-1 空闲检查（占用告警不阻断）----
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', ' '$1<=1 && $2+0>2000 {print $1}')
[ -n "$busy" ] && log "⚠️ 卡 [$busy] 仍占用，可能 OOM。"

# ---- 2 臂并行：v6=卡0 / base=卡1（episode 模式，--score 开）----
# 臂：tag | lora参数 | retrieval | card
run_arm(){  # $1=tag $2=loraarg $3=retr $4=card
  local tag="$1" loraarg="$2" retr="$3" card="$4"
  log "推理臂 [$tag] retrieval=$retr → 卡 $card（episode=$EPISODE_ID, ${NUM_CLIPS} clip）..."
  CUDA_VISIBLE_DEVICES="$card" python src/pipeline/v6/latentconcat_infer.py \
    --ckpt_dir "$CKPT" $loraarg \
    --dataset_dir "$DS" --metadata "$METADATA" --episode_id "$EPISODE_ID" \
    --num_clips "$NUM_CLIPS" --frame_num 81 \
    --retrieval "$retr" --num_inference_steps "$STEPS" \
    --prompt_source data --score \
    --run_name "$RUN" --tag "$tag" \
    > "$LOGDIR/infer_${tag}.log" 2>&1 &
}

run_arm v6   "--lora_path $LORA" bank 0
run_arm base ""                  none 1
log "两臂已并行启动（卡 0=v6 / 卡 1=base），等待..."
wait
log "两臂推理完成。"

# ---- 汇总产物路径 ----
log "===== 产物 ====="
find "$OUTROOT" -name long_video.mp4 -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' 2>/dev/null | sort | tee -a "$MASTER"
log "scores（--score 若检出重访点才有）："
find "$OUTROOT" -name "scores.csv" 2>/dev/null | tee -a "$MASTER"
log "=== 全部完成。日志: $LOGDIR ==="
