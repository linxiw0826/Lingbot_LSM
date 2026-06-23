#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v5/run_infer_v5.sh
# ============================================================
# v5 in-context KV 记忆的多 clip 长视频自然画质生成（demo）。
# 单进程、单卡，不用 accelerate / torchrun。
# 产出走 paths.py：OUTPUT_ROOT/v5/infer/<run_name>/<tag>/ 下放
#   long_video.mp4 + infer.log + config 快照（config.yaml 或 config.json）。
#
# 必填：MEMORY_ENCODER_CKPT —— 训练产出的 memory_encoder.pth（train_v5 save_memory_encoder）。
#
# 可用环境变量覆盖（无需编辑本文件）：
#   CUDA_VISIBLE_DEVICES / MEMORY_ENCODER_CKPT / IMAGE / ACTION_PATH / SAVE_FILE
#   PROMPT / NUM_CLIPS / FRAME_NUM / SIZE / SAMPLE_STEPS / GUIDE_SCALE / SEED / FPS
#   TAG / RUN_NAME / GRID / ENCODER_DEPTH / OUTPUT_ROOT
#
# 示例（指定权重 + 首帧 + 动作轨迹，单卡 0）：
#   MEMORY_ENCODER_CKPT=/home/nvme02/wlx/Memory/outputs/v5/train/<run>/epoch_3/memory_encoder.pth \
#     IMAGE=/data/clip_000/image.jpg \
#     ACTION_PATH=/data/clip_000 \
#     bash src/scripts/v5/run_infer_v5.sh
# 示例（换卡）：
#   CUDA_VISIBLE_DEVICES=2 MEMORY_ENCODER_CKPT=.../memory_encoder.pth \
#     IMAGE=... ACTION_PATH=... bash src/scripts/v5/run_infer_v5.sh
# ============================================================

# ---- 模型权重 ----
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"     # lingbot-world 预训练权重
MEMORY_ENCODER_CKPT="${MEMORY_ENCODER_CKPT:-}"                          # 必填：训练好的 memory_encoder.pth

# ---- 数据（走环境变量 + 合理默认）----
IMAGE="${IMAGE:-/home/nvme02/Memory-dataset/demo/clip_000/image.jpg}"              # 首帧图像
ACTION_PATH="${ACTION_PATH:-/home/nvme02/Memory-dataset/demo/clip_000}"            # 含 poses/action/intrinsics.npy
SAVE_FILE="${SAVE_FILE:-}"                                              # 空 → 落 infer_run_dir/long_video.mp4

# ---- 生成参数 ----
PROMPT="${PROMPT:-First-person view of CS:GO competitive gameplay}"
NUM_CLIPS="${NUM_CLIPS:-12}"            # 多 clip 连续生成数（默认 12）
FRAME_NUM="${FRAME_NUM:-81}"            # 每 clip 帧数（默认 81）
SIZE="${SIZE:-480*832}"                 # 分辨率 H*W
SAMPLE_STEPS="${SAMPLE_STEPS:-70}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-10.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"

# ---- v5 超参（留空 → infer 从 training_metadata 自动采纳，与训练一致）----
GRID="${GRID:-}"                         # 空=从 training_metadata 采纳
ENCODER_DEPTH="${ENCODER_DEPTH:-}"       # 空=从 training_metadata 采纳

# ---- 产出 ----
RUN_NAME="${RUN_NAME:-}"                 # 空 → default_run_name('v5_infer')
TAG="${TAG:-long_video}"                # infer 场景 tag

# infer 单进程，默认单卡（可用 CUDA_VISIBLE_DEVICES 覆盖）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---- 环境护栏（项目记忆）----
export TMPDIR=/tmp                       # 防 pymp/torchelastic 孤儿落进 repo
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # 卡号按 PCI 总线序

# ---------- 路径检查 ----------
_err=0
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置" >&2; _err=1
fi
if [ -z "${MEMORY_ENCODER_CKPT}" ]; then
    echo "[ERROR] MEMORY_ENCODER_CKPT 未设置（必填：传环境变量 MEMORY_ENCODER_CKPT=.../memory_encoder.pth）" >&2; _err=1
fi
if [ -z "${IMAGE}" ]; then
    echo "[ERROR] IMAGE 未设置（首帧图像路径）" >&2; _err=1
fi
if [ -z "${ACTION_PATH}" ]; then
    echo "[ERROR] ACTION_PATH 未设置（含 poses.npy/action.npy/intrinsics.npy 的目录）" >&2; _err=1
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

if [ ! -f "${IMAGE}" ]; then
    echo "[ERROR] 首帧图像不存在：${IMAGE}" >&2; exit 1
fi
if [ ! -d "${ACTION_PATH}" ]; then
    echo "[ERROR] action_path 不是目录：${ACTION_PATH}" >&2; exit 1
fi
for _need in poses.npy action.npy intrinsics.npy; do
    if [ ! -f "${ACTION_PATH}/${_need}" ]; then
        echo "[ERROR] ${ACTION_PATH}/${_need} 不存在" >&2; exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

INFER_SCRIPT="${PROJECT_ROOT}/src/pipeline/v5/infer_v5.py"
if [ ! -f "${INFER_SCRIPT}" ]; then
    echo "[ERROR] 推理脚本不存在：${INFER_SCRIPT}" >&2; exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

# ---------- 参数数组 ----------
INFER_ARGS=(
    --ckpt_dir              "${CKPT_DIR}"
    --memory_encoder_ckpt   "${MEMORY_ENCODER_CKPT}"
    --image                 "${IMAGE}"
    --action_path           "${ACTION_PATH}"
    --prompt                "${PROMPT}"
    --num_clips             "${NUM_CLIPS}"
    --frame_num             "${FRAME_NUM}"
    --size                  "${SIZE}"
    --sample_steps          "${SAMPLE_STEPS}"
    --sample_shift          "${SAMPLE_SHIFT}"
    --guide_scale           "${GUIDE_SCALE}"
    --seed                  "${SEED}"
    --fps                   "${FPS}"
    --tag                   "${TAG}"
)

# save_file：空 → infer_v5 默认落 infer_run_dir/long_video.mp4
if [ -n "${SAVE_FILE}" ]; then
    INFER_ARGS+=(--save_file "${SAVE_FILE}")
fi
# run_name 留空 → infer 用 default_run_name
if [ -n "${RUN_NAME}" ]; then
    INFER_ARGS+=(--run_name "${RUN_NAME}")
fi
# grid/encoder_depth：仅当显式给出才透传（空=从 training_metadata 自动采纳）
if [ -n "${GRID}" ]; then
    INFER_ARGS+=(--grid "${GRID}")
fi
if [ -n "${ENCODER_DEPTH}" ]; then
    INFER_ARGS+=(--encoder_depth "${ENCODER_DEPTH}")
fi
# 默认 low-only：不加 --inject_high。

echo "====================================================="
echo "  LingBot-World Memory v5 推理启动（多 clip 长视频 demo）"
echo "  MEMORY_ENCODER_CKPT : ${MEMORY_ENCODER_CKPT}"
echo "  IMAGE               : ${IMAGE}"
echo "  ACTION_PATH         : ${ACTION_PATH}"
echo "  SAVE_FILE           : ${SAVE_FILE:-<infer_run_dir/long_video.mp4>}"
echo "  NUM_CLIPS           : ${NUM_CLIPS}"
echo "  FRAME_NUM           : ${FRAME_NUM}"
echo "  SIZE                : ${SIZE}"
echo "  PROMPT              : ${PROMPT}"
echo "  GRID                : ${GRID:-<从 training_metadata 采纳>}"
echo "  ENCODER_DEPTH       : ${ENCODER_DEPTH:-<从 training_metadata 采纳>}"
echo "  TAG                 : ${TAG}"
echo "  RUN_NAME            : ${RUN_NAME:-<default_run_name>}"
echo "  OUTPUT_ROOT         : ${OUTPUT_ROOT:-<paths.py 默认>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  LOG_FILE            : ${LOG_FILE}"
echo "====================================================="

CMD=(
    python "${INFER_SCRIPT}"
    "${INFER_ARGS[@]}"
)

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; _EXIT="${PIPESTATUS[0]}"
if [ "${_EXIT}" -ne 0 ]; then
    echo "[ERROR] v5 推理失败，退出码 ${_EXIT}" >&2
    exit "${_EXIT}"
fi

echo ""
echo "===== v5 推理完成（产出走 paths.py：OUTPUT_ROOT/v5/infer/<run_name>/<tag>/）====="
echo "===== 默认 low-only in-context KV 注入；demo 视频 + infer.log + config 快照同目录 ====="
