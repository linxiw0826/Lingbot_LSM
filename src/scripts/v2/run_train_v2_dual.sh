#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_train_v2_dual.sh
# ============================================================
# Stage1 双模型训练：先运行 low_noise_model，完成后自动运行 high_noise_model
# 两个模型共用同一配置，输出分别保存至 OUTPUT_DIR/low_noise_model/ 和 OUTPUT_DIR/high_noise_model/

LORA_RANK=0                # LoRA rank：0=全参微调；32/64=LoRA微调
LORA_TARGET_MODULES=""     # LoRA目标模块（留空自动检测）

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
OUTPUT_BASE="/home/nvme02/wlx/Memory/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/train/v2_stage1_dual"
RESUME_FROM_LOW=""          # low_noise_model 已完成（epoch_2），跳过
RESUME_FROM_HIGH="/home/nvme02/wlx/Memory/outputs/train/v2_stage1_dual/high_noise_model/epoch_1"

NUM_EPOCHS=2
LEARNING_RATE=1e-4
LR_DIT=1e-5
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=8    # 4 GPU × 8 accum = effective batch 32（与 8-GPU × 4 等效）
MAX_GRAD_NORM=1.0
SAVE_EVERY_N_EPOCHS=1
DATASET_REPEAT=1
NUM_FRAMES=81
HEIGHT=480
WIDTH=832
NFP_LOSS_WEIGHT=0.1

CUDA_VISIBLE_DEVICES="0,1,2,3"

# ============================================================
# 以下内容通常无需修改
# ============================================================

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | xargs)

# ---------- 路径检查 ----------
_err=0
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置" >&2; _err=1
fi
if [ -z "${DATASET_DIR}" ]; then
    echo "[ERROR] DATASET_DIR 未设置" >&2; _err=1
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/accelerate_stage1.yaml"
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[ERROR] accelerate 配置文件不存在：${ACCEL_CONFIG}" >&2; exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}"

# ---------- 公共参数数组（两个模型共用）----------
COMMON_ARGS=(
    --config_file "${ACCEL_CONFIG}"
    --num_processes "${NUM_GPUS}"
)
TRAIN_ARGS=(
    --ckpt_dir                    "${CKPT_DIR}"
    --dataset_dir                 "${DATASET_DIR}"
    --output_dir                  "${OUTPUT_DIR}"
    --stage                       1
    --num_epochs                  "${NUM_EPOCHS}"
    --learning_rate               "${LEARNING_RATE}"
    --lr_dit                      "${LR_DIT}"
    --weight_decay                "${WEIGHT_DECAY}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_grad_norm               "${MAX_GRAD_NORM}"
    --save_every_n_epochs         "${SAVE_EVERY_N_EPOCHS}"
    --dataset_repeat              "${DATASET_REPEAT}"
    --num_frames                  "${NUM_FRAMES}"
    --height                      "${HEIGHT}"
    --width                       "${WIDTH}"
    --nfp_loss_weight             "${NFP_LOSS_WEIGHT}"
    --gradient_checkpointing
)

if [ "${LORA_RANK}" -gt 0 ]; then
    TRAIN_ARGS+=(--lora_rank "${LORA_RANK}")
    if [ -n "${LORA_TARGET_MODULES}" ]; then
        TRAIN_ARGS+=(--lora_target_modules "${LORA_TARGET_MODULES}")
    fi
fi

TRAIN_SCRIPT="${PROJECT_ROOT}/src/pipeline/v2/train_v2_stage1_dual.py"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 双模型训练 v2 启动"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS           : ${NUM_GPUS}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "  LOG_FILE           : ${LOG_FILE}"
echo "====================================================="

# ============================================================
# Step 1/2: 训练 low_noise_model
# ============================================================
echo ""
echo "===== Step 1/2: 训练 low_noise_model（t < 0.947）====="
CMD_LOW=(
    accelerate launch "${COMMON_ARGS[@]}"
    "${TRAIN_SCRIPT}"
    "${TRAIN_ARGS[@]}"
    --model_type low
)
if [ -n "${RESUME_FROM_LOW}" ]; then
    CMD_LOW+=(--resume "${RESUME_FROM_LOW}")
fi

echo "===== low_noise_model 已完成，跳过（epoch_2 checkpoint 已存在）====="
# echo "执行命令（low）："
# echo "${CMD_LOW[*]}"
# echo ""
# "${CMD_LOW[@]}" 2>&1 | tee -a "${LOG_FILE}"

# ============================================================
# Step 2/2: 训练 high_noise_model
# ============================================================
echo ""
echo "===== Step 2/2: 训练 high_noise_model（t >= 0.947）====="
CMD_HIGH=(
    accelerate launch "${COMMON_ARGS[@]}"
    "${TRAIN_SCRIPT}"
    "${TRAIN_ARGS[@]}"
    --model_type high
)
if [ -n "${RESUME_FROM_HIGH}" ]; then
    CMD_HIGH+=(--resume "${RESUME_FROM_HIGH}")
fi

echo "执行命令（high）："
echo "${CMD_HIGH[*]}"
echo ""
"${CMD_HIGH[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "===== 双模型训练完成 ====="
echo "  low_noise_model  → ${OUTPUT_DIR}/low_noise_model/"
echo "  high_noise_model → ${OUTPUT_DIR}/high_noise_model/"
