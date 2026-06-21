#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_train_v3_stage2_dual.sh
# ============================================================
# 实验配置 ⑧：v3 数据（8ch action）+ Stage2 + 双模型全参数联合微调（最终目标）
# 状态：PENDING — 依赖：① train_v3_stage2_dual.py 实现 + ② v3 数据就绪 + ③ D-03 解除
# PENDING[D-03]：CKPT_DIR 待定（选项A: 原始预训练权重；选项B: 对方 CSGO-DiT 权重）

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"   # PENDING[D-03]: 确认后改为 Stage1 输出或 CSGO-DiT 权重
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3_8ch"
OUTPUT_BASE="/home/nvme02/wlx/Memory/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/train/v3_stage2_dual"
RESUME_FROM_LOW=""
RESUME_FROM_HIGH=""

NUM_EPOCHS=5
LEARNING_RATE=1e-4             # 记忆模块 lr
LR_DIT=1e-5                    # DiT blocks lr（Stage2 全参微调）
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=8
MAX_GRAD_NORM=1.0
SAVE_EVERY_N_EPOCHS=5
DATASET_REPEAT=1
NUM_FRAMES=81
HEIGHT=480
WIDTH=832
NFP_LOSS_WEIGHT=0.1
ACTION_DIM=8

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

# ============================================================
# 以下内容通常无需修改
# ============================================================

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | xargs)

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

# Stage2 使用 accelerate_stage2.yaml（ZeRO-3 + CPU offload，全参解冻需要）
ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/accelerate_stage2.yaml"
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[ERROR] accelerate 配置文件不存在：${ACCEL_CONFIG}" >&2; exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --config_file "${ACCEL_CONFIG}"
    --num_processes "${NUM_GPUS}"
)
TRAIN_ARGS=(
    --ckpt_dir                    "${CKPT_DIR}"
    --dataset_dir                 "${DATASET_DIR}"
    --output_dir                  "${OUTPUT_DIR}"
    --stage                       2
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
    --action_dim                  "${ACTION_DIM}"
    --gradient_checkpointing
)

TRAIN_SCRIPT="${PROJECT_ROOT}/src/pipeline/v3/train_v3_stage2_dual.py"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 双模型训练 v3 Stage2 启动"
echo "  实验配置 ⑧：v3（8ch action）+ Stage2 + 双模型（最终目标）"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS           : ${NUM_GPUS}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "  LOG_FILE           : ${LOG_FILE}"
echo "====================================================="

# ============================================================
# Step 1/2: 训练 low_noise_model Stage2
# ============================================================
echo ""
echo "===== Step 1/2: 训练 low_noise_model Stage2（t < 0.947）====="
CMD_LOW=(
    accelerate launch "${COMMON_ARGS[@]}"
    "${TRAIN_SCRIPT}"
    "${TRAIN_ARGS[@]}"
    --model_type low
)
if [ -n "${RESUME_FROM_LOW}" ]; then
    CMD_LOW+=(--resume "${RESUME_FROM_LOW}")
fi
echo "执行命令（low）："
echo "${CMD_LOW[*]}"
echo ""
"${CMD_LOW[@]}" 2>&1 | tee -a "${LOG_FILE}"
echo ""
echo "===== low_noise_model Stage2 训练完成 ====="

# ============================================================
# Step 2/2: 训练 high_noise_model Stage2
# ============================================================
echo ""
echo "===== Step 2/2: 训练 high_noise_model Stage2（t >= 0.947）====="
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
echo "===== 双模型 Stage2 训练完成 ====="
echo "  low_noise_model  → ${OUTPUT_DIR}/low_noise_model/"
echo "  high_noise_model → ${OUTPUT_DIR}/high_noise_model/"
