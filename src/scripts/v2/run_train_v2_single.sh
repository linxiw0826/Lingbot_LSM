#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_train_v2_single.sh
# ============================================================
STAGE=1                    # 训练阶段：1（冻结DiT只训Memory）或 2（全参联合微调）
LORA_RANK=0                # LoRA rank：0=全参微调；32/64=LoRA微调（显存少50%）
LORA_TARGET_MODULES=""     # LoRA目标模块（留空自动检测）

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
DATASET_DIR="/home/nvme02/lingbot-world/datasets/processed_csgo_v3"
OUTPUT_BASE="/home/nvme02/wlx/Memory/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/train/v2_stage${STAGE}_single"
RESUME_FROM=""  # 从头训练

NUM_EPOCHS=2
LEARNING_RATE=1e-4
LR_DIT=1e-5
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=8    # 4 GPU × 8 accum = effective batch 32
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

# ---------- 路径计算 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------- 根据 STAGE 选择 accelerate 配置文件 ----------
if [ "${STAGE}" -eq 1 ]; then
    ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/accelerate_stage1.yaml"
elif [ "${STAGE}" -eq 2 ]; then
    ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/accelerate_stage2.yaml"
else
    echo "[ERROR] STAGE 必须为 1 或 2，当前值：${STAGE}" >&2; exit 1
fi
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[ERROR] accelerate 配置文件不存在：${ACCEL_CONFIG}" >&2; exit 1
fi

# ---------- 日志目录 & 日志文件 ----------
LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 训练 v2-single 启动"
echo "  Stage              : ${STAGE}"
echo "  LoRA rank          : ${LORA_RANK}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS           : ${NUM_GPUS}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "  LOG_FILE           : ${LOG_FILE}"
echo "====================================================="

CMD=(
    accelerate launch
    --config_file "${ACCEL_CONFIG}"
    --num_processes "${NUM_GPUS}"
    "${PROJECT_ROOT}/src/pipeline/v2/train_v2_stage${STAGE}.py"
    --ckpt_dir                    "${CKPT_DIR}"
    --dataset_dir                 "${DATASET_DIR}"
    --output_dir                  "${OUTPUT_DIR}"
    --stage                       "${STAGE}"
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
    CMD+=(--lora_rank "${LORA_RANK}")
    if [ -n "${LORA_TARGET_MODULES}" ]; then
        CMD+=(--lora_target_modules "${LORA_TARGET_MODULES}")
    fi
fi

if [ -n "${RESUME_FROM}" ]; then
    CMD+=(--resume "${RESUME_FROM}")
fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; exit "${PIPESTATUS[0]}"
