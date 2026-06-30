#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v6/run_train_v6.sh
# ============================================================
# v6：latent-concat 训练（frame-dim anchor concat + rank-128 LoRA + flow-matching）。
# DiT 全冻，唯一可训练 = LoRA（self_attn q/k/v/o + ffn）。anchor = 同 episode 较早帧（自监督）。
# 产出走 paths.py：OUTPUT_ROOT/v6/train/<run_name>/（--run_name 留空 → default_run_name）。
#
# ⚠️ 前置：本脚本只在 v6 #1 latent-concat 多 clip 理想复验（Step 44 / S-V5）GO 后才应运行。
#
# 可用环境变量覆盖（无需编辑本文件）：
#   TRAIN_GPUS / DATASET_DIR / PHASE / NUM_EPOCHS / LORA_RANK / LORA_ALPHA / LORA_TARGETS
#   NUM_ANCHOR_FRAMES / LEARNING_RATE / RUN_NAME / RESUME_FROM / OUTPUT_ROOT
#
# 示例（6 卡从头训，默认）：
#   bash src/scripts/v6/run_train_v6.sh
# 示例（自定 rank + targets）：
#   LORA_RANK=128 LORA_TARGETS=self_attn,ffn,cross_attn bash src/scripts/v6/run_train_v6.sh
# 示例（续训，从某 epoch 目录恢复）：
#   RESUME_FROM=/home/nvme02/wlx/Memory/outputs/v6/train/<run>/epoch_2 bash src/scripts/v6/run_train_v6.sh
#
# 默认 fresh 训练（不 resume）；如需续训设置 RESUME_FROM。
# ============================================================

# ---- 模型 / 数据 / 阶段 ----
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"                              # lingbot-world 预训练权重
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"           # 默认=revisit verify 集（同 v5，可 env 覆盖）
PHASE="${PHASE:-verify}"                 # 训练阶段："verify" / "exp" / "full"

# ---- v6 LoRA 超参 ----
LORA_RANK="${LORA_RANK:-128}"            # LoRA 秩 r（StoryMem 同骨干先例 r128）
LORA_ALPHA="${LORA_ALPHA:-128}"          # scaling=alpha/r；默认 alpha=r → scaling=1
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"      # LoRA dropout
LORA_TARGETS="${LORA_TARGETS:-self_attn,ffn}"   # 挂载层组（self_attn/ffn/cross_attn/cam）
NUM_ANCHOR_FRAMES="${NUM_ANCHOR_FRAMES:-1}"     # 拼接 anchor 帧数（token 预算）

# ---- 训练超参（对齐 v5：1e-4 量级）----
NUM_EPOCHS="${NUM_EPOCHS:-3}"            # 验证阶段 3 epoch
LEARNING_RATE="${LEARNING_RATE:-1e-4}"   # 回退稳定值（同 v5）
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=4
MAX_GRAD_NORM=1.0
SAVE_EVERY_N_EPOCHS=1
DATASET_REPEAT=1
NUM_FRAMES=81
HEIGHT=480
WIDTH=832
MAX_CONTEXT_CLIPS="${MAX_CONTEXT_CLIPS:-6}"

# ---- run 名 / 续训钩子 ----
RUN_NAME="${RUN_NAME:-}"                 # 空 → train_v6 用 default_run_name('latentconcat_lora')
RESUME_FROM="${RESUME_FROM:-}"           # 空=fresh（默认）；非空=从该 epoch 目录续训

# 默认 0-5 共 6 卡（ZeRO-3 需 6 卡分摊 22.7B；项目记忆：4 卡会 OOM/死锁）。
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5}"

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---- 环境护栏（项目记忆，必须内置）----
export TMPDIR=/tmp                                          # 防 pymp/torchelastic 孤儿落进 repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # 缓解显存碎片
export CUDA_DEVICE_ORDER=PCI_BUS_ID                         # 卡号按 PCI 总线序
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

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

ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/v6/accelerate_v6.yaml"
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[ERROR] accelerate 配置文件不存在：${ACCEL_CONFIG}" >&2; exit 1
fi

TRAIN_SCRIPT="${PROJECT_ROOT}/src/pipeline/v6/train_v6.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] 训练脚本不存在：${TRAIN_SCRIPT}" >&2; exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

# ---------- 参数数组 ----------
TRAIN_ARGS=(
    --ckpt_dir                    "${CKPT_DIR}"
    --dataset_dir                 "${DATASET_DIR}"
    --phase                       "${PHASE}"
    --lora_rank                   "${LORA_RANK}"
    --lora_alpha                  "${LORA_ALPHA}"
    --lora_dropout                "${LORA_DROPOUT}"
    --lora_targets                "${LORA_TARGETS}"
    --num_anchor_frames           "${NUM_ANCHOR_FRAMES}"
    --learning_rate               "${LEARNING_RATE}"
    --weight_decay                "${WEIGHT_DECAY}"
    --num_epochs                  "${NUM_EPOCHS}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_grad_norm               "${MAX_GRAD_NORM}"
    --save_every_n_epochs         "${SAVE_EVERY_N_EPOCHS}"
    --dataset_repeat              "${DATASET_REPEAT}"
    --num_frames                  "${NUM_FRAMES}"
    --height                      "${HEIGHT}"
    --width                       "${WIDTH}"
    --max_context_clips           "${MAX_CONTEXT_CLIPS}"
)

# run_name 留空 → 交给 train_v6 用 default_run_name；非空才透传
if [ -n "${RUN_NAME}" ]; then
    TRAIN_ARGS+=(--run_name "${RUN_NAME}")
fi
# 续训钩子：默认 fresh（RESUME_FROM 空）
if [ -n "${RESUME_FROM}" ]; then
    TRAIN_ARGS+=(--resume "${RESUME_FROM}")
fi

echo "====================================================="
echo "  LingBot-World Memory v6 训练启动（latent-concat + LoRA）"
echo "  PHASE              : ${PHASE}"
echo "  LORA_RANK           : ${LORA_RANK}"
echo "  LORA_ALPHA          : ${LORA_ALPHA}"
echo "  LORA_TARGETS        : ${LORA_TARGETS}"
echo "  NUM_ANCHOR_FRAMES   : ${NUM_ANCHOR_FRAMES}"
echo "  NUM_EPOCHS          : ${NUM_EPOCHS}"
echo "  LEARNING_RATE       : ${LEARNING_RATE}"
echo "  RUN_NAME            : ${RUN_NAME:-<default_run_name>}"
echo "  RESUME_FROM         : ${RESUME_FROM:-<fresh, 从头训>}"
echo "  DATASET_DIR         : ${DATASET_DIR}"
echo "  OUTPUT_ROOT         : ${OUTPUT_ROOT:-<paths.py 默认>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS            : ${NUM_GPUS}"
echo "  ACCEL_CONFIG        : ${ACCEL_CONFIG}"
echo "  LOG_FILE            : ${LOG_FILE}"
echo "====================================================="

CMD=(
    accelerate launch
    --config_file "${ACCEL_CONFIG}"
    --num_processes "${NUM_GPUS}"
    "${TRAIN_SCRIPT}"
    "${TRAIN_ARGS[@]}"
)

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; _EXIT="${PIPESTATUS[0]}"
if [ "${_EXIT}" -ne 0 ]; then
    echo "[ERROR] v6 训练失败，退出码 ${_EXIT}" >&2
    exit "${_EXIT}"
fi

echo ""
echo "===== v6 训练完成（产出走 paths.py：OUTPUT_ROOT/v6/train/<run_name>/）====="
