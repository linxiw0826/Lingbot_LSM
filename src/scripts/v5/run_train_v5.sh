#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v5/run_train_v5.sh
# ============================================================
# v5：in-context KV 注入（MemoryEncoder，A_frozen），DiT 冻结只训 encoder。
# 产出走 paths.py：OUTPUT_ROOT/v5/train/<run_name>/（--run_name 留空 → default_run_name）。
#
# 可用环境变量覆盖（无需编辑本文件）：
#   TRAIN_GPUS / DATASET_DIR / PHASE / NUM_EPOCHS / GRID / ENCODER_DEPTH
#   MEMORY_LAYERS / LEARNING_RATE / RUN_NAME / RESUME_FROM / OUTPUT_ROOT
#
# 示例（6 卡从头训，默认）：
#   bash src/scripts/v5/run_train_v5.sh
# 示例（指定 6 卡 + 自定 run_name）：
#   TRAIN_GPUS=0,1,2,3,4,5 RUN_NAME=inctxkv_A_frozen_v5verify bash src/scripts/v5/run_train_v5.sh
# 示例（输出改到别处，避免覆盖基线 ckpt）：
#   OUTPUT_ROOT=/home/nvme02/wlx/Memory/outputs_exp bash src/scripts/v5/run_train_v5.sh
# 示例（续训，从某 epoch 目录恢复）：
#   RESUME_FROM=/home/nvme02/wlx/Memory/outputs/v5/train/<run>/epoch_2 bash src/scripts/v5/run_train_v5.sh
#
# 默认 fresh 训练（不 resume）；如需续训设置 RESUME_FROM。
# ============================================================

# ---- 模型 / 数据 / 阶段 ----
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"                              # lingbot-world 预训练权重
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"           # 默认=revisit verify 集（同 v4，可 env 覆盖）
PHASE="${PHASE:-verify}"                 # 训练阶段："verify" / "exp" / "full"

# ---- v5 模型超参 ----
GRID="${GRID:-16}"                       # MemoryEncoder 每帧 grid×grid token
ENCODER_DEPTH="${ENCODER_DEPTH:-1}"      # MemoryEncoder 残差块层数
MEMORY_LAYERS="${MEMORY_LAYERS:-}"       # 注入层索引（逗号分隔，如 '0,10,20,39'）；空=全部 block

# ---- 训练超参（对齐 v4：1e-4 量级）----
NUM_EPOCHS="${NUM_EPOCHS:-3}"            # 验证阶段 3 epoch
LEARNING_RATE="${LEARNING_RATE:-1e-4}"   # 回退稳定值（同 v4 low）
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=4
MAX_GRAD_NORM=1.0
SAVE_EVERY_N_EPOCHS=1
DATASET_REPEAT=1
NUM_FRAMES=81
HEIGHT=480
WIDTH=832

# ---- run 名 / 续训钩子 ----
RUN_NAME="${RUN_NAME:-}"                 # 空 → train_v5 用 default_run_name('inctxkv_A_frozen')
RESUME_FROM="${RESUME_FROM:-}"           # 空=fresh（默认）；非空=从该 epoch 目录续训

# 默认 0-5 共 6 卡（ZeRO-3 需 6 卡分摊 22.7B；项目记忆：4 卡会 OOM/死锁）。
# 用专属变量 TRAIN_GPUS 覆盖卡组（不用 CUDA_VISIBLE_DEVICES，避免 shell 残留单卡值误覆盖）。
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5}"

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---- 环境护栏（项目记忆，必须内置）----
export TMPDIR=/tmp                                          # 防 pymp/torchelastic 孤儿落进 repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # 缓解显存碎片
export CUDA_DEVICE_ORDER=PCI_BUS_ID                         # 卡号按 PCI 总线序，与 TRAIN_GPUS 对齐
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

ACCEL_CONFIG="${PROJECT_ROOT}/src/configs/v5/accelerate_v5.yaml"
if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "[ERROR] accelerate 配置文件不存在：${ACCEL_CONFIG}" >&2; exit 1
fi

TRAIN_SCRIPT="${PROJECT_ROOT}/src/pipeline/v5/train_v5.py"
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
    --grid                        "${GRID}"
    --encoder_depth               "${ENCODER_DEPTH}"
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
)

# run_name 留空 → 交给 train_v5 用 default_run_name；非空才透传
if [ -n "${RUN_NAME}" ]; then
    TRAIN_ARGS+=(--run_name "${RUN_NAME}")
fi
# 注入层：仅当非空才透传（空=全部 block）
if [ -n "${MEMORY_LAYERS}" ]; then
    TRAIN_ARGS+=(--memory_layers "${MEMORY_LAYERS}")
fi
# 续训钩子：默认 fresh（RESUME_FROM 空）
if [ -n "${RESUME_FROM}" ]; then
    TRAIN_ARGS+=(--resume "${RESUME_FROM}")
fi

echo "====================================================="
echo "  LingBot-World Memory v5 训练启动（in-context KV，A_frozen）"
echo "  PHASE              : ${PHASE}"
echo "  GRID               : ${GRID}"
echo "  ENCODER_DEPTH       : ${ENCODER_DEPTH}"
echo "  MEMORY_LAYERS       : ${MEMORY_LAYERS:-<none, 全部 block>}"
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
    echo "[ERROR] v5 训练失败，退出码 ${_EXIT}" >&2
    exit "${_EXIT}"
fi

echo ""
echo "===== v5 训练完成（产出走 paths.py：OUTPUT_ROOT/v5/train/<run_name>/）====="
