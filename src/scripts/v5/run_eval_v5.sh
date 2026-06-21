#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v5/run_eval_v5.sh
# ============================================================
# v5 三臂 bank_revisit 评测（off / oracle / random）。单进程、单卡，不用 accelerate。
# 产出走 paths.py：OUTPUT_ROOT/v5/eval/<run_name>/<tag>/；
#   跑完自动出 summary.md + 追加 INDEX.md。
# 判据：oracle 的 DINO 指标应 > off（说明模型能用上注入的记忆）。
#
# 必填：MEMORY_ENCODER_CKPT —— 训练产出的 memory_encoder.pth（train_v5 save_memory_encoder）。
#
# 可用环境变量覆盖（无需编辑本文件）：
#   CUDA_VISIBLE_DEVICES / MEMORY_ENCODER_CKPT / DATASET_DIR / METADATA
#   MODES / MAX_EPISODES / TAG / RUN_NAME / GRID / ENCODER_DEPTH / OUTPUT_ROOT
#
# 示例（指定权重，单卡 0）：
#   MEMORY_ENCODER_CKPT=/home/nvme02/wlx/Memory/outputs/v5/train/<run>/epoch_3/memory_encoder.pth \
#     bash src/scripts/v5/run_eval_v5.sh
# 示例（换卡）：
#   CUDA_VISIBLE_DEVICES=2 MEMORY_ENCODER_CKPT=.../memory_encoder.pth bash src/scripts/v5/run_eval_v5.sh
# ============================================================

# ---- 模型权重 ----
CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"     # lingbot-world 预训练权重
MEMORY_ENCODER_CKPT="${MEMORY_ENCODER_CKPT:-}"                          # 必填：训练好的 memory_encoder.pth

# ---- 数据 ----
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"   # 含重访的数据集根
METADATA="${METADATA:-metadata_verify_train.csv}"                       # 相对 dataset_dir 的 CSV
MAX_EPISODES="${MAX_EPISODES:-5}"                                       # 0=不限；>0 取前 N 个 episode

# ---- 三臂 / 注入 ----
MODES="${MODES:-off,oracle,random}"      # 注入臂子集
WEAKEN_FIRST_FRAME="zero"                # F-18 护栏：zero=置零中性灰（默认温和锚点）
# 注意：默认不加 --inject_high（low-only，对齐训练；high 用未训练 encoder 仅消融）。

# ---- v5 超参（留空 → eval 从 training_metadata 自动采纳，与训练一致）----
GRID="${GRID:-}"                         # 空=从 training_metadata 采纳
ENCODER_DEPTH="${ENCODER_DEPTH:-}"       # 空=从 training_metadata 采纳

# ---- 产出 ----
RUN_NAME="${RUN_NAME:-}"                 # 空 → default_run_name('v5_eval')
TAG="${TAG:-bank_revisit}"              # eval 场景 tag

# eval 单进程，默认单卡（可用 CUDA_VISIBLE_DEVICES 覆盖）
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
if [ -z "${DATASET_DIR}" ]; then
    echo "[ERROR] DATASET_DIR 未设置" >&2; _err=1
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

EVAL_SCRIPT="${PROJECT_ROOT}/src/pipeline/v5/eval_v5.py"
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[ERROR] 评测脚本不存在：${EVAL_SCRIPT}" >&2; exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

# ---------- 参数数组 ----------
EVAL_ARGS=(
    --ckpt_dir              "${CKPT_DIR}"
    --memory_encoder_ckpt   "${MEMORY_ENCODER_CKPT}"
    --dataset_dir           "${DATASET_DIR}"
    --metadata              "${METADATA}"
    --modes                 "${MODES}"
    --max_episodes          "${MAX_EPISODES}"
    --weaken_first_frame    "${WEAKEN_FIRST_FRAME}"
    --tag                   "${TAG}"
)

# run_name 留空 → eval 用 default_run_name
if [ -n "${RUN_NAME}" ]; then
    EVAL_ARGS+=(--run_name "${RUN_NAME}")
fi
# grid/encoder_depth：仅当显式给出才透传（空=从 training_metadata 自动采纳）
if [ -n "${GRID}" ]; then
    EVAL_ARGS+=(--grid "${GRID}")
fi
if [ -n "${ENCODER_DEPTH}" ]; then
    EVAL_ARGS+=(--encoder_depth "${ENCODER_DEPTH}")
fi
# 默认 low-only：不加 --inject_high。

echo "====================================================="
echo "  LingBot-World Memory v5 评测启动（三臂 bank_revisit）"
echo "  MEMORY_ENCODER_CKPT : ${MEMORY_ENCODER_CKPT}"
echo "  DATASET_DIR         : ${DATASET_DIR}"
echo "  METADATA            : ${METADATA}"
echo "  MODES               : ${MODES}"
echo "  MAX_EPISODES        : ${MAX_EPISODES}"
echo "  WEAKEN_FIRST_FRAME  : ${WEAKEN_FIRST_FRAME}"
echo "  GRID                : ${GRID:-<从 training_metadata 采纳>}"
echo "  ENCODER_DEPTH       : ${ENCODER_DEPTH:-<从 training_metadata 采纳>}"
echo "  TAG                 : ${TAG}"
echo "  RUN_NAME            : ${RUN_NAME:-<default_run_name>}"
echo "  OUTPUT_ROOT         : ${OUTPUT_ROOT:-<paths.py 默认>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  LOG_FILE            : ${LOG_FILE}"
echo "  判据：oracle DINO > off"
echo "====================================================="

CMD=(
    python "${EVAL_SCRIPT}"
    "${EVAL_ARGS[@]}"
)

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; _EXIT="${PIPESTATUS[0]}"
if [ "${_EXIT}" -ne 0 ]; then
    echo "[ERROR] v5 评测失败，退出码 ${_EXIT}" >&2
    exit "${_EXIT}"
fi

echo ""
echo "===== v5 评测完成（产出走 paths.py：OUTPUT_ROOT/v5/eval/<run_name>/<tag>/）====="
echo "===== 自动产出 summary.md + 追加 INDEX.md；判据：oracle DINO > off ====="
