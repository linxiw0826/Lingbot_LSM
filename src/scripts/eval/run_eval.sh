#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_eval.sh
# ============================================================

# [选项A] Demo 模式：已有视频文件夹（非空 → 跳过推理，直接评分）
VIDEO_DIR="/home/nvme02/wlx/Memory/outputs/inference/v4_stage1_dual_epoch_5_n5"

# [选项B] Full pipeline 模式（VIDEO_DIR 留空时生效；命名逻辑与 run_infer_v3.sh 对齐）
CKPT_DIR=""                  # 基础模型目录（必填，full pipeline 模式）
FT_MODEL_DIR=""              # 全参微调 / dual-low 目录（如 .../low_noise_model/epoch_5）
FT_HIGH_MODEL_DIR=""         # dual-high 目录（有值时视为 dual 模式，FT_MODEL_DIR 也必须填）
USE_MEMORY=true              # 是否启用 Memory Bank
LORA_PATH=""                 # LoRA 权重路径（可选）
NUM_CLIPS=5                  # 连续推理 clip 数（影响 run_name 命名）
INFER_SCRIPT="src/pipeline/v3/infer_v3.py"  # 推理脚本（相对项目根目录）
FORCE=true                  # true = 忽略 all_runs.csv 缓存，强制重新评测

# 推理参数（full pipeline 和 demo 均用于 VBench）
TEST_IMAGES_DIR="eval_data/images/"       # 测试图片目录（full pipeline 推理用）
TEST_TRAJ_DIR="eval_data/trajectories/"  # 相机轨迹目录（full pipeline 推理用）
FRAME_NUM=81
SIZE="480*832"
PROMPT="First-person view of CS:GO competitive gameplay"

# VBench 评测维度（对应论文 Table 2 的 6 个维度）
DIMENSIONS="imaging_quality aesthetic_quality dynamic_degree motion_smoothness temporal_flickering subject_consistency"

CUDA_VISIBLE_DEVICES="1,2,3,4,5"   # 使用哪些 GPU

# ============================================================
# 以下内容通常无需修改
# ============================================================

export CUDA_VISIBLE_DEVICES

# ---------- 路径计算 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

EVAL_SCRIPT="${PROJECT_ROOT}/src/pipeline/eval/eval_vbench.py"

# ---------- RUN_NAME 推导（与 run_infer_v3.sh 命名逻辑对齐）----------
if [ -n "${VIDEO_DIR}" ]; then
    # demo 模式：取视频文件夹 basename
    RUN_NAME="$(basename "${VIDEO_DIR}")"
else
    # full pipeline 模式：与 run_infer_v3.sh 完全相同的 EXP_NAME 逻辑
    if [ -n "${FT_HIGH_MODEL_DIR}" ]; then
        _train_dir="$(basename "$(dirname "$(dirname "${FT_MODEL_DIR}")")")"
        _epoch="$(basename "${FT_MODEL_DIR}")"
        _base_exp="${_train_dir}_${_epoch}"
    elif [ -n "${FT_MODEL_DIR}" ]; then
        _train_dir="$(basename "$(dirname "${FT_MODEL_DIR}")")"
        _epoch="$(basename "${FT_MODEL_DIR}")"
        _base_exp="${_train_dir}_${_epoch}"
    else
        _base_exp="baseline"
    fi

    if [ -n "${FT_HIGH_MODEL_DIR}" ] || [ -n "${FT_MODEL_DIR}" ]; then
        if [ "${USE_MEMORY}" = "true" ]; then
            _base_exp="${_base_exp}_mem"
        else
            _base_exp="${_base_exp}_nomem"
        fi
    else
        if [ "${USE_MEMORY}" = "true" ]; then
            _base_exp="baseline_mem"
        else
            _base_exp="baseline"
        fi
    fi
    RUN_NAME="${_base_exp}_n${NUM_CLIPS}"
fi

# ---------- 推理结果根目录（与 run_infer_v3.sh 一致）----------
MEMORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
OUTPUT_BASE="${MEMORY_ROOT}/outputs"

# ---------- OUTPUT_DIR 使用 RUN_NAME ----------
OUTPUT_DIR="${OUTPUT_BASE}/eval_vbench/${RUN_NAME}"

# ---------- EVAL_SCRIPT 存在性检查 ----------
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "[ERROR] eval_vbench.py 不存在：${EVAL_SCRIPT}" >&2
    echo "请确认项目目录结构完整。" >&2
    exit 1
fi

# ---------- 日志目录 & 日志文件 ----------
LOG_DIR="${PROJECT_ROOT}/logs/run_eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"

# ---------- OUTPUT_DIR 创建 ----------
mkdir -p "${OUTPUT_DIR}"

# ---------- TEST_IMAGES_DIR 路径解析 ----------
if [[ "${TEST_IMAGES_DIR}" != /* ]]; then
    TEST_IMAGES_DIR="${PROJECT_ROOT}/${TEST_IMAGES_DIR}"
fi

# ---------- TEST_TRAJ_DIR 路径解析 ----------
if [[ "${TEST_TRAJ_DIR}" != /* ]]; then
    TEST_TRAJ_DIR="${PROJECT_ROOT}/${TEST_TRAJ_DIR}"
fi

echo "====================================================="
echo "  LingBot-World Memory Enhancement — VBench 评测"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  RUN_NAME        : ${RUN_NAME}"
echo "  VIDEO_DIR       : ${VIDEO_DIR}"
echo "  TEST_IMAGES_DIR : ${TEST_IMAGES_DIR}"
echo "  TEST_TRAJ_DIR   : ${TEST_TRAJ_DIR}"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "  LOG_FILE        : ${LOG_FILE}"
echo "====================================================="

# ---------- 拼接评测命令 ----------
# shellcheck disable=SC2206
DIMENSIONS_ARRAY=( ${DIMENSIONS} )

CMD=(
    python "${EVAL_SCRIPT}"
    --run_name         "${RUN_NAME}"
    --test_images_dir  "${TEST_IMAGES_DIR}"
    --test_traj_dir    "${TEST_TRAJ_DIR}"
    --output_dir       "${OUTPUT_DIR}"
    --dimensions       "${DIMENSIONS_ARRAY[@]}"
    --frame_num        "${FRAME_NUM}"
    --size             "${SIZE}"
    --prompt           "${PROMPT}"
)

if [ -n "${VIDEO_DIR}" ]; then
    CMD+=(--video_dir "${VIDEO_DIR}")
else
    # full pipeline 参数
    CMD+=(--infer_script "${PROJECT_ROOT}/${INFER_SCRIPT}")
    if [ -n "${CKPT_DIR}" ];          then CMD+=(--ckpt_dir "${CKPT_DIR}"); fi
    if [ -n "${FT_MODEL_DIR}" ];      then CMD+=(--ft_model_dir "${FT_MODEL_DIR}"); fi
    if [ -n "${FT_HIGH_MODEL_DIR}" ]; then CMD+=(--ft_high_model_dir "${FT_HIGH_MODEL_DIR}"); fi
    if [ "${USE_MEMORY}" = "true" ];  then CMD+=(--use_memory); fi
    if [ -n "${LORA_PATH}" ];         then CMD+=(--lora_path "${LORA_PATH}"); fi
fi

if [ "${FORCE}" = "true" ]; then CMD+=(--force); fi

echo "执行命令："
echo "${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; exit "${PIPESTATUS[0]}"
