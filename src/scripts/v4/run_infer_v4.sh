#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_infer_v4.sh
# ============================================================
# v4 推理（Innovations 9+10）：ThreeTierMemoryBank 默认启用，12 clip 连续推理
#
# 与 run_infer_v3.sh 的差异：
#   - 调用 infer_v4.py（tier_ids 传递 + visual_emb query fusion）
#   - LONG_CAP 默认值改为 32（v4 扩容）
#   - 新增 VISUAL_FUSION_ALPHA（Visual Feature Fusion，Innovation 9）

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"   # 基础模型目录（必填）
IMAGE="/home/nvme02/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected/Ep000027_p0007_77s_86s_two_windows_revisit/image.jpg"          # 每次推理前填写：<clip目录>/image.jpg
ACTION_PATH="/home/nvme02/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected/Ep000027_p0007_77s_86s_two_windows_revisit"    # 每次推理前填写：<clip目录>
PROMPT="First-person CS:GO competitive gameplay"

# 微调权重（三选一，留空则跑 baseline）
LORA_PATH=""             # LoRA 权重路径（lora_weights.pth）
FT_MODEL_DIR="/home/nvme02/wlx/Memory/outputs/train/v4_stage1_dual/low_noise_model/epoch_5"
                         # 全参微调 / dual-low 目录（如 .../train_v4_stage1_dual/low_noise_model/epoch_5）
FT_HIGH_MODEL_DIR="/home/nvme02/wlx/Memory/outputs/train/v4_stage1_dual/high_noise_model/epoch_5"
                         # dual-high 目录（如 .../train_v4_stage1_dual/high_noise_model/epoch_5）
                         # FT_HIGH_MODEL_DIR 有值时视为 dual 模式，此时 FT_MODEL_DIR 也必须填写

# 推理参数
FRAME_NUM=81             # 单 clip 帧数（81 帧 @ 16fps ≈ 5 秒）
NUM_CLIPS=5             # v4 默认 12 clip 连续推理
                         # 确保 ACTION_PATH 内 action.npy 帧数 >= FRAME_NUM * NUM_CLIPS
SAMPLE_STEPS=40
SAMPLE_SHIFT=10.0
GUIDE_SCALE=5.0
SIZE="480*832"

# ---- ThreeTierMemoryBank 超参数（保持 v4 默认值；通常无需修改）----
SHORT_CAP=1              # ShortTermBank 容量（FIFO，保证 chunk 间衔接）
MEDIUM_CAP=8             # MediumTermBank 容量（高 surprise 帧 + age decay 淘汰）
LONG_CAP=32              # LongTermBank 容量（v4 扩容，v3 为 16）
SURPRISE_THRESHOLD=0.4   # Medium 写入下限
STABILITY_THRESHOLD=0.2  # Long stable 写入上限
NOVELTY_THRESHOLD=0.7    # Long novelty 写入上限
HALF_LIFE=10.0           # Medium age decay 半衰期（单位 chunk）
HYBRID_MEDIUM_K=3        # 混合检索预算中 Medium 层 top-K
HYBRID_LONG_K=2          # 混合检索预算中 Long 层 top-K
DUP_THRESHOLD=0.95       # cross-tier dedup 阈值

VISUAL_FUSION_ALPHA=0.7    # v4 Visual Feature Fusion pose 权重（Innovation 9，default=0.7）
USE_MEMORY=true            # 启用 ThreeTierMemoryBank（false 跑 baseline 对比）

OUTPUT_BASE="/home/nvme02/wlx/Memory/outputs"   # 推理结果根目录
CUDA_VISIBLE_DEVICES="1,2,3,4,5"               # 5 卡 Ulysses SP（40 heads ÷ 5 = 8）

# ============================================================
# 以下内容通常无需修改
# ============================================================

export CUDA_VISIBLE_DEVICES

# ---------- GPU 计数 ----------
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)

# Wan14B 固定 40 个 attention heads；Ulysses SP 要求 num_heads % GPU数 == 0
# memory 模式下多卡需要整除检查（baseline 模式单卡亦可跑）
if [ "${NUM_GPUS}" -gt 1 ]; then
    if [ $((40 % NUM_GPUS)) -ne 0 ]; then
        echo "[ERROR] Ulysses SP：${NUM_GPUS} 个 GPU 不能整除 Wan14B 的 40 个 attention heads（余数 $((40 % NUM_GPUS))）。" >&2
        echo "[ERROR] 请将 CUDA_VISIBLE_DEVICES 的 GPU 数量改为 40 的因数，推荐：1 / 2 / 4 / 5 / 8 / 10" >&2
        exit 1
    fi
fi

# ---------- FT 权重提示 ----------
if [ -z "${FT_MODEL_DIR}" ] && [ -z "${FT_HIGH_MODEL_DIR}" ] && [ -z "${LORA_PATH}" ]; then
    echo "[INFO] 未设置 FT 权重（FT_MODEL_DIR/FT_HIGH_MODEL_DIR/LORA_PATH 均为空），将使用基础模型推理" >&2
    echo "[INFO] 如需加载训练权重，请设置 FT_MODEL_DIR 或 FT_HIGH_MODEL_DIR" >&2
fi

# ---------- 路径检查 ----------
_err=0
if [ -z "${CKPT_DIR}" ];    then echo "[ERROR] CKPT_DIR 未设置"    >&2; _err=1; fi
if [ -z "${IMAGE}" ];       then echo "[ERROR] IMAGE 未设置"        >&2; _err=1; fi
if [ -z "${ACTION_PATH}" ]; then echo "[ERROR] ACTION_PATH 未设置"  >&2; _err=1; fi
# dual 模式：FT_HIGH_MODEL_DIR 非空时 FT_MODEL_DIR 也必须填写
if [ -n "${FT_HIGH_MODEL_DIR}" ] && [ -z "${FT_MODEL_DIR}" ]; then
    echo "[ERROR] dual 模式下 FT_MODEL_DIR 未设置（FT_HIGH_MODEL_DIR 有值时必须同时填写 FT_MODEL_DIR）" >&2; _err=1
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

# ---------- 路径计算 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------- 自动推导实验名 → 输出目录 ----------
# 规则：
#   dual  模式（FT_HIGH_MODEL_DIR 非空）: EXP_NAME = <训练dir>_<epoch>
#   single 模式（只有 FT_MODEL_DIR）     : EXP_NAME = <训练dir>_<epoch>
#   baseline（均为空）                  : EXP_NAME = baseline
if [ -n "${FT_HIGH_MODEL_DIR}" ]; then
    _train_dir="$(basename "$(dirname "$(dirname "${FT_MODEL_DIR}")")")"
    _epoch="$(basename "${FT_MODEL_DIR}")"
    EXP_NAME="${_train_dir}_${_epoch}"
elif [ -n "${FT_MODEL_DIR}" ]; then
    _train_dir="$(basename "$(dirname "${FT_MODEL_DIR}")")"
    _epoch="$(basename "${FT_MODEL_DIR}")"
    EXP_NAME="${_train_dir}_${_epoch}"
else
    EXP_NAME="baseline"
fi

EXP_NAME="${EXP_NAME}_n${NUM_CLIPS}"

# clip 名称取自 ACTION_PATH 目录名
CLIP_NAME="$(basename "${ACTION_PATH}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_FILE="${OUTPUT_BASE}/inference/${EXP_NAME}/${CLIP_NAME}_v4_${TIMESTAMP}.mp4"

# ---------- 日志目录 & 日志文件 ----------
LOG_DIR="${PROJECT_ROOT}/logs/$(basename "${BASH_SOURCE[0]}" .sh)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"

mkdir -p "$(dirname "${SAVE_FILE}")"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 推理 v4 启动"
echo "  Innovations 9+10（visual fusion + tier embedding）"
echo "  NUM_CLIPS: ${NUM_CLIPS} | NUM_GPUS: ${NUM_GPUS} | USE_MEMORY: ${USE_MEMORY}"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  CLIP_NAME  : ${CLIP_NAME}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  IMAGE      : ${IMAGE}"
echo "  ACTION_PATH: ${ACTION_PATH}"
echo "  NUM_CLIPS  : ${NUM_CLIPS}"
echo "  VISUAL_FUSION_ALPHA : ${VISUAL_FUSION_ALPHA}"
echo "  SAVE_FILE  : ${SAVE_FILE}"
echo "  LOG_FILE   : ${LOG_FILE}"
echo "====================================================="

# ---------- 拼接推理命令 ----------
CMD=(
    python "${PROJECT_ROOT}/src/pipeline/v4/infer_v4.py"
    --ckpt_dir           "${CKPT_DIR}"
    --image              "${IMAGE}"
    --action_path        "${ACTION_PATH}"
    --save_file          "${SAVE_FILE}"
    --prompt             "${PROMPT}"
    --frame_num          "${FRAME_NUM}"
    --num_clips          "${NUM_CLIPS}"
    --sample_steps       "${SAMPLE_STEPS}"
    --sample_shift       "${SAMPLE_SHIFT}"
    --guide_scale        "${GUIDE_SCALE}"
    --size               "${SIZE}"
    --short_cap          "${SHORT_CAP}"
    --medium_cap         "${MEDIUM_CAP}"
    --long_cap           "${LONG_CAP}"
    --surprise_threshold "${SURPRISE_THRESHOLD}"
    --stability_threshold "${STABILITY_THRESHOLD}"
    --novelty_threshold  "${NOVELTY_THRESHOLD}"
    --half_life          "${HALF_LIFE}"
    --hybrid_medium_k    "${HYBRID_MEDIUM_K}"
    --hybrid_long_k      "${HYBRID_LONG_K}"
    --dup_threshold      "${DUP_THRESHOLD}"
    --visual_fusion_alpha "${VISUAL_FUSION_ALPHA}"
)

# 可选：use_memory 开关
if [ "${USE_MEMORY}" = "true" ]; then
    CMD+=(--use_memory)
fi

# 可选：LoRA 权重
if [ -n "${LORA_PATH}" ]; then
    CMD+=(--lora_path "${LORA_PATH}")
fi

# 可选：全参微调 / dual 模型目录
if [ -n "${FT_MODEL_DIR}" ]; then
    CMD+=(--ft_model_dir "${FT_MODEL_DIR}")
fi
if [ -n "${FT_HIGH_MODEL_DIR}" ]; then
    CMD+=(--ft_high_model_dir "${FT_HIGH_MODEL_DIR}")
fi

# ---------- 启动方式：多卡用 torchrun，单卡用 python ----------
if [ "${NUM_GPUS}" -gt 1 ]; then
    CMD+=(--ulysses_size "${NUM_GPUS}")
    CMD+=(--t5_fsdp)          # 多卡时对 T5 做 FSDP 分片，避免每卡复制全量 T5（~22 GiB）
    # 动态选取空闲端口，避免 EADDRINUSE
    MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    # 替换脚本中第一个元素 python → torchrun
    LAUNCH=(
        torchrun
        --nproc_per_node "${NUM_GPUS}"
        --master_port "${MASTER_PORT}"
    )
    # CMD 第一个元素是 "python"，第二个是脚本路径
    # 重新构造：去掉 python，用 torchrun + 脚本路径
    FULL_CMD=("${LAUNCH[@]}" "${CMD[@]:1}")
else
    FULL_CMD=("${CMD[@]}")
fi

echo "执行命令："
echo "${FULL_CMD[*]}"
echo ""

"${FULL_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; exit "${PIPESTATUS[0]}"
