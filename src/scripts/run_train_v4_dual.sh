#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash run_train_v4_dual.sh
# ============================================================
# 可用环境变量覆盖（无需编辑本文件）：
#   CUDA_VISIBLE_DEVICES / RESUME_FROM_LOW / RESUME_FROM_HIGH / DATASET_DIR / PHASE / NUM_EPOCHS
# 示例（2 卡从头训）：CUDA_VISIBLE_DEVICES=0,1 bash src/scripts/run_train_v4_dual.sh
# 默认从头训练（RESUME_FROM_LOW 为空）；如需续训：RESUME_FROM_LOW=.../epoch_N bash run_train_v4_dual.sh
# ============================================================
# 实验配置 ⑦：v4 数据 + Stage1 + 双模型，Innovation 6-10
# 方法：Method B 多 clip 顺序训练，ThreeTierMemoryBank
#
# 训练流程：sequential（先 low_noise_model，完成后自动运行 high_noise_model）
# 输出目录：OUTPUT_DIR/low_noise_model/ 和 OUTPUT_DIR/high_noise_model/
#
# 与 run_train_v3_dual.sh 的差异：
#   - 调用 train_v4_stage1_dual.py（Innovation 6-10）
#   - 移除 --num_context_clips，改为 --max_context_clips（Stochastic N-clip，Innovation 6）
#   - 新增 --context_drop_p_max（Context Drop-off，Innovation 7）
#   - 新增 --visual_fusion_alpha（Visual Feature Fusion，Innovation 9）
#   - LONG_CAP 默认值改为 32（v4 扩容）

CKPT_DIR="/home/nvme02/lingbot-world/models/lingbot-world-base-act"
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_all46}"            # v4 数据集目录
PHASE="${PHASE:-exp}"                  # 训练阶段："exp"（ep01-11）或 "full"（ep01-46）
OUTPUT_BASE="/home/nvme02/wlx/Memory/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/train/v4_stage1_dual"
# RESUME_FROM_LOW=""         # low 模型断点续训路径（留空从头开始）
RESUME_FROM_LOW="${RESUME_FROM_LOW:-}"
RESUME_FROM_HIGH="${RESUME_FROM_HIGH:-}"        # high 模型断点续训路径（留空从头开始）

LORA_RANK=0                # LoRA rank：0=全参微调；32/64=LoRA微调
LORA_TARGET_MODULES=""     # LoRA目标模块（留空自动检测）

NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE=1e-4
LR_DIT=1e-5
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION_STEPS=8    # 6 GPU × 8 accum = effective batch 48
MAX_GRAD_NORM=1.0
SAVE_EVERY_N_EPOCHS=1
KEEP_LAST_N_CHECKPOINTS=2     # 只保留最近 2 个 checkpoint，第 3 个存下来时自动删第 1 个
DATASET_REPEAT=1
NUM_FRAMES=81
HEIGHT=480
WIDTH=832
NFP_LOSS_WEIGHT=0.1

# ---- v4 新增：Stochastic N-clip + Context Drop-off + Visual Feature Fusion ----
MAX_CONTEXT_CLIPS=6        # v4 Stochastic N-clip 上限（N ~ Uniform(2, max_context_clips)）
CONTEXT_DROP_P_MAX=0.3     # v4 Context Drop-off 最大丢弃比例（Uniform(0, p_max)，Innovation 7）
VISUAL_FUSION_ALPHA=0.7    # v4 Visual Feature Fusion pose 权重（Innovation 9，default=0.7）

# ---- ThreeTierMemoryBank 超参数（保持 v4 默认值）----
SHORT_CAP=1                # ShortTermBank 容量（FIFO，保证 chunk 间衔接）
MEDIUM_CAP=8               # MediumTermBank 容量（高 surprise 帧 + age decay 淘汰）
LONG_CAP=32                # LongTermBank 容量（v4 扩容，v3 为 16）
SURPRISE_THRESHOLD=0.4     # Medium 写入下限（surprise >= threshold 才写入）
STABILITY_THRESHOLD=0.2    # Long stable 写入上限（surprise < threshold 视为 stable）
NOVELTY_THRESHOLD=0.7      # Long novelty 写入上限（max cosine_sim < threshold 视为 novel）
HALF_LIFE=10.0             # Medium age decay 半衰期（单位 chunk）
HYBRID_MEDIUM_K=3          # 混合检索预算中 Medium 层 top-K
HYBRID_LONG_K=2            # 混合检索预算中 Long 层 top-K
DUP_THRESHOLD=0.95         # cross-tier dedup 阈值（pose_emb cosine_sim > 阈值则去重）

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
    --phase                       "${PHASE}"
    --stage                       1
    --learning_rate               "${LEARNING_RATE}"
    --lr_dit                      "${LR_DIT}"
    --weight_decay                "${WEIGHT_DECAY}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_grad_norm               "${MAX_GRAD_NORM}"
    --save_every_n_epochs         "${SAVE_EVERY_N_EPOCHS}"
    --keep_last_n_checkpoints     "${KEEP_LAST_N_CHECKPOINTS}"
    --dataset_repeat              "${DATASET_REPEAT}"
    --num_frames                  "${NUM_FRAMES}"
    --height                      "${HEIGHT}"
    --width                       "${WIDTH}"
    --nfp_loss_weight             "${NFP_LOSS_WEIGHT}"
    --gradient_checkpointing
    --max_context_clips           "${MAX_CONTEXT_CLIPS}"
    --context_drop_p_max          "${CONTEXT_DROP_P_MAX}"
    --visual_fusion_alpha         "${VISUAL_FUSION_ALPHA}"
    --short_cap                   "${SHORT_CAP}"
    --medium_cap                  "${MEDIUM_CAP}"
    --long_cap                    "${LONG_CAP}"
    --surprise_threshold          "${SURPRISE_THRESHOLD}"
    --stability_threshold         "${STABILITY_THRESHOLD}"
    --novelty_threshold           "${NOVELTY_THRESHOLD}"
    --half_life                   "${HALF_LIFE}"
    --hybrid_medium_k             "${HYBRID_MEDIUM_K}"
    --hybrid_long_k               "${HYBRID_LONG_K}"
    --dup_threshold               "${DUP_THRESHOLD}"
)

if [ "${LORA_RANK}" -gt 0 ]; then
    TRAIN_ARGS+=(--lora_rank "${LORA_RANK}")
    if [ -n "${LORA_TARGET_MODULES}" ]; then
        TRAIN_ARGS+=(--lora_target_modules "${LORA_TARGET_MODULES}")
    fi
fi

TRAIN_SCRIPT="${PROJECT_ROOT}/src/pipeline/train_v4_stage1_dual.py"

echo "====================================================="
echo "  LingBot-World Memory Enhancement 双模型训练 v4 启动"
echo "  实验配置 ⑦：v4（Innovations 6-10）"
echo "  PHASE              : ${PHASE}"
echo "  MAX_CONTEXT_CLIPS   : ${MAX_CONTEXT_CLIPS}"
echo "  CONTEXT_DROP_P_MAX  : ${CONTEXT_DROP_P_MAX}"
echo "  VISUAL_FUSION_ALPHA : ${VISUAL_FUSION_ALPHA}"
echo "  ThreeTierMemoryBank: short=${SHORT_CAP} medium=${MEDIUM_CAP} long=${LONG_CAP}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS           : ${NUM_GPUS}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "  LOG_FILE           : ${LOG_FILE}"
echo "====================================================="

# ============================================================
# Step 1/2: 训练 low_noise_model（每 epoch 独立启动 accelerate launch）
# ============================================================
echo ""
echo "===== Step 1/2: 训练 low_noise_model（t < 0.947，per-epoch 子进程）====="

_LOW_RESUME="${RESUME_FROM_LOW}"
_LOW_START=1
if [ -n "${RESUME_FROM_LOW}" ]; then
    _ep=$(basename "${RESUME_FROM_LOW}" | sed 's/^epoch_//')
    if ! echo "${_ep}" | grep -qE '^[0-9]+$'; then
        echo "[ERROR] 无法从 RESUME_FROM_LOW 解析 epoch 编号，期望路径格式：.../epoch_N" >&2
        exit 1
    fi
    _LOW_START=$((_ep + 1))
fi

for _EPOCH in $(seq "${_LOW_START}" "${NUM_EPOCHS}"); do
    echo ""
    echo "----- low_noise_model Epoch ${_EPOCH}/${NUM_EPOCHS} -----"
    CMD_LOW=(
        accelerate launch "${COMMON_ARGS[@]}"
        "${TRAIN_SCRIPT}"
        "${TRAIN_ARGS[@]}"
        --num_epochs "${_EPOCH}"
        --model_type low
    )
    if [ -n "${_LOW_RESUME}" ]; then
        CMD_LOW+=(--resume "${_LOW_RESUME}")
    fi
    echo "执行命令（low epoch ${_EPOCH}）："
    echo "${CMD_LOW[*]}"
    echo ""
    "${CMD_LOW[@]}" 2>&1 | tee -a "${LOG_FILE}"; _EXIT="${PIPESTATUS[0]}"
    if [ "${_EXIT}" -ne 0 ]; then
        echo "[ERROR] low_noise_model Epoch ${_EPOCH} 训练失败，退出码 ${_EXIT}" >&2
        exit "${_EXIT}"
    fi
    _LOW_RESUME="${OUTPUT_DIR}/low_noise_model/epoch_${_EPOCH}"
done

echo ""
echo "===== low_noise_model 训练完成（${NUM_EPOCHS} epochs）====="

# ============================================================
# Step 2/2: 训练 high_noise_model（每 epoch 独立启动 accelerate launch）
# ============================================================
echo ""
echo "===== Step 2/2: 训练 high_noise_model（t >= 0.947，per-epoch 子进程）====="

_HIGH_RESUME="${RESUME_FROM_HIGH}"
_HIGH_START=1
if [ -n "${RESUME_FROM_HIGH}" ]; then
    _ep=$(basename "${RESUME_FROM_HIGH}" | sed 's/^epoch_//')
    if ! echo "${_ep}" | grep -qE '^[0-9]+$'; then
        echo "[ERROR] 无法从 RESUME_FROM_HIGH 解析 epoch 编号，期望路径格式：.../epoch_N" >&2
        exit 1
    fi
    _HIGH_START=$((_ep + 1))
fi

for _EPOCH in $(seq "${_HIGH_START}" "${NUM_EPOCHS}"); do
    echo ""
    echo "----- high_noise_model Epoch ${_EPOCH}/${NUM_EPOCHS} -----"
    CMD_HIGH=(
        accelerate launch "${COMMON_ARGS[@]}"
        "${TRAIN_SCRIPT}"
        "${TRAIN_ARGS[@]}"
        --num_epochs "${_EPOCH}"
        --model_type high
    )
    if [ -n "${_HIGH_RESUME}" ]; then
        CMD_HIGH+=(--resume "${_HIGH_RESUME}")
    fi
    echo "执行命令（high epoch ${_EPOCH}）："
    echo "${CMD_HIGH[*]}"
    echo ""
    "${CMD_HIGH[@]}" 2>&1 | tee -a "${LOG_FILE}"; _EXIT="${PIPESTATUS[0]}"
    if [ "${_EXIT}" -ne 0 ]; then
        echo "[ERROR] high_noise_model Epoch ${_EPOCH} 训练失败，退出码 ${_EXIT}" >&2
        exit "${_EXIT}"
    fi
    _HIGH_RESUME="${OUTPUT_DIR}/high_noise_model/epoch_${_EPOCH}"
done

echo ""
echo "===== 双模型训练完成 ====="
echo "  low_noise_model  → ${OUTPUT_DIR}/low_noise_model/"
echo "  high_noise_model → ${OUTPUT_DIR}/high_noise_model/"
