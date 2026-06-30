#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v5/run_render_qual_sharded.sh
# ============================================================
# v5-KV 理想注入【定性渲染】（S-V4 / Step 43 定性部分）—— **6 卡分片版**。
# 复用 ideal_inject_diag.py 的 --render_qual 路径：对每个 revisit case 渲一条完整多 clip
# 自回归长视频（off vs ideal_A），不算 DINO / 不判 GO/NO-GO，纯出视频供眼看「回到旧地」的
# 重访 clip（最后一个 clip）。把 revisit case 全集（先 detect 完整列表，再在切分前应用
# --max_cases 全局上限）按 **case 全局序号取模** 切成 NUM_SHARDS 份，每张卡渲若干 case 的长视频
# （不同 --tag → 独立 run_dir，无写冲突）。**定性不合并、不判决**（与定量分片版的区别）。
#
# 产出布局（走 paths.py，与定量分片版一致）：
#   OUTPUT_ROOT/v5/eval/<run_name>/<tag>_s0/videos/<ep>/q<query>/long_video_off.mp4
#   OUTPUT_ROOT/v5/eval/<run_name>/<tag>_s0/videos/<ep>/q<query>/long_video_ideal_A.mp4
#   ...（s1..s5 同构）
#
# 可用环境变量覆盖（无需编辑本文件）：
#   CKPT_DIR / MEMORY_ENCODER_CKPT / DATASET_DIR / METADATA
#   QUAL_ARMS / MAX_CASES / NUM_CLIPS_QUAL / FRAME_NUM / TAG / RUN_NAME
#   NUM_SHARDS / DIAG_GPUS / WEAKEN_FIRST_FRAME / INJECT_HIGH
#   PROMPT / SIZE / SAMPLE_STEPS / SAMPLE_SHIFT / GUIDE_SCALE / SEED / FPS
#   GRID / ENCODER_DEPTH / OUTPUT_ROOT
#
# 示例（6 卡 0-5，tier-A 默认权重）：
#   bash src/scripts/v5/run_render_qual_sharded.sh
# ============================================================

# ---- 模型权重（默认 = tier-A 跑法）----
CKPT_DIR="${CKPT_DIR:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
MEMORY_ENCODER_CKPT="${MEMORY_ENCODER_CKPT:-/home/nvme02/wlx/Memory/outputs/v5/train/inctxkv_A_frozen/checkpoints/epoch_3/memory_encoder.pth}"

# ---- 数据（默认 = tier-A 跑法）----
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"
METADATA="${METADATA:-metadata_verify_train.csv}"

# ---- 定性臂 / 上限 / 长视频规格 ----
QUAL_ARMS="${QUAL_ARMS:-off,ideal_A}"                                   # 定性臂（默认 off,ideal_A；可加 random_A）
MAX_CASES="${MAX_CASES:-5}"                                             # case 全局上限（切分前应用）
NUM_CLIPS_QUAL="${NUM_CLIPS_QUAL:-5}"                                   # 长视频 clip 数（默认 5 → 405 帧 ≈ 25s）
FRAME_NUM="${FRAME_NUM:-81}"                                            # 每 clip 帧数（默认 81）

# ---- 分片 / GPU ----
NUM_SHARDS="${NUM_SHARDS:-6}"                                           # 分片数（默认 6）
DIAG_GPUS="${DIAG_GPUS:-0,1,2,3,4,5}"                                   # 逗号分隔的 GPU 列表，数量须 == NUM_SHARDS

# ---- 注入 ----
WEAKEN_FIRST_FRAME="${WEAKEN_FIRST_FRAME:-zero}"                        # 透传给 diag（定性不弱化生成首帧，仅捕获/对照口径用）
INJECT_HIGH="${INJECT_HIGH:-0}"                                         # 默认不开（low-only，对齐训练）

# ---- 生成参数（默认对齐 infer_v5）----
PROMPT="${PROMPT:-First-person view of CS:GO competitive gameplay}"
SIZE="${SIZE:-480*832}"
SAMPLE_STEPS="${SAMPLE_STEPS:-70}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-10.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SEED="${SEED:-42}"
FPS="${FPS:-16}"

# ---- v5 超参（留空 → diag 从 training_metadata 自动采纳，与训练一致）----
GRID="${GRID:-}"
ENCODER_DEPTH="${ENCODER_DEPTH:-}"

# ---- 产出 ----
RUN_NAME="${RUN_NAME:-default}"                                         # 渲染 run 名
TAG="${TAG:-tierA_qual}"                                                # 场景 tag（shard tag 自动加 _s<i>）

# ---- OUTPUT_ROOT（须与 paths.py 一致；env 可覆盖）----
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme02/wlx/Memory/outputs}"

# ============================================================
# 以下内容通常无需修改
# ============================================================

# ---- 环境护栏（项目记忆）----
export TMPDIR=/tmp                       # 防 pymp/torchelastic 孤儿落进 repo
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # 卡号按 PCI 总线序
export OUTPUT_ROOT                       # 透传给 python（paths.py 读同一个 env）

# ---------- 路径检查 ----------
_err=0
if [ -z "${MEMORY_ENCODER_CKPT}" ]; then
    echo "[ERROR] MEMORY_ENCODER_CKPT 未设置" >&2; _err=1
fi
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置" >&2; _err=1
fi
if [ -z "${DATASET_DIR}" ]; then
    echo "[ERROR] DATASET_DIR 未设置" >&2; _err=1
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DIAG_SCRIPT="${PROJECT_ROOT}/src/pipeline/v5/ideal_inject_diag.py"
if [ ! -f "${DIAG_SCRIPT}" ]; then
    echo "[ERROR] 诊断脚本不存在：${DIAG_SCRIPT}" >&2; exit 1
fi

# ---- 把 DIAG_GPUS 拆成数组，校验数量 == NUM_SHARDS ----
IFS=',' read -ra GPUS_ARR <<< "${DIAG_GPUS}"
if [ "${#GPUS_ARR[@]}" -ne "${NUM_SHARDS}" ]; then
    echo "[ERROR] DIAG_GPUS 的 GPU 数量(${#GPUS_ARR[@]}) != NUM_SHARDS(${NUM_SHARDS})" >&2
    echo "        DIAG_GPUS='${DIAG_GPUS}'（逗号分隔），NUM_SHARDS=${NUM_SHARDS}" >&2
    exit 1
fi

# ---- 日志目录 ----
LOG_DIR="${PROJECT_ROOT}/logs/run_render_qual_sharded"
mkdir -p "${LOG_DIR}"

# ---- inject_high 开关 → flag ----
INJECT_HIGH_FLAG=""
if [ "${INJECT_HIGH}" = "1" ] || [ "${INJECT_HIGH,,}" = "true" ] \
   || [ "${INJECT_HIGH,,}" = "on" ] || [ "${INJECT_HIGH,,}" = "yes" ]; then
    INJECT_HIGH_FLAG="--inject_high"
fi

echo "====================================================="
echo "  LingBot-World Memory v5 理想注入【定性渲染】（6 卡分片版）"
echo "  MEMORY_ENCODER_CKPT : ${MEMORY_ENCODER_CKPT}"
echo "  CKPT_DIR            : ${CKPT_DIR}"
echo "  DATASET_DIR         : ${DATASET_DIR}"
echo "  METADATA            : ${METADATA}"
echo "  QUAL_ARMS           : ${QUAL_ARMS}"
echo "  MAX_CASES           : ${MAX_CASES}（case 全集上限；在按 case 取模切分前应用）"
echo "  NUM_CLIPS_QUAL      : ${NUM_CLIPS_QUAL}（× FRAME_NUM=${FRAME_NUM} 帧/clip @ ${FPS}fps）"
echo "  INJECT_HIGH         : ${INJECT_HIGH_FLAG:-<关>}"
echo "  GRID                : ${GRID:-<从 training_metadata 采纳>}"
echo "  ENCODER_DEPTH       : ${ENCODER_DEPTH:-<从 training_metadata 采纳>}"
echo "  NUM_SHARDS          : ${NUM_SHARDS}"
echo "  DIAG_GPUS           : ${DIAG_GPUS}"
echo "  RUN_NAME            : ${RUN_NAME}"
echo "  TAG                 : ${TAG}（shard tag = ${TAG}_s<i>）"
echo "  OUTPUT_ROOT         : ${OUTPUT_ROOT}"
echo "  LOG_DIR             : ${LOG_DIR}/shard_<i>.log"
echo "  说明                : 定性渲染纯出长视频，不算 DINO / 不合并 / 不判 GO/NO-GO"
echo "====================================================="

declare -a SHARD_PIDS

echo ""
echo "===== 并发启动 ${NUM_SHARDS} 个 shard（定性渲染）====="

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPUS_ARR[$i]}"
    shard_tag="${TAG}_s${i}"
    shard_run_dir="${OUTPUT_ROOT}/v5/eval/${RUN_NAME}/${shard_tag}"

    SHARD_ARGS=(
        --ckpt_dir              "${CKPT_DIR}"
        --memory_encoder_ckpt   "${MEMORY_ENCODER_CKPT}"
        --dataset_dir           "${DATASET_DIR}"
        --metadata              "${METADATA}"
        --render_qual
        --qual_arms             "${QUAL_ARMS}"
        --num_clips_qual        "${NUM_CLIPS_QUAL}"
        --frame_num             "${FRAME_NUM}"
        --max_cases             "${MAX_CASES}"
        --weaken_first_frame    "${WEAKEN_FIRST_FRAME}"
        --prompt                "${PROMPT}"
        --size                  "${SIZE}"
        --num_inference_steps   "${SAMPLE_STEPS}"
        --sample_shift          "${SAMPLE_SHIFT}"
        --guide_scale           "${GUIDE_SCALE}"
        --seed                  "${SEED}"
        --fps                   "${FPS}"
        --run_name              "${RUN_NAME}"
        --tag                   "${shard_tag}"
        --device                "cuda:0"
        --shard_index           "${i}"
        --shard_count           "${NUM_SHARDS}"
    )
    if [ -n "${INJECT_HIGH_FLAG}" ]; then
        SHARD_ARGS+=("${INJECT_HIGH_FLAG}")
    fi
    if [ -n "${GRID}" ]; then
        SHARD_ARGS+=(--grid "${GRID}")
    fi
    if [ -n "${ENCODER_DEPTH}" ]; then
        SHARD_ARGS+=(--encoder_depth "${ENCODER_DEPTH}")
    fi

    shard_log="${LOG_DIR}/shard_${i}.log"
    echo "  [shard ${i}] GPU=${gpu} → ${shard_run_dir}/videos/ (log: ${shard_log})"

    CUDA_VISIBLE_DEVICES="${gpu}" TMPDIR=/tmp python "${DIAG_SCRIPT}" "${SHARD_ARGS[@]}" \
        > "${shard_log}" 2>&1 &
    SHARD_PIDS[$i]=$!
done

echo ""
echo "===== 等待所有 shard 结束（PIDs: ${SHARD_PIDS[*]}) ====="
FAIL=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    pid="${SHARD_PIDS[$i]}"
    if ! wait "${pid}"; then
        echo "[ERROR] shard ${i} (PID ${pid}) 失败，详见 ${LOG_DIR}/shard_${i}.log" >&2
        FAIL=1
    fi
done

echo ""
echo "===== v5 定性渲染完成 ====="
echo "  长视频产出 : ${OUTPUT_ROOT}/v5/eval/${RUN_NAME}/${TAG}_s<i>/videos/<ep>/q<query>/long_video_{off,ideal_A}.mp4"
echo "  并排眼看   : 同一 q<query> 目录下 off vs ideal_A，重点看最后一个 clip（重访段）+ gt_first_visit.png"
if [ "${FAIL}" -ne 0 ]; then
    echo "[WARN] 有 shard 失败，已成功的 shard 的长视频仍可用。" >&2
    exit 1
fi
