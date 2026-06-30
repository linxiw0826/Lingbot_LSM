#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v6/run_latentconcat_infer_sharded.sh
# ============================================================
# v6 latent-concat 正常推理（bank 检索历史帧 + anchor-concat，可部署多 clip 自回归长视频）
# —— **多卡分片版**（镜像 run_latentconcat_ideal_diag_sharded.sh，0-5 卡）。
#
# 与诊断脚本（run_latentconcat_ideal_diag_sharded.sh）的区别：
#   - 这是**正常推理 demo**，不三臂 / 不算 DINO / 不判 GO（可选 --score 旁路）。
#   - anchor 来源 = **bank 检索的历史帧**（retrieve_revisit），非 GT oracle 首访帧。
#   - 切分轴 = **episode 全局序号**（episode 模式多 ep 时按序号取模分片，多卡并行推理）。
#   - 每张卡用独立 --tag（${TAG}_s<i> → 独立 run_dir），各跑分到的 episode，无并发写冲突。
#   - 无合并步骤（推理产出 = 各 episode 一条 long_video.mp4）。
#
# 产出布局（走 paths.py，version=v6）：
#   OUTPUT_ROOT/v6/infer/<run_name>/<tag>_s0/<episode_id>/long_video.mp4   ← shard 0 的各 episode
#   OUTPUT_ROOT/v6/infer/<run_name>/<tag>_s1/<episode_id>/long_video.mp4   ← shard 1
#   ...
#
# 可用环境变量覆盖（无需编辑本文件）：
#   CKPT_DIR / FT_MODEL_DIR / FT_HIGH_MODEL_DIR / DATASET_DIR / METADATA / EPISODE_ID
#   RETRIEVAL / NUM_ANCHOR_FRAMES / NUM_CLIPS / FRAME_NUM / PROMPT_SOURCE / PROMPT
#   MAX_EPISODES / SCORE / TAG / RUN_NAME / NUM_SHARDS / INFER_GPUS / OUTPUT_ROOT
#
# 示例（6 卡 0-5，跑全部 episode）：
#   EPISODE_ID=all bash src/scripts/v6/run_latentconcat_infer_sharded.sh
# ============================================================

# ---- 模型权重 ----
CKPT_DIR="${CKPT_DIR:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
FT_MODEL_DIR="${FT_MODEL_DIR:-}"             # 可选：v4 low_noise_model 目录（仅影响 DiT 主干）
FT_HIGH_MODEL_DIR="${FT_HIGH_MODEL_DIR:-}"   # 可选：dual high_noise_model 目录

# ---- 数据（episode 模式）----
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"
METADATA="${METADATA:-metadata_verify_train.csv}"
EPISODE_ID="${EPISODE_ID:-all}"              # 'all'=跑全部（配分片）/ 'first'/'top' / 具体 ep id

# ---- 注入 / anchor ----
RETRIEVAL="${RETRIEVAL:-bank}"               # bank（默认，检索历史帧注入）/ none（纯 base i2v 对照）
NUM_ANCHOR_FRAMES="${NUM_ANCHOR_FRAMES:-1}"  # 每 clip 注入 anchor 帧数上限

# ---- 多 clip ----
NUM_CLIPS="${NUM_CLIPS:-12}"                 # 自回归 clip 数（上限 = T//frame_num）
FRAME_NUM="${FRAME_NUM:-81}"

# ---- prompt 对齐 ----
PROMPT_SOURCE="${PROMPT_SOURCE:-data}"       # data（逐 clip prompt.txt）/ fixed（--prompt）
PROMPT="${PROMPT:-First-person view of CS:GO competitive gameplay}"  # PROMPT_SOURCE=fixed 时用

# ---- 范围 / 评分 ----
MAX_EPISODES="${MAX_EPISODES:-0}"            # 0=不限；>0 取前 N 个 episode（切分前应用）
SCORE="${SCORE:-0}"                          # 1=开启旁路评分（--score，写 scores.csv，不判 GO）

# ---- 分片 / GPU ----
NUM_SHARDS="${NUM_SHARDS:-6}"                # 分片数（默认 6）
INFER_GPUS="${INFER_GPUS:-0,1,2,3,4,5}"      # 逗号分隔 GPU 列表，数量须 == NUM_SHARDS

# ---- 产出 ----
RUN_NAME="${RUN_NAME:-default}"
TAG="${TAG:-long_video}"                     # 场景 tag（shard tag 自动加 _s<i>）

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
if [ -z "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR 未设置" >&2; _err=1
fi
if [ -z "${DATASET_DIR}" ]; then
    echo "[ERROR] DATASET_DIR 未设置" >&2; _err=1
fi
if [ "${_err}" -ne 0 ]; then exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

INFER_SCRIPT="${PROJECT_ROOT}/src/pipeline/v6/latentconcat_infer.py"
if [ ! -f "${INFER_SCRIPT}" ]; then
    echo "[ERROR] 推理脚本不存在：${INFER_SCRIPT}" >&2; exit 1
fi

# ---- 把 INFER_GPUS 拆成数组，校验数量 == NUM_SHARDS ----
IFS=',' read -ra GPUS_ARR <<< "${INFER_GPUS}"
if [ "${#GPUS_ARR[@]}" -ne "${NUM_SHARDS}" ]; then
    echo "[ERROR] INFER_GPUS 的 GPU 数量(${#GPUS_ARR[@]}) != NUM_SHARDS(${NUM_SHARDS})" >&2
    echo "        INFER_GPUS='${INFER_GPUS}'（逗号分隔），NUM_SHARDS=${NUM_SHARDS}" >&2
    exit 1
fi

# ---- 日志目录 ----
LOG_DIR="${PROJECT_ROOT}/logs/run_latentconcat_infer_sharded"
mkdir -p "${LOG_DIR}"

echo "====================================================="
echo "  LingBot-World Memory v6 latent-concat 正常推理（bank 检索 + anchor-concat，多卡分片版）"
echo "  CKPT_DIR            : ${CKPT_DIR}"
echo "  FT_MODEL_DIR        : ${FT_MODEL_DIR:-<无>}"
echo "  FT_HIGH_MODEL_DIR   : ${FT_HIGH_MODEL_DIR:-<无>}"
echo "  DATASET_DIR         : ${DATASET_DIR}"
echo "  METADATA            : ${METADATA}"
echo "  EPISODE_ID          : ${EPISODE_ID}"
echo "  RETRIEVAL           : ${RETRIEVAL}"
echo "  NUM_ANCHOR_FRAMES   : ${NUM_ANCHOR_FRAMES}"
echo "  NUM_CLIPS           : ${NUM_CLIPS}"
echo "  FRAME_NUM           : ${FRAME_NUM}"
echo "  PROMPT_SOURCE       : ${PROMPT_SOURCE}"
echo "  MAX_EPISODES        : ${MAX_EPISODES}"
echo "  SCORE               : ${SCORE}"
echo "  NUM_SHARDS          : ${NUM_SHARDS}"
echo "  INFER_GPUS          : ${INFER_GPUS}"
echo "  RUN_NAME            : ${RUN_NAME}"
echo "  TAG                 : ${TAG}（shard tag = ${TAG}_s<i>）"
echo "  OUTPUT_ROOT         : ${OUTPUT_ROOT}"
echo "  LOG_DIR             : ${LOG_DIR}/shard_<i>.log"
echo "====================================================="

declare -a SHARD_PIDS

echo ""
echo "===== 并发启动 ${NUM_SHARDS} 个 shard ====="

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPUS_ARR[$i]}"
    shard_tag="${TAG}_s${i}"
    shard_run_dir="${OUTPUT_ROOT}/v6/infer/${RUN_NAME}/${shard_tag}"

    SHARD_ARGS=(
        --ckpt_dir              "${CKPT_DIR}"
        --dataset_dir           "${DATASET_DIR}"
        --metadata              "${METADATA}"
        --episode_id            "${EPISODE_ID}"
        --retrieval             "${RETRIEVAL}"
        --num_anchor_frames     "${NUM_ANCHOR_FRAMES}"
        --num_clips             "${NUM_CLIPS}"
        --frame_num             "${FRAME_NUM}"
        --prompt_source         "${PROMPT_SOURCE}"
        --prompt                "${PROMPT}"
        --max_episodes          "${MAX_EPISODES}"
        --run_name              "${RUN_NAME}"
        --tag                   "${shard_tag}"
        --device                "cuda:0"
        --shard_index           "${i}"
        --shard_count           "${NUM_SHARDS}"
    )
    if [ "${SCORE}" = "1" ]; then
        SHARD_ARGS+=(--score)
    fi
    if [ -n "${FT_MODEL_DIR}" ]; then
        SHARD_ARGS+=(--ft_model_dir "${FT_MODEL_DIR}")
    fi
    if [ -n "${FT_HIGH_MODEL_DIR}" ]; then
        SHARD_ARGS+=(--ft_high_model_dir "${FT_HIGH_MODEL_DIR}")
    fi

    shard_log="${LOG_DIR}/shard_${i}.log"
    echo "  [shard ${i}] GPU=${gpu} → ${shard_run_dir} (log: ${shard_log})"

    CUDA_VISIBLE_DEVICES="${gpu}" TMPDIR=/tmp python "${INFER_SCRIPT}" "${SHARD_ARGS[@]}" \
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
echo "===== v6 分片推理完成 ====="
echo "  各 shard 产出 : ${OUTPUT_ROOT}/v6/infer/${RUN_NAME}/${TAG}_s<i>/<episode_id>/long_video.mp4"
if [ "${FAIL}" -ne 0 ]; then
    echo "[WARN] 有 shard 失败，已成功的 shard 的 long_video.mp4 仍可用。" >&2
    exit 1
fi
