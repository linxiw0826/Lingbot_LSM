#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 用户配置区 — 修改以下变量后运行 bash src/scripts/v5/run_ideal_inject_diag_sharded.sh
# ============================================================
# v5-KV 理想注入诊断（S-V4 / Step 43）—— **6 卡分片版**（off / ideal_A / random_A [/ ideal_B]）。
# 把 revisit case 全集（先 detect 完整列表，再在切分前应用 --max_cases 全局上限）按 **case
# 全局序号取模** 切成 NUM_SHARDS 份，每张卡跑一个 shard（不同 --tag → 独立 run_dir，
# 无 per_window.csv 并发写冲突），全部跑完后用 merge_diag_shards.py 合并出三臂 GO/NO-GO 判决。
#
# 产出布局（走 paths.py，与单进程版一致）：
#   OUTPUT_ROOT/v5/eval/<run_name>/<tag>_s0/   ← shard 0 的 per_window.csv + videos/
#   OUTPUT_ROOT/v5/eval/<run_name>/<tag>_s1/   ← shard 1
#   ...
#   OUTPUT_ROOT/v5/eval/<run_name>/<tag>/      ← 合并后的 per_window.csv + summary.md
#
# 可用环境变量覆盖（无需编辑本文件）：
#   CKPT_DIR / MEMORY_ENCODER_CKPT / DATASET_DIR / METADATA
#   ARMS / MAX_CASES / TAG / RUN_NAME
#   NUM_SHARDS / DIAG_GPUS / GO_MARGIN / WEAKEN_FIRST_FRAME / INJECT_HIGH
#   GRID / ENCODER_DEPTH / OUTPUT_ROOT
#
# 示例（6 卡 0-5，tier-A 默认权重）：
#   bash src/scripts/v5/run_ideal_inject_diag_sharded.sh
# ============================================================

# ---- 模型权重（默认 = tier-A 跑法）----
CKPT_DIR="${CKPT_DIR:-/home/nvme02/lingbot-world/models/lingbot-world-base-act}"
MEMORY_ENCODER_CKPT="${MEMORY_ENCODER_CKPT:-/home/nvme02/wlx/Memory/outputs/v5/train/inctxkv_A_frozen/checkpoints/epoch_3/memory_encoder.pth}"

# ---- 数据（默认 = tier-A 跑法）----
DATASET_DIR="${DATASET_DIR:-/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3}"
METADATA="${METADATA:-metadata_verify_train.csv}"

# ---- 臂 / 上限（默认 = tier-A 跑法）----
ARMS="${ARMS:-off,ideal_A,random_A}"                                    # 诊断臂子集
MAX_CASES="${MAX_CASES:-5}"                                             # case 全局上限（切分前应用）

# ---- 分片 / GPU ----
NUM_SHARDS="${NUM_SHARDS:-6}"                                           # 分片数（默认 6）
DIAG_GPUS="${DIAG_GPUS:-0,1,2,3,4,5}"                                   # 逗号分隔的 GPU 列表，数量须 == NUM_SHARDS

# ---- 判据 / 注入 ----
GO_MARGIN="${GO_MARGIN:-0.01}"                                          # GO 判据 margin
WEAKEN_FIRST_FRAME="${WEAKEN_FIRST_FRAME:-zero}"                        # F-18 护栏：zero=置零中性灰
INJECT_HIGH="${INJECT_HIGH:-0}"                                         # 默认不开（low-only，对齐训练）

# ---- v5 超参（留空 → diag 从 training_metadata 自动采纳，与训练一致）----
GRID="${GRID:-}"
ENCODER_DEPTH="${ENCODER_DEPTH:-}"

# ---- 产出 ----
RUN_NAME="${RUN_NAME:-default}"                                         # 诊断 run 名
TAG="${TAG:-tierA_ideal}"                                              # 场景 tag（shard tag 自动加 _s<i>）

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
MERGE_SCRIPT="${PROJECT_ROOT}/src/pipeline/v5/merge_diag_shards.py"
if [ ! -f "${DIAG_SCRIPT}" ]; then
    echo "[ERROR] 诊断脚本不存在：${DIAG_SCRIPT}" >&2; exit 1
fi
if [ ! -f "${MERGE_SCRIPT}" ]; then
    echo "[ERROR] 合并脚本不存在：${MERGE_SCRIPT}" >&2; exit 1
fi

# ---- 把 DIAG_GPUS 拆成数组，校验数量 == NUM_SHARDS ----
IFS=',' read -ra GPUS_ARR <<< "${DIAG_GPUS}"
if [ "${#GPUS_ARR[@]}" -ne "${NUM_SHARDS}" ]; then
    echo "[ERROR] DIAG_GPUS 的 GPU 数量(${#GPUS_ARR[@]}) != NUM_SHARDS(${NUM_SHARDS})" >&2
    echo "        DIAG_GPUS='${DIAG_GPUS}'（逗号分隔），NUM_SHARDS=${NUM_SHARDS}" >&2
    exit 1
fi

# ---- 合并后的 run_dir 路径（与 paths.eval_run_dir 布局一致）----
MERGE_OUT_DIR="${OUTPUT_ROOT}/v5/eval/${RUN_NAME}/${TAG}"

# ---- 日志目录 ----
LOG_DIR="${PROJECT_ROOT}/logs/run_ideal_inject_diag_sharded"
mkdir -p "${LOG_DIR}"

# ---- inject_high 开关 → flag ----
INJECT_HIGH_FLAG=""
if [ "${INJECT_HIGH}" = "1" ] || [ "${INJECT_HIGH,,}" = "true" ] \
   || [ "${INJECT_HIGH,,}" = "on" ] || [ "${INJECT_HIGH,,}" = "yes" ]; then
    INJECT_HIGH_FLAG="--inject_high"
fi

echo "====================================================="
echo "  LingBot-World Memory v5 理想注入诊断（6 卡分片版）"
echo "  MEMORY_ENCODER_CKPT : ${MEMORY_ENCODER_CKPT}"
echo "  CKPT_DIR            : ${CKPT_DIR}"
echo "  DATASET_DIR         : ${DATASET_DIR}"
echo "  METADATA            : ${METADATA}"
echo "  ARMS                : ${ARMS}"
echo "  MAX_CASES           : ${MAX_CASES}（case 全集上限；在按 case 取模切分前应用）"
echo "  GO_MARGIN           : ${GO_MARGIN}"
echo "  WEAKEN_FIRST_FRAME  : ${WEAKEN_FIRST_FRAME}"
echo "  INJECT_HIGH         : ${INJECT_HIGH_FLAG:-<关>}"
echo "  GRID                : ${GRID:-<从 training_metadata 采纳>}"
echo "  ENCODER_DEPTH       : ${ENCODER_DEPTH:-<从 training_metadata 采纳>}"
echo "  NUM_SHARDS          : ${NUM_SHARDS}"
echo "  DIAG_GPUS           : ${DIAG_GPUS}"
echo "  RUN_NAME            : ${RUN_NAME}"
echo "  TAG                 : ${TAG}（shard tag = ${TAG}_s<i>）"
echo "  OUTPUT_ROOT         : ${OUTPUT_ROOT}"
echo "  合并目标            : ${MERGE_OUT_DIR}"
echo "  LOG_DIR             : ${LOG_DIR}/shard_<i>.log"
echo "  判据                : ideal_A DINO > off+margin 且 > random_A+margin（合并后判）"
echo "====================================================="

# ---- 记录每个 shard 的 run_dir 路径（合并用）----
declare -a SHARD_DIRS
declare -a SHARD_PIDS

echo ""
echo "===== 并发启动 ${NUM_SHARDS} 个 shard ====="

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu="${GPUS_ARR[$i]}"
    shard_tag="${TAG}_s${i}"
    shard_run_dir="${OUTPUT_ROOT}/v5/eval/${RUN_NAME}/${shard_tag}"
    SHARD_DIRS[$i]="${shard_run_dir}"

    SHARD_ARGS=(
        --ckpt_dir              "${CKPT_DIR}"
        --memory_encoder_ckpt   "${MEMORY_ENCODER_CKPT}"
        --dataset_dir           "${DATASET_DIR}"
        --metadata              "${METADATA}"
        --arms                  "${ARMS}"
        --max_cases             "${MAX_CASES}"
        --go_margin             "${GO_MARGIN}"
        --weaken_first_frame    "${WEAKEN_FIRST_FRAME}"
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
    echo "  [shard ${i}] GPU=${gpu} → ${shard_run_dir} (log: ${shard_log})"

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

if [ "${FAIL}" -ne 0 ]; then
    echo "[ERROR] 有 shard 失败，但仍尝试合并（已成功的 shard 的 per_window.csv 仍可用）。" >&2
fi

echo ""
echo "===== 合并 per_window.csv + 出判决 ====="
MERGE_SHARD_ARGS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    MERGE_SHARD_ARGS+=("${SHARD_DIRS[$i]}")
done

python "${MERGE_SCRIPT}" \
    --shard_dirs "${MERGE_SHARD_ARGS[@]}" \
    --out_dir    "${MERGE_OUT_DIR}" \
    --margin     "${GO_MARGIN}"

echo ""
echo "===== v5 分片诊断完成 ====="
echo "  合并 summary.md : ${MERGE_OUT_DIR}/summary.md"
echo "  合并 per_window : ${MERGE_OUT_DIR}/per_window.csv"
echo "  合并 note       : ${MERGE_OUT_DIR}/merged_note.md"
echo "  各 shard videos : ${OUTPUT_ROOT}/v5/eval/${RUN_NAME}/${TAG}_s<i>/videos/（未搬运）"
