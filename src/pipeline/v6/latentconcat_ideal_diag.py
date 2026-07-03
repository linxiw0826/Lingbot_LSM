"""
latentconcat_ideal_diag.py — v6 latent-concat 理想注入诊断（zero-training；experiment_design Step 44 / S-V5）
============================================================================================================

**目的**（decisions.md 讨论 10 + 通用闸门补记；承 Step 43 in-context KV 主线判死 [[F-30]]）：
in-context KV（范式 A）整条判死后，latent-concat 升为执行主线。但单 clip 上界（[[F-23]]，
stage1_upperbound.py 零训练 oracle，12/12 oracle>off）**≠** 多 clip 端到端注入闭环成立。
本脚本 = 在**多 clip 自回归**长视频上对 latent-concat 通道做**零训练理想复验**——把 GT 首访帧
当尾部 clean anchor 帧 concat 进 latent 时间维（Context-as-Memory / FramePack 式），走骨干原生
i2v 条件流，回答「latent-concat 通道在多 clip 端到端闭环里是否真能用记忆」。这是 latent-concat
通道自己的 GO 闸门。

**零训练、骨干一字不改、不挂 LoRA、不用 negative-RoPE**：anchor 走骨干**已训练**的 36 通道
i2v 条件路径（msk=1 clean + anchor latent + anchor 位姿 plucker，尾部 append），patch_embedding /
q/k/v/o / cam_* / blocks / head 全冻，全程 torch.no_grad。复用 stage1_upperbound 已验证的三个
帧维 anchor-concat 函数（_generate_with_anchor / _build_conditioning_with_anchor /
_encode_anchor_latent），把它们从**单 clip** 升到**多 clip 自回归**（镜像 infer_v5:589-668）。

**三臂（定量）**（每个 revisit case，多 clip 自回归长视频）：
  · off            : 无 anchor，纯自回归 i2v（baseline）。
  · anchor_ideal   : GT 首访帧当尾部 clean anchor 帧（理想记忆，逐 clip 注入）。
  · anchor_random  : 错帧（非首访随机历史帧）当 anchor（内容对照，复用 _pick_random_hist_frame）。
打分：复用 oracle_injection._revisit_consistency（frame-aligned DINO）。长视频窗口锚定使
**重访 query_frame 落在最后一个 clip**，对**最后一个 clip（= 重访 clip）** vs GT 首访帧算
revisit_consistency_dino_mean，三臂同口径。

**判据（GO/NO-GO）**：
  GO  ⇔ anchor_ideal 的 DINO 均值 > off + margin（默认 0.01）**且** > anchor_random + margin
        → latent-concat 通道在多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 配方训练（通道维
          concat + negative-RoPE + r128 LoRA on 全 DiT 线性层 + 位姿）。
  NO-GO ⇔ 否则 → 升级更重干预（LoRA / 解冻）。

**风险处理（报告 R1/R2/R4，逐条落地）**：
  - **R2（anchor 尾部丢弃在多 clip 下的帧对齐）**：每 clip 的生成由 _generate_with_anchor 内部
    丢弃尾部 vae_stride_t*n_anchor 个 anchor pixel 帧、并断言 query 段 = frame_num（见
    stage1_upperbound BLOCK-1 修复）→ 返回的 clip 恰好 frame_num 帧，自回归取「末帧」永远是
    query 帧、不会取到 anchor 帧。故多 clip 链条不被 anchor 污染。
  - **R4（anchor 位姿逐 clip 相对当前 clip query[0] 重算）**：anchor 的**绝对 c2w**（ep.poses[fi]）
    每个 case 只取一次，但每 clip 调用 _generate_with_anchor 时把**该 clip 自己的 query_poses**
    传进去，_build_conditioning_with_anchor 内部用该 clip 的 query[0] 作参考帧把 anchor 位姿
    重新相对化（构造 [query_ref, anchor] 2 帧序列取 frame[1]）→ 逐 clip 自动重算，不会位姿错位。
  - **R1（多 clip 漂移污染 DINO 口径）**：frame-aligned DINO 只比「重访 clip vs GT 首访帧」，
    三臂同窗口同 seed 链，漂移对三臂同等作用，差值仍反映 anchor 贡献。

**prompt 对齐（用户硬指令）**：`--prompt_source data`（默认）逐 clip 从该 clip 目录的 prompt.txt
取数据 prompt（与训练 target_clip["prompt"] 同源），不再固定串；`--prompt_source fixed` 回退
`--prompt`。所有臂用同一对齐 prompt（per-clip 一致）。

**历史帧选取**：`--retrieval gt_oracle`（默认）= GT 首访帧（_find_revisit_points，与 v5 ideal
同源，隔离注入）；`--retrieval bank`（正常推理用 bank 检索）本组件**占位 NotImplemented**，
留给下一个组件（bank 检索版）。

服务器跑前置：`export TMPDIR=/tmp`（否则 kill 掉的 run 会在仓库留 pymp-* 孤儿）。
本地无 torch/CUDA/DINO 真跑不动；--help / py_compile 走通即可（真跑待服务器）。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path（与 stage1_upperbound / ideal_inject_diag 一致）
# ---------------------------------------------------------------------------
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                            # → src/
_PROJECT_ROOT = dirname(_SRC_DIR)                            # → Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, "refs", "lingbot-world")

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 复用 oracle_injection.py（重访点检测 / 帧切片 / 指标 / IO，import 不重写）
# ---------------------------------------------------------------------------
from pipeline.eval.oracle_injection import (  # noqa: E402
    RevisitPoint,
    _frame_to_clip_slice,
    _revisit_consistency,
    _dino_feat,
    _frame_to_pil,
    _save_frame_png,
    _save_video,
    _read_video_back,
)
import pipeline.eval.oracle_injection as _oracle_inj  # noqa: E402

# 复用 retrieval_probe.py（episode 加载 + 解码；_enumerate_cases 内部用 build_episode_data /
# _find_revisit_points，本文件不直接调）
from pipeline.eval.retrieval_probe import (  # noqa: E402
    EpisodeData,
    load_episode_clips,
    build_episode_data,
    _decode_episode_video,
)

# 复用 stage1_upperbound 的**已验证**帧维 anchor-concat 三函数 + anchor 构造 + 分辨率换算
# （[[F-23]] 单 clip oracle>off 12/12；本脚本只把它从单 clip 升到多 clip 自回归，不改其内部）
from pipeline.eval.stage1_upperbound import (  # noqa: E402
    _generate_with_anchor,
    _build_anchor_inputs,
    _resize_hw,
)

# 复用 eval_v5 的 random 历史帧选取（anchor_random 臂；与 v5 ideal 同源）
from pipeline.v5.eval_v5 import _pick_random_hist_frame  # noqa: E402

# 复用 ideal_inject_diag 的 case 枚举（分片用；max_cases 在切分前应用，各分片看同一全集）
from pipeline.v5.ideal_inject_diag import _enumerate_cases  # noqa: E402

# 复用 infer_v4 的 pipeline 装载/转换（memory 不参与本实验，仅复用 DiT 主干）
from pipeline.v4.infer_v4 import (  # noqa: E402
    _load_ft_model_and_prepare_ckpt,
    _convert_pipeline_to_memory,
)

from pipeline.common.paths import (  # noqa: E402
    eval_run_dir,
    snapshot_config,
    default_run_name,
)


# 臂顺序（off 在前作 baseline）。
ALL_ARMS = ("off", "anchor_ideal", "anchor_random")
DEFAULT_ARMS = ("off", "anchor_ideal", "anchor_random")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "v6 latent-concat 理想注入诊断（zero-training；Step 44 / S-V5）："
            "off / anchor_ideal（GT 首访帧当尾部 clean anchor）/ anchor_random（错帧对照）× "
            "多 clip 自回归 × frame-aligned DINO。服务器跑前置：export TMPDIR=/tmp。"
        )
    )
    # ---- 模型权重（与 stage1_upperbound / infer_v4 对齐）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="lingbot-world 预训练权重目录（含 low/high noise_model / VAE / T5）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="（可选）v4 low_noise_model checkpoint 目录（仅影响 DiT 主干）")
    p.add_argument("--ft_high_model_dir", type=str, default=None,
                   help="（可选）dual 训练 high_noise_model 目录")

    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_verify_train.csv")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")
    p.add_argument("--max_cases", type=int, default=5,
                   help="总 revisit case 上限（默认 5，对齐 Step 44；**在天花板过滤之后、切分之前**应用）")

    # ---- GT 天花板过滤（真重访筛选；丢弃几何匹配到的假重访）----
    # 动态 CS:GO 下 compute_gt_revisit 靠绝对位置+yaw 匹配，会挑出「同位姿但画面全变」的假重访
    # （DINO(首访,重访) ≤ DINO(首访,随机)）。这些假重访会污染闸门判读，故在跑生成**之前**用
    # GT 帧 DINO 天花板把它们过滤掉（零生成、只 DINO + 已解码帧、不加载骨干）。判据复用
    # revisit_metric_sanity 的 GT 天花板算法：d_vr=DINO(GT[首访],GT[重访])，
    # d_vrand=DINO(GT[首访],随机帧)。保留 ⇔ d_vr - d_vrand >= min_ceiling_delta **且**
    # d_vr >= min_ceiling_abs。
    p.add_argument("--ceiling_filter", dest="ceiling_filter", action="store_true", default=True,
                   help="开启 GT 天花板过滤（**默认 on**）：跑生成前丢弃假重访 case。")
    p.add_argument("--no_ceiling_filter", dest="ceiling_filter", action="store_false",
                   help="关闭 GT 天花板过滤，跑全部（几何匹配到的）case（向后兼容 / 对照用；"
                        "此时行为与加过滤前逐位一致）。")
    p.add_argument("--min_ceiling_delta", type=float, default=0.1,
                   help="保留判据 Δ：d_vr - d_vrand 须 >= 此值（默认 0.1）。")
    p.add_argument("--min_ceiling_abs", type=float, default=0.5,
                   help="保留判据绝对下限：d_vr 须 >= 此值（默认 0.5，防 d_vr/d_vrand 都低）。")

    # ---- 臂 / 判据 ----
    p.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS),
                   help="逗号分隔的臂子集（默认 off,anchor_ideal,anchor_random）")
    p.add_argument("--go_margin", type=float, default=0.01,
                   help="GO 判据 margin：anchor_ideal 的 DINO 均值须 > off 且 > anchor_random "
                        "至少这么多（默认 0.01）")

    # ---- 历史帧选取（注入源）----
    p.add_argument("--retrieval", type=str, default="gt_oracle",
                   choices=["gt_oracle", "bank"],
                   help="anchor 记忆源：gt_oracle（默认，GT 首访帧，隔离注入）/ "
                        "bank（正常推理用 bank 检索，本组件占位 NotImplemented，留给下个组件）")

    # ---- 多 clip 自回归 ----
    p.add_argument("--num_clips", type=int, default=5,
                   help="定量多 clip 自回归的 clip 数（默认 5；5 × frame_num=81 @ 16fps ≈ 25s）。"
                        "窗口锚定使重访 query_frame 落在最后一个 clip。")

    # ---- 重访点判定（复用 oracle_injection 口径）----
    p.add_argument("--hit_dist", type=float, default=40.0)
    p.add_argument("--hit_yaw", type=float, default=30.0)
    p.add_argument("--intermediate_separation", type=float, default=100.0)
    p.add_argument("--min_time_gap_sec", type=float, default=5.0)
    p.add_argument("--clip_overlap_frames", type=int, default=0)
    p.add_argument("--max_revisit_points", type=int, default=2)

    # ---- 生成参数（与 stage1_upperbound 对齐；_generate_with_anchor 直接读 args 这些字段）----
    p.add_argument("--frame_num", type=int, default=81,
                   help="每 clip 帧数（4n+1）")
    p.add_argument("--num_inference_steps", type=int, default=70,
                   help="diffusion 采样步数")
    p.add_argument("--sample_shift", type=float, default=10.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay",
                   help="prompt_source=fixed 时的固定 prompt（data 模式仅作 prompt.txt 缺失回退）")
    p.add_argument("--prompt_source", type=str, default="data",
                   choices=["data", "fixed"],
                   help="data（默认，逐 clip 从 clip 目录 prompt.txt 取数据 prompt，与训练同源）/ "
                        "fixed（所有 clip 用 --prompt）")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42,
                   help="基础 seed；每 clip 用 seed+clip_idx（镜像 infer_v5 自回归）")
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（min_time_gap_sec→帧数换算 + 保存）")

    # ---- 定性渲染（off vs anchor_ideal 的多 clip 长视频，眼看重访段）----
    p.add_argument("--render_qual", action="store_true", default=False,
                   help="定性渲染开关（向后兼容 additive）：开启时只渲 off vs anchor_ideal 的多 clip "
                        "自回归长视频，不算 DINO / 不判 GO/NO-GO。不开时定量三臂行为逐位不变。")
    p.add_argument("--num_clips_qual", type=int, default=5,
                   help="定性长视频的 clip 数（默认 5；5 × frame_num=81 @ 16fps ≈ 25s / 405 帧）")
    p.add_argument("--qual_arms", type=str, default="off,anchor_ideal",
                   help="定性渲染的臂子集（逗号分隔，默认 off,anchor_ideal；可加 anchor_random）")

    # ---- 产出 ----
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--tag", type=str, default="latentconcat_ideal",
                   help="eval 场景 tag（INDEX 区分用，默认 latentconcat_ideal）")

    # ---- 分片（additive：shard_count 默认 1 → 逐位与单进程一致）----
    # 切分轴 = revisit case 全局序号（与 ideal_inject_diag 一致）；**在切分前应用 --max_cases
    # 全局上限**保证各分片看同一全集；本分片只处理 case_global_index % shard_count == shard_index。
    p.add_argument("--shard_index", type=int, default=0,
                   help="当前分片索引（0-based），多卡并行诊断用。默认 0。")
    p.add_argument("--shard_count", type=int, default=1,
                   help="总分片数。默认 1=不分片（走原始路径）；>1 时按 case 全局序号取模分片。")

    # ---- 确定性 case manifest（治本：全体 shard 消费同一份 case 列表，见 Phase A0 修改 A）----
    # 分片切分不重不漏的前提是各 shard 看到**逐字节相同**的有序 case 全集。天花板过滤依赖
    # DINO 前向（GPU 非确定）+ filter_rng（随机抽历史帧），borderline case 的 keep 一翻转，
    # 全局序号 gi 整体错位 → gi % N 在各独立进程里切出既重叠又漏的划分（ep05_p09/q560 曾落进
    # 两个 shard）。解法：枚举+过滤**只跑一次**落盘成 manifest，所有 shard 只读该 manifest。
    p.add_argument("--emit_manifest", type=str, default=None,
                   help="（与 --cases_manifest 互斥）单进程模式：跑完枚举+天花板过滤+max_cases 截断，"
                        "把最终有序 case 列表（含 RevisitPoint 全字段 + d_vr/d_vrand/keep 诊断量）"
                        "序列化为 JSON 落盘到该路径，然后退出，**不生成视频**。")
    p.add_argument("--cases_manifest", type=str, default=None,
                   help="（与 --emit_manifest 互斥）若给定且文件存在 → 跳过现场枚举+过滤，直接从该 "
                        "JSON 反序列化出有序 case 全集，再走 gi %% shard_count 分片（保证各 shard 看"
                        "同一确定性全集，切分不重不漏）。文件不存在 → 回退现场枚举+过滤（附 warning）。")

    return p.parse_args()


# ---------------------------------------------------------------------------
# per-clip prompt（报告落点 3 + 用户 prompt 对齐指令）
# ---------------------------------------------------------------------------

def _clip_prompt(ep: EpisodeData, clip_start: int, args) -> str:
    """逐 clip 取数据 prompt（与训练 target_clip["prompt"] 同源）。

    prompt_source=fixed → 直接返回 --prompt。
    prompt_source=data  → 用 ep.frame_to_clip 把该 clip 的全局起始帧映射到 clip_array_idx，
      读 ep.clips[clip_array_idx].clip_path/prompt.txt（每 clip 目录已存，D-02 / 报告 §5）。
      缺失/读失败 → 回退 --prompt（带 warning）。

    采用「读 prompt.txt」路径（报告 §5.3(b)）而非扩 ClipMeta（§5.3(a)），以**不修改**
    retrieval_probe.py（code_standards §5：不改既有 src 文件）。
    """
    if args.prompt_source == "fixed":
        return args.prompt
    T = ep.poses.shape[0]
    if not ep.frame_to_clip:
        return args.prompt
    idx = int(max(0, min(int(clip_start), T - 1)))
    clip_array_idx = ep.frame_to_clip[idx][0]
    if clip_array_idx < 0 or clip_array_idx >= len(ep.clips):
        return args.prompt
    ppath = os.path.join(ep.clips[clip_array_idx].clip_path, "prompt.txt")
    try:
        with open(ppath, "r", encoding="utf-8") as fh:
            txt = fh.read().strip()
        if txt:
            return txt
        logger.warning("prompt.txt 为空 → 回退 --prompt：%s", ppath)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读 prompt.txt 失败（%s）→ 回退 --prompt：%s", exc, ppath)
    return args.prompt


def _clip_args(args, prompt: str, seed: int):
    """浅拷贝 args，覆盖 prompt + seed（供逐 clip 调 stage1._generate_with_anchor）。

    _generate_with_anchor 直接读 args 的 prompt/seed/frame_num/num_inference_steps/
    sample_shift/guide_scale/size 等字段；逐 clip 只需变 prompt（数据 prompt）+ seed
    （base+clip_idx，镜像 infer_v5），其余字段原样继承。
    """
    ns = argparse.Namespace(**vars(args))
    ns.prompt = prompt
    ns.seed = seed
    return ns


# ---------------------------------------------------------------------------
# 多 clip 自回归 rollout（镜像 infer_v5:589-668，把 generate 换成帧维 anchor-concat）
# ---------------------------------------------------------------------------

def _rollout_case_arm(
    wan_i2v,
    ep: EpisodeData,
    pt: RevisitPoint,
    frames: np.ndarray,             # [T,3,H,W] in [-1,1]
    arm: str,
    anchor_latent: Optional[torch.Tensor],   # [16,n_anchor,lat_h,lat_w] 或 None（off）
    anchor_poses: Optional[np.ndarray],       # [n_anchor,4,4] 绝对 c2w 或 None（off）
    args,
    device: torch.device,
    num_clips: int,
) -> Tuple[List[np.ndarray], int]:
    """对单个 (case, 臂) 跑 num_clips 自回归，返回 (clip_videos, win_start)。

    镜像 infer_v5:589-668 的循环（逐 clip 切 poses/actions/intrinsics、seed=base+clip_idx、
    上一 clip 末帧 → 下一 clip current_img），但把 wan_i2v.generate 换成 stage1_upperbound 的
    **帧维 anchor-concat** 生成 _generate_with_anchor（无 memory cross-attn、无 in-context KV）。

    窗口锚定：win_start = query_frame - (num_clips-1)*frame_num，clamp 到 [0, T-total]，使重访
    query_frame 落在**最后一个 clip**（clip_videos[-1] = 重访 clip）。

    注入排期：anchor **逐 clip 都注入**（"can't hurt, might help"——冻结骨干能忽略无关记忆，
    见 injection_idea_soundness；非 OPEN 决策，无 PENDING）。如日后要改成「只在重访 clip 注入」，
    在下方循环里 clip_idx < num_clips-1 时把 a_lat/a_pose 置 None 即可。

    R2（anchor 尾部丢弃）：_generate_with_anchor 内部已丢尾部 anchor pixel 帧并断言 query 段
    = frame_num → 返回 clip 恰 frame_num 帧，自回归取末帧永远是 query 帧，不被 anchor 污染。
    R4（anchor 位姿逐 clip 重算）：anchor 绝对 c2w 每 case 取一次，但每 clip 把**该 clip 自己的
    query_poses** 传进 _generate_with_anchor → _build_conditioning_with_anchor 内部以该 clip 的
    query[0] 为参考帧重新相对化 anchor 位姿，逐 clip 自动重算。
    """
    T = ep.poses.shape[0]
    frame_num = args.frame_num
    total_frames = num_clips * frame_num
    win_start = pt.query_frame - (num_clips - 1) * frame_num
    win_start = int(max(0, min(win_start, max(0, T - total_frames))))

    a_lat = anchor_latent if arm != "off" else None
    a_pose = anchor_poses if arm != "off" else None

    current_img = _frame_to_pil(frames[win_start])  # 起始首帧 = 窗口首帧
    clip_videos: List[np.ndarray] = []

    for clip_idx in range(num_clips):
        clip_start = win_start + clip_idx * frame_num
        clip_end = clip_start + frame_num
        if clip_end <= T:
            clip_poses = ep.poses[clip_start:clip_end].astype(np.float32)
            clip_actions = ep.actions[clip_start:clip_end].astype(np.float32)
            clip_intr = ep.intrinsics[clip_start:clip_end].astype(np.float32)
        else:
            # 数据不足：取末尾 frame_num 帧 + 末帧 pad（对齐 infer_v5 / _frame_to_clip_slice fallback）
            clip_poses, clip_actions, clip_intr, _ = _frame_to_clip_slice(
                ep, clip_start, frame_num)
            clip_poses = clip_poses.astype(np.float32)
            clip_actions = clip_actions.astype(np.float32)
            clip_intr = clip_intr.astype(np.float32)

        prompt = _clip_prompt(ep, clip_start, args)
        clip_args = _clip_args(args, prompt, args.seed + clip_idx)

        video = _generate_with_anchor(
            wan_i2v, current_img, a_lat, a_pose,
            clip_poses, clip_actions, clip_intr, clip_args, device)
        if video is None:
            logger.warning("ep=%s q=%d [%s] clip %d/%d：生成 None，终止该臂",
                           ep.episode_id, pt.query_frame, arm, clip_idx + 1, num_clips)
            break
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().float().numpy()
        video = np.asarray(video, dtype=np.float32)  # [3,frame_num,H,W] in [-1,1]
        clip_videos.append(video)

        # 自回归链接：本 clip 末帧（必是 query 帧，R2）→ 下一 clip 首帧（镜像 infer_v5）
        last_chw = video[:, -1]                                   # [3,H,W]
        last_hwc = (last_chw.transpose(1, 2, 0) * 127.5 + 127.5
                    ).clip(0, 255).astype(np.uint8)               # [H,W,3] uint8
        current_img = Image.fromarray(last_hwc)
        logger.info("ep=%s q=%d [%s]：clip %d/%d done（prompt=%.40r）",
                    ep.episode_id, pt.query_frame, arm, clip_idx + 1, num_clips, prompt)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return clip_videos, win_start


# ---------------------------------------------------------------------------
# anchor 构造（按臂取记忆帧索引 → clean latent + 绝对位姿）
# ---------------------------------------------------------------------------

def _build_arm_anchor(
    wan_i2v, ep: EpisodeData, frames: np.ndarray, pt: RevisitPoint, arm: str,
    h: int, w: int, device: torch.device, rng: np.random.Generator,
) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """为指定臂构造 anchor (latent [16,1,lat_h,lat_w], poses [1,4,4]) 或 None。

    anchor_ideal  : GT 首访帧（--retrieval gt_oracle）。
    anchor_random : 非首访随机历史帧（_pick_random_hist_frame）。
    off           : 调用方不会传入 off。
    """
    T = ep.poses.shape[0]
    if arm == "anchor_ideal":
        if args_retrieval_is_bank():  # 占位：bank 检索版在下个组件实现
            raise NotImplementedError(
                "--retrieval bank 尚未实现（本组件只做 gt_oracle 隔离注入；"
                "bank 检索版留给下一个组件）。")
        fi = int(pt.first_visit_frame)
        if not (0 <= fi < T):
            return None
        return _build_anchor_inputs(ep, frames, None, wan_i2v.vae, [fi], h, w, device)
    if arm == "anchor_random":
        rfi = _pick_random_hist_frame(pt, T, rng)
        if rfi is None:
            return None
        return _build_anchor_inputs(ep, frames, None, wan_i2v.vae, [int(rfi)], h, w, device)
    return None


# --retrieval 全局（避免把 args 透传进每个 helper；main 启动时设置一次）
_RETRIEVAL_MODE = "gt_oracle"


def args_retrieval_is_bank() -> bool:
    return _RETRIEVAL_MODE == "bank"


# ---------------------------------------------------------------------------
# GT 天花板过滤（真重访筛选；零生成、只 DINO + 已解码 GT 帧、不加载骨干）
#
#   复用 revisit_metric_sanity 的 GT 天花板算法（`_dino_cosine` = 两帧各 `_dino_feat` 取余弦；
#   DINO(首访,重访) vs DINO(首访,随机)），在跑生成**之前**丢弃几何匹配到的假重访 case
#   （同位姿但画面全变：d_vr ≤ d_vrand）。这些假重访会污染 v6 闸门判读。
# ---------------------------------------------------------------------------

def _dino_cosine(frame_a: np.ndarray, frame_b: np.ndarray,
                 device: torch.device) -> Optional[float]:
    """DINO(frame_a, frame_b) 余弦相似度（复用 _dino_feat；与 revisit_metric_sanity 同算法）。

    两帧各过 _dino_feat（lazy-load DINOv2，无骨干）取 pooled 特征再算 cosine。任一帧特征
    算不出（DINO 不可用）→ None（graceful，调用方据此不过滤该 case）。
    """
    fa = _dino_feat(frame_a, device)
    fb = _dino_feat(frame_b, device)
    if fa is None or fb is None:
        return None
    cos = F.cosine_similarity(fa.unsqueeze(0), fb.unsqueeze(0), dim=-1)
    return float(cos.item())


def _case_gt_ceiling(frames: np.ndarray, pt: RevisitPoint,
                     device: torch.device,
                     filter_rng: np.random.Generator) -> Tuple[Optional[float], Optional[float]]:
    """算单 case 的 GT 天花板：(d_vr, d_vrand)。

    d_vr    = DINO(GT[first_visit_frame], GT[query_frame])   —— 同地点真值一致性（天花板）。
    d_vrand = DINO(GT[first_visit_frame], 随机历史帧)         —— 随机对照（_pick_random_hist_frame）。
    两帧越界 / 随机帧取不到 → 对应值 None（graceful）。
    """
    T = frames.shape[0]
    fi = int(pt.first_visit_frame)
    qi = int(pt.query_frame)
    if not (0 <= fi < T and 0 <= qi < T):
        return None, None
    d_vr = _dino_cosine(frames[fi], frames[qi], device)
    ri = _pick_random_hist_frame(pt, T, filter_rng)
    d_vrand = _dino_cosine(frames[fi], frames[int(ri)], device) if ri is not None else None
    return d_vr, d_vrand


def _keep_real_revisit(d_vr: Optional[float], d_vrand: Optional[float], args) -> bool:
    """保留判据：d_vr - d_vrand >= min_ceiling_delta **且** d_vr >= min_ceiling_abs。

    graceful：DINO 不可用（d_vr is None）→ 保留（不因 DINO 装载失败误删全部 case）。
    d_vrand 单独 None（随机帧取不到）→ 只用绝对下限 min_ceiling_abs 判。
    """
    if d_vr is None:
        return True  # DINO 不可用 → 不过滤（避免全丢）
    if d_vr < args.min_ceiling_abs:
        return False
    if d_vrand is not None and (d_vr - d_vrand) < args.min_ceiling_delta:
        return False
    return True


def _enumerate_and_filter_cases(ep_ids, ep_groups, args, min_time_gap_frames,
                                device, height, width, diag_out=None):
    """枚举 revisit case 全集 →（可选）GT 天花板过滤真重访 → 取前 max_cases。

    **`_enumerate_cases` 的过滤增强版**，与其返回同构 `(ordered_cases, ep_cache)`，供定量主
    路径 + 定性路径共用（分片仍由调用方对返回列表按全局序号取模，确定性、并集完整、不重不漏）。

    - `--no_ceiling_filter`：**逐位回退** `_enumerate_cases(ep_ids, ep_groups, args, ...)`
      （含其内部 max_cases 截断），与加过滤前完全一致。
    - `--ceiling_filter`（默认 on）：先以 **max_cases=0** 枚举**完整**全集（否则 max_cases
      名额会被假重访占掉）→ 逐 case 算 GT 天花板过滤 → 对**已过滤**列表取前 max_cases。
      过滤发生在 max_cases 与分片**之前**（枚举全集 → 过滤 → 取 max_cases → 分片）。

    `diag_out`（可选，Phase A0 修改 A/D）：若传入 dict，则以 `(ep_id, query_frame)` 为键写入每个
    **保留 case** 的过滤诊断量 `{"d_vr", "d_vrand", "keep"}`，供 --emit_manifest 序列化进 manifest
    （天花板强度观测，见修改 D）。no_ceiling_filter 分支无诊断量，diag_out 保持为空。
    """
    if not args.ceiling_filter:
        return _enumerate_cases(ep_ids, ep_groups, args, min_time_gap_frames)

    # 枚举完整全集（禁用 max_cases：先过滤后截断，防假重访占名额）
    args_full = argparse.Namespace(**vars(args))
    args_full.max_cases = 0
    ordered_cases, ep_cache = _enumerate_cases(
        ep_ids, ep_groups, args_full, min_time_gap_frames)
    if not ordered_cases:
        return ordered_cases, ep_cache

    # 过滤用**独立** rng（不扰动主生成 rng → anchor_random 选取与 no_ceiling_filter 一致）
    filter_rng = np.random.default_rng(args.seed)

    # 按 episode 分组（保持全局序），逐 episode 解码一次算 DINO 天花板（不加载骨干）
    from collections import OrderedDict
    by_ep: "OrderedDict[str, List[RevisitPoint]]" = OrderedDict()
    for ep_id, pt in ordered_cases:
        by_ep.setdefault(ep_id, []).append(pt)

    keep_flags: Dict[Tuple[str, int], bool] = {}
    for ep_id, pts in by_ep.items():
        ep = ep_cache[ep_id]
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
        except Exception as exc:  # noqa: BLE001
            logger.warning("天花板过滤：episode %s 解码失败: %s；该 episode 全部 case 保留"
                           "（不因解码失败误删）", ep_id, exc)
            for pt in pts:
                keep_flags[(ep_id, int(pt.query_frame))] = True
            continue
        for pt in pts:
            d_vr, d_vrand = _case_gt_ceiling(frames, pt, device, filter_rng)
            keep = _keep_real_revisit(d_vr, d_vrand, args)
            keep_flags[(ep_id, int(pt.query_frame))] = keep
            if diag_out is not None:
                diag_out[(ep_id, int(pt.query_frame))] = {
                    "d_vr": d_vr, "d_vrand": d_vrand, "keep": bool(keep)}
            _s_vr = "nan" if d_vr is None else f"{d_vr:.4f}"
            _s_vrand = "nan" if d_vrand is None else f"{d_vrand:.4f}"
            if keep:
                logger.info("kept real-revisit ep=%s q=%d d_vr=%s d_vrand=%s",
                            ep_id, pt.query_frame, _s_vr, _s_vrand)
            else:
                logger.info("dropped fake-revisit ep=%s q=%d d_vr=%s d_vrand=%s",
                            ep_id, pt.query_frame, _s_vr, _s_vrand)
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    filtered = [(ep_id, pt) for (ep_id, pt) in ordered_cases
                if keep_flags.get((ep_id, int(pt.query_frame)), True)]
    n_dropped = len(ordered_cases) - len(filtered)
    logger.info("GT 天花板过滤：全集 %d → 保留 %d（丢弃 %d 假重访），"
                "min_ceiling_delta=%.3f min_ceiling_abs=%.3f",
                len(ordered_cases), len(filtered), n_dropped,
                args.min_ceiling_delta, args.min_ceiling_abs)

    # 过滤后再取前 max_cases（名额只给真重访）
    if args.max_cases > 0 and len(filtered) > args.max_cases:
        filtered = filtered[:args.max_cases]
        logger.info("过滤后取前 max_cases=%d 个", args.max_cases)

    return filtered, ep_cache


# ---------------------------------------------------------------------------
# 确定性 case manifest（Phase A0 修改 A：全体 shard 消费同一份有序 case 列表）
#
#   生产方 = --emit_manifest 路径（单进程枚举+过滤+截断 → 落盘 JSON）。
#   消费方 = --cases_manifest 路径（各 shard 反序列化 → gi %% shard_count 分片）。
#   契约：JSON 顶层 {schema_version, created, ceiling_filter, min_ceiling_delta,
#     min_ceiling_abs, max_cases, retrieval, num_cases, cases:[...]}；每个 case
#     {episode_id, query_frame, first_visit_frame, gt_past_frames, d_vr, d_vrand, keep}。
#     cases **已按 (episode_id, query_frame) 稳定排序**，各 shard 读同一顺序 → 切分不重不漏。
# ---------------------------------------------------------------------------

_MANIFEST_SCHEMA_VERSION = 1


def _revisitpoint_to_dict(ep_id: str, pt: RevisitPoint,
                          diag: Optional[Dict]) -> Dict:
    """RevisitPoint（+ 可选过滤诊断）→ JSON-safe dict。

    RevisitPoint 全字段（见 oracle_injection.py:310-315）：episode_id / query_frame /
    first_visit_frame / gt_past_frames，全部序列化以便消费方**完整还原**。
    """
    d = {
        "episode_id": str(ep_id),
        "query_frame": int(pt.query_frame),
        "first_visit_frame": int(pt.first_visit_frame),
        "gt_past_frames": [int(x) for x in (pt.gt_past_frames or [])],
    }
    if diag is not None:
        d["d_vr"] = None if diag.get("d_vr") is None else float(diag["d_vr"])
        d["d_vrand"] = None if diag.get("d_vrand") is None else float(diag["d_vrand"])
        d["keep"] = bool(diag.get("keep", True))
    return d


def _dict_to_revisitpoint(d: Dict) -> RevisitPoint:
    """JSON dict → RevisitPoint（完整还原；缺 gt_past_frames 时退化为空 list）。"""
    return RevisitPoint(
        episode_id=str(d["episode_id"]),
        query_frame=int(d["query_frame"]),
        first_visit_frame=int(d["first_visit_frame"]),
        gt_past_frames=[int(x) for x in d.get("gt_past_frames", [])],
    )


def _sort_ordered_cases(ordered_cases: List) -> List:
    """按稳定键 (episode_id, query_frame) 排序，防御 dict/set 迭代序，保证分片确定性。"""
    return sorted(ordered_cases, key=lambda ep_pt: (str(ep_pt[0]), int(ep_pt[1].query_frame)))


def _write_cases_manifest(path: str, ordered_cases: List,
                          diag_out: Optional[Dict], args) -> None:
    """把最终有序 case 列表（+ 过滤诊断量）序列化落盘为确定性 manifest（生产方）。"""
    cases_sorted = _sort_ordered_cases(ordered_cases)
    cases_json = []
    for ep_id, pt in cases_sorted:
        diag = diag_out.get((str(ep_id), int(pt.query_frame))) if diag_out else None
        cases_json.append(_revisitpoint_to_dict(ep_id, pt, diag))
    payload = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "created": datetime.now().isoformat(),
        "ceiling_filter": bool(args.ceiling_filter),
        "min_ceiling_delta": float(args.min_ceiling_delta),
        "min_ceiling_abs": float(args.min_ceiling_abs),
        "max_cases": int(args.max_cases),
        "retrieval": str(args.retrieval),
        "num_cases": len(cases_json),
        "cases": cases_json,
    }
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _load_cases_manifest(path: str) -> Tuple[List, Dict]:
    """从 manifest 反序列化有序 case 全集（消费方）。

    Returns:
        (ordered_cases, payload)
        ordered_cases: List[(ep_id, RevisitPoint)]，按 (episode_id, query_frame) 稳定排序
          （防御性 re-sort，即便生产方顺序被外部改动仍确定）。
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cases = payload.get("cases", [])
    ordered = [(str(c["episode_id"]), _dict_to_revisitpoint(c)) for c in cases]
    ordered = _sort_ordered_cases(ordered)
    return ordered, payload


def _get_episode(ep_cache: Dict, ep_id: str, ep_groups: Dict, args) -> Optional[EpisodeData]:
    """惰性取 EpisodeData：ep_cache 命中直接返回，否则 build 并缓存（manifest 模式各 shard 自建）。

    manifest 消费路径下 ep_cache 初始为空（枚举未跑），各 shard 只为**它分到的** episode
    调 build_episode_data（自行解码它分到的 episode 视频），不重跑全集枚举。
    """
    ep = ep_cache.get(ep_id)
    if ep is not None:
        return ep
    if ep_id not in ep_groups:
        logger.warning("episode %s 不在 metadata CSV（manifest 与数据集不匹配？），跳过", ep_id)
        return None
    ep = build_episode_data(ep_id, ep_groups[ep_id],
                            clip_overlap_frames=args.clip_overlap_frames)
    if ep is None:
        logger.warning("episode %s build_episode_data 失败，跳过", ep_id)
        return None
    ep_cache[ep_id] = ep
    return ep


def _resolve_ordered_cases(args, ep_ids, ep_groups, min_time_gap_frames,
                           device, height, width) -> Tuple[List, Dict]:
    """统一 case 来源：优先读 --cases_manifest（确定性、跳过枚举+过滤），否则现场枚举+过滤。

    manifest 命中 → 返回 (ordered_cases, {})，ep_cache 空（后续 _get_episode 惰性 build）。
    manifest 缺失但指定了 → warning 后回退现场枚举（单进程仍安全；多进程分片应先 emit_manifest）。
    """
    if args.cases_manifest and os.path.exists(args.cases_manifest):
        ordered_cases, _payload = _load_cases_manifest(args.cases_manifest)
        logger.info("从 cases_manifest 加载 %d 个 case（跳过现场枚举+过滤）：%s",
                    len(ordered_cases), args.cases_manifest)
        return ordered_cases, {}
    if args.cases_manifest:
        logger.warning("--cases_manifest 指定但文件不存在（%s）→ 回退现场枚举+过滤"
                       "（分片下各进程独立枚举有不重不漏风险，建议先 --emit_manifest）",
                       args.cases_manifest)
    return _enumerate_and_filter_cases(
        ep_ids, ep_groups, args, min_time_gap_frames, device, height, width)


def _shard_and_dedup(ordered_cases: List, args) -> List:
    """对有序 case 全集按 gi %% shard_count 分片 + 防御性去重（Phase A0 修改 B）。

    Returns: sel = List[(gi, ep_id, pt)]，本 shard 内 (episode_id, query_frame) 唯一。
    去重后 assert 无重复（治本靠 manifest 保证不重叠，这里是第二道防线，兜异常输入）。
    """
    if args.shard_count > 1:
        sel = [(gi, e, p) for gi, (e, p) in enumerate(ordered_cases)
               if gi % args.shard_count == args.shard_index]
    else:
        sel = [(gi, e, p) for gi, (e, p) in enumerate(ordered_cases)]

    seen: set = set()
    deduped: List = []
    dropped = 0
    for gi, e, p in sel:
        key = (str(e), int(p.query_frame))
        if key in seen:
            dropped += 1
            logger.warning("shard %d/%d 去重：丢弃重复 case ep=%s q=%d",
                           args.shard_index, args.shard_count, e, p.query_frame)
            continue
        seen.add(key)
        deduped.append((gi, e, p))
    if dropped:
        logger.warning("shard %d/%d：共去重 %d 个重复 case", args.shard_index,
                       args.shard_count, dropped)
    # assert 本 shard 内无重复 case（去重后必然成立；作显式不变量检查）
    _keys = [(str(e), int(p.query_frame)) for _gi, e, p in deduped]
    assert len(_keys) == len(set(_keys)), \
        f"shard {args.shard_index}/{args.shard_count} 仍含重复 case: {_keys}"
    return deduped


# ---------------------------------------------------------------------------
# per_window.csv（增量写，抗崩 / 抗分片）
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "episode_id", "query_frame", "first_visit_frame", "arm",
    "retrieval", "prompt_source", "num_clips", "win_start", "revisit_clip_idx",
    "video_path", "gt_first_visit_png",
    "dino_max", "dino_mean", "dino_last",      # DINO（主判据）
    "ssim_max", "ssim_mean", "ssim_last",      # SSIM（对照）
    "status",                                  # ok / dino_empty / error（修改 C：空值不静默）
]


def _append_csv(run_dir: str, record: Dict) -> None:
    csv_path = os.path.join(run_dir, "per_window.csv")
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 per_window.csv 失败: %s", exc)


def _is_bad_dino(v) -> bool:
    """dino_mean 是否为 None / NaN（打分失败标记，修改 C：不静默写空当成功）。"""
    if v is None:
        return True
    try:
        return v != v  # NaN
    except Exception:  # noqa: BLE001
        return True


def _score_and_record(all_records, run_dir, args, ep_id, pt, arm, video, gt_first,
                      mp4_path, gt_png_path, win_start, revisit_clip_idx,
                      device) -> Optional[float]:
    """算 DINO + SSIM → record → append + 落盘。返回 dino_mean（判据用）。

    修改 C（DINO 空值不静默）：打分抛异常 → status=error；dino_mean 为 None/NaN → status=
    dino_empty；均记 logger.warning（含 ep/q/arm/win）。正常行 status=ok。merge 侧 GO/NO-GO
    统计排除 status != ok 的行。
    """
    status = "ok"
    try:
        metrics = _revisit_consistency(video, gt_first, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DINO/SSIM 打分抛异常 ep=%s q=%d arm=%s win_start=%d：%s",
                       ep_id, pt.query_frame, arm, win_start, exc)
        metrics = {}
        status = "error"
    record = {
        "episode_id": ep_id, "query_frame": pt.query_frame,
        "first_visit_frame": pt.first_visit_frame, "arm": arm,
        "retrieval": args.retrieval, "prompt_source": args.prompt_source,
        "num_clips": args.num_clips, "win_start": win_start,
        "revisit_clip_idx": revisit_clip_idx,
        "video_path": mp4_path, "gt_first_visit_png": gt_png_path,
        "dino_max": metrics.get("revisit_consistency_dino_max"),
        "dino_mean": metrics.get("revisit_consistency_dino_mean"),
        "dino_last": metrics.get("revisit_consistency_dino_last"),
        "ssim_max": metrics.get("revisit_consistency_max"),
        "ssim_mean": metrics.get("revisit_consistency_mean"),
        "ssim_last": metrics.get("revisit_consistency_last"),
    }
    if status == "ok" and _is_bad_dino(record["dino_mean"]):
        logger.warning("DINO 打分为空/NaN（不静默）ep=%s q=%d arm=%s win_start=%d "
                       "→ 标记 status=dino_empty（不计入 GO/NO-GO）",
                       ep_id, pt.query_frame, arm, win_start)
        status = "dino_empty"
    record["status"] = status
    logger.info("ep=%s q=%d [%s] dino_mean=%s ssim_mean=%s status=%s",
                ep_id, pt.query_frame, arm, record["dino_mean"], record["ssim_mean"], status)
    all_records.append(record)
    _append_csv(run_dir, record)
    dm = record["dino_mean"]
    return float(dm) if (dm is not None and not _is_bad_dino(dm)) else None


# ---------------------------------------------------------------------------
# 单 case 处理（定量）
# ---------------------------------------------------------------------------

def _process_case(wan_i2v, ep, ep_id, pt, frames, args, device, rng,
                  videos_root, run_dir, all_records, arms, h_pix, w_pix):
    """处理单个 (episode, revisit case)：逐臂多 clip 自回归 + 重访 clip 打分落盘。"""
    T = ep.poses.shape[0]
    q_dir = os.path.join(videos_root, ep_id, f"q{pt.query_frame}")
    os.makedirs(q_dir, exist_ok=True)
    gt_first = frames[pt.first_visit_frame]  # [3,H,W]
    gt_png_path = os.path.join(q_dir, "gt_first_visit.png")
    _save_frame_png(gt_first, gt_png_path)

    num_clips = max(1, args.num_clips)
    revisit_clip_idx = num_clips - 1

    # ---- 逐臂 anchor 构造（off 无 anchor）----
    anchors: Dict[str, Optional[Tuple[torch.Tensor, np.ndarray]]] = {}
    for arm in arms:
        if arm == "off":
            anchors[arm] = None
            continue
        ai = _build_arm_anchor(wan_i2v, ep, frames, pt, arm, h_pix, w_pix, device, rng)
        if ai is None:
            logger.warning("ep=%s q=%d [%s]：anchor 构造为空 → 跳过该臂",
                           ep_id, pt.query_frame, arm)
        anchors[arm] = ai

    # ---- 逐臂 rollout + 打分 ----
    for arm in arms:
        if arm != "off" and anchors.get(arm) is None:
            continue
        mp4_path = os.path.join(q_dir, f"{arm}_revisit.mp4")
        # 可续：重访 clip mp4 已存在 → 读回重算指标，跳过 rollout
        if os.path.exists(mp4_path):
            video = _read_video_back(mp4_path)
            if video is not None:
                logger.info("ep=%s q=%d [%s]：revisit mp4 已存在 → 读回重算指标",
                            ep_id, pt.query_frame, arm)
                # win_start 仅用于记录；读回路径按窗口锚定公式复算（与 rollout 一致）
                _ws = pt.query_frame - (num_clips - 1) * args.frame_num
                _ws = int(max(0, min(_ws, max(0, T - num_clips * args.frame_num))))
                _score_and_record(all_records, run_dir, args, ep_id, pt, arm,
                                  video, gt_first, mp4_path, gt_png_path, _ws,
                                  revisit_clip_idx, device)
                continue

        anchor_latent = anchors[arm][0] if anchors.get(arm) is not None else None
        anchor_poses = anchors[arm][1] if anchors.get(arm) is not None else None
        clip_videos, win_start = _rollout_case_arm(
            wan_i2v, ep, pt, frames, arm, anchor_latent, anchor_poses,
            args, device, num_clips)
        if not clip_videos:
            logger.warning("ep=%s q=%d [%s]：无 clip 生成，跳过", ep_id, pt.query_frame, arm)
            continue
        revisit_video = clip_videos[-1]   # [3,frame_num,H,W]，重访 clip
        _save_video(revisit_video, mp4_path, fps=args.fps)
        _score_and_record(all_records, run_dir, args, ep_id, pt, arm,
                          revisit_video, gt_first, mp4_path, gt_png_path,
                          win_start, revisit_clip_idx, device)


# ---------------------------------------------------------------------------
# 定性渲染（off vs anchor_ideal 的多 clip 自回归长视频，眼看重访段）
# ---------------------------------------------------------------------------

def _render_case_qual(wan_i2v, ep, ep_id, pt, frames, args, device, rng,
                      videos_root, qual_arms, num_clips, h_pix, w_pix):
    """对单个 case 渲完整多 clip 自回归长视频（每臂一条 long_video_<arm>.mp4）。不算 DINO。"""
    q_dir = os.path.join(videos_root, ep_id, f"q{pt.query_frame}")
    os.makedirs(q_dir, exist_ok=True)
    try:
        _save_frame_png(frames[int(pt.first_visit_frame)],
                        os.path.join(q_dir, "gt_first_visit.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("定性 ep=%s q=%d：存 gt_first_visit.png 失败（非致命）: %s",
                       ep_id, pt.query_frame, exc)

    for arm in qual_arms:
        out_path = os.path.join(q_dir, f"long_video_{arm}.mp4")
        if os.path.exists(out_path):
            logger.info("定性 ep=%s q=%d [%s]：long_video 已存在 → 跳过",
                        ep_id, pt.query_frame, arm)
            continue
        anchor = None
        if arm != "off":
            anchor = _build_arm_anchor(wan_i2v, ep, frames, pt, arm, h_pix, w_pix, device, rng)
            if anchor is None:
                logger.warning("定性 ep=%s q=%d [%s]：anchor 构造为空 → 跳过该臂",
                               ep_id, pt.query_frame, arm)
                continue
        a_lat = anchor[0] if anchor is not None else None
        a_pose = anchor[1] if anchor is not None else None
        clip_videos, win_start = _rollout_case_arm(
            wan_i2v, ep, pt, frames, arm, a_lat, a_pose, args, device, num_clips)
        if not clip_videos:
            logger.warning("定性 ep=%s q=%d [%s]：无 clip 生成，跳过保存",
                           ep_id, pt.query_frame, arm)
            continue
        full_video = np.concatenate(clip_videos, axis=1)   # [3, total_F, H, W]
        _save_video(full_video, out_path, fps=args.fps)
        logger.info("定性 ep=%s q=%d [%s]：长视频已存 → %s（%d 帧 @ %dfps，win_start=%d）",
                    ep_id, pt.query_frame, arm, out_path, full_video.shape[1],
                    args.fps, win_start)


def _render_qualitative(wan_i2v, ep_groups, ep_ids, args, device, rng,
                        height, width, videos_root, min_time_gap_frames, h_pix, w_pix):
    """定性渲染主驱动：枚举 revisit case（复用 _enumerate_cases）→ 分片 → 逐 case 渲长视频。"""
    _requested = [a.strip() for a in args.qual_arms.split(",") if a.strip() in ALL_ARMS]
    # off 在前（baseline）；按 ALL_ARMS 顺序稳定
    qual_arms = [a for a in ALL_ARMS if a in _requested]
    if not qual_arms:
        qual_arms = ["off", "anchor_ideal"]
    num_clips = max(1, args.num_clips_qual)
    logger.info("定性渲染：qual_arms=%s num_clips=%d frame_num=%d fps=%d",
                qual_arms, num_clips, args.frame_num, args.fps)

    ordered_cases, ep_cache = _resolve_ordered_cases(
        args, ep_ids, ep_groups, min_time_gap_frames, device, height, width)
    if not ordered_cases:
        logger.error("定性渲染：无 revisit case（或全被天花板过滤丢弃 / manifest 为空），退出。")
        return

    sel = _shard_and_dedup(ordered_cases, args)
    logger.info("定性渲染：case 全集 %d，本分片处理 %d（shard %d/%d，max_cases=%d 已在切分前应用）",
                len(ordered_cases), len(sel), args.shard_index, args.shard_count, args.max_cases)
    if not sel:
        logger.error("定性渲染：shard %d/%d 分到 0 个 case，退出。",
                     args.shard_index, args.shard_count)
        return

    from collections import OrderedDict
    by_ep: "OrderedDict[str, List]" = OrderedDict()
    for _gi, ep_id, pt in sel:
        by_ep.setdefault(ep_id, []).append(pt)

    for ep_id, pts in by_ep.items():
        ep = _get_episode(ep_cache, ep_id, ep_groups, args)
        if ep is None:
            continue
        logger.info("定性 Episode %s：T=%d，本分片 %d 个 case", ep_id, ep.poses.shape[0], len(pts))
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
        except Exception as exc:  # noqa: BLE001
            logger.warning("定性 episode %s 解码失败: %s；跳过", ep_id, exc)
            continue
        for pt in pts:
            try:
                _render_case_qual(wan_i2v, ep, ep_id, pt, frames, args, device, rng,
                                  videos_root, qual_arms, num_clips, h_pix, w_pix)
            except Exception as exc:  # noqa: BLE001
                logger.exception("定性 case 渲染失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                continue
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 判据 / summary
# ---------------------------------------------------------------------------

def _verdict(all_records: List[Dict], arms: List[str], margin: float, run_dir: str) -> str:
    """逐 case 三臂表 + 均值 + GO/NO-GO 判决，打印并写 summary.md。

    GO ⇔ anchor_ideal 的 DINO 均值 > off + margin **且** > anchor_random + margin。
    """
    cases: Dict[tuple, Dict[str, float]] = {}
    for r in all_records:
        # 修改 C：status != ok 的行（dino_empty / error）不计入 GO/NO-GO
        if str(r.get("status", "ok")).strip().lower() not in ("", "ok"):
            continue
        key = (r["episode_id"], r["query_frame"])
        dm = r["dino_mean"]
        if dm is None or _is_bad_dino(dm):
            continue
        cases.setdefault(key, {})[r["arm"]] = float(dm)

    lines: List[str] = []
    lines.append("# v6 latent-concat 理想注入诊断（S-V5 / Step 44）—— GO/NO-GO\n")
    lines.append(f"- timestamp: {datetime.now().isoformat()}\n")
    lines.append(f"判据: GO ⇔ anchor_ideal 的 DINO 均值 > off + {margin} **且** "
                 f"> anchor_random + {margin}\n")
    lines.append("（GO = latent-concat 通道在多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 训练；"
                 "NO-GO = 升级更重干预 LoRA/解冻）\n")

    header = "| episode | query | " + " | ".join(arms) + " | ideal−off | ideal−random |"
    sep = "|" + "---|" * (len(arms) + 4)
    lines.append("\n## 逐 case（frame-aligned DINO mean，重访 clip vs GT 首访帧）\n")
    lines.append(header)
    lines.append(sep)

    means: Dict[str, List[float]] = {a: [] for a in arms}
    for key in sorted(cases.keys()):
        row = cases[key]
        cells = []
        for a in arms:
            v = row.get(a)
            cells.append("nan" if v is None else f"{v:.4f}")
            if v is not None:
                means[a].append(v)
        d_off = (row.get("anchor_ideal", float("nan")) - row.get("off", float("nan")))
        d_rnd = (row.get("anchor_ideal", float("nan")) - row.get("anchor_random", float("nan")))
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) +
                     f" | {d_off:+.4f} | {d_rnd:+.4f} |")

    mean_off = float(np.mean(means["off"])) if means.get("off") else float("nan")
    mean_ideal = float(np.mean(means["anchor_ideal"])) if means.get("anchor_ideal") else float("nan")
    mean_rnd = float(np.mean(means["anchor_random"])) if means.get("anchor_random") else float("nan")

    lines.append("\n## 均值\n")
    for a in arms:
        mv = float(np.mean(means[a])) if means.get(a) else float("nan")
        lines.append(f"- {a}: {mv:.4f}  (n={len(means.get(a, []))})")

    go = (
        ("anchor_ideal" in arms and "off" in arms and "anchor_random" in arms)
        and (mean_ideal == mean_ideal)  # not nan
        and (mean_off == mean_off) and (mean_rnd == mean_rnd)
        and (mean_ideal > mean_off + margin)
        and (mean_ideal > mean_rnd + margin)
    )
    verdict = "GO" if go else "NO-GO"
    route = ("latent-concat 通道多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 配方训练"
             "（通道维 concat + negative-RoPE + r128 LoRA on 全 DiT 线性层 + 位姿）"
             if go else
             "latent-concat 多 clip 端到端不成立 → 升级更重干预（LoRA / 解冻）")
    lines.append("\n## 判决\n")
    lines.append(f"**{verdict}** — anchor_ideal={mean_ideal:.4f} vs off={mean_off:.4f} "
                 f"(Δ={mean_ideal - mean_off:+.4f}) vs anchor_random={mean_rnd:.4f} "
                 f"(Δ={mean_ideal - mean_rnd:+.4f})，margin={margin}")
    lines.append(f"\n**路由**：{route}")

    summary = "\n".join(lines) + "\n"
    try:
        with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 summary.md 失败: %s", exc)
    print("\n" + summary)
    logger.info("verdict=%s", verdict)
    return verdict


# ---------------------------------------------------------------------------
# 模型装载（镜像 stage1_upperbound main：WanI2V + 转换骨干；memory 不参与）
# ---------------------------------------------------------------------------

def _load_pipeline(args, device):
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS

    ckpt_dir = args.ckpt_dir
    if args.ft_model_dir:
        _fake = argparse.Namespace(ckpt_dir=args.ckpt_dir)
        ckpt_dir = _load_ft_model_and_prepare_ckpt(_fake)

    cfg = WAN_CONFIGS["i2v-A14B"]
    local_rank = device.index if device.type == "cuda" and device.index is not None else 0
    wan_i2v = WanI2V(
        config=cfg, checkpoint_dir=ckpt_dir, device_id=local_rank, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False,
    )
    logger.info("转换 pipeline → WanModelWithMemory（memory 不参与本实验，仅复用 DiT 主干）...")
    wan_i2v = _convert_pipeline_to_memory(
        wan_i2v, memory_ckpt_path=None,
        high_model_dir=args.ft_high_model_dir, low_model_dir=args.ft_model_dir,
    )
    return wan_i2v


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.emit_manifest and args.cases_manifest:
        raise SystemExit("--emit_manifest 与 --cases_manifest 互斥，不能同时给定。")

    global _RETRIEVAL_MODE
    _RETRIEVAL_MODE = args.retrieval
    if args.retrieval == "bank":
        # 占位：bank 检索版在下个组件实现（本组件只做 gt_oracle 隔离注入）。
        raise NotImplementedError(
            "--retrieval bank 尚未实现：本组件只做 gt_oracle（GT 首访帧）隔离注入闸门；"
            "bank 检索版（正常推理用 bank 检索注入）留给下一个组件。请用 --retrieval gt_oracle。")

    arms = [a for a in ALL_ARMS if a in
            {x.strip() for x in args.arms.split(",") if x.strip()}]
    if not arms:
        arms = list(DEFAULT_ARMS)

    # oracle_injection 全局（_build_* 读它；本脚本不调那些，仍按 stage1 习惯设置）
    _oracle_inj._SIZE_HW = tuple(args.size.split("*"))
    _oracle_inj._ORACLE_CLIP_FRAMES = args.frame_num

    run_name = args.run_name or default_run_name("v6_latentconcat_diag")
    run_dir = eval_run_dir("v6", run_name, args.tag)
    videos_root = os.path.join(str(run_dir), "videos")
    os.makedirs(videos_root, exist_ok=True)
    log_path = os.path.join(str(run_dir), "diag.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    snapshot_config(run_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})
    logger.info("latentconcat_ideal_diag run_dir=%s | arms=%s | retrieval=%s | prompt_source=%s",
                run_dir, arms, args.retrieval, args.prompt_source)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    h_pix, w_pix = _resize_hw(height, width, args)  # anchor VAE encode 须与 query 同 h/w
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))
    logger.info("Args: %s", vars(args))

    # ---- episode CSV ----
    ep_filter = None
    if args.episode_ids:
        ep_filter = [s.strip() for s in args.episode_ids.split(",") if s.strip()]
    ep_groups = load_episode_clips(args.dataset_dir, args.metadata, episode_ids_filter=ep_filter)
    ep_ids = list(ep_groups.keys())
    if args.max_episodes > 0 and len(ep_ids) > args.max_episodes:
        ep_ids = ep_ids[:args.max_episodes]
    if not ep_ids:
        logger.error("无 episode 可处理，退出。")
        return

    # ---- --emit_manifest：单进程枚举+过滤+截断 → 落盘确定性 manifest → 退出（不加载骨干/不生成）----
    # 天花板过滤只需 DINO（lazy-load，无骨干）+ 解码 GT 帧，不需要 WanI2V pipeline，故在 _load_pipeline
    # 之前完成并 return，避免为「只出 manifest」白白装载 14B 骨干。
    if args.emit_manifest:
        diag_out: Dict = {}
        ordered_cases, _ep_cache = _enumerate_and_filter_cases(
            ep_ids, ep_groups, args, min_time_gap_frames, device, height, width,
            diag_out=diag_out)
        _write_cases_manifest(args.emit_manifest, ordered_cases, diag_out, args)
        logger.info("cases manifest 已写出（%d cases）→ %s；不生成视频，退出。",
                    len(ordered_cases), args.emit_manifest)
        return

    # ---- 加载 pipeline ----
    wan_i2v = _load_pipeline(args, device)

    # ---- 定性渲染路径（--render_qual）：只渲长视频，不算 DINO / 不判 GO/NO-GO，提前返回 ----
    if args.render_qual:
        _render_qualitative(wan_i2v, ep_groups, ep_ids, args, device, rng,
                            height, width, videos_root, min_time_gap_frames, h_pix, w_pix)
        logger.info("定性渲染完成。输出目录: %s", run_dir)
        return

    all_records: List[Dict] = []
    run_dir_str = str(run_dir)

    # ---- 定量：确定性 case 全集（优先 manifest；否则现场枚举+过滤）→ 分片 + 去重 ----
    ordered_cases, ep_cache = _resolve_ordered_cases(
        args, ep_ids, ep_groups, min_time_gap_frames, device, height, width)
    if not ordered_cases:
        logger.error("无 revisit case（或全被天花板过滤丢弃 / manifest 为空），退出。")
        return
    sel = _shard_and_dedup(ordered_cases, args)
    logger.info("分片 %d/%d：case 全集 %d，本分片处理 %d（全局序号 %% %d == %d）",
                args.shard_index, args.shard_count, len(ordered_cases), len(sel),
                args.shard_count, args.shard_index)
    if not sel:
        logger.error("shard %d/%d 分到 0 个 case（全集太小？），退出。",
                     args.shard_index, args.shard_count)
        return

    from collections import OrderedDict
    by_ep: "OrderedDict[str, List]" = OrderedDict()
    for _gi, ep_id, pt in sel:
        by_ep.setdefault(ep_id, []).append(pt)

    for ep_id, pts in by_ep.items():
        ep = _get_episode(ep_cache, ep_id, ep_groups, args)
        if ep is None:
            continue
        T = ep.poses.shape[0]
        logger.info("Episode %s: T=%d, 本分片 %d 个 case", ep_id, T, len(pts))
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码失败: %s；跳过", ep_id, exc)
            continue
        for pt in pts:
            try:
                _process_case(wan_i2v, ep, ep_id, pt, frames, args, device, rng,
                              videos_root, run_dir_str, all_records, arms, h_pix, w_pix)
            except Exception as exc:  # noqa: BLE001
                logger.exception("case 处理失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                continue
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_records:
        logger.error("无任何记录（无 case / 全失败），退出。")
        return

    _verdict(all_records, arms, args.go_margin, run_dir_str)
    logger.info("Done. 输出目录: %s", run_dir)


if __name__ == "__main__":
    main()
