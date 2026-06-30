"""
latentconcat_infer.py — v6 latent-concat 正常推理（bank 检索 + anchor-concat，可部署长视频）
============================================================================================================

**目的**（decisions.md 讨论 10 + 通用闸门补记；承 [[F-30]] in-context KV 判死 → latent-concat
主线）：把 v6 的帧维 anchor-concat（零训练）从 ideal_diag 的「GT oracle 首访帧隔离注入」升级到
**「bank 检索的历史帧」**，做**可部署的多 clip 自回归长视频推理 demo**。

与 `latentconcat_ideal_diag.py` 的关系（落点 b：新建镜像，**不改 ideal_diag**）：
  - ideal_diag = 诊断脚本（三臂 off/anchor_ideal/anchor_random + frame-aligned DINO + GO/NO-GO），
    anchor 来源 = GT 首访帧（隔离注入，每 clip 固定同一 anchor）。其 `--retrieval bank` 是
    占位 NotImplemented。**本文件不动它，gt_oracle/ideal 诊断路径逐位不变。**
  - 本文件 = 正常推理（部署形态），anchor 来源 = **bank 检索**：每 clip 生成后把帧写入 3-tier
    bank（populate，复用 infer_v5._update_bank_from_clip），下一 clip 按当前 clip 绝对位置从 bank
    `retrieve_revisit`（绝对位置键 + 排近期，[[F-14]]/OP-2，p@1=0.542）检索历史帧 → 把检索到的
    历史帧 latent + 真实位姿当 **anchor** 走 v6 anchor-concat 注入（复用 stage1
    `_generate_with_anchor`，无 memory cross-attn、无 in-context KV）。检索不到 → 该 clip 不注入
    anchor（`_generate_with_anchor` 的 anchor=None 退化为纯 i2v）。

**注入后端 = 帧维 anchor-concat（Context-as-Memory / FramePack 式）**：把检索到的历史帧 latent
当尾部 clean anchor 槽 concat 进 latent 时间维，走骨干**已训练**的 36 通道 i2v 条件流；anchor 位姿
挂原生 plucker（相对**当前 clip** query[0] 重算）。**骨干全冻、no_grad、无新可训练参数、不挂 LoRA、
不用 negative-RoPE**（零训练 anchor-concat 推理；训练版是另一个组件）。

核心流程（多 clip 自回归，镜像 infer_v5:589-708，注入从 in-context KV encoder 换成 anchor-concat）：
  1. 模型装载：复用 ideal_diag._load_pipeline（WanI2V + 转 WanModelWithMemory；memory 不参与，
     仅复用冻结 DiT 主干，与 ideal_diag 同源）。
  2. bank = ThreeTierMemoryBank（默认值对齐 train_v5 / infer_v5）。
  3. for clip_idx in range(num_clips)：
       a. 切本 clip poses/actions/intrinsics（[c*frame_num:(c+1)*frame_num]，infer_v5 口径）。
       b. 检索 anchor（仅 clip_idx>0 且 bank 非空）：query_location = 本 clip 第一 latent 帧绝对
          位置 → bank.retrieve_revisit(return_latents=True) → 取 top num_anchor_frames 帧的 latent
          ([16,n,lat_h,lat_w]) + 真实位姿（[n,4,4]，按 timestep 从 timestep→pose 映射取回）。
       c. 生成：_generate_with_anchor(current_img, anchor_latent, anchor_poses, clip_poses,
          clip_actions, clip_intrinsics, clip_args)（per-clip 数据 prompt + seed=base+clip_idx）。
          anchor=None 时退化纯 i2v。
       d. bank 更新（surprise-independent，复用 infer_v5._update_bank_from_clip）：VAE encode 本 clip
          → 逐 latent 帧 update（location=绝对位置，surprise=0 全进 long tier）；同时把每 latent 帧
          的**完整 c2w 位姿**记入 timestep→pose 映射（anchor 位姿挂载用，bank 只存 location[3]，
          plucker 需完整 4×4）。
       e. 末帧 → 下一 clip current_img（autoregressive，镜像 infer_v5）。
  4. 拼接所有 clip → 一条 long_video.mp4，存 v6/infer/<run>/<tag>/<episode_id>/。

**数据入口（二选一，镜像 infer_v5）**：
  A. action_path 模式：--image 首帧 + --action_path 目录（poses.npy/action.npy/intrinsics.npy 长轨迹）。
  B. episode 模式：--episode_id（或 'first'/'top'/'all'）+ --dataset_dir + --metadata，从重访数据集
     整条 episode 取轨迹（demo 用，省手动拼 action；可逐 clip 取数据 prompt + 可选 --score）。

**prompt 对齐**：`--prompt_source data`（默认，episode 模式逐 clip 从 clip 目录 prompt.txt 取数据
prompt，复用 ideal_diag._clip_prompt，与训练同源）/ `fixed`（用 --prompt）。action_path 模式无
per-clip prompt.txt → 始终用 --prompt。

**评分（可选）**：默认是纯推理 demo，**不算 DINO / 不判 GO**。`--score`（仅 episode 模式）旁路复用
oracle_injection._revisit_consistency：检测重访点 → 对落在生成范围内的重访 clip vs GT 首访帧算
frame-aligned DINO/SSIM，写 scores.csv（仅供旁观，不判 GO/NO-GO）。

**分片**（复用 --shard_index/--shard_count）：切分轴 = episode 全局序号（episode 模式多 ep 时按
序号取模分片，多卡并行推理）。action_path 模式单轨迹，shard_count>1 时仅 shard 0 处理。

服务器跑前置：`export TMPDIR=/tmp`（否则 kill 掉的 run 会在仓库留 pymp-* 孤儿）。
本地无 torch/CUDA 真跑不动；--help / py_compile 走通即可（真跑待服务器）。
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path（与 latentconcat_ideal_diag / infer_v5 一致）
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
# 复用既有组件（import，不重写；最大限度复用 infer_v5 bank 循环 + stage1 anchor-concat）
# ---------------------------------------------------------------------------
# infer_v5：bank populate（VAE encode + 逐 latent 帧 update，含 location）+ 每 latent 帧绝对位置
from pipeline.v5.infer_v5 import (  # noqa: E402
    _per_latent_abs_locations,
    _update_bank_from_clip,
)

# stage1_upperbound：**已验证**帧维 anchor-concat 生成（[[F-23]] 12/12）
from pipeline.eval.stage1_upperbound import _generate_with_anchor  # noqa: E402

# latentconcat_ideal_diag：per-clip 数据 prompt + 模型装载（复用，确保与诊断同源；不改 ideal_diag）
from pipeline.v6.latentconcat_ideal_diag import (  # noqa: E402
    _clip_prompt,
    _clip_args,
    _load_pipeline,
)

# episode 加载 / 解码（与 infer_v5 / ideal_diag 同源）
from pipeline.eval.retrieval_probe import (  # noqa: E402
    load_episode_clips,
    build_episode_data,
    _decode_episode_video,
)

# oracle_injection：可选评分（重访点检测 + frame-aligned DINO/SSIM）+ 视频/帧 IO
from pipeline.eval.oracle_injection import (  # noqa: E402
    _find_revisit_points,
    _revisit_consistency,
    _save_video,
    _save_frame_png,
)
import pipeline.eval.oracle_injection as _oracle_inj  # noqa: E402

from pipeline.common.paths import (  # noqa: E402
    infer_run_dir,
    snapshot_config,
    default_run_name,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "v6 latent-concat 正常推理（bank 检索历史帧 + anchor-concat，可部署多 clip 自回归长视频）。"
            "服务器跑前置：export TMPDIR=/tmp。"
        )
    )

    # ---- 模型权重（与 ideal_diag._load_pipeline 对齐）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="lingbot-world 预训练权重目录（含 low/high noise_model / VAE / T5）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="（可选）v4 low_noise_model checkpoint 目录（仅影响 DiT 主干）")
    p.add_argument("--ft_high_model_dir", type=str, default=None,
                   help="（可选）dual 训练 high_noise_model 目录")

    # ---- 数据（二选一：action_path 模式 / episode 模式，镜像 infer_v5）----
    p.add_argument("--image", type=str, default=None,
                   help="action_path 模式首帧图像（PNG/JPG）。episode 模式忽略（首帧取 frames[0]）。")
    p.add_argument("--action_path", type=str, default=None,
                   help="action_path 模式目录：poses.npy[T,4,4]/action.npy[T,4]/intrinsics.npy[T,4]"
                        "（整条 GT 控制轨迹）。episode 模式忽略。")
    p.add_argument("--episode_id", type=str, default=None,
                   help="episode 模式 episode id；'first'/'top'=CSV 第一个；'all'=跑全部（配 shard）。"
                        "设此参数即 episode 模式，--dataset_dir 必填。")
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="含 metadata CSV + clips/ 的数据集根（episode 模式必填）。")
    p.add_argument("--metadata", type=str, default="metadata_verify_train.csv",
                   help="相对 dataset_dir 的 CSV 路径（默认 metadata_verify_train.csv）。")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="episode 模式：仅跑这些 episode（逗号分隔），覆盖 --episode_id 的单 ep 选择。")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="episode 模式：0=不限；>0 时取前 N 个 episode（切分前应用）。")

    # ---- 注入源 / anchor ----
    p.add_argument("--retrieval", type=str, default="bank",
                   choices=["bank", "none"],
                   help="anchor 记忆源：bank（默认，3-tier bank retrieve_revisit 检索历史帧）/ "
                        "none（纯 base i2v 自回归，不检索/不注入 anchor，base 对照）。")
    p.add_argument("--num_anchor_frames", type=int, default=1,
                   help="每 clip 注入的 anchor 帧数上限（取 retrieve_revisit top-N，默认 1；token "
                        "预算 R3：每 anchor 帧占完整 1560 token，少量为宜）。")

    # ---- 多 clip 自回归 ----
    p.add_argument("--num_clips", type=int, default=12,
                   help="多 clip 自回归 clip 数（默认 12；episode 模式上限 = T//frame_num）")
    p.add_argument("--frame_num", type=int, default=81,
                   help="每 clip 帧数（4n+1，默认 81）")

    # ---- 生成参数（_generate_with_anchor 直接读 args 这些字段）----
    p.add_argument("--num_inference_steps", type=int, default=70,
                   help="diffusion 采样步数（默认 70）")
    p.add_argument("--sample_shift", type=float, default=10.0,
                   help="sigma shift（默认 10.0）")
    p.add_argument("--guide_scale", type=float, default=5.0,
                   help="CFG scale（默认 5.0）")
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W（默认 480*832）")
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay",
                   help="prompt_source=fixed 时的固定 prompt（data 模式仅作 prompt.txt 缺失回退）")
    p.add_argument("--prompt_source", type=str, default="data",
                   choices=["data", "fixed"],
                   help="data（默认，episode 模式逐 clip 从 clip 目录 prompt.txt 取，与训练同源）/ "
                        "fixed（所有 clip 用 --prompt）。action_path 模式无 prompt.txt → 始终 fixed。")
    p.add_argument("--seed", type=int, default=42,
                   help="基础 seed；每 clip 用 seed+clip_idx（镜像 infer_v5 自回归）")
    p.add_argument("--fps", type=int, default=16, help="输出视频 fps（默认 16）")
    p.add_argument("--device", type=str, default="cuda:0")

    # ---- bank 参数（默认值对齐 train_v5 / infer_v5）----
    p.add_argument("--short_cap", type=int, default=1)
    p.add_argument("--medium_cap", type=int, default=8)
    p.add_argument("--long_cap", type=int, default=256,
                   help="LongTermBank 容量（默认 256，对齐 infer_v5 长视频 demo）")
    p.add_argument("--surprise_threshold", type=float, default=0.4)
    p.add_argument("--stability_threshold", type=float, default=0.2)
    p.add_argument("--novelty_threshold", type=float, default=0.7)
    p.add_argument("--half_life", type=float, default=10.0)
    p.add_argument("--dup_threshold", type=float, default=0.95)
    p.add_argument("--revisit_top_k", type=int, default=5,
                   help="retrieve_revisit 取回帧数上限（默认 5，对齐 train_v5）")
    p.add_argument("--revisit_min_gap_frames", type=int, default=0,
                   help="retrieve_revisit 排除 timestep 距离 < 此值的近邻帧（默认 0，对齐 train_v5）")

    # ---- 可选评分（旁路，仅 episode 模式；不判 GO）----
    p.add_argument("--score", action="store_true", default=False,
                   help="可选：episode 模式下检测重访点，对落在生成范围内的重访 clip vs GT 首访帧算 "
                        "frame-aligned DINO/SSIM（复用 _revisit_consistency），写 scores.csv。"
                        "默认 False（纯推理 demo，不算 DINO / 不判 GO）。")
    p.add_argument("--hit_dist", type=float, default=40.0)
    p.add_argument("--hit_yaw", type=float, default=30.0)
    p.add_argument("--intermediate_separation", type=float, default=100.0)
    p.add_argument("--min_time_gap_sec", type=float, default=5.0)
    p.add_argument("--max_revisit_points", type=int, default=2)

    # ---- 产出 ----
    p.add_argument("--run_name", type=str, default=None,
                   help="infer run 名（默认 default_run_name('v6_latentconcat_infer')）")
    p.add_argument("--tag", type=str, default="long_video",
                   help="infer 场景 tag（默认 long_video）")

    # ---- 分片（additive：shard_count 默认 1 → 逐位与单进程一致）----
    p.add_argument("--shard_index", type=int, default=0,
                   help="当前分片索引（0-based），episode 模式多 ep 时按 episode 全局序号取模分片。")
    p.add_argument("--shard_count", type=int, default=1,
                   help="总分片数。默认 1=不分片；>1 时按 episode 全局序号取模分片。")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 每 latent 帧的完整 c2w 位姿（anchor 位姿挂载用；与 _per_latent_abs_locations 同口径）
# ---------------------------------------------------------------------------

def _per_latent_abs_poses(poses_np: np.ndarray, lat_f: int) -> np.ndarray:
    """从 clip 的 [F,4,4] 位姿取每个 latent 帧的**完整 c2w 矩阵**（poses[::4] 裁/补到 lat_f）。

    与 infer_v5._per_latent_abs_locations 完全同口径（后者取 poses[::4,:3,3] 的平移；本函数取
    poses[::4] 的完整 4×4）→ 同一 latent 帧的 _per_latent_abs_poses(...)[t][:3,3] ==
    _per_latent_abs_locations(...)[t]，保证 bank 存的 location 与本映射存的 pose 平移逐元素一致。

    bank 只存 location[3]（retrieve_revisit 用），但 v6 anchor-concat 的 plucker 需要完整 4×4
    c2w（旋转进 plucker 射线）→ 用本映射在 timestep→pose 字典里另存完整位姿。

    Args:
        poses_np: [F,4,4] 该 clip 的绝对 c2w 位姿。
        lat_f:    该 clip 的 latent 帧数（= VAE encode 后 latent.shape[1]）。

    Returns:
        [lat_f,4,4] np.float32 绝对 c2w。
    """
    sub = np.asarray(poses_np, dtype=np.float32)[::4]  # [~F/4, 4, 4]
    if sub.shape[0] >= lat_f:
        sub = sub[:lat_f]
    elif sub.shape[0] > 0:
        pad = np.repeat(sub[-1:], lat_f - sub.shape[0], axis=0)
        sub = np.concatenate([sub, pad], axis=0)
    return sub.astype(np.float32)  # [lat_f, 4, 4]


# ---------------------------------------------------------------------------
# bank 检索 → anchor（latent + 完整位姿）
# ---------------------------------------------------------------------------

def _retrieve_anchor(
    bank,
    timestep_to_pose: Dict[int, np.ndarray],
    query_location: torch.Tensor,
    query_timestep: int,
    num_anchor_frames: int,
    top_k: int,
    min_gap_frames: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """从 bank 检索历史帧 → 构造 anchor (latent [16,n,lat_h,lat_w], poses [n,4,4]) 或 None。

    复用 bank.retrieve_revisit（绝对位置键 + 排近期，[[F-14]]/OP-2）；检索结果 frames/latents 同序
    （retrieve_by_location 升序 = 位置最近优先）。取 top num_anchor_frames：
      - anchor_latent：latents[:n] 形状 [n,16,lat_h,lat_w] → permute 成 [16,n,lat_h,lat_w]
        （_generate_with_anchor 的 anchor 时间维布局）。
      - anchor_poses ：按 frames[i].timestep 从 timestep_to_pose 取回**完整 4×4** c2w（bank 只存
        location[3]，plucker 需完整位姿）。某帧 timestep 不在映射里（理论不应发生）→ 跳过该帧。

    Returns:
        (anchor_latent, anchor_poses) 或 None（bank 空 / 检索空 / 无可用位姿）。
    """
    if bank.size() == 0:
        return None
    frames, latents = bank.retrieve_revisit(
        query_location=query_location,
        query_timestep=query_timestep,
        top_k=top_k,
        min_gap_frames=min_gap_frames,
        device=device,
        return_latents=True,
    )
    if not frames or latents is None:
        return None

    n = min(max(1, int(num_anchor_frames)), len(frames))
    lat_list: List[torch.Tensor] = []
    pose_list: List[np.ndarray] = []
    for i in range(n):
        ts = int(frames[i].timestep)
        pose = timestep_to_pose.get(ts)
        if pose is None:
            logger.warning("retrieve: timestep=%d 无对应位姿（跳过该 anchor 帧）", ts)
            continue
        lat_list.append(latents[i])          # [16, lat_h, lat_w]
        pose_list.append(pose)               # [4, 4]
    if not lat_list:
        return None

    anchor_latent = torch.stack(lat_list, dim=1).to(device).float()  # [16, n, lat_h, lat_w]
    anchor_poses = np.stack(pose_list, axis=0).astype(np.float32)    # [n, 4, 4]
    return anchor_latent, anchor_poses


# ---------------------------------------------------------------------------
# 数据入口（action_path 模式 / episode 模式 二选一）
# ---------------------------------------------------------------------------

def _build_job_episode(args, ep_id: str, ep_groups, height: int, width: int):
    """episode 模式：build_episode_data + 解码 → 返回 job dict（含 ep / 轨迹 / 首帧 / GT 帧）。"""
    ep = build_episode_data(ep_id, ep_groups[ep_id], clip_overlap_frames=0)
    if ep is None:
        logger.warning("episode %s: build_episode_data 返回 None，跳过", ep_id)
        return None
    poses_np = ep.poses.astype(np.float32)
    actions_np = ep.actions.astype(np.float32)
    intrinsics_np = ep.intrinsics.astype(np.float32)
    T = poses_np.shape[0]
    frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W] in [-1,1]
    first_hwc = (frames[0].transpose(1, 2, 0) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    first_img = Image.fromarray(first_hwc)
    return {
        "kind": "episode", "episode_id": ep_id, "ep": ep,
        "poses": poses_np, "actions": actions_np, "intrinsics": intrinsics_np,
        "first_img": first_img, "gt_frames": frames, "T": T,
    }


def _build_job_action(args):
    """action_path 模式：np.load 长轨迹 + 首帧 → 返回 job dict（ep=None，无 GT 帧）。"""
    if not (args.action_path and os.path.isdir(args.action_path)):
        raise SystemExit(f"action_path 不存在或非目录：{args.action_path}")
    if not args.image:
        raise SystemExit("action_path 模式要求 --image（首帧图像路径）")
    poses_np = np.load(os.path.join(args.action_path, "poses.npy")).astype(np.float32)
    actions_np = np.load(os.path.join(args.action_path, "action.npy")).astype(np.float32)
    intrinsics_np = np.load(os.path.join(args.action_path, "intrinsics.npy")).astype(np.float32)
    first_img = Image.open(args.image).convert("RGB")
    logger.info("action_path 模式 loaded: poses=%s actions=%s intrinsics=%s",
                poses_np.shape, actions_np.shape, intrinsics_np.shape)
    return {
        "kind": "action", "episode_id": os.path.basename(str(args.action_path).rstrip("/")),
        "ep": None, "poses": poses_np, "actions": actions_np, "intrinsics": intrinsics_np,
        "first_img": first_img, "gt_frames": None, "T": poses_np.shape[0],
    }


def _resolve_episode_ids(args) -> List[str]:
    """episode 模式：解析要跑的 episode id 列表（filter/cap，切分前应用）。"""
    if not args.dataset_dir:
        raise ValueError("episode 模式（--episode_id 设）要求 --dataset_dir 非空")
    ep_filter = None
    if args.episode_ids:
        ep_filter = [s.strip() for s in args.episode_ids.split(",") if s.strip()]
    elif args.episode_id not in ("first", "top", "all"):
        ep_filter = [args.episode_id]
    ep_groups = load_episode_clips(args.dataset_dir, args.metadata, episode_ids_filter=ep_filter)
    ep_ids = list(ep_groups.keys())
    # 'first'/'top' 只取第一个；'all' 或显式列表取全部
    if args.episode_id in ("first", "top") and not args.episode_ids:
        ep_ids = ep_ids[:1]
    if args.max_episodes > 0 and len(ep_ids) > args.max_episodes:
        ep_ids = ep_ids[:args.max_episodes]
    return ep_ids, ep_groups


# ---------------------------------------------------------------------------
# 多 clip 自回归 rollout（bank populate/retrieve + anchor-concat 注入）
# ---------------------------------------------------------------------------

def _rollout_long_video(wan_i2v, job, args, device, num_clips):
    """对单条轨迹跑 num_clips 自回归 + 每 clip bank retrieve→anchor 注入 / populate。

    镜像 infer_v5:589-708（逐 clip 切片、seed=base+clip_idx、末帧→下一 current_img、bank
    increment_age/update），但把生成从 wan_i2v.generate + in-context KV 换成 stage1
    _generate_with_anchor 的**帧维 anchor-concat**，anchor 来源从 encoder 换成 **bank 检索**。

    Returns:
        clip_videos: List[np.ndarray]，各 [3, frame_num, H, W] in [-1,1]。
    """
    from memory_module.memory_bank import ThreeTierMemoryBank

    poses_np = job["poses"]
    actions_np = job["actions"]
    intrinsics_np = job["intrinsics"]
    ep = job["ep"]
    T = job["T"]
    frame_num = args.frame_num
    use_bank = (args.retrieval == "bank")

    bank = ThreeTierMemoryBank(
        short_cap=args.short_cap, medium_cap=args.medium_cap, long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life, dup_threshold=args.dup_threshold,
    )
    dim = wan_i2v.low_noise_model.dim  # pose_emb 占位维度（对齐 infer_v5）
    timestep_to_pose: Dict[int, np.ndarray] = {}  # global_t → 完整 c2w[4,4]（anchor 位姿挂载）
    global_t = 0

    current_img = job["first_img"]
    clip_videos: List[np.ndarray] = []

    for clip_idx in range(num_clips):
        logger.info("[%s] Generating clip %d/%d ...", job["episode_id"], clip_idx + 1, num_clips)
        if clip_idx > 0:
            bank.increment_age()

        # a. 切本 clip 轨迹（infer_v5 口径：不足 frame_num 取末尾 frame_num 帧）
        clip_start = clip_idx * frame_num
        clip_end = clip_start + frame_num
        if clip_end <= len(poses_np):
            clip_poses = poses_np[clip_start:clip_end]
            clip_actions = actions_np[clip_start:clip_end]
            clip_intr = intrinsics_np[clip_start:clip_end]
        else:
            clip_poses = poses_np[-frame_num:] if len(poses_np) >= frame_num else poses_np
            clip_actions = actions_np[-frame_num:] if len(actions_np) >= frame_num else actions_np
            clip_intr = intrinsics_np[-frame_num:] if len(intrinsics_np) >= frame_num else intrinsics_np
            logger.warning("[%s] clip %d: 轨迹不足（%d<%d），用末尾 %d 帧",
                           job["episode_id"], clip_idx + 1, len(poses_np), clip_end, len(clip_poses))
        clip_poses = clip_poses.astype(np.float32)
        clip_actions = clip_actions.astype(np.float32)
        clip_intr = clip_intr.astype(np.float32)
        poses_tensor = torch.from_numpy(clip_poses).float()

        # b. bank 检索 anchor（仅 clip_idx>0 且 bank 模式，bank 非空）
        anchor_latent = None
        anchor_poses = None
        if clip_idx > 0 and use_bank:
            _est_lat_f = max(1, len(clip_poses) // 4)
            query_location = _per_latent_abs_locations(poses_tensor, _est_lat_f)[0]  # [3]
            anchor = _retrieve_anchor(
                bank, timestep_to_pose, query_location, global_t,
                num_anchor_frames=args.num_anchor_frames,
                top_k=args.revisit_top_k, min_gap_frames=args.revisit_min_gap_frames,
                device=device)
            if anchor is not None:
                anchor_latent, anchor_poses = anchor
            _k = 0 if anchor_latent is None else anchor_latent.shape[1]
            logger.info("[%s] clip %d: bank retrieved %d anchor frame(s) (size=%d).",
                        job["episode_id"], clip_idx + 1, _k, bank.size())
        elif not use_bank:
            logger.info("[%s] clip %d: --retrieval none, pure base i2v (no anchor).",
                        job["episode_id"], clip_idx + 1)

        # c. per-clip 数据 prompt（episode 模式复用 _clip_prompt；action 模式回退 --prompt）+ 生成
        if ep is not None:
            prompt = _clip_prompt(ep, clip_start, args)
        else:
            if args.prompt_source == "data" and clip_idx == 0:
                logger.warning("action_path 模式无 per-clip prompt.txt → 用 --prompt（fixed）")
            prompt = args.prompt
        clip_args = _clip_args(args, prompt, args.seed + clip_idx)

        video = _generate_with_anchor(
            wan_i2v, current_img, anchor_latent, anchor_poses,
            clip_poses, clip_actions, clip_intr, clip_args, device)
        if video is None:
            logger.error("[%s] clip %d: 生成 None，终止该轨迹", job["episode_id"], clip_idx + 1)
            break
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().float().numpy()
        video = np.asarray(video, dtype=np.float32)  # [3, frame_num, H, W] in [-1,1]
        clip_videos.append(video)

        # d. bank 更新（surprise-independent，复用 infer_v5._update_bank_from_clip）+ 记完整位姿
        if use_bank:
            new_global_t = _update_bank_from_clip(
                bank=bank, video=video, wan_i2v=wan_i2v, device=device,
                poses_clip=poses_tensor, global_t_start=global_t, dim=dim, clip_idx=clip_idx)
            lat_f = new_global_t - global_t  # = VAE encode 后 latent.shape[1]（与 bank update 一致）
            clip_latent_poses = _per_latent_abs_poses(clip_poses, lat_f)  # [lat_f,4,4]
            for t in range(lat_f):
                timestep_to_pose[global_t + t] = clip_latent_poses[t]
            global_t = new_global_t

        # e. 末帧 → 下一 clip current_img（autoregressive，镜像 infer_v5）
        last_hwc = (video[:, -1].transpose(1, 2, 0) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        current_img = Image.fromarray(last_hwc)
        logger.info("[%s] clip %d done. bank.size=%d global_t=%d.",
                    job["episode_id"], clip_idx + 1, bank.size(), global_t)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return clip_videos


# ---------------------------------------------------------------------------
# 可选评分（旁路；仅 episode 模式有 GT 帧；不判 GO）
# ---------------------------------------------------------------------------

_SCORE_FIELDS = [
    "episode_id", "query_frame", "first_visit_frame", "revisit_clip_idx",
    "dino_max", "dino_mean", "dino_last", "ssim_max", "ssim_mean", "ssim_last",
]


def _score_job(job, clip_videos, args, device, out_dir, min_time_gap_frames):
    """旁路评分：检测重访点 → 对落在生成范围内的重访 clip vs GT 首访帧算 DINO/SSIM，写 scores.csv。"""
    ep = job["ep"]
    gt_frames = job["gt_frames"]
    if ep is None or gt_frames is None:
        logger.warning("[%s] --score 仅 episode 模式可用（无 GT 帧），跳过评分", job["episode_id"])
        return
    pts = _find_revisit_points(ep, args, min_time_gap_frames)
    if not pts:
        logger.info("[%s] --score：无重访点，跳过", job["episode_id"])
        return
    n_clips = len(clip_videos)
    csv_path = os.path.join(out_dir, "scores.csv")
    rows: List[Dict] = []
    for pt in pts:
        clip_idx = pt.query_frame // args.frame_num
        if clip_idx < 0 or clip_idx >= n_clips:
            continue  # 重访点不在生成范围内
        if not (0 <= pt.first_visit_frame < gt_frames.shape[0]):
            continue
        gt_first = gt_frames[pt.first_visit_frame]  # [3,H,W]
        metrics = _revisit_consistency(clip_videos[clip_idx], gt_first, device=device)
        rows.append({
            "episode_id": job["episode_id"], "query_frame": pt.query_frame,
            "first_visit_frame": pt.first_visit_frame, "revisit_clip_idx": clip_idx,
            "dino_max": metrics.get("revisit_consistency_dino_max"),
            "dino_mean": metrics.get("revisit_consistency_dino_mean"),
            "dino_last": metrics.get("revisit_consistency_dino_last"),
            "ssim_max": metrics.get("revisit_consistency_max"),
            "ssim_mean": metrics.get("revisit_consistency_mean"),
            "ssim_last": metrics.get("revisit_consistency_last"),
        })
        try:
            _save_frame_png(gt_first, os.path.join(out_dir, f"gt_first_visit_q{pt.query_frame}.png"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] 存 gt_first_visit png 失败（非致命）: %s", job["episode_id"], exc)
        logger.info("[%s] score q=%d clip=%d dino_mean=%s",
                    job["episode_id"], pt.query_frame, clip_idx, rows[-1]["dino_mean"])
    if not rows:
        logger.info("[%s] --score：无重访点落在生成范围内，未写 scores.csv", job["episode_id"])
        return
    try:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SCORE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info("[%s] scores.csv 已写 → %s（%d 行，旁路指标，不判 GO）",
                    job["episode_id"], csv_path, len(rows))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 写 scores.csv 失败: %s", job["episode_id"], exc)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 产出目录（paths.py 新布局，version=v6）----
    run_name = args.run_name or default_run_name("v6_latentconcat_infer")
    run_dir = infer_run_dir("v6", run_name, args.tag)
    log_path = os.path.join(str(run_dir), "infer.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    snapshot_config(run_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})
    logger.info("v6 latentconcat_infer run_dir=%s | retrieval=%s | prompt_source=%s | num_anchor=%d",
                run_dir, args.retrieval, args.prompt_source, args.num_anchor_frames)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    # oracle_injection 全局（_revisit_consistency / 帧切片读它，对齐 ideal_diag 习惯）
    _oracle_inj._SIZE_HW = tuple(args.size.split("*"))
    _oracle_inj._ORACLE_CLIP_FRAMES = args.frame_num
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))
    logger.info("Args: %s", vars(args))

    # ---- 解析 job 列表（episode 模式多 ep / action_path 单轨迹）+ 分片 ----
    episode_mode = args.episode_id is not None or args.episode_ids is not None
    if episode_mode:
        ep_ids, ep_groups = _resolve_episode_ids(args)
        if not ep_ids:
            logger.error("episode 模式：无 episode 可处理，退出。")
            return
        if args.shard_count > 1:
            sel_ids = [(i, e) for i, e in enumerate(ep_ids)
                       if i % args.shard_count == args.shard_index]
            logger.info("分片 %d/%d：episode 全集 %d，本分片处理 %d",
                        args.shard_index, args.shard_count, len(ep_ids), len(sel_ids))
        else:
            sel_ids = list(enumerate(ep_ids))
            logger.info("单进程：处理全部 %d 个 episode", len(sel_ids))
        if not sel_ids:
            logger.error("shard %d/%d 分到 0 个 episode，退出。", args.shard_index, args.shard_count)
            return
    else:
        # action_path 单轨迹：shard_count>1 时仅 shard 0 处理（其余分片空跑退出）
        if args.shard_count > 1 and args.shard_index != 0:
            logger.info("action_path 模式单轨迹：shard %d 非 0，跳过。", args.shard_index)
            return

    # ---- 加载 pipeline（复用 ideal_diag._load_pipeline；冻结 DiT，memory 不参与）----
    wan_i2v = _load_pipeline(args, device)

    # ---- 逐 job rollout + 保存长视频 ----
    if episode_mode:
        for _gi, ep_id in sel_ids:
            try:
                job = _build_job_episode(args, ep_id, ep_groups, height, width)
            except Exception as exc:  # noqa: BLE001
                logger.exception("episode %s 构建失败: %s；跳过", ep_id, exc)
                continue
            if job is None:
                continue
            num_clips = max(1, min(args.num_clips, max(1, job["T"] // args.frame_num)))
            if num_clips < args.num_clips:
                logger.warning("[%s] T=%d frame_num=%d → num_clips 裁到 %d",
                               ep_id, job["T"], args.frame_num, num_clips)
            _run_and_save(wan_i2v, job, args, device, run_dir, num_clips, min_time_gap_frames)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        job = _build_job_action(args)
        num_clips = max(1, args.num_clips)
        _run_and_save(wan_i2v, job, args, device, run_dir, num_clips, min_time_gap_frames)

    logger.info("Done. 输出目录: %s", run_dir)


def _run_and_save(wan_i2v, job, args, device, run_dir, num_clips, min_time_gap_frames):
    """单 job：rollout → 拼接长视频保存 → 可选评分。"""
    out_dir = os.path.join(str(run_dir), job["episode_id"])
    os.makedirs(out_dir, exist_ok=True)
    save_file = os.path.join(out_dir, "long_video.mp4")
    if os.path.exists(save_file):
        logger.info("[%s] long_video.mp4 已存在 → 跳过 rollout", job["episode_id"])
        return
    # 另存首帧供对照
    try:
        job["first_img"].save(os.path.join(out_dir, "first_frame.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 存 first_frame.png 失败（非致命）: %s", job["episode_id"], exc)

    clip_videos = _rollout_long_video(wan_i2v, job, args, device, num_clips)
    if not clip_videos:
        logger.error("[%s] 无 clip 生成，未保存长视频", job["episode_id"])
        return
    full_video = np.concatenate(clip_videos, axis=1)  # [3, total_F, H, W]
    _save_video(full_video, save_file, fps=args.fps)
    logger.info("[%s] 长视频已存 → %s（%d 帧 @ %dfps，%d clips）",
                job["episode_id"], save_file, full_video.shape[1], args.fps, len(clip_videos))

    if args.score:
        _score_job(job, clip_videos, args, device, out_dir, min_time_gap_frames)


if __name__ == "__main__":
    main()
