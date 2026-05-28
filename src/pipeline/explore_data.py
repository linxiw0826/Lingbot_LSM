"""explore_data.py — v4 数据集重访结构探查脚本

目的
-----
在 Exp1/Exp2（Memory Bank 快速验证实验）设计前，先量化现有 v4 数据集的
**真实重访结构**，区分以下三类帧对：
  1. **位置重访**（Long Tier 主打）：相同地点 + 相同朝向，时间差较长，
     对应"agent 走开后绕回来"的场景。
  2. **镜头摆动**（Short Tier 范围）：相同地点 + 相同朝向，时间差很短，
     对应"几秒内左右摆头又看回来"。
  3. **位置近但视角不同**：地点接近但朝向相差大，对应"同一地点不同视野"。

依据 OP-1 / experiment_design.md「快速验证实验设计」，
本脚本输出的统计将用于：
  - 判断 v4_dynamic_all46 中"长程位置重访"是否足够支撑 idea Long Tier 测试
  - 在 eval 集中筛选真正含 loop 的 episode，构建 Exp1 检索探针的 GT

设计约束
---------
- **CPU only**：只用 numpy / opencv-python / matplotlib / pandas / tqdm，
  不导入 PyTorch，不导入 train_v4_stage1_dual / memory 模块。
- **不动训练/推理代码**：本脚本独立运行，输出全部落到 --output_dir，
  不修改任何 src/ 代码与数据集本身。
- **复用 episode 分组逻辑思路**：episode 分组 + clip_idx 整数排序与
  CSGOMultiClipDataset._build_episode_groups（train_v4_stage1_dual.py:247）
  保持一致；但本脚本自带简洁实现，不 import 训练 dataset 类。
- **可重入**：同一 output_dir 重跑时已生成的图直接跳过；--force 强制重生成。
- **CSV 列依赖**：本脚本假设 metadata CSV 含 `episode_id`/`clip_idx`/`clip_path`
  三列；`clip_path` 用于定位 `poses.npy` 与 `video.mp4`。若数据集 CSV
  schema 变化（如 clip_path 被改名），需同步更新 load_clip_metas()。

输出文件结构
------------
  output_dir/
    ├── summary.md                          # 人类可读全局报告
    ├── summary.json                        # 机器可读版本
    ├── trajectories/<ep_id>_bev.png        # BEV 2D 轨迹，时间渐变着色
    └── revisit_pairs/
          ├── <ep_id>_pairs.csv             # 该 episode 全部三类重访帧对
          └── <ep_id>_<type>_<i>.png        # 抽样帧对并排截图

用法示例
--------
仅分析 eval 5 视频（指定 episode_id）：
    python explore_data.py \\
        --dataset_dir /home/nvme02/Memory-dataset/v4_dynamic_all46 \\
        --metadata metadata_full_val.csv \\
        --output_dir /tmp/explore_v4_eval5 \\
        --episode_ids player12_ep12,player13_ep13

整集采样：
    python explore_data.py \\
        --dataset_dir /home/nvme02/Memory-dataset/v4_dynamic_all46 \\
        --metadata metadata_full_train.csv \\
        --output_dir /tmp/explore_v4_pilot \\
        --max_episodes 20
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# matplotlib 必须在导入前设 backend，避免 server 上无 DISPLAY 时崩溃
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import cv2
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ClipMeta:
    """单个 clip 的元信息（来自 metadata_*.csv 单行 + 派生路径）。"""

    episode_id: str
    clip_idx: int
    clip_path: str          # 绝对路径，clip 目录
    csv_row: dict           # 原 CSV 行，调试用


@dataclass
class EpisodeData:
    """一个 episode 拼接后的完整轨迹数据。"""

    episode_id: str
    clips: List[ClipMeta]           # 按 clip_idx 排序
    poses: np.ndarray               # [T, 4, 4]
    frame_to_clip: List[Tuple[int, int]]  # 长度 T，(clip_array_idx, local_frame_idx)
    # 注意：clip_array_idx 是 self.clips 列表中的下标（非 csv 的 clip_idx）

    @property
    def total_frames(self) -> int:
        return self.poses.shape[0]


@dataclass
class RevisitPair:
    """一对帧之间的重访信息。"""

    frame_a: int            # 全局帧号（拼接后）
    frame_b: int
    pair_type: str          # "location_revisit" / "camera_swing" / "same_loc_diff_view"
    dist_m: float           # XZ 平面距离（米）
    yaw_diff_deg: float     # yaw 差（度，0-180）
    time_diff_sec: float    # 时间差（秒）


@dataclass
class EpisodeStats:
    """单 episode 的统计结果。"""

    episode_id: str
    total_frames: int
    duration_sec: float
    n_location_revisit: int = 0
    n_camera_swing: int = 0
    n_same_loc_diff_view: int = 0
    n_grey_zone: int = 0   # 灰色地带：同地点同朝向，但 time_short <= dt <= time_long
    pairs: List[RevisitPair] = field(default_factory=list)
    bev_path: Optional[str] = None     # 相对 output_dir 的路径
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Episode 分组（与 CSGOMultiClipDataset._build_episode_groups 思路一致）
# ---------------------------------------------------------------------------

def load_clip_metas(
    dataset_dir: str,
    metadata_rel_path: str,
    episode_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, List[ClipMeta]]:
    """从 metadata CSV 加载 clip 元信息，按 episode_id 分组 + clip_idx 整数排序。

    Args:
        dataset_dir: 数据集根目录（含 metadata_*.csv 和 clips/）。
        metadata_rel_path: 相对 dataset_dir 的 CSV 路径，如 "metadata_full_train.csv"。
        episode_ids_filter: 若给定，只保留这些 episode_id。

    Returns:
        OrderedDict-like: episode_id -> List[ClipMeta]（已按 clip_idx 升序）。

    复用思路（不复用类）：
        train_v4_stage1_dual.py:247 `_build_episode_groups`
          - 强制要求 episode_id / clip_idx 列存在
          - 按 episode_id 分组，clip_idx 整数排序
        本函数保留同样的字段名与排序规则，但 standalone（不依赖训练 Dataset）。
    """
    csv_path = os.path.join(dataset_dir, metadata_rel_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

    episode_clips: Dict[str, List[ClipMeta]] = defaultdict(list)
    n_rows = 0
    n_kept = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if "episode_id" not in fieldnames:
            raise ValueError(
                f"CSV {csv_path} missing required column 'episode_id'. "
                "explore_data.py 与 CSGOMultiClipDataset 共用 episode_id/clip_idx 列约定。"
            )
        if "clip_idx" not in fieldnames:
            raise ValueError(
                f"CSV {csv_path} missing required column 'clip_idx'."
            )
        if "clip_path" not in fieldnames:
            raise ValueError(
                f"CSV {csv_path} missing required column 'clip_path'."
            )

        filter_set = set(episode_ids_filter) if episode_ids_filter else None
        for row in reader:
            n_rows += 1
            ep_id = row["episode_id"]
            if filter_set is not None and ep_id not in filter_set:
                continue
            try:
                clip_idx_int = int(row["clip_idx"])
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping row with invalid clip_idx=%r in episode %s",
                    row.get("clip_idx"), ep_id,
                )
                continue
            clip_path = row["clip_path"]
            # clip_path 可能是相对路径；统一拼成绝对路径
            if not os.path.isabs(clip_path):
                clip_path_abs = os.path.normpath(os.path.join(dataset_dir, clip_path))
            else:
                clip_path_abs = clip_path
            episode_clips[ep_id].append(
                ClipMeta(
                    episode_id=ep_id,
                    clip_idx=clip_idx_int,
                    clip_path=clip_path_abs,
                    csv_row=row,
                )
            )
            n_kept += 1

    # 排序（与 train_v4_stage1_dual.py:296 一致：int clip_idx 升序）
    for ep_id in episode_clips:
        episode_clips[ep_id].sort(key=lambda c: c.clip_idx)

    logger.info(
        "Loaded %d clips (kept %d after filter) across %d episodes from %s",
        n_rows, n_kept, len(episode_clips), csv_path,
    )
    if filter_set is not None:
        missing = filter_set - set(episode_clips.keys())
        if missing:
            logger.warning(
                "episode_ids_filter requested %s but missing from CSV: %s",
                sorted(filter_set), sorted(missing),
            )
    return episode_clips


def build_episode_data(
    episode_id: str,
    clips: List[ClipMeta],
    clip_overlap_frames: int = 0,
) -> Tuple[Optional[EpisodeData], List[str]]:
    """拼接一个 episode 的所有 poses，构造 EpisodeData。

    Args:
        episode_id: episode 标识。
        clips: 该 episode 已按 clip_idx 升序排列的 ClipMeta 列表。
        clip_overlap_frames: 相邻 clip 之间的 overlap 帧数。
            - 0（默认）：假设 clip 间首尾连续，无 overlap。
            - >0：对 clip_array_idx >= 1 的 clip 跳过前 N 帧 poses，避免 overlap
              帧被错误归类为 camera_swing。被跳过帧对应的 local_idx 不进入
              frame_to_clip，因此 video.mp4 解码定位时仍指向原始帧位置。

    Returns:
        (EpisodeData 或 None, overlap_hints):
          - 当 poses 全部缺失/为 0 时返回 (None, hints)
          - hints: 自动检测产出的 overlap 提示字符串列表（若相邻 clip 末/首帧
            xz 平移几乎重合）

    缺 poses.npy 的 clip 整段跳过（按任务要求，若缺关键文件则跳过该 episode）。
    """
    pose_list: List[np.ndarray] = []
    frame_to_clip: List[Tuple[int, int]] = []
    skipped_clips: List[str] = []
    overlap_hints: List[str] = []
    # 用于自动检测：保存每段 raw（trim 之前）的首/末帧平移
    prev_last_xyz: Optional[np.ndarray] = None

    valid_clip_count = 0  # 已加入 pose_list 的 clip 数；用于判断是否为首段
    for clip_array_idx, clip in enumerate(clips):
        poses_path = os.path.join(clip.clip_path, "poses.npy")
        if not os.path.isfile(poses_path):
            skipped_clips.append(clip.clip_path)
            continue
        try:
            poses = np.load(poses_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", poses_path, exc)
            skipped_clips.append(clip.clip_path)
            continue
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            logger.warning(
                "Unexpected poses shape %s in %s, skipping clip",
                poses.shape, poses_path,
            )
            skipped_clips.append(clip.clip_path)
            continue
        poses = poses.astype(np.float32)
        n_frames_raw = poses.shape[0]

        # ---- 自动检测：相邻 clip 首帧 vs 上一 clip 末帧是否几乎重合 ----
        if prev_last_xyz is not None and n_frames_raw > 0:
            first_xyz = poses[0, :3, 3]
            dist = float(np.linalg.norm(first_xyz - prev_last_xyz))
            if dist < 0.01:
                overlap_hints.append(
                    f"Episode {episode_id}: clip {clip.clip_idx} 的首帧 pose 与上"
                    f"一 clip 末帧几乎重合（dist={dist:.4f}m），可能存在 overlap，"
                    f"建议设 --clip_overlap_frames=8"
                )

        # 末帧记录（先于 trim，用上一 clip 的原始末帧）
        if n_frames_raw > 0:
            prev_last_xyz = poses[-1, :3, 3].copy()

        # ---- Overlap 处理：非首段 clip 跳过前 N 帧 ----
        if valid_clip_count >= 1 and clip_overlap_frames > 0:
            trim = min(clip_overlap_frames, n_frames_raw)
            poses_kept = poses[trim:]
            local_start = trim
        else:
            poses_kept = poses
            local_start = 0

        if poses_kept.shape[0] == 0:
            # overlap > 实际帧数，整段被跳过；记 warning 但不算缺失
            logger.warning(
                "Episode %s: clip %s 被 overlap (%d) 完全跳过 (raw_frames=%d)",
                episode_id, clip.clip_idx, clip_overlap_frames, n_frames_raw,
            )
            valid_clip_count += 1
            continue

        pose_list.append(poses_kept)
        for local_idx in range(local_start, local_start + poses_kept.shape[0]):
            frame_to_clip.append((clip_array_idx, local_idx))
        valid_clip_count += 1

    if not pose_list:
        logger.warning(
            "Episode %s has no usable poses.npy (skipped %d clips), skipping.",
            episode_id, len(skipped_clips),
        )
        return None, overlap_hints

    if skipped_clips:
        logger.warning(
            "Episode %s: skipped %d/%d clips with missing/invalid poses.npy",
            episode_id, len(skipped_clips), len(clips),
        )

    for h in overlap_hints:
        logger.warning(h)

    poses_cat = np.concatenate(pose_list, axis=0)
    return EpisodeData(
        episode_id=episode_id,
        clips=clips,
        poses=poses_cat,
        frame_to_clip=frame_to_clip,
    ), overlap_hints


# ---------------------------------------------------------------------------
# 几何工具：BEV 投影 + yaw 计算
# ---------------------------------------------------------------------------

def extract_xz_and_yaw(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从 c2w 矩阵 [T,4,4] 提取 BEV 位置 (x,z) 和 yaw 角（度）。

    BEV 投影：取平移分量的 0/2 维（x, z），y 是上方向。
    Yaw 计算：标准做法 yaw = atan2(R[0,2], R[2,2])（弧度），再转度数。
      - R 为 c2w 旋转部分 poses[:, :3, :3]
      - 该 yaw 衡量相机绕 y 轴的方向，与 BEV 平面一致
    返回 yaw 范围 (-180, 180]。
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape [T,4,4], got {poses.shape}")
    xz = poses[:, [0, 2], 3].astype(np.float32)      # [T, 2]
    R = poses[:, :3, :3].astype(np.float32)
    yaw_rad = np.arctan2(R[:, 0, 2], R[:, 2, 2])     # [T]
    yaw_deg = np.degrees(yaw_rad)
    return xz, yaw_deg


def angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组角度（度）之间的最小角度差，结果在 [0, 180]。

    支持广播：a, b 可为同 shape 数组或可广播的两个数组。
    """
    diff = np.abs(a - b) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


# ---------------------------------------------------------------------------
# 帧对分类（核心统计逻辑）
# ---------------------------------------------------------------------------

def _sample_frame_indices(total_frames: int, max_pairs_dim: int = 500) -> np.ndarray:
    """大 episode 稀疏采样：T > max_pairs_dim 时按 step 抽样，避免 O(T^2) 爆炸。

    Args:
        total_frames: 帧总数 T。
        max_pairs_dim: 采样后的目标帧数（默认 500，→ 最多 ~125k 对）。

    Returns:
        排序好的全局帧号 ndarray。
    """
    if total_frames <= max_pairs_dim:
        return np.arange(total_frames, dtype=np.int64)
    step = max(1, total_frames // max_pairs_dim)
    return np.arange(0, total_frames, step, dtype=np.int64)


def classify_revisit_pairs(
    episode: EpisodeData,
    distance_eps: float,
    yaw_eps: float,
    time_short: float,
    time_long: float,
    fps: int,
    diff_view_yaw: float = 60.0,
    max_pairs_dim: int = 500,
) -> Tuple[List[RevisitPair], Dict[str, int]]:
    """对单 episode 内所有帧对做三类重访分类（+ 灰色地带计数）。

    类别定义（与任务规范一致）：
      - location_revisit  : dist < distance_eps && |yaw| < yaw_eps && time_diff > time_long
      - camera_swing      : dist < distance_eps*0.5 && |yaw| < yaw_eps && time_diff < time_short
      - same_loc_diff_view: dist < distance_eps && |yaw| > diff_view_yaw && 任意时间差
      - grey_zone（仅计数）: dist < distance_eps && |yaw| < yaw_eps
                            && time_short <= time_diff <= time_long
        （不进入 pairs 列表也不画截图；仅用于诊断阈值合理性）

    返回 (pairs, counts)，counts 含 location_revisit/camera_swing/
    same_loc_diff_view/grey_zone 四项。
    """
    xz, yaw_deg = extract_xz_and_yaw(episode.poses)
    T = episode.poses.shape[0]

    # 稀疏采样防止 O(T^2) 爆炸
    sample_idx = _sample_frame_indices(T, max_pairs_dim=max_pairs_dim)
    n = sample_idx.shape[0]
    xz_s = xz[sample_idx]                # [n, 2]
    yaw_s = yaw_deg[sample_idx]          # [n]

    # 两两矩阵（n x n）
    diff_xz = xz_s[:, None, :] - xz_s[None, :, :]
    dist_mat = np.sqrt(np.sum(diff_xz * diff_xz, axis=-1))   # [n, n]
    yaw_mat = angular_diff_deg(yaw_s[:, None], yaw_s[None, :])  # [n, n]
    time_mat = np.abs(sample_idx[:, None] - sample_idx[None, :]).astype(np.float32) / float(fps)

    # 仅取上三角（i < j），避免重复
    iu, ju = np.triu_indices(n, k=1)

    dist_u = dist_mat[iu, ju]
    yaw_u = yaw_mat[iu, ju]
    time_u = time_mat[iu, ju]

    # 类别 mask
    loc_revisit_mask = (
        (dist_u < distance_eps) &
        (yaw_u < yaw_eps) &
        (time_u > time_long)
    )
    cam_swing_mask = (
        (dist_u < distance_eps * 0.5) &
        (yaw_u < yaw_eps) &
        (time_u < time_short)
    )
    same_loc_diff_view_mask = (
        (dist_u < distance_eps) &
        (yaw_u > diff_view_yaw)
    )
    grey_zone_mask = (
        (dist_u < distance_eps) &
        (yaw_u < yaw_eps) &
        (time_u >= time_short) &
        (time_u <= time_long)
    )

    pairs: List[RevisitPair] = []
    counts = {
        "location_revisit": int(loc_revisit_mask.sum()),
        "camera_swing": int(cam_swing_mask.sum()),
        "same_loc_diff_view": int(same_loc_diff_view_mask.sum()),
        "grey_zone": int(grey_zone_mask.sum()),
    }

    def _append(mask: np.ndarray, label: str) -> None:
        sel = np.nonzero(mask)[0]
        for k in sel:
            ai = int(sample_idx[iu[k]])
            bi = int(sample_idx[ju[k]])
            pairs.append(
                RevisitPair(
                    frame_a=ai,
                    frame_b=bi,
                    pair_type=label,
                    dist_m=float(dist_u[k]),
                    yaw_diff_deg=float(yaw_u[k]),
                    time_diff_sec=float(time_u[k]),
                )
            )

    _append(loc_revisit_mask, "location_revisit")
    _append(cam_swing_mask, "camera_swing")
    _append(same_loc_diff_view_mask, "same_loc_diff_view")

    return pairs, counts


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def render_bev(
    episode: EpisodeData,
    out_path: str,
    force: bool = False,
) -> None:
    """渲染 BEV 鸟瞰图：x-z 平面 + 时间渐变色 + colorbar。"""
    if os.path.isfile(out_path) and not force:
        return
    xz, _ = extract_xz_and_yaw(episode.poses)
    T = xz.shape[0]
    times = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    norm = Normalize(vmin=0, vmax=max(T - 1, 1))
    cmap = cm.get_cmap("viridis")
    sc = ax.scatter(xz[:, 0], xz[:, 1], c=times, cmap=cmap, norm=norm, s=4, alpha=0.7)
    # 起止点强调
    ax.scatter([xz[0, 0]], [xz[0, 1]], c="lime", s=80, marker="o", edgecolor="black",
               label="start", zorder=5)
    ax.scatter([xz[-1, 0]], [xz[-1, 1]], c="red", s=80, marker="X", edgecolor="black",
               label="end", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"BEV trajectory — {episode.episode_id}\nT={T} frames, "
                 f"{len(episode.clips)} clips")
    ax.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("frame index (time →)")
    ax.grid(True, linestyle=":", alpha=0.4)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _decode_video_frame(video_path: str, local_frame_idx: int,
                        max_side: int = 512) -> Optional[np.ndarray]:
    """从 video.mp4 解码指定本地帧号（0-based），返回 RGB ndarray 或 None。

    按任务要求：截图最长边 ≤ max_side（默认 512）。
    """
    if not os.path.isfile(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames > 0:
            local_frame_idx = min(local_frame_idx, n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return rgb
    finally:
        cap.release()


def render_pair_screenshot(
    episode: EpisodeData,
    pair: RevisitPair,
    out_path: str,
    force: bool = False,
) -> Optional[str]:
    """渲染一对帧的并排截图，标注类别 + 距离 + yaw + 时间差。

    返回成功写入的路径，或 None（无法解码视频时记录 warning）。
    """
    if os.path.isfile(out_path) and not force:
        return out_path

    a_clip_idx, a_local = episode.frame_to_clip[pair.frame_a]
    b_clip_idx, b_local = episode.frame_to_clip[pair.frame_b]
    a_video = os.path.join(episode.clips[a_clip_idx].clip_path, "video.mp4")
    b_video = os.path.join(episode.clips[b_clip_idx].clip_path, "video.mp4")

    img_a = _decode_video_frame(a_video, a_local)
    img_b = _decode_video_frame(b_video, b_local)
    if img_a is None or img_b is None:
        logger.warning(
            "Skip pair screenshot for %s frames %d/%d: cannot decode video(s) "
            "(a_ok=%s, b_ok=%s)",
            episode.episode_id, pair.frame_a, pair.frame_b,
            img_a is not None, img_b is not None,
        )
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    axes[0].imshow(img_a)
    axes[0].set_title(f"frame {pair.frame_a} (clip {episode.clips[a_clip_idx].clip_idx})")
    axes[0].axis("off")
    axes[1].imshow(img_b)
    axes[1].set_title(f"frame {pair.frame_b} (clip {episode.clips[b_clip_idx].clip_idx})")
    axes[1].axis("off")
    fig.suptitle(
        f"{pair.pair_type} | dist={pair.dist_m:.2f}m | "
        f"|yaw|={pair.yaw_diff_deg:.1f}° | Δt={pair.time_diff_sec:.1f}s",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore v4 dataset revisit structure: classify location revisit "
            "vs camera swing vs same-loc-different-view pairs."
        ),
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="数据集根目录（含 metadata_*.csv 和 clips/）")
    parser.add_argument("--metadata", required=True,
                        help="相对 dataset_dir 的 metadata CSV 路径，例如 "
                             "metadata_full_train.csv")
    parser.add_argument("--output_dir", required=True,
                        help="所有产出目录（summary.md/json + trajectories/ + "
                             "revisit_pairs/）")
    parser.add_argument("--episode_ids", default=None,
                        help="仅分析指定 episode_id（逗号分隔），缺省则分析全部")
    parser.add_argument("--max_episodes", type=int, default=50,
                        help="未指定 episode_ids 时，从全部 episode 中随机采样的"
                             "上限（默认 50）；负数或 0 表示不限")
    parser.add_argument("--distance_eps", type=float, default=1.0,
                        help="同地点距离阈值，米（默认 1.0）")
    parser.add_argument("--yaw_eps", type=float, default=30.0,
                        help="同朝向角度阈值，度（默认 30.0）")
    parser.add_argument("--time_short", type=float, default=2.0,
                        help="镜头摆动时间窗，秒（默认 2.0）")
    parser.add_argument("--time_long", type=float, default=5.0,
                        help="位置重访时间下界，秒（默认 5.0）")
    parser.add_argument("--fps", type=int, default=16,
                        help="视频帧率，用于帧号→时间换算（v4 默认 16）")
    parser.add_argument("--sample_pairs_per_episode", type=int, default=5,
                        help="每条 episode 每个类别抽样画截图的对数（默认 5）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机抽样种子（默认 42）")
    parser.add_argument("--max_pairs_dim", type=int, default=500,
                        help="两两矩阵采样后帧数上限（默认 500），防止 O(T^2) 爆炸")
    parser.add_argument("--clip_overlap_frames", type=int, default=0,
                        help="相邻 clip 之间的 overlap 帧数（默认 0=假设无 overlap）；"
                             "v4 数据若按 experiment_design.md 的「5s clip + 0.5s overlap」"
                             "切分则应显式设为 8。脚本会在拼接前对 clip_array_idx >= 1 "
                             "的 clip 跳过前 N 帧，避免 overlap 帧被错误归类为 camera_swing。"
                             "脚本同时自动检测相邻 clip pose 几乎重合的情况并写 warning。")
    parser.add_argument("--diff_view_yaw", type=float, default=60.0,
                        help="same_loc_diff_view 的 yaw 阈值，度（默认 60.0）；"
                             "|yaw| > diff_view_yaw 视为视角显著不同")
    parser.add_argument("--force", action="store_true",
                        help="即使输出文件已存在也强制重新生成（BEV / 截图）")
    return parser.parse_args(argv)


def _select_episodes(
    all_episode_ids: Sequence[str],
    explicit_ids: Optional[Sequence[str]],
    max_episodes: int,
    seed: int,
) -> List[str]:
    """决定本次实际处理的 episode 列表。

    - 显式 episode_ids：取交集，保持显式给定的顺序。
    - 否则：若 max_episodes <= 0 或 ≥ 全部数量 → 全部；否则随机抽样（可复现）。
    """
    all_set = set(all_episode_ids)
    if explicit_ids:
        kept = [e for e in explicit_ids if e in all_set]
        # WARN C-2: 显式 episode_ids 与 max_episodes 同传时静默歧义
        # 默认值是 50；只有当用户显式传非默认正值时才提示
        if max_episodes is not None and max_episodes > 0 and max_episodes != 50:
            logger.warning(
                "--episode_ids 已显式指定 %d 个 episode，将忽略 --max_episodes=%d",
                len(kept), max_episodes,
            )
        return kept

    all_sorted = sorted(all_episode_ids)
    if max_episodes is None or max_episodes <= 0 or max_episodes >= len(all_sorted):
        return all_sorted

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_sorted), size=max_episodes, replace=False)
    return [all_sorted[int(i)] for i in sorted(idx)]


def process_episode(
    episode: EpisodeData,
    args: argparse.Namespace,
    output_dir: str,
    rng: np.random.Generator,
) -> EpisodeStats:
    """处理单个 episode：分类 + 写 pairs.csv + 抽样截图 + 渲 BEV。"""
    stats = EpisodeStats(
        episode_id=episode.episode_id,
        total_frames=episode.total_frames,
        duration_sec=episode.total_frames / float(args.fps),
    )

    # 1. 分类
    pairs, counts = classify_revisit_pairs(
        episode,
        distance_eps=args.distance_eps,
        yaw_eps=args.yaw_eps,
        time_short=args.time_short,
        time_long=args.time_long,
        fps=args.fps,
        diff_view_yaw=args.diff_view_yaw,
        max_pairs_dim=args.max_pairs_dim,
    )
    stats.n_location_revisit = counts["location_revisit"]
    stats.n_camera_swing = counts["camera_swing"]
    stats.n_same_loc_diff_view = counts["same_loc_diff_view"]
    stats.n_grey_zone = counts.get("grey_zone", 0)
    stats.pairs = pairs

    # 2. 写 pairs CSV
    pairs_csv = os.path.join(output_dir, "revisit_pairs",
                             f"{episode.episode_id}_pairs.csv")
    os.makedirs(os.path.dirname(pairs_csv), exist_ok=True)
    df_rows = [
        {
            "frame_a": p.frame_a,
            "frame_b": p.frame_b,
            "type": p.pair_type,
            "dist_m": p.dist_m,
            "yaw_diff_deg": p.yaw_diff_deg,
            "time_diff_sec": p.time_diff_sec,
        }
        for p in pairs
    ]
    pd.DataFrame(df_rows, columns=[
        "frame_a", "frame_b", "type", "dist_m", "yaw_diff_deg", "time_diff_sec",
    ]).to_csv(pairs_csv, index=False)

    # 3. BEV 图
    bev_path = os.path.join(output_dir, "trajectories",
                            f"{episode.episode_id}_bev.png")
    try:
        render_bev(episode, bev_path, force=args.force)
        stats.bev_path = os.path.relpath(bev_path, output_dir)
    except Exception as exc:  # noqa: BLE001
        msg = f"BEV render failed for {episode.episode_id}: {exc}"
        logger.warning(msg)
        stats.warnings.append(msg)

    # 4. 抽样画帧对截图
    n_sample = max(0, int(args.sample_pairs_per_episode))
    if n_sample > 0 and pairs:
        by_type: Dict[str, List[RevisitPair]] = defaultdict(list)
        for p in pairs:
            by_type[p.pair_type].append(p)
        for ptype, plist in by_type.items():
            if not plist:
                continue
            k = min(n_sample, len(plist))
            sel_idx = rng.choice(len(plist), size=k, replace=False)
            for i_out, j in enumerate(sorted(int(x) for x in sel_idx)):
                pair = plist[j]
                out_png = os.path.join(
                    output_dir, "revisit_pairs",
                    f"{episode.episode_id}_{ptype}_{i_out}.png",
                )
                try:
                    render_pair_screenshot(episode, pair, out_png, force=args.force)
                except Exception as exc:  # noqa: BLE001
                    msg = (f"Pair screenshot failed ({ptype}#{i_out}) for "
                           f"{episode.episode_id}: {exc}")
                    logger.warning(msg)
                    stats.warnings.append(msg)

    return stats


def write_summary(
    args: argparse.Namespace,
    output_dir: str,
    per_episode_stats: List[EpisodeStats],
    global_warnings: List[str],
) -> None:
    """写 summary.md + summary.json，全局表 + per-episode 表 + 结论提示。"""
    total_frames = sum(s.total_frames for s in per_episode_stats)
    total_loc = sum(s.n_location_revisit for s in per_episode_stats)
    total_swing = sum(s.n_camera_swing for s in per_episode_stats)
    total_diffview = sum(s.n_same_loc_diff_view for s in per_episode_stats)
    total_grey = sum(s.n_grey_zone for s in per_episode_stats)
    total_pairs_classified = total_loc + total_swing + total_diffview

    def _pct(n: int, denom: int) -> str:
        if denom <= 0:
            return "n/a"
        return f"{100.0 * n / denom:.2f}%"

    # 占比相对全部已分类对数（让相对量级直观可比）
    loc_pct = _pct(total_loc, total_pairs_classified)
    swing_pct = _pct(total_swing, total_pairs_classified)
    diffview_pct = _pct(total_diffview, total_pairs_classified)

    avg_frames = total_frames / max(1, len(per_episode_stats))
    avg_duration = avg_frames / float(args.fps)

    # 结论提示
    hints: List[str] = []
    if total_pairs_classified == 0:
        hints.append("⚠️ 三类重访对均为 0；可能采样太稀疏或阈值过严，建议调大 "
                     "--max_pairs_dim 或放宽 --distance_eps / --yaw_eps。")
    else:
        loc_ratio = total_loc / float(total_pairs_classified)
        swing_ratio = total_swing / float(total_pairs_classified)
        if loc_ratio < 0.05:
            hints.append(f"⚠️ 位置重访仅占 {loc_ratio*100:.2f}% (< 5%)；"
                         "现有数据集对 idea Long Tier（场景重访）的覆盖偏低，"
                         "建议挑选/补充含明显 loop 的 episode。")
        if swing_ratio > 0.5:
            hints.append(f"ℹ️ 镜头摆动占主导 ({swing_ratio*100:.1f}%)，"
                         "证实用户观察『重访很多是镜头摆动』；这类对应 Short Tier，"
                         "不能用来检验 Long Tier 重访能力。")
        if total_diffview > total_loc:
            hints.append(f"ℹ️ '位置近视角不同' ({total_diffview}) 多于 "
                         f"'位置重访' ({total_loc})，说明轨迹常在同地点转向，"
                         "Long Tier 检验需筛选『朝向也对齐』的子集。")
        if total_grey > total_loc * 2 and total_loc > 0:
            hints.append(
                f"⚠️ 灰色地带（同地点同朝向，{args.time_short}s ≤ Δt ≤ {args.time_long}s）"
                f"对数 {total_grey} > 2× 位置重访 ({total_loc})，"
                f"大量同地点对落在 [time_short, time_long] 灰色地带，"
                "可考虑调宽 --time_long 阈值或降低 --time_short。"
            )
    if global_warnings:
        hints.append(f"⚠️ 处理过程产生 {len(global_warnings)} 条 warning（详见日志）。")

    # ----- JSON -----
    json_obj = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "metadata": args.metadata,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "distance_eps": args.distance_eps,
            "yaw_eps": args.yaw_eps,
            "time_short": args.time_short,
            "time_long": args.time_long,
            "fps": args.fps,
            "max_pairs_dim": args.max_pairs_dim,
            "sample_pairs_per_episode": args.sample_pairs_per_episode,
            "seed": args.seed,
            "clip_overlap_frames": args.clip_overlap_frames,
            "diff_view_yaw": args.diff_view_yaw,
        },
        "global": {
            "n_episodes": len(per_episode_stats),
            "total_frames": total_frames,
            "avg_frames_per_episode": avg_frames,
            "avg_duration_sec_per_episode": avg_duration,
            "n_location_revisit": total_loc,
            "n_camera_swing": total_swing,
            "n_same_loc_diff_view": total_diffview,
            "n_grey_zone": total_grey,
            "pct_location_revisit": loc_pct,
            "pct_camera_swing": swing_pct,
            "pct_same_loc_diff_view": diffview_pct,
        },
        "per_episode": [
            {
                "episode_id": s.episode_id,
                "total_frames": s.total_frames,
                "duration_sec": s.duration_sec,
                "counts": {
                    "location_revisit": s.n_location_revisit,
                    "camera_swing": s.n_camera_swing,
                    "same_loc_diff_view": s.n_same_loc_diff_view,
                    "grey_zone": s.n_grey_zone,
                },
                "n_location_revisit": s.n_location_revisit,
                "n_camera_swing": s.n_camera_swing,
                "n_same_loc_diff_view": s.n_same_loc_diff_view,
                "n_grey_zone": s.n_grey_zone,
                "bev_path": s.bev_path,
                "warnings": s.warnings,
            }
            for s in per_episode_stats
        ],
        "hints": hints,
    }
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_obj, fh, ensure_ascii=False, indent=2)

    # ----- Markdown -----
    md_lines: List[str] = []
    md_lines.append("# 数据探查报告")
    md_lines.append("")
    md_lines.append(f"- 数据集: `{args.dataset_dir}` ({args.metadata})")
    md_lines.append(f"- 分析时间: {json_obj['generated_at']}")
    md_lines.append(
        f"- 分析参数: distance_eps={args.distance_eps}m, yaw_eps={args.yaw_eps}°, "
        f"time_short={args.time_short}s, time_long={args.time_long}s, fps={args.fps}, "
        f"diff_view_yaw={args.diff_view_yaw}°"
    )
    md_lines.append(
        f"- Clip overlap 处理: clip_overlap_frames={args.clip_overlap_frames} "
        f"(0=假设无 overlap；experiment_design.md 声明 5s clip + 0.5s overlap → 8)"
    )
    md_lines.append(f"- 采样上限: max_pairs_dim={args.max_pairs_dim} "
                    f"(单 episode 用于两两矩阵的帧数上限)")
    md_lines.append("")
    md_lines.append("## 全局统计")
    md_lines.append("")
    md_lines.append("| 项 | 值 |")
    md_lines.append("|---|---|")
    md_lines.append(f"| episode 数 | {len(per_episode_stats)} |")
    md_lines.append(f"| 总帧数 | {total_frames:,} |")
    md_lines.append(
        f"| 平均 episode 长度 | {avg_frames:,.0f} 帧 "
        f"(≈ {avg_duration:.1f} 秒 @ {args.fps}fps) |"
    )
    md_lines.append(
        f"| **位置重访对** (Δt > {args.time_long}s, dist < {args.distance_eps}m, "
        f"|yaw| < {args.yaw_eps}°) | {total_loc:,} ({loc_pct}) |"
    )
    md_lines.append(
        f"| **镜头摆动对** (Δt < {args.time_short}s, dist < "
        f"{args.distance_eps*0.5}m, |yaw| < {args.yaw_eps}°) | "
        f"{total_swing:,} ({swing_pct}) |"
    )
    md_lines.append(
        f"| **位置近视角不同** (dist < {args.distance_eps}m, "
        f"|yaw| > {args.diff_view_yaw}°) | "
        f"{total_diffview:,} ({diffview_pct}) |"
    )
    md_lines.append(
        f"| 灰色地带 ([{args.time_short}s, {args.time_long}s] 内同地点同朝向) | "
        f"{total_grey:,} |"
    )
    md_lines.append("")
    md_lines.append("> 占比 = 该类对数 / 三类对数总和（灰色地带不计入占比，仅诊断用）。")
    md_lines.append("")
    md_lines.append("## 结论提示")
    md_lines.append("")
    if hints:
        for h in hints:
            md_lines.append(f"- {h}")
    else:
        md_lines.append("- 无显著异常。")
    md_lines.append("")
    md_lines.append("## per-episode 表")
    md_lines.append("")
    md_lines.append("| episode_id | T 帧 | 时长(s) | 位置重访 | 镜头摆动 | "
                    "位置近视角不同 | 灰色地带 | BEV |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for s in per_episode_stats:
        bev_md = f"[bev]({s.bev_path})" if s.bev_path else "—"
        md_lines.append(
            f"| {s.episode_id} | {s.total_frames:,} | {s.duration_sec:.1f} | "
            f"{s.n_location_revisit:,} | {s.n_camera_swing:,} | "
            f"{s.n_same_loc_diff_view:,} | {s.n_grey_zone:,} | {bev_md} |"
        )
    md_lines.append("")
    if global_warnings:
        md_lines.append("## 全局 warning")
        md_lines.append("")
        for w in global_warnings:
            md_lines.append(f"- {w}")
        md_lines.append("")

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    logger.info("Wrote %s and %s", md_path, json_path)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "revisit_pairs"), exist_ok=True)

    explicit_ids: Optional[List[str]] = None
    if args.episode_ids:
        explicit_ids = [s.strip() for s in args.episode_ids.split(",") if s.strip()]

    all_clips = load_clip_metas(
        dataset_dir=dataset_dir,
        metadata_rel_path=args.metadata,
        episode_ids_filter=explicit_ids,
    )

    if not all_clips:
        logger.error("No episodes found after filtering, exiting.")
        return 1

    chosen_ids = _select_episodes(
        all_episode_ids=list(all_clips.keys()),
        explicit_ids=explicit_ids,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    logger.info("Will analyze %d episodes (out of %d candidate)",
                len(chosen_ids), len(all_clips))

    rng = np.random.default_rng(args.seed)
    per_episode_stats: List[EpisodeStats] = []
    global_warnings: List[str] = []

    for ep_id in tqdm(chosen_ids, desc="episodes"):
        clips = all_clips[ep_id]
        episode, overlap_hints = build_episode_data(
            ep_id, clips,
            clip_overlap_frames=args.clip_overlap_frames,
        )
        # 把自动检测的 overlap 提示进入全局 hint 池（不阻塞）
        for h in overlap_hints:
            global_warnings.append(h)
        if episode is None:
            global_warnings.append(f"Episode {ep_id} skipped (no usable poses.npy)")
            continue
        try:
            stats = process_episode(episode, args, output_dir, rng)
        except Exception as exc:  # noqa: BLE001
            logger.exception("process_episode failed for %s", ep_id)
            global_warnings.append(f"Episode {ep_id} crashed: {exc}")
            continue
        # 把 overlap 检测提示也归入该 episode 的 warnings（per-episode 可见）
        for h in overlap_hints:
            stats.warnings.append(h)
        per_episode_stats.append(stats)

    if not per_episode_stats:
        logger.error("No episode successfully processed.")
        write_summary(args, output_dir, per_episode_stats, global_warnings)
        return 2

    write_summary(args, output_dir, per_episode_stats, global_warnings)
    logger.info("Done. Output written to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
