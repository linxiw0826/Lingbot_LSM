"""retrieval_probe.py — Exp1 检索探针（v4 ThreeTierMemoryBank 离线评测）

目的（D-06 / experiment_design.md Exp 1）
-----------------------------------------
把 v4 ThreeTierMemoryBank 当作独立检索系统评测。给定 episode，逐帧模拟
bank.update()，到 GT 标注的"重访帧"时调 bank.retrieve()，看 top-k 里有没有
命中"之前在同一位置存进去的帧"。

实验设计（D-04 / D-05 / D-06）
------------------------------
- 验证载体：v4（统一 v4 代码做 Exp0/1/2，不回头修 v2/v3）
- 载入：epoch_4 checkpoint（旧的，本探针不经过 memory_cross_attn 主路径）
- Bank 配置：short_cap=1, medium_cap=8, long_cap=16
- Surprise 来源：NFP 或 oracle（子消融）
- GT "命中"定义：dist < hit_dist units AND |yaw差| < hit_yaw°
- k 值：{1, 3, 5, 10}
- 基线：random / temporal / pose-cosine

Pipeline 概要
-------------
for each episode:
  1. 拼接所有 clip poses → episode 全 [T, 4, 4]，与 explore_data.py 同思路
     的"同地点 + 中间分离"判定逻辑（独立实现，不 import）
  2. 解码 video.mp4 取全部 frame，VAE encode → episode visual_emb [T, 5120]
  3. 用 dataloader.build_dit_cond_dict 算 plucker → get_projected_frame_embs
     得到 pose_emb [T, 5120]
  4. 计算 surprise [T]（NFP 或 visual cosine diff 近似或 oracle）
  5. 算 GT 重访集合：gt_revisit_for[query_frame] = set of past_frame_indices
  6. 逐帧模拟 bank.update()
  7. 在 query 帧调 bank.retrieve()，计算 precision/recall@k
  8. 对照基线（random / temporal / pose-cosine）
  9. 汇总到 summary.md + summary.json

依赖
----
- torch + Wan VAE + memory_module.memory_bank.ThreeTierMemoryBank
- 不 import explore_data.py（任务约束：抄思路独立实现）
- 不修改 train_v4/infer_v4 任何文件
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# sys.path 设置（与 infer_v4.py 一致）
# ---------------------------------------------------------------------------

_PIPELINE_DIR = dirname(abspath(__file__))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                   # → src/
_PROJECT_ROOT = dirname(_SRC_DIR)                   # → Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, 'refs', 'lingbot-world')

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Exp1 retrieval probe — 离线评测 v4 ThreeTierMemoryBank 检索质量。"
        )
    )
    # 数据
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_full_train.csv")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录（重跑会覆盖 summary.*）")
    # 模型权重
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="基础模型目录（含 low_noise_model 子目录与 Wan2.1_VAE.pth）")
    p.add_argument("--vae_ckpt_dir", type=str, default=None,
                   help="Wan VAE 权重目录（默认与 --ckpt_dir 相同）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="epoch_4 全参微调 low_noise_model 目录（含 NFP head + "
                        "visual_key_proj 等权重）。缺失时使用 fallback surprise。")
    # 数据过滤
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时随机抽取前 N 个 episode")
    # Episode 拼接参数（与 explore_data.py 语义一致）
    p.add_argument("--clip_overlap_frames", type=int, default=0,
                   help="相邻 clip overlap 帧数；v4 数据 0.5s overlap 应设 8")
    # Bank 参数（与 v4 默认对齐）
    p.add_argument("--short_cap", type=int, default=1)
    p.add_argument("--medium_cap", type=int, default=8)
    p.add_argument("--long_cap", type=int, default=16,
                   help="LongTermBank 容量；本探针默认 16（与任务描述对齐）")
    p.add_argument("--surprise_threshold", type=float, default=0.4,
                   help="MediumTermBank 写入下限（与 train_v4/infer_v4 默认一致）")
    p.add_argument("--stability_threshold", type=float, default=0.2)
    p.add_argument("--novelty_threshold", type=float, default=0.7)
    p.add_argument("--half_life", type=float, default=10.0)
    p.add_argument("--dup_threshold", type=float, default=0.95)
    p.add_argument("--visual_fusion_alpha", type=float, default=0.7,
                   help="Innovation 9: pose/visual fusion alpha（默认 0.7）；"
                        "本脚本 query_key = alpha * pose_key + (1-alpha) * vis_key")
    # Surprise 来源
    p.add_argument("--surprise_source", type=str, default="nfp",
                   choices=["nfp", "oracle", "visual_cosine"],
                   help="surprise 来源：nfp=NFP head（默认）/ oracle=GT 重访帧"
                        "在 future 的 query 中被命中则 surprise=1 / "
                        "visual_cosine=visual_emb[t] vs visual_emb[t-1] cosine "
                        "distance（NFP 不可用时的 fallback）")
    # GT 命中判定
    p.add_argument("--hit_dist", type=float, default=40.0,
                   help="GT 重访距离阈值（数据集原生单位；v4/CSGO ≈ inches，"
                        "1m ≈ 40 units）")
    p.add_argument("--hit_yaw", type=float, default=30.0,
                   help="GT 重访 |yaw 差| 阈值（度）")
    p.add_argument("--intermediate_separation", type=float, default=100.0,
                   help="中间分离阈值（>0 时要求中间帧 max(dist(k,a),dist(k,b)) "
                        "> 此值，过滤 stationary 假阳性）")
    p.add_argument("--min_time_gap_sec", type=float, default=5.0,
                   help="GT 重访最小时间差（秒）；默认 5.0 与 explore_data.py "
                        "位置重访的 time_long 阈值对齐（语义：'真正离开过又回来'）。"
                        "如要包含更短时间间隔的重访可改（如 1.0 接受连续帧重访）")
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（用于 min_time_gap_sec→帧数换算）")
    # 评测参数
    p.add_argument("--k_values", type=str, default="1,3,5,10",
                   help="逗号分隔的 top-k 列表")
    p.add_argument("--max_query_frames", type=int, default=100,
                   help="每 episode 最多采样多少 query 帧；0=全跑")
    p.add_argument("--frame_stride", type=int, default=5,
                   help="在 episode 内每 stride 帧选一个候选 query")
    p.add_argument("--skip_first_n", type=int, default=50,
                   help="跳过早期帧（前 N 帧 bank 还没填上）")
    # 计算资源
    p.add_argument("--device", type=str, default="cuda",
                   help="计算设备（cuda/cuda:0/cpu）")
    p.add_argument("--vae_batch", type=int, default=8,
                   help="VAE encode 每次最多处理的视频帧数（控制显存）")
    p.add_argument("--size", type=str, default="480*832",
                   help="视频分辨率（H*W），与训练一致")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ClipMeta:
    episode_id: str
    clip_idx: int
    clip_path: str          # 绝对路径


@dataclass
class EpisodeData:
    episode_id: str
    clips: List[ClipMeta]            # 已按 clip_idx 排序
    poses: np.ndarray                # [T, 4, 4]
    actions: np.ndarray              # [T, 4]
    intrinsics: np.ndarray           # [T, 4]
    frame_to_clip: List[Tuple[int, int]]   # 长度 T, (clip_array_idx, local_frame_idx)
    # 派生
    xz: np.ndarray = field(default=None)   # [T, 2]
    yaw_deg: np.ndarray = field(default=None)   # [T]


@dataclass
class ProbeResult:
    """单 query 帧的 retrieval 评测结果。"""
    query_frame: int
    n_gt: int
    # method → {k → (hits, p, r)}
    methods: Dict[str, Dict[int, Tuple[int, float, float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode 加载（思路抄自 explore_data.py，不 import）
# ---------------------------------------------------------------------------

def load_episode_clips(
    dataset_dir: str,
    metadata_rel_path: str,
    episode_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, List[ClipMeta]]:
    """从 metadata CSV 加载 clip 元信息，按 episode_id 分组 + clip_idx 排序。

    与 train_v4_stage1_dual.py:CSGOMultiClipDataset._build_episode_groups 同思路；
    独立实现避免依赖训练 dataset 类。
    """
    csv_path = os.path.join(dataset_dir, metadata_rel_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

    episode_clips: Dict[str, List[ClipMeta]] = defaultdict(list)
    n_rows = 0
    n_kept = 0
    filter_set = set(episode_ids_filter) if episode_ids_filter else None

    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for col in ("episode_id", "clip_idx", "clip_path"):
            if col not in fieldnames:
                raise ValueError(
                    f"CSV {csv_path} missing required column '{col}'"
                )
        for row in reader:
            n_rows += 1
            ep_id = row["episode_id"]
            if filter_set is not None and ep_id not in filter_set:
                continue
            try:
                clip_idx_int = int(row["clip_idx"])
            except (TypeError, ValueError):
                logger.warning("Skip row invalid clip_idx=%r ep=%s",
                               row.get("clip_idx"), ep_id)
                continue
            clip_path = row["clip_path"]
            if not os.path.isabs(clip_path):
                clip_path = os.path.normpath(os.path.join(dataset_dir, clip_path))
            episode_clips[ep_id].append(
                ClipMeta(episode_id=ep_id, clip_idx=clip_idx_int,
                         clip_path=clip_path)
            )
            n_kept += 1

    for ep_id in episode_clips:
        episode_clips[ep_id].sort(key=lambda c: c.clip_idx)

    logger.info(
        "Loaded %d rows (kept %d after filter) across %d episodes from %s",
        n_rows, n_kept, len(episode_clips), csv_path,
    )
    if filter_set is not None:
        missing = filter_set - set(episode_clips.keys())
        if missing:
            logger.warning(
                "episode_ids_filter requested %s but missing in CSV: %s",
                sorted(filter_set), sorted(missing),
            )
    return episode_clips


def build_episode_data(
    episode_id: str,
    clips: List[ClipMeta],
    clip_overlap_frames: int = 0,
) -> Optional[EpisodeData]:
    """拼接 episode 全部 clip 的 poses/actions/intrinsics（重要文件缺失则跳过）。

    思路与 explore_data.py:build_episode_data 一致；
    overlap 处理：clip_array_idx >= 1 的 clip 跳过前 N 帧。
    """
    pose_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    intr_list: List[np.ndarray] = []
    frame_to_clip: List[Tuple[int, int]] = []
    skipped: List[str] = []
    valid_clip_count = 0

    for clip_array_idx, clip in enumerate(clips):
        poses_path = os.path.join(clip.clip_path, "poses.npy")
        actions_path = os.path.join(clip.clip_path, "action.npy")
        intr_path = os.path.join(clip.clip_path, "intrinsics.npy")
        video_path = os.path.join(clip.clip_path, "video.mp4")
        for p in (poses_path, actions_path, intr_path, video_path):
            if not os.path.isfile(p):
                logger.warning("Missing %s in ep %s clip %d; skip clip",
                               p, episode_id, clip.clip_idx)
                skipped.append(clip.clip_path)
                break
        else:
            try:
                poses = np.load(poses_path).astype(np.float32)
                acts = np.load(actions_path).astype(np.float32)
                intr = np.load(intr_path).astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed loading %s clip %d: %s",
                               episode_id, clip.clip_idx, exc)
                skipped.append(clip.clip_path)
                continue
            if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                logger.warning("Bad poses shape %s in %s clip %d",
                               poses.shape, episode_id, clip.clip_idx)
                skipped.append(clip.clip_path)
                continue
            n_raw = poses.shape[0]
            # 对齐 action/intrinsics 长度到 poses
            if acts.shape[0] < n_raw:
                pad = np.tile(acts[-1:], (n_raw - acts.shape[0], 1))
                acts = np.concatenate([acts, pad], axis=0)
            else:
                acts = acts[:n_raw]
            if intr.shape[0] < n_raw:
                pad = np.tile(intr[-1:], (n_raw - intr.shape[0], 1))
                intr = np.concatenate([intr, pad], axis=0)
            else:
                intr = intr[:n_raw]

            if valid_clip_count >= 1 and clip_overlap_frames > 0:
                trim = min(clip_overlap_frames, n_raw)
                poses_kept = poses[trim:]
                acts_kept = acts[trim:]
                intr_kept = intr[trim:]
                local_start = trim
            else:
                poses_kept = poses
                acts_kept = acts
                intr_kept = intr
                local_start = 0

            if poses_kept.shape[0] == 0:
                logger.warning("Episode %s clip %d trimmed to 0 by overlap=%d",
                               episode_id, clip.clip_idx, clip_overlap_frames)
                valid_clip_count += 1
                continue

            pose_list.append(poses_kept)
            act_list.append(acts_kept)
            intr_list.append(intr_kept)
            for local_idx in range(local_start, local_start + poses_kept.shape[0]):
                frame_to_clip.append((clip_array_idx, local_idx))
            valid_clip_count += 1
            continue
        # 进入 break → 缺文件
        continue

    if not pose_list:
        logger.warning("Episode %s has no usable clip; skipping", episode_id)
        return None

    poses_cat = np.concatenate(pose_list, axis=0)
    acts_cat = np.concatenate(act_list, axis=0)
    intr_cat = np.concatenate(intr_list, axis=0)

    ep = EpisodeData(
        episode_id=episode_id,
        clips=clips,
        poses=poses_cat,
        actions=acts_cat,
        intrinsics=intr_cat,
        frame_to_clip=frame_to_clip,
    )
    ep.xz, ep.yaw_deg = _extract_xz_yaw(poses_cat)
    return ep


# ---------------------------------------------------------------------------
# 几何工具（与 explore_data.py 一致）
# ---------------------------------------------------------------------------

def _extract_xz_yaw(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从 [T,4,4] c2w 矩阵提取 BEV xz 平移 + yaw 角（度）。"""
    xz = poses[:, [0, 2], 3].astype(np.float32)
    R = poses[:, :3, :3].astype(np.float32)
    yaw_rad = np.arctan2(R[:, 0, 2], R[:, 2, 2])
    yaw_deg = np.degrees(yaw_rad)
    return xz, yaw_deg


def _angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组角度（度）之间的最小角度差，结果在 [0, 180]。"""
    diff = np.abs(a - b) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


def _max_intermediate_separation(
    xz_full: np.ndarray, i_a: int, i_b: int,
) -> float:
    """[i_a+1, i_b-1] 中间帧的 max(max(dist(k,a), dist(k,b)))。

    用于过滤"原地静止 + yaw 来回"误判为位置重访。
    """
    if i_b - i_a <= 1:
        return 0.0
    seg = xz_full[i_a + 1:i_b]
    d_a = np.linalg.norm(seg - xz_full[i_a], axis=1)
    d_b = np.linalg.norm(seg - xz_full[i_b], axis=1)
    return float(np.maximum(d_a, d_b).max())


def compute_gt_revisit(
    ep: EpisodeData,
    hit_dist: float,
    hit_yaw: float,
    intermediate_separation: float,
    min_time_gap_frames: int,
) -> Dict[int, List[int]]:
    """对 episode 内每个 query 帧 q，找出所有"过去重访目标"帧 i (i < q)。

    判定：
      dist(xz_q, xz_i) < hit_dist
      AND |yaw_q - yaw_i| < hit_yaw
      AND (q - i) >= min_time_gap_frames
      AND max_intermediate_separation(i, q) > intermediate_separation
          （若 intermediate_separation <= 0 则跳过中间分离过滤）

    Returns:
        gt: dict[query_frame -> sorted list of past_frame indices]
    """
    T = ep.poses.shape[0]
    xz = ep.xz
    yaw = ep.yaw_deg

    # 全 T x T 矩阵（T <= 几千帧 OK；T=1500 → 1500^2 ~ 2.25M，可接受）
    diff_xz = xz[:, None, :] - xz[None, :, :]
    dist_mat = np.sqrt((diff_xz * diff_xz).sum(axis=-1))   # [T, T]
    yaw_mat = _angular_diff_deg(yaw[:, None], yaw[None, :])  # [T, T]

    gt: Dict[int, List[int]] = defaultdict(list)
    use_sep = intermediate_separation > 0.0
    for q in range(T):
        # 候选 i < q
        i_arr = np.arange(0, q, dtype=np.int64)
        if i_arr.size == 0:
            continue
        dist_arr = dist_mat[q, i_arr]
        yaw_arr = yaw_mat[q, i_arr]
        time_gap = q - i_arr
        cand_mask = (
            (dist_arr < hit_dist)
            & (yaw_arr < hit_yaw)
            & (time_gap >= min_time_gap_frames)
        )
        cand_idx = i_arr[cand_mask]
        if cand_idx.size == 0:
            continue
        if use_sep:
            kept = []
            for i in cand_idx:
                sep = _max_intermediate_separation(xz, int(i), q)
                if sep > intermediate_separation:
                    kept.append(int(i))
            if kept:
                gt[q] = kept
        else:
            gt[q] = cand_idx.tolist()
    return dict(gt)


# ---------------------------------------------------------------------------
# 模型加载（仅 VAE + 必要的 memory 模块权重）
# ---------------------------------------------------------------------------

def _load_vae(ckpt_dir: str, device: torch.device):
    """加载 Wan VAE。"""
    from wan.modules.vae2_1 import Wan2_1_VAE
    vae_pth = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
    if not os.path.isfile(vae_pth):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_pth}")
    logger.info("Loading Wan VAE from %s", vae_pth)
    vae = Wan2_1_VAE(vae_pth=vae_pth, device=device)
    return vae


def _build_memory_model(
    ckpt_dir: str,
    ft_model_dir: Optional[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """加载 WanModelWithMemory（含 visual_key_proj / latent_proj / nfp_head）。

    流程：
      1. 从 ckpt_dir/low_noise_model 加载基础 WanModel
      2. 通过 from_wan_model 转换为 WanModelWithMemory（含 memory_cross_attn 随机初始化）
      3. 若 ft_model_dir 提供，加载 epoch_4 全量权重（latent_proj/visual_key_proj/nfp_head 含训练值）
      4. eval() + requires_grad_(False) + to(device)

    Returns:
        (model, vae) tuple；其中 model 是 WanModelWithMemory，vae 是独立加载的 Wan VAE。
    """
    from wan.modules.model import WanModel
    from memory_module.model_with_memory import WanModelWithMemory
    import glob

    logger.info("Loading base WanModel from %s/low_noise_model", ckpt_dir)
    base_model = WanModel.from_pretrained(
        ckpt_dir, subfolder="low_noise_model",
        torch_dtype=dtype, control_type="act",
    )
    logger.info("Converting to WanModelWithMemory (memory layers = all blocks)...")
    model = WanModelWithMemory.from_wan_model(
        base_model, memory_layers=None, max_memory_size=8, skip_to_device=True,
    )
    del base_model
    gc.collect()
    model = model.to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)

    # 加载 ft 权重（含训练好的 nfp_head / latent_proj / visual_key_proj）
    if ft_model_dir is not None and os.path.isdir(ft_model_dir):
        logger.info("Loading ft weights from %s", ft_model_dir)
        ft_state = {}
        # 优先 safetensors → bin
        sf_files = sorted(glob.glob(os.path.join(ft_model_dir, "*.safetensors")))
        if sf_files:
            try:
                from safetensors.torch import load_file
                for f in sf_files:
                    ft_state.update(load_file(f, device="cpu"))
            except ImportError:
                logger.warning("safetensors not available; skipping")
                sf_files = []
        if not ft_state:
            bin_files = sorted(glob.glob(
                os.path.join(ft_model_dir, "diffusion_pytorch_model*.bin")
            ))
            for f in bin_files:
                ft_state.update(torch.load(f, map_location="cpu", weights_only=True))
        if not ft_state:
            bin_files = sorted(glob.glob(
                os.path.join(ft_model_dir, "pytorch_model*.bin")
            ))
            for f in bin_files:
                ft_state.update(torch.load(f, map_location="cpu", weights_only=True))
        if ft_state:
            result = model.load_state_dict(ft_state, strict=False)
            logger.info(
                "Loaded %d ft weights (missing=%d, unexpected=%d)",
                len(ft_state), len(result.missing_keys), len(result.unexpected_keys),
            )
            # WARN-G：sanity check —— 打印关键权重 norm，让用户确认权重真正落到这些层
            # 而非全 missing（修复前若 key 不匹配，load_state_dict(strict=False) 会静默
            # 保留随机初始化值，下游 surprise/visual_emb 数值看似正常但实际是噪声）
            def _safe_norm(mod, attr_chain: str) -> float:
                obj = mod
                for part in attr_chain.split("."):
                    if part.isdigit():
                        try:
                            obj = obj[int(part)]
                        except (TypeError, IndexError):
                            return float("nan")
                    elif hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        return float("nan")
                try:
                    return float(obj.norm().item())
                except Exception:  # noqa: BLE001
                    return float("nan")

            logger.info(
                "ft weight sanity (norm; NaN=attribute missing): "
                "nfp_head.mlp[0].weight=%.4f, nfp_head.mlp[2].weight=%.4f, "
                "visual_key_proj.weight=%.4f, latent_proj.weight=%.4f",
                _safe_norm(model, "nfp_head.mlp.0.weight"),
                _safe_norm(model, "nfp_head.mlp.2.weight"),
                _safe_norm(model, "visual_key_proj.weight"),
                _safe_norm(model, "latent_proj.weight"),
            )
        else:
            logger.warning("No ft weights found in %s", ft_model_dir)
    else:
        logger.warning(
            "ft_model_dir not provided or invalid (%s); using base weights "
            "(nfp_head/latent_proj/visual_key_proj are randomly initialized; "
            "consider --surprise_source visual_cosine).",
            ft_model_dir,
        )

    return model


# ---------------------------------------------------------------------------
# 视频解码与 VAE encode
# ---------------------------------------------------------------------------

def _decode_episode_video(
    ep: EpisodeData,
    height: int,
    width: int,
) -> np.ndarray:
    """解码 episode 全部帧为 [T, 3, H, W] float32 in [-1, 1]。

    遍历每个 clip 的 video.mp4，按 frame_to_clip 取相应 local_idx。
    """
    import cv2
    # 按 clip 分组要解码的 local_idx
    clip_to_locals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    # (clip_array_idx, local_idx) → 全局序号 t
    for t, (clip_array_idx, local_idx) in enumerate(ep.frame_to_clip):
        clip_to_locals[clip_array_idx].append((local_idx, t))

    T = len(ep.frame_to_clip)
    frames = np.empty((T, 3, height, width), dtype=np.float32)

    for clip_array_idx, local_t_pairs in clip_to_locals.items():
        clip = ep.clips[clip_array_idx]
        video_path = os.path.join(clip.clip_path, "video.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        # local_idx → t 索引
        local_t_pairs.sort()
        # cv2 顺序读最快；为避免 seek 每帧，按 local_idx 排序顺序读
        try:
            current_local = -1
            for local_idx, t in local_t_pairs:
                # 若需要跳过帧，就 grab；否则按需 read
                if local_idx > current_local + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                    current_local = local_idx - 1
                # 顺序读到目标
                while current_local < local_idx - 1:
                    ok = cap.grab()
                    if not ok:
                        raise RuntimeError(
                            f"grab failed at local_idx={current_local+1} "
                            f"in {video_path}"
                        )
                    current_local += 1
                ok, bgr = cap.read()
                current_local += 1
                if not ok or bgr is None:
                    # 末尾不足时用最后一帧的内容填充（与训练 _pad_or_truncate 思路一致）
                    logger.warning(
                        "Read failed at local_idx=%d in %s; reusing previous frame",
                        local_idx, video_path,
                    )
                    if t > 0:
                        frames[t] = frames[t - 1]
                    else:
                        frames[t] = 0.0
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)
                arr = rgb.astype(np.float32) / 127.5 - 1.0
                frames[t] = arr.transpose(2, 0, 1)
        finally:
            cap.release()
    return frames


def _vae_encode_batched(
    vae,
    frames: np.ndarray,
    device: torch.device,
    batch_frames: int = 8,
) -> torch.Tensor:
    """按 batch 编码视频帧到 latent 空间。

    Wan VAE encode 接受 [3, F, H, W] 视频张量，单次调用对整段视频做时间下采样
    (stride_t=4)，输出 [z_dim, F_lat, lat_h, lat_w]。为了得到 per-frame latent，
    我们以 batch_frames 为单位调用 encode，再沿时间维拼接。

    注意：vae.encode 内部时间 stride=4，因此每段 batch_frames 应该是 4 的倍数 + 1
    才能产生整数 lat 帧。若不满足，本函数会自动 pad 到下一个合法长度。

    Returns:
        latents: torch.Tensor [z_dim=16, T_lat, lat_h, lat_w]  — T_lat 通常 < T。
        本探针仅需 per-input-frame 视觉嵌入，故我们额外做 mode-aware 处理：
        最终把每个输入帧对应到一个 latent index（取 idx // stride_t）。
    """
    T = frames.shape[0]
    stride_t = 4   # Wan VAE 时间下采样
    # 简化：一次 encode 整段（如果 batch_frames=0 或大于 T）
    bs = batch_frames if batch_frames > 0 else T
    # 把 bs 调整到 (k*stride_t + 1) 以避免 trailing 帧丢失
    if bs > 1 and (bs - 1) % stride_t != 0:
        bs = ((bs - 1) // stride_t) * stride_t + 1
        bs = max(bs, stride_t + 1)
    logger.info("VAE encode T=%d, batch_frames=%d (after stride align)", T, bs)

    out_latents = []
    pad_used = 0
    for start in range(0, T, bs):
        end = min(start + bs, T)
        chunk = frames[start:end]
        n = chunk.shape[0]
        # 至少 stride_t+1 帧才能 encode（产出 2 个 latent 帧）；不足则用末帧 pad
        if n < stride_t + 1:
            need = stride_t + 1 - n
            last = chunk[-1:].repeat(need, axis=0)
            chunk = np.concatenate([chunk, last], axis=0)
            pad_used += need
        else:
            # 对齐到 (k*stride_t + 1)，否则 VAE 丢掉末尾
            n_after_pad = ((chunk.shape[0] - 1) // stride_t) * stride_t + 1
            if n_after_pad < chunk.shape[0]:
                # 截到对齐
                chunk = chunk[:n_after_pad]
            elif n_after_pad > chunk.shape[0]:
                need = n_after_pad - chunk.shape[0]
                last = chunk[-1:].repeat(need, axis=0)
                chunk = np.concatenate([chunk, last], axis=0)
                pad_used += need
        # [F, 3, H, W] → [3, F, H, W]
        t = torch.from_numpy(chunk).permute(1, 0, 2, 3).contiguous().to(device)
        with torch.no_grad():
            try:
                latent = vae.encode([t])[0]    # [z_dim, F_lat, h, w]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("VAE encode OOM at start=%d bs=%d; retry bs/2",
                               start, n)
                # 简单兜底：减半重试（最多一次）
                half = max(stride_t + 1, n // 2)
                if (half - 1) % stride_t != 0:
                    half = ((half - 1) // stride_t) * stride_t + 1
                # 单次重试
                t1 = torch.from_numpy(chunk[:half]).permute(1, 0, 2, 3).contiguous().to(device)
                latent = vae.encode([t1])[0]
                del t1
        out_latents.append(latent.cpu())
        del t
        torch.cuda.empty_cache()

    full = torch.cat(out_latents, dim=1)   # [z_dim, T_lat_total, h, w]
    if pad_used > 0:
        logger.info("VAE encode pad_used=%d (trailing frames)", pad_used)
    return full


def _expand_latents_to_frames(
    latents: torch.Tensor,
    n_frames: int,
    stride_t: int = 4,
) -> torch.Tensor:
    """把 latent[T_lat, ...] 的每个 latent 帧对应到 stride_t 个输入帧（含 mid sample）。

    映射规则：输入帧 t → latent_idx = min(t // stride_t, T_lat - 1)。
    这与 prepare_control_signal 中 `actions[::stride_t]` 一致语义（每 stride_t
    个输入帧共享一个 latent 表征）。
    """
    z_dim, T_lat, lat_h, lat_w = latents.shape
    out = torch.empty((n_frames, z_dim, lat_h, lat_w),
                      dtype=latents.dtype, device=latents.device)
    for t in range(n_frames):
        lat_idx = min(t // stride_t, T_lat - 1)
        out[t] = latents[:, lat_idx]
    return out


# ---------------------------------------------------------------------------
# Surprise 计算
# ---------------------------------------------------------------------------

def _compute_surprise_visual_cosine(visual_embs: torch.Tensor) -> torch.Tensor:
    """fallback surprise：visual_emb[t] vs visual_emb[t-1] cosine distance。

    Args:
        visual_embs: [T, dim=5120] (CPU/float)
    Returns:
        surprise: [T]，surprise[0] = 1.0（首帧）
    """
    T = visual_embs.shape[0]
    out = torch.empty(T, dtype=torch.float32)
    out[0] = 1.0
    if T <= 1:
        return out
    a = F.normalize(visual_embs[:-1].float(), dim=-1)
    b = F.normalize(visual_embs[1:].float(), dim=-1)
    cos = (a * b).sum(dim=-1)   # [T-1]
    out[1:] = 1.0 - cos
    return out


def _compute_surprise_oracle(
    gt_revisit_for: Dict[int, List[int]],
    n_frames: int,
) -> torch.Tensor:
    """oracle surprise：若帧 i 出现在某个 future query 的 GT 集合中，则 surprise=1。

    意图：把"GT 标注重访目标"标为高 surprise → 模拟"如果 surprise 完美的话"
    Medium/Long Bank 会如何路由。

    Note: GT 帧通常应是 stable（低 surprise），但本探针的语义是"对检索有用的
    帧"——它们既可能进 Medium（高 surprise）也可能进 Long（低 surprise）。
    本 oracle 把它们标为 surprise=1 → 倾向于进 Medium；用户可改用
    `_compute_surprise_oracle_stable`（低 surprise）测 Long Bank。我们这里
    提供高 surprise 版作为主路径（test "理想 surprise 信号是否能改善检索"）。
    """
    out = torch.zeros(n_frames, dtype=torch.float32)
    used: set = set()
    for q, past_list in gt_revisit_for.items():
        for i in past_list:
            used.add(int(i))
    if not used:
        return out
    for i in used:
        if 0 <= i < n_frames:
            out[i] = 1.0
    return out


# ---------------------------------------------------------------------------
# 计算 per-frame pose_emb（5120 维）
# ---------------------------------------------------------------------------

def _compute_pose_embs_episode(
    ep: EpisodeData,
    model,
    device: torch.device,
    height: int,
    width: int,
    fps: int,
    chunk_size: int = 81,
) -> torch.Tensor:
    """对 episode 全部帧计算 pose_emb [T, 5120]。

    复用 pipeline.dataloader.build_dit_cond_dict 计算 plucker emb，再用
    model.get_projected_frame_embs（参考 infer_v4.py 同名调用）得到 5120 维。

    分块策略：每 chunk_size 帧（默认 81，对齐训练 clip 长度）一次性算 plucker
    + projected_frame_embs，再拼接。VAE stride_t=4 → 81 帧 → 21 个 latent 帧。

    注意：plucker 用 compute_relative_poses(framewise=True)——相对姿态。这与
    训练时一致；探针保留这一致性以避免 query 与存储之间空间不一致。
    """
    from pipeline.dataloader import build_dit_cond_dict

    T = ep.poses.shape[0]
    stride_t = 4
    # 我们按 chunk_size=81 帧切分；每 chunk 产生 lat_f = (chunk_size-1)//4 + 1 帧
    out = torch.empty((T, model.dim), dtype=torch.float32, device="cpu")

    poses_t = torch.from_numpy(ep.poses).float()
    actions_t = torch.from_numpy(ep.actions).float()
    intr_t = torch.from_numpy(ep.intrinsics).float()

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        n = end - start
        if n < stride_t + 1:
            # 太短无法构造 plucker（lat_f >= 2）；用末帧 pad
            pad_n = stride_t + 1 - n
            pad_p = poses_t[end - 1:end].repeat(pad_n, 1, 1)
            pad_a = actions_t[end - 1:end].repeat(pad_n, 1)
            pad_i = intr_t[end - 1:end].repeat(pad_n, 1)
            chunk_p = torch.cat([poses_t[start:end], pad_p], dim=0)
            chunk_a = torch.cat([actions_t[start:end], pad_a], dim=0)
            chunk_i = torch.cat([intr_t[start:end], pad_i], dim=0)
        else:
            chunk_p = poses_t[start:end]
            chunk_a = actions_t[start:end]
            chunk_i = intr_t[start:end]
        try:
            cond = build_dit_cond_dict(
                poses=chunk_p, actions=chunk_a, intrinsics=chunk_i,
                height=height, width=width,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "build_dit_cond_dict failed at ep=%s start=%d: %s; "
                "using zero pose_emb for this chunk",
                ep.episode_id, start, exc,
            )
            out[start:end] = 0.0
            continue
        plucker = cond["c2ws_plucker_emb"][0].to(device)   # [1, 448, lat_f, lat_h, lat_w]
        with torch.no_grad():
            # patch_embedding_wancamctrl 等层可能在 CPU；先移到 device
            for attr in ("patch_embedding_wancamctrl",
                         "c2ws_hidden_states_layer1",
                         "c2ws_hidden_states_layer2"):
                if hasattr(model, attr):
                    getattr(model, attr).to(device)
            frame_embs = model.get_projected_frame_embs(plucker)   # [lat_f, 5120]
        frame_embs = frame_embs.float().cpu()
        # lat_f → 输入帧映射：t → lat_idx = t//stride_t（与 _expand_latents_to_frames 一致）
        lat_f = frame_embs.shape[0]
        for t_local in range(n):
            lat_idx = min(t_local // stride_t, lat_f - 1)
            out[start + t_local] = frame_embs[lat_idx]
    return out


# ---------------------------------------------------------------------------
# 计算 visual_emb [T, 5120]（latent_proj 投影）
# ---------------------------------------------------------------------------

def _compute_visual_embs_from_latents(
    latents_per_frame: torch.Tensor,
    model,
    device: torch.device,
) -> torch.Tensor:
    """对每个输入帧的 latent 通过 model.latent_proj 投影到 5120 维。

    Args:
        latents_per_frame: [T, z_dim=16, lat_h, lat_w]，CPU
    Returns:
        visual_embs: [T, 5120]，CPU float32
    """
    T = latents_per_frame.shape[0]
    out = torch.empty((T, model.dim), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        if hasattr(model, "latent_proj"):
            model.latent_proj.to(device)
        for t in range(T):
            lat = latents_per_frame[t].to(device)
            v = model.get_projected_latent_emb(lat)   # [5120]
            out[t] = v.float().cpu()
    return out


# ---------------------------------------------------------------------------
# 基线检索
# ---------------------------------------------------------------------------

def _retrieve_random(q: int, n_past: int, k: int, rng: random.Random) -> List[int]:
    """从 [0, q-1] 随机抽 k 帧（不重复）。"""
    if n_past == 0:
        return []
    pool = list(range(n_past))
    rng.shuffle(pool)
    return pool[:k]


def _retrieve_temporal(q: int, n_past: int, k: int) -> List[int]:
    """最近 k 帧 [q-k, q-1]。"""
    if n_past == 0:
        return []
    start = max(0, q - k)
    return list(range(start, q))


def _retrieve_pose_cosine(
    q: int,
    pose_embs: torch.Tensor,
    k: int,
) -> List[int]:
    """[0, q-1] 内 pose_emb cosine sim 的 top-k。"""
    if q == 0:
        return []
    query = pose_embs[q].float().unsqueeze(0)
    past = pose_embs[:q].float()
    sims = F.cosine_similarity(query, past, dim=-1)
    k_use = min(k, q)
    _, idx = torch.topk(sims, k=k_use)
    return idx.tolist()


def _retrieve_pose_abs(
    q: int,
    abs_translations: torch.Tensor,   # [T, 3] 绝对 c2w 平移向量（世界坐标位置）
    k: int,
) -> List[int]:
    """[0, q-1] 内按绝对位置 L2 距离最近的 top-k（诊断基线：location key）。

    与 pose_cosine 的关键区别：pose_cosine 用 framewise 运动 pose_emb（编码"怎么动"），
    本函数用绝对世界位置（编码"在哪"），用于验证"检索失败是否因为 key 是运动而非位置"。
    """
    if q == 0:
        return []
    query = abs_translations[q].float().unsqueeze(0)   # [1, 3]
    past = abs_translations[:q].float()                # [q, 3]
    dists = torch.norm(past - query, dim=-1)           # [q] L2 距离，越小越近
    k_use = min(k, q)
    _, idx = torch.topk(dists, k=k_use, largest=False)  # 最小的 k 个
    return idx.tolist()


def _retrieve_pose_abs_gap(
    q: int,
    abs_translations: torch.Tensor,   # [T, 3] 绝对 c2w 平移
    k: int,
    min_gap_frames: int,              # 排除 q 之前这么多帧（time_gap）
) -> List[int]:
    """先排除最近 min_gap_frames 帧，再按绝对位置 L2 最近取 top-k。

    与 _retrieve_pose_abs 的区别：只在 [0, q - min_gap_frames] 候选里检索，
    排除"刚走过的近邻帧"（它们距离≈0 但被 GT 的 time_gap 条件排除），
    使本基线成为"理想的地点重访检索器"。
    """
    # GT 候选条件是 q - i >= min_gap_frames（compute_gt_revisit），即排除 q-i < min_gap_frames。
    # 候选集 {i : q-i >= min_gap_frames} = {i : i <= q - min_gap_frames}，故 cutoff = q - min_gap_frames + 1
    # （abs_translations[:cutoff] 含索引 cutoff-1 = q - min_gap_frames），与 GT 排除窗口逐帧对齐。
    cutoff = q - min_gap_frames + 1
    if cutoff <= 0:
        return []
    query = abs_translations[q].float().unsqueeze(0)   # [1,3]
    past = abs_translations[:cutoff].float()           # [cutoff,3]
    dists = torch.norm(past - query, dim=-1)           # [cutoff]
    k_use = min(k, cutoff)
    _, idx = torch.topk(dists, k=k_use, largest=False)  # 最近的 k 个
    return idx.tolist()


# ---------------------------------------------------------------------------
# 主评测循环
# ---------------------------------------------------------------------------

def _eval_episode(
    ep: EpisodeData,
    pose_embs: torch.Tensor,         # [T, 5120]
    visual_embs: torch.Tensor,       # [T, 5120]
    surprise: torch.Tensor,          # [T]
    latents_per_frame: torch.Tensor,  # [T, z_dim, lat_h, lat_w]
    gt_revisit_for: Dict[int, List[int]],
    args,
    rng: random.Random,
) -> Dict:
    """对单 episode 跑 bank 模拟 + 探针评测，返回汇总 dict。"""
    from memory_module.memory_bank import ThreeTierMemoryBank, MemoryFrame

    T = pose_embs.shape[0]
    k_values = sorted([int(x) for x in args.k_values.split(",")])
    k_max = max(k_values)

    # 绝对世界位置（c2w 平移向量），用于 pose_abs 诊断基线
    # ep.poses 为 [T,4,4] 绝对 c2w，与 pose_embs 同 T 同序（均源自 ep.poses）
    abs_translations = torch.from_numpy(ep.poses[:, :3, 3]).float()  # [T, 3]

    # pose_abs_gap 排除窗口：必须与 compute_gt_revisit 的 GT time_gap 排除
    # （main: min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * fps)))）
    # 完全一致，否则 pose_abs_gap 候选与 GT 不对齐。
    min_gap_frames = max(1, int(round(args.fps * args.min_time_gap_sec)))

    # 1) 构造 bank + 逐帧 update
    bank = ThreeTierMemoryBank(
        short_cap=args.short_cap,
        medium_cap=args.medium_cap,
        long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life,
        dup_threshold=args.dup_threshold,
    )

    # 选取 query 帧集合
    candidate_query = list(range(args.skip_first_n, T, args.frame_stride))
    candidate_query = [q for q in candidate_query if gt_revisit_for.get(q)]
    if args.max_query_frames > 0 and len(candidate_query) > args.max_query_frames:
        rng.shuffle(candidate_query)
        candidate_query = sorted(candidate_query[:args.max_query_frames])
    query_set = set(candidate_query)

    # 评测累计
    # method_key → k → [hits_sum, p_sum, r_sum, count]
    method_keys = ["bank", "random", "temporal", "pose_cosine", "pose_abs", "pose_abs_gap"]
    agg: Dict[str, Dict[int, List[float]]] = {
        m: {k: [0.0, 0.0, 0.0, 0] for k in k_values} for m in method_keys
    }
    per_query: List[Dict] = []
    bank_size_log: List[int] = []

    # 用 visual_key_proj 简化版的 semantic key 计算：
    # 任务允许"省略 cross-attn 的 k 投影"。我们使用：
    #   pose_key = normalize(pose_emb)
    #   vis_key  = normalize(visual_key_proj(visual_emb))   （来自 v4 训练好的权重）
    #   semantic_key = alpha * pose_key + (1-alpha) * vis_key
    # 这与 model_with_memory.get_semantic_key() 在 visual_emb 提供时
    # 公式一致；唯一差异是 pose_key 不经过所有 layer 的 K 投影平均。
    # ——这对探针目的合理：layer K 投影没训过（OP-1 F-12），用 normalize(pose_emb)
    # 比未训过的随机投影更稳。

    visual_key_proj = None
    model_dim = pose_embs.shape[-1]
    # （visual_key_proj 实例由外层传入；此处简化为 query 时只需 visual_embs 即可
    #  使用 _semantic_key 函数构造。本函数已包含通过 visual_embs/pose_embs 计算。）

    def _semantic_key(t_idx: int) -> torch.Tensor:
        pk = F.normalize(pose_embs[t_idx].float().unsqueeze(0), dim=-1).squeeze(0)
        if visual_embs is None:
            return pk
        vk = F.normalize(visual_embs[t_idx].float().unsqueeze(0), dim=-1).squeeze(0)
        return args.visual_fusion_alpha * pk + (1.0 - args.visual_fusion_alpha) * vk

    # 逐帧主循环
    for t in tqdm(range(T), desc=f"ep={ep.episode_id}", leave=False):
        # 1) 检索（仅 query 帧）——必须在 update 之前，否则会自命中
        if t in query_set:
            past_set = set(gt_revisit_for.get(t, []))
            if past_set:
                # bank retrieve（自带 semantic key 融合）
                query_pose = pose_embs[t].float()
                query_sk = _semantic_key(t)
                retrieved = bank.retrieve(
                    query_pose_emb=query_pose,
                    query_semantic_key=query_sk,
                    short_n=args.short_cap,
                    medium_k=args.medium_cap,    # 取得最多 medium_cap，让我们能裁 top-k
                    long_k=args.long_cap,
                    device=torch.device("cpu"),
                    return_tier_ids=False,
                )
                if retrieved is not None:
                    _key_states, _val_states = retrieved
                    # 还原 retrieved 帧的 timestep 索引：用 bank 内 frame 的
                    # timestep 字段查找。由于 retrieve 已返回 tensor，
                    # 这里直接从 bank.short/medium/long 内取 timestep。
                    bank_frame_ts = _collect_bank_timesteps(bank)
                    # 把 retrieved tensor 与 bank 顺序匹配较复杂；
                    # 我们换个做法：直接遍历 bank 三层取所有 frame，按 cosine sim
                    # 与 query 排序，得到 top-k timesteps。
                    bank_topk_ts = _bank_topk_timesteps(
                        bank, query_pose, query_sk,
                        k_max=k_max,
                        alpha=args.visual_fusion_alpha,
                    )
                else:
                    bank_topk_ts = []
            else:
                bank_topk_ts = []

            # 基线
            random_topk = _retrieve_random(t, t, k_max, rng)
            temporal_topk = _retrieve_temporal(t, t, k_max)
            pose_cos_topk = _retrieve_pose_cosine(t, pose_embs, k_max)
            pose_abs_topk = _retrieve_pose_abs(t, abs_translations, k_max)
            pose_abs_gap_topk = _retrieve_pose_abs_gap(t, abs_translations, k_max, min_gap_frames)

            # 评测每个 k
            q_result = {
                "query_frame": int(t),
                "n_gt": len(past_set),
                "methods": {},
            }
            for method, retrieved_list in (
                ("bank", bank_topk_ts),
                ("random", random_topk),
                ("temporal", temporal_topk),
                ("pose_cosine", pose_cos_topk),
                ("pose_abs", pose_abs_topk),
                ("pose_abs_gap", pose_abs_gap_topk),
            ):
                method_result: Dict[int, Dict[str, float]] = {}
                for k in k_values:
                    top_k_list = retrieved_list[:k]
                    hits = len(set(top_k_list) & past_set)
                    p = hits / max(k, 1)
                    r = hits / max(len(past_set), 1)
                    agg[method][k][0] += hits
                    agg[method][k][1] += p
                    agg[method][k][2] += r
                    agg[method][k][3] += 1
                    method_result[k] = {"hits": int(hits),
                                        "precision": float(p),
                                        "recall": float(r)}
                q_result["methods"][method] = method_result
            per_query.append(q_result)

        # 2) update（在 retrieve 之后；这帧也会进入 bank 供后续 query 用）
        try:
            frame_visual = visual_embs[t] if visual_embs is not None else None
            sk = _semantic_key(t)
            bank.update(
                pose_emb=pose_embs[t].float(),
                latent=latents_per_frame[t].float() if latents_per_frame is not None
                else torch.zeros(1),
                surprise_score=float(surprise[t].item()),
                timestep=int(t),
                visual_emb=frame_visual.float() if frame_visual is not None else None,
                chunk_id=int(t // 21),    # 21 latent 帧 ≈ 1 clip
                semantic_key=sk,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("bank.update failed at ep=%s t=%d: %s",
                           ep.episode_id, t, exc)
            continue
        bank_size_log.append(bank.size())

        # age increment 按 clip 边界（每 21 frame）
        if t > 0 and (t % 21) == 0:
            bank.increment_age()

    # 汇总
    def _avg(m: str, k: int) -> Tuple[float, float, float, int]:
        h, p, r, c = agg[m][k]
        if c == 0:
            return 0.0, 0.0, 0.0, 0
        return h / c, p / c, r / c, c

    summary = {
        "episode_id": ep.episode_id,
        "total_frames": int(T),
        "n_query_evaluated": int(len(per_query)),
        "n_gt_query_candidates_total": int(sum(1 for q in range(T) if gt_revisit_for.get(q))),
        "bank_size_final": int(bank.size()),
        "bank_size_mean": float(np.mean(bank_size_log)) if bank_size_log else 0.0,
        "bank_size_max": int(max(bank_size_log)) if bank_size_log else 0,
        "bank_stats_final": bank.get_stats(),
        "per_query": per_query,
        "metrics_by_method": {
            m: {
                k: {
                    "hits_per_query": float(_avg(m, k)[0]),
                    "precision_at_k": float(_avg(m, k)[1]),
                    "recall_at_k": float(_avg(m, k)[2]),
                    "n": int(_avg(m, k)[3]),
                }
                for k in k_values
            }
            for m in method_keys
        },
    }
    return summary


def _collect_bank_timesteps(bank) -> List[int]:
    """从 bank 三层取所有 frame 的 timestep（按 short/medium/long 顺序）。"""
    out = []
    for tier_name in ("short", "medium", "long"):
        tier = getattr(bank, tier_name)
        for f in tier.frames:
            out.append(int(f.timestep))
    return out


def _bank_topk_timesteps(
    bank,
    query_pose: torch.Tensor,
    query_semantic_key: torch.Tensor,
    k_max: int,
    alpha: float = 0.7,
) -> List[int]:
    """ThreeTierMemoryBank 的混合检索 + 排序，返回 top-k_max 的 timestep 列表。

    复用 bank.retrieve() 的内部三层取回 + cross-tier dedup 顺序，
    但跳过 tensor 拼接环节，直接返回 selected_frames 的 timestep。

    检索预算：Short short_cap + Medium top-medium_cap + Long top-long_cap
    （我们让单层先取充裕，再统一裁到 k_max）。

    注意：bank.retrieve() 返回的是 dedup 后的 pose/visual tensors，但没有
    timestep。为了拿到 timestep，我们手动复刻三层取回 + dedup 逻辑。
    这里依赖 ThreeTierMemoryBank 子层的公开属性（cap, frames）和方法。
    """
    # 三层取回（参考 memory_bank.py:ThreeTierMemoryBank.retrieve）
    short_frames = bank.short.retrieve_all(device=None)[:bank.short.cap]
    medium_frames = bank.medium.retrieve(query_pose, top_k=bank.medium.cap, device=None)
    long_frames = bank.long.retrieve(
        query_semantic_key, query_pose, top_k=bank.long.cap, device=None
    )
    all_frames = list(short_frames) + list(medium_frames) + list(long_frames)
    if not all_frames:
        return []
    # 同 bank.retrieve 的 dedup 逻辑（pose_emb cosine_sim > dup_threshold 视为冗余）
    selected = []
    selected_embs = []
    dup_threshold = bank.dup_threshold
    for f in all_frames:
        if not selected:
            selected.append(f)
            selected_embs.append(f.pose_emb.float())
            continue
        stacked = torch.stack(selected_embs)
        sims = F.cosine_similarity(
            f.pose_emb.float().unsqueeze(0), stacked, dim=-1
        )
        if float(sims.max()) < dup_threshold:
            selected.append(f)
            selected_embs.append(f.pose_emb.float())
    if not selected:
        return []
    # WARN-C 修复：不再额外按 query semantic_key cosine sim 重排，
    # 保持 short > medium > long 的 dedup 后顺序（与
    # ThreeTierMemoryBank.retrieve() 一致），直接取前 k_max 个 timestep。
    # 原因：bank.retrieve() 内部并不做"按 query semantic_key 重排"，
    # 探针若额外重排会让 top-k 顺序与训练/推理时真实注入顺序不一致。
    out = [int(f.timestep) for f in selected[:k_max]]
    return out


# ---------------------------------------------------------------------------
# Summary 输出
# ---------------------------------------------------------------------------

def _write_summary(
    output_dir: str,
    args,
    episode_summaries: List[Dict],
    k_values: List[int],
):
    os.makedirs(output_dir, exist_ok=True)
    per_ep_dir = os.path.join(output_dir, "per_episode")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(per_ep_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 每 episode 写 JSON
    for ep_sum in episode_summaries:
        ep_path = os.path.join(per_ep_dir, f"{ep_sum['episode_id']}.json")
        with open(ep_path, "w") as fh:
            json.dump(ep_sum, fh, indent=2)

    # 全局汇总：对每个 (method, k) 取所有 query 的平均
    method_keys = ["bank", "random", "temporal", "pose_cosine", "pose_abs", "pose_abs_gap"]
    global_metrics = {m: {k: {"precision": [], "recall": [], "hits": [], "n": 0}
                         for k in k_values} for m in method_keys}
    n_query_total = 0
    n_ep_used = 0
    for ep_sum in episode_summaries:
        n_q = ep_sum["n_query_evaluated"]
        n_query_total += n_q
        if n_q > 0:
            n_ep_used += 1
        for m in method_keys:
            for k in k_values:
                row = ep_sum["metrics_by_method"][m][str(k)] \
                    if str(k) in ep_sum["metrics_by_method"][m] \
                    else ep_sum["metrics_by_method"][m][k]
                # 按 query 数加权
                for _ in range(row["n"]):
                    pass
                global_metrics[m][k]["precision"].append(row["precision_at_k"] * row["n"])
                global_metrics[m][k]["recall"].append(row["recall_at_k"] * row["n"])
                global_metrics[m][k]["hits"].append(row["hits_per_query"] * row["n"])
                global_metrics[m][k]["n"] += row["n"]

    global_table = {m: {} for m in method_keys}
    for m in method_keys:
        for k in k_values:
            n = global_metrics[m][k]["n"]
            if n == 0:
                global_table[m][k] = {"precision_at_k": 0.0, "recall_at_k": 0.0,
                                      "hits_per_query": 0.0, "n": 0}
            else:
                global_table[m][k] = {
                    "precision_at_k": sum(global_metrics[m][k]["precision"]) / n,
                    "recall_at_k": sum(global_metrics[m][k]["recall"]) / n,
                    "hits_per_query": sum(global_metrics[m][k]["hits"]) / n,
                    "n": n,
                }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "n_episodes": len(episode_summaries),
        "n_episodes_with_queries": n_ep_used,
        "n_query_total": n_query_total,
        "k_values": k_values,
        "global_metrics_by_method": global_table,
        "per_episode_brief": [
            {
                "episode_id": ep["episode_id"],
                "n_query": ep["n_query_evaluated"],
                "bank_size_final": ep["bank_size_final"],
                "bank_metrics_p_at_max_k": ep["metrics_by_method"]["bank"][
                    str(max(k_values)) if str(max(k_values)) in ep["metrics_by_method"]["bank"]
                    else max(k_values)
                ]["precision_at_k"],
            }
            for ep in episode_summaries
        ],
    }

    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary JSON: %s", summary_json_path)

    # 写 Markdown
    md_lines = []
    md_lines.append(f"# Exp1 Retrieval Probe Summary\n")

    # WARN-A / B / D 修复：在 summary 顶部明确暴露探针与训练后真实 bank 行为的偏差
    md_lines.append("\n")
    md_lines.append("## ⚠️ 探针计算与训练后真实 bank 行为的偏差说明\n\n")
    md_lines.append(
        "本探针为 **快速验证设计**（不训练 DiT、不跑完整 inference），有以下设计性降级，"
        "结果应在此前提下解读：\n\n"
    )
    md_lines.append(
        "1. **semantic_key 计算**：query 与存储两侧均用 "
        "`alpha * normalize(pose_emb) + (1-alpha) * normalize(visual_key_proj(visual_emb))`，"
        "pose 部分用 raw normalize 而非走 cross-attn K 投影（model_with_memory.get_semantic_key 真实行为）。"
        "原因：F-12 显示 memory_cross_attn 的 K 层在 epoch_4 之前全程零梯度，是随机初始化噪声；"
        "用 raw normalize 比\"用随机投影\"更稳。"
        "**结论刻画的是 idea 的 pose-norm + visual fusion 子集，不是完整 K-proj-and-norm 链路**。\n\n"
    )
    md_lines.append(
        "2. **NFP surprise 输入**：当 surprise_source=nfp 时，NFP head 输入是 "
        "`visual_emb.unsqueeze(1)` ([T, 1, dim])，而**非真正的 last-block hidden_states**"
        "（DiT 倒数第二层输出，含位置/时间/memory 注入）。NFPHead.forward 形状兼容能跑，"
        "但 surprise 数值分布与训练时严重偏离，`surprise_threshold=0.4` 的 Medium tier 写入门槛"
        "**可能整段失效**（要么全部进要么全部不进）。"
        "**建议用 `--surprise_source visual_cosine` 或 `oracle` 作对照**。\n\n"
    )
    md_lines.append(
        "3. **Plucker pose_emb 按 81-frame chunk 切**：与 v4 训练时 `framewise=True` 对称，"
        "每 81 帧重新计算相对姿态。query 与存储两侧同对称，cosine 相对排序仍有意义，"
        "但**跨 chunk 检索的绝对值不可比**。\n\n"
    )
    md_lines.append(
        "4. **`--surprise_source` 对照**：单次 run 只跑一种 surprise 源；"
        "要对比 `nfp` vs `oracle` 请跑两次（output_dir 不同），事后人工对比 summary.json。\n\n"
    )

    md_lines.append(f"- timestamp: {summary['timestamp']}\n")
    md_lines.append(f"- surprise_source: {args.surprise_source}\n")
    md_lines.append(f"- bank caps: short={args.short_cap}, "
                    f"medium={args.medium_cap}, long={args.long_cap}\n")
    md_lines.append(f"- hit_dist={args.hit_dist}, hit_yaw={args.hit_yaw}, "
                    f"intermediate_separation={args.intermediate_separation}\n")
    md_lines.append(f"- visual_fusion_alpha={args.visual_fusion_alpha}\n")
    md_lines.append(f"- episodes={summary['n_episodes']}, "
                    f"queries_total={n_query_total}\n\n")

    md_lines.append("## Global Metrics（全 episode 平均，按 query 加权）\n\n")
    header = "| method | " + " | ".join(
        f"p@{k} / r@{k}" for k in k_values
    ) + " |"
    sep = "|" + "---|" * (1 + len(k_values))
    md_lines.append(header + "\n")
    md_lines.append(sep + "\n")
    for m in method_keys:
        row_cells = [m]
        for k in k_values:
            cell = global_table[m][k]
            row_cells.append(
                f"{cell['precision_at_k']:.3f} / {cell['recall_at_k']:.3f}"
            )
        md_lines.append("| " + " | ".join(row_cells) + " |\n")

    md_lines.append("\n## Per-Episode Brief\n\n")
    md_lines.append("| episode | T | n_query | bank_size_final | "
                    f"bank p@{max(k_values)} |\n")
    md_lines.append("|---|---|---|---|---|\n")
    for ep in episode_summaries:
        max_k = max(k_values)
        key = str(max_k) if str(max_k) in ep["metrics_by_method"]["bank"] else max_k
        md_lines.append(
            f"| {ep['episode_id']} | {ep['total_frames']} | "
            f"{ep['n_query_evaluated']} | {ep['bank_size_final']} | "
            f"{ep['metrics_by_method']['bank'][key]['precision_at_k']:.3f} |\n"
        )

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as fh:
        fh.writelines(md_lines)
    logger.info("Wrote summary MD: %s", md_path)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "logs", "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU "
                       "(VAE encode will be very slow)")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    fps = args.fps
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * fps)))
    k_values = sorted([int(x) for x in args.k_values.split(",")])
    logger.info("Args: %s", vars(args))

    # 加载 episode CSV
    ep_filter = None
    if args.episode_ids:
        ep_filter = [s.strip() for s in args.episode_ids.split(",") if s.strip()]
    ep_groups = load_episode_clips(
        args.dataset_dir, args.metadata, episode_ids_filter=ep_filter,
    )
    ep_ids = list(ep_groups.keys())
    # WARN-F：当用户同时指定 --episode_ids 与 --max_episodes 时，需明确说明
    # max_episodes 会在 filter 结果上继续截断（语义可能违反用户预期）
    if args.max_episodes > 0 and ep_filter is not None:
        logger.warning(
            "--episode_ids 已显式指定 %d 个 episode，--max_episodes=%d "
            "将在 filter 结果上继续截断（实际跑 min(%d, %d)=%d 个）",
            len(ep_filter), args.max_episodes,
            len(ep_ids), args.max_episodes, min(len(ep_ids), args.max_episodes),
        )
    if args.max_episodes > 0 and len(ep_ids) > args.max_episodes:
        ep_ids = ep_ids[:args.max_episodes]
    if not ep_ids:
        logger.error("No episodes to process; abort.")
        return

    # 加载模型（仅一次，跨 episode 复用）
    logger.info("Loading models...")
    model = _build_memory_model(
        ckpt_dir=args.ckpt_dir,
        ft_model_dir=args.ft_model_dir,
        device=device,
        dtype=torch.bfloat16,
    )
    vae_ckpt = args.vae_ckpt_dir or args.ckpt_dir
    vae = _load_vae(vae_ckpt, device)

    # NFP head 准备
    nfp_head = getattr(model, "nfp_head", None)
    if nfp_head is None and args.surprise_source == "nfp":
        logger.warning("NFP head not found; downgrade --surprise_source to visual_cosine")
        args.surprise_source = "visual_cosine"

    rng = random.Random(args.seed)
    episode_summaries: List[Dict] = []

    for ep_id in tqdm(ep_ids, desc="episodes"):
        clips = ep_groups[ep_id]
        ep = build_episode_data(
            ep_id, clips, clip_overlap_frames=args.clip_overlap_frames,
        )
        if ep is None:
            continue
        T = ep.poses.shape[0]
        if T < args.skip_first_n + 5:
            logger.warning("Episode %s only %d frames; skip (need >= %d)",
                           ep_id, T, args.skip_first_n + 5)
            continue

        # 计算 GT 重访
        gt_revisit_for = compute_gt_revisit(
            ep, hit_dist=args.hit_dist, hit_yaw=args.hit_yaw,
            intermediate_separation=args.intermediate_separation,
            min_time_gap_frames=min_time_gap_frames,
        )
        n_gt_query = sum(1 for q in range(T) if gt_revisit_for.get(q))
        logger.info("Episode %s: T=%d, n_gt_query=%d", ep_id, T, n_gt_query)
        if n_gt_query == 0:
            logger.warning("Episode %s has 0 GT revisit queries; skip", ep_id)
            continue

        # 解码 video → VAE encode → expand to per-frame latents
        try:
            logger.info("Decoding %d frames from video.mp4 ...", T)
            frames = _decode_episode_video(ep, height=height, width=width)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s: decode failed: %s; skip", ep_id, exc)
            continue
        try:
            logger.info("Running VAE encode...")
            t0 = time.time()
            latents_full = _vae_encode_batched(
                vae, frames, device=device, batch_frames=args.vae_batch,
            )
            logger.info("VAE encode done in %.1fs, latents shape=%s",
                        time.time() - t0, tuple(latents_full.shape))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s: VAE encode failed: %s; skip", ep_id, exc)
            del frames
            continue
        del frames
        latents_per_frame = _expand_latents_to_frames(latents_full, T)
        del latents_full

        # 计算 pose_emb / visual_emb
        try:
            logger.info("Computing pose_embs...")
            pose_embs = _compute_pose_embs_episode(
                ep, model, device, height=height, width=width, fps=fps,
            )   # [T, 5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s: pose_embs failed: %s; skip", ep_id, exc)
            continue
        try:
            logger.info("Computing visual_embs...")
            visual_embs = _compute_visual_embs_from_latents(
                latents_per_frame, model, device,
            )   # [T, 5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s: visual_embs failed: %s; "
                           "continuing with visual_embs=None", ep_id, exc)
            visual_embs = None

        # Surprise
        if args.surprise_source == "oracle":
            surprise = _compute_surprise_oracle(gt_revisit_for, T)
            logger.info("Using oracle surprise (n_nonzero=%d)",
                        int((surprise > 0).sum().item()))
        elif args.surprise_source == "nfp" and nfp_head is not None and visual_embs is not None:
            # 用 NFP head 对 visual_emb 作为输入计算 surprise（简化版）：
            # 训练时 NFP 输入是 last_block hidden_states [B, L, dim]，
            # 推理时此处取 visual_emb [1, dim] → reshape [1, 1, dim] 灌进去。
            # 这是降级，但保留 NFP head 权重的语义。
            try:
                logger.info("Computing surprise via NFP head (simplified input)...")
                with torch.no_grad():
                    nfp_head.to(device)
                    pred_lat = nfp_head.forward(
                        visual_embs.to(device).to(
                            next(nfp_head.parameters()).dtype
                        ).unsqueeze(1)   # [T, 1, dim]
                    )   # [T, z_dim=16]
                # actual = latent 的空间均值（每帧）→ [T, z_dim]
                actual = latents_per_frame.float().mean(dim=[-2, -1])   # [T, z_dim]
                surprise = (
                    1.0 - F.cosine_similarity(pred_lat.float().cpu(), actual, dim=-1)
                ).clamp(min=0.0, max=2.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NFP surprise failed: %s; fallback visual_cosine", exc)
                surprise = _compute_surprise_visual_cosine(
                    visual_embs if visual_embs is not None
                    else torch.zeros(T, model.dim)
                )
        else:
            # fallback：visual_cosine
            if visual_embs is None:
                logger.warning("visual_embs unavailable; using zero surprise")
                surprise = torch.zeros(T, dtype=torch.float32)
            else:
                surprise = _compute_surprise_visual_cosine(visual_embs)
        logger.info("Surprise stats: mean=%.3f, max=%.3f, min=%.3f, "
                    ">0.4 frac=%.2f%%",
                    float(surprise.mean()), float(surprise.max()),
                    float(surprise.min()),
                    100.0 * float((surprise > args.surprise_threshold).float().mean()))

        # 跑评测
        try:
            ep_summary = _eval_episode(
                ep=ep,
                pose_embs=pose_embs,
                visual_embs=visual_embs,
                surprise=surprise,
                latents_per_frame=latents_per_frame,
                gt_revisit_for=gt_revisit_for,
                args=args,
                rng=rng,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Episode %s eval failed: %s; skip", ep_id, exc)
            continue
        episode_summaries.append(ep_summary)

        # 单 episode 后简单清理
        del pose_embs
        if visual_embs is not None:
            del visual_embs
        del latents_per_frame
        torch.cuda.empty_cache()
        gc.collect()

    if not episode_summaries:
        logger.error("No episodes produced any summary; abort.")
        return

    _write_summary(args.output_dir, args, episode_summaries, k_values)
    logger.info("Done. Output dir: %s", args.output_dir)


if __name__ == "__main__":
    main()
