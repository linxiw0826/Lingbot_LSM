"""oracle_injection.py — Exp2 Oracle 记忆注入测试（v4 ThreeTierMemoryBank / 冻结 DiT）

目的（D-06 / experiment_design.md Exp 2）
-----------------------------------------
剥离检索，单独测「冻结 DiT 能否利用注入的记忆」。在重访 target clip 处直接注入
GT 历史帧（oracle memory）——绕过 bank 检索/NFP surprise/存储逻辑，直接构造注入用
的 memory K/V——并弱化首帧 image condition（逼模型靠记忆），对照 oracle / off /
wrong-memory，看生成的重访帧是否更一致。

⚠️ 重要前提（结果有效性依赖）
----------------------------
本脚本的有意义结果**依赖 memory_cross_attn 已训练**（OP-1 / Bug1 已修 + 重训）。
在 epoch_4（memory_cross_attn 随机初始化，见 OP-1 / F-12：use_reentrant=True 梯度
检查点 + Stage1 冻结 backbone 导致 memory_cross_attn 全程零梯度）上跑，只能作为
**负对照基线**——此时 gate=0.1 固定 + 投影随机，注入任何 K/V 都只是加结构化噪声，
oracle / off / wrong 的差别**不能**用来肯定或否定 idea。
主结果须用 **Exp0 重训后的 checkpoint**（memory 模块真正训练过）跑。

三档 Tier 对照（D-06）
---------------------
| --tier_config | 含义 | 测什么 |
|---------------|------|-------|
| full         | 弱化首帧 + Short/Medium/Long 全保留（+ oracle 帧注入） | Memory Bank 整体能不能用（最宽松） |
| medium_long  | 弱化首帧 + 关 Short（Medium+Long 保留，+ oracle 帧注入） | Medium+Long 能不能用（idea 主打） |
| oracle_only  | 弱化首帧 + 三层全关，**只注入 oracle GT 帧** | 注入本身有没有被用到（最严格） |

注入内容对照（--memory_mode）
-----------------------------
- oracle : 注入该重访点首次访问时的 GT 历史帧（正确记忆）
- off    : 不注入记忆（memory_states=None，baseline）
- wrong  : 注入同 episode 远距离位置的 GT 帧（错误场景）——检验"是不是任何注入都改变输出"

首帧弱化（--weaken_first_frame，D-06）
-------------------------------------
砍掉首帧 image condition 这条强信号通道，制造"模型不靠记忆就答不出"的清洁实验条件。
- noise : 随机噪声替代首帧（默认，相对温和）
- zero  : 首帧置零（中性灰）
- none  : 不弱化（首帧条件保留）
首帧弱化在 WanI2V.generate() 的 `img` 入口处替换 PIL 图像实现——不修改 image2video.py。

依赖与约束
----------
- 复用 infer_v4.py 的 load/convert/patch pipeline（import，不重写）
- 复用 retrieval_probe.py 的 episode 加载 + 绝对位置 GT 重访判定（import）
- 不修改 infer_v4.py / train_v4 / memory_bank.py / model_with_memory.py
- 代码风格与 infer_v4 / retrieval_probe 一致（中文注释 / type hints / docstring）
- 单 GPU 即可运行（不强制多卡；本脚本不支持 Ulysses SP，逻辑更清晰）

跨模块数据契约（oracle 帧注入）
-------------------------------
- memory_states (K)       : [1, K, 5120]，oracle GT 帧的 pose_emb（get_projected_frame_embs）
- memory_value_states (V) : [1, K, 5120]，oracle GT 帧的 visual_emb（get_projected_latent_emb）
- tier_ids                : [K] int64（oracle_only 模式标 long=2）或 None
  消费方：_patch_pipeline_memory → MemoryCrossAttention.forward（与 bank 检索路径同接口）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# sys.path 设置（与 infer_v4.py / retrieval_probe.py 一致）
# ---------------------------------------------------------------------------

_PIPELINE_DIR = dirname(abspath(__file__))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                   # → src/
_PROJECT_ROOT = dirname(_SRC_DIR)                   # → Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, "refs", "lingbot-world")

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
            "Exp2 Oracle 记忆注入测试 — 冻结 DiT，注入 GT 历史帧 + 弱化首帧，"
            "对照 oracle/off/wrong × 三档 tier_config。"
        )
    )
    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_full_train.csv")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录（视频 + summary.md）")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")

    # ---- 模型权重（与 infer_v4 一致）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="基础模型目录（lingbot-world checkpoint，同 infer_v4）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="重训后的 v4 low_noise_model checkpoint 目录"
                        "（memory 模块训练好的）；缺失则用 base 权重（memory 随机，仅负对照）")
    p.add_argument("--ft_high_model_dir", type=str, default=None,
                   help="dual 训练 high_noise_model 目录（可选）")

    # ---- 三档 Tier 对照（D-06）----
    p.add_argument("--tier_config", type=str, default="oracle_only",
                   choices=["full", "medium_long", "oracle_only"],
                   help="full=Short/Medium/Long 全保留；medium_long=关 Short；"
                        "oracle_only=三层全关只注入 oracle GT 帧（默认）")
    # ---- 注入内容对照 ----
    p.add_argument("--memory_mode", type=str, default="oracle",
                   choices=["oracle", "off", "wrong"],
                   help="oracle=注入 GT 历史帧 / off=不注入 / wrong=注入错误场景帧")
    # ---- 首帧弱化（D-06）----
    p.add_argument("--weaken_first_frame", type=str, default="noise",
                   choices=["noise", "zero", "none"],
                   help="noise=随机噪声替首帧（默认）/ zero=置零 / none=不弱化")

    # ---- 重访点判定（复用 retrieval_probe 口径）----
    p.add_argument("--hit_dist", type=float, default=40.0,
                   help="GT 重访距离阈值（数据集原生单位；v4/CSGO ≈ inches，1m≈40）")
    p.add_argument("--hit_yaw", type=float, default=30.0,
                   help="GT 重访 |yaw 差| 阈值（度）")
    p.add_argument("--intermediate_separation", type=float, default=100.0,
                   help="中间分离阈值（过滤 stationary 假位置重访；<=0 跳过）")
    p.add_argument("--min_time_gap_sec", type=float, default=5.0,
                   help="GT 重访最小时间差（秒），默认 5.0")
    p.add_argument("--clip_overlap_frames", type=int, default=0,
                   help="相邻 clip overlap 帧数；v4 数据 0.5s overlap 应设 8")
    p.add_argument("--max_revisit_points", type=int, default=2,
                   help="每 episode 最多取多少个重访点（控制生成耗时）")
    p.add_argument("--num_oracle_frames", type=int, default=2,
                   help="每个重访点注入多少 oracle/wrong GT 帧（K）")

    # ---- 生成参数 ----
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=70,
                   help="diffusion 采样步数（infer_v4 中名为 sample_steps）")
    p.add_argument("--sample_shift", type=float, default=10.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（min_time_gap_sec→帧数换算 + 保存）")

    # ---- Bank 超参数（full / medium_long 模式构建 bank 时使用，与 v4 默认对齐）----
    p.add_argument("--medium_cap", type=int, default=8)
    p.add_argument("--long_cap", type=int, default=32)
    p.add_argument("--surprise_threshold", type=float, default=0.4)
    p.add_argument("--stability_threshold", type=float, default=0.2)
    p.add_argument("--novelty_threshold", type=float, default=0.7)
    p.add_argument("--half_life", type=float, default=10.0)
    p.add_argument("--dup_threshold", type=float, default=0.95)
    p.add_argument("--visual_fusion_alpha", type=float, default=0.7)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 重访点数据结构
# ---------------------------------------------------------------------------

@dataclass
class RevisitPoint:
    """一个重访点：query 帧 q 重新到达了首访帧 first_visit_frame 所在的地点。"""
    episode_id: str
    query_frame: int                 # 全局帧索引（episode 拼接后）
    first_visit_frame: int           # 该地点首次访问的 GT 历史帧（全局索引）
    gt_past_frames: List[int]        # 全部命中的过去帧（用于诊断）


# ---------------------------------------------------------------------------
# 复用 retrieval_probe 的 episode 加载 + GT 重访判定（import，不重写）
# ---------------------------------------------------------------------------

from pipeline.retrieval_probe import (  # noqa: E402
    EpisodeData,
    load_episode_clips,
    build_episode_data,
    compute_gt_revisit,
    _decode_episode_video,
    _vae_encode_batched,
    _expand_latents_to_frames,
)

# 复用 infer_v4 的 pipeline 装载/转换/注入（import，不重写）
from pipeline.infer_v4 import (  # noqa: E402
    _load_ft_model_and_prepare_ckpt,
    _convert_pipeline_to_memory,
    _patch_pipeline_memory,
    _unpatch_pipeline_memory,
)


def _find_revisit_points(
    ep: EpisodeData,
    args,
    min_time_gap_frames: int,
) -> List[RevisitPoint]:
    """对 episode 找重访点：复用 retrieval_probe.compute_gt_revisit 的绝对位置 +
    yaw + time_gap + 中间分离判定。

    对每个 query 帧 q（有 GT 过去帧），取最早的过去帧作为 first_visit_frame
    （"该地点首次访问"语义）。按 query_frame 升序，取前 max_revisit_points 个。
    """
    gt = compute_gt_revisit(
        ep,
        hit_dist=args.hit_dist,
        hit_yaw=args.hit_yaw,
        intermediate_separation=args.intermediate_separation,
        min_time_gap_frames=min_time_gap_frames,
    )
    points: List[RevisitPoint] = []
    for q in sorted(gt.keys()):
        past = sorted(gt[q])
        if not past:
            continue
        points.append(
            RevisitPoint(
                episode_id=ep.episode_id,
                query_frame=int(q),
                first_visit_frame=int(past[0]),   # 最早访问 = 首访
                gt_past_frames=[int(x) for x in past],
            )
        )
    if args.max_revisit_points > 0:
        points = points[:args.max_revisit_points]
    return points


# ---------------------------------------------------------------------------
# 帧索引 → clip 切片（用于取 query clip 的 pose/action/intrinsics 与首帧图像）
# ---------------------------------------------------------------------------

def _frame_to_clip_slice(
    ep: EpisodeData,
    center_frame: int,
    frame_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """从 episode 拼接后的全序列里，截取以 center_frame 起始的 frame_num 帧
    pose/action/intrinsics（不足时回退到末尾对齐）。

    Returns:
        (poses[frame_num,4,4], actions[frame_num,4], intrinsics[frame_num,4], start)
    """
    T = ep.poses.shape[0]
    start = center_frame
    end = start + frame_num
    if end > T:
        start = max(0, T - frame_num)
        end = start + frame_num
    poses = ep.poses[start:end]
    actions = ep.actions[start:end]
    intr = ep.intrinsics[start:end]
    # 不足 frame_num 时用末帧 pad（与 infer_v4 fallback 思路一致）
    if poses.shape[0] < frame_num:
        pad_n = frame_num - poses.shape[0]
        poses = np.concatenate([poses, np.tile(poses[-1:], (pad_n, 1, 1))], axis=0)
        actions = np.concatenate([actions, np.tile(actions[-1:], (pad_n, 1))], axis=0)
        intr = np.concatenate([intr, np.tile(intr[-1:], (pad_n, 1))], axis=0)
    return poses, actions, intr, start


def _weaken_image(img: Image.Image, mode: str, rng: np.random.Generator) -> Image.Image:
    """按 mode 弱化首帧 image condition（砍掉 I2V 强信号通道，D-06）。

    在 PIL 入口替换图像，避免修改 image2video.py。

    Args:
        img:  原始首帧 PIL 图像
        mode: noise（随机噪声）/ zero（中性灰）/ none（不弱化）
        rng:  numpy Generator（保证可复现）

    Returns:
        弱化后的 PIL 图像（mode=none 时原样返回）
    """
    if mode == "none":
        return img
    w, h = img.size
    if mode == "noise":
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if mode == "zero":
        # 置零 = [-1,1] 空间的 0 → 像素 127.5（中性灰），避免纯黑被 VAE 当强信号
        arr = np.full((h, w, 3), 128, dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    return img


# ---------------------------------------------------------------------------
# Oracle 帧 → memory K/V 构造（绕过 bank 检索 / NFP surprise / 存储逻辑）
# ---------------------------------------------------------------------------

def _build_oracle_memory_kv(
    model,
    pipeline,
    ep: EpisodeData,
    latents_per_frame: torch.Tensor,    # [T, z_dim, lat_h, lat_w]，CPU
    frame_indices: List[int],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """直接构造注入用 memory K/V（最干净做法，task 注意点 2）。

    对每个 GT 帧 i：
      K (pose_emb) ：用该帧所在 clip 的 pose 经 get_projected_frame_embs 取对应帧
      V (visual_emb)：用该帧的 VAE latent 经 get_projected_latent_emb

    这条路径绕过 bank.retrieve / NFP surprise / 三层存储，把 GT 历史帧直接 VAE
    encode 后作为 memory_states 传给 memory_cross_attn（与 bank 检索路径同 shape 契约）。

    Args:
        model:             WanModelWithMemory（pipeline.low_noise_model）
        pipeline:          WanI2V
        ep:                EpisodeData（提供 poses/actions/intrinsics）
        latents_per_frame: episode 全帧 VAE latent（用于 V）
        frame_indices:     要注入的 GT 帧全局索引列表
        device:            目标设备

    Returns:
        (key_states [K,5120], value_states [K,5120])，CPU；无有效帧时 None
    """
    from memory_module.model_with_memory import WanModelWithMemory
    from pipeline.dataloader import build_dit_cond_dict

    if not isinstance(model, WanModelWithMemory):
        logger.warning("model 非 WanModelWithMemory，无法构造 oracle K/V")
        return None
    if not frame_indices:
        return None

    T = ep.poses.shape[0]
    _h, _w = (int(x) for x in _SIZE_HW)
    vae_stride_t = 4

    # 确保相关投影层在 device（offload 安全）
    for _attr in ("patch_embedding_wancamctrl", "c2ws_hidden_states_layer1",
                  "c2ws_hidden_states_layer2"):
        if hasattr(model, _attr):
            getattr(model, _attr).to(device)
    if hasattr(model, "latent_proj"):
        model.latent_proj.to(device)

    key_list: List[torch.Tensor] = []
    val_list: List[torch.Tensor] = []

    for fi in frame_indices:
        if fi < 0 or fi >= T:
            continue
        # --- K：该帧的 pose_emb ---
        # 取以 fi 为中心的 clip pose 段，算 plucker → projected frame embs，
        # 取段内对应该帧的 latent index。
        poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(ep, fi, _ORACLE_CLIP_FRAMES)
        try:
            cond = build_dit_cond_dict(
                poses=torch.from_numpy(poses_c).float(),
                actions=torch.from_numpy(acts_c).float(),
                intrinsics=torch.from_numpy(intr_c).float(),
                height=_h, width=_w,
            )
            plucker = cond["c2ws_plucker_emb"][0].to(device)  # [1,448,lat_f,lat_h,lat_w]
            with torch.no_grad():
                frame_embs = model.get_projected_frame_embs(plucker)  # [lat_f,5120]
            local = fi - seg_start
            lat_idx = min(max(local // vae_stride_t, 0), frame_embs.shape[0] - 1)
            pose_emb = frame_embs[lat_idx].float().cpu()  # [5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("oracle K 计算失败 fi=%d: %s；跳过该帧", fi, exc)
            continue

        # --- V：该帧的 visual_emb ---
        try:
            lat = latents_per_frame[fi].to(device)  # [z_dim,lat_h,lat_w]
            with torch.no_grad():
                visual_emb = model.get_projected_latent_emb(lat).float().cpu()  # [5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("oracle V 计算失败 fi=%d: %s；跳过该帧", fi, exc)
            continue

        key_list.append(pose_emb)
        val_list.append(visual_emb)

    if not key_list:
        return None
    key_states = torch.stack(key_list)   # [K,5120]
    value_states = torch.stack(val_list)  # [K,5120]
    return key_states, value_states


# 全局占位（main 中设置，供 _build_oracle_memory_kv 读取分辨率 / clip 长度）
_SIZE_HW: Tuple[str, str] = ("480", "832")
_ORACLE_CLIP_FRAMES: int = 81


# ---------------------------------------------------------------------------
# 轻量一致性指标（D-06：第一轮用最简单可跑的）
# ---------------------------------------------------------------------------

def _to_gray_uint8(frame_chw: np.ndarray) -> np.ndarray:
    """[3,H,W] in [-1,1] → 灰度 [H,W] uint8。"""
    hwc = ((frame_chw.transpose(1, 2, 0) * 127.5 + 127.5)
           .clip(0, 255).astype(np.uint8))
    # RGB → 灰度（Rec.601）
    gray = (0.299 * hwc[..., 0] + 0.587 * hwc[..., 1]
            + 0.114 * hwc[..., 2])
    return gray.astype(np.float32)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """全局单窗口 SSIM（灰度，[H,W] float32）。

    简单可跑实现（D-06 已定：第一轮指标用现有/最简可跑指标，不追求完美；
    若定性模糊再补 Pose-Paired SSIM / DINO cosine / LPIPS。此处选 SSIM 是其中最简实现）。
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den > 0 else 0.0


def _revisit_consistency(
    gen_video: np.ndarray,        # [3, F, H, W] in [-1,1]（生成的 query clip）
    gt_first_visit_frame: np.ndarray,  # [3, H, W] in [-1,1]（首访 GT 帧）
) -> Dict[str, float]:
    """计算生成的重访帧 vs 首访 GT 帧的相似度。

    revisit_consistency = max_t SSIM(gen_video[:,t], gt_first_visit_frame)
      —— 取整段 query clip 里与首访帧最相似的一帧（重访发生的精确帧不确定，取 max）。
    同时返回 mean / last 供诊断。
    """
    F_ = gen_video.shape[1]
    gt_gray = _to_gray_uint8(gt_first_visit_frame)
    ssims = []
    for t in range(F_):
        ssims.append(_ssim(_to_gray_uint8(gen_video[:, t]), gt_gray))
    ssims_np = np.asarray(ssims, dtype=np.float32)
    return {
        "revisit_consistency_max": float(ssims_np.max()) if F_ > 0 else 0.0,
        "revisit_consistency_mean": float(ssims_np.mean()) if F_ > 0 else 0.0,
        "revisit_consistency_last": float(ssims_np[-1]) if F_ > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 单个重访点的一次生成（给定 tier_config × memory_mode）
# ---------------------------------------------------------------------------

def _generate_for_point(
    wan_i2v,
    bank_or_none,
    oracle_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    wrong_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    point: RevisitPoint,
    ep: EpisodeData,
    base_img: Image.Image,
    args,
    device: torch.device,
    rng: np.random.Generator,
    tmp_action_dir: str,
) -> Optional[np.ndarray]:
    """对一个重访点跑一次 diffusion 生成，返回生成视频 [3,F,H,W]。

    流程：
      a. 弱化首帧条件（按 --weaken_first_frame）
      b. 配置注入 K/V：
         - memory_mode=off：memory_states=None（不注入）
         - memory_mode=oracle：注入 oracle_kv
         - memory_mode=wrong：注入 wrong_kv
         - tier_config=full/medium_long：bank 路径预留（REVIEW-NOTE，见下）
      c. 跑生成（复用 infer_v4 的 _patch_pipeline_memory + wan_i2v.generate）
    """
    from memory_module.model_with_memory import WanModelWithMemory
    from wan.configs import MAX_AREA_CONFIGS

    # --- 首帧图像：base_img = query clip 首帧 GT 图（外层从解码帧构造并传入），按需弱化 ---
    # query clip 起始帧 = 以 query_frame 对齐的 clip 段起点
    poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(
        ep, point.query_frame, args.frame_num
    )
    img = _weaken_image(base_img, args.weaken_first_frame, rng)

    # --- 写 query clip 的 action 切片到临时目录（供 generate action_path 读取）---
    np.save(os.path.join(tmp_action_dir, "poses.npy"), poses_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "action.npy"), acts_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "intrinsics.npy"), intr_c.astype(np.float32))

    # --- 选择注入 K/V ---
    memory_states = None
    memory_value_states = None
    tier_ids = None

    if args.memory_mode == "off":
        memory_states = None
    else:
        kv = oracle_kv if args.memory_mode == "oracle" else wrong_kv
        if kv is not None:
            key_states, value_states = kv
            memory_states = key_states.unsqueeze(0).to(device)         # [1,K,5120]
            memory_value_states = value_states.unsqueeze(0).to(device)  # [1,K,5120]
            # tier_ids：oracle_only 标 long(2)；full/medium_long 也标 long（oracle 帧语义=长期重访记忆）
            tier_ids = torch.full(
                (key_states.shape[0],), 2, dtype=torch.long, device=device
            )

    # REVIEW-NOTE（无对应 OPEN 决策；待 ReviewAgent / 用户拍板，见汇报「需决策点」）：
    # full / medium_long 模式下，"oracle GT 帧" 与 "bank 三层检索帧" 如何混合注入尚未定。
    # 当前实现（第一版）：oracle 模式只注入 oracle GT 帧，不叠加 bank 检索帧；
    # tier_config 仅通过 _build_bank_for_config 控制启用哪些层、但本版未把 bank 检索结果
    # 与 oracle K/V 合并 → 三档当前差异主要体现在 tier_ids 语义与 bank 是否构建上。
    # 候选方案：full=oracle帧 + Short/Medium/Long 检索帧并集；
    #          medium_long=oracle帧 + Medium/Long 检索帧；oracle_only=仅 oracle 帧（已实现）。

    max_area = MAX_AREA_CONFIGS[args.size]

    if memory_states is not None:
        _patch_pipeline_memory(wan_i2v, memory_states, memory_value_states, tier_ids=tier_ids)
    try:
        video = wan_i2v.generate(
            args.prompt,
            img,
            action_path=tmp_action_dir,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.num_inference_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            offload_model=True,
        )
    finally:
        if memory_states is not None:
            _unpatch_pipeline_memory(wan_i2v)

    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    return video  # [3,F,H,W]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    global _SIZE_HW, _ORACLE_CLIP_FRAMES
    _SIZE_HW = tuple(args.size.split("*"))  # ("480","832")
    _ORACLE_CLIP_FRAMES = args.frame_num

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))

    logger.info("Args: %s", vars(args))
    logger.warning(
        "⚠️ Exp2 有效性前提：memory_cross_attn 须已训练（OP-1 修复 + 重训后的 "
        "--ft_model_dir）。epoch_4 随机 memory 上结果仅作负对照。"
    )

    # ---- 加载 episode CSV + 过滤 ----
    ep_filter = None
    if args.episode_ids:
        ep_filter = [s.strip() for s in args.episode_ids.split(",") if s.strip()]
    ep_groups = load_episode_clips(args.dataset_dir, args.metadata,
                                   episode_ids_filter=ep_filter)
    ep_ids = list(ep_groups.keys())
    if args.max_episodes > 0 and len(ep_ids) > args.max_episodes:
        ep_ids = ep_ids[:args.max_episodes]
    if not ep_ids:
        logger.error("无 episode 可处理，退出。")
        return

    # ---- 加载 WanI2V pipeline（复用 infer_v4，转换为 WanModelWithMemory）----
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS

    # 全参微调模式：准备 tmp ckpt_dir（base 权重，ft 权重由 _convert 注入）
    ckpt_dir = args.ckpt_dir
    if args.ft_model_dir:
        _fake = argparse.Namespace(ckpt_dir=args.ckpt_dir)
        ckpt_dir = _load_ft_model_and_prepare_ckpt(_fake)

    cfg = WAN_CONFIGS["i2v-A14B"]
    local_rank = device.index if device.type == "cuda" and device.index is not None else 0
    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
    )
    logger.info("转换 pipeline → WanModelWithMemory ...")
    wan_i2v = _convert_pipeline_to_memory(
        wan_i2v,
        memory_ckpt_path=None,
        high_model_dir=args.ft_high_model_dir,
        low_model_dir=args.ft_model_dir,
    )
    model = wan_i2v.low_noise_model

    # ---- 逐 episode → 找重访点 → 三注入对照生成 ----
    import tempfile

    all_records: List[Dict] = []

    for ep_id in ep_ids:
        clips = ep_groups[ep_id]
        ep = build_episode_data(ep_id, clips,
                                clip_overlap_frames=args.clip_overlap_frames)
        if ep is None:
            continue
        T = ep.poses.shape[0]
        points = _find_revisit_points(ep, args, min_time_gap_frames)
        if not points:
            logger.warning("Episode %s 无重访点；跳过", ep_id)
            continue
        logger.info("Episode %s: T=%d, 重访点 %d 个", ep_id, T, len(points))

        # 解码 video + VAE encode（oracle V + 首帧图像 + GT 一致性参照都要用）
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W] in [-1,1]
            latents_full = _vae_encode_batched(wan_i2v.vae, frames, device=device,
                                               batch_frames=8)
            latents_per_frame = _expand_latents_to_frames(latents_full, T)
            del latents_full
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
            continue

        ep_out_dir = os.path.join(args.output_dir, ep_id)
        os.makedirs(ep_out_dir, exist_ok=True)

        for pt in points:
            # 首访 GT 帧（一致性参照 + 人工定性对比）
            gt_first = frames[pt.first_visit_frame]  # [3,H,W]
            # 保存首访 GT 帧供人工对比
            _save_frame_png(gt_first,
                            os.path.join(ep_out_dir,
                                         f"q{pt.query_frame}_gt_first_visit.png"))

            # query clip 首帧 GT 图像（弱化前的 base，作为 generate 的 img 入口）
            _pc, _ac, _ic, seg_start = _frame_to_clip_slice(
                ep, pt.query_frame, args.frame_num)
            query_first_np = frames[seg_start]  # [3,H,W]
            base_img = _frame_to_pil(query_first_np)

            # 构造 oracle K/V（绕过 bank）：注入首访点附近的 GT 历史帧
            oracle_indices = _pick_oracle_indices(pt, args.num_oracle_frames, T)
            oracle_kv = _build_oracle_memory_kv(
                model, wan_i2v, ep, latents_per_frame, oracle_indices, device)
            # wrong K/V：同 episode 远距离位置的 GT 帧
            wrong_indices = _pick_wrong_indices(pt, args.num_oracle_frames, T, rng)
            wrong_kv = _build_oracle_memory_kv(
                model, wan_i2v, ep, latents_per_frame, wrong_indices, device)

            # full / medium_long 模式下构建 bank（预留；REVIEW-NOTE 见 _generate_for_point）
            bank = _build_bank_for_config(args) if args.tier_config != "oracle_only" else None

            _tmp_action = tempfile.mkdtemp(prefix=f"oracle_inj_{ep_id}_q{pt.query_frame}_")
            try:
                video = _generate_for_point(
                    wan_i2v, bank, oracle_kv, wrong_kv, pt, ep, base_img, args,
                    device, rng, _tmp_action)
            except Exception as exc:  # noqa: BLE001
                logger.exception("生成失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                video = None
            finally:
                import shutil
                shutil.rmtree(_tmp_action, ignore_errors=True)

            if video is None:
                continue

            # 保存生成视频
            mp4_name = f"q{pt.query_frame}_{args.tier_config}_{args.memory_mode}.mp4"
            mp4_path = os.path.join(ep_out_dir, mp4_name)
            _save_video(video, mp4_path, fps=args.fps)

            # 一致性指标
            metrics = _revisit_consistency(video, gt_first)
            logger.info("ep=%s q=%d [%s/%s] %s",
                        ep_id, pt.query_frame, args.tier_config,
                        args.memory_mode, metrics)

            all_records.append({
                "episode_id": ep_id,
                "query_frame": pt.query_frame,
                "first_visit_frame": pt.first_visit_frame,
                "tier_config": args.tier_config,
                "memory_mode": args.memory_mode,
                "weaken_first_frame": args.weaken_first_frame,
                "n_oracle_frames": len(oracle_indices),
                "video_path": mp4_path,
                "gt_first_visit_png": os.path.join(
                    ep_out_dir, f"q{pt.query_frame}_gt_first_visit.png"),
                **metrics,
            })

        del frames, latents_per_frame
        torch.cuda.empty_cache()

    _write_summary(args, all_records)
    logger.info("Done. 输出目录: %s", args.output_dir)


# ---------------------------------------------------------------------------
# oracle / wrong 帧选择
# ---------------------------------------------------------------------------

def _pick_oracle_indices(pt: RevisitPoint, n: int, T: int) -> List[int]:
    """从 GT 过去帧里取 n 个作为 oracle 注入帧（优先首访帧及其邻近）。"""
    cand = sorted(set(pt.gt_past_frames))
    if not cand:
        return []
    # 取最早 n 个（首访点附近，最贴"该地点首次访问的真实历史帧"语义）
    return cand[:max(1, n)]


def _pick_wrong_indices(pt: RevisitPoint, n: int, T: int,
                        rng: np.random.Generator) -> List[int]:
    """取同 episode 远离 query / 首访点的随机帧作为 wrong-memory 注入帧。"""
    forbidden = set(pt.gt_past_frames) | {pt.query_frame}
    # 远离 query_frame：要求 |i - query_frame| > T//4，避免误命中真实重访
    margin = max(1, T // 4)
    pool = [i for i in range(T)
            if i not in forbidden and abs(i - pt.query_frame) > margin
            and abs(i - pt.first_visit_frame) > margin]
    if not pool:
        pool = [i for i in range(T) if i not in forbidden]
    if not pool:
        return []
    rng.shuffle(pool)
    return sorted(pool[:max(1, n)])


def _build_bank_for_config(args):
    """按 tier_config 构建 ThreeTierMemoryBank（关掉的层 cap=0）。

    - full        : Short/Medium/Long 全保留
    - medium_long : Short cap=0（关 Short），Medium/Long 保留
    （oracle_only 不构建 bank，main 中传 None）

    注：cap=0 时该层 update 不写入、retrieve 返回空，等价"关层"。
    """
    from memory_module.memory_bank import ThreeTierMemoryBank
    short_cap = 1 if args.tier_config == "full" else 0
    return ThreeTierMemoryBank(
        short_cap=short_cap,
        medium_cap=args.medium_cap,
        long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life,
        dup_threshold=args.dup_threshold,
    )


# ---------------------------------------------------------------------------
# 图像 / 视频 IO
# ---------------------------------------------------------------------------

def _frame_to_pil(frame_chw: np.ndarray) -> Image.Image:
    """[3,H,W] in [-1,1] → PIL RGB。"""
    hwc = ((frame_chw.transpose(1, 2, 0) * 127.5 + 127.5)
           .clip(0, 255).astype(np.uint8))
    return Image.fromarray(hwc, mode="RGB")


def _save_frame_png(frame_chw: np.ndarray, path: str) -> None:
    _frame_to_pil(frame_chw).save(path)


def _save_video(video: np.ndarray, path: str, fps: int) -> None:
    """保存 [3,F,H,W] in [-1,1] 视频，复用 wan.utils.save_video。"""
    from wan.utils.utils import save_video
    t = torch.from_numpy(video) if isinstance(video, np.ndarray) else video
    save_video(
        tensor=t[None],   # [1,3,F,H,W]
        save_file=path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info("保存视频 → %s", path)


# ---------------------------------------------------------------------------
# Summary 输出
# ---------------------------------------------------------------------------

def _write_summary(args, records: List[Dict]) -> None:
    """输出 summary.md（一致性数值表 + 视频路径）+ summary.json。"""
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "records": records,
        }, fh, indent=2)

    md = []
    md.append("# Exp2 Oracle Injection Summary\n\n")
    md.append(f"- timestamp: {datetime.now().isoformat()}\n")
    md.append(f"- tier_config: **{args.tier_config}** | "
              f"memory_mode: **{args.memory_mode}** | "
              f"weaken_first_frame: **{args.weaken_first_frame}**\n")
    md.append(f"- ft_model_dir: {args.ft_model_dir}\n\n")
    md.append("> ⚠️ 有效性前提：memory_cross_attn 须已训练（OP-1 修复 + 重训）。"
              "epoch_4 随机 memory 上仅作负对照。\n\n")
    md.append("> 单次 run 跑一种 (tier_config × memory_mode)。完整对照需多次 run "
              "（不同 --tier_config / --memory_mode），事后对比各 summary。\n\n")
    md.append("## 一致性数值（revisit_consistency = max_t SSIM(gen[:,t], 首访GT帧)）\n\n")
    md.append("| episode | query_frame | tier | mode | "
              "rc_max | rc_mean | rc_last | video |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for r in records:
        md.append(
            f"| {r['episode_id']} | {r['query_frame']} | {r['tier_config']} | "
            f"{r['memory_mode']} | {r['revisit_consistency_max']:.4f} | "
            f"{r['revisit_consistency_mean']:.4f} | "
            f"{r['revisit_consistency_last']:.4f} | "
            f"`{os.path.basename(r['video_path'])}` |\n"
        )
    md.append("\n## 人工定性对比\n\n")
    md.append("每个重访点已保存：\n")
    md.append("- `q<frame>_<tier>_<mode>.mp4`：生成的 query clip\n")
    md.append("- `q<frame>_gt_first_visit.png`：该地点首访 GT 帧（参照）\n\n")
    md.append("判读：oracle 是否优于 off（rc 更高）？wrong 是否变差（rc 更低 / 接近 off）？\n")

    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as fh:
        fh.writelines(md)
    logger.info("写 summary: %s", md_path)


if __name__ == "__main__":
    main()
