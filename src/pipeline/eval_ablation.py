"""eval_ablation.py — 「记忆开/关」loss-based ablation（目标 B「记忆有没有用」首道信号）

目的（experiment_design.md「快速验证实验设计」Exp2 / open_problems.md OP-1）
-------------------------------------------------------------------------------
Exp0 已活体复验 memory_cross_attn 在收梯度（[[F-15]]，gate_grad≠0）。本脚本回答下一问：
**「训出来的记忆到底有没有用」**。最快信号 = 在 revisit 评测窗口上，比较模型在
「记忆检索 ON」vs「记忆 OFF」两种条件下的前向 loss：
  若 memory_ON 的 diffusion / NFP loss **显著低于** memory_OFF → 记忆有用。

这是 loss-based ablation，比「生成 + VBench / 生成 + SSIM」便宜得多
（不跑 diffusion 采样、不解码视频），作为目标 B 的第一道信号。

⚠️ 与 oracle_injection.py 的分工（为何新建本文件而非复用）
--------------------------------------------------------
oracle_injection.py 做的是 **生成 + SSIM 一致性**（跑完整 `wan_i2v.generate()`
diffusion 采样，再对生成帧与首访 GT 帧算 SSIM），**不输出 diffusion / NFP loss**，
也没有「真实 bank 检索 ON vs OFF 的 loss 对照」。本任务要的是 **单步前向 loss 对照**
（forward-only，不采样），口径与 train_v4 训练 loss 一致 → oracle_injection 无法覆盖，
故新建本脚本（复用其同源的 episode/bank/检索 + train_v4 的 loss 计算前向）。

memory_ON vs memory_OFF 的语义（与 infer_v4 USE_MEMORY 对齐）
-----------------------------------------------------------
- **memory_ON**：对 target clip query 走已验证有效的 `retrieve_revisit` 路径
  （[[F-14]] bank_revisit p@1=0.542，绝对位置 L2 + 排近邻 + 只走 Long tier），
  把检索帧构造成 memory K(pose_emb)/V(visual_emb) 传给 model.forward(memory_states=...)。
- **memory_OFF**：`memory_states=None` 调 model.forward。
  与 infer_v4.py `--use_memory` 未设置时**完全同源**：infer_v4 不设 --use_memory 时
  从不调 `_patch_pipeline_memory`、memory_states 恒为 None；而
  MemoryBlockWrapper.forward（model_with_memory.py:115）仅在 dit_cond_dict 含
  `_MEMORY_STATES_KEY` 时才执行 memory_cross_attn，否则整条 cross-attn 旁路，
  行为与原始 WanModel 一致。本脚本 OFF 档不构造 memory_states、不写 _TIER_IDS_KEY，
  即与该旁路语义逐字一致（最干净、与现有开关一致的禁用方式）。
  → 两档**唯一差异**就是「有没有注入检索到的记忆 K/V」，其余前向完全相同
    （同一 noisy_latent / 同一 timestep / 同一 control signal），保证 Δloss 干净归因于记忆。

数据 & 评测窗口（task 要求 3）
----------------------------
- 默认 verify val 集（dataset_dir + metadata_verify_val.csv）。
- 优先在 **is_revisit=1 且 gap∈[gap_lo, gap_hi]（默认 [2,6]）** 的窗口上评测
  （记忆最该起作用处）；CSV 若无 is_revisit/gap 列则回退到「全部含 GT 重访的窗口」
  （用 compute_gt_revisit 在 episode 内判定，与 retrieval_probe 同口径），并在 summary 标注降级。

输出（task 要求 4）
------------------
- per-window CSV：每窗口 ON / OFF 的 diffusion_loss / nfp_loss + Δ + Δ%。
- aggregate：ON vs OFF 均值 + Δ + Δ% + 一句话判据。
- summary.md（顶部「降级说明」块，仿 retrieval_probe）+ summary.json。

诚实标注的局限（task 要求 5，仿 retrieval_probe summary 降级块）
-------------------------------------------------------------
1. gate 当前约 0.1（才训少量 epoch，记忆贡献小，ON/OFF 差可能很弱）。
2. loss-based ≠ 生成质量（低 loss 不直接等于重访一致性变好，是 cheap proxy）。
3. 样本量小（verify val ~5 episode / ~64 窗口）。
4. **若 ON ≈ OFF 不能立即判「记忆无用」——必须先排除训练不足**（gate 没打开、
   检索帧少），多训几 epoch 再测（沿用 current_focus 阶段三十五/三十六关键陷阱，
   避免重蹈 F-12「以 non-functional memory 的结果否定 idea」）。

依赖与约束
----------
- 复用 retrieval_probe.py 的 ckpt/VAE/model 加载 + episode/VAE/emb/bank 口径（import）。
- 复用 train_v4_stage1_dual.py 的 LingBotMemoryTrainer（FlowMatchingSchedule /
  encode_text / prepare_y / prepare_control_signal / encode_video）+ NFPHead.compute_loss
  → loss 前向口径与训练**逐行对齐**，保证 ON/OFF 对照与训练 loss 同源。
- 不修改 train_v4 / infer_v4 / oracle_injection / memory_bank / model_with_memory 主路径。
- CLI 用 --ft_model_dir 指向 epoch 目录（与 infer_v4 / retrieval_probe 一致）；
  训练还在跑、checkpoint 尚未落盘——脚本可在 checkpoint 出现后直接跑。
- 单 GPU 即可（不强制多卡；本脚本不走 Ulysses SP）。
- 不实际运行（无 GPU + checkpoint 未出）；只产出可运行代码。

跨模块数据契约（memory_ON 注入 K/V，与 oracle_injection / infer_v4 同契约）
--------------------------------------------------------------------------
- memory_states (K)       : [1, K, dim=5120]，pose_emb（bank 检索帧 frame.pose_emb）
- memory_value_states (V) : [1, K, dim=5120]，visual_emb（latent_proj 重算或 frame.visual_emb）
- tier_ids                : [K] int64（retrieve_revisit 只走 Long → 全标 2）或 None
  消费方：WanModelWithMemory.forward(memory_states=, memory_value_states=) +
          dit_cond_dict[_TIER_IDS_KEY]（model_with_memory.py:115-121）
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
# 复用 retrieval_probe 的 episode 加载 + VAE/emb + GT 重访（import，不重写）
# ---------------------------------------------------------------------------

from pipeline.retrieval_probe import (  # noqa: E402
    EpisodeData,
    load_episode_clips,
    build_episode_data,
    compute_gt_revisit,
    _decode_episode_video,
    _vae_encode_batched,
    _expand_latents_to_frames,
    _compute_pose_embs_episode,
    _compute_visual_embs_from_latents,
    _compute_surprise_visual_cosine,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "「记忆开/关」loss-based ablation —— 在 revisit 窗口上对比 memory_ON "
            "(retrieve_revisit) vs memory_OFF (memory_states=None) 的 diffusion/nfp loss。"
        )
    )
    # ---- 数据（默认 verify val）----
    p.add_argument("--dataset_dir", type=str,
                   default="/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3",
                   help="数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, default="metadata_verify_val.csv",
                   help="相对 dataset_dir 的 CSV 路径（默认 verify val）")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录（per_window.csv + summary.md/json）")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")

    # ---- 模型权重（与 infer_v4 / retrieval_probe 一致）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="基础模型目录（含 low_noise_model 子目录 + Wan2.1_VAE.pth + T5）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="verify 训练产出的 low_noise_model/epoch_N 目录（含 memory 权重 + "
                        "diffusion_pytorch_model.bin）。缺失时用 base 权重（memory 随机，仅负对照）。")
    p.add_argument("--model_type", type=str, default="low", choices=["low", "high"],
                   help="评测哪个子模型（low_noise_model / high_noise_model）；默认 low")

    # ---- 评测窗口选择（task 要求 3：优先 is_revisit=1 且 gap∈[lo,hi]）----
    p.add_argument("--revisit_only", action="store_true", default=True,
                   help="仅评测重访窗口（默认开）；--no_revisit_only 关闭可评全部窗口")
    p.add_argument("--no_revisit_only", dest="revisit_only", action="store_false")
    p.add_argument("--gap_lo", type=int, default=2,
                   help="is_revisit 窗口 gap 下界（含），默认 2")
    p.add_argument("--gap_hi", type=int, default=6,
                   help="is_revisit 窗口 gap 上界（含），默认 6")
    p.add_argument("--max_windows_per_episode", type=int, default=0,
                   help="每 episode 最多评测多少窗口；0=不限")

    # ---- 重访点判定（CSV 无 is_revisit 列时回退用，复用 retrieval_probe 口径）----
    p.add_argument("--hit_dist", type=float, default=40.0,
                   help="GT 重访距离阈值（数据集原生单位；v4/CSGO ≈ inches，1m≈40）")
    p.add_argument("--hit_yaw", type=float, default=30.0,
                   help="GT 重访 |yaw 差| 阈值（度）")
    p.add_argument("--intermediate_separation", type=float, default=100.0,
                   help="中间分离阈值（过滤 stationary 假位置重访；<=0 跳过）")
    p.add_argument("--min_time_gap_sec", type=float, default=5.0,
                   help="GT 重访最小时间差（秒），默认 5.0（与 retrieval_probe 同口径）")
    p.add_argument("--clip_overlap_frames", type=int, default=0,
                   help="相邻 clip overlap 帧数；v4 数据 0.5s overlap 应设 8")

    # ---- Bank 超参数（memory_ON populate 时用，与 v4 / retrieval_probe 默认对齐）----
    p.add_argument("--short_cap", type=int, default=1)
    p.add_argument("--medium_cap", type=int, default=8)
    p.add_argument("--long_cap", type=int, default=32)
    p.add_argument("--surprise_threshold", type=float, default=0.4)
    p.add_argument("--stability_threshold", type=float, default=0.2)
    p.add_argument("--novelty_threshold", type=float, default=0.7)
    p.add_argument("--half_life", type=float, default=10.0)
    p.add_argument("--dup_threshold", type=float, default=0.95)
    p.add_argument("--visual_fusion_alpha", type=float, default=0.7)
    p.add_argument("--retrieve_topk", type=int, default=6,
                   help="memory_ON 每窗口注入的检索帧数上限（与训练 retrieve 预算 6 对齐）")

    # ---- loss / 前向参数（与 train_v4 对齐）----
    p.add_argument("--nfp_loss_weight", type=float, default=0.1,
                   help="NFP loss 权重（与 train_v4 默认 0.1 对齐；仅用于 total 展示，"
                        "本脚本主对照看 diffusion_loss / nfp_loss 原始分量）")
    p.add_argument("--num_loss_samples", type=int, default=4,
                   help="每窗口重采样多少次 timestep/noise 取平均（降低 Flow Matching "
                        "随机性，使 ON/OFF Δ 更稳定）；ON 与 OFF 用同一组采样保证可比")

    # ---- 生成/数据规格 ----
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（min_time_gap_sec→帧数换算）")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--vae_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 评测窗口数据结构
# ---------------------------------------------------------------------------

@dataclass
class EvalWindow:
    """一个 target-clip 评测窗口。

    query_frame      ：target clip 起始帧（全局拼接索引），bank 只填 [0, query_frame)。
    gt_past_frames   ：该 target clip 命中的过去重访帧（全局索引；is_revisit 路径下可空）。
    is_revisit       ：是否标注为重访窗口。
    gap              ：重访 gap（CSV 提供则用 CSV 值；否则按 GT 推算）。
    """
    episode_id: str
    query_frame: int
    gt_past_frames: List[int]
    is_revisit: bool
    gap: int


# ---------------------------------------------------------------------------
# CSV is_revisit/gap 列读取（task 要求 3 优先项）
# ---------------------------------------------------------------------------

def _load_csv_revisit_flags(
    dataset_dir: str, metadata_rel: str,
) -> Tuple[Dict[Tuple[str, int], Tuple[bool, int]], bool]:
    """读取 CSV 的 (episode_id, clip_idx) → (is_revisit, gap) 映射。

    Returns:
        (flags, has_columns)：
          flags        ：{(episode_id, clip_idx): (is_revisit_bool, gap_int)}
          has_columns  ：CSV 是否含 is_revisit 列（决定是否走 CSV 优先路径）
    """
    csv_path = os.path.join(dataset_dir, metadata_rel)
    flags: Dict[Tuple[str, int], Tuple[bool, int]] = {}
    if not os.path.isfile(csv_path):
        return flags, False
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        has_revisit = "is_revisit" in fieldnames
        has_gap = "gap" in fieldnames
        if not has_revisit:
            return flags, False
        for row in reader:
            ep_id = row.get("episode_id")
            try:
                clip_idx = int(row.get("clip_idx"))
            except (TypeError, ValueError):
                continue
            try:
                is_rev = int(float(row.get("is_revisit", 0))) != 0
            except (TypeError, ValueError):
                is_rev = False
            gap = 0
            if has_gap:
                try:
                    gap = int(float(row.get("gap", 0)))
                except (TypeError, ValueError):
                    gap = 0
            flags[(ep_id, clip_idx)] = (is_rev, gap)
    return flags, True


# ---------------------------------------------------------------------------
# 评测窗口构造
# ---------------------------------------------------------------------------

def _frame_to_clip_slice(
    ep: EpisodeData, center_frame: int, frame_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """从 episode 拼接全序列截取 [center_frame, center_frame+frame_num) 的
    pose/action/intrinsics（不足时末尾对齐 + 末帧 pad；与 oracle_injection 同思路）。

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
    if poses.shape[0] < frame_num:
        pad_n = frame_num - poses.shape[0]
        poses = np.concatenate([poses, np.tile(poses[-1:], (pad_n, 1, 1))], axis=0)
        actions = np.concatenate([actions, np.tile(actions[-1:], (pad_n, 1))], axis=0)
        intr = np.concatenate([intr, np.tile(intr[-1:], (pad_n, 1))], axis=0)
    return poses, actions, intr, start


def _build_windows(
    ep: EpisodeData,
    args,
    csv_flags: Dict[Tuple[str, int], Tuple[bool, int]],
    csv_has_revisit: bool,
    min_time_gap_frames: int,
) -> List[EvalWindow]:
    """构造该 episode 的评测窗口列表。

    优先路径（CSV 含 is_revisit）：
      逐 clip（clip_idx）取 is_revisit=1 且 gap∈[gap_lo, gap_hi] 的 clip 作为 target clip，
      其全局起始帧 = ep.frame_to_clip 中该 clip 的首帧索引；bank 填 [0, 起始帧)。

    回退路径（CSV 无 is_revisit）：
      用 compute_gt_revisit 在 episode 内找重访 query 帧（与 retrieval_probe 同口径），
      把每个有 GT 过去帧的 query 帧（按 target-clip 边界对齐）作为窗口；gap 近似为
      (query_clip_idx - first_visit_clip_idx)。
    """
    windows: List[EvalWindow] = []

    if args.revisit_only and csv_has_revisit:
        # clip_array_idx → 全局起始帧（episode 拼接序列里该 clip 的首帧）
        clip_to_global_start: Dict[int, int] = {}
        for t, (clip_array_idx, _local) in enumerate(ep.frame_to_clip):
            if clip_array_idx not in clip_to_global_start:
                clip_to_global_start[clip_array_idx] = t
        for clip_array_idx, clip in enumerate(ep.clips):
            key = (ep.episode_id, clip.clip_idx)
            if key not in csv_flags:
                continue
            is_rev, gap = csv_flags[key]
            if not is_rev:
                continue
            if not (args.gap_lo <= gap <= args.gap_hi):
                continue
            g_start = clip_to_global_start.get(clip_array_idx)
            if g_start is None:
                continue
            windows.append(EvalWindow(
                episode_id=ep.episode_id,
                query_frame=int(g_start),
                gt_past_frames=[],
                is_revisit=True,
                gap=int(gap),
            ))
    else:
        # 回退：GT 重访判定（CSV 无 is_revisit 列，或显式 --no_revisit_only）
        gt = compute_gt_revisit(
            ep,
            hit_dist=args.hit_dist,
            hit_yaw=args.hit_yaw,
            intermediate_separation=args.intermediate_separation,
            min_time_gap_frames=min_time_gap_frames,
        )
        seen_starts: set = set()
        for q in sorted(gt.keys()):
            past = sorted(gt[q])
            if not past:
                continue
            # 对齐到 target-clip 起始帧（与 _frame_to_clip_slice 同口径）
            _p, _a, _i, seg_start = _frame_to_clip_slice(ep, q, args.frame_num)
            if seg_start in seen_starts:
                continue
            seen_starts.add(seg_start)
            # gap 近似：query 与最早过去帧相差多少个 clip（21 latent 帧 ≈ 1 clip → 用帧/frame_num）
            approx_gap = max(1, int(round((q - past[0]) / max(1, args.frame_num))))
            windows.append(EvalWindow(
                episode_id=ep.episode_id,
                query_frame=int(seg_start),
                gt_past_frames=[int(x) for x in past],
                is_revisit=True,
                gap=approx_gap,
            ))

    if args.max_windows_per_episode > 0:
        windows = windows[:args.max_windows_per_episode]
    return windows


# ---------------------------------------------------------------------------
# Bank populate + retrieve_revisit（memory_ON 路径，复用 retrieval_probe 口径）
# ---------------------------------------------------------------------------

def _semantic_key_for_frame(
    pose_emb: torch.Tensor,
    visual_emb: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """逐帧 semantic_key（口径与 retrieval_probe._eval_episode._semantic_key /
    oracle_injection._semantic_key_for_frame 完全一致）：
        alpha * normalize(pose_emb) + (1-alpha) * normalize(visual_emb)
    """
    pk = F.normalize(pose_emb.float().unsqueeze(0), dim=-1).squeeze(0)
    if visual_emb is None:
        return pk
    vk = F.normalize(visual_emb.float().unsqueeze(0), dim=-1).squeeze(0)
    return alpha * pk + (1.0 - alpha) * vk


def _build_bank(args):
    """构建 ThreeTierMemoryBank（容量/阈值与 v4 / retrieval_probe 默认对齐）。"""
    from memory_module.memory_bank import ThreeTierMemoryBank
    return ThreeTierMemoryBank(
        short_cap=args.short_cap,
        medium_cap=args.medium_cap,
        long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life,
        dup_threshold=args.dup_threshold,
    )


def _populate_bank(
    bank,
    ep: EpisodeData,
    query_clip_start: int,
    pose_embs: torch.Tensor,        # [T, dim] CPU
    visual_embs: Optional[torch.Tensor],   # [T, dim] CPU 或 None
    surprise: torch.Tensor,         # [T] CPU
    latents_per_frame: torch.Tensor,  # [T, z_dim, lat_h, lat_w] CPU
    abs_translations: torch.Tensor,  # [T, 3] CPU 绝对位置
    args,
) -> None:
    """对 [0, query_clip_start) 逐帧 bank.update（口径与 retrieval_probe._eval_episode /
    oracle_injection._populate_bank 完全一致：semantic_key + 绝对位置 location +
    chunk_id=t//21 + 每 21 帧 increment_age）。"""
    T = pose_embs.shape[0]
    end = max(0, min(query_clip_start, T))
    for t in range(end):
        try:
            frame_visual = visual_embs[t] if visual_embs is not None else None
            sk = _semantic_key_for_frame(
                pose_embs[t], frame_visual, args.visual_fusion_alpha
            )
            bank.update(
                pose_emb=pose_embs[t].float(),
                latent=(latents_per_frame[t].float()
                        if latents_per_frame is not None else torch.zeros(1)),
                surprise_score=float(surprise[t].item()),
                timestep=int(t),
                visual_emb=frame_visual.float() if frame_visual is not None else None,
                chunk_id=int(t // 21),
                semantic_key=sk,
                location=abs_translations[t].float(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("populate bank.update failed at ep=%s t=%d: %s",
                           ep.episode_id, t, exc)
            continue
        if t > 0 and (t % 21) == 0:
            bank.increment_age()


def _retrieve_revisit_kv(
    bank,
    query_location: torch.Tensor,   # [3] CPU
    query_timestep: int,
    model,
    latents_per_frame: torch.Tensor,  # [T, z_dim, lat_h, lat_w] CPU
    args,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """走已验证有效的 retrieve_revisit（[[F-14]]：绝对位置 L2 + 排近邻 + 只走 Long tier）
    检索 → 构造注入 K/V + tier_ids。

    每帧：
      K = frame.pose_emb（populate 时存入）
      V = get_projected_latent_emb(latents_per_frame[frame.timestep])（与 oracle_injection
          / train_v4 dummy_memory_v 同样构造方式；timestep 越界则回退 frame.visual_emb）
      tier_id = 2（retrieve_revisit 只走 Long tier）

    Returns:
        (memory_states_kv [K,dim], memory_value_kv [K,dim], tier_ids [K] int64)，CPU；
        无检索结果返回 None。
    """
    from memory_module.model_with_memory import WanModelWithMemory

    if not isinstance(model, WanModelWithMemory):
        logger.warning("model 非 WanModelWithMemory，无法构造检索 K/V")
        return None

    min_gap_frames = max(1, int(round(args.fps * args.min_time_gap_sec)))
    long_frames = bank.retrieve_revisit(
        query_location=query_location,
        query_timestep=query_timestep,
        top_k=args.retrieve_topk,
        min_gap_frames=min_gap_frames,
    )
    if not long_frames:
        return None

    T = latents_per_frame.shape[0] if latents_per_frame is not None else 0
    if hasattr(model, "latent_proj"):
        model.latent_proj.to(device)

    key_list: List[torch.Tensor] = []
    val_list: List[torch.Tensor] = []
    for frame in long_frames:
        try:
            pose_emb = frame.pose_emb.float().cpu()  # [dim]
        except Exception as exc:  # noqa: BLE001
            logger.warning("检索 K 取用失败 t=%s: %s；跳过",
                           getattr(frame, "timestep", "?"), exc)
            continue
        fi = int(getattr(frame, "timestep", -1))
        if 0 <= fi < T and latents_per_frame is not None:
            try:
                lat = latents_per_frame[fi].to(device)
                with torch.no_grad():
                    visual_emb = model.get_projected_latent_emb(lat).float().cpu()
            except Exception as exc:  # noqa: BLE001
                logger.warning("检索 V 计算失败 t=%d: %s；回退 frame.visual_emb", fi, exc)
                visual_emb = (frame.visual_emb.float().cpu()
                              if getattr(frame, "visual_emb", None) is not None
                              else pose_emb)
        else:
            visual_emb = (frame.visual_emb.float().cpu()
                          if getattr(frame, "visual_emb", None) is not None
                          else pose_emb)
        key_list.append(pose_emb)
        val_list.append(visual_emb)

    if not key_list:
        return None
    key_states = torch.stack(key_list)    # [K, dim]
    value_states = torch.stack(val_list)  # [K, dim]
    tier_ids = torch.full((key_states.shape[0],), 2, dtype=torch.long)  # Long=2
    return key_states, value_states, tier_ids


# ---------------------------------------------------------------------------
# 单窗口前向 loss（memory_ON / memory_OFF 共用；唯一差异 = 是否注入 K/V）
# ---------------------------------------------------------------------------

def _forward_loss_one_condition(
    trainer,
    model,
    ep: EpisodeData,
    window: EvalWindow,
    frames: np.ndarray,             # [T, 3, H, W] in [-1,1]
    bank_kv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    args,
    device: torch.device,
    timestep_samples: List[Tuple[float, torch.Tensor, float, torch.Tensor]],
) -> Dict[str, float]:
    """对单窗口 target clip 跑前向，返回 diffusion_loss / nfp_loss（多次采样取平均）。

    与 train_v4 multi_clip_training_step 的 target-clip loss 段**逐行对齐**：
      - encode_video → video_latent
      - prepare_y / prepare_control_signal / encode_text
      - Flow Matching：noisy = (1-sigma)*lat + sigma*noise；target = noise - lat
      - diffusion_loss = MSE(pred[:,1:], target[:,1:]) * training_weight（排除第一帧）
      - nfp_loss：last_block hook 捕获 hidden_states → nfp_head → compute_loss
                  （target = video_latent 最后帧空间均值）
    memory_ON 与 memory_OFF 共用同一组 timestep_samples（sigma/t/weight/noise），
    保证 Δloss 只来自记忆注入，不受采样随机性干扰。

    Args:
        bank_kv: memory_ON 时为 (K, V, tier_ids)；memory_OFF 时为 None。
        timestep_samples: 预采样的 [(sigma, t, training_weight, noise)] 列表（外层构造）。
    """
    from memory_module.model_with_memory import _TIER_IDS_KEY
    from memory_module.nfp_head import NFPHead

    # --- target clip 切片（pose/action/intrinsics + 起始帧）---
    poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(
        ep, window.query_frame, args.frame_num
    )
    end = min(seg_start + args.frame_num, frames.shape[0])
    video_np = frames[seg_start:end]                      # [F, 3, H, W]
    if video_np.shape[0] < args.frame_num:
        pad_n = args.frame_num - video_np.shape[0]
        video_np = np.concatenate(
            [video_np, np.tile(video_np[-1:], (pad_n, 1, 1, 1))], axis=0)
    # [F,3,H,W] → [3,F,H,W]（与 dataset / train_v4 video 排布一致）
    video = torch.from_numpy(video_np).permute(1, 0, 2, 3).contiguous().to(device)
    h, w = video.shape[2], video.shape[3]

    poses = torch.from_numpy(poses_c).float()
    actions = torch.from_numpy(acts_c).float()
    intrinsics = torch.from_numpy(intr_c).float()

    with torch.no_grad():
        video_latent = trainer.encode_video(video)       # [16, lat_f, lat_h, lat_w]
    lat_f, lat_h, lat_w = (
        video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
    )
    seq_len = lat_f * lat_h * lat_w // (trainer.patch_size[1] * trainer.patch_size[2])

    with torch.no_grad():
        context = trainer.encode_text(window_prompt(window))
        context = [c.to(torch.bfloat16)
                   if hasattr(c, "dtype") and c.dtype != torch.bfloat16 else c
                   for c in context]
        y = trainer.prepare_y(video, video_latent)

    dit_cond_dict_target = trainer.prepare_control_signal(
        poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
    )

    # --- 注入记忆（memory_ON）或不注入（memory_OFF）---
    memory_states = None
    memory_value_states = None
    if bank_kv is not None:
        key_states, value_states, tier_ids = bank_kv
        memory_states = key_states.unsqueeze(0).to(device)         # [1, K, dim]
        memory_value_states = value_states.unsqueeze(0).to(device)  # [1, K, dim]
        # _TIER_IDS_KEY 仅在 memory_ON 写入；memory_OFF 完全不写 → 与 infer_v4
        # USE_MEMORY=false 旁路语义一致（MemoryBlockWrapper 不执行 cross-attn）
        dit_cond_dict_target[_TIER_IDS_KEY] = tier_ids
    # memory_OFF：memory_states=None 且 dit_cond_dict 不含 _MEMORY_STATES_KEY

    nfp_loss_weight = args.nfp_loss_weight
    last_block = model.blocks[-1]

    diff_vals: List[float] = []
    nfp_vals: List[float] = []

    for (sigma, t, training_weight, noise) in timestep_samples:
        noise = noise.to(device=video_latent.device, dtype=video_latent.dtype)
        noisy_latent = (1.0 - sigma) * video_latent + sigma * noise
        target = noise - video_latent
        t_in = t.to(device).unsqueeze(0)

        captured: Dict[str, torch.Tensor] = {}

        def _nfp_hook(module, inp, output):
            if isinstance(output, torch.Tensor):
                captured["hs"] = output
            elif isinstance(output, (list, tuple)):
                captured["hs"] = output[0]

        handle = last_block.register_forward_hook(_nfp_hook)
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(
                        [noisy_latent],
                        t=t_in,
                        context=context,
                        seq_len=seq_len,
                        y=[y],
                        dit_cond_dict=dit_cond_dict_target,
                        memory_states=memory_states,
                        memory_value_states=memory_value_states,
                    )[0]

                # diffusion loss（排除第一帧，与 train_v4 完全对齐）
                pred_rest = pred[:, 1:]
                target_rest = target[:, 1:]
                diffusion_loss = F.mse_loss(
                    pred_rest, target_rest.to(pred_rest.dtype)
                ) * training_weight
                diff_vals.append(float(diffusion_loss.item()))

                # nfp loss（last_block hidden_states → nfp_head；target=最后帧空间均值）
                if "hs" in captured and nfp_loss_weight > 0.0:
                    hidden_states = captured["hs"]
                    nfp_head = model.nfp_head
                    pred_latent = nfp_head(hidden_states)  # [B, 16]
                    actual_latent = video_latent[:, -1].mean(
                        dim=[-2, -1]
                    ).unsqueeze(0).to(pred_latent.dtype)   # [1, 16]
                    nfp_loss_dict = NFPHead.compute_loss(
                        pred_latent, actual_latent,
                        mse_weight=1.0, cosine_weight=1.0,
                    )
                    nfp_vals.append(float(nfp_loss_dict["total"].item()))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("OOM during forward ep=%s q=%d; skip this sample",
                           ep.episode_id, window.query_frame)
        finally:
            handle.remove()

    return {
        "diffusion_loss": float(np.mean(diff_vals)) if diff_vals else float("nan"),
        "nfp_loss": float(np.mean(nfp_vals)) if nfp_vals else float("nan"),
        "n_samples_ok": len(diff_vals),
    }


def window_prompt(window: EvalWindow) -> str:
    """target clip 的 prompt。verify 数据各 clip prompt 同源（CS:GO），统一用占位文本，
    与 oracle_injection 默认 prompt 一致，避免依赖逐 clip prompt.txt。"""
    return "First-person view of CS:GO competitive gameplay"


# ---------------------------------------------------------------------------
# Trainer 构造（复用 train_v4 的 load_models，保证 loss 前向与训练同源）
# ---------------------------------------------------------------------------

def _build_trainer_and_model(args, device: torch.device):
    """实例化 LingBotMemoryTrainer 并 load_models → 返回 (trainer, model)。

    复用 train_v4 的模型/VAE/T5/schedule/cam_utils 装载，确保 encode_video /
    prepare_y / prepare_control_signal / FlowMatchingSchedule 与训练逐行一致。
    随后用 retrieval_probe 的 ft 权重加载逻辑把 verify checkpoint 注入 model
    （含 memory_cross_attn / nfp_head / latent_proj / visual_key_proj）。
    """
    from pipeline.train_v4_stage1_dual import LingBotMemoryTrainer
    import glob

    trainer_args = argparse.Namespace(
        ckpt_dir=args.ckpt_dir,
        model_type=args.model_type,
    )
    trainer = LingBotMemoryTrainer(trainer_args)
    model = trainer.load_models(device)   # 加载 base WanModelWithMemory + VAE + T5

    # --- 注入 verify ft 权重（口径与 retrieval_probe._build_memory_model 一致）---
    if args.ft_model_dir and os.path.isdir(args.ft_model_dir):
        logger.info("Loading ft weights from %s", args.ft_model_dir)
        ft_state: Dict[str, torch.Tensor] = {}
        sf_files = sorted(glob.glob(os.path.join(args.ft_model_dir, "*.safetensors")))
        if sf_files:
            try:
                from safetensors.torch import load_file
                for f in sf_files:
                    ft_state.update(load_file(f, device="cpu"))
            except ImportError:
                logger.warning("safetensors not available; skipping")
        if not ft_state:
            for pat in ("diffusion_pytorch_model*.bin", "pytorch_model*.bin"):
                for f in sorted(glob.glob(os.path.join(args.ft_model_dir, pat))):
                    ft_state.update(torch.load(f, map_location="cpu", weights_only=True))
                if ft_state:
                    break
        if ft_state:
            result = model.load_state_dict(ft_state, strict=False)
            logger.info(
                "Loaded %d ft weights (missing=%d, unexpected=%d)",
                len(ft_state), len(result.missing_keys), len(result.unexpected_keys),
            )
            _log_memory_weight_sanity(model)
        else:
            logger.warning("No ft weights found in %s (memory 仍为随机初始化，仅负对照)",
                           args.ft_model_dir)
    else:
        logger.warning(
            "ft_model_dir 缺失或无效 (%s)：memory_cross_attn 为随机初始化 → "
            "结果仅作负对照，不能据此肯定/否定 idea（见 summary 降级说明）。",
            args.ft_model_dir,
        )

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    return trainer, model


def _log_memory_weight_sanity(model) -> None:
    """打印 memory_cross_attn / gate / nfp_head 关键权重 norm + gate 均值，
    让用户确认 ft 权重真正落到了 memory 层（非全 missing 静默保留随机初始化）。
    gate≈0.1 → 记忆贡献小（OP-1 已知陷阱），ON/OFF Δ 可能很弱。"""
    def _safe_norm(attr_chain: str) -> float:
        obj = model
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

    # gate 均值（各 MemoryBlockWrapper 的 gate；OP-1 关注它是否仍 ≈0.1）
    gate_vals: List[float] = []
    for blk in getattr(model, "blocks", []):
        ca = getattr(blk, "memory_cross_attn", None)
        g = getattr(ca, "gate", None) if ca is not None else None
        if g is not None:
            try:
                gate_vals.append(float(g.detach().float().mean().item()))
            except Exception:  # noqa: BLE001
                pass
    gate_mean = float(np.mean(gate_vals)) if gate_vals else float("nan")
    logger.info(
        "ft weight sanity: blocks.0.memory_cross_attn.q.weight norm=%.4f, "
        "nfp_head.mlp.0.weight norm=%.4f, gate_mean=%.4f (≈0.1 → 记忆贡献小，"
        "ON/OFF Δ 可能弱，见 OP-1)",
        _safe_norm("blocks.0.memory_cross_attn.q.weight"),
        _safe_norm("nfp_head.mlp.0.weight"),
        gate_mean,
    )
    return gate_mean


# ---------------------------------------------------------------------------
# Summary 输出
# ---------------------------------------------------------------------------

def _pct_delta(on: float, off: float) -> float:
    """Δ% = (OFF - ON) / |OFF| * 100（正值 = ON 比 OFF 低，记忆有帮助）。"""
    if off == 0 or not np.isfinite(off) or not np.isfinite(on):
        return float("nan")
    return (off - on) / abs(off) * 100.0


def _write_outputs(args, records: List[Dict], gate_mean: float,
                   csv_has_revisit: bool) -> None:
    """写 per_window.csv + summary.json + summary.md。"""
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- per-window CSV ----
    csv_path = os.path.join(args.output_dir, "per_window.csv")
    cols = ["episode_id", "query_frame", "is_revisit", "gap", "retrieved_k",
            "diffusion_on", "diffusion_off", "diffusion_delta", "diffusion_delta_pct",
            "nfp_on", "nfp_off", "nfp_delta", "nfp_delta_pct", "n_samples_ok"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in records:
            writer.writerow({c: r.get(c, "") for c in cols})
    logger.info("Wrote per-window CSV: %s", csv_path)

    # ---- aggregate ----
    def _finite(key: str) -> List[float]:
        return [r[key] for r in records
                if isinstance(r.get(key), float) and np.isfinite(r[key])]

    agg = {}
    for metric in ("diffusion", "nfp"):
        on_vals = _finite(f"{metric}_on")
        off_vals = _finite(f"{metric}_off")
        on_mean = float(np.mean(on_vals)) if on_vals else float("nan")
        off_mean = float(np.mean(off_vals)) if off_vals else float("nan")
        agg[metric] = {
            "on_mean": on_mean,
            "off_mean": off_mean,
            "delta": (off_mean - on_mean)
                     if (np.isfinite(on_mean) and np.isfinite(off_mean)) else float("nan"),
            "delta_pct": _pct_delta(on_mean, off_mean),
            "n": len(on_vals),
        }

    # 判据：diffusion_loss ON 显著低于 OFF（Δ% > 0 且有实质幅度）
    d = agg["diffusion"]
    if not np.isfinite(d["delta_pct"]):
        verdict = "无法判定（无有效样本）"
    elif d["delta_pct"] > 1.0:
        verdict = (f"memory_ON diffusion_loss 比 OFF 低 {d['delta_pct']:.2f}% "
                   f"→ 初步信号：记忆有用（仍需结合 gate 数值 + 生成质量确认）")
    elif d["delta_pct"] < -1.0:
        verdict = (f"memory_ON diffusion_loss 比 OFF 高 {-d['delta_pct']:.2f}% "
                   f"→ 注入记忆反而抬高 loss（疑为 non-functional memory 加结构化噪声，见 OP-1）")
    else:
        verdict = (f"memory_ON ≈ memory_OFF（Δ={d['delta_pct']:.2f}%）"
                   f"→ ⚠️ 不能据此判「记忆无用」：须先排除训练不足"
                   f"（gate_mean={gate_mean:.4f}，若 ≈0.1 说明 gate 没打开），多训几 epoch 再测")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "gate_mean": gate_mean,
        "csv_has_revisit_column": csv_has_revisit,
        "n_windows": len(records),
        "aggregate": agg,
        "verdict": verdict,
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary JSON: %s", json_path)

    # ---- summary.md ----
    md: List[str] = []
    md.append("# 「记忆开/关」loss-based ablation Summary\n\n")
    md.append("## ⚠️ 降级 / 局限说明（结果须在此前提下解读）\n\n")
    md.append(
        "1. **gate 当前可能 ≈0.1**（才训少量 epoch，记忆贡献小）→ "
        f"ON/OFF 的 loss 差可能很弱甚至看不出（gate_mean={gate_mean:.4f}）。\n"
    )
    md.append(
        "2. **loss-based ≠ 生成质量**：diffusion/NFP loss 是 cheap proxy，"
        "低 loss 不直接等于「重访一致性变好」；这是目标 B 的首道信号，非终判。\n"
    )
    md.append(
        f"3. **样本量小**：本次 {len(records)} 个窗口（verify val ~5 episode）；"
        "结论置信度有限。\n"
    )
    md.append(
        "4. **若 ON ≈ OFF，不能立即判「记忆无用」**：必须先排除训练不足"
        "（gate 没打开、检索帧少、memory_cross_attn 才刚开始收敛），"
        "**多训几 epoch 再测**（沿用 current_focus 阶段三十五/三十六关键陷阱，避免重蹈 F-12："
        "用 non-functional memory 的结果否定 idea）。\n"
    )
    if not args.ft_model_dir:
        md.append(
            "5. **未提供 --ft_model_dir**：memory_cross_attn 为随机初始化 → "
            "本次结果**仅作负对照**，不能据此肯定/否定 idea。\n"
        )
    if not csv_has_revisit:
        md.append(
            "6. **CSV 无 is_revisit 列**：评测窗口由 compute_gt_revisit 在 episode 内"
            "判定（与 retrieval_probe 同口径），gap 为按帧数近似，非 CSV 标注值。\n"
        )
    md.append("\n")

    md.append("## 配置\n\n")
    md.append(f"- timestamp: {summary['timestamp']}\n")
    md.append(f"- ft_model_dir: {args.ft_model_dir}\n")
    md.append(f"- model_type: {args.model_type}\n")
    md.append(f"- metadata: {args.metadata}\n")
    md.append(f"- revisit_only: {args.revisit_only} | gap∈[{args.gap_lo},{args.gap_hi}]\n")
    md.append(f"- retrieve path: retrieve_revisit (Long tier, abs-pos L2 + 排近邻; [[F-14]])\n")
    md.append(f"- num_loss_samples: {args.num_loss_samples} | "
              f"gate_mean: {gate_mean:.4f}\n\n")

    md.append("## Aggregate（memory_ON vs memory_OFF，全窗口均值）\n\n")
    md.append("| 指标 | ON | OFF | Δ(OFF-ON) | Δ% | n |\n")
    md.append("|---|---|---|---|---|---|\n")
    for metric, label in (("diffusion", "diffusion_loss"), ("nfp", "nfp_loss")):
        a = agg[metric]
        md.append(
            f"| {label} | {a['on_mean']:.5f} | {a['off_mean']:.5f} | "
            f"{a['delta']:.5f} | {a['delta_pct']:.2f}% | {a['n']} |\n"
        )
    md.append("\n> Δ% = (OFF - ON)/|OFF|×100；**正值 = ON 比 OFF 低 = 记忆有帮助**。\n\n")

    md.append("## 判据\n\n")
    md.append(f"**{verdict}**\n\n")

    md.append("## Per-Window 明细（见 per_window.csv）\n\n")
    md.append("| episode | q_frame | gap | k | diff_ON | diff_OFF | diff_Δ% | "
              "nfp_ON | nfp_OFF | nfp_Δ% |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in records:
        md.append(
            f"| {r['episode_id']} | {r['query_frame']} | {r['gap']} | "
            f"{r['retrieved_k']} | {r['diffusion_on']:.5f} | {r['diffusion_off']:.5f} | "
            f"{r['diffusion_delta_pct']:.2f}% | {r['nfp_on']:.5f} | "
            f"{r['nfp_off']:.5f} | {r['nfp_delta_pct']:.2f}% |\n"
        )

    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as fh:
        fh.writelines(md)
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
    log_path = os.path.join(args.output_dir, "run.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（前向会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))

    logger.info("Args: %s", vars(args))
    logger.warning(
        "⚠️ 有效性前提：memory_cross_attn 须已训练（Exp0 修复后的 --ft_model_dir）。"
        "gate≈0.1（训练不足）时 ON≈OFF 不能判记忆无用——见 summary 降级说明。"
    )

    # ---- episode CSV + is_revisit 列 ----
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

    csv_flags, csv_has_revisit = _load_csv_revisit_flags(args.dataset_dir, args.metadata)
    if args.revisit_only and not csv_has_revisit:
        logger.warning(
            "CSV 无 is_revisit 列 → 回退到 compute_gt_revisit 判定重访窗口"
            "（与 retrieval_probe 同口径）；summary 会标注此降级。"
        )

    # ---- 加载 trainer + model（复用 train_v4 load_models + retrieval_probe ft 加载）----
    logger.info("Loading trainer + model (train_v4 LingBotMemoryTrainer)...")
    trainer, model = _build_trainer_and_model(args, device)
    gate_mean = _log_memory_weight_sanity(model)

    rng_seed_base = args.seed
    all_records: List[Dict] = []

    for ep_id in ep_ids:
        clips = ep_groups[ep_id]
        ep = build_episode_data(ep_id, clips,
                                clip_overlap_frames=args.clip_overlap_frames)
        if ep is None:
            continue
        T = ep.poses.shape[0]

        windows = _build_windows(
            ep, args, csv_flags, csv_has_revisit, min_time_gap_frames
        )
        if not windows:
            logger.warning("Episode %s 无评测窗口；跳过", ep_id)
            continue
        logger.info("Episode %s: T=%d, 评测窗口 %d 个", ep_id, T, len(windows))

        # 解码 video + VAE encode（前向 video_latent + bank V + populate 用）
        try:
            frames = _decode_episode_video(ep, height=height, width=width)
            latents_full = _vae_encode_batched(trainer.vae, frames, device=device,
                                               batch_frames=args.vae_batch)
            latents_per_frame = _expand_latents_to_frames(latents_full, T)
            del latents_full
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
            continue

        # bank populate 所需逐帧量（pose_emb / visual_emb / surprise / 绝对位置），
        # 口径与 retrieval_probe 同源。surprise 用 visual_cosine（自包含，无 NFP 依赖）。
        try:
            ep_pose_embs = _compute_pose_embs_episode(
                ep, model, device, height=height, width=width, fps=args.fps,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s pose_embs 计算失败: %s；跳过", ep_id, exc)
            del frames, latents_per_frame
            continue
        try:
            ep_visual_embs = _compute_visual_embs_from_latents(
                latents_per_frame, model, device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s visual_embs 计算失败: %s；continue with None",
                           ep_id, exc)
            ep_visual_embs = None
        ep_surprise = (_compute_surprise_visual_cosine(ep_visual_embs)
                       if ep_visual_embs is not None
                       else torch.zeros(T, dtype=torch.float32))
        ep_abs_translations = torch.from_numpy(ep.poses[:, :3, 3]).float()  # [T,3]

        for w_idx, window in enumerate(windows):
            # query clip 起始帧（与 _frame_to_clip_slice 对齐；windows 里已是 seg_start）
            q_clip_start = window.query_frame

            # ---- memory_ON：populate [0, q_clip_start) → retrieve_revisit → K/V ----
            bank = _build_bank(args)
            _populate_bank(
                bank, ep, q_clip_start,
                ep_pose_embs, ep_visual_embs, ep_surprise,
                latents_per_frame, ep_abs_translations, args,
            )
            q_idx = min(max(q_clip_start, 0), T - 1)
            bank_kv = _retrieve_revisit_kv(
                bank,
                query_location=ep_abs_translations[q_idx],
                query_timestep=int(q_clip_start),
                model=model,
                latents_per_frame=latents_per_frame,
                args=args,
                device=device,
            )
            retrieved_k = 0 if bank_kv is None else int(bank_kv[0].shape[0])
            if bank_kv is None:
                logger.warning(
                    "ep=%s q=%d：retrieve_revisit 为空（无满足条件的 Long 记忆帧）→ "
                    "memory_ON 退化为不注入，本窗口 ON≡OFF（记录但 Δ=0）",
                    ep_id, q_clip_start)

            # ---- 预采样 timestep/noise（ON 与 OFF 共用，保证 Δ 干净）----
            torch.manual_seed(rng_seed_base + ep_ids.index(ep_id) * 1000 + w_idx)
            timestep_samples: List[Tuple[float, torch.Tensor, float, torch.Tensor]] = []
            # 用 target clip 真实 latent 形状采 noise（先 encode 一次以拿形状）
            _poses_c, _acts_c, _intr_c, seg_start = _frame_to_clip_slice(
                ep, window.query_frame, args.frame_num)
            _end = min(seg_start + args.frame_num, frames.shape[0])
            _vid_np = frames[seg_start:_end]
            if _vid_np.shape[0] < args.frame_num:
                _pad = args.frame_num - _vid_np.shape[0]
                _vid_np = np.concatenate(
                    [_vid_np, np.tile(_vid_np[-1:], (_pad, 1, 1, 1))], axis=0)
            _vid = torch.from_numpy(_vid_np).permute(1, 0, 2, 3).contiguous().to(device)
            with torch.no_grad():
                _lat = trainer.encode_video(_vid)
            for _ in range(max(1, args.num_loss_samples)):
                sigma, t, training_weight = trainer.schedule.sample_timestep(
                    model_type=args.model_type
                )
                noise = torch.randn_like(_lat)
                timestep_samples.append((sigma, t, training_weight, noise.cpu()))
            del _vid, _lat

            # ---- 两条件前向 loss ----
            try:
                loss_on = _forward_loss_one_condition(
                    trainer, model, ep, window, frames, bank_kv, args, device,
                    timestep_samples,
                )
                loss_off = _forward_loss_one_condition(
                    trainer, model, ep, window, frames, None, args, device,
                    timestep_samples,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("前向 loss 失败 ep=%s q=%d: %s",
                                 ep_id, q_clip_start, exc)
                continue

            rec = {
                "episode_id": ep_id,
                "query_frame": int(q_clip_start),
                "is_revisit": int(window.is_revisit),
                "gap": int(window.gap),
                "retrieved_k": retrieved_k,
                "diffusion_on": loss_on["diffusion_loss"],
                "diffusion_off": loss_off["diffusion_loss"],
                "diffusion_delta": (loss_off["diffusion_loss"] - loss_on["diffusion_loss"]),
                "diffusion_delta_pct": _pct_delta(
                    loss_on["diffusion_loss"], loss_off["diffusion_loss"]),
                "nfp_on": loss_on["nfp_loss"],
                "nfp_off": loss_off["nfp_loss"],
                "nfp_delta": (loss_off["nfp_loss"] - loss_on["nfp_loss"]),
                "nfp_delta_pct": _pct_delta(loss_on["nfp_loss"], loss_off["nfp_loss"]),
                "n_samples_ok": min(loss_on["n_samples_ok"], loss_off["n_samples_ok"]),
            }
            all_records.append(rec)
            logger.info(
                "ep=%s q=%d k=%d | diff ON=%.5f OFF=%.5f (Δ%%=%.2f) | "
                "nfp ON=%.5f OFF=%.5f (Δ%%=%.2f)",
                ep_id, q_clip_start, retrieved_k,
                rec["diffusion_on"], rec["diffusion_off"], rec["diffusion_delta_pct"],
                rec["nfp_on"], rec["nfp_off"], rec["nfp_delta_pct"],
            )

        del frames, latents_per_frame, ep_pose_embs, ep_abs_translations
        if ep_visual_embs is not None:
            del ep_visual_embs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_records:
        logger.error("无任何窗口产出结果；退出。")
        return

    _write_outputs(args, all_records, gate_mean, csv_has_revisit)
    logger.info("Done. 输出目录: %s", args.output_dir)


if __name__ == "__main__":
    main()
