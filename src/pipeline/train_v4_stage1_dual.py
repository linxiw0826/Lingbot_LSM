"""
train_v4_stage1_dual.py — LingBot-World Memory Enhancement 训练脚本 v4
=======================================================================

基础：train_v3_stage1_dual.py（Method B 多 clip 顺序训练）
在 v3 基础上新增 5 项 Innovation：

  Innovation 6 — 随机 N-clip 串联训练（Stochastic N-clip Training）
    • 移除 --num_context_clips，改为 --max_context_clips（default=6）
    • 每 step 随机采样 n_ctx ~ Uniform(2, max_context_clips)
    • 数据集窗口大小 = max_context_clips + 1（最多 7 clips）
    • target clip = batch_clips[n_ctx]，其余 clips 本 step 不使用

  Innovation 7 — Context Drop-off（retrieve 前随机丢弃部分 bank entries）
    • 新增 --context_drop_p_max（default=0.3）
    • 在 target clip 调用 bank.retrieve() 之前对三层各自随机丢弃部分 entries
    • 模拟推理初期 bank 稀疏状态（借鉴 MoC arXiv 2508.21058）

  Innovation 8 — Short Bank Clip-Level Update
    • context clip 帧循环内 Medium/Long 维持逐帧 update
    • ShortTermBank 改为每 clip 结束后调用 1 次 bank.short.update(最后一帧)
    • 修复 v3 中 ShortTermBank 实际只存最近 2 latent 帧（≈0.5s）的 bug

  Innovation 9 — Visual Feature Fusion（传递 visual_emb 给 get_semantic_key）
    • context clip 每帧计算 semantic_key 时传入 visual_emb
    • target clip query 也传入 visual_emb（从第一帧 get_projected_latent_emb 计算）
    • 新增 --visual_fusion_alpha（default=0.7）

  Innovation 10 — Tier Embedding（传递 tier_ids 到 dit_cond_dict）
    • bank.retrieve(return_tier_ids=True) 获取 tier_ids [K] int64
    • dit_cond_dict_target[_TIER_IDS_KEY] = tier_ids
    • MemoryCrossAttention 内部 k = k + tier_emb(tier_ids)

继承 v3 的所有修复：
  BUG-1/2/3/4/5/6，阶段十一～十九全部修复（use_reentrant=True，RMSNorm fused kernel，
  backward DDP-safe guard，gc.collect + empty_cache，_reset_deepspeed_zero_state，etc.）

PENDING 标记：3 个 PENDING[D-03]（Stage2 逻辑，继承自 v3_dual）
"""

import argparse
import csv
import dataclasses
import gc
import logging
import math
import os
import random
import shutil
import sys
import warnings
from collections import defaultdict
from functools import wraps
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 日志（code_standards.md §4）
# ---------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path 设置
# ---------------------------------------------------------------------------
_PIPELINE_DIR = dirname(abspath(__file__))          # src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                   # src/
_PROJECT_ROOT = dirname(_SRC_DIR)                   # Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, 'refs', 'lingbot-world')

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)


# ============================================================
# CSGODataset（复用 v2_dual，用于 collate 内部单 clip 加载）
# ============================================================

class CSGODataset(Dataset):
    """Loads preprocessed CSGO clips for LingBot training.

    完全复用 train_v2_stage1_dual.py 的 CSGODataset，不做改动。
    """

    def __init__(self, dataset_dir, split="train", num_frames=81,
                 height=480, width=832, repeat=1):
        self.dataset_dir = dataset_dir
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.repeat = repeat

        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {csv_path}")
        logging.info(f"Loaded {len(self.samples)} {split} samples (x{repeat} repeat)")

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.samples)
        sample = self.samples[idx]
        clip_dir = os.path.join(self.dataset_dir, sample["clip_path"])
        return self._load_clip(clip_dir, sample)

    def _load_clip(self, clip_dir: str, sample: dict) -> dict:
        """加载单个 clip 数据（可被 CSGOMultiClipDataset 复用）。"""
        try:
            import cv2
            video_path = os.path.join(clip_dir, "video.mp4")
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.width, self.height),
                                   interpolation=cv2.INTER_LANCZOS4)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
                frames.append(frame)
            cap.release()

            while len(frames) < self.num_frames:
                frames.append(frames[-1].clone())

            video_tensor = torch.stack(frames, dim=1)  # [3, F, H, W]

            poses = np.load(os.path.join(clip_dir, "poses.npy"))
            actions = np.load(os.path.join(clip_dir, "action.npy"))
            intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))

            poses = self._pad_or_truncate(poses, self.num_frames)
            actions = self._pad_or_truncate(actions, self.num_frames)
            intrinsics = self._pad_or_truncate(intrinsics, self.num_frames)

            return {
                "video": video_tensor,
                "prompt": sample.get("prompt", ""),
                "poses": torch.from_numpy(poses).float(),
                "actions": torch.from_numpy(actions).float(),
                "intrinsics": torch.from_numpy(intrinsics).float(),
                "clip_path": sample.get("clip_path", ""),
            }
        except Exception as e:
            # code_standards.md §3：跳过损坏样本
            logging.warning(f"Error loading clip ({clip_dir}): {e}")
            raise

    def _pad_or_truncate(self, arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
        pad = np.tile(arr[-1:], rep_shape)
        return np.concatenate([arr, pad], axis=0)


# ============================================================
# CSGOMultiClipDataset（v4 更新：window_size = max_context_clips + 1）
# ============================================================

class CSGOMultiClipDataset(Dataset):
    """按 episode 分组，每次返回 max_context_clips+1 个连续 clip 的数据列表。

    v4 变更（Innovation 6）：
      - 参数从 num_context_clips 改为 max_context_clips
      - window_size = max_context_clips + 1（返回最多 max_context_clips+1 个连续 clips）
      - 训练循环内通过随机采样 n_ctx ~ Uniform(2, max_context_clips) 决定实际使用多少个 context

    每个 clip 数据（视频帧使用 cv2 + pad/truncate，与 CSGODataset 对齐）：
        {
            "video":      Tensor [3, F, H, W],
            "prompt":     str,
            "poses":      Tensor [81, 4, 4],
            "actions":    Tensor [81, 4],   # WASD 前4维
            "intrinsics": Tensor [81, 4],
            "clip_path":  str,
        }
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        phase: str = "exp",
        max_context_clips: int = 6,  # Innovation 6: 改为 max_context_clips
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        repeat: int = 1,
    ):
        """
        Args:
            dataset_dir:        数据集根目录（含 metadata_{phase}_{split}.csv）
            split:              "train" / "val"
            phase:              训练阶段，"exp"（ep01-11）或 "full"（ep01-46）；CSV 路径为 metadata_{phase}_{split}.csv
            max_context_clips:  最大 context clip 数量；window_size = max_context_clips + 1
            num_frames:         每个 clip 的帧数（81）
            height:             视频高度
            width:              视频宽度
            repeat:             数据集重复次数（用于模拟更长的 epoch）
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.phase = phase
        self.max_context_clips = max_context_clips
        # Innovation 6: window_size = max_context_clips + 1（返回足够多 clip，训练时随机截取）
        self.num_clips = max_context_clips + 1
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.repeat = repeat

        csv_path = os.path.join(dataset_dir, f"metadata_{phase}_{split}.csv")
        self._build_episode_groups(csv_path)
        logging.info(
            f"CSGOMultiClipDataset(v4): {len(self.samples)} valid episode windows "
            f"(window_size={self.num_clips}, max_context_clips={max_context_clips}, "
            f"repeat={repeat}) from {csv_path}"
        )

    def _build_episode_groups(self, csv_path: str):
        """从 CSV 中按 episode_id 分组，构建 (episode_id → sorted clips) 映射。

        退化规则：
          - 若 CSV 无 episode_id 列 → 用 clip_path 的父目录名作为 episode_id
          - 若 CSV 有 stem 列 → 按 stem 排序；否则按 clip_path 字母序

        注意：metadata_all.csv 中的列语义：
          - episode_id（字符串，如 "player01_ep01"）：clip 的所属 episode 标识符，本方法用于分组
          - episode_idx（整数，1-46）：episode 的全局编号，供 prepare_v4_splits.py 按 phase 过滤使用
          二者是不同列，功能不同。
        """
        all_rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_episode_id = "episode_id" in fieldnames
            has_stem = "stem" in fieldnames
            has_clip_idx = "clip_idx" in fieldnames
            if not has_episode_id:
                raise ValueError(
                    f"CSV '{csv_path}' missing required column 'episode_id'. "
                    "v4 数据集 metadata_all.csv 必须包含 'episode_id'（字符串标识符，如 player01_ep01）列。"
                )
            if not has_clip_idx:
                raise ValueError(
                    f"CSV '{csv_path}' missing required column 'clip_idx'. "
                    "v4 数据集 metadata_all.csv 必须包含 'clip_idx'（整数，clip 在 episode 内的序号）列。"
                )
            # assert 保证此处 has_episode_id 始终为 True，退化路径保留以兼容旧格式 CSV
            for row in reader:
                all_rows.append(row)

        if not all_rows:
            raise ValueError(f"No rows found in {csv_path}")

        # 按 episode 分组
        episode_clips: Dict[str, List[dict]] = defaultdict(list)
        for row in all_rows:
            if has_episode_id:
                ep_id = row["episode_id"]
            else:
                # 退化：用 clip_path 的父目录名
                ep_id = os.path.dirname(row.get("clip_path", ""))
            episode_clips[ep_id].append(row)

        # 对每个 episode 的 clips 排序（优先按 clip_idx 整数排序，回退到 stem，再回退到 clip_path）
        for ep_id in episode_clips:
            clips = episode_clips[ep_id]
            if has_clip_idx:
                clips.sort(key=lambda r: int(r.get("clip_idx", 0)))
            elif has_stem:
                clips.sort(key=lambda r: r.get("stem", ""))
            else:
                clips.sort(key=lambda r: r.get("clip_path", ""))
            episode_clips[ep_id] = clips

        # 构建所有合法的 (episode, start_idx) 窗口
        # 每个 episode 至少要有 num_clips 个 clip 才能构成一个训练样本
        self.samples: List[List[dict]] = []
        for ep_id, clips in sorted(episode_clips.items()):
            n = len(clips)
            if n < self.num_clips:
                logger.debug(
                    "CSGOMultiClipDataset: episode %s has %d clips < %d, skipping",
                    ep_id, n, self.num_clips,
                )
                continue
            # 使用重叠滑动窗口，stride=1（v4 数据集预生成 6524 overlapping windows，充分利用所有窗口）
            for start in range(0, n - self.num_clips + 1, 1):
                window = clips[start: start + self.num_clips]
                self.samples.append(window)

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid episode windows found (num_clips={self.num_clips}). "
                f"Check that episodes have at least {self.num_clips} clips."
            )

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx) -> List[dict]:
        """返回 max_context_clips+1 个连续 clip 的数据列表。

        每个元素与 CSGODataset.__getitem__ 返回格式相同：
          {"video": Tensor, "prompt": str, "poses": Tensor, "actions": Tensor,
           "intrinsics": Tensor, "clip_path": str}

        若加载失败，递归跳到下一个样本（code_standards.md §3）。
        """
        window = self.samples[idx % len(self.samples)]
        clips_data = []
        for row in window:
            ep_id = row["episode_id"]
            clip_idx = int(row["clip_idx"])
            clip_rel = f"clips/{ep_id}/{ep_id}_clip{clip_idx:02d}"
            clip_dir = os.path.join(self.dataset_dir, clip_rel)
            try:
                data = self._load_single_clip(clip_dir, clip_rel, row)
                clips_data.append(data)
            except Exception as e:
                logging.warning(
                    f"Error loading clip {clip_dir}: {e}. "
                    f"Falling back to next sample."
                )
                next_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(next_idx)
        return clips_data

    def _load_single_clip(self, clip_dir: str, clip_rel: str, row: dict) -> dict:
        """加载单个 clip，返回与 CSGODataset.__getitem__ 格式相同的 dict。

        Args:
            clip_dir:   clip 的绝对路径（由 dataset_dir + clip_rel 构成）
            clip_rel:   clip 的相对路径（Method B 约定：clips/{ep_id}/{ep_id}_clip{idx:02d}）
            row:        CSV 行 dict（用于读取 prompt 等元数据）
        """
        import cv2

        video_path = os.path.join(clip_dir, "video.mp4")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_LANCZOS4)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from {video_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        video_tensor = torch.stack(frames, dim=1)  # [3, F, H, W]

        poses = np.load(os.path.join(clip_dir, "poses.npy"))
        actions = np.load(os.path.join(clip_dir, "action.npy"))
        intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))

        poses = _pad_or_truncate(poses, self.num_frames)
        actions = _pad_or_truncate(actions, self.num_frames)
        intrinsics = _pad_or_truncate(intrinsics, self.num_frames)

        return {
            "video": video_tensor,
            "prompt": row.get("prompt", ""),
            "poses": torch.from_numpy(poses).float(),
            "actions": torch.from_numpy(actions).float(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "clip_path": clip_rel,
        }


def _pad_or_truncate(arr, target_len):
    """模块级 pad/truncate（供 CSGOMultiClipDataset 调用）。"""
    if len(arr) >= target_len:
        return arr[:target_len]
    rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
    pad = np.tile(arr[-1:], rep_shape)
    return np.concatenate([arr, pad], axis=0)


def multi_clip_collate_fn(batch: List[List[dict]]) -> List[dict]:
    """collate_fn：将 batch（通常 batch_size=1）的 N 个 clip 列表展开。

    因为 DataLoader batch_size=1，batch 为 [List[dict]]（外层长度1）。
    返回 List[dict]（N 个 clip，每个 dict 含 batch 维度 = 1 的 tensor）。
    """
    assert len(batch) == 1, "CSGOMultiClipDataset 只支持 batch_size=1"
    clips = batch[0]  # List[dict] of N clips

    # 为每个 clip 的 tensor 加 batch 维（squeeze 保持与 v2 一致的接口）
    result = []
    for clip in clips:
        result.append({
            "video": clip["video"].unsqueeze(0),      # [1, 3, F, H, W]
            "prompt": clip["prompt"],
            "poses": clip["poses"].unsqueeze(0),      # [1, 81, 4, 4]
            "actions": clip["actions"].unsqueeze(0),  # [1, 81, 4]
            "intrinsics": clip["intrinsics"].unsqueeze(0),  # [1, 81, 4]
            "clip_path": clip["clip_path"],
        })
    return result


# ============================================================
# FlowMatchingSchedule（直接复用 v3_dual）
# ============================================================

class FlowMatchingSchedule:
    """预计算 Flow Matching sigma schedule 及训练权重。

    与 v3_dual 完全对齐（shift=10.0, boundary=0.947）。
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 10.0,
        boundary: float = 0.947,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.boundary = boundary

        sigmas_linear = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        self.sigmas = shift * sigmas_linear / (1 + (shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * num_train_timesteps

        max_timestep = boundary * num_train_timesteps
        self.valid_train_indices = torch.where(self.timesteps_schedule < max_timestep)[0]
        self.high_noise_indices = torch.where(self.timesteps_schedule >= max_timestep)[0]

        x = self.timesteps_schedule
        steps = num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        self.training_weights = y_shifted * (steps / y_shifted.sum())

        logging.info(
            f"Sigma schedule: shift={shift}, "
            f"{len(self.valid_train_indices)}/{num_train_timesteps} "
            f"valid timesteps for low_noise_model (t < {max_timestep})"
        )

    def sample_timestep(self, model_type: str = "low") -> Tuple[float, torch.Tensor, float]:
        """从有效范围均匀随机采样一个时间步。"""
        indices = self.high_noise_indices if model_type == "high" else self.valid_train_indices
        idx = indices[torch.randint(len(indices), (1,)).item()].item()
        sigma = self.sigmas[idx].item()
        t = self.timesteps_schedule[idx]
        training_weight = self.training_weights[idx].item()
        return sigma, t, training_weight


# ============================================================
# LoRA 设置（直接复用 v3_dual）
# ============================================================

def setup_lora(model, lora_rank: int, lora_target_modules: str = "") -> nn.Module:
    """Apply LoRA to the model（与 v3_dual 完全对齐）。"""
    from peft import LoraConfig, inject_adapter_in_model

    if lora_target_modules:
        target_modules = lora_target_modules.split(",")
    else:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                v2_patterns = [
                    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                    "ffn.0", "ffn.2",
                    "cam_injector_layer1", "cam_injector_layer2",
                    "cam_scale_layer", "cam_shift_layer",
                ]
                memory_patterns = [
                    "memory_cross_attn.q",
                    "memory_cross_attn.k",
                    "memory_cross_attn.v",
                    "memory_cross_attn.o",
                ]
                all_patterns = v2_patterns + memory_patterns
                for pattern in all_patterns:
                    if pattern in name:
                        target_modules.append(name)
                        break

    logging.info(f"LoRA target modules ({len(target_modules)}): {target_modules[:5]}...")

    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                             target_modules=target_modules)
    model = inject_adapter_in_model(lora_config, model)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"LoRA: {trainable:,} trainable / {total:,} total params "
                 f"({100*trainable/total:.2f}%)")
    return model


# ============================================================
# Stage1/Stage2 冻结策略（v4 更新：新增 visual_key_proj 解冻）
# ============================================================

def freeze_for_stage(model: nn.Module, stage: int, lora_rank: int) -> list:
    """根据训练阶段设置参数冻结策略。

    v4 变更（Innovation 9）：
      Stage1 新增解冻 visual_key_proj（需在 Stage1 训练）
      Stage2 optimizer 新增 visual_key_proj 到 memory param group
    """
    if stage == 1:
        model.requires_grad_(False)

        num_unfrozen_blocks = 0
        for block in model.blocks:
            if hasattr(block, 'memory_cross_attn'):
                block.memory_cross_attn.requires_grad_(True)
                if hasattr(block, 'memory_norm'):
                    block.memory_norm.requires_grad_(True)
                num_unfrozen_blocks += 1

        model.nfp_head.requires_grad_(True)

        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        if hasattr(model, 'latent_proj'):
            model.latent_proj.requires_grad_(True)
            logging.info("Stage1: 解冻 latent_proj (F-03/F5 fix)")

        # Innovation 9: visual_key_proj 需要在 Stage1 训练（解冻）
        if hasattr(model, 'visual_key_proj'):
            model.visual_key_proj.requires_grad_(True)
            logging.info("Stage1: 解冻 visual_key_proj (Innovation 9: Visual Feature Fusion)")

        if lora_rank > 0:
            n_lora_params = 0
            for name, param in model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    param.requires_grad_(True)
                    n_lora_params += 1
            logging.info(
                f"Stage1 LoRA: 额外解冻 {n_lora_params} 个 LoRA adapter 参数（lora_A/lora_B）"
            )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        logging.info(
            f"Stage1: 解冻 {num_unfrozen_blocks} 个 MemoryBlockWrapper memory 组件 + NFPHead | "
            f"可训参数 {trainable_count:,} / 总参数 {total_count:,} ({100.0*trainable_count/max(total_count,1):.2f}%)"
        )

    elif stage == 2:
        # PENDING[D-03]: Stage2 起点权重未定，当前假设从预训练权重开始，全参解冻。
        # D-03 解除后需要修改（同 train_v3_stage1_dual.py）
        model.requires_grad_(True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        logging.info(
            f"Stage2: 全参数解冻 | "
            f"可训参数 {trainable_count:,} / 总参数 {total_count:,} ({100.0*trainable_count/max(total_count,1):.2f}%)"
        )
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")

    return [p for p in model.parameters() if p.requires_grad]


# ============================================================
# 梯度检查点（直接复用 v3_dual，use_reentrant=True）
# ============================================================

def enable_gradient_checkpointing(model: nn.Module) -> int:
    """对每个 DiT block 启用梯度检查点（ZeRO-3 兼容版，use_reentrant=True）。"""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    patched = 0
    block_container = None
    for attr in ['blocks', 'layers', 'transformer_blocks']:
        if hasattr(model, attr):
            block_container = getattr(model, attr)
            break
    if block_container is None:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 10:
                block_container = mod
                break

    if block_container is None:
        logging.warning("Gradient checkpointing: could not find DiT blocks, skipping")
        return 0

    for block in block_container:
        if not any(p.requires_grad for p in block.parameters()):
            continue
        orig_forward = block.forward

        def _make_ckpt_fn(module, fn):
            # _in_ckpt prevents infinite recursion:
            # _ckpt_forward → checkpoint(_run_via_module) → module() → _ckpt_forward → fn (direct)
            # ZeRO-3 allgather pre-hook fires when module() is called, ensuring params are gathered
            # during both the initial checkpoint forward and the backward recomputation.
            _in_ckpt = [False]

            @wraps(fn)
            def _ckpt_forward(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict=None):
                if _in_ckpt[0]:
                    return fn(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict)

                def _run_via_module(x, e, seq_lens, grid_sizes, freqs,
                                    context, context_lens, dit_cond_dict):
                    _in_ckpt[0] = True
                    try:
                        return module(x, e, seq_lens, grid_sizes, freqs,
                                      context, context_lens, dit_cond_dict)
                    finally:
                        _in_ckpt[0] = False

                # Exp0 fix (OP-1 / F-12 / D-05): activate block input grad for reentrant checkpoint.
                # With Stage1 frozen backbone, block inputs lose requires_grad; reentrant ckpt then
                # builds no backward graph, starving memory_cross_attn / gate of gradients.
                if torch.is_grad_enabled() and not x.requires_grad:
                    x = x.requires_grad_(True)

                # use_reentrant=True: avoids check_recomputed_tensors_match which fails with ZeRO-3 in-place param release
                return torch_checkpoint(
                    _run_via_module, x, e, seq_lens, grid_sizes, freqs,
                    context, context_lens, dit_cond_dict,
                    use_reentrant=True,
                )
            return _ckpt_forward

        block.forward = _make_ckpt_fn(block, orig_forward)
        patched += 1

    logging.info(f"Gradient checkpointing: patched {patched} DiT blocks")
    return patched


def _reset_deepspeed_zero_state(accelerator, optimizer=None) -> None:
    """Reset DeepSpeed ZeRO stage-1/2 internal reduce state after a mid-backward OOM.

    When backward() raises OOM, ZeRO gradient hooks may have already marked some
    parameters as reduced (params_already_reduced[i] = True) and partially filled
    the IPG bucket.  optimizer.zero_grad() clears gradient tensors but does NOT
    touch this internal bookkeeping, so the next backward hits:
        AssertionError: The parameter X has already been reduced.
    This function tries multiple paths to find the ZeRO optimizer and resets it.
    """
    candidates = []
    # Try the caller-supplied optimizer first (most direct path)
    if optimizer is not None:
        candidates.append(optimizer)
        if hasattr(optimizer, "optimizer"):
            candidates.append(optimizer.optimizer)
    # Try through the accelerator's DeepSpeed engine
    if hasattr(accelerator, "deepspeed_engine_wrapped") and accelerator.deepspeed_engine_wrapped is not None:
        eng = getattr(accelerator.deepspeed_engine_wrapped, "engine", None)
        if eng is not None:
            zero_opt = getattr(eng, "optimizer", None)
            if zero_opt is not None:
                candidates.append(zero_opt)
                if hasattr(zero_opt, "optimizer"):
                    candidates.append(zero_opt.optimizer)

    for cand in candidates:
        if not hasattr(cand, "params_already_reduced"):
            continue
        n = len(cand.params_already_reduced)
        for i in range(n):
            cand.params_already_reduced[i] = False
        for attr in ("ipg_bucket", "grads_in_ipg_bucket", "params_in_ipg_bucket",
                     "extra_large_params_truncated", "extra_large_param_grads",
                     "previous_reduce_events", "reduce_scatter_gradients_remaining_events"):
            if hasattr(cand, attr):
                setattr(cand, attr, [])
        if hasattr(cand, "elements_in_ipg_bucket"):
            cand.elements_in_ipg_bucket = 0
        # micro_step_id 追踪梯度累积边界；OOM 后 reset 避免 sync/non-sync 混乱
        # 导致同一 backward 内 double-reduce AssertionError
        if hasattr(cand, "micro_step_id"):
            cand.micro_step_id = -1
        # Clear averaged_gradients: stale partial gradients from failed backwards
        # corrupt ZeRO's state machine (causing double-reduce AssertionError) and
        # keep ~1.7 GB allocated that prevents the next backward's recomputation.
        if hasattr(cand, "averaged_gradients"):
            avg_grads = cand.averaged_gradients
            if isinstance(avg_grads, dict):
                avg_grads.clear()
            elif isinstance(avg_grads, list):
                for i in range(len(avg_grads)):
                    avg_grads[i] = None
        logging.info(f"DeepSpeed ZeRO state reset: {n} params cleared via {type(cand).__name__}")
        return

    logging.warning(
        "_reset_deepspeed_zero_state: ZeRO optimizer not found — "
        "params_already_reduced NOT reset; next backward may AssertionError"
    )


# ============================================================
# Trainer（主训练器）
# ============================================================

class LingBotMemoryTrainer:
    """Memory Enhancement 训练器（v4 多 clip 版本，含 Innovations 6-10）。

    核心差异（相对 v3_dual）：
      - multi_clip_training_step 实现 Innovation 6/7/8/9/10
      - freeze_for_stage 新增 visual_key_proj 解冻（Innovation 9）
      - load_models / encode_video / encode_text / prepare_y / prepare_control_signal 与 v3_dual 完全复用
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")

        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.schedule = FlowMatchingSchedule(
            num_train_timesteps=1000,
            shift=10.0,
            boundary=0.947,
        )

        # T5 prompt cache
        self._t5_cache: dict = {}

        self.cam_utils = {}

    def load_models(self, device: torch.device):
        """加载 WanModelWithMemory、VAE、T5（与 v3_dual 完全对齐）。"""
        self.device = device
        ckpt_dir = self.args.ckpt_dir

        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.cam_utils import (
            interpolate_camera_poses, compute_relative_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )
        from memory_module.model_with_memory import WanModelWithMemory

        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

        _subfolder = "low_noise_model" if self.args.model_type == "low" else "high_noise_model"
        logging.info(f"Loading base WanModel ({_subfolder})...")
        base_wan_model = WanModel.from_pretrained(
            ckpt_dir,
            subfolder=_subfolder,
            torch_dtype=torch.bfloat16,
            control_type="act",
        )

        wancamctrl = base_wan_model.patch_embedding_wancamctrl
        logging.info(
            f"patch_embedding_wancamctrl: Linear({wancamctrl.in_features}, "
            f"{wancamctrl.out_features})"
        )

        logging.info("Converting to WanModelWithMemory...")
        model = WanModelWithMemory.from_wan_model(base_wan_model)
        del base_wan_model
        gc.collect()
        model.train()

        logging.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
            device=self.device,
        )

        logging.info("Loading T5 text encoder...")
        self.t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(ckpt_dir, "google", "umt5-xxl"),
        )

        return model

    @torch.no_grad()
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """video_tensor: [3, F, H, W] -> latent [16, lat_f, lat_h, lat_w]（与 v3_dual 对齐）。"""
        latent = self.vae.encode([video_tensor.to(self.device)])[0]
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def encode_text(self, prompt: str) -> list:
        """prompt string -> list of text embedding tensors（与 v3_dual 对齐，含 CPU cache）。"""
        if prompt in self._t5_cache:
            return [t.to(self.device) for t in self._t5_cache[prompt]]
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        self._t5_cache[prompt] = [t.cpu() for t in context]
        return [t.to(self.device) for t in self._t5_cache[prompt]]

    def prepare_y(self, video_tensor: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Prepare conditional input y（与 v3_dual 完全对齐）。"""
        lat_f, lat_h, lat_w = latent.shape[1], latent.shape[2], latent.shape[3]
        F_total = video_tensor.shape[1]
        h, w = video_tensor.shape[2], video_tensor.shape[3]

        first_frame = video_tensor[:, 0:1, :, :]
        zeros = torch.zeros(3, F_total - 1, h, w, device=video_tensor.device)
        vae_input = torch.concat([first_frame, zeros], dim=1)
        y_latent = self.vae.encode([vae_input.to(self.device)])[0]

        msk = torch.ones(1, F_total, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # [4, lat_f, lat_h, lat_w]

        y = torch.concat([msk, y_latent])  # [20, lat_f, lat_h, lat_w]
        return y

    def prepare_control_signal(
        self,
        poses: torch.Tensor,
        actions: torch.Tensor,
        intrinsics: torch.Tensor,
        h: int, w: int,
        lat_f: int, lat_h: int, lat_w: int,
    ) -> dict:
        """Prepare dit_cond_dict（与 v3_dual 完全对齐）。

        v3/v4 action 格式：[81, 4]（与 v2 相同，前 4 维 WASD）。
        """
        interpolate_camera_poses = self.cam_utils["interpolate_camera_poses"]
        compute_relative_poses = self.cam_utils["compute_relative_poses"]
        get_plucker_embeddings = self.cam_utils["get_plucker_embeddings"]
        get_Ks_transformed = self.cam_utils["get_Ks_transformed"]

        num_frames = poses.shape[0]

        Ks = get_Ks_transformed(
            intrinsics,
            height_org=480, width_org=832,
            height_resize=h, width_resize=w,
            height_final=h, width_final=w,
        )
        Ks_single = Ks[0]

        len_c2ws = num_frames
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=poses[:, :3, :3].cpu().numpy(),
            src_trans_vec=poses[:, :3, 3].cpu().numpy(),
            tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)

        Ks_repeated = Ks_single.repeat(len(c2ws_infer), 1)
        c2ws_infer = c2ws_infer.to(self.device)
        Ks_repeated = Ks_repeated.to(self.device)

        # v3/v4 action 格式：[81, 4]（前 4 维 WASD，与 v2 相同）
        wasd = actions[::4, :4].to(self.device)
        if len(wasd) > len(c2ws_infer):
            wasd = wasd[:len(c2ws_infer)]
        elif len(wasd) < len(c2ws_infer):
            pad = wasd[-1:].repeat(len(c2ws_infer) - len(wasd), 1)
            wasd = torch.cat([wasd, pad], dim=0)

        c2ws_plucker_emb = get_plucker_embeddings(
            c2ws_infer, Ks_repeated, h, w, only_rays_d=True
        )  # [lat_f, h, w, 3]

        c1 = int(h // lat_h)
        c2 = int(w // lat_w)

        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=c1, c2=c2,
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w,
        ).to(torch.bfloat16)

        wasd_tensor = wasd[:, None, None, :].repeat(1, h, w, 1)
        wasd_tensor = rearrange(
            wasd_tensor,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=c1, c2=c2,
        )
        wasd_tensor = wasd_tensor[None, ...]
        wasd_tensor = rearrange(
            wasd_tensor, 'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w,
        ).to(torch.bfloat16)

        c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_tensor], dim=1)

        dit_cond_dict = {
            "c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0),
        }
        return dit_cond_dict


# ============================================================
# Checkpoint 保存（直接复用 v3_dual）
# ============================================================

def save_checkpoint(accelerator, model, args, tag: str, epoch: int = 0, global_step: int = 0):
    """保存 checkpoint（与 v3_dual 完全对齐）。"""
    save_dir = os.path.join(args.output_dir, tag)

    if args.lora_rank > 0:
        if accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            lora_state_dict = {
                name: param.data.cpu()
                for name, param in unwrapped.named_parameters()
                if param.requires_grad
            }
            torch.save(lora_state_dict, os.path.join(save_dir, "lora_weights.pth"))
            logging.info(
                f"Saved LoRA checkpoint ({len(lora_state_dict)} params) -> {save_dir}"
            )
    else:
        if accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        model.save_16bit_model(save_dir, "diffusion_pytorch_model.bin")

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_config(save_dir)
            saved_path = os.path.join(save_dir, "diffusion_pytorch_model.bin")
            sd = torch.load(saved_path, map_location="cpu", weights_only=True)
            n_empty = sum(1 for v in sd.values() if v.numel() == 0)
            logging.info(
                f"Saved full checkpoint -> {save_dir} "
                f"({len(sd)} params, {n_empty} empty)"
            )
            if n_empty > 0:
                logging.error(f"WARNING: {n_empty} parameters have empty tensors!")
            del sd

        accelerator.save_state(os.path.join(save_dir, "training_state"))
        gc.collect()
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            import json
            metadata = {"epoch": epoch, "global_step": global_step}
            with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f)
            logging.info(
                f"Saved training state -> {save_dir}/training_state "
                f"(epoch={epoch}, global_step={global_step})"
            )

    accelerator.wait_for_everyone()


# ============================================================
# 多 clip 训练步骤（v4 核心，实现 Innovations 6/7/8/9/10）
# ============================================================

def multi_clip_training_step(
    trainer: LingBotMemoryTrainer,
    model: nn.Module,
    batch_clips: List[dict],
    args,
    n_ctx: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """多 clip 顺序训练步骤（v4：Innovations 6/7/8/9/10）。

    args:
        trainer:     LingBotMemoryTrainer 实例
        model:       accelerator.unwrap_model(model)，WanModelWithMemory
        batch_clips: List[dict]，max_context_clips+1 个 clip（来自 multi_clip_collate_fn）
        args:        argparse.Namespace

    返回：
        total_loss:          Tensor scalar，用于 accelerator.backward(loss)
        _loss_components:    Dict[str, float]，各分量标量值（diffusion/nfp/vis_align/latent_proj），
                             未计算的分量填 0.0，供 W&B 精细报告（M-01 修复）

    Innovation 6 — 随机 N-clip：
        n_ctx = random.randint(2, args.max_context_clips)  每 step 随机采样
        context clips = batch_clips[:n_ctx]（前 n_ctx 个）
        target clip   = batch_clips[n_ctx]（第 n_ctx+1 个）
        余下 clips 本 step 不使用

    Innovation 7 — Context Drop-off：
        target clip 的 bank.retrieve() 之前随机丢弃三层部分 entries

    Innovation 8 — Short Bank Clip-Level Update：
        Medium/Long 维持逐帧 update；Short 每 clip 结束后调 1 次（最后一帧）

    Innovation 9 — Visual Feature Fusion：
        get_semantic_key(pose_emb, visual_emb=visual_emb_t, alpha=args.visual_fusion_alpha)

    Innovation 10 — Tier Embedding：
        bank.retrieve(return_tier_ids=True)，tier_ids 放入 dit_cond_dict_target[_TIER_IDS_KEY]
    """
    from memory_module.memory_bank import ThreeTierMemoryBank, MemoryFrame
    from memory_module.nfp_head import NFPHead
    from memory_module.model_with_memory import (
        WanModelWithMemory, _MEMORY_STATES_KEY, _MEMORY_VALUE_KEY, _TIER_IDS_KEY
    )

    device = trainer.device

    # ----------------------------------------------------------------
    # Innovation 6：随机采样 n_ctx ~ Uniform(2, max_context_clips)
    # n_ctx 由调用方在 broadcast 后传入（ZeRO-3 要求所有 rank forward 次数一致）
    # ----------------------------------------------------------------
    if n_ctx is None:
        n_ctx = random.randint(2, args.max_context_clips)
    context_clips = batch_clips[:n_ctx]
    target_clip = batch_clips[n_ctx]
    # batch_clips[n_ctx+1:] 本 step 不使用

    logging.debug(
        f"v4 Stochastic N-clip: n_ctx={n_ctx}, total_clips={len(batch_clips)}, "
        f"using context[0:{n_ctx}] + target[{n_ctx}]"
    )

    # 每次 training_step 创建新 bank（每个训练样本独立，不跨样本积累）
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

    # ----------------------------------------------------------------
    # Context clips：no_grad，填充 ThreeTierMemoryBank
    # Innovation 8：Short 每 clip 只 update 1 次（最后一帧）
    # Innovation 9：semantic_key 融合 visual_emb
    # ----------------------------------------------------------------
    for clip_idx, clip in enumerate(context_clips):
        video = clip["video"].squeeze(0).to(device)   # [3, F, H, W]
        poses = clip["poses"].squeeze(0)               # [81, 4, 4]
        actions = clip["actions"].squeeze(0)           # [81, 4]
        intrinsics = clip["intrinsics"].squeeze(0)     # [81, 4]
        prompt = clip["prompt"]

        h, w = video.shape[2], video.shape[3]

        with torch.no_grad():
            # 1. VAE encode
            video_latent = trainer.encode_video(video)
            lat_f, lat_h, lat_w = (
                video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
            )
            seq_len = lat_f * lat_h * lat_w // (trainer.patch_size[1] * trainer.patch_size[2])

            # 2. prepare_control_signal
            assert not torch.is_grad_enabled(), "context clip 不应有梯度（no_grad 失效）"
            context = trainer.encode_text(prompt)
            # M-2 fix: encode_text 可能返回 float32，model forward 期望 bfloat16
            context = [c.to(torch.bfloat16) if hasattr(c, 'dtype') and c.dtype != torch.bfloat16 else c for c in context]
            y = trainer.prepare_y(video, video_latent)
            dit_cond_dict = trainer.prepare_control_signal(
                poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
            )

            # 3. 用 get_projected_frame_embs 计算 frame_embs: [lat_f, 5120]
            c2ws_emb_raw = dit_cond_dict["c2ws_plucker_emb"][0]  # [1, C, lat_f, lat_h, lat_w]
            frame_embs = model.get_projected_frame_embs(c2ws_emb_raw).detach()  # [lat_f, 5120]

            # 4. bank 非空时先 retrieve，用于 context forward（若 bank 为空则用 dummy）
            query_emb_ctx = frame_embs[0]  # 第一帧 pose emb
            # Innovation 9: context query 也用 visual_emb 融合
            if hasattr(model, 'latent_proj'):
                _ctx_latent_frame_0 = video_latent[:, 0]  # [z_dim, lat_h, lat_w]
                _ctx_visual_emb_0 = model.get_projected_latent_emb(_ctx_latent_frame_0)  # [5120]
            else:
                _ctx_visual_emb_0 = None
            query_semantic_key_ctx = model.get_semantic_key(
                query_emb_ctx,
                visual_emb=_ctx_visual_emb_0,
                alpha=args.visual_fusion_alpha,
            )

            if bank.size() > 0:
                retrieved_ctx = bank.retrieve(
                    query_emb_ctx,
                    query_semantic_key=query_semantic_key_ctx,
                    short_n=args.short_cap,
                    medium_k=args.hybrid_medium_k,
                    long_k=args.hybrid_long_k,
                    device=device,
                )
            else:
                retrieved_ctx = None

            if retrieved_ctx is not None:
                key_states_ctx, value_states_ctx = retrieved_ctx
                memory_states_ctx = key_states_ctx.unsqueeze(0)    # [1, K, 5120]
                memory_value_states_ctx = value_states_ctx.unsqueeze(0)  # [1, K, 5120]
            else:
                # 无 memory 时用 dummy（与 v3 training_step 对齐）
                memory_states_ctx = torch.zeros(
                    1, 1, model.dim, device=device, dtype=torch.bfloat16
                )
                if hasattr(model, 'latent_proj'):
                    _v_feat = video_latent[:, -1].float().mean(dim=[-2, -1])
                    _v_emb = model.latent_proj(
                        _v_feat.to(model.latent_proj.weight.dtype)
                    )
                    memory_value_states_ctx = _v_emb.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
                else:
                    memory_value_states_ctx = memory_states_ctx

            # 5. 注册 NFP hook 在 model.blocks[-1] 上（捕获 hidden_states）
            captured_hidden_states_ctx = {}

            def _nfp_hook_ctx(module, input, output):
                if isinstance(output, torch.Tensor):
                    captured_hidden_states_ctx["hs"] = output
                elif isinstance(output, (list, tuple)):
                    captured_hidden_states_ctx["hs"] = output[0]

            last_block = model.blocks[-1]
            hook_handle_ctx = last_block.register_forward_hook(_nfp_hook_ctx)

            try:
                # 6. Forward with no_grad（context clip 不参与反传）
                with torch.no_grad():
                    sigma_ctx, t_ctx, _ = trainer.schedule.sample_timestep(
                        model_type=args.model_type
                    )
                    t_ctx = t_ctx.to(device).unsqueeze(0)
                    noise_ctx = torch.randn_like(video_latent)
                    noisy_latent_ctx = (1.0 - sigma_ctx) * video_latent + sigma_ctx * noise_ctx

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _ = model(
                            [noisy_latent_ctx],
                            t=t_ctx,
                            context=context,
                            seq_len=seq_len,
                            y=[y],
                            dit_cond_dict=dit_cond_dict,
                            memory_states=memory_states_ctx,
                            memory_value_states=memory_value_states_ctx,
                        )
            finally:
                hook_handle_ctx.remove()

            # 7. 从 hook 捕获的 hidden_states 用 NFPHead 计算 clip-level surprise
            clip_surprise = 0.4   # 默认 surprise（若 hook 未捕获）
            if "hs" in captured_hidden_states_ctx:
                hidden_states_ctx = captured_hidden_states_ctx["hs"]  # [B, L, 5120]
                nfp_head = model.nfp_head
                with torch.no_grad():
                    pred_latent_ctx = nfp_head(hidden_states_ctx)  # [B, 16]
                    actual_latent_ctx = video_latent[:, -1].mean(
                        dim=[-2, -1]
                    ).unsqueeze(0).to(pred_latent_ctx.dtype)  # [1, 16]
                    surprise_scores = NFPHead.compute_surprise(
                        pred_latent_ctx, actual_latent_ctx
                    )
                    clip_surprise = float(surprise_scores[0].item())

            # 8. 逐帧计算 visual_emb 和 semantic_key，然后更新 bank
            # Innovation 8: Medium/Long 逐帧 update；Short 在帧循环结束后 1 次 update（最后一帧）
            # Innovation 9: semantic_key 融合 visual_emb
            with torch.no_grad():
                # 保存最后一帧的变量（Innovation 8：Short clip-level update 使用）
                pose_emb_last = None
                latent_last = None
                visual_emb_last = None
                surprise_last = clip_surprise
                semantic_key_last = None

                for t_idx in range(lat_f):
                    pose_emb_t = frame_embs[t_idx]  # [5120]

                    if hasattr(model, 'latent_proj'):
                        latent_frame = video_latent[:, t_idx]  # [z_dim, lat_h, lat_w]
                        visual_emb_t = model.get_projected_latent_emb(latent_frame)  # [5120]
                    else:
                        visual_emb_t = None

                    # Innovation 9: semantic_key 融合 visual_emb
                    semantic_key_t = model.get_semantic_key(
                        pose_emb_t,
                        visual_emb=visual_emb_t,
                        alpha=args.visual_fusion_alpha,
                    )  # [5120]

                    # Innovation 8: 逐帧更新 Medium + Long（通过子 bank 直接调用）
                    _frame = MemoryFrame(
                        pose_emb=pose_emb_t.detach().cpu(),
                        latent=video_latent[:, t_idx].detach().cpu(),
                        surprise_score=float(clip_surprise),
                        timestep=int(clip_idx * lat_f + t_idx),
                        visual_emb=visual_emb_t.detach().cpu() if visual_emb_t is not None else None,
                        chunk_id=int(clip_idx),
                        age=0,
                        semantic_key=semantic_key_t.detach().cpu() if semantic_key_t is not None else None,
                    )
                    bank.medium.update(dataclasses.replace(_frame))
                    bank.long.update(dataclasses.replace(_frame))

                    # 记录最后一帧（供 Short clip-level update 使用）
                    pose_emb_last = pose_emb_t
                    latent_last = video_latent[:, t_idx]
                    visual_emb_last = visual_emb_t
                    semantic_key_last = semantic_key_t

                # Innovation 8: Short clip-level update（每 clip 只调 1 次，使用最后一帧）
                if pose_emb_last is not None:
                    _last_frame = MemoryFrame(
                        pose_emb=pose_emb_last.detach().cpu(),
                        latent=latent_last.detach().cpu(),
                        surprise_score=float(surprise_last),
                        timestep=int(clip_idx * lat_f + lat_f - 1),
                        visual_emb=visual_emb_last.detach().cpu() if visual_emb_last is not None else None,
                        chunk_id=int(clip_idx),
                        age=0,
                        semantic_key=semantic_key_last.detach().cpu() if semantic_key_last is not None else None,
                    )
                    bank.short.update(_last_frame)

            # 9. 若不是最后一个 context clip：bank.increment_age()
            if clip_idx < len(context_clips) - 1:
                bank.increment_age()

    # ----------------------------------------------------------------
    # Target clip：有梯度，计算 loss
    # Innovation 7：Context Drop-off（retrieve 前随机丢弃三层 entries）
    # Innovation 10：bank.retrieve(return_tier_ids=True)
    # ----------------------------------------------------------------
    video = target_clip["video"].squeeze(0).to(device)   # [3, F, H, W]
    poses = target_clip["poses"].squeeze(0)
    actions = target_clip["actions"].squeeze(0)
    intrinsics = target_clip["intrinsics"].squeeze(0)
    prompt = target_clip["prompt"]

    h, w = video.shape[2], video.shape[3]

    with torch.no_grad():
        video_latent = trainer.encode_video(video)

    lat_f, lat_h, lat_w = (
        video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
    )
    seq_len = lat_f * lat_h * lat_w // (trainer.patch_size[1] * trainer.patch_size[2])

    with torch.no_grad():
        context = trainer.encode_text(prompt)
        # M-3 fix: encode_text 可能返回 float32，model forward 期望 bfloat16
        context = [c.to(torch.bfloat16) if hasattr(c, 'dtype') and c.dtype != torch.bfloat16 else c for c in context]
        y = trainer.prepare_y(video, video_latent)

    dit_cond_dict_target = trainer.prepare_control_signal(
        poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
    )

    # 1. 用 get_projected_frame_embs 计算 query_emb（第一帧）
    with torch.no_grad():
        c2ws_emb_raw = dit_cond_dict_target["c2ws_plucker_emb"][0]  # [1, C, lat_f, lat_h, lat_w]
        frame_embs_target = model.get_projected_frame_embs(c2ws_emb_raw)  # [lat_f, 5120]
        query_pose_emb = frame_embs_target[0]  # [5120]

        # Innovation 9: target query 也融合 visual_emb（第一帧）
        if hasattr(model, 'latent_proj'):
            query_visual_emb = model.get_projected_latent_emb(
                video_latent[:, 0]
            )  # [5120]（第一帧 visual_emb）
        else:
            query_visual_emb = None

        query_semantic_key = model.get_semantic_key(
            query_pose_emb,
            visual_emb=query_visual_emb,
            alpha=args.visual_fusion_alpha,
        )  # [5120]

    # [Innovation 9] visual_key_proj 梯度路径
    # 模式同 latent_proj 的 dummy_memory_v：在 torch.no_grad() 外调用，确保参数梯度积累
    # 即使 query_visual_emb 是 detached tensor，visual_key_proj.weight 也能积累梯度
    _vis_proj_for_grad: Optional[torch.Tensor] = None
    if (
        query_visual_emb is not None
        and hasattr(model, "visual_key_proj")
        and model.visual_key_proj.weight.requires_grad
    ):
        _vis_proj_for_grad = torch.nn.functional.normalize(
            model.visual_key_proj(
                query_visual_emb.to(
                    device=model.visual_key_proj.weight.device,
                    dtype=model.visual_key_proj.weight.dtype,
                )
            ),
            dim=-1,
        )  # [5120], grad 流经 visual_key_proj.weight

    # [H-01 fix] latent_proj 梯度路径（对称 Innovation 9 的 _vis_proj_for_grad 模式）
    # get_projected_latent_emb 在 no_grad 内调用（上方 with torch.no_grad() 块），latent_proj 无梯度。
    # 此处在 no_grad 外对同一帧重新调用，建立 latent_proj.weight 的梯度路径。
    # 即使输入 video_latent[:, 0] 是 detached tensor（来自 VAE encode），权重梯度仍能积累。
    _latent_proj_for_grad: Optional[torch.Tensor] = None
    if hasattr(model, "latent_proj") and model.latent_proj.weight.requires_grad:
        _first_frame_latent = video_latent[:, 0].detach()  # [z_dim=16, lat_h, lat_w]
        _latent_proj_for_grad = torch.nn.functional.normalize(
            model.latent_proj(
                _first_frame_latent.to(
                    device=model.latent_proj.weight.device,
                    dtype=model.latent_proj.weight.dtype,
                ).mean(dim=[-2, -1])  # [z_dim=16] 空间平均，与 get_projected_latent_emb 逻辑一致
            ),
            dim=-1,
        )  # [5120], grad 流经 latent_proj.weight

    # ----------------------------------------------------------------
    # Innovation 7：Context Drop-off（retrieve 之前，对三层随机丢弃部分 entries）
    # 模拟推理初期 bank 稀疏状态（借鉴 MoC arXiv 2508.21058）
    # ----------------------------------------------------------------
    if args.context_drop_p_max > 0.0:
        for tier_bank in [bank.short, bank.medium, bank.long]:
            n_entries = len(tier_bank.frames)
            if n_entries == 0:
                continue
            p_drop = random.uniform(0.0, args.context_drop_p_max)
            n_drop = int(p_drop * n_entries)
            if n_drop > 0:
                drop_idxs = random.sample(range(n_entries), n_drop)
                for idx in sorted(drop_idxs, reverse=True):
                    tier_bank.frames.pop(idx)

    # 2. bank.retrieve（bank 可能包含 context clips 填充的帧）
    # Innovation 10：return_tier_ids=True 获取 tier_ids
    if bank.size() > 0:
        retrieved = bank.retrieve(
            query_pose_emb,
            query_semantic_key=query_semantic_key,
            short_n=args.short_cap,
            medium_k=args.hybrid_medium_k,
            long_k=args.hybrid_long_k,
            device=device,
            return_tier_ids=True,  # Innovation 10: Tier Embedding
        )
    else:
        retrieved = None

    _retrieved_k: int = 0  # 记录实际检索到的帧数（0 表示 retrieve 失败或 bank 为空）
    if retrieved is not None:
        # Innovation 10: 解包三元组 (key_states, value_states, tier_ids)
        key_states, value_states, tier_ids = retrieved
        _retrieved_k = key_states.shape[0]  # 记录实际检索帧数
        assert key_states.shape[0] <= 6, (
            f"retrieve() returned {key_states.shape[0]} > 6 frames"
        )
        memory_states = key_states.unsqueeze(0)        # [1, K, 5120]
        memory_value_states = value_states.unsqueeze(0)  # [1, K, 5120]
        # Innovation 10: 注入 tier_ids 到 dit_cond_dict_target
        dit_cond_dict_target[_TIER_IDS_KEY] = tier_ids  # [K] int64, CPU 也可（forward 内 .to(x.device)）
    else:
        # bank 为空（n_ctx=0 或 drop-off 清空全部），回退到 dummy，与 v3 对齐
        memory_states = torch.zeros(
            1, 1, model.dim, device=device, dtype=torch.bfloat16
        )
        if hasattr(model, 'latent_proj'):
            _v_feat = video_latent[:, -1].float().mean(dim=[-2, -1])
            _v_emb = model.latent_proj(
                _v_feat.to(model.latent_proj.weight.dtype)
            )
            memory_value_states = _v_emb.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        else:
            memory_value_states = memory_states

    # 采样 timestep
    sigma, t, training_weight = trainer.schedule.sample_timestep(
        model_type=args.model_type
    )
    t = t.to(device).unsqueeze(0)

    # Flow Matching：加噪
    noise = torch.randn_like(video_latent)
    noisy_latent = (1.0 - sigma) * video_latent + sigma * noise
    target = noise - video_latent

    # NFP hook 注册（与 v3_dual.training_step 完全对齐）
    last_block = model.blocks[-1]
    captured_hidden_states = {}

    def _nfp_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured_hidden_states["hs"] = output
        elif isinstance(output, (list, tuple)):
            captured_hidden_states["hs"] = output[0]

    hook_handle = last_block.register_forward_hook(_nfp_hook)

    # M-01 修复：各分量标量初始值（未计算时保持 0.0，防止 NameError）
    _diffusion_loss_val: float = 0.0
    _nfp_loss_val: float = 0.0
    _vis_align_loss_val: float = 0.0
    _latent_proj_loss_val: float = 0.0

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(
                [noisy_latent],
                t=t,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond_dict_target,
                memory_states=memory_states,
                memory_value_states=memory_value_states,
            )[0]

        # Diffusion loss（排除第一帧，与 v3_dual 完全对齐）
        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        diffusion_loss = F.mse_loss(pred_rest, target_rest.to(pred_rest.dtype))
        diffusion_loss = diffusion_loss * training_weight
        _diffusion_loss_val = diffusion_loss.item()  # M-01：保存纯 diffusion loss 标量

        # NFP Loss（只在 target clip 计算，与 v3_dual 对齐）
        nfp_loss_weight = args.nfp_loss_weight
        if "hs" in captured_hidden_states and nfp_loss_weight > 0.0:
            hidden_states = captured_hidden_states["hs"]  # [B, L, 5120]
            nfp_head = model.nfp_head
            pred_latent = nfp_head(hidden_states)  # [B, 16]

            # NFP target：clip 最后帧空间均值（与 v3_dual M-2 修复对齐）
            actual_latent = video_latent[:, -1].mean(
                dim=[-2, -1]
            ).unsqueeze(0).to(pred_latent.dtype)  # [1, 16]

            nfp_loss_dict = NFPHead.compute_loss(
                pred_latent, actual_latent,
                mse_weight=1.0, cosine_weight=1.0,
            )
            nfp_loss = nfp_loss_dict['total']
            _nfp_loss_val = nfp_loss.item()  # M-01：保存 nfp loss 标量
            total_loss = diffusion_loss + nfp_loss_weight * nfp_loss
        else:
            total_loss = diffusion_loss

    finally:
        hook_handle.remove()

    # [Innovation 9 + H-01 fix] 辅助对齐 loss（visual_key_proj 和 latent_proj 梯度路径）
    # 共用同一个 _pose_key_ref（只计算一次，避免重复 get_semantic_key 调用）
    if _vis_proj_for_grad is not None or _latent_proj_for_grad is not None:
        with torch.no_grad():
            _pose_key_ref = model.get_semantic_key(query_pose_emb)  # pose-only，detached [5120]
        if _vis_proj_for_grad is not None:
            _vis_align_loss = (
                1.0
                - torch.nn.functional.cosine_similarity(
                    _vis_proj_for_grad.to(_pose_key_ref.device, _pose_key_ref.dtype).unsqueeze(0),
                    _pose_key_ref.unsqueeze(0),
                )
            ).clamp(min=0.0).mean()
            _vis_align_loss_val = _vis_align_loss.item()  # M-01：保存 vis_align loss 标量
            total_loss = total_loss + 0.05 * _vis_align_loss
        if _latent_proj_for_grad is not None:
            # _latent_proj_for_grad shape: [5120]（与 _vis_proj_for_grad 完全对称，无 batch 维度）
            _latent_proj_loss = (
                1.0
                - torch.nn.functional.cosine_similarity(
                    _latent_proj_for_grad.to(_pose_key_ref.device, _pose_key_ref.dtype).unsqueeze(0),
                    _pose_key_ref.unsqueeze(0),
                )
            ).clamp(min=0.0).mean()
            _latent_proj_loss_val = _latent_proj_loss.item()  # M-01：保存 latent_proj loss 标量
            total_loss = total_loss + 0.05 * _latent_proj_loss

    # 诊断 key（不参与 loss 计算和 W&B 上报，仅供日志使用）
    _loss_components: Dict[str, float] = {
        "diffusion": _diffusion_loss_val,
        "nfp": _nfp_loss_val,
        "vis_align": _vis_align_loss_val,
        "latent_proj": _latent_proj_loss_val,
    }
    _loss_components["bank_short"]       = float(len(bank.short.frames))
    _loss_components["bank_medium"]      = float(len(bank.medium.frames))
    _loss_components["bank_long"]        = float(len(bank.long.frames))
    _loss_components["bank_retrieved_k"] = float(_retrieved_k)
    return total_loss, _loss_components


# ============================================================
# 命令行参数（v4 更新：移除 --num_context_clips，新增 --max_context_clips / --context_drop_p_max / --visual_fusion_alpha）
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    v4 变更（相对 v3）：
      - 移除 --num_context_clips
      - 新增 --max_context_clips（type=int, default=6，Innovation 6）
      - 新增 --context_drop_p_max（type=float, default=0.3，Innovation 7）
      - 新增 --visual_fusion_alpha（type=float, default=0.7，Innovation 9）
      - --long_cap 的 default 改为 32（v4 新默认值，比 v3 的 16 更大）
    """
    parser = argparse.ArgumentParser(
        description="LingBot-World Memory Enhancement Training v4 "
                    "(Stochastic N-clip + Context Drop-off + Short Clip-Level + Visual Fusion + Tier Emb)"
    )

    # ---- v3_dual 原有参数 ----
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="lingbot-world 预训练权重目录（含 low_noise_model）")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="CSGO 预处理数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="checkpoint 保存目录")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=0,
                        help="保留最近 N 个 epoch checkpoint，0=全部保留")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="每 N steps 保存一次（None = 只按 epoch 保存）")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank. 0 = full fine-tuning")
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数（code_standards.md §3 默认 4）")
    # code_standards.md §2：必须有 --resume
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 目录恢复训练（accelerator.load_state）")
    # code_standards.md §2：必须有 --dry_run
    parser.add_argument("--dry_run", action="store_true",
                        help="只跑 2 steps 验证训练流程")

    # ---- Memory Enhancement 参数（与 v3_dual 一致）----
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="训练阶段：1=只训 memory 模块；2=全参数解冻")
    parser.add_argument("--model_type", type=str, default="low", choices=["low", "high"],
                        help="模型类型：low=低噪声模型（t < 0.947）；high=高噪声模型（t >= 0.947）")
    parser.add_argument("--nfp_loss_weight", type=float, default=0.1,
                        help="NFP loss 权重（L_total = L_diffusion + w * L_nfp）")
    # PENDING[D-03]: Stage2 DiT 学习率，与 v3_dual 相同假设
    parser.add_argument("--lr_dit", type=float, default=1e-5,
                        help="Stage2 DiT 学习率（Stage1 时忽略）")
    parser.add_argument("--wandb_project", type=str, default="lingbot-memory",
                        help="W&B project 名称")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/团队名（可选）")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run 名称（默认自动生成）")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="W&B 模式：online/offline/disabled")
    parser.add_argument("--log_every_steps", type=int, default=10,
                        help="W&B 日志记录频率（每 N 步记录一次）")

    # ---- v4 新增：训练阶段选择（exp / full 数据集）----
    parser.add_argument("--phase", type=str, default="exp",
                        choices=["exp", "full"],
                        help="训练阶段：exp（ep01-11，小规模验证）/ full（ep01-46，全量训练）；"
                             "决定 CSV 路径：metadata_{phase}_{split}.csv")

    # ---- v4 新增：随机 N-clip 训练参数（Innovation 6）----
    # 注：v3 的 --num_context_clips 已移除，替换为 --max_context_clips
    parser.add_argument("--max_context_clips", type=int, default=6,
                        help="最大 context clip 数量（N ~ Uniform(2, max_context_clips)），"
                             "数据集 window_size = max_context_clips+1（Innovation 6）")

    # ---- v4 新增：Context Drop-off 参数（Innovation 7）----
    parser.add_argument("--context_drop_p_max", type=float, default=0.3,
                        help="retrieve 前随机丢弃 bank entries 的最大概率，"
                             "0.0=关闭（Innovation 7）")

    # ---- v4 新增：Visual Feature Fusion 参数（Innovation 9）----
    parser.add_argument("--visual_fusion_alpha", type=float, default=0.7,
                        help="Semantic Key 中 pose 权重（0=纯visual，1=纯pose，默认 0.7）"
                             "（Innovation 9）")

    # ---- ThreeTierMemoryBank 超参数（v4 更新：--long_cap default=32）----
    parser.add_argument("--short_cap", type=int, default=1,
                        help="ShortTermBank 容量（FIFO，默认 1）")
    parser.add_argument("--medium_cap", type=int, default=8,
                        help="MediumTermBank 容量（高 surprise 帧，默认 8）")
    parser.add_argument("--long_cap", type=int, default=32,
                        help="LongTermBank 容量（稳定场景帧，v4 默认 32，v3 为 16）")
    parser.add_argument("--surprise_threshold", type=float, default=0.4,
                        help="MediumTermBank 写入下限（surprise > threshold 时写入，默认 0.4）")
    parser.add_argument("--stability_threshold", type=float, default=0.2,
                        help="LongTermBank stable 写入上限（surprise < threshold，默认 0.2）")
    parser.add_argument("--novelty_threshold", type=float, default=0.7,
                        help="LongTermBank novelty 写入上限（max cosine_sim < threshold，默认 0.7）")
    parser.add_argument("--half_life", type=float, default=10.0,
                        help="MediumTermBank age decay 半衰期（单位 chunk，默认 10.0）")
    parser.add_argument("--hybrid_medium_k", type=int, default=3,
                        help="MediumTermBank 检索帧数（默认 3）")
    parser.add_argument("--hybrid_long_k", type=int, default=2,
                        help="LongTermBank 检索帧数（默认 2）")
    parser.add_argument("--dup_threshold", type=float, default=0.95,
                        help="Cross-tier dedup 阈值（pose_emb cosine_sim > 此值认为冗余，默认 0.95）")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    import accelerate
    from accelerate.utils import DataLoaderConfiguration

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    # ---- W&B 日志（可选）----
    wb_logger = None
    if args.wandb_mode != "disabled":
        try:
            from scripts.wandb_utils import WandBLogger
            wb_logger = WandBLogger(args, accelerator)
        except Exception as _wb_e:
            logging.warning("W&B init failed (non-fatal, training continues): %s", _wb_e)

    # 自动在 output_dir 下创建模型类型子目录
    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_noise_model")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Args: {args}")
        logging.info(
            f"v4 训练配置：max_context_clips={args.max_context_clips}, "
            f"window_size={args.max_context_clips + 1}, "
            f"context_drop_p_max={args.context_drop_p_max}, "
            f"visual_fusion_alpha={args.visual_fusion_alpha}, "
            f"ThreeTierMemoryBank(short={args.short_cap}, "
            f"medium={args.medium_cap}, long={args.long_cap})"
        )

    trainer = LingBotMemoryTrainer(args)
    model = trainer.load_models(accelerator.device)

    # ---- 参数冻结 / LoRA 设置 ----
    if args.lora_rank > 0:
        model = setup_lora(model, args.lora_rank, args.lora_target_modules)
        trainable_params = freeze_for_stage(model, args.stage, args.lora_rank)
    else:
        trainable_params = freeze_for_stage(model, args.stage, 0)

    # ---- 梯度检查点 ----
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # ---- 数据集（v4：CSGOMultiClipDataset 使用 max_context_clips + phase）----
    dataset = CSGOMultiClipDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        phase=args.phase,
        max_context_clips=args.max_context_clips,  # Innovation 6: 使用 max_context_clips
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_clip_collate_fn,
    )

    # ---- 优化器（v4 更新：Stage2 新增 visual_key_proj 到 memory param group）----
    if args.stage == 2:
        # PENDING[D-03]: Stage2 起点权重未定，与 v3_dual 相同处理方式
        memory_param_ids = set()
        for block in model.blocks:
            if hasattr(block, 'memory_cross_attn'):
                for p in block.memory_cross_attn.parameters():
                    memory_param_ids.add(id(p))
                if hasattr(block, 'memory_norm'):
                    for p in block.memory_norm.parameters():
                        memory_param_ids.add(id(p))
        for p in model.nfp_head.parameters():
            memory_param_ids.add(id(p))
        if hasattr(model, 'latent_proj'):
            for p in model.latent_proj.parameters():
                memory_param_ids.add(id(p))
        # Innovation 9: visual_key_proj 加入 memory param group（lr=1e-4，与 latent_proj 相同）
        if hasattr(model, 'visual_key_proj'):
            for p in model.visual_key_proj.parameters():
                memory_param_ids.add(id(p))

        memory_params = [p for p in model.parameters()
                         if p.requires_grad and id(p) in memory_param_ids]
        dit_params = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in memory_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": memory_params, "lr": args.learning_rate},
                {"params": dit_params, "lr": args.lr_dit},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        # Stage1：只优化 memory 模块参数
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * max(1, len(dataloader) // args.gradient_accumulation_steps),
        eta_min=1e-6,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # ---- Resume（code_standards.md §2：必须支持 --resume）----
    start_epoch = 0
    start_global_step = 0
    if args.resume:
        ckpt_file = os.path.join(args.resume, "diffusion_pytorch_model.bin")
        training_state_dir = os.path.join(args.resume, "training_state")
        metadata_file = os.path.join(args.resume, "training_metadata.json")

        if os.path.exists(ckpt_file):
            state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            unwrapped = accelerator.unwrap_model(model)
            try:
                import deepspeed
                with deepspeed.zero.GatheredParameters(
                    list(unwrapped.parameters()), modifier_rank=0
                ):
                    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
            except (ImportError, AttributeError):
                missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"Resume: missing keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                logging.warning(f"Resume: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
            logging.info(f"Resumed model weights from {ckpt_file}")

        if os.path.exists(training_state_dir):
            try:
                accelerator.load_state(training_state_dir)
                logging.info(f"Resumed optimizer/scheduler from {training_state_dir}")
            except Exception as e:
                logging.warning(
                    f"Skipping optimizer/scheduler state load from {training_state_dir} "
                    f"(ZeRO stage mismatch or missing files): {e}. "
                    "Model weights loaded. Optimizer/scheduler will start fresh."
                )

        if os.path.exists(metadata_file):
            import json
            with open(metadata_file) as f:
                meta = json.load(f)
            start_epoch = meta.get("epoch", 0) + 1
            start_global_step = meta.get("global_step", 0)
            logging.info(f"Resuming from epoch {start_epoch}, global_step {start_global_step}")

    # gate gradient hook（ZeRO-3 下 param.grad 永远 None，需在 reduce-scatter 前用 hook 捕获）
    _last_gate_grad: float = 0.0
    def _gate_grad_hook(grad):
        nonlocal _last_gate_grad
        if grad is not None:
            _last_gate_grad = grad.abs().item()
    accelerator.unwrap_model(model).blocks[0].memory_cross_attn.gate.register_hook(
        _gate_grad_hook
    )

    # ---- 训练循环 ----
    global_step = start_global_step
    try:
        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            gc.collect()
            torch.cuda.empty_cache()
            epoch_loss = 0.0
            num_batches = 0
            epoch_diffusion_loss = 0.0
            epoch_nfp_loss       = 0.0
            epoch_vis_loss       = 0.0
            # 诊断 log 变量（每 epoch 重置）
            _last_grad_norm: float = 0.0
            _last_gate_grad = 0.0  # epoch 开始重置；hook 在 backward 时更新

            progress = tqdm(
                dataloader,
                disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch+1}/{args.num_epochs} [v4, stage={args.stage}, {args.model_type}]",
            )

            for batch_clips in progress:
                # batch_clips: List[dict] of max_context_clips+1 clips（来自 multi_clip_collate_fn）
                # code_standards.md §2：OOM 防御（DDP-safe）
                # 先 forward 并同步 OOM 标志，确保所有 rank 一致决定跳过，避免 NCCL 死锁
                # ZeRO-3：所有 rank 必须执行相同次数的 model.forward()（等于 n_ctx 次 context clip）
                # 若各 rank 独立采样 n_ctx，forward 次数不同 → allgather 序列错位 → NCCL 死锁
                # 修复：rank 0 采样后 broadcast，所有 rank 使用同一 n_ctx
                _n_ctx_t = torch.zeros(1, dtype=torch.long, device=accelerator.device)
                if accelerator.is_main_process:
                    _n_ctx_t[0] = random.randint(2, args.max_context_clips)
                dist.broadcast(_n_ctx_t, src=0)
                _synced_n_ctx = int(_n_ctx_t.item())

                _skip = torch.zeros(1, device=accelerator.device)
                loss = None
                _loss_components: Dict[str, float] = {
                    "diffusion": 0.0, "nfp": 0.0, "vis_align": 0.0, "latent_proj": 0.0
                }
                try:
                    loss, _loss_components = multi_clip_training_step(
                        trainer,
                        accelerator.unwrap_model(model),
                        batch_clips,
                        args,
                        n_ctx=_synced_n_ctx,
                    )
                except torch.cuda.OutOfMemoryError:
                    try:
                        del loss
                    except NameError:
                        pass
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    _skip[0] = 1.0

                # 在任何 backward collective 之前同步：任一 rank OOM → 全部跳过
                if accelerator.num_processes > 1:
                    dist.all_reduce(_skip, op=dist.ReduceOp.MAX)

                if _skip.item() > 0:
                    logger.warning(
                        f"OOM at step {global_step}, batch_size=1. Skipping batch."
                    )
                    continue

                # backward OOM guard（DDP-safe，继承自 v3_dual 阶段十七）
                _back_skip = torch.zeros(1, device=accelerator.device)
                try:
                    with accelerator.accumulate(model):
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            _last_grad_norm = float(
                                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            )
                        optimizer.step()
                        if accelerator.sync_gradients:
                            lr_scheduler.step()
                        # W&B 步骤日志（梯度 norm 必须在 zero_grad 之前采集）
                        if wb_logger is not None and accelerator.sync_gradients:
                            # M-01 修复：loss/diffusion 报告纯 diffusion loss（而非 total_loss）
                            _loss_dict = {
                                "loss/total": loss.item(),
                                "loss/diffusion": _loss_components["diffusion"],
                                "loss/nfp": _loss_components["nfp"],
                                "loss/vis_align": _loss_components["vis_align"],
                                "loss/latent_proj": _loss_components["latent_proj"],
                            }
                            wb_logger.log_step(
                                global_step + 1,
                                _loss_dict,
                                model=accelerator.unwrap_model(model),
                                lr=lr_scheduler.get_last_lr()[0],
                            )
                        optimizer.zero_grad()
                except (torch.cuda.OutOfMemoryError, AssertionError) as _bwd_exc:
                    if isinstance(_bwd_exc, AssertionError) and "already been reduced" not in str(_bwd_exc):
                        raise
                    if loss is not None:
                        loss.detach_()
                    del loss
                    torch.cuda.synchronize()
                    optimizer.zero_grad(set_to_none=True)
                    _reset_deepspeed_zero_state(accelerator, optimizer)
                    torch.cuda.empty_cache()
                    gc.collect()
                    _back_skip[0] = 1.0

                if accelerator.num_processes > 1:
                    dist.all_reduce(_back_skip, op=dist.ReduceOp.MAX)

                if _back_skip.item() > 0:
                    logger.warning(
                        f"OOM (backward recompute) at step {global_step}. Skipping batch."
                    )
                    continue

                epoch_loss += loss.item()
                epoch_diffusion_loss += _loss_components["diffusion"]
                epoch_nfp_loss       += _loss_components["nfp"]
                epoch_vis_loss       += _loss_components["vis_align"]
                num_batches += 1
                global_step += 1

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    nfp=f"{_loss_components['nfp']:.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                if accelerator.is_main_process and global_step % 10 == 0:
                    # 修改1：多 block gate 值（blocks 0/10/20/39）
                    _unwrapped = accelerator.unwrap_model(model)
                    _gate_vals = [
                        _unwrapped.blocks[idx].memory_cross_attn._last_gate_value
                        for idx in (0, 10, 20, 39)
                    ]
                    # 修改2：attn_out_norm
                    _attn_norm = _unwrapped.blocks[0].memory_cross_attn._last_attn_out_norm
                    logger.info(
                        f"step {global_step} | n_ctx={_synced_n_ctx} | "
                        f"gates=[{_gate_vals[0]:.6f}, {_gate_vals[1]:.6f}, "
                        f"{_gate_vals[2]:.6f}, {_gate_vals[3]:.6f}] | "
                        f"attn_norm={_attn_norm:.4f} | "
                        f"gate_grad={_last_gate_grad:.2e} | "
                        f"grad_norm={_last_grad_norm:.3f} | "
                        f"bank=[{int(_loss_components.get('bank_short', 0))}s/"
                        f"{int(_loss_components.get('bank_medium', 0))}m/"
                        f"{int(_loss_components.get('bank_long', 0))}l/{args.long_cap}] "
                        f"retr={int(_loss_components.get('bank_retrieved_k', 0))} | "
                        f"loss={loss.item():.4f} "
                        f"(diff={_loss_components['diffusion']:.4f} "
                        f"nfp={_loss_components['nfp']:.4f} "
                        f"vis={_loss_components['vis_align']:.4f})"
                    )

                # 定期保存 checkpoint
                if args.save_steps and global_step % args.save_steps == 0:
                    save_checkpoint(
                        accelerator, model, args,
                        f"step_{global_step}", epoch=epoch, global_step=global_step
                    )

                # dry_run
                if args.dry_run and global_step >= 2:
                    logging.info("dry_run=True, stopping after 2 steps.")
                    break

            avg_loss = epoch_loss / max(num_batches, 1)
            if accelerator.is_main_process:
                logging.info(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"avg_loss={avg_loss:.4f} "
                    f"(diff={epoch_diffusion_loss/max(num_batches,1):.4f} "
                    f"nfp={epoch_nfp_loss/max(num_batches,1):.4f} "
                    f"vis={epoch_vis_loss/max(num_batches,1):.4f}) | "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e} | "
                    f"stage={args.stage}"
                )

            if (epoch + 1) % args.save_every_n_epochs == 0:
                save_checkpoint(
                    accelerator, model, args,
                    f"epoch_{epoch+1}", epoch=epoch, global_step=global_step
                )
                if accelerator.is_main_process:
                    prev_epoch_num = epoch + 1 - args.save_every_n_epochs
                    if prev_epoch_num >= 1:
                        prev_state_dir = os.path.join(
                            args.output_dir, f"epoch_{prev_epoch_num}", "training_state"
                        )
                        if os.path.isdir(prev_state_dir):
                            shutil.rmtree(prev_state_dir)
                            logging.info(f"Auto-deleted old training_state: {prev_state_dir}")
                    if args.keep_last_n_checkpoints > 0:
                        old_epoch_num = epoch + 1 - args.keep_last_n_checkpoints * args.save_every_n_epochs
                        if old_epoch_num >= 1:
                            old_ckpt_dir = os.path.join(args.output_dir, f"epoch_{old_epoch_num}")
                            if os.path.isdir(old_ckpt_dir):
                                shutil.rmtree(old_ckpt_dir)
                                logging.info(f"Auto-deleted old checkpoint: {old_ckpt_dir}")

            if args.dry_run:
                break

        if args.num_epochs % args.save_every_n_epochs != 0:
            save_checkpoint(
                accelerator, model, args,
                "final", epoch=args.num_epochs - 1, global_step=global_step
            )
        if accelerator.is_main_process:
            logging.info("Training complete!")
    except Exception as _exc:
        if wb_logger is not None:
            _slurm_log = f"slurm-{os.environ.get('SLURM_JOB_ID', 'local')}.out"
            wb_logger.log_crash(_exc, log_path=_slurm_log)
        raise
    finally:
        if wb_logger is not None:
            wb_logger.finish()


if __name__ == "__main__":
    main()
