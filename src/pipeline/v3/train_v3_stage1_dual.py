"""
train_v3_stage1_dual.py — LingBot-World Memory Enhancement 训练脚本 v3（多 clip 顺序训练）
==========================================================================================

与 train_v2_stage1_dual.py 的核心差异（Method B：多 clip 顺序训练）：
  1. CSGOMultiClipDataset：按 episode_id 分组，返回 N 个连续 clip 的列表
  2. multi_clip_training_step：
     - Context clips（前 N-1 个）：no_grad forward → 填充 ThreeTierMemoryBank
     - Target clip（最后 1 个）：有梯度 forward + backward → 计算 loss
  3. CLI 新增 ThreeTierMemoryBank 所有超参数（--short_cap / --medium_cap / ...）

训练流程：
  Context clips → 填充 ThreeTierMemoryBank（no_grad，不参与反传）
  Target clip  → 检索 bank，带 memory 做 forward，计算 diffusion loss + NFP loss

两个模型（--model_type low|high）均使用此训练流程，与 v2_dual 一样先跑 low 再跑 high。
v3 action 格式：action.npy [81, 4]（与 v2 相同，用前 4 维 WASD）。
"""

import argparse
import csv
import gc
import logging
import math
import os
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
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # src/pipeline/
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
# CSGOMultiClipDataset（v3 新增，多 clip 顺序训练）
# ============================================================

class CSGOMultiClipDataset(Dataset):
    """按 episode 分组，每次返回 N 个连续 clip 的数据列表。

    设计：
      - 读 metadata CSV，按 episode_id 列分组
      - 若无 episode_id 列，退化为用 clip_path 的父目录名
      - 每个 episode 中按 stem 列排序（若存在），否则按 clip_path 字母序
      - 返回 N 个连续 clip 的数据列表（每个元素与 CSGODataset.__getitem__ 返回格式相同）
      - 若某 episode clip 数 < num_context_clips+1，跳过该 episode

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
        num_context_clips: int = 1,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        repeat: int = 1,
    ):
        """
        Args:
            dataset_dir:       数据集根目录（含 metadata_{split}.csv）
            split:             "train" / "val"
            num_context_clips: context clip 数量（N-1），总 clip 数 = num_context_clips + 1
            num_frames:        每个 clip 的帧数（81）
            height:            视频高度
            width:             视频宽度
            repeat:            数据集重复次数（用于模拟更长的 epoch）
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.num_context_clips = num_context_clips
        self.num_clips = num_context_clips + 1   # 总 clips = context + target
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.repeat = repeat

        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        self._build_episode_groups(csv_path)
        logging.info(
            f"CSGOMultiClipDataset: {len(self.samples)} valid episode windows "
            f"(num_clips={self.num_clips}, repeat={repeat}) from {csv_path}"
        )

    def _build_episode_groups(self, csv_path: str):
        """从 CSV 中按 episode_id 分组，构建 (episode_id → sorted clips) 映射。

        退化规则：
          - 若 CSV 无 episode_id 列 → 用 clip_path 的父目录名作为 episode_id
          - 若 CSV 有 stem 列 → 按 stem 排序；否则按 clip_path 字母序
        """
        all_rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_episode_id = "episode_id" in fieldnames
            has_stem = "stem" in fieldnames
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

        # 对每个 episode 的 clips 排序
        for ep_id in episode_clips:
            clips = episode_clips[ep_id]
            if has_stem:
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
            # 使用滑动窗口，每隔 num_clips 步取一个窗口（不重叠）
            for start in range(0, n - self.num_clips + 1, self.num_clips):
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
        """返回 N 个连续 clip 的数据列表。

        每个元素与 CSGODataset.__getitem__ 返回格式相同：
          {"video": Tensor, "prompt": str, "poses": Tensor, "actions": Tensor,
           "intrinsics": Tensor, "clip_path": str}

        若加载失败，递归跳到下一个样本（code_standards.md §3）。
        """
        window = self.samples[idx % len(self.samples)]
        clips_data = []
        for row in window:
            clip_dir = os.path.join(self.dataset_dir, row["clip_path"])
            try:
                data = self._load_single_clip(clip_dir, row)
                clips_data.append(data)
            except Exception as e:
                logging.warning(
                    f"Error loading clip {clip_dir}: {e}. "
                    f"Falling back to next sample."
                )
                next_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(next_idx)
        return clips_data

    def _load_single_clip(self, clip_dir: str, row: dict) -> dict:
        """加载单个 clip，返回与 CSGODataset.__getitem__ 格式相同的 dict。"""
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
            "clip_path": row.get("clip_path", ""),
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
# FlowMatchingSchedule（直接复用 v2_dual）
# ============================================================

class FlowMatchingSchedule:
    """预计算 Flow Matching sigma schedule 及训练权重。

    与 v2_dual 完全对齐（shift=10.0, boundary=0.947）。
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
# LoRA 设置（直接复用 v2_dual）
# ============================================================

def setup_lora(model, lora_rank: int, lora_target_modules: str = "") -> nn.Module:
    """Apply LoRA to the model（与 v2_dual 完全对齐）。"""
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
# Stage1/Stage2 冻结策略（直接复用 v2_dual）
# ============================================================

def freeze_for_stage(model: nn.Module, stage: int, lora_rank: int) -> list:
    """根据训练阶段设置参数冻结策略（与 v2_dual 完全对齐）。"""
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
        # D-03 解除后需要修改（同 train_v2_stage1_dual.py）
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
# 梯度检查点（直接复用 v2_dual）
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
    """Memory Enhancement 训练器（v3 多 clip 版本）。

    核心差异（相对 v2_dual）：
      - training_step → multi_clip_training_step（使用 ThreeTierMemoryBank）
      - load_models / encode_video / encode_text / prepare_y / prepare_control_signal 与 v2_dual 完全复用
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
        """加载 WanModelWithMemory、VAE、T5（与 v2_dual 完全对齐）。"""
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
        """video_tensor: [3, F, H, W] -> latent [16, lat_f, lat_h, lat_w]（与 v2_dual 对齐）。"""
        latent = self.vae.encode([video_tensor.to(self.device)])[0]
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def encode_text(self, prompt: str) -> list:
        """prompt string -> list of text embedding tensors（与 v2_dual 对齐，含 CPU cache）。"""
        if prompt in self._t5_cache:
            return [t.to(self.device) for t in self._t5_cache[prompt]]
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        self._t5_cache[prompt] = [t.cpu() for t in context]
        return [t.to(self.device) for t in self._t5_cache[prompt]]

    def prepare_y(self, video_tensor: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Prepare conditional input y（与 v2_dual 完全对齐）。"""
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
        """Prepare dit_cond_dict（与 v2_dual 完全对齐）。

        v3 action 格式：[81, 4]（与 v2 相同，前 4 维 WASD）。
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

        # v3 action 格式：[81, 4]（前 4 维 WASD，与 v2 相同）
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
# Checkpoint 保存（直接复用 v2_dual）
# ============================================================

def save_checkpoint(accelerator, model, args, tag: str, epoch: int = 0, global_step: int = 0):
    """保存 checkpoint（与 v2_dual 完全对齐）。"""
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
# 多 clip 训练步骤（Method B 核心，v3 新增）
# ============================================================

def multi_clip_training_step(
    trainer: LingBotMemoryTrainer,
    model: nn.Module,
    clips_batch: List[dict],
    args,
) -> torch.Tensor:
    """多 clip 顺序训练步骤（Method B）。

    args:
        trainer:      LingBotMemoryTrainer 实例（提供 encode_video / encode_text / prepare_y / prepare_control_signal）
        model:        accelerator.unwrap_model(model)，WanModelWithMemory
        clips_batch:  List[dict]，N 个 clip 的数据（来自 multi_clip_collate_fn）
        args:         argparse.Namespace

    返回：
        total_loss:  Tensor scalar，用于 accelerator.backward(loss)

    训练流程：
        Context clips（前 N-1 个）：torch.no_grad() forward → 填充 ThreeTierMemoryBank
        Target clip（最后 1 个）：正常 forward + backward → 计算 loss
    """
    from memory_module.memory_bank import ThreeTierMemoryBank
    from memory_module.nfp_head import NFPHead

    device = trainer.device
    N = len(clips_batch)
    context_clips = clips_batch[:N - 1]
    target_clip = clips_batch[N - 1]

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
            # M-2 fix: encode_text 可能返回 float32，model forward 期望 bfloat16（与 target clip line 1084 对称）
            context = [c.to(torch.bfloat16) if hasattr(c, 'dtype') and c.dtype != torch.bfloat16 else c for c in context]
            y = trainer.prepare_y(video, video_latent)
            dit_cond_dict = trainer.prepare_control_signal(
                poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
            )

            # 3. 用 get_projected_frame_embs 计算 frame_embs: [lat_f, 5120]
            c2ws_emb_raw = dit_cond_dict["c2ws_plucker_emb"][0]  # [1, C, lat_f, lat_h, lat_w]
            frame_embs = model.get_projected_frame_embs(c2ws_emb_raw).detach()  # [lat_f, 5120]，显式 detach 确保不携带梯度（已在 no_grad 内，此为双重保障）

            # 4. bank 非空时先 retrieve，用于 context forward（若 bank 为空则用 dummy）
            query_emb_ctx = frame_embs[0]  # 第一帧 pose emb
            query_semantic_key_ctx = model.get_semantic_key(query_emb_ctx)

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
                # 无 memory 时用 dummy（与 v2 training_step 对齐）
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
            # NOTE: 使用 clip-level surprise（一个 clip 共享同一分数）与 infer_v3.py
            # _update_memory_bank_v3 的 NFPHead 主路径对齐（per-frame cosine 为推理降级路径）
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
                    # compute_surprise 返回 [B] tensor，取第一个元素作为 clip-level surprise
                    clip_surprise = float(surprise_scores[0].item())

            # 8. 用 latent_proj 计算 visual_embs（每帧）
            # 每帧取 video_latent[:, t_idx] 空间均值 → latent_proj
            # 用 clip-level surprise（所有帧共享同一分数）
            with torch.no_grad():
                for t_idx in range(lat_f):
                    pose_emb_t = frame_embs[t_idx]  # [5120]

                    if hasattr(model, 'latent_proj'):
                        latent_frame = video_latent[:, t_idx]  # [z_dim, lat_h, lat_w]
                        visual_emb_t = model.get_projected_latent_emb(latent_frame)  # [5120]
                    else:
                        visual_emb_t = None

                    # 9. 用 model.get_semantic_key(pose_emb_t) 计算 semantic_key
                    semantic_key_t = model.get_semantic_key(pose_emb_t)  # [5120]

                    # 10. bank.update 每帧（传入 semantic_key）
                    bank.update(
                        pose_emb=pose_emb_t,
                        latent=video_latent[:, t_idx],  # [z_dim, lat_h, lat_w]
                        surprise_score=clip_surprise,
                        timestep=t_idx,
                        visual_emb=visual_emb_t,
                        chunk_id=clip_idx,
                        semantic_key=semantic_key_t,
                    )

            # 11. 若不是最后一个 context clip：bank.increment_age()
            if clip_idx < len(context_clips) - 1:
                bank.increment_age()

    # ----------------------------------------------------------------
    # Target clip：有梯度，计算 loss
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

    dit_cond_dict = trainer.prepare_control_signal(
        poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
    )

    # 1. 用 get_projected_frame_embs 计算 query_emb（第一帧）
    with torch.no_grad():
        c2ws_emb_raw = dit_cond_dict["c2ws_plucker_emb"][0]  # [1, C, lat_f, lat_h, lat_w]
        frame_embs_target = model.get_projected_frame_embs(c2ws_emb_raw)  # [lat_f, 5120]
        query_emb = frame_embs_target[0]  # [5120]

        # 2. 用 model.get_semantic_key(query_emb) 计算 query_semantic_key
        query_semantic_key = model.get_semantic_key(query_emb)  # [5120]

    # 3. bank.retrieve（bank 可能包含 context clips 填充的帧）
    if bank.size() > 0:
        retrieved = bank.retrieve(
            query_emb,
            query_semantic_key=query_semantic_key,
            short_n=args.short_cap,
            medium_k=args.hybrid_medium_k,
            long_k=args.hybrid_long_k,
            device=device,
        )
    else:
        retrieved = None

    if retrieved is not None:
        key_states, value_states = retrieved
        memory_states = key_states.unsqueeze(0)        # [1, K, 5120]
        memory_value_states = value_states.unsqueeze(0)  # [1, K, 5120]
    else:
        # bank 为空（num_context_clips=0 情况，回退到 dummy，与 v2 对齐）
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

    # NFP hook 注册（与 v2_dual.training_step 完全对齐）
    last_block = model.blocks[-1]
    captured_hidden_states = {}

    def _nfp_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured_hidden_states["hs"] = output
        elif isinstance(output, (list, tuple)):
            captured_hidden_states["hs"] = output[0]

    hook_handle = last_block.register_forward_hook(_nfp_hook)

    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(
                [noisy_latent],
                t=t,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond_dict,
                memory_states=memory_states,
                memory_value_states=memory_value_states,
            )[0]

        # Diffusion loss（排除第一帧，与 v2_dual 完全对齐）
        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        diffusion_loss = F.mse_loss(pred_rest, target_rest.to(pred_rest.dtype))
        diffusion_loss = diffusion_loss * training_weight

        # NFP Loss（只在 target clip 计算，与 v2_dual 对齐）
        nfp_loss_weight = args.nfp_loss_weight
        if "hs" in captured_hidden_states and nfp_loss_weight > 0.0:
            hidden_states = captured_hidden_states["hs"]  # [B, L, 5120]
            nfp_head = model.nfp_head
            pred_latent = nfp_head(hidden_states)  # [B, 16]

            # NFP target：clip 最后帧空间均值（与 v2_dual M-2 修复对齐）
            actual_latent = video_latent[:, -1].mean(
                dim=[-2, -1]
            ).unsqueeze(0).to(pred_latent.dtype)  # [1, 16]

            nfp_loss_dict = NFPHead.compute_loss(
                pred_latent, actual_latent,
                mse_weight=1.0, cosine_weight=1.0,
            )
            nfp_loss = nfp_loss_dict['total']
            total_loss = diffusion_loss + nfp_loss_weight * nfp_loss
        else:
            total_loss = diffusion_loss

    finally:
        hook_handle.remove()

    return total_loss


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    在 v2_dual 参数基础上，新增 ThreeTierMemoryBank 所有超参数和 --num_context_clips。
    """
    parser = argparse.ArgumentParser(
        description="LingBot-World Memory Enhancement Training v3 (Multi-Clip, Method B)"
    )

    # ---- v2_dual 原有参数 ----
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

    # ---- Memory Enhancement 参数（与 v2_dual 一致）----
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="训练阶段：1=只训 memory 模块；2=全参数解冻")
    parser.add_argument("--model_type", type=str, default="low", choices=["low", "high"],
                        help="模型类型：low=低噪声模型（t < 0.947）；high=高噪声模型（t >= 0.947）")
    parser.add_argument("--nfp_loss_weight", type=float, default=0.1,
                        help="NFP loss 权重（L_total = L_diffusion + w * L_nfp）")
    # PENDING[D-03]: Stage2 DiT 学习率，与 v2_dual 相同假设
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

    # ---- v3 新增：多 clip 顺序训练参数 ----
    parser.add_argument("--num_context_clips", type=int, default=1,
                        help="Context clip 数量（N-1 个填充 bank，最后1个为 target）")

    # ---- v3 新增：ThreeTierMemoryBank 超参数 ----
    parser.add_argument("--short_cap", type=int, default=2,
                        help="ShortTermBank 容量（FIFO，默认 2）")
    parser.add_argument("--medium_cap", type=int, default=8,
                        help="MediumTermBank 容量（高 surprise 帧，默认 8）")
    parser.add_argument("--long_cap", type=int, default=16,
                        help="LongTermBank 容量（稳定场景帧，默认 16）")
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
            from pipeline.common.wandb_utils import WandBLogger
            wb_logger = WandBLogger(args, accelerator)
        except Exception as _wb_e:
            logging.warning("W&B init failed (non-fatal, training continues): %s", _wb_e)

    # 自动在 output_dir 下创建模型类型子目录
    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_noise_model")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Args: {args}")
        logging.info(
            f"v3 训练配置：num_context_clips={args.num_context_clips}, "
            f"num_total_clips={args.num_context_clips + 1}, "
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

    # ---- 数据集（v3 新增：CSGOMultiClipDataset）----
    dataset = CSGOMultiClipDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        num_context_clips=args.num_context_clips,
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

    # ---- 优化器（与 v2_dual 对齐）----
    if args.stage == 2:
        # PENDING[D-03]: Stage2 起点权重未定，与 v2_dual 相同处理方式
        from memory_module.model_with_memory import WanModelWithMemory

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
            # ZeRO-3 将参数分片存储，直接 load_state_dict 会写入错误分片；
            # GatheredParameters 临时汇聚所有 GPU 的分片 → 写入完整权重 → 重新分片。
            # ZeRO-2 下 GatheredParameters 是 no-op，兼容两种配置。
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
                # ZeRO-2 保存的 training_state 在 ZeRO-3 下文件格式不兼容，
                # DeepSpeed 找不到分片文件会触发 AssertionError。
                # 模型权重已在上方正确加载，optimizer 从零热身影响可忽略。
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

    # ---- 训练循环 ----
    global_step = start_global_step
    try:
        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            gc.collect()
            torch.cuda.empty_cache()
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(
                dataloader,
                disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch+1}/{args.num_epochs} [v3, stage={args.stage}, {args.model_type}]",
            )

            for clips_batch in progress:
                # clips_batch: List[dict] of N clips（来自 multi_clip_collate_fn）
                # code_standards.md §2：OOM 防御（DDP-safe）
                # 先 forward 并同步 OOM 标志，确保所有 rank 一致决定跳过，避免 NCCL 死锁
                _skip = torch.zeros(1, device=accelerator.device)
                loss = None
                try:
                    loss = multi_clip_training_step(
                        trainer,
                        accelerator.unwrap_model(model),
                        clips_batch,
                        args,
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

                # backward OOM guard（DDP-safe）
                # 注：不在此处调用 empty_cache()。d249643 曾加入该调用，041063d 证明它会
                # 把 PyTorch allocator cache 返还给 CUDA，导致 backward recompute 申请
                # 新的连续内存时触发 allocator fragmentation，造成系统性每 batch 必 OOM。
                # averaged_gradients 的清理已在 _reset_deepspeed_zero_state 中处理。
                # 全部 rank 在 gradient checkpoint recomputation 阶段同时 OOM →
                # 无 ALLREDUCE 参与，可安全捕获后 all_reduce 跳过标志。
                _back_skip = torch.zeros(1, device=accelerator.device)
                try:
                    with accelerator.accumulate(model):
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            # 与 v2_dual H-2 修复对齐：使用 accelerator-prepared model.parameters()
                            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        if accelerator.sync_gradients:
                            lr_scheduler.step()
                        # W&B 步骤日志（梯度 norm 必须在 zero_grad 之前采集）
                        if wb_logger is not None and accelerator.sync_gradients:
                            _loss_dict = {
                                "loss/total": loss.item(),
                                "loss/diffusion": loss.item(),
                            }
                            wb_logger.log_step(
                                global_step + 1,
                                _loss_dict,
                                model=accelerator.unwrap_model(model),
                                lr=lr_scheduler.get_last_lr()[0],
                            )
                        optimizer.zero_grad()
                except (torch.cuda.OutOfMemoryError, AssertionError) as _bwd_exc:
                    # AssertionError "already been reduced" = ZeRO micro_step_id 混乱导致
                    # double-reduce，与 OOM 同源（多次失败后状态机累积残留），统一做 ZeRO reset。
                    # 若是其他 AssertionError（代码 bug）则重新抛出。
                    if isinstance(_bwd_exc, AssertionError) and "already been reduced" not in str(_bwd_exc):
                        raise
                    if loss is not None:
                        loss.detach_()
                    del loss
                    torch.cuda.synchronize()  # 冲刷残留 CUDA 操作，避免 NCCL 与释放竞争
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
                num_batches += 1
                global_step += 1

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
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
                    f"avg_loss: {avg_loss:.4f} | "
                    f"lr: {lr_scheduler.get_last_lr()[0]:.2e} | "
                    f"stage: {args.stage}"
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
