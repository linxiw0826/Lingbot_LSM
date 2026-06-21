"""
train_v2_stage1_dual.py — LingBot-World Memory Enhancement 训练脚本 v2（双模型版）
======================================================================================
与 train_v2_stage1.py 的差异（仅 dual-model 相关）：
  1. 新增 --model_type low|high（从对应 low_noise_model 或 high_noise_model 子文件夹加载权重）
  2. FlowMatchingSchedule 新增 high_noise_indices（t >= 0.947）+ sample_timestep() 接受 model_type 参数
  3. load_models() 按 model_type 路由 subfolder
  4. training_step() 把 model_type 传给 sample_timestep()
  5. main() 自动在 output_dir 下创建 low_noise_model/ 或 high_noise_model/ 子目录

两个模型均使用 WanModelWithMemory + Stage1 freeze（冻结 DiT，只训 memory 模块）。
运行：先 --model_type low，完成后再 --model_type high（见 run_train_v2_dual.sh）。
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
from functools import wraps
from os.path import abspath, dirname, join
from typing import Optional, Tuple

import numpy as np
import torch
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
# 本文件位于 src/pipeline/v2/，向上三层到 Lingbot_LSM/，再进入 refs/lingbot-world
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
# Dataset（与 v2 完全对齐，直接复用）
# ============================================================

class CSGODataset(Dataset):
    """Loads preprocessed CSGO clips for LingBot training.

    完全复用 csgo-finetune-v2 的 CSGODataset，不做改动。
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
                "prompt": sample["prompt"],
                "poses": torch.from_numpy(poses).float(),
                "actions": torch.from_numpy(actions).float(),
                "intrinsics": torch.from_numpy(intrinsics).float(),
            }
        except Exception as e:
            # code_standards.md §3：跳过损坏样本
            logging.warning(f"Error loading sample {idx} ({clip_dir}): {e}")
            # 返回相邻样本（递归向后取，避免无限循环直接用 0 号）
            fallback_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(fallback_idx)

    def _pad_or_truncate(self, arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
        pad = np.tile(arr[-1:], rep_shape)
        return np.concatenate([arr, pad], axis=0)


# ============================================================
# FlowMatchingSchedule（从 v2 LingBotTrainer.__init__ 提取为独立类）
# ============================================================

class FlowMatchingSchedule:
    """预计算 Flow Matching sigma schedule 及训练权重。

    与 v2 LingBotTrainer.__init__ 完全对齐：
      - shift = 10.0
      - boundary = 0.947
      - num_train_timesteps = 1000
      - sigma 计算：shift * sigma_linear / (1 + (shift - 1) * sigma_linear)
      - 训练权重：高斯加权 exp(-2 * ((x - steps/2) / steps)^2)，归一化
      - valid_train_indices：timestep < 947 的索引

    Args:
        num_train_timesteps: sigma schedule 步数，默认 1000
        shift:               schedule shift 参数，LingBot 为 10.0
        boundary:            low_noise_model 有效范围上界（0.947）
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

        # sigma schedule（与 v2 完全对齐）
        sigmas_linear = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
        self.sigmas = shift * sigmas_linear / (1 + (shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * num_train_timesteps

        # 有效训练范围（low_noise_model 负责 t < 947）
        max_timestep = boundary * num_train_timesteps  # 947
        self.valid_train_indices = torch.where(self.timesteps_schedule < max_timestep)[0]

        # 高噪声模型训练范围（high_noise_model 负责 t >= 947）
        self.high_noise_indices = torch.where(self.timesteps_schedule >= max_timestep)[0]

        # 训练权重（高斯加权，以 t=500 为中心；与 v2 完全对齐）
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
        """从有效范围均匀随机采样一个时间步。

        Args:
            model_type: "low" → valid_train_indices（t < 0.947）；"high" → high_noise_indices（t >= 0.947）

        Returns:
            sigma:           float
            t:               Tensor scalar（传给模型的 t）
            training_weight: float
        """
        indices = self.high_noise_indices if model_type == "high" else self.valid_train_indices
        idx = indices[
            torch.randint(len(indices), (1,)).item()
        ].item()
        sigma = self.sigmas[idx].item()
        t = self.timesteps_schedule[idx]
        training_weight = self.training_weights[idx].item()
        return sigma, t, training_weight


# ============================================================
# LoRA 设置（与 v2 完全对齐，额外包含 memory 模块）
# ============================================================

def setup_lora(model, lora_rank: int, lora_target_modules: str = "") -> nn.Module:
    """Apply LoRA to the model.

    与 v2 完全对齐，额外包含 memory 模块的 Linear 层：
      memory_cross_attn.q / .k / .v / .o

    Args:
        model:               WanModelWithMemory 实例
        lora_rank:           LoRA rank（lora_alpha = lora_rank）
        lora_target_modules: 逗号分隔的模块名（空 = 自动检测）
    """
    from peft import LoraConfig, inject_adapter_in_model

    if lora_target_modules:
        target_modules = lora_target_modules.split(",")
    else:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # v2 原始模式匹配
                v2_patterns = [
                    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                    "ffn.0", "ffn.2",
                    "cam_injector_layer1", "cam_injector_layer2",
                    "cam_scale_layer", "cam_shift_layer",
                ]
                # Memory Enhancement：额外包含 memory_cross_attn 的线性层
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
# Stage1/Stage2 冻结策略（Memory Enhancement 新增）
# ============================================================

def freeze_for_stage(model: nn.Module, stage: int, lora_rank: int) -> list:
    """根据训练阶段设置参数冻结策略。

    Stage1：冻结所有参数，只解冻 memory 模块（memory_cross_attn + memory_norm + nfp_head）
    Stage2：全参数解冻

    Args:
        model:      WanModelWithMemory 实例
        stage:      1 或 2
        lora_rank:  LoRA rank（> 0 表示 LoRA 模式）

    Returns:
        trainable_params: 可训参数列表（传给 optimizer）
    """
    if stage == 1:
        # 冻结所有参数
        model.requires_grad_(False)

        # 只解冻 memory 模块
        num_unfrozen_blocks = 0
        for block in model.blocks:
            if hasattr(block, 'memory_cross_attn'):
                block.memory_cross_attn.requires_grad_(True)
                if hasattr(block, 'memory_norm'):
                    block.memory_norm.requires_grad_(True)
                num_unfrozen_blocks += 1

        # 解冻 NFPHead
        model.nfp_head.requires_grad_(True)

        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        # 解冻 latent_proj（VAE latent → 模型空间的线性投影，用于生成 visual_emb 作为 cross-attn V）
        if hasattr(model, 'latent_proj'):
            model.latent_proj.requires_grad_(True)
            logging.info("Stage1: 解冻 latent_proj (F-03/F5 fix)")

        # M-3 修复：LoRA 模式下额外解冻 LoRA adapter（lora_A/lora_B）
        # 避免 setup_lora() 注入的 adapter 被 model.requires_grad_(False) 误冻结
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
        # D-03 解除后需要修改：
        #   - 若选从 Stage1 checkpoint 恢复：main() 中 --resume 加载 Stage1 权重后再调用此函数
        #   - 若选对方提供 CSGO-DiT 权重：在 main() 的模型加载处替换 base_model，
        #     再 from_wan_model() + 加载 Stage1 memory 参数，然后调用此函数
        # 待修改位置：本处的注释及 main() 中 "# PENDING[D-03]" 标注的代码段

        # 全参解冻
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
# 梯度检查点（与 v2 完全对齐）
# ============================================================

def enable_gradient_checkpointing(model: nn.Module) -> int:
    """对每个 DiT block 启用梯度检查点（monkey-patch）。

    与 v2 完全对齐，自动查找 block container（尝试 blocks/layers/transformer_blocks），
    patch forward，use_reentrant=False。

    Returns:
        patched: 成功 patch 的 block 数量
    """
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

        def _make_ckpt_fn(fn):
            @wraps(fn)
            def _ckpt_forward(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict=None):
                # use_reentrant=False 不支持 kwargs，dit_cond_dict 作为位置参数传入
                return torch_checkpoint(
                    fn, x, e, seq_lens, grid_sizes, freqs,
                    context, context_lens, dit_cond_dict,
                    use_reentrant=False,
                )
            return _ckpt_forward

        block.forward = _make_ckpt_fn(orig_forward)
        patched += 1

    logging.info(f"Gradient checkpointing: patched {patched} DiT blocks")
    return patched


# ============================================================
# Trainer（主训练器）
# ============================================================

class LingBotMemoryTrainer:
    """Memory Enhancement 训练器。

    核心设计：在 v2 LingBotTrainer 基础上：
      1. 模型加载改为 WanModelWithMemory
      2. training_step 加入 NFP Loss（forward hook）
      3. memory_states=None（训练时不注入 memory bank）
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu")

        # Flow Matching Schedule（与 v2 完全对齐）
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.schedule = FlowMatchingSchedule(
            num_train_timesteps=1000,
            shift=10.0,
            boundary=0.947,
        )

        # T5 prompt cache：避免重复编码同一 prompt（per-prompt, stored on CPU）
        self._t5_cache: dict = {}

        # cam_utils 在 load_models 后可用
        self.cam_utils = {}

    def load_models(self, device: torch.device):
        """加载 WanModelWithMemory、VAE、T5。

        与 v2 load_models 对齐，模型加载改为 WanModelWithMemory.from_wan_model()。
        """
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

        # 根据 model_type 路由到对应子目录（低噪声 or 高噪声）
        _subfolder = "low_noise_model" if self.args.model_type == "low" else "high_noise_model"
        logging.info(f"Loading base WanModel ({_subfolder})...")
        base_wan_model = WanModel.from_pretrained(
            ckpt_dir,
            subfolder=_subfolder,
            torch_dtype=torch.bfloat16,
            control_type="act",
        )

        # 验证控制信号嵌入维度（act mode 应为 Linear(1792, 5120)）
        wancamctrl = base_wan_model.patch_embedding_wancamctrl
        logging.info(
            f"patch_embedding_wancamctrl: Linear({wancamctrl.in_features}, "
            f"{wancamctrl.out_features})"
        )

        logging.info("Converting to WanModelWithMemory...")
        # Memory Enhancement：转换为 WanModelWithMemory
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
        """video_tensor: [3, F, H, W] -> latent [16, lat_f, lat_h, lat_w]

        与 v2 完全对齐。
        """
        latent = self.vae.encode([video_tensor.to(self.device)])[0]
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def encode_text(self, prompt: str) -> list:
        """prompt string -> list of text embedding tensors

        与 v2 完全对齐；新增 CPU cache 避免重复编码。
        """
        if prompt in self._t5_cache:
            return [t.to(self.device) for t in self._t5_cache[prompt]]
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        self._t5_cache[prompt] = [t.cpu() for t in context]
        return [t.to(self.device) for t in self._t5_cache[prompt]]

    def prepare_y(self, video_tensor: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Prepare conditional input y (mask + first frame VAE encoding).

        与 v2 完全对齐。
        """
        lat_f, lat_h, lat_w = latent.shape[1], latent.shape[2], latent.shape[3]
        F_total = video_tensor.shape[1]
        h, w = video_tensor.shape[2], video_tensor.shape[3]

        first_frame = video_tensor[:, 0:1, :, :]  # [3, 1, H, W]
        zeros = torch.zeros(3, F_total - 1, h, w, device=video_tensor.device)
        vae_input = torch.concat([first_frame, zeros], dim=1)  # [3, F, H, W]
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
        """Prepare dit_cond_dict for act mode.

        与 v2 完全对齐：
          get_Ks_transformed → interpolate_camera_poses → compute_relative_poses
          → get_plucker_embeddings
          c1 = h // lat_h, c2 = w // lat_w（与 v2 一致）
          最终输出：{"c2ws_plucker_emb": tensor.chunk(1, dim=0)}，shape [1, 448, lat_f, lat_h, lat_w]
        """
        interpolate_camera_poses = self.cam_utils["interpolate_camera_poses"]
        compute_relative_poses = self.cam_utils["compute_relative_poses"]
        get_plucker_embeddings = self.cam_utils["get_plucker_embeddings"]
        get_Ks_transformed = self.cam_utils["get_Ks_transformed"]

        num_frames = poses.shape[0]

        # Transform intrinsics for current resolution（与 v2 完全对齐）
        Ks = get_Ks_transformed(
            intrinsics,
            height_org=480, width_org=832,
            height_resize=h, width_resize=w,
            height_final=h, width_final=w,
        )
        Ks_single = Ks[0]

        # Interpolate poses to latent temporal resolution（与 v2 完全对齐）
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

        # wasd action（与 v2 完全对齐）
        # processed_csgo_v3 使用 8ch action，截断到前 4 维（WASD）以兼容 v2 模型架构
        wasd = actions[::4, :4].to(self.device)
        if len(wasd) > len(c2ws_infer):
            wasd = wasd[:len(c2ws_infer)]
        elif len(wasd) < len(c2ws_infer):
            pad = wasd[-1:].repeat(len(c2ws_infer) - len(wasd), 1)
            wasd = torch.cat([wasd, pad], dim=0)

        # rays_d: only_rays_d=True → 3 channels
        c2ws_plucker_emb = get_plucker_embeddings(
            c2ws_infer, Ks_repeated, h, w, only_rays_d=True
        )  # [lat_f, h, w, 3]

        # c1 = h // lat_h, c2 = w // lat_w（与 v2 完全对齐）
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

        # wasd 空间折叠（与 v2 完全对齐）
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

        # rays_d + wasd → [1, 448, lat_f, lat_h, lat_w]（与 v2 完全对齐）
        c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_tensor], dim=1)

        dit_cond_dict = {
            "c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0),
        }
        return dit_cond_dict

    def training_step(
        self,
        model: nn.Module,
        batch: dict,
        nfp_loss_weight: float = 0.1,
    ) -> torch.Tensor:
        """Single training step with Flow Matching loss + NFP Loss.

        在 v2 training_step 基础上：
          1. memory_states=None（训练时不注入 memory bank）
          2. 增加 NFP loss（forward hook 注册在 model.blocks[-1]）
          3. total_loss = diffusion_loss + nfp_loss_weight * nfp_loss['total']

        NFP Loss hook 注册位置：model.blocks[-1]（最后一个 DiT block）
        """
        from memory_module.nfp_head import NFPHead

        video = batch["video"].to(self.device)
        prompt = batch["prompt"]
        poses = batch["poses"]
        actions = batch["actions"]
        intrinsics = batch["intrinsics"]

        h, w = video.shape[2], video.shape[3]

        with torch.no_grad():
            video_latent = self.encode_video(video)

        lat_f, lat_h, lat_w = (
            video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
        )
        seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        with torch.no_grad():
            context = self.encode_text(prompt)
            y = self.prepare_y(video, video_latent)

        dit_cond_dict = self.prepare_control_signal(
            poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
        )

        # Sample timestep from shifted schedule，按 model_type 路由噪声区间
        sigma, t, training_weight = self.schedule.sample_timestep(
            model_type=self.args.model_type
        )
        t = t.to(self.device).unsqueeze(0)

        # Flow Matching: add noise（与 v2 完全对齐）
        noise = torch.randn_like(video_latent)
        noisy_latent = (1.0 - sigma) * video_latent + sigma * noise

        # Target: velocity = noise - clean（与 v2 完全对齐）
        target = noise - video_latent

        noisy_latent = noisy_latent.requires_grad_(True)

        # ----------------------------------------------------------------
        # NFP Loss：forward hook 注册在 model.blocks[-1]
        # 捕获 hidden_states [B, L, dim] → nfp_head.forward → pred_latent
        # ----------------------------------------------------------------
        # 找到最后一个 block（可能是 MemoryBlockWrapper 或原始 WanAttentionBlock）
        last_block = model.blocks[-1]
        captured_hidden_states = {}

        def _nfp_hook(module, input, output):
            # output 是 [B, L, dim] tensor（WanAttentionBlock 返回 tensor）
            if isinstance(output, torch.Tensor):
                captured_hidden_states["hs"] = output
            elif isinstance(output, (list, tuple)):
                captured_hidden_states["hs"] = output[0]

        hook_handle = last_block.register_forward_hook(_nfp_hook)

        try:
            # Forward（训练时注入 dummy memory K + 真实 latent 投影 V，确保 MemoryCrossAttention 和 latent_proj 接收梯度）
            # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
            # latent_proj 需要在训练 forward 中被调用才有梯度路径。
            # 策略：dummy_memory_k（零初始化）作为 K；用真实 latent 投影的 visual_emb 作为 V。
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # K: dummy zero memory，shape [1, 1, dim=model.dim]
                dummy_memory_k = torch.zeros(
                    1, 1, model.dim,
                    device=noisy_latent.device,
                    dtype=torch.bfloat16,
                )
                # V: 真实 latent 的最后帧空间均值经 latent_proj 投影（确保 latent_proj 有梯度）
                # MODIFIED: F-03/F5 fix — 不用 no_grad，让 latent_proj 参与反传
                if hasattr(model, 'latent_proj'):
                    # video_latent: [z_dim=16, lat_f, lat_h, lat_w]，取最后帧空间均值 [16]
                    _v_feat = video_latent[:, -1].float().mean(dim=[-2, -1])  # [16]
                    _v_emb = model.latent_proj(_v_feat)                        # [5120]
                    dummy_memory_v = _v_emb.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)  # [1, 1, 5120]
                else:
                    dummy_memory_v = dummy_memory_k  # 退化路径（latent_proj 未加载）
                pred = model(
                    [noisy_latent],
                    t=t,
                    context=context,
                    seq_len=seq_len,
                    y=[y],
                    dit_cond_dict=dit_cond_dict,
                    memory_states=dummy_memory_k,       # K: dummy zero memory，激活 MemoryCrossAttention
                    memory_value_states=dummy_memory_v, # V: 真实 latent 投影，确保 latent_proj 梯度路径
                )[0]

            # Diffusion loss（与 v2 完全对齐：排除第一帧）
            # Keep pred_rest in its native bf16 dtype (from autocast model forward) so
            # the autograd graph stays in bf16, satisfying DeepSpeed bf16 engine backward.
            # target_rest has no requires_grad so casting it is safe.
            pred_rest = pred[:, 1:]
            target_rest = target[:, 1:]
            diffusion_loss = F.mse_loss(pred_rest, target_rest.to(pred_rest.dtype))
            diffusion_loss = diffusion_loss * training_weight

            # NFP Loss（Memory Enhancement 新增）
            if "hs" in captured_hidden_states and nfp_loss_weight > 0.0:
                hidden_states = captured_hidden_states["hs"]  # [B, L, dim]
                nfp_head = model.nfp_head
                pred_latent = nfp_head(hidden_states)  # [B, z_dim=16]

                # M-2 修复：用 clip 最后一帧空间均值替代全 clip 全局均值，
                # 更接近"预测下一帧"的语义（下一帧 ≈ clip 最后一帧的延续）
                # video_latent shape: [z_dim=16, lat_f, lat_h, lat_w]
                # Cast to pred_latent dtype (bf16) to match autograd graph dtype.
                actual_latent = video_latent[:, -1].mean(dim=[-2, -1]).unsqueeze(0).to(pred_latent.dtype)  # [1, z_dim=16]

                nfp_loss_dict = NFPHead.compute_loss(
                    pred_latent, actual_latent,
                    mse_weight=1.0, cosine_weight=1.0,
                )
                nfp_loss = nfp_loss_dict['total']
                total_loss = diffusion_loss + nfp_loss_weight * nfp_loss
            else:
                total_loss = diffusion_loss

        finally:
            # hook 在 try/finally 中移除（保证异常时也清理）
            hook_handle.remove()

        return total_loss


# ============================================================
# Checkpoint 保存（与 v2 完全对齐）
# ============================================================

def save_checkpoint(accelerator, model, args, tag: str, epoch: int = 0, global_step: int = 0):
    """保存 checkpoint。

    LoRA 模式：保存 lora_weights.pth（只保存 requires_grad 参数）
    全参模式：保存 diffusion_pytorch_model.bin（ZeRO-3 集体操作）+ training_state/

    与 v2 完全对齐。
    """
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

        # ZeRO-3：use DeepSpeed's native save_16bit_model
        model.save_16bit_model(save_dir, "diffusion_pytorch_model.bin")

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_config(save_dir)
            # 验证保存权重有效性
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

        # 保存完整训练状态（optimizer + scheduler），供续训使用
        accelerator.save_state(os.path.join(save_dir, "training_state"))
        # ZeRO-3 save_state() re-assembles all shards/optimizer moments; release
        # that residual memory before returning so the next training step does
        # not OOM (observed: GPUs nearly full at the first step of epoch N+1).
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
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    保留 v2 的所有参数，新增：--stage / --nfp_loss_weight / --lr_dit
    """
    parser = argparse.ArgumentParser(
        description="LingBot-World Memory Enhancement Training v2"
    )

    # ---- v2 原有参数 ----
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

    # ---- Memory Enhancement 新增参数 ----
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="训练阶段：1=只训 memory 模块；2=全参数解冻")
    parser.add_argument("--model_type", type=str, default="low", choices=["low", "high"],
                        help="模型类型：low=低噪声模型（t < 0.947）；high=高噪声模型（t >= 0.947）")
    parser.add_argument("--nfp_loss_weight", type=float, default=0.1,
                        help="NFP loss 权重（L_total = L_diffusion + w * L_nfp）")
    # PENDING[D-03]: Stage2 DiT 学习率，当前假设从预训练权重开始，设为 1e-5（lr/10）。
    # D-03 解除后若选择不同起点权重，可能需要调整此默认值。
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

    # ---- W&B 日志（可选，--wandb_mode=disabled 时跳过）----
    wb_logger = None
    if args.wandb_mode != "disabled":
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from pipeline.common.wandb_utils import WandBLogger
            wb_logger = WandBLogger(args, accelerator)
        except Exception as _wb_e:
            logging.warning("W&B init failed (non-fatal, training continues): %s", _wb_e)

    # 自动在 output_dir 下创建模型类型子目录（low_noise_model/ 或 high_noise_model/）
    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_noise_model")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Args: {args}")

    trainer = LingBotMemoryTrainer(args)
    model = trainer.load_models(accelerator.device)

    # ---- 参数冻结 / LoRA 设置 ----
    if args.lora_rank > 0:
        # LoRA 模式：先 setup_lora（注入 adapter），再 freeze_for_stage
        # Stage1 时 LoRA 参数也只在 memory 模块中存在（target_modules 包含 memory_cross_attn）
        model = setup_lora(model, args.lora_rank, args.lora_target_modules)
        trainable_params = freeze_for_stage(model, args.stage, args.lora_rank)
    else:
        # 全参模式：直接 freeze_for_stage
        trainable_params = freeze_for_stage(model, args.stage, 0)

    # ---- 梯度检查点（与 v2 完全对齐）----
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # ---- 数据集（与 v2 完全对齐）----
    dataset = CSGODataset(
        args.dataset_dir, split="train",
        num_frames=args.num_frames, height=args.height, width=args.width,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    # ---- 优化器（与 v2 对齐，但 Stage1 只优化 memory 模块参数）----
    if args.stage == 2:
        # PENDING[D-03]: Stage2 起点权重未定。
        # 当前假设：从 args.resume 指定的 Stage1 checkpoint 恢复（含 memory 模块权重），
        # DiT 参数使用 lingbot-world 预训练权重，lr_dit 通常为 learning_rate 的 1/10。
        # D-03 解除后若采用对方 CSGO-DiT 权重作为 Stage2 起点，需要在模型加载处替换 base_model。
        # 届时修改位置：load_models() 中的 WanModel.from_pretrained 调用处。
        from memory_module.model_with_memory import WanModelWithMemory

        # 区分 memory 模块参数 vs DiT 参数（用于 Stage2 差异化学习率）
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
        # N-02 修复：latent_proj 在 model 上直接挂载，不在 blocks 中，需单独收集
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
        # Stage1：只优化 memory 模块参数（与 AdamW+CosineAnnealingLR 对齐）
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
            missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"Resume: missing keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                logging.warning(f"Resume: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
            logging.info(f"Resumed model weights from {ckpt_file}")

        if os.path.exists(training_state_dir):
            accelerator.load_state(training_state_dir)
            logging.info(f"Resumed optimizer/scheduler from {training_state_dir}")

        if os.path.exists(metadata_file):
            import json
            with open(metadata_file) as f:
                meta = json.load(f)
            start_epoch = meta.get("epoch", 0) + 1  # +1 因为保存的是已完成的 epoch
            start_global_step = meta.get("global_step", 0)
            logging.info(f"Resuming from epoch {start_epoch}, global_step {start_global_step}")

    # ---- 训练循环 ----
    global_step = start_global_step
    try:
        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            # 清理上一轮 checkpoint save_state() 留下的内存碎片
            gc.collect()
            torch.cuda.empty_cache()
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(
                dataloader,
                disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch+1}/{args.num_epochs} [stage={args.stage}]",
            )

            for batch in progress:
                # code_standards.md §2：OOM 防御
                try:
                    with accelerator.accumulate(model):
                        loss = trainer.training_step(
                            accelerator.unwrap_model(model),
                            batch,
                            nfp_loss_weight=args.nfp_loss_weight,
                        )
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            # H-2 修复：使用 accelerator-prepared model.parameters() 替代 pre-prepare 的
                            # trainable_params 列表，避免 ZeRO-3 下参数指针已迁移导致梯度裁剪静默失效
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
                                global_step + 1,   # +1 因为 global_step 在 accumulate 块外才递增
                                _loss_dict,
                                model=accelerator.unwrap_model(model),
                                lr=lr_scheduler.get_last_lr()[0],
                            )
                        optimizer.zero_grad()

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"OOM at step {global_step}, batch_size=1. Skipping batch."
                    )
                    optimizer.zero_grad()
                    continue

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                # code_standards.md §2：定期保存 checkpoint
                if args.save_steps and global_step % args.save_steps == 0:
                    save_checkpoint(accelerator, model, args, f"step_{global_step}", epoch=epoch, global_step=global_step)

                # dry_run：只跑 2 steps（code_standards.md §2）
                if args.dry_run and global_step >= 2:
                    logging.info("dry_run=True, stopping after 2 steps.")
                    break

            avg_loss = epoch_loss / max(num_batches, 1)
            if accelerator.is_main_process:
                # code_standards.md §4：关键步骤 INFO 日志
                logging.info(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"avg_loss: {avg_loss:.4f} | "
                    f"lr: {lr_scheduler.get_last_lr()[0]:.2e} | "
                    f"stage: {args.stage}"
                )

            if (epoch + 1) % args.save_every_n_epochs == 0:
                save_checkpoint(accelerator, model, args, f"epoch_{epoch+1}", epoch=epoch, global_step=global_step)
                # 新 epoch checkpoint 保存成功后，自动删除上一个 epoch 的 training_state
                if accelerator.is_main_process:
                    prev_epoch_num = epoch + 1 - args.save_every_n_epochs
                    if prev_epoch_num >= 1:
                        prev_state_dir = os.path.join(args.output_dir, f"epoch_{prev_epoch_num}", "training_state")
                        if os.path.isdir(prev_state_dir):
                            shutil.rmtree(prev_state_dir)
                            logging.info(f"Auto-deleted old training_state: {prev_state_dir}")

            if args.dry_run:
                break

        if args.num_epochs % args.save_every_n_epochs != 0:
            save_checkpoint(accelerator, model, args, "final", epoch=args.num_epochs-1, global_step=global_step)
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
