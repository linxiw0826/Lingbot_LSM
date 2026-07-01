"""
train_v6.py — LingBot-World Memory Enhancement 训练脚本 v6（latent-concat + LoRA）
====================================================================================

experiment_design Step 45（S-V6，承 Step 44 latent-concat 多 clip 理想复验 GO 后才训）。
decisions.md 讨论 10：in-context KV（范式 A）判死 [[F-30]] → latent-concat 升为执行主线。

== 设计要点（为何 frame-dim + LoRA 而非 StoryMem 通道维 concat）==

  报告 `20260701_latentconcat_interface_modeA.md` §3 指出 StoryMem 式**通道维 concat**要扩
  `patch_embedding=Conv3d(in_dim=36,…)` 的输入通道（新权重随机初始化）+ 改 rope（negative-RoPE）
  = 动骨干输入层。而**帧维 anchor-concat 已零训练可用**（[[F-23]] 12/12 单 clip + v6 #1 多 clip
  理想复验），训练版只需在其上**加 LoRA 让冻结骨干更会用 anchored 记忆**，不扩 conv、不改 rope，
  最小侵入、直接长在已验证的零训练版上 = 「零训练 work 后的增强训练」。

== 与 v5 train_v5 的核心差异 ==

  1. **模型 = 原生 base WanModel（无 memory_encoder、无 in-context KV 注入）**：
     `WanModel.from_pretrained(ckpt_dir, subfolder='low_noise_model', ...)`，**骨干全冻**，
     再用 `attach_lora_to_backbone` 给各 block 的线性层（self_attn q/k/v/o + ffn；可选
     cross_attn / cam injector）挂 **rank-128 LoRA**。**唯一可训练 = LoRA 参数**（A/B），
     base 权重 requires_grad=False。开训前断言：可训练参数全部含 '.lora_'。

  2. **注入 = frame-dim anchor-concat（复用 stage1_upperbound 已验证机制）**：
     训练 forward 里把**一个 anchor 帧 latent** 拼到 query latent 的**时间维尾部**（msk=1 clean
     + anchor clean latent + anchor 位姿 plucker，相对 query[0] 重算）。直接复用
     stage1_upperbound 的 `_encode_anchor_latent`（VAE 编码单帧 anchor，no_grad，因 anchor 是
     clean 上下文）与 `_build_conditioning_with_anchor`（plucker 含 anchor 尾部 append）。
     **梯度走骨干原生 36 通道 i2v 条件路径 → LoRA 收梯度**（forward 本身不在 no_grad 里，
     只有 VAE encode anchor / query 那步 no_grad，与 v5 同口径）。

  3. **loss = 单步 flow-matching 重建（照搬 v5/v4 公式），只在 query（非 anchor）帧上算**：
     pred[:, 1:lat_f_query]（排除 query 首帧=i2v image 条件 + 排除 anchor 尾部帧）。
     对齐 stage1「丢弃 anchor 输出」的做法。

  4. **训练 anchor 自监督选取（避开 D-07 revisit 稀疏）**：anchor = **同一 episode 的一个较早
     帧**——从 target clip 之前的 context clips 里随机取一帧 VAE encode 作 anchor（StoryMem 式
     自监督，不依赖真重访对，数据充足）。context clips 整段早于 query clip → 天然满足「较早帧 +
     时间间隔」。见 `_select_anchor`。

  5. **prompt = target_clip["prompt"]（数据 prompt，与 v5 同源同口径）。**

  6. **save = 只存 LoRA 权重（小）**；eval 时复用 base ckpt 重建骨干 + 挂 LoRA + load。
     ZeRO-3 下用 deepspeed.zero.GatheredParameters 聚合分片（参照 v5 save_memory_encoder）。

  7. **R6 类探针（开训前强制，过不了 raise）**：
       (a) ON/OFF 差异：同一 query batch，跑 anchor 拼接 vs 无 anchor，query 帧输出 max abs diff
           > 阈值（即便 LoRA=0，anchor 仍经骨干原生 i2v 条件路径影响输出 → 注入机制非 no-op）。
       (b) 梯度流：一次 backward 后 **至少一个 LoRA 参数 grad 非零**（防 F-12 类零梯度）。
           注：LoRA 冷启动 B=0 → 首步 lora_A grad=0、lora_B grad≠0 是**预期**（非 bug），故判据
           是「至少一个 LoRA 参数非零梯度」。

== checkpoint 存取约定 ==
  保存：run_dir/checkpoints/<tag>/lora.pth = {name: cpu tensor}，name 形如
        'blocks.0.self_attn.q.lora_A.weight'（即 named_parameters 里 requires_grad 的全集，
        已确证全部含 '.lora_'）。同目录 training_metadata.json 存 epoch/global_step + LoRA 重建
        配置（lora_rank/lora_alpha/lora_targets），供 eval 以 metadata 为准重建。
  加载（eval / resume）：
        base = WanModel.from_pretrained(ckpt_dir, subfolder='low_noise_model', ...)
        attach_lora_to_backbone(base, rank, alpha, dropout, targets)  # 同 metadata
        sd = torch.load('lora.pth'); base.load_state_dict(sd, strict=False)  # 骨干来自 base

本地无 torch/CUDA 真跑不动；--help / py_compile 走通即可（真跑待服务器、且须 v6 #1 ideal 闸门 GO）。
"""

import argparse
import gc
import logging
import math
import os
import random
import sys
from functools import wraps
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# sys.path（与 v5 对齐）
# ---------------------------------------------------------------------------
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                            # src/
_PROJECT_ROOT = dirname(_SRC_DIR)                            # Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, 'refs', 'lingbot-world')

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

# ---- 复用 v4 脚手架（dataset / schedule / collate / deepspeed reset）----
from pipeline.v4.train_v4_stage1_dual import (  # noqa: E402
    CSGOMultiClipDataset,
    FlowMatchingSchedule,
    multi_clip_collate_fn,
    _reset_deepspeed_zero_state,
)
# ---- 复用 v5 trainer（encode_video / encode_text / prepare_y），override load_models ----
from pipeline.v5.train_v5 import LingBotMemoryTrainerV5  # noqa: E402
# ---- 复用 stage1_upperbound 已验证的帧维 anchor-concat 机制（[[F-23]]）----
from pipeline.eval.stage1_upperbound import (  # noqa: E402
    _encode_anchor_latent,
    _build_conditioning_with_anchor,
)

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# LoRA：手写低秩 adapter（W 冻结 + (alpha/r)·B·A，仅 A/B 可训）
# ============================================================

class LoRALinear(nn.Module):
    """包裹一个已存在的 nn.Linear：out = base(x) + scaling * B(A(dropout(x)))。

    - `base`：原线性层，**冻结**（requires_grad=False），权重不变。
    - `lora_A` (in→r) / `lora_B` (r→out)：唯一可训练参数。
    - **B 零初始化** → 开训瞬间 LoRA 贡献=0 → 等价原骨干（idea-sound「can't hurt, might
      help」：模型可忽略无用记忆，幅度从 0 起步不破坏冻结骨干）。
    - scaling = alpha / r。
    - dtype 跟随 base（bf16），保证 `base(x)+lora` 不发生 dtype 冲突；autocast(bf16) 下安全。

    LoRA 冷启动梯度提示：B=0 → 首步 d L/d A = scaling·Bᵀ·… = 0（lora_A 首步零梯度），
    d L/d B = scaling·A(x) ≠ 0（lora_B 首步非零梯度）。这是 LoRA 的**标准**行为，**不是**
    F-12 类零梯度 bug；A 在 B 更新后即获梯度。R6 PROBE B 因此判「至少一个 LoRA 参数非零梯度」。
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "LoRALinear 只能包裹 nn.Linear"
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        _dtype = base_linear.weight.dtype
        _device = base_linear.weight.device

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        # 标准 LoRA 初始化：A ~ kaiming, B = 0（B=0 → 起步等价骨干）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # dtype/device 对齐 base（trainable，ZeRO-3 bf16 下保持统一 dtype）
        self.lora_A.to(device=_device, dtype=_dtype)
        self.lora_B.to(device=_device, dtype=_dtype)
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return out + lora * self.scaling


# 各 target 组到 block 内子模块路径的映射（值 = (父模块属性链, 线性层属性名)）
def _iter_block_linear_targets(block: nn.Module, targets: List[str]):
    """生成 (owner_module, attr_name, full_suffix) 三元组，指向 block 内要挂 LoRA 的 nn.Linear。

    targets 可含：
      'self_attn'  → self_attn.{q,k,v,o}
      'ffn'        → ffn[0], ffn[2]（nn.Sequential 内两个 Linear）
      'cross_attn' → cross_attn.{q,k,v,o}
      'cam'        → cam_injector_layer1/2 + cam_scale_layer + cam_shift_layer
    """
    if "self_attn" in targets and hasattr(block, "self_attn"):
        sa = block.self_attn
        for name in ("q", "k", "v", "o"):
            if hasattr(sa, name) and isinstance(getattr(sa, name), nn.Linear):
                yield sa, name, f"self_attn.{name}"
    if "cross_attn" in targets and hasattr(block, "cross_attn"):
        ca = block.cross_attn
        for name in ("q", "k", "v", "o"):
            if hasattr(ca, name) and isinstance(getattr(ca, name), nn.Linear):
                yield ca, name, f"cross_attn.{name}"
    if "ffn" in targets and hasattr(block, "ffn"):
        ffn = block.ffn
        # nn.Sequential：遍历找 Linear（通常 index 0 与 2）
        for idx, sub in enumerate(ffn):
            if isinstance(sub, nn.Linear):
                yield ffn, idx, f"ffn.{idx}"
    if "cam" in targets:
        for name in ("cam_injector_layer1", "cam_injector_layer2",
                     "cam_scale_layer", "cam_shift_layer"):
            if hasattr(block, name) and isinstance(getattr(block, name), nn.Linear):
                yield block, name, name


def attach_lora_to_backbone(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    targets: List[str],
) -> Tuple[int, int]:
    """给 model.blocks 各 block 的目标线性层挂 LoRALinear（in-place 替换）。

    前置：调用方应先 model.requires_grad_(False) 冻结全骨干。本函数替换后，只有新建的
    lora_A/lora_B requires_grad=True。

    Returns:
        (n_layers_wrapped, n_lora_params)
    """
    block_container = _find_blocks(model)
    if block_container is None:
        raise RuntimeError("attach_lora_to_backbone: 找不到 DiT blocks 容器")

    n_wrapped = 0
    for block in block_container:
        # 先收集再替换（避免遍历时修改 ffn Sequential 的元素索引语义）
        replacements = list(_iter_block_linear_targets(block, targets))
        for owner, key, _suffix in replacements:
            if isinstance(key, int):       # nn.Sequential 用整数索引
                base_linear = owner[key]
                if isinstance(base_linear, LoRALinear):
                    continue
                owner[key] = LoRALinear(base_linear, rank, alpha, dropout)
            else:                           # 普通属性
                base_linear = getattr(owner, key)
                if isinstance(base_linear, LoRALinear):
                    continue
                setattr(owner, key, LoRALinear(base_linear, rank, alpha, dropout))
            n_wrapped += 1

    n_lora = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and ".lora_" in n)
    logging.info(
        "attach_lora_to_backbone: 挂 LoRA 到 %d 个线性层（rank=%d, alpha=%.1f, targets=%s），"
        "可训练 LoRA 参数 %.1fM。",
        n_wrapped, rank, alpha, ",".join(targets), n_lora / 1e6,
    )
    return n_wrapped, n_lora


def _find_blocks(model: nn.Module):
    for attr in ("blocks", "layers", "transformer_blocks"):
        if hasattr(model, attr):
            return getattr(model, attr)
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 10:
            return mod
    return None


# ============================================================
# v6 梯度检查点（强制全 block，use_reentrant=True + 输入 requires_grad_）
# ============================================================

def enable_gradient_checkpointing_v6(model: nn.Module) -> int:
    """对每个 backbone block 强制启用梯度检查点（重算激活、不驻留 → 省显存）。

    与 v5 版同构（去掉 memory stash，v6 无 in-context KV）：
      - **强制对每个 block patch**（不因冻结 block 无 requires_grad 而跳过：v6 block 内已挂
        LoRA → 实际有可训练参数，但用 v5 同款 reentrant 写法更稳）。
      - **use_reentrant=True**：躲开 ZeRO-3 的 check_recomputed_tensors_match（reentrant=False
        在 ZeRO-3 把参数重新分片成 shape[0] 时崩，见 v5 注释）。
      - reentrant 靠输入 grad_fn 建反传图；全冻骨干下 block 输入 x 无 requires_grad → 调
        checkpoint 前强制 x.requires_grad_(True)，否则不建图、LoRA 拿不到梯度（F-12 类）。
        梯度经重算 forward 连回 block 内 LoRA 的 lora_A/lora_B 参数。

    block.forward 签名（model.py:225）：
        (x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None)
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    block_container = _find_blocks(model)
    if block_container is None:
        logging.warning("v6 gradient checkpointing: 找不到 DiT blocks，跳过")
        return 0

    patched = 0
    for block in block_container:
        orig_forward = block.forward

        def _make_ckpt_fn(fn):
            _in_ckpt = [False]

            @wraps(fn)
            def _ckpt_forward(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict=None):
                if _in_ckpt[0]:
                    return fn(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict)

                def _run(x, e, seq_lens, grid_sizes, freqs,
                         context, context_lens, dit_cond_dict):
                    _in_ckpt[0] = True
                    try:
                        return fn(x, e, seq_lens, grid_sizes, freqs,
                                  context, context_lens, dit_cond_dict)
                    finally:
                        _in_ckpt[0] = False

                if (torch.is_grad_enabled() and isinstance(x, torch.Tensor)
                        and not x.requires_grad):
                    x = x.requires_grad_(True)

                return torch_checkpoint(
                    _run, x, e, seq_lens, grid_sizes, freqs,
                    context, context_lens, dit_cond_dict,
                    use_reentrant=True,
                )
            return _ckpt_forward

        block.forward = _make_ckpt_fn(orig_forward)
        patched += 1

    logging.info(
        "v6 gradient checkpointing: 强制 patch %d 个 block（use_reentrant=True + 输入 "
        "requires_grad_）。", patched,
    )
    return patched


# ============================================================
# v6 Trainer：复用 v5 的 VAE/T5/encode/prepare_y，模型换成 base WanModel + LoRA
# ============================================================

class LingBotTrainerV6(LingBotMemoryTrainerV5):
    """v6 训练器。继承 v5 trainer 复用 encode_video / encode_text / prepare_y /
    prepare_control_signal（公式不改），仅 override load_models：加载 base WanModel（无
    memory_encoder）、冻结骨干、挂 LoRA。
    """

    def load_models(self, device: torch.device):
        self.device = device
        ckpt_dir = self.args.ckpt_dir

        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.cam_utils import (
            interpolate_camera_poses, compute_relative_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )

        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

        # v6 只训 low_noise_model（等价 v4 TRAIN_HIGH=0 / v5）
        logging.info("Loading base WanModel (low_noise_model)...")
        model = WanModel.from_pretrained(
            ckpt_dir,
            subfolder="low_noise_model",
            torch_dtype=torch.bfloat16,
            control_type="act",
        )

        # ---- 冻结全骨干，再挂 LoRA（唯一可训练）----
        model.requires_grad_(False)
        targets = [t for t in self.args.lora_targets.split(",") if t]
        attach_lora_to_backbone(
            model,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            dropout=self.args.lora_dropout,
            targets=targets,
        )
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


# ============================================================
# anchor 自监督选取 + 单步 forward（frame-dim anchor-concat）
# ============================================================

def _select_anchor(
    trainer: LingBotTrainerV6,
    context_clips: List[dict],
    args,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """自监督 anchor 选取：从 target clip 之前的 context clips 里随机取 num_anchor_frames 个
    较早帧，VAE encode（no_grad）作 clean anchor latent + 取其绝对 c2w 位姿。

    StoryMem 式自监督（不依赖真重访对，避开 D-07 revisit 稀疏）：context clips 整段早于
    query clip → 天然满足「较早帧 + 跨 clip 时间间隔」。

    Returns:
        (anchor_latent [16, n_anchor, lat_h, lat_w] 或 None,
         anchor_poses_np [n_anchor, 4, 4] 或 None)
        无 context clip 时返回 (None, None)（退化为纯 i2v，正常）。
    """
    if not context_clips or args.num_anchor_frames <= 0:
        return None, None

    h, w = args.height, args.width
    latents: List[torch.Tensor] = []
    poses_np: List[np.ndarray] = []
    for _ in range(args.num_anchor_frames):
        clip = random.choice(context_clips)
        video = clip["video"].squeeze(0)          # [3, F, H, W] in [-1,1]
        poses = clip["poses"].squeeze(0)          # [F, 4, 4]
        n_frames = video.shape[1]
        fi = random.randint(0, n_frames - 1)
        anchor_chw = video[:, fi].detach().cpu().numpy()   # [3, H, W] in [-1,1]
        # 复用 stage1 已验证的单帧 anchor VAE 编码（no_grad；anchor 为 clean 上下文）
        anchor_lat = _encode_anchor_latent(trainer.vae, anchor_chw, h, w, device)  # [16,1,lat_h,lat_w]
        latents.append(anchor_lat)
        poses_np.append(poses[fi].detach().cpu().numpy().astype(np.float32))  # [4,4]

    anchor_latent = torch.cat(latents, dim=1)             # [16, n_anchor, lat_h, lat_w]
    anchor_poses = np.stack(poses_np, axis=0)             # [n_anchor, 4, 4]
    return anchor_latent, anchor_poses


def _forward_with_optional_anchor(
    trainer: LingBotTrainerV6,
    model: nn.Module,
    ctx: dict,
    anchor_latent: Optional[torch.Tensor],
    anchor_poses_np: Optional[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """组装 frame-dim anchor-concat 输入并跑一次骨干 forward（梯度走 LoRA）。

    anchor_latent / anchor_poses_np 同时为 None → off 臂（纯 query i2v，无 anchor）。

    Returns:
        (pred [16, lat_f_total, lat_h, lat_w], target_full [同 shape],
         lat_f_query, lat_f_total)
    """
    device = trainer.device
    video_latent = ctx["video_latent"]            # [16, lat_f_query, lat_h, lat_w]（query clean）
    lat_f_query = video_latent.shape[1]
    lat_h, lat_w = video_latent.shape[2], video_latent.shape[3]
    n_anchor = 0 if anchor_latent is None else int(anchor_latent.shape[1])
    lat_f_total = lat_f_query + n_anchor

    # ---- full clean latent（query 在前，anchor 尾部 append）----
    if n_anchor > 0:
        anchor_clean = anchor_latent.to(video_latent.dtype).to(device)
        full_clean = torch.cat([video_latent, anchor_clean], dim=1)
    else:
        full_clean = video_latent

    # ---- noise：query 段固定（ctx['query_noise']，供 ON/OFF 探针逐元素一致）；anchor 段独立 ----
    query_noise = ctx["query_noise"]
    if n_anchor > 0:
        anchor_noise = torch.randn(16, n_anchor, lat_h, lat_w,
                                   device=device, dtype=query_noise.dtype)
        noise_full = torch.cat([query_noise, anchor_noise], dim=1)
    else:
        noise_full = query_noise

    sigma = ctx["sigma"]
    noisy_latent = (1.0 - sigma) * full_clean + sigma * noise_full
    target_full = noise_full - full_clean          # FM 速度（noise - clean），与 v5 同口径

    # ---- y = [msk(4), cond_latent(16)]；anchor 槽 msk=1 clean + clean anchor latent ----
    y_query = ctx["y_query"]                        # [20, lat_f_query]（trainer.prepare_y 产出）
    if n_anchor > 0:
        msk_q = y_query[:4]
        ylat_q = y_query[4:]
        msk_anchor = torch.ones(4, n_anchor, lat_h, lat_w,
                                device=device, dtype=y_query.dtype)
        msk = torch.cat([msk_q, msk_anchor], dim=1)            # [4, lat_f_total]
        cond_latent = torch.cat([ylat_q, anchor_clean.to(y_query.dtype)], dim=1)  # [16, lat_f_total]
        y_full = torch.cat([msk, cond_latent], dim=0)          # [20, lat_f_total]
    else:
        y_full = y_query

    # ---- plucker（含 anchor 尾部 append；anchor 位姿相对当前 clip query[0] 重算）----
    plucker, _ = _build_conditioning_with_anchor(
        anchor_poses=anchor_poses_np,
        query_poses=ctx["poses_np"],
        query_actions=ctx["actions_np"],
        query_intrinsics=ctx["intrinsics_np"],
        h=ctx["h"], w=ctx["w"], lat_h=lat_h, lat_w=lat_w,
        control_type="act",
        param_dtype=ctx["bb_dtype"],
        patch_size=trainer.patch_size,
        device=device,
    )
    assert plucker.shape[2] == lat_f_total, (
        f"plucker 时间维 {plucker.shape[2]} != lat_f_total {lat_f_total}（anchor/query 错位）"
    )
    dit_cond_dict = {"c2ws_plucker_emb": plucker.chunk(1, dim=0)}

    seq_len = lat_f_total * lat_h * lat_w // (
        trainer.patch_size[1] * trainer.patch_size[2])

    # dtype 对齐（VAE float32 → 骨干 bf16），与 v5 同口径
    noisy_latent = noisy_latent.to(dtype=ctx["bb_dtype"])
    y_full = y_full.to(dtype=ctx["bb_dtype"])

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(
            [noisy_latent],
            t=ctx["t"],
            context=ctx["context"],
            seq_len=seq_len,
            y=[y_full],
            dit_cond_dict=dit_cond_dict,
        )[0]
    return pred, target_full, lat_f_query, lat_f_total


def _build_target_ctx(
    trainer: LingBotTrainerV6,
    model: nn.Module,
    target_clip: dict,
) -> dict:
    """从 target clip 准备一次 forward 共享的上下文（query latent / text / y / sigma / 噪声 等）。

    复用 trainer.encode_video / encode_text / prepare_y（公式不改）。query_noise 在此固定，
    供同一 batch 的 ON / OFF 探针逐元素一致。
    """
    device = trainer.device
    video = target_clip["video"].squeeze(0).to(device)      # [3, F, H, W]
    poses = target_clip["poses"].squeeze(0)                 # [F, 4, 4]
    actions = target_clip["actions"].squeeze(0)
    intrinsics = target_clip["intrinsics"].squeeze(0)
    prompt = target_clip["prompt"]
    h, w = video.shape[2], video.shape[3]

    with torch.no_grad():
        video_latent = trainer.encode_video(video)          # [16, lat_f, lat_h, lat_w]
        context = trainer.encode_text(prompt)
        context = [c.to(torch.bfloat16) if hasattr(c, "dtype") and c.dtype != torch.bfloat16
                   else c for c in context]
        y_query = trainer.prepare_y(video, video_latent)    # [20, lat_f]

    sigma, t, training_weight = trainer.schedule.sample_timestep(model_type="low")
    t = t.to(device).unsqueeze(0)
    query_noise = torch.randn_like(video_latent)
    bb_dtype = model.patch_embedding.weight.dtype

    return {
        "video_latent": video_latent,
        "y_query": y_query,
        "context": context,
        "poses_np": poses.detach().cpu().numpy().astype(np.float32),
        "actions_np": actions.detach().cpu().numpy().astype(np.float32),
        "intrinsics_np": intrinsics.detach().cpu().numpy().astype(np.float32),
        "h": h, "w": w,
        "sigma": sigma, "t": t, "training_weight": training_weight,
        "query_noise": query_noise,
        "bb_dtype": bb_dtype,
    }


def multi_clip_training_step_v6(
    trainer: LingBotTrainerV6,
    model: nn.Module,
    batch_clips: List[dict],
    args,
    n_ctx: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """v6 单步训练：自监督 anchor frame-dim concat + 单步 FM 重建（只在 query 帧算 loss）。"""
    context_clips = batch_clips[:n_ctx]
    target_clip = batch_clips[n_ctx]

    anchor_latent, anchor_poses_np = _select_anchor(trainer, context_clips, args, trainer.device)
    ctx = _build_target_ctx(trainer, model, target_clip)

    pred, target_full, lat_f_query, lat_f_total = _forward_with_optional_anchor(
        trainer, model, ctx, anchor_latent, anchor_poses_np
    )

    # loss mask：排除 query 首帧（i2v image 条件）+ 排除 anchor 尾部帧（clean 上下文，不进 loss）
    pred_q = pred[:, 1:lat_f_query]
    target_q = target_full[:, 1:lat_f_query]
    diffusion_loss = F.mse_loss(pred_q, target_q.to(pred_q.dtype))
    total_loss = diffusion_loss * ctx["training_weight"]

    n_anchor = 0 if anchor_latent is None else int(anchor_latent.shape[1])
    loss_components = {
        "diffusion": float(total_loss.item()),
        "n_anchor": float(n_anchor),
        "lat_f_query": float(lat_f_query),
    }
    return total_loss, loss_components


# ============================================================
# Checkpoint：只存 LoRA 可训练权重
# ============================================================

def save_lora(accelerator, model, run_dir, tag: str,
              epoch: int = 0, global_step: int = 0,
              lora_rank: Optional[int] = None,
              lora_alpha: Optional[float] = None,
              lora_dropout: Optional[float] = None,
              lora_targets: Optional[str] = None) -> None:
    """只保存 LoRA 可训练权重（骨干冻结，eval 复用 base ckpt 重建后挂 LoRA + load）。

    保存到 run_dir/checkpoints/<tag>/lora.pth（dict: name -> cpu tensor），name 形如
    'blocks.0.self_attn.q.lora_A.weight'。ZeRO-3 下用 GatheredParameters 聚合分片。
    同目录 training_metadata.json 存 epoch/global_step + LoRA 重建配置（eval 以此为准）。
    """
    save_dir = os.path.join(str(run_dir), "checkpoints", tag)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    trainable = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]

    state = None
    try:
        import deepspeed
        with deepspeed.zero.GatheredParameters([p for _, p in trainable], modifier_rank=0):
            if accelerator.is_main_process:
                state = {n: p.data.detach().cpu().clone() for n, p in trainable}
    except (ImportError, AttributeError):
        if accelerator.is_main_process:
            state = {n: p.data.detach().cpu().clone() for n, p in trainable}

    if accelerator.is_main_process:
        bad = [n for n in state if ".lora_" not in n]
        assert not bad, f"save_lora: 发现非 LoRA 的可训练参数！keys={bad[:10]}"
        out_path = os.path.join(save_dir, "lora.pth")
        torch.save(state, out_path)
        import json
        with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
            json.dump({
                "epoch": epoch, "global_step": global_step,
                "n_params": len(state),
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_targets": lora_targets,
            }, f)
        logging.info("Saved LoRA checkpoint (%d tensors) -> %s", len(state), out_path)

    accelerator.wait_for_everyone()


# ============================================================
# 训练健康诊断可复用测量（PROBE A 核心 + LoRA 范数）
# ============================================================

def _measure_anchor_onoff_diff(
    trainer: LingBotTrainerV6,
    model: nn.Module,
    ctx: dict,
    anchor_latent: torch.Tensor,
    anchor_poses_np: np.ndarray,
) -> float:
    """PROBE A 核心（可复用）：同一 ctx（query_noise 已固定 → 两臂逐元素一致）跑
    anchor frame-dim concat（ON）vs 无 anchor（OFF），返回 query 帧输出的 max abs diff。

    **旁路 no_grad 测量**：只读输出、不建反传图、不进 loss/优化器 → 不污染梯度、
    不改训练数值行为。run_r6_probes_v6 的 PROBE A 与每-epoch 诊断共用此函数（不重写）。
    """
    with torch.no_grad():
        pred_on, _, lat_f_query, _ = _forward_with_optional_anchor(
            trainer, model, ctx, anchor_latent, anchor_poses_np)
        pred_off, _, _, _ = _forward_with_optional_anchor(
            trainer, model, ctx, None, None)
    return (pred_on[:, :lat_f_query].float() - pred_off.float()).abs().max().item()


def _measure_lora_norms(model: nn.Module, accelerator) -> Dict[str, float]:
    """聚合所有 LoRA 层的 ‖A‖ / ‖B‖ 总范数（旁路测量，只读权重不改任何状态）。

    B 零初始化 → ‖B‖ 长期≈0 表示 LoRA 没学到东西（等价没训）；‖A‖ 反映 A 的漂移。

    **ZeRO-3 正确聚合**：分片模式下 p.data 只是本 rank 的分片（partition），直接算范数会得到
    size-0 的假 0 / 偏小值。故用 `deepspeed.zero.GatheredParameters` 在 with 块内把各 LoRA
    参数聚合成完整张量后再算范数（与 save_lora 同一思路）。GatheredParameters 是集合通信 →
    必须由**所有 rank**进入（本函数在所有 rank 上被调用）。非 ZeRO-3 / 无 deepspeed 时参数本就
    完整，直接算。范数按参数平方和的平方根聚合（等价把所有 LoRA A/B 拼成一个向量取 L2 范数）。
    """
    unwrapped = accelerator.unwrap_model(model)
    lora_named = [(n, p) for n, p in unwrapped.named_parameters()
                  if p.requires_grad and ".lora_" in n]
    a_named = [(n, p) for n, p in lora_named if ".lora_A." in n]
    b_named = [(n, p) for n, p in lora_named if ".lora_B." in n]

    def _agg_norm(named):
        total_sq = 0.0
        for _, p in named:
            d = p.data
            if d is None or d.numel() == 0:
                continue
            total_sq += float(d.detach().float().norm().item()) ** 2
        return total_sq ** 0.5

    try:
        import deepspeed
        with deepspeed.zero.GatheredParameters([p for _, p in lora_named], modifier_rank=0):
            a_norm = _agg_norm(a_named)
            b_norm = _agg_norm(b_named)
    except (ImportError, AttributeError):
        a_norm = _agg_norm(a_named)
        b_norm = _agg_norm(b_named)

    return {"diag/lora_A_norm": float(a_norm), "diag/lora_B_norm": float(b_norm)}


def run_epoch_diagnostics(
    trainer: LingBotTrainerV6,
    model: nn.Module,
    diag_batch: List[dict],
    args,
    accelerator,
    wb_logger,
    epoch: int,
    global_step: int,
    last_grad_norm: float,
) -> None:
    """每-epoch（或每 N epoch）训练健康诊断，防止「跑完才发现白训」。

    记录（logger.info + W&B，仅主进程 log）：
      - diag/lora_A_norm, diag/lora_B_norm：LoRA 权重范数增长（B 从 0 长出 = 真在学）
      - diag/anchor_onoff_diff：复用 PROBE A，看注入是否**持续**有效（不退化成 no-op）
      - diag/grad_norm：最近一次同步步的梯度范数趋势（_last_grad_norm）

    **不改训练数值行为**：anchor ON/OFF 用 no_grad 旁路测量（不建图、不进 loss/优化器）；
    LoRA 范数只读权重。测量前后临时 eval() → 恢复原 train/eval 状态。

    **WARN1（防分布式集合通信不对齐 hang）**：整个诊断主体（_measure_lora_norms +
    anchor ON/OFF forward）用统一 try/except 只 warn 包裹，任何失败都不中断训练、不 re-raise。
    集合通信段（GatheredParameters / ZeRO-3 forward）严格「全 rank 一起做、要么全 rank 一起跳」：
      - _measure_lora_norms 的 GatheredParameters 只遍历 named_parameters（各 rank 结构一致、
        不因单 rank 数据异常而只在部分 rank 抛）→ 全 rank 直接进。
      - anchor 集合 forward 前的准备（_select_anchor random / _build_target_ctx）是**非集合**且
        可能因数据在部分 rank 失败 → 各 rank 先本地 try 准备，再用 dist.all_reduce(MIN) 广播
        prep_ok；任一 rank 准备失败即全 rank 一致跳过后续集合 forward，绝不只部分 rank 进。

    **WARN2（防诊断消耗全局 RNG 扰动主训练随机序列）**：进入时快照 torch(CPU)+cuda(all
    devices)+python random(+numpy) 的全局 RNG，finally 里恢复 → 诊断的 _select_anchor(random)/
    sample_timestep/torch.randn 推进的 RNG 不影响后续 epoch，diag-on/off 训练轨迹 bitwise 一致。
    """
    was_training = model.training
    model.eval()

    # ---- WARN2: 快照全局 RNG（进入即存，finally 恢复；get_* 均只读、不推进 RNG）----
    _cpu_rng_state = torch.get_rng_state()
    _cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    _py_rng_state = random.getstate()
    _np_rng_state = np.random.get_state()

    def _all_rank_agree(local_ok: bool) -> bool:
        """全 rank 一致的 ok 标志：dist.all_reduce(MIN) → 任一 rank 失败则全 rank 返回 False，
        保证后续集合 op「全 rank 一起进 / 全 rank 一起跳」（防 ZeRO-3 集合通信不对齐 hang）。
        单进程或 dist 未初始化时直接返回本地标志（无集合可对齐）。"""
        if accelerator.num_processes <= 1 or not (
            dist.is_available() and dist.is_initialized()
        ):
            return local_ok
        flag = torch.tensor(
            [1 if local_ok else 0], device=trainer.device, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() > 0)

    diff = None
    try:
        metrics: Dict[str, float] = {}

        # 1. LoRA 权重范数（GatheredParameters 集合通信；结构对称 → 全 rank 一起进）
        metrics.update(_measure_lora_norms(model, accelerator))

        # 2. anchor ON/OFF diff（复用 PROBE A 核心；ZeRO-3 集合 forward，须全 rank 对称进入）
        #    准备段（非集合、可能因数据在部分 rank 失败）先本地 try，再全 rank 对齐 prep_ok。
        unwrapped = accelerator.unwrap_model(model)
        anchor_latent = None
        anchor_poses_np = None
        ctx = None
        prep_ok = True
        try:
            n_ctx = max(1, len(diag_batch) - 1)
            context_clips = diag_batch[:n_ctx]
            target_clip = diag_batch[n_ctx]
            anchor_latent, anchor_poses_np = _select_anchor(
                trainer, context_clips, args, trainer.device)
            if anchor_latent is not None:
                ctx = _build_target_ctx(trainer, unwrapped, target_clip)
            else:
                prep_ok = False  # 无 anchor → 无集合 forward 可跑（diag_batch 全 rank 相同）
        except Exception as _prep_e:  # 非集合准备失败：本地记标志，随后全 rank 一致跳过集合段
            prep_ok = False
            logging.warning(
                "epoch diag: anchor 准备失败（非致命，全 rank 一致跳过集合 forward）：%s",
                _prep_e,
            )

        # 全 rank 对齐：任一 rank 准备失败 → 全 rank 一起跳过集合 forward（绝不只部分 rank 进）
        if _all_rank_agree(prep_ok):
            diff = _measure_anchor_onoff_diff(
                trainer, unwrapped, ctx, anchor_latent, anchor_poses_np)
            metrics["diag/anchor_onoff_diff"] = float(diff)

        # 3. grad_norm 趋势（每步已入 W&B 的 train/grad_norm；此处再落 epoch 粒度 diag/grad_norm）
        metrics["diag/grad_norm"] = float(last_grad_norm)

        if accelerator.is_main_process:
            logging.info(
                "[DIAG] epoch %d | lora_A_norm=%.4e | lora_B_norm=%.4e | "
                "anchor_onoff_diff=%s | grad_norm=%.3e",
                epoch + 1,
                metrics.get("diag/lora_A_norm", float("nan")),
                metrics.get("diag/lora_B_norm", float("nan")),
                ("%.3e" % diff) if diff is not None else "n/a",
                metrics["diag/grad_norm"],
            )
            if wb_logger is not None and getattr(wb_logger, "enabled", False):
                try:
                    import wandb
                    wandb.log(metrics, step=global_step)
                except Exception as _we:
                    logging.warning("epoch diag W&B log failed (non-fatal): %s", _we)
    except Exception as _diag_e:  # WARN1: 诊断整体非致命，只 warn，绝不中断训练、绝不 re-raise
        logging.warning("epoch diag 整体失败（非致命，训练继续）：%s", _diag_e)
    finally:
        if was_training:
            model.train()
        # ---- WARN2: 恢复全局 RNG（CPU + CUDA all devices + python random + numpy）----
        torch.set_rng_state(_cpu_rng_state)
        if _cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(_cuda_rng_states)
        random.setstate(_py_rng_state)
        np.random.set_state(_np_rng_state)
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()


# ============================================================
# R6 探针（开训前强制，过不了 raise）
# ============================================================

def run_r6_probes_v6(
    trainer: LingBotTrainerV6,
    model: nn.Module,
    batch_clips: List[dict],
    args,
    accelerator,
) -> None:
    """R6 两探针（训练循环之前调用）。任一不过即 raise RuntimeError 退出。

    探针 A（ON/OFF 差异）：同一 target batch，跑 anchor frame-dim concat vs 无 anchor，
        query 帧输出 max abs diff 必须 > args.probe_onoff_thresh，否则注入是 no-op。
        （注：即便 LoRA=0，anchor 经骨干原生 36 通道 i2v 条件路径仍影响输出 → 应非 no-op。）
    探针 B（梯度流）：对 ON 输出做一次 backward，**至少一个 LoRA 参数** grad 非零（防 F-12）。
        LoRA 冷启动 B=0 → 首步 lora_A grad=0、lora_B grad≠0 为预期，故判「至少一个非零」。
    """
    unwrapped = accelerator.unwrap_model(model)
    device = trainer.device

    logging.info("=" * 72)
    logging.info("R6 PROBES (v6 latent-concat + LoRA；强制；过不了即退出)")
    logging.info("=" * 72)

    n_ctx = max(1, len(batch_clips) - 1)
    context_clips = batch_clips[:n_ctx]
    target_clip = batch_clips[n_ctx]

    anchor_latent, anchor_poses_np = _select_anchor(trainer, context_clips, args, device)
    if anchor_latent is None:
        raise RuntimeError(
            "R6 探针：未能选出 anchor（context_clips 为空或 num_anchor_frames<=0）。"
            "无法验证注入；请检查 max_context_clips / num_anchor_frames。"
        )
    logging.info("R6: selected %d anchor frame(s) for probe.", anchor_latent.shape[1])

    ctx = _build_target_ctx(trainer, unwrapped, target_clip)

    # ---- 探针 A：ON vs OFF（复用 _measure_anchor_onoff_diff；query_noise 由 ctx 固定，两臂一致）----
    diff = _measure_anchor_onoff_diff(
        trainer, unwrapped, ctx, anchor_latent, anchor_poses_np)
    logging.info("R6 PROBE A (ON/OFF): max_abs_diff=%.3e (threshold=%.3e)",
                 diff, args.probe_onoff_thresh)
    if diff <= args.probe_onoff_thresh:
        raise RuntimeError(
            f"R6 PROBE A 失败：anchor 注入是 no-op（max_abs_diff={diff:.3e} "
            f"<= {args.probe_onoff_thresh:.3e}）。frame-dim concat 未影响输出，退出。"
        )
    logging.info("R6 PROBE A 通过：anchor 注入对输出有影响。")

    # ---- 探针 B：梯度流（LoRA 非零梯度）。本前向**不在** no_grad 里（必须建图供 backward）----
    lora_params = [(n, p) for n, p in unwrapped.named_parameters()
                   if p.requires_grad and ".lora_" in n]
    for _, p in lora_params:
        p.grad = None

    pred_grad, target_grad, lat_f_query, _ = _forward_with_optional_anchor(
        trainer, unwrapped, ctx, anchor_latent, anchor_poses_np)
    # loss 只在 query 帧（与训练同 mask），保证梯度路径与训练一致
    probe_loss = F.mse_loss(
        pred_grad[:, 1:lat_f_query],
        target_grad[:, 1:lat_f_query].to(pred_grad.dtype),
    )
    accelerator.backward(probe_loss)

    # ZeRO-3：优先 deepspeed safe_get_full_grad 读完整梯度（p.grad 常为 None/0 → 误判）
    _get_grad = None
    _grad_reader = "p.grad"
    try:
        from deepspeed.utils import safe_get_full_grad as _ds_full_grad
        _get_grad = _ds_full_grad
        _grad_reader = "deepspeed.safe_get_full_grad"
    except Exception:
        try:
            from deepspeed.utils import safe_get_local_grad as _ds_local_grad
            _get_grad = _ds_local_grad
            _grad_reader = "deepspeed.safe_get_local_grad"
        except Exception:
            _get_grad = None
            _grad_reader = "p.grad (deepspeed grad API 不可用)"

    def _read_grad(p):
        if _get_grad is not None:
            try:
                g = _get_grad(p)
                if g is not None:
                    return g
            except Exception:
                pass
        ds_t = getattr(p, "ds_tensor", None)
        if ds_t is not None and getattr(ds_t, "grad", None) is not None:
            return ds_t.grad
        return p.grad

    grad_total_sq = 0.0
    n_with_grad = 0
    n_A_with_grad = 0
    n_B_with_grad = 0
    for n, p in lora_params:
        g = _read_grad(p)
        if g is not None:
            gn = float(g.detach().float().norm().item())
            grad_total_sq += gn * gn
            if gn > 0:
                n_with_grad += 1
                if ".lora_A." in n:
                    n_A_with_grad += 1
                elif ".lora_B." in n:
                    n_B_with_grad += 1
    grad_norm = grad_total_sq ** 0.5
    if _get_grad is None:
        logging.warning(
            "R6 PROBE B: 无法经 deepspeed 读梯度，回退 p.ds_tensor.grad/p.grad，"
            "ZeRO-3 下可能误判零梯度。"
        )
    logging.info(
        "R6 PROBE B (grad flow): LoRA grad_norm=%.3e, #params_nonzero=%d "
        "(A=%d, B=%d)（grad_reader=%s）",
        grad_norm, n_with_grad, n_A_with_grad, n_B_with_grad, _grad_reader,
    )
    for _, p in lora_params:
        p.grad = None

    if not (grad_norm > 0.0 and n_with_grad > 0):
        raise RuntimeError(
            "R6 PROBE B 失败：LoRA 零梯度（F-12 类）。"
            f"grad_norm={grad_norm:.3e}, n_with_grad={n_with_grad}。"
            "anchor 注入未建立到 LoRA 的反传路径，退出。"
        )
    logging.info("R6 PROBE B 通过：LoRA 梯度非零（B=0 冷启动下 lora_B 先获梯度，符合预期）。")
    logging.info("=" * 72)
    logging.info("R6 PROBES 全部通过，进入训练循环。")
    logging.info("=" * 72)


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LingBot-World Memory Enhancement Training v6 "
                    "(latent-concat frame-dim anchor + rank-128 LoRA, frozen backbone, flow-matching)"
    )

    # ---- 路径（输出走 paths.py；OUTPUT_ROOT 环境变量可覆盖根）----
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="lingbot-world 预训练权重目录（含 low_noise_model / VAE / T5）")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="CSGO 预处理数据集根目录（含 metadata_{phase}_{split}.csv）")
    parser.add_argument("--run_name", type=str, default=None,
                        help="训练 run 名（默认 default_run_name('latentconcat_lora')）")
    parser.add_argument("--phase", type=str, default="verify",
                        choices=["exp", "full", "verify"],
                        help="数据集 phase：决定 CSV 路径 metadata_{phase}_{split}.csv")

    # ---- LoRA 超参 ----
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA 秩 r（StoryMem 同骨干先例 r128；默认 128）")
    parser.add_argument("--lora_alpha", type=float, default=128.0,
                        help="LoRA scaling alpha（scaling=alpha/r；默认 alpha=r=128 → scaling=1）")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout（默认 0）")
    parser.add_argument("--lora_targets", type=str, default="self_attn,ffn",
                        help="LoRA 挂载层组，逗号分隔，可选 self_attn/ffn/cross_attn/cam；"
                             "默认 'self_attn,ffn'（anchor 经 self-attn 与 query 交互最相关 + ffn 适配）")

    # ---- anchor 自监督选取 ----
    parser.add_argument("--num_anchor_frames", type=int, default=1,
                        help="拼接的 anchor 帧数（token 预算 R3：每帧占整段 token，默认 1）")

    # ---- 训练超参（对齐 v5）----
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
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 目录恢复（含 lora.pth）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只跑 2 steps 验证训练流程")

    # ---- Stochastic N-clip（沿用 v5 习惯）----
    parser.add_argument("--max_context_clips", type=int, default=6,
                        help="最大 context clip 数（n_ctx ~ Uniform(2, max_context_clips)），"
                             "数据集 window_size = max_context_clips+1；context clips 供 anchor 选取")

    # ---- R6 探针阈值 ----
    parser.add_argument("--probe_onoff_thresh", type=float, default=1e-3,
                        help="R6 探针 A：ON/OFF 输出 max_abs_diff 下限阈值")

    # ---- 训练全程健康诊断 ----
    parser.add_argument("--diag_every_epochs", type=int, default=1,
                        help="每 N 个 epoch 末尾跑一次训练健康诊断（LoRA ‖A‖/‖B‖ 范数 + "
                             "anchor ON/OFF diff + grad_norm 趋势）；0=关闭周期诊断（只保留开训 "
                             "R6 探针）。诊断为 no_grad 旁路测量，不改训练数值行为。")

    # ---- W&B ----
    parser.add_argument("--wandb_project", type=str, default="lingbot-memory")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--log_every_steps", type=int, default=10)

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    import accelerate
    from accelerate.utils import DataLoaderConfiguration
    from pipeline.common.paths import (
        train_run_dir, snapshot_config, default_run_name,
    )

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    run_name = args.run_name or default_run_name("latentconcat_lora")
    run_dir = train_run_dir("v6", run_name)
    ckpt_root = os.path.join(str(run_dir), "checkpoints")

    if accelerator.is_main_process:
        _log_path = os.path.join(str(run_dir), "logs", "train.log")
        _fh = logging.FileHandler(_log_path)
        _fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        logging.getLogger().addHandler(_fh)
        snapshot_config(run_dir, {k: v for k, v in vars(args).items()
                                  if not k.startswith("_")})
        logging.info("v6 run_dir = %s", run_dir)
        logging.info("Args: %s", args)

    wb_logger = None
    if args.wandb_mode != "disabled":
        try:
            from pipeline.common.wandb_utils import WandBLogger
            wb_logger = WandBLogger(args, accelerator)
        except Exception as _wb_e:
            logging.warning("W&B init failed (non-fatal): %s", _wb_e)

    trainer = LingBotTrainerV6(args)
    model = trainer.load_models(accelerator.device)

    # ---- 断言：可训练参数全部是 LoRA，总量合理 ----
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable_named if ".lora_" not in n]
    assert not bad, (
        "v6 断言失败：发现非 LoRA 的可训练参数（应只有 LoRA A/B）："
        f"{bad[:10]}"
    )
    n_train = sum(p.numel() for _, p in trainable_named)
    logging.info(
        "v6 可训练参数：%d 个张量，共 %.1fM（应全是 LoRA A/B）。",
        len(trainable_named), n_train / 1e6,
    )
    if not (5e7 < n_train < 1.5e9):
        logging.warning(
            "可训练 LoRA 参数量 %.1fM 不在预期区间 [50M, 1.5B]（r128 self_attn+ffn ≈ 400M）；"
            "若刻意调小 rank/targets 可忽略。", n_train / 1e6,
        )

    trainable_params = [p for _, p in trainable_named]

    # ---- 梯度检查点（OOM 修法，v6 专用）----
    if args.gradient_checkpointing:
        enable_gradient_checkpointing_v6(model)

    # ---- 数据集（复用 v4 CSGOMultiClipDataset / collate）----
    dataset = CSGOMultiClipDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        phase=args.phase,
        max_context_clips=args.max_context_clips,
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

    # ---- 优化器：只优化 LoRA ----
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

    # ---- Resume（只 load lora.pth）----
    start_epoch = 0
    start_global_step = 0
    if args.resume:
        lora_file = os.path.join(args.resume, "lora.pth")
        meta_file = os.path.join(args.resume, "training_metadata.json")
        if os.path.exists(lora_file):
            sd = torch.load(lora_file, map_location="cpu", weights_only=True)
            unwrapped = accelerator.unwrap_model(model)
            _lora_ps = [p for n, p in unwrapped.named_parameters()
                        if p.requires_grad and ".lora_" in n]
            try:
                import deepspeed
                with deepspeed.zero.GatheredParameters(_lora_ps, modifier_rank=0):
                    missing, unexpected = unwrapped.load_state_dict(sd, strict=False)
            except (ImportError, AttributeError):
                missing, unexpected = unwrapped.load_state_dict(sd, strict=False)
            lora_missing = [k for k in missing if ".lora_" in k]
            if lora_missing:
                logging.warning("Resume: LoRA missing keys: %s", lora_missing[:5])
            if unexpected:
                logging.warning("Resume: unexpected keys: %s", unexpected[:5])
            logging.info("Resumed LoRA from %s", lora_file)
        if os.path.exists(meta_file):
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            start_epoch = meta.get("epoch", 0) + 1
            start_global_step = meta.get("global_step", 0)
            logging.info("Resuming from epoch %d, global_step %d",
                         start_epoch, start_global_step)

    # ---- R6 探针（训练循环之前，过不了 raise 退出）----
    # 同一 batch 留作每-epoch 诊断的固定探针 batch（--diag_every_epochs），使 anchor ON/OFF
    # diff 跨 epoch 可比（固定输入看注入是否随训练退化成 no-op），不额外占数据。
    _diag_batch = next(iter(dataloader))
    run_r6_probes_v6(trainer, model, _diag_batch, args, accelerator)
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # 训练循环（脚手架照搬 v5：N-clip broadcast + OOM guard + accumulate）
    # ----------------------------------------------------------------
    global_step = start_global_step
    _last_grad_norm = 0.0
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
                desc=f"Epoch {epoch+1}/{args.num_epochs} [v6 latentconcat+LoRA]",
            )

            for batch_clips in progress:
                # ZeRO-3：所有 rank forward 次数须一致 → rank0 采样 n_ctx 后 broadcast
                _n_ctx_t = torch.zeros(1, dtype=torch.long, device=accelerator.device)
                if accelerator.is_main_process:
                    _n_ctx_t[0] = random.randint(2, args.max_context_clips)
                if accelerator.num_processes > 1:
                    dist.broadcast(_n_ctx_t, src=0)
                _synced_n_ctx = int(_n_ctx_t.item())

                _skip = torch.zeros(1, device=accelerator.device)
                loss = None
                _loss_components: Dict[str, float] = {
                    "diffusion": 0.0, "n_anchor": 0.0, "lat_f_query": 0.0
                }
                try:
                    loss, _loss_components = multi_clip_training_step_v6(
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

                if accelerator.num_processes > 1:
                    dist.all_reduce(_skip, op=dist.ReduceOp.MAX)
                if _skip.item() > 0:
                    logger.warning("OOM at step %d, skipping batch.", global_step)
                    continue

                _back_skip = torch.zeros(1, device=accelerator.device)
                try:
                    with accelerator.accumulate(model):
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            _last_grad_norm = float(
                                accelerator.clip_grad_norm_(model.parameters(),
                                                            args.max_grad_norm)
                            )
                        optimizer.step()
                        if accelerator.sync_gradients:
                            lr_scheduler.step()
                            if wb_logger is not None:
                                _loss_dict = {
                                    "loss/total": loss.item(),
                                    "loss/diffusion": _loss_components["diffusion"],
                                    "memory/n_anchor": _loss_components["n_anchor"],
                                    "train/grad_norm": _last_grad_norm,
                                }
                                wb_logger.log_step(
                                    global_step + 1, _loss_dict, model=None,
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
                    logger.warning("OOM (backward) at step %d, skipping.", global_step)
                    continue

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    na=f"{int(_loss_components['n_anchor'])}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                if accelerator.is_main_process and global_step % args.log_every_steps == 0:
                    logger.info(
                        "step %d | n_ctx=%d | loss=%.4f (diff=%.4f) | "
                        "n_anchor=%d | gnorm=%.3e",
                        global_step, _synced_n_ctx, loss.item(),
                        _loss_components["diffusion"],
                        int(_loss_components["n_anchor"]),
                        _last_grad_norm,
                    )

                if args.save_steps and global_step % args.save_steps == 0:
                    save_lora(
                        accelerator, model, run_dir, f"step_{global_step}",
                        epoch=epoch, global_step=global_step,
                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout, lora_targets=args.lora_targets,
                    )

                if args.dry_run and global_step >= 2:
                    logging.info("dry_run=True, stopping after 2 steps.")
                    break

            avg_loss = epoch_loss / max(num_batches, 1)
            if accelerator.is_main_process:
                logging.info(
                    "Epoch %d/%d | avg_loss=%.4f | lr=%.2e",
                    epoch + 1, args.num_epochs, avg_loss,
                    lr_scheduler.get_last_lr()[0],
                )
                if wb_logger is not None and getattr(wb_logger, "enabled", False):
                    try:
                        import wandb
                        wandb.log({"diag/avg_loss": avg_loss, "diag/epoch": epoch + 1},
                                  step=global_step)
                    except Exception as _we:
                        logging.warning("epoch avg_loss W&B log failed (non-fatal): %s", _we)

            # ---- 每-epoch 训练健康诊断（旁路 no_grad，不改训练数值行为）----
            if args.diag_every_epochs > 0 and (epoch + 1) % args.diag_every_epochs == 0:
                run_epoch_diagnostics(
                    trainer, model, _diag_batch, args, accelerator, wb_logger,
                    epoch=epoch, global_step=global_step,
                    last_grad_norm=_last_grad_norm,
                )

            if (epoch + 1) % args.save_every_n_epochs == 0:
                save_lora(
                    accelerator, model, run_dir, f"epoch_{epoch+1}",
                    epoch=epoch, global_step=global_step,
                    lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout, lora_targets=args.lora_targets,
                )

            if args.dry_run:
                break

        if args.num_epochs % args.save_every_n_epochs != 0:
            save_lora(
                accelerator, model, run_dir, "final",
                epoch=args.num_epochs - 1, global_step=global_step,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, lora_targets=args.lora_targets,
            )
        if accelerator.is_main_process:
            logging.info("v6 training complete! ckpt_root=%s", ckpt_root)
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
