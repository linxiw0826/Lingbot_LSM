"""
train_v5.py — LingBot-World Memory Enhancement 训练脚本 v5（in-context KV，方案A）
====================================================================================

experiment_design Step 41（S-V2）。**最大化复用 v4 train_v4_stage1_dual 的训练循环脚手架**，
只做必要替换。不改 v4 文件、不改 refs。

== 与 v4 train_v4_stage1_dual 的核心差异 ==

  1. 模型：`WanModelWithMemoryV5.from_wan_model(base, memory_layers=None(全 40 层),
     grid=16, encoder_depth=1)`（不是 v4 的 WanModelWithMemory）。from_wan_model 内已做
     R4 断言 + 冻结骨干（除 memory_encoder.* 外全部 requires_grad_(False)）+ encoder
     切片初始化。本脚本直接用。

  2. 优化器只含 memory_encoder：`[p for n,p in model.named_parameters() if p.requires_grad]`
     应只有 memory_encoder（from_wan_model 已冻其余）。开训前断言：可训练参数全部
     name.startswith('memory_encoder')、总量 ≈ 315M（grid=16, depth=1）。

  3. 喂记忆（scope: surprise-independent）：训练每步用
     `bank.retrieve_revisit(query_location, query_timestep, ..., return_latents=True)`
     取回记忆帧 latent `[K,16,h,w]`，作为 `memory_latents` 传入
     `model.forward(x, t, context, seq_len, y=..., dit_cond_dict=..., memory_latents=...)`。
     无记忆可取时 memory_latents=None（退化为纯 i2v，正常）。

  4. bank 构建/update（沿用 v4 Method B 多 clip 顺序路径，但**不依赖 surprise/NFP**）：
     v5 第一版搁置 surprise/NFP 分层（见 decisions.md「讨论 8」scope）。bank.update 的
     surprise_score 维度在 v5 走**中性值**（surprise_score=0.0）：
       - MediumTermBank 写入门槛 surprise > surprise_threshold(0.4)：0.0 不入 medium（无所谓，
         retrieve_revisit 只读 long）；
       - LongTermBank stable 门槛 surprise < stability_threshold(0.2)：0.0 < 0.2 → 永远 stable，
         **每帧都进 long**（这正是 surprise-independent revisit 检索想要的：地点全收录）；
       - novelty check：传 semantic_key=None → LongTermBank.update 跳过 novelty check（退化路径，
         纯 novelty-only/全收录，不引 NFP）。
     **不创建 nfp_head、不算 surprise loss、不算 vis_align/latent_proj 辅助 loss。**
     `location` = 每个 latent 帧的绝对 c2w 平移向量（poses[::4][:, :3, 3]），retrieve_revisit
     按位置 L2 检索（OP-2 Bug2 口径，与 retrieval_probe / eval_ablation 一致）。

  5. R6 开训前强制探针（过不了即 raise 退出，不进训练循环）：
       (a) ON/OFF 差异：同一 batch 跑 memory_latents=mem vs None，输出 max abs diff > 阈值
           (--probe_onoff_thresh, default 1e-3)；否则报「注入是 no-op」并退出。
       (b) 梯度流：一次 backward 后 memory_encoder 至少一个参数 grad 非零、范数 > 0；否则报
           「memory_encoder 零梯度（F-12 类）」退出。
     两个探针结果打到日志显眼处。**位置：在训练循环之前，accelerator.prepare 之后**（用真模型/
     真 batch）。

  6. 去掉（相对 v4）：gate 诊断/日志、dual(high/low) 逻辑（只训单模型 low，等价 v4 TRAIN_HIGH=0）、
     nfp loss、tier_emb、Innovation 7/9/10 的 context-drop / visual-fusion / tier-id。
     loss = 单步 flow-matching 重建（照搬 v4 公式：MSE(pred[:,1:], (noise-latent)[:,1:]) * weight）。

  7. 产出走 paths.py：
       run_dir = train_run_dir('v5', run_name)（run_name 用 default_run_name('inctxkv_A_frozen')
                 或 --run_name 传入）；
       snapshot_config(run_dir, 超参 dict)；
       checkpoint 存 run_dir/checkpoints/epoch_N/（见下「checkpoint 存取约定」）；
       日志 run_dir/logs/train.log；
       OUTPUT_ROOT 可由环境变量覆盖（见 paths.py）。

== checkpoint 存取约定 ==
  **只存 memory_encoder 的可训练权重**（骨干冻结，不重存——eval 时复用 base ckpt 重建骨干后
  load memory_encoder）。保存格式：run_dir/checkpoints/epoch_N/memory_encoder.pth，内容为
  `{name: tensor}`，name 形如 'memory_encoder.in_proj.weight'（即 model.named_parameters()
  里 requires_grad 的全集，已确证全部以 'memory_encoder' 开头）。
  额外保存 run_dir/checkpoints/epoch_N/training_metadata.json（epoch / global_step）。
  **加载约定（eval / resume）**：
     base = WanModel.from_pretrained(ckpt_dir, subfolder='low_noise_model', ...)
     model = WanModelWithMemoryV5.from_wan_model(base, memory_layers=None, grid=.., encoder_depth=..)
     sd = torch.load('memory_encoder.pth')   # 只含 memory_encoder.*
     model.load_state_dict(sd, strict=False) # 骨干来自 base，已就位
  ZeRO-3 下保存需用 deepspeed.zero.GatheredParameters 聚合分片参数（见 save_memory_encoder）。

== 已知限制 / PENDING ==
  - 注入路径 B=1（model_with_memory_v5 / injection.py 契约）；本脚本 batch_size=1，天然满足。
  - padded-batch 不支持（injection.py：memory 注入要求 seq_lens 全 == L，即当前帧无 padding）。
    本脚本单 clip 单样本、num_frames 固定 → seq_lens 全 == L，天然满足。
  - bank.update 不依赖 surprise（走中性 surprise_score=0.0 + semantic_key=None 的全收录 long
    路径）；若后续要分层，再引 NFP。

本地无 torch/CUDA 真跑不动；--help/argparse 与 py_compile 走通即可（真跑待服务器）。
"""

import argparse
import gc
import logging
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
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# sys.path 设置（与 v4 对齐）
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

# ---- 复用 v4 的脚手架组件（dataset / schedule / control-signal / ckpting helper）----
# 直接 import v4 模块，不改动 v4 文件。
from pipeline.v4.train_v4_stage1_dual import (  # noqa: E402
    CSGOMultiClipDataset,
    FlowMatchingSchedule,
    multi_clip_collate_fn,
    enable_gradient_checkpointing,
    _reset_deepspeed_zero_state,
)

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# v5 专用梯度检查点（强制全 block，use_reentrant=True + 输入 requires_grad_，OOM 修法）
# ============================================================

def enable_gradient_checkpointing_v5(model: nn.Module) -> int:
    """对 v5 的**每个** backbone block 强制启用梯度检查点（不因 requires_grad 跳过）。

    背景（OOM 根因）：v5 全冻骨干 → block 内无 requires_grad 参数 → v4 的
    `enable_gradient_checkpointing`（line ~633 `if not any(p.requires_grad ...): continue`）
    对 40 个 block 全跳过 → no-op → 40 层 FFN 中间激活（~4×dim/层）全驻留 → 爆 95GB。
    修法：对**所有** block 强制 checkpoint，backward 时重算激活、不驻留 → 释放约 50GB。

    与 v4 函数的关键区别：
      1. **去掉 requires_grad 跳过**：对每个 block 一律 patch。
      2. **use_reentrant=True + 强制 block 输入 x.requires_grad_(True)**（照搬 v4 成功组合）：
         use_reentrant=False 会触发 ZeRO-3 的 check_recomputed_tensors_match —— ZeRO-3 在
         forward 后把参数重新分片成 shape [0]，backward 重算时 saved [5120] vs recomputed
         [0] 对不上 → 崩（CheckpointError: Recomputed values ... different metadata）。
         改用 reentrant=True 躲开该 check；但 reentrant 在「输入不 require grad」时不建反传图
         （F-12/OP-1），故在调 checkpoint **之前**强制把传入的 x 设为 requires_grad_(True)
         （v4/OP-1 写法）。ZeRO-3 在 reentrant 重算 forward 时会重新 gather 参数（v4 已验证可行）。
      3. **显式把 mem 传入被 checkpoint 的函数**（保留）：memory 本通过 self_attn 的 stash
         副信道（block.self_attn._mem_tokens）进入 block，不是 block.forward 的显式参数。
         为确保 backward 重算时拿到的 mem 是「本步、且 require grad、且能连回 memory_encoder」
         的同一张量，这里在 wrap 内**先读出本步 stash 的 mem**，把它作为 checkpoint 的**显式
         入参**传进去；被 checkpoint 的内层函数在重算时用该 mem 重新 set_memory 再跑原 forward。

    梯度为何能到 memory_encoder（use_reentrant=True 下）：
      - mem = memory_encoder(...) 在**checkpoint 区域之外**算好（在 model.forward 里，
        super().forward 之前），是一个 require-grad 的中间张量。
      - 它作为 checkpoint(fn, x, ..., mem) 的**显式输入**；reentrant 模式对**所有 require-grad
        的输入张量**回传梯度（x 被手动设 require grad、mem 本就 require grad，二者都满足）。
      - backward 时 reentrant 重算 fn → 内层 set_memory(mem) → block.self_attn 用
        self.k(mem)/self.v(mem) 重建注意力 → 重算激活的反传图连到 mem → mem 再连回
        memory_encoder 的参数 → 梯度正确回流到唯一可训练件。
      - block 的冻结参数无 grad（无所谓）；reentrant 之所以需要 require-grad 输入，正是因为
        它靠输入的 grad_fn 建反传图，故必须手动给 x（全冻骨干下输入无 grad）打 requires_grad_。

    返回 patch 的 block 数量。
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    # 找 block 容器（同 v4 探测逻辑）
    block_container = None
    for attr in ['blocks', 'layers', 'transformer_blocks']:
        if hasattr(model, attr):
            block_container = getattr(model, attr)
            break
    if block_container is None:
        for _name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 10:
                block_container = mod
                break
    if block_container is None:
        logging.warning("v5 gradient checkpointing: could not find DiT blocks, skipping")
        return 0

    patched = 0
    for block in block_container:
        orig_forward = block.forward

        def _make_ckpt_fn(module, fn):
            # _in_ckpt 防无限递归：_ckpt_forward → checkpoint(_run) → fn（直接）。
            _in_ckpt = [False]

            @wraps(fn)
            def _ckpt_forward(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict=None):
                if _in_ckpt[0]:
                    return fn(x, e, seq_lens, grid_sizes, freqs,
                              context, context_lens, dit_cond_dict)

                # 读出本步 stash 的 mem（model.forward 已在 super().forward 前 set 好；
                # 注入时为本步 require-grad 张量，不注入时为 None）。把它作为 checkpoint
                # 的显式入参，确保重算时喂回同一张量、梯度连回 memory_encoder。
                self_attn = getattr(module, "self_attn", None)
                mem = getattr(self_attn, "_mem_tokens", None) if self_attn is not None else None

                def _run(x, e, seq_lens, grid_sizes, freqs,
                         context, context_lens, dit_cond_dict, mem):
                    _in_ckpt[0] = True
                    try:
                        # 重算时用显式传入的 mem 重新 set（防 stash 在重算时被其它步覆盖）。
                        if self_attn is not None:
                            self_attn.set_memory(mem)
                        return fn(x, e, seq_lens, grid_sizes, freqs,
                                  context, context_lens, dit_cond_dict)
                    finally:
                        _in_ckpt[0] = False

                # OP-1 fix：reentrant checkpoint 靠输入的 grad_fn 建反传图，全冻骨干下 block
                # 输入 x 无 requires_grad → reentrant 不建图 → memory_encoder 拿不到梯度
                # （F-12）。故在调 checkpoint 前强制把传入的 x 设 requires_grad_(True)。
                if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and not x.requires_grad:
                    x = x.requires_grad_(True)

                # use_reentrant=True（照搬 v4）：躲开 ZeRO-3 的 check_recomputed_tensors_match
                # （reentrant=False 在 ZeRO-3 重新分片参数到 shape [0] 时崩）。位置传参（含
                # dit_cond_dict 非 tensor），与 v4 N-01 写法一致；mem 作显式 require-grad 入参。
                return torch_checkpoint(
                    _run, x, e, seq_lens, grid_sizes, freqs,
                    context, context_lens, dit_cond_dict, mem,
                    use_reentrant=True,
                )
            return _ckpt_forward

        block.forward = _make_ckpt_fn(block, orig_forward)
        patched += 1

    logging.info(
        "v5 gradient checkpointing: 强制 patch %d 个 block（use_reentrant=True + 输入 "
        "requires_grad_，mem 显式入参）。",
        patched,
    )
    return patched


# ============================================================
# Trainer：复用 v4 的 VAE/T5/control-signal 逻辑，模型换成 V5
# ============================================================

class LingBotMemoryTrainerV5:
    """v5 训练器。复用 v4 的 encode_video / encode_text / prepare_y /
    prepare_control_signal（逐字搬运，公式不改），仅 load_models 换成
    WanModelWithMemoryV5.from_wan_model。
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
        self._t5_cache: dict = {}
        self.cam_utils = {}

    def load_models(self, device: torch.device):
        """加载 base WanModel(low_noise_model) → WanModelWithMemoryV5；VAE；T5。"""
        self.device = device
        ckpt_dir = self.args.ckpt_dir

        from wan.modules.model import WanModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.modules.t5 import T5EncoderModel
        from wan.utils.cam_utils import (
            interpolate_camera_poses, compute_relative_poses,
            get_plucker_embeddings, get_Ks_transformed,
        )
        from memory_module.v5_incontext.model_with_memory_v5 import WanModelWithMemoryV5

        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

        # v5 只训 low_noise_model（等价 v4 TRAIN_HIGH=0）
        _subfolder = "low_noise_model"
        logging.info("Loading base WanModel (%s)...", _subfolder)
        base_wan_model = WanModel.from_pretrained(
            ckpt_dir,
            subfolder=_subfolder,
            torch_dtype=torch.bfloat16,
            control_type="act",
        )

        # memory_layers: None=全部 block
        _memory_layers = None
        if self.args.memory_layers:
            _memory_layers = [int(x) for x in self.args.memory_layers.split(",") if x != ""]

        logging.info(
            "Converting to WanModelWithMemoryV5 (memory_layers=%s, grid=%d, encoder_depth=%d)...",
            _memory_layers if _memory_layers is not None else "ALL",
            self.args.grid, self.args.encoder_depth,
        )
        model = WanModelWithMemoryV5.from_wan_model(
            base_wan_model,
            memory_layers=_memory_layers,
            grid=self.args.grid,
            encoder_depth=self.args.encoder_depth,
        )
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

    # ------------------------------------------------------------------
    # 以下 4 个方法逐字复用 v4（公式不改）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode([video_tensor.to(self.device)])[0]
        torch.cuda.empty_cache()
        return latent

    @torch.no_grad()
    def encode_text(self, prompt: str) -> list:
        if prompt in self._t5_cache:
            return [t.to(self.device) for t in self._t5_cache[prompt]]
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        torch.cuda.empty_cache()
        self._t5_cache[prompt] = [t.cpu() for t in context]
        return [t.to(self.device) for t in self._t5_cache[prompt]]

    def prepare_y(self, video_tensor: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Prepare conditional input y（与 v4 完全对齐）。"""
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
        """Prepare dit_cond_dict（与 v4 完全对齐）。"""
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
# 工具：每帧绝对位置（latent 帧对齐）
# ============================================================

def _per_latent_abs_locations(poses: torch.Tensor, lat_f: int) -> torch.Tensor:
    """从 clip 的 [F,4,4] 位姿取每个 latent 帧的绝对 c2w 平移向量。

    与 prepare_control_signal 的 latent-帧子采样口径一致：每 4 个原始帧 → 1 个 latent 帧
    （poses[::4]），再裁/补到 lat_f 个。

    Args:
        poses:  [F, 4, 4] 该 clip 的绝对 c2w 位姿（CPU/GPU 均可）。
        lat_f:  该 clip 的 latent 帧数。

    Returns:
        locations: [lat_f, 3] 绝对位置（CPU float）。
    """
    abs_trans = poses[::4, :3, 3].float().cpu()  # [~F/4, 3]
    if abs_trans.shape[0] >= lat_f:
        abs_trans = abs_trans[:lat_f]
    else:
        pad = abs_trans[-1:].repeat(lat_f - abs_trans.shape[0], 1)
        abs_trans = torch.cat([abs_trans, pad], dim=0)
    return abs_trans  # [lat_f, 3]


# ============================================================
# 多 clip 训练步骤（v5：surprise-independent revisit 喂记忆）
# ============================================================

def multi_clip_training_step_v5(
    trainer: LingBotMemoryTrainerV5,
    model: nn.Module,
    batch_clips: List[dict],
    args,
    n_ctx: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """v5 多 clip 顺序训练步骤。

    流程（沿用 v4 Method B 骨架，但去 surprise/NFP/gate/visual-fusion/tier）：
      1. context clips（前 n_ctx 个）：no_grad，VAE encode 后逐 latent 帧 update bank.long
         （surprise_score=0.0 全收录 + semantic_key=None 跳 novelty + location=绝对平移）。
      2. target clip（第 n_ctx 个）：retrieve_revisit 取回记忆帧 latent → memory_latents；
         有梯度地 forward + flow-matching 单步重建 loss。

    Returns:
        total_loss, loss_components（含 bank_long / retrieved_k 供日志）。
    """
    from memory_module.memory_bank import ThreeTierMemoryBank

    device = trainer.device

    context_clips = batch_clips[:n_ctx]
    target_clip = batch_clips[n_ctx]

    # 每个训练样本独立 bank（不跨样本积累）
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

    # 全局 latent-帧 timestep 计数（跨 clip 递增，供 retrieve_revisit 的 min_gap_frames 口径）
    _global_t = 0

    # ----------------------------------------------------------------
    # Context clips：no_grad，逐 latent 帧填充 bank.long（surprise-independent 全收录）
    # ----------------------------------------------------------------
    for clip_idx, clip in enumerate(context_clips):
        video = clip["video"].squeeze(0).to(device)   # [3, F, H, W]
        poses = clip["poses"].squeeze(0)               # [F, 4, 4]

        with torch.no_grad():
            video_latent = trainer.encode_video(video)  # [16, lat_f, lat_h, lat_w]
            lat_f = video_latent.shape[1]
            locations = _per_latent_abs_locations(poses, lat_f)  # [lat_f, 3]

            for t_idx in range(lat_f):
                latent_frame = video_latent[:, t_idx]  # [16, lat_h, lat_w]
                # v5 surprise-independent：surprise_score=0.0（永远 stable，全进 long），
                # semantic_key=None（LongTermBank.update 跳过 novelty check）。
                # pose_emb 仅占位（retrieve_revisit 不用 pose_emb，只用 location）。
                _pose_placeholder = torch.zeros(model.dim)
                bank.update(
                    pose_emb=_pose_placeholder,
                    latent=latent_frame,
                    surprise_score=0.0,
                    timestep=_global_t,
                    visual_emb=None,
                    chunk_id=clip_idx,
                    semantic_key=None,
                    location=locations[t_idx],
                )
                _global_t += 1

        if clip_idx < len(context_clips) - 1:
            bank.increment_age()

    # ----------------------------------------------------------------
    # Target clip：retrieve_revisit 喂记忆 + 有梯度 forward + loss
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
        context = [c.to(torch.bfloat16) if hasattr(c, 'dtype') and c.dtype != torch.bfloat16 else c
                   for c in context]
        y = trainer.prepare_y(video, video_latent)

    dit_cond_dict_target = trainer.prepare_control_signal(
        poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
    )

    # 取回记忆帧 latent（surprise-independent revisit：按当前 clip 第一帧位置检索）
    target_locations = _per_latent_abs_locations(poses, lat_f)  # [lat_f, 3]
    query_location = target_locations[0]                        # [3]
    query_timestep = _global_t                                  # 当前 clip 第一 latent 帧的全局 t

    memory_latents = _retrieve_memory_latents(
        bank, query_location, query_timestep, args, device
    )
    _retrieved_k = 0 if memory_latents is None else memory_latents.shape[0]

    # 采样 timestep + Flow Matching 加噪（照搬 v4 公式）
    sigma, t, training_weight = trainer.schedule.sample_timestep(model_type="low")
    t = t.to(device).unsqueeze(0)

    noise = torch.randn_like(video_latent)
    noisy_latent = (1.0 - sigma) * video_latent + sigma * noise
    target = noise - video_latent

    # dtype 对齐：noised latent x 与条件 y 来自 VAE(float32)，但骨干为 bf16。
    # autocast 在 CUDA 不可用/被禁用时是 no-op，patch_embedding(Conv3d) 会因
    # "Input type (float) and bias type (BFloat16)" 崩溃 → forward 前显式对齐。
    # 骨干 dtype 直接读骨干权重（ZeRO-3 下 next(parameters()) 不可靠，
    # 与 memory_encoder 读 in_proj.weight.dtype、WanModel.forward 读
    # patch_embedding.weight.device 同口径）。
    _bb_dtype = model.patch_embedding.weight.dtype
    noisy_latent = noisy_latent.to(dtype=_bb_dtype)
    y = y.to(dtype=_bb_dtype)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(
            [noisy_latent],
            t=t,
            context=context,
            seq_len=seq_len,
            y=[y],
            dit_cond_dict=dit_cond_dict_target,
            memory_latents=memory_latents,
        )[0]

    # Diffusion loss（排除第一帧，与 v4 完全对齐）
    pred_rest = pred[:, 1:]
    target_rest = target[:, 1:]
    diffusion_loss = F.mse_loss(pred_rest, target_rest.to(pred_rest.dtype))
    total_loss = diffusion_loss * training_weight

    loss_components: Dict[str, float] = {
        "diffusion": float(total_loss.item()),
        "bank_long": float(len(bank.long.frames)),
        "bank_retrieved_k": float(_retrieved_k),
    }
    return total_loss, loss_components


def _retrieve_memory_latents(
    bank,
    query_location: torch.Tensor,
    query_timestep: int,
    args,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """surprise-independent revisit 检索，返回记忆帧 latent [K,16,h,w] 或 None。"""
    if bank.size() == 0:
        return None
    frames, latents = bank.retrieve_revisit(
        query_location=query_location,
        query_timestep=query_timestep,
        top_k=args.revisit_top_k,
        min_gap_frames=args.revisit_min_gap_frames,
        device=device,
        return_latents=True,
    )
    if not frames or latents is None:
        return None
    return latents  # [K, 16, h, w]


# ============================================================
# Checkpoint：只存 memory_encoder 可训练权重
# ============================================================

def save_memory_encoder(accelerator, model, run_dir, tag: str,
                        epoch: int = 0, global_step: int = 0,
                        grid: Optional[int] = None,
                        encoder_depth: Optional[int] = None,
                        memory_layers=None):
    """只保存 memory_encoder 的可训练权重（骨干冻结，复用 base ckpt）。

    保存到 run_dir/checkpoints/<tag>/memory_encoder.pth（dict: name -> cpu tensor），
    name 形如 'memory_encoder.in_proj.weight'。ZeRO-3 下用 GatheredParameters 聚合分片。

    同目录写 training_metadata.json，除 epoch/global_step/n_params 外，**增存
    grid / encoder_depth / memory_layers 三项模型重建配置**（W1：供 eval 以 metadata
    为准重建模型，堵死 grid 错配导致的 strict=False 静默载入假阴性）。memory_layers
    为注入层索引 list 或 None（全部 block）。
    """
    save_dir = os.path.join(str(run_dir), "checkpoints", tag)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    trainable = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]

    # ZeRO-3：参数被分片，需聚合到 rank0 再取 .data
    try:
        import deepspeed
        with deepspeed.zero.GatheredParameters(
            [p for _, p in trainable], modifier_rank=0
        ):
            if accelerator.is_main_process:
                state = {n: p.data.detach().cpu().clone() for n, p in trainable}
    except (ImportError, AttributeError):
        if accelerator.is_main_process:
            state = {n: p.data.detach().cpu().clone() for n, p in trainable}

    if accelerator.is_main_process:
        assert all(n.startswith("memory_encoder") for n in state), (
            "save_memory_encoder: 发现非 memory_encoder 的可训练参数！"
            f"keys={[n for n in state if not n.startswith('memory_encoder')]}"
        )
        out_path = os.path.join(save_dir, "memory_encoder.pth")
        torch.save(state, out_path)
        import json
        with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
            json.dump({"epoch": epoch, "global_step": global_step,
                       "n_params": len(state),
                       # W1: 模型重建配置（eval 以此为准，防 grid/层数错配静默载入）
                       "grid": grid,
                       "encoder_depth": encoder_depth,
                       "memory_layers": memory_layers}, f)
        logging.info(
            "Saved memory_encoder checkpoint (%d tensors) -> %s",
            len(state), out_path,
        )

    accelerator.wait_for_everyone()


# ============================================================
# R6 探针（开训前强制，过不了 raise）
# ============================================================

def run_r6_probes(
    trainer: LingBotMemoryTrainerV5,
    model: nn.Module,
    batch_clips: List[dict],
    args,
    accelerator,
) -> None:
    """R6 两探针（训练循环之前调用）。任一不过即 raise RuntimeError 退出。

    探针 A（ON/OFF 差异）：同一 target batch，跑 memory_latents=mem vs None，
        max abs diff 必须 > args.probe_onoff_thresh，否则注入是 no-op。
    探针 B（梯度流）：对 ON 输出做一次 backward，memory_encoder 至少一个参数 grad 非零
        且 grad 范数 > 0，否则零梯度（F-12 类）。
    """
    unwrapped = accelerator.unwrap_model(model)
    device = trainer.device

    logging.info("=" * 72)
    logging.info("R6 PROBES (强制；过不了即退出，不进训练循环)")
    logging.info("=" * 72)

    # ---- 构造一个真 batch：用 batch_clips 的最后一个 clip 作 target，
    #      其余作 context 填 bank（至少 1 个 context 才能 retrieve 出 memory）----
    n_ctx = max(1, len(batch_clips) - 1)

    from memory_module.memory_bank import ThreeTierMemoryBank
    bank = ThreeTierMemoryBank(
        short_cap=args.short_cap, medium_cap=args.medium_cap, long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life, dup_threshold=args.dup_threshold,
    )
    _global_t = 0
    for clip in batch_clips[:n_ctx]:
        video = clip["video"].squeeze(0).to(device)
        poses = clip["poses"].squeeze(0)
        with torch.no_grad():
            vlat = trainer.encode_video(video)
            lat_f = vlat.shape[1]
            locs = _per_latent_abs_locations(poses, lat_f)
            for t_idx in range(lat_f):
                bank.update(
                    pose_emb=torch.zeros(unwrapped.dim),
                    latent=vlat[:, t_idx],
                    surprise_score=0.0,
                    timestep=_global_t,
                    visual_emb=None,
                    chunk_id=0,
                    semantic_key=None,
                    location=locs[t_idx],
                )
                _global_t += 1

    target_clip = batch_clips[n_ctx]
    video = target_clip["video"].squeeze(0).to(device)
    poses = target_clip["poses"].squeeze(0)
    actions = target_clip["actions"].squeeze(0)
    intrinsics = target_clip["intrinsics"].squeeze(0)
    prompt = target_clip["prompt"]
    h, w = video.shape[2], video.shape[3]

    with torch.no_grad():
        video_latent = trainer.encode_video(video)
    lat_f, lat_h, lat_w = video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
    seq_len = lat_f * lat_h * lat_w // (trainer.patch_size[1] * trainer.patch_size[2])

    with torch.no_grad():
        context = trainer.encode_text(prompt)
        context = [c.to(torch.bfloat16) if hasattr(c, 'dtype') and c.dtype != torch.bfloat16 else c
                   for c in context]
        y = trainer.prepare_y(video, video_latent)
    dit_cond_dict = trainer.prepare_control_signal(
        poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
    )

    target_locations = _per_latent_abs_locations(poses, lat_f)
    mem = _retrieve_memory_latents(
        bank, target_locations[0], _global_t, args, device
    )
    if mem is None:
        raise RuntimeError(
            "R6 探针：bank 检索不到任何 memory_latents（retrieve_revisit 返回空）。"
            "无法验证注入；请检查 context clip 数 / revisit_min_gap_frames / long_cap。"
        )
    logging.info("R6: retrieved %d memory frames for probe.", mem.shape[0])

    sigma, t_sched, _ = trainer.schedule.sample_timestep(model_type="low")
    t_sched = t_sched.to(device).unsqueeze(0)
    noise = torch.randn_like(video_latent)
    noisy_latent = (1.0 - sigma) * video_latent + sigma * noise

    # dtype 对齐：noised latent x 与条件 y(来自 VAE,float32) → 骨干 bf16。
    # 与训练步同口径（读骨干 patch_embedding.weight.dtype）。一次对齐覆盖
    # _fwd 的 ON / OFF / 梯度流三处 forward（闭包捕获 noisy_latent / y）。
    _bb_dtype = unwrapped.patch_embedding.weight.dtype
    noisy_latent = noisy_latent.to(dtype=_bb_dtype)
    y = y.to(dtype=_bb_dtype)

    def _fwd(mem_arg):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model(
                [noisy_latent], t=t_sched, context=context, seq_len=seq_len,
                y=[y], dit_cond_dict=dit_cond_dict, memory_latents=mem_arg,
            )[0]

    # ---- 探针 A：ON vs OFF ----
    with torch.no_grad():
        out_on = _fwd(mem).float()
        out_off = _fwd(None).float()
    max_abs_diff = (out_on - out_off).abs().max().item()
    logging.info(
        "R6 PROBE A (ON/OFF): max_abs_diff=%.3e  (threshold=%.3e)",
        max_abs_diff, args.probe_onoff_thresh,
    )
    if max_abs_diff <= args.probe_onoff_thresh:
        raise RuntimeError(
            f"R6 PROBE A 失败：memory 注入是 no-op（max_abs_diff={max_abs_diff:.3e} "
            f"<= {args.probe_onoff_thresh:.3e}）。memory_latents 未影响输出，退出。"
        )
    logging.info("R6 PROBE A 通过：注入对输出有影响。")

    # ---- 探针 B：梯度流（memory_encoder 非零梯度）----
    for p in unwrapped.memory_encoder.parameters():
        if p.grad is not None:
            p.grad = None
    out_on_grad = _fwd(mem)
    probe_loss = out_on_grad.float().pow(2).mean()
    accelerator.backward(probe_loss)

    grad_total_sq = 0.0
    n_with_grad = 0
    for _, p in unwrapped.memory_encoder.named_parameters():
        if p.grad is not None:
            g = p.grad.float().norm().item()
            grad_total_sq += g * g
            if g > 0:
                n_with_grad += 1
    grad_norm = grad_total_sq ** 0.5
    logging.info(
        "R6 PROBE B (grad flow): memory_encoder grad_norm=%.3e, "
        "#params_with_nonzero_grad=%d", grad_norm, n_with_grad,
    )
    # 清理本次探针产生的梯度，避免污染第一步训练
    for p in unwrapped.memory_encoder.parameters():
        p.grad = None

    if not (grad_norm > 0.0 and n_with_grad > 0):
        raise RuntimeError(
            "R6 PROBE B 失败：memory_encoder 零梯度（F-12 类）。"
            f"grad_norm={grad_norm:.3e}, n_with_grad={n_with_grad}。"
            "注入未建立到 memory_encoder 的反传路径，退出。"
        )
    logging.info("R6 PROBE B 通过：memory_encoder 梯度非零。")
    logging.info("=" * 72)
    logging.info("R6 PROBES 全部通过，进入训练循环。")
    logging.info("=" * 72)


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LingBot-World Memory Enhancement Training v5 "
                    "(in-context KV, frozen backbone, MemoryEncoder-only, surprise-independent revisit)"
    )

    # ---- 路径（输出走 paths.py；OUTPUT_ROOT 环境变量可覆盖根）----
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="lingbot-world 预训练权重目录（含 low_noise_model / VAE / T5）")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="CSGO 预处理数据集根目录（含 metadata_{phase}_{split}.csv）")
    parser.add_argument("--run_name", type=str, default=None,
                        help="训练 run 名（默认 default_run_name('inctxkv_A_frozen')）")
    parser.add_argument("--phase", type=str, default="exp",
                        choices=["exp", "full", "verify"],
                        help="数据集 phase：决定 CSV 路径 metadata_{phase}_{split}.csv")

    # ---- v5 模型超参 ----
    parser.add_argument("--grid", type=int, default=16,
                        help="MemoryEncoder 每帧 grid×grid token（默认 16→256/帧）")
    parser.add_argument("--encoder_depth", type=int, default=1,
                        help="MemoryEncoder 残差 Transformer 块层数（默认 1，精简起步）")
    parser.add_argument("--memory_layers", type=str, default=None,
                        help="注入层索引逗号分隔（如 '0,10,20,39'）；None/空=全部 block")

    # ---- 训练超参（对齐 v4 习惯）----
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
                        help="从指定 checkpoint 目录恢复（含 memory_encoder.pth）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只跑 2 steps 验证训练流程")

    # ---- Stochastic N-clip（沿用 v4 习惯）----
    parser.add_argument("--max_context_clips", type=int, default=6,
                        help="最大 context clip 数（n_ctx ~ Uniform(2, max_context_clips)），"
                             "数据集 window_size = max_context_clips+1")

    # ---- surprise-independent revisit 检索 ----
    parser.add_argument("--revisit_top_k", type=int, default=5,
                        help="retrieve_revisit 返回的记忆帧数上限")
    parser.add_argument("--revisit_min_gap_frames", type=int, default=0,
                        help="retrieve_revisit 排除 timestep 距离 < 此值的近邻帧")

    # ---- ThreeTierMemoryBank 超参（v5 走 long 全收录；保留以兼容 bank 构造）----
    parser.add_argument("--short_cap", type=int, default=1)
    parser.add_argument("--medium_cap", type=int, default=8)
    parser.add_argument("--long_cap", type=int, default=256,
                        help="LongTermBank 容量（v5 全收录 revisit，需较大；默认 256）")
    parser.add_argument("--surprise_threshold", type=float, default=0.4)
    parser.add_argument("--stability_threshold", type=float, default=0.2,
                        help="LongTermBank stable 上限；v5 surprise=0.0 恒 < 此值（全收录）")
    parser.add_argument("--novelty_threshold", type=float, default=0.7)
    parser.add_argument("--half_life", type=float, default=10.0)
    parser.add_argument("--dup_threshold", type=float, default=0.95)

    # ---- R6 探针阈值 ----
    parser.add_argument("--probe_onoff_thresh", type=float, default=1e-3,
                        help="R6 探针 A：ON/OFF 输出 max_abs_diff 下限阈值")

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

    # ---- 产出目录（paths.py）----
    run_name = args.run_name or default_run_name("inctxkv_A_frozen")
    run_dir = train_run_dir("v5", run_name)
    ckpt_root = os.path.join(str(run_dir), "checkpoints")

    if accelerator.is_main_process:
        # 文件日志（co-located）
        _log_path = os.path.join(str(run_dir), "logs", "train.log")
        _fh = logging.FileHandler(_log_path)
        _fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        logging.getLogger().addHandler(_fh)
        snapshot_config(run_dir, {k: v for k, v in vars(args).items()
                                  if not k.startswith("_")})
        logging.info("v5 run_dir = %s", run_dir)
        logging.info("Args: %s", args)

    # ---- W&B（复用 common.wandb_utils；v5 无 gate/memory_cross_attn，helper 会自动跳过）----
    wb_logger = None
    if args.wandb_mode != "disabled":
        try:
            from pipeline.common.wandb_utils import WandBLogger
            wb_logger = WandBLogger(args, accelerator)
        except Exception as _wb_e:
            logging.warning("W&B init failed (non-fatal): %s", _wb_e)

    trainer = LingBotMemoryTrainerV5(args)
    model = trainer.load_models(accelerator.device)

    # ---- W1: 解析注入层（用于 checkpoint metadata，与 load_models 内口径一致）----
    _memory_layers_meta = None
    if args.memory_layers:
        _memory_layers_meta = [int(x) for x in args.memory_layers.split(",") if x != ""]

    # ---- 断言：可训练参数全部是 memory_encoder，总量 ≈ 315M（from_wan_model 已冻骨干）----
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable_named if not n.startswith("memory_encoder")]
    assert not bad, (
        "v5 断言失败：发现非 memory_encoder 的可训练参数（from_wan_model 应冻骨干）："
        f"{bad[:10]}"
    )
    n_train = sum(p.numel() for _, p in trainable_named)
    logging.info(
        "v5 可训练参数：%d 个张量，共 %.1fM（应全在 memory_encoder）。",
        len(trainable_named), n_train / 1e6,
    )
    if not (1e8 < n_train < 6e8):
        logging.warning(
            "可训练参数量 %.1fM 不在 ~315M(grid=16,depth=1) 的预期区间 [100M,600M]；"
            "若刻意调小 grid/depth 可忽略。", n_train / 1e6,
        )

    trainable_params = [p for _, p in trainable_named]

    # ---- 梯度检查点（OOM 修法）：必须用 v5 专用 enable_gradient_checkpointing_v5。
    #      v4 的 enable_gradient_checkpointing 因 `if not any(p.requires_grad ...): continue`
    #      会对 v5 全冻骨干的所有 40 个 block 跳过 → no-op → 40 层 FFN 激活全驻留 → OOM。
    #      v5 版强制对**每个** block 用 use_reentrant=True（躲 ZeRO-3 check）+ 输入
    #      x.requires_grad_(True)（reentrant 建图前提）检查点（重算激活、不驻留），
    #      并把 stash 的 mem 作显式入参传入 → backward 重算正确、梯度回流 memory_encoder。
    if args.gradient_checkpointing:
        enable_gradient_checkpointing_v5(model)

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

    # ---- 优化器：只优化 memory_encoder ----
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

    # ---- Resume（只 load memory_encoder.pth）----
    start_epoch = 0
    start_global_step = 0
    if args.resume:
        me_file = os.path.join(args.resume, "memory_encoder.pth")
        meta_file = os.path.join(args.resume, "training_metadata.json")
        if os.path.exists(me_file):
            sd = torch.load(me_file, map_location="cpu", weights_only=True)
            unwrapped = accelerator.unwrap_model(model)
            try:
                import deepspeed
                with deepspeed.zero.GatheredParameters(
                    list(unwrapped.memory_encoder.parameters()), modifier_rank=0
                ):
                    missing, unexpected = unwrapped.load_state_dict(sd, strict=False)
            except (ImportError, AttributeError):
                missing, unexpected = unwrapped.load_state_dict(sd, strict=False)
            # missing 应全是骨干（来自 base，已就位）；只关心 memory_encoder.* 是否齐
            me_missing = [k for k in missing if k.startswith("memory_encoder")]
            if me_missing:
                logging.warning("Resume: memory_encoder missing keys: %s", me_missing[:5])
            if unexpected:
                logging.warning("Resume: unexpected keys: %s", unexpected[:5])
            logging.info("Resumed memory_encoder from %s", me_file)
        if os.path.exists(meta_file):
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            start_epoch = meta.get("epoch", 0) + 1
            start_global_step = meta.get("global_step", 0)
            logging.info("Resuming from epoch %d, global_step %d",
                         start_epoch, start_global_step)

    # ----------------------------------------------------------------
    # R6 探针（训练循环之前，过不了 raise 退出）
    # ----------------------------------------------------------------
    _probe_batch = next(iter(dataloader))
    run_r6_probes(trainer, model, _probe_batch, args, accelerator)
    del _probe_batch
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # 训练循环（脚手架照搬 v4：N-clip broadcast + OOM guard + accumulate）
    # ----------------------------------------------------------------
    global_step = start_global_step
    # 全程可见的梯度范数（F-12 教训）；仅在 sync_gradients 分支更新，
    # OOM/未 sync 的步保留上一次值，循环外初始化避免 NameError。
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
                desc=f"Epoch {epoch+1}/{args.num_epochs} [v5 inctxkv]",
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
                    "diffusion": 0.0, "bank_long": 0.0, "bank_retrieved_k": 0.0
                }
                try:
                    loss, _loss_components = multi_clip_training_step_v5(
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

                # backward OOM guard（DDP-safe，照搬 v4）
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
                                    "memory/bank_long": _loss_components["bank_long"],
                                    "memory/retrieved_k": _loss_components["bank_retrieved_k"],
                                    "train/grad_norm": _last_grad_norm,
                                }
                                # model=None：v5 无 gate/memory_cross_attn，跳过相关诊断
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
                    retr=f"{int(_loss_components['bank_retrieved_k'])}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                if accelerator.is_main_process and global_step % args.log_every_steps == 0:
                    logger.info(
                        "step %d | n_ctx=%d | loss=%.4f (diff=%.4f) | "
                        "bank_long=%d retr=%d | gnorm=%.3e",
                        global_step, _synced_n_ctx, loss.item(),
                        _loss_components["diffusion"],
                        int(_loss_components["bank_long"]),
                        int(_loss_components["bank_retrieved_k"]),
                        _last_grad_norm,
                    )

                if args.save_steps and global_step % args.save_steps == 0:
                    save_memory_encoder(
                        accelerator, model, run_dir, f"step_{global_step}",
                        epoch=epoch, global_step=global_step,
                        grid=args.grid, encoder_depth=args.encoder_depth,
                        memory_layers=_memory_layers_meta,
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

            if (epoch + 1) % args.save_every_n_epochs == 0:
                save_memory_encoder(
                    accelerator, model, run_dir, f"epoch_{epoch+1}",
                    epoch=epoch, global_step=global_step,
                    grid=args.grid, encoder_depth=args.encoder_depth,
                    memory_layers=_memory_layers_meta,
                )

            if args.dry_run:
                break

        if args.num_epochs % args.save_every_n_epochs != 0:
            save_memory_encoder(
                accelerator, model, run_dir, "final",
                epoch=args.num_epochs - 1, global_step=global_step,
                grid=args.grid, encoder_depth=args.encoder_depth,
                memory_layers=_memory_layers_meta,
            )
        if accelerator.is_main_process:
            logging.info("v5 training complete! ckpt_root=%s", ckpt_root)
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
