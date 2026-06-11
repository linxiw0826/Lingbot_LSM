"""memory_injection_diag.py — Exp2 注入 forward-only 诊断脚本（嫌疑 A/B 排查 + 注入贡献量化）

目的（experiment_design.md Exp2 / decisions.md D-06 / open_problems.md OP-2 之后）
-------------------------------------------------------------------------------------
oracle_injection.py 发现：注入正确记忆(oracle)/不注入(off)/注入错误记忆(wrong)三种模式
生成结果几乎完全一致（SSIM 0.317/0.321/0.318，Δ≈0.004），即注入对生成毫无影响。在投入
「重设计 V 表征」之前，本脚本用一个**纯前向诊断**排除两个会「假装是 V 问题」的简单原因，
并定位注入到底死在哪一环：

  - 嫌疑 A（代码 bug）：oracle/off/wrong 三模式是否其实喂了**相同**的 memory K/V 张量？
    若相同，生成一致是 bug 而非机制问题。→ 本脚本报告 ||V_oracle−V_wrong||、
    cosine(V_oracle,V_wrong)、||K_oracle−K_wrong||，并确认 off 模式确实 memory_states=None。
  - 嫌疑 B（注意力塌缩）：记忆 cross-attention 的注意力权重是否塌成近均匀分布（永远读
    8 个记忆帧的平均，无区分力）？→ 在 hook 内 **eager 重算** softmax(q·kᵀ/√head_dim)，
    报告平均熵(nats)与最大权重。均匀塌缩 → 熵≈ln(K)。
  - 量化：记忆经过 gate 后到底把隐状态 x 推动了多少（贡献 norm 占残差流 norm 的比例）。

⚠️ 本脚本是 **forward-only 诊断，不生成**
----------------------------------------
对每个 (mode, timestep) 只跑**一次** model.forward——不采样、不 VAE decode、不算 SSIM。
timestep 通过 --timesteps（逗号分隔整数，0..999，默认 "999,500,100"）探多个噪声水平，
每个整数映射到 FlowMatchingSchedule 的最近调度索引，确定地取出 (sigma, t)，三模式共享
**同一 noise + 同一 timestep**（paired 比较，保证三模式差异只来自注入的 K/V）。

单 GPU 用法
-----------
    export CUDA_VISIBLE_DEVICES=<gpu>     # 选定物理 GPU
    python src/pipeline/memory_injection_diag.py \
        --ckpt_dir <base ckpt> --ft_model_dir <verify low epoch_N> \
        --dataset_dir <data> --metadata metadata_verify_val.csv \
        --device cuda:0 --timesteps 999,500,100 --output_dir <out>
本脚本不走 Ulysses SP；--device cuda:0 即指当前可见的第 0 张卡。

复用而非复制（task 硬性要求 1）
------------------------------
- 模型/数据加载、forward-only 加噪路径：复用 eval_ablation.py 的
  `_build_trainer_and_model`（= train_v4 load_models + retrieval_probe ft 加载）+
  `_frame_to_clip_slice` + trainer.encode_video/prepare_y/prepare_control_signal/encode_text +
  FlowMatchingSchedule（noisy=(1-sigma)·lat+sigma·noise，三模式共享 noise/timestep）。
- 重访点判定 + oracle/off/wrong 三模式 K/V 构造：复用 oracle_injection.py 的
  `_find_revisit_points` / `RevisitPoint` / `_build_oracle_memory_kv` /
  `_pick_oracle_indices` / `_pick_wrong_indices`（import，不重写）。
- episode/VAE/emb：复用 retrieval_probe.py（经 eval_ablation 间接 import）。
- 仪表化：本脚本独有——在 40 个 memory_cross_attn 上挂 forward hook + memory_norm pre-hook，
  采集 gate / ||out|| / 贡献比 / 注意力熵；不修改任何被复用模块。

跨模块数据契约（注入 K/V，与 oracle_injection / eval_ablation / infer_v4 同契约）
-------------------------------------------------------------------------------
- memory_states (K)       : [1, K, dim=5120]，pose_emb（get_projected_frame_embs）
- memory_value_states (V) : [1, K, dim=5120]，visual_emb（get_projected_latent_emb）
- tier_ids                : [K] int64（oracle/wrong 全标 long=2）或 None
  消费方：WanModelWithMemory.forward(memory_states=, memory_value_states=) +
          MemoryCrossAttention.forward（model_with_memory.py:115-121）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# sys.path 设置（与 oracle_injection.py / eval_ablation.py 一致）
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
# 复用：oracle_injection 的重访点 + 三模式 K/V 构造（import，不重写）
# ---------------------------------------------------------------------------

from pipeline.oracle_injection import (  # noqa: E402
    RevisitPoint,
    _find_revisit_points,
    _frame_to_clip_slice,
    _build_oracle_memory_kv,
    _pick_oracle_indices,
    _pick_wrong_indices,
)
import pipeline.oracle_injection as _oracle_mod  # noqa: E402（设置 _SIZE_HW / _ORACLE_CLIP_FRAMES）

# 复用：eval_ablation 的 trainer/model 加载（= train_v4 load_models + retrieval_probe ft 加载）
from pipeline.eval_ablation import (  # noqa: E402
    _build_trainer_and_model,
    _log_memory_weight_sanity,
)

# 复用：retrieval_probe 的 episode 加载 + VAE/emb（经 eval_ablation 同源 import 路径）
from pipeline.retrieval_probe import (  # noqa: E402
    EpisodeData,
    load_episode_clips,
    build_episode_data,
    _decode_episode_video,
    _vae_encode_batched,
    _expand_latents_to_frames,
)


# ---------------------------------------------------------------------------
# CLI（task 硬性要求 6：对齐 oracle_injection.py 的模型/数据加载部分）
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Exp2 注入 forward-only 诊断 —— 每个 (mode, timestep) 只跑一次 model.forward，"
            "挂 hook 仪表化全部 40 个 memory_cross_attn，采集 gate/||out||/贡献比/注意力熵，"
            "并做跨模式张量 sanity（嫌疑 A）。不生成、不采样、不算 SSIM。"
        )
    )
    # ---- 数据（对齐 oracle_injection.py）----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_verify_val.csv")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录（diag.json + diag.md + run.log）")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=1,
                   help="0=不限；>0 时取前 N 个 episode（诊断默认 1，够定位机制）")
    p.add_argument("--max_revisit_points", type=int, default=1,
                   help="每 episode 最多取多少个重访点（诊断默认 1）")

    # ---- 模型权重（对齐 oracle_injection.py / eval_ablation.py）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="基础模型目录（含 low_noise_model 子目录 + Wan2.1_VAE.pth + T5）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="重训后的 v4 low_noise_model/epoch_N 目录（memory 权重训练好的）；"
                        "缺失则用 base 权重（memory 随机，仅负对照，诊断仍可定位 bug）")
    p.add_argument("--ft_high_model_dir", type=str, default=None,
                   help="dual 训练 high_noise_model 目录（可选；本诊断默认评 low）")
    p.add_argument("--model_type", type=str, default="low", choices=["low", "high"],
                   help="诊断哪个子模型（low_noise_model / high_noise_model）；默认 low")
    p.add_argument("--tier_config", type=str, default="oracle_only",
                   choices=["oracle_only"],
                   help="本诊断只走 oracle_only（直接注入 GT 帧，绕过 bank，最干净地隔离"
                        "「注入这件事本身有没有被用到」）。full/medium_long 的 bank 检索"
                        "已由 oracle_injection / eval_ablation 覆盖，此处不重复。")

    # ---- 重访点判定（复用 oracle_injection / retrieval_probe 口径）----
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
    p.add_argument("--num_oracle_frames", type=int, default=2,
                   help="每个重访点注入多少 oracle/wrong GT 帧（K）")

    # ---- 噪声水平（task 硬性要求 2）----
    p.add_argument("--timesteps", type=str, default="999,500,100",
                   help="逗号分隔的整数噪声水平（0..999）。每个映射到 FlowMatchingSchedule "
                        "最近调度索引，确定取出 (sigma,t)；三模式共享同一 noise+timestep。")

    # ---- 数据/前向规格 ----
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（min_time_gap_sec→帧数换算）")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--vae_batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 40 个 memory_cross_attn 的仪表化 hook（本脚本独有，不改任何被复用模块）
# ---------------------------------------------------------------------------

@dataclass
class _BlockProbe:
    """单个 block 在一次 forward 中采集到的诊断量（hook 写入）。"""
    gate: float = float("nan")            # gate 标量
    out_norm: float = float("nan")        # pre-gate ||out||（gate 缩放前）
    contrib_norm: float = float("nan")    # post-gate ||gate*out||
    x_norm: float = float("nan")          # 注入点残差流 ||x||（memory_norm 的输入）
    contrib_ratio: float = float("nan")   # contrib_norm / x_norm（记忆把隐状态推动了多少）
    attn_entropy: float = float("nan")    # 平均注意力熵（nats）；均匀塌缩 → ln(K)
    attn_max: float = float("nan")        # 平均最大注意力权重
    K: int = 0                            # 记忆 token 数
    contrib_vec: Optional[torch.Tensor] = None  # post-gate 贡献张量（CPU，用于 Δx 跨模式比较）


class MemoryDiagInstrument:
    """对模型所有 MemoryBlockWrapper 的 memory_cross_attn 挂 hook，仪表化采集诊断量。

    对每个 wrapper 注册两个 hook：
      1. memory_norm 的 forward_pre_hook：捕获其输入 x（= block 输出 = 注入点残差流），
         记录 ||x||（注入贡献比的分母）。
      2. memory_cross_attn 的 forward_hook（with_kwargs=True）：拿到
         (normed_x, memory_key_states, memory_value_states) 位置参数 + tier_ids kwarg
         + 该模块返回的 `gate*out`（post-gate 贡献）。在 hook 内：
           - gate / pre-gate ||out||（用 _last_attn_out_norm，模块已 detach 记录）
           - post-gate 贡献 ||gate*out||（= 返回张量 norm）+ 贡献比
           - **eager 重算注意力权重**（flash_attention 不返回权重）：
             复算 q=norm_q(q(normed_x))、k=norm_k(k(mem_key))(+tier_emb)，按 head reshape，
             float32 算 softmax(q·kᵀ/√head_dim) → 熵(nats) + 最大权重。

    一次 forward 后通过 `collect()` 取出 {block_idx: _BlockProbe}，再 `reset()`。
    """

    def __init__(self, model):
        self.model = model
        self._handles: List = []
        self._probes: Dict[int, _BlockProbe] = {}
        # block_idx → (wrapper, memory_cross_attn)
        self._wrappers: Dict[int, Tuple] = {}
        self._register()

    def _register(self) -> None:
        from memory_module.model_with_memory import MemoryBlockWrapper

        for idx, blk in enumerate(getattr(self.model, "blocks", [])):
            if not isinstance(blk, MemoryBlockWrapper):
                continue
            ca = getattr(blk, "memory_cross_attn", None)
            if ca is None:
                continue
            self._wrappers[idx] = (blk, ca)

            # pre-hook on memory_norm：捕获注入点残差流 x（memory_norm 的输入）
            mn = getattr(blk, "memory_norm", None)
            if mn is not None:
                h_pre = mn.register_forward_pre_hook(self._make_norm_pre_hook(idx))
                self._handles.append(h_pre)

            # forward-hook on memory_cross_attn（with_kwargs 拿到 tier_ids kwarg）
            h_ca = ca.register_forward_hook(
                self._make_ca_hook(idx, ca), with_kwargs=True
            )
            self._handles.append(h_ca)

        logger.info("MemoryDiagInstrument: 已挂 hook 的 memory_cross_attn 数 = %d",
                    len(self._wrappers))

    # ---- 残差流 x norm（memory_norm 输入）----
    def _make_norm_pre_hook(self, idx: int):
        def _pre_hook(module, args):
            try:
                x = args[0]
                if isinstance(x, torch.Tensor):
                    probe = self._probes.setdefault(idx, _BlockProbe())
                    with torch.no_grad():
                        probe.x_norm = float(x.detach().float().norm().item())
            except Exception:  # noqa: BLE001
                pass
            return None
        return _pre_hook

    # ---- gate / ||out|| / 贡献比 / 注意力熵 ----
    def _make_ca_hook(self, idx: int, ca):
        def _hook(module, args, kwargs, output):
            try:
                probe = self._probes.setdefault(idx, _BlockProbe())
                # args = (normed_x, memory_key_states, memory_value_states?)（位置）
                normed_x = args[0] if len(args) > 0 else kwargs.get("x")
                memory_key_states = (args[1] if len(args) > 1
                                     else kwargs.get("memory_key_states"))
                tier_ids = kwargs.get("tier_ids")

                # gate 标量（模块已在 forward 内 detach 记录 _last_gate_value）
                probe.gate = float(getattr(module, "_last_gate_value", float("nan")))
                # pre-gate ||out||（gate 缩放前）：模块记录的 _last_attn_out_norm
                probe.out_norm = float(getattr(module, "_last_attn_out_norm", float("nan")))

                # post-gate 贡献 = 返回张量（gate*out），即 wrapper 加到 x 上的量
                if isinstance(output, torch.Tensor):
                    with torch.no_grad():
                        contrib = output.detach().float()
                    probe.contrib_norm = float(contrib.norm().item())
                    probe.contrib_vec = contrib.cpu()
                    if np.isfinite(probe.x_norm) and probe.x_norm > 0:
                        probe.contrib_ratio = probe.contrib_norm / probe.x_norm

                # 注意力权重：eager 重算（flash_attention 不返回权重）
                if isinstance(normed_x, torch.Tensor) and memory_key_states is not None:
                    ent, amax, k_cnt = _eager_attention_stats(
                        module, normed_x, memory_key_states, tier_ids
                    )
                    probe.attn_entropy = ent
                    probe.attn_max = amax
                    probe.K = k_cnt
            except Exception as exc:  # noqa: BLE001
                logger.warning("ca hook 采集失败 block=%d: %s", idx, exc)
            return None
        return _hook

    def collect(self) -> Dict[int, _BlockProbe]:
        return dict(self._probes)

    def reset(self) -> None:
        self._probes = {}

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []


def _eager_attention_stats(
    ca,
    normed_x: torch.Tensor,        # [B, L, dim]（memory_cross_attn 的输入，已 norm 过的 x）
    memory_key_states: torch.Tensor,  # [B, K, dim]，pose_emb（K 投影前）
    tier_ids: Optional[torch.Tensor],
) -> Tuple[float, float, int]:
    """在 hook 内 eager 重算注意力权重（flash_attention 不返回权重），报告：
      - 平均熵(nats)：对所有 query token、所有 head 的 softmax 分布求熵后取均值；
        均匀塌缩 → 熵≈ln(K)。
      - 平均最大权重：每个 (head, query) 分布的最大权重求均值。
      - K：记忆 token 数。

    复算与 MemoryCrossAttention.forward 完全同口径（task 注意点 7）：
      q = norm_q(q(normed_x))，k = norm_k(k(memory_key_states))（+tier_emb if tier_ids）。
      flash_attention 的 q,k 布局为 [B,L,H,D]——这里 view 成 [B,L,H,D] 后转置到
      [B,H,L,D]/[B,H,K,D]，按 head 维正确计算。softmax_scale = 1/√head_dim（与
      flash_attention 默认 head_dim**-0.5 一致）。数值用 float32 防溢出。
    """
    with torch.no_grad():
        target_dtype = ca.q.weight.dtype
        x = normed_x.to(target_dtype)
        mk = memory_key_states.to(target_dtype)

        B, L, _ = x.shape
        K = mk.shape[1]
        H, D = ca.num_heads, ca.head_dim

        # 与 forward 同口径的投影 + QK-norm
        q = ca.norm_q(ca.q(x)).view(B, L, H, D)
        k = ca.norm_k(ca.k(mk)).view(B, K, H, D)

        # Innovation 10：tier embedding 叠加到 K（与 forward 一致）
        if tier_ids is not None:
            try:
                _te = ca.tier_emb(tier_ids.to(x.device))          # [K, dim]
                _te = _te.unsqueeze(0).expand(B, -1, -1)           # [B, K, dim]
                k = k + _te.view(B, K, H, D)
            except Exception:  # noqa: BLE001
                pass

        # [B,L,H,D] → [B,H,L,D] / [B,H,K,D]，float32 算 softmax 防溢出
        q = q.permute(0, 2, 1, 3).float()    # [B,H,L,D]
        k = k.permute(0, 2, 1, 3).float()    # [B,H,K,D]
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,L,K]
        attn = torch.softmax(scores, dim=-1)                   # [B,H,L,K]

        # 熵(nats)：-Σ p log p，对 (B,H,L) 求均值
        eps = 1e-9
        ent = -(attn * (attn + eps).log()).sum(dim=-1)         # [B,H,L]
        attn_entropy = float(ent.mean().item())
        attn_max = float(attn.max(dim=-1).values.mean().item())
    return attn_entropy, attn_max, int(K)


# ---------------------------------------------------------------------------
# timestep 整数 → 调度 (sigma, t)（forward-only，确定映射，不随机采样）
# ---------------------------------------------------------------------------

def _resolve_timesteps(
    schedule, timesteps_str: str, model_type: str,
) -> List[Tuple[int, float, torch.Tensor]]:
    """把 --timesteps 的整数（0..999）映射到 FlowMatchingSchedule 的最近调度索引，
    确定取出 (sigma, t)。返回 [(req_timestep, sigma, t_tensor)]。

    与 train_v4 FlowMatchingSchedule 同源：sigmas / timesteps_schedule 已预计算；
    诊断不随机 sample_timestep，而是按用户指定的噪声水平确定地取（paired 可复现）。
    """
    ts_sched = schedule.timesteps_schedule  # [num_train_timesteps]，= sigmas * 1000
    out: List[Tuple[int, float, torch.Tensor]] = []
    for tok in timesteps_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            req = int(float(tok))
        except ValueError:
            logger.warning("非法 timestep '%s'，跳过", tok)
            continue
        # 最近调度索引（按 |timesteps_schedule - req| 最小）
        idx = int(torch.argmin((ts_sched - float(req)).abs()).item())
        sigma = float(schedule.sigmas[idx].item())
        t = ts_sched[idx].clone()
        out.append((req, sigma, t))
        logger.info("timestep 请求=%d → 调度 idx=%d (sigma=%.4f, t=%.2f)",
                    req, idx, sigma, float(t.item()))
    return out


# ---------------------------------------------------------------------------
# 单次 forward（给定 mode × timestep）→ 采集 per-block 诊断量
# ---------------------------------------------------------------------------

def _forward_once(
    trainer,
    model,
    instrument: MemoryDiagInstrument,
    ep: EpisodeData,
    point: RevisitPoint,
    frames: np.ndarray,             # [T,3,H,W] in [-1,1]
    video_latent: torch.Tensor,     # [16, lat_f, lat_h, lat_w]（外层 encode 一次，三模式/多timestep 复用）
    context,                        # encode_text 结果（复用）
    y: torch.Tensor,                # prepare_y 结果（复用）
    dit_cond_dict_target: Dict,     # control signal（复用）
    seq_len: int,
    kv: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    sigma: float,
    t: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> Dict[int, _BlockProbe]:
    """对一个 (mode, timestep) 跑**一次** model.forward（forward-only，无采样/无 decode）。

    三模式 paired：noisy_latent = (1-sigma)·video_latent + sigma·noise 用**外层共享**的
    同一 noise 与同一 (sigma,t)，唯一差异 = 是否注入 K/V（kv=None 即 off）。

    Args:
        kv: (K[k,dim], V[k,dim], tier_ids[k]|None)；off 模式传 None。
    Returns:
        {block_idx: _BlockProbe}（仪表化采集结果）。
    """
    instrument.reset()

    noise = noise.to(device=video_latent.device, dtype=video_latent.dtype)
    noisy_latent = (1.0 - sigma) * video_latent + sigma * noise
    t_in = t.to(device).unsqueeze(0)

    memory_states = None
    memory_value_states = None
    if kv is not None:
        key_states, value_states, tier_ids = kv
        memory_states = key_states.unsqueeze(0).to(device)         # [1,K,dim]
        memory_value_states = value_states.unsqueeze(0).to(device)  # [1,K,dim]
        # tier_ids 通过 dit_cond_dict 注入（与 model_with_memory.py:118 消费方一致）
        from memory_module.model_with_memory import _TIER_IDS_KEY
        dit_cond_dict_target = dict(dit_cond_dict_target)
        if tier_ids is not None:
            dit_cond_dict_target[_TIER_IDS_KEY] = tier_ids.to(device)

    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(
                    [noisy_latent],
                    t=t_in,
                    context=context,
                    seq_len=seq_len,
                    y=[y],
                    dit_cond_dict=dit_cond_dict_target,
                    memory_states=memory_states,
                    memory_value_states=memory_value_states,
                )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("OOM during diag forward ep=%s q=%d; 返回部分采集",
                       ep.episode_id, point.query_frame)

    return instrument.collect()


def _prepare_forward_inputs(
    trainer, model, ep: EpisodeData, point: RevisitPoint,
    frames: np.ndarray, args, device: torch.device,
):
    """构造该重访 target clip 的前向输入（与 eval_ablation._forward_loss_one_condition
    的输入构造段逐行对齐）：encode_video → prepare_y → prepare_control_signal → encode_text。

    三模式 × 多 timestep 共享同一组输入（只换 noise/timestep/注入 K/V），保证 paired。

    Returns:
        (video_latent, context, y, dit_cond_dict_target, seq_len)
    """
    poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(
        ep, point.query_frame, args.frame_num
    )
    end = min(seg_start + args.frame_num, frames.shape[0])
    video_np = frames[seg_start:end]
    if video_np.shape[0] < args.frame_num:
        pad_n = args.frame_num - video_np.shape[0]
        video_np = np.concatenate(
            [video_np, np.tile(video_np[-1:], (pad_n, 1, 1, 1))], axis=0)
    video = torch.from_numpy(video_np).permute(1, 0, 2, 3).contiguous().to(device)
    h, w = video.shape[2], video.shape[3]

    poses = torch.from_numpy(poses_c).float()
    actions = torch.from_numpy(acts_c).float()
    intrinsics = torch.from_numpy(intr_c).float()

    with torch.no_grad():
        video_latent = trainer.encode_video(video)        # [16, lat_f, lat_h, lat_w]
    lat_f, lat_h, lat_w = (
        video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
    )
    seq_len = lat_f * lat_h * lat_w // (trainer.patch_size[1] * trainer.patch_size[2])

    with torch.no_grad():
        context = trainer.encode_text(args.prompt)
        context = [c.to(torch.bfloat16)
                   if hasattr(c, "dtype") and c.dtype != torch.bfloat16 else c
                   for c in context]
        y = trainer.prepare_y(video, video_latent)

    dit_cond_dict_target = trainer.prepare_control_signal(
        poses, actions, intrinsics, h, w, lat_f, lat_h, lat_w
    )
    return video_latent, context, y, dit_cond_dict_target, seq_len


# ---------------------------------------------------------------------------
# 跨模式张量 sanity（嫌疑 A 的直接检验，task 硬性要求 4）
# ---------------------------------------------------------------------------

def _kv_tensor_sanity(
    oracle_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    wrong_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict:
    """报告 oracle/wrong 的 K/V 张量差异 + off 确认 None。

    包含：||V_oracle−V_wrong||、cosine(V_oracle,V_wrong)、||K_oracle−K_wrong||、
    每模式 ||V||、||K||、记忆 token 数 K，以及 off 模式 memory_states=None 的确认。
    """
    def _norm(x: Optional[torch.Tensor]) -> float:
        return float(x.float().norm().item()) if x is not None else float("nan")

    def _flat_cos(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
        if a is None or b is None:
            return float("nan")
        af, bf = a.float().reshape(-1), b.float().reshape(-1)
        n = min(af.numel(), bf.numel())
        if n == 0:
            return float("nan")
        return float(F.cosine_similarity(af[:n], bf[:n], dim=0).item())

    def _flat_l2(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
        if a is None or b is None:
            return float("nan")
        af, bf = a.float().reshape(-1), b.float().reshape(-1)
        n = min(af.numel(), bf.numel())
        if n == 0:
            return float("nan")
        return float((af[:n] - bf[:n]).norm().item())

    k_o = oracle_kv[0] if oracle_kv is not None else None
    v_o = oracle_kv[1] if oracle_kv is not None else None
    k_w = wrong_kv[0] if wrong_kv is not None else None
    v_w = wrong_kv[1] if wrong_kv is not None else None

    sanity = {
        "off_memory_states_is_none": True,   # off 模式恒不注入（_forward_once kv=None）
        "K_oracle": int(k_o.shape[0]) if k_o is not None else 0,
        "K_wrong": int(k_w.shape[0]) if k_w is not None else 0,
        "norm_K_oracle": _norm(k_o),
        "norm_V_oracle": _norm(v_o),
        "norm_K_wrong": _norm(k_w),
        "norm_V_wrong": _norm(v_w),
        "l2_K_oracle_minus_wrong": _flat_l2(k_o, k_w),
        "l2_V_oracle_minus_wrong": _flat_l2(v_o, v_w),
        "cosine_V_oracle_wrong": _flat_cos(v_o, v_w),
        "cosine_K_oracle_wrong": _flat_cos(k_o, k_w),
    }
    return sanity


# ---------------------------------------------------------------------------
# 聚合 + verdict（task 硬性要求 5）
# ---------------------------------------------------------------------------

def _aggregate_probes(probes: Dict[int, _BlockProbe]) -> Dict:
    """对所有 block 取均值，得到 per-mode/timestep 的聚合 gate/||out||/贡献比/注意力熵。"""
    def _mean(attr: str) -> float:
        vals = [getattr(p, attr) for p in probes.values()
                if isinstance(getattr(p, attr), float) and np.isfinite(getattr(p, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    Ks = [p.K for p in probes.values() if p.K > 0]
    return {
        "n_blocks": len(probes),
        "gate_mean": _mean("gate"),
        "out_norm_mean": _mean("out_norm"),
        "contrib_norm_mean": _mean("contrib_norm"),
        "contrib_ratio_mean": _mean("contrib_ratio"),
        "x_norm_mean": _mean("x_norm"),
        "attn_entropy_mean": _mean("attn_entropy"),
        "attn_max_mean": _mean("attn_max"),
        "K": int(np.round(np.mean(Ks))) if Ks else 0,
    }


def _delta_x_between_modes(
    probes_a: Dict[int, _BlockProbe],
    probes_b: Dict[int, _BlockProbe],
) -> float:
    """逐 block 计算 ||Δx_a − Δx_b||（Δx = 该 block 注入记忆后加到 x 的贡献 = gate*out），
    跨所有 block 取 L2 后求均值。off 模式 Δx=0（不注入），故传 None probes 时按 0 处理。

    Δx_a / Δx_b 来自同一 noise/timestep 的 paired forward → 差异纯由注入 K/V 不同导致。
    """
    diffs: List[float] = []
    idxs = set(probes_a.keys()) | set(probes_b.keys())
    for idx in idxs:
        va = probes_a.get(idx)
        vb = probes_b.get(idx)
        ta = va.contrib_vec if (va is not None and va.contrib_vec is not None) else None
        tb = vb.contrib_vec if (vb is not None and vb.contrib_vec is not None) else None
        if ta is None and tb is None:
            continue
        if ta is None:
            diffs.append(float(tb.float().norm().item()))
            continue
        if tb is None:
            diffs.append(float(ta.float().norm().item()))
            continue
        n = min(ta.numel(), tb.numel())
        diffs.append(float((ta.float().reshape(-1)[:n] - tb.float().reshape(-1)[:n]).norm().item()))
    return float(np.mean(diffs)) if diffs else float("nan")


def _make_verdict(sanity: Dict, agg_oracle: Dict, ln_K: float) -> List[str]:
    """启发式自动判读（task 硬性要求 5c）。"""
    verdicts: List[str] = []
    cr = agg_oracle.get("contrib_ratio_mean", float("nan"))
    if np.isfinite(cr) and cr < 0.01:
        verdicts.append(
            f"贡献比≈0（contrib_ratio_mean={cr:.4f}）→ 记忆对隐状态近零影响"
            "（gate 关死或 out 太小）。")
    cos_v = sanity.get("cosine_V_oracle_wrong", float("nan"))
    if np.isfinite(cos_v) and cos_v > 0.99:
        verdicts.append(
            f"cosine(V_oracle,V_wrong)={cos_v:.4f}>0.99 → ⚠️ BUG：三模式注入了相同 V"
            "（生成一致是代码 bug，非机制问题）。")
    ent = agg_oracle.get("attn_entropy_mean", float("nan"))
    if np.isfinite(ent) and np.isfinite(ln_K) and ln_K > 0 and abs(ent - ln_K) < 0.05 * ln_K:
        verdicts.append(
            f"注意力熵≈ln(K)（entropy_mean={ent:.4f}, ln(K)={ln_K:.4f}）→ "
            "注意力塌成均匀（永远读记忆帧平均，无区分力）。")
    if not verdicts:
        verdicts.append(
            "未触发上述三条启发式阈值：注入 K/V 三模式不同、贡献比非零、注意力非均匀；"
            "若生成仍一致，瓶颈可能在更下游（V 表征 / 残差被后续层抹平），需进一步分析。")
    return verdicts


# ---------------------------------------------------------------------------
# 输出（diag.json + diag.md）
# ---------------------------------------------------------------------------

def _write_outputs(args, results: Dict, ln_K_by_point: Dict) -> None:
    """写 diag.json（机器可读）+ diag.md（人读）。

    diag.md 含：(a) 跨模式张量差异表（bug 检验）；(b) per-mode 聚合（按 block 均值，分 timestep）；
    (c) Δx 跨模式差异；(d) 自动 verdict。
    """
    json_path = os.path.join(args.output_dir, "diag.json")
    with open(json_path, "w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": results,
        }, fh, indent=2, default=_json_default)
    logger.info("Wrote diag.json: %s", json_path)

    md: List[str] = []
    md.append("# Exp2 注入 forward-only 诊断 Summary\n\n")
    md.append(f"- timestamp: {datetime.now().isoformat()}\n")
    md.append(f"- ft_model_dir: {args.ft_model_dir}\n")
    md.append(f"- model_type: {args.model_type} | tier_config: {args.tier_config}\n")
    md.append(f"- metadata: {args.metadata} | timesteps: {args.timesteps}\n")
    md.append("- 本脚本是 **forward-only 诊断**：每个 (mode,timestep) 只 forward 一次，"
              "不采样/不 decode/不算 SSIM。\n\n")
    if not args.ft_model_dir:
        md.append("> ⚠️ 未提供 --ft_model_dir：memory_cross_attn 为随机初始化，结果仅作"
                  "负对照（仍能检验嫌疑 A 是否为代码 bug）。\n\n")

    for key, res in results.items():
        ep_id = res["episode_id"]
        q = res["query_frame"]
        ln_K = ln_K_by_point.get(key, float("nan"))
        md.append(f"## 重访点 ep={ep_id} q={q}（K={res['sanity']['K_oracle']}, "
                  f"ln(K)={ln_K:.4f}）\n\n")

        # (a) 跨模式张量 sanity（嫌疑 A）
        s = res["sanity"]
        md.append("### (a) 跨模式张量差异（嫌疑 A：三模式是否喂了相同 K/V）\n\n")
        md.append("| 量 | 值 |\n|---|---|\n")
        md.append(f"| off memory_states is None | {s['off_memory_states_is_none']} |\n")
        md.append(f"| K_oracle / K_wrong (token 数) | {s['K_oracle']} / {s['K_wrong']} |\n")
        md.append(f"| \\|\\|K_oracle\\|\\| / \\|\\|K_wrong\\|\\| | "
                  f"{s['norm_K_oracle']:.4f} / {s['norm_K_wrong']:.4f} |\n")
        md.append(f"| \\|\\|V_oracle\\|\\| / \\|\\|V_wrong\\|\\| | "
                  f"{s['norm_V_oracle']:.4f} / {s['norm_V_wrong']:.4f} |\n")
        md.append(f"| \\|\\|K_oracle − K_wrong\\|\\| | {s['l2_K_oracle_minus_wrong']:.4f} |\n")
        md.append(f"| \\|\\|V_oracle − V_wrong\\|\\| | {s['l2_V_oracle_minus_wrong']:.4f} |\n")
        md.append(f"| cosine(K_oracle, K_wrong) | {s['cosine_K_oracle_wrong']:.4f} |\n")
        md.append(f"| cosine(V_oracle, V_wrong) | {s['cosine_V_oracle_wrong']:.4f} |\n\n")

        # (b) per-mode 聚合（分 timestep）
        md.append("### (b) per-mode 聚合（按 40 block 取均值，分 timestep）\n\n")
        md.append("| timestep | mode | gate | \\|\\|out\\|\\| | 贡献norm | 贡献比 | "
                  "\\|\\|x\\|\\| | attn熵(nats) | attn_max | K |\n")
        md.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for ts_block in res["per_timestep"]:
            ts = ts_block["req_timestep"]
            for mode in ("oracle", "off", "wrong"):
                a = ts_block["modes"].get(mode)
                if a is None:
                    continue
                md.append(
                    f"| {ts} | {mode} | {a['gate_mean']:.4f} | {a['out_norm_mean']:.4f} | "
                    f"{a['contrib_norm_mean']:.4f} | {a['contrib_ratio_mean']:.4f} | "
                    f"{a['x_norm_mean']:.2f} | {a['attn_entropy_mean']:.4f} | "
                    f"{a['attn_max_mean']:.4f} | {a['K']} |\n"
                )
        md.append("\n")

        # (c) Δx 跨模式差异
        md.append("### (c) 注入后隐状态差 Δx（paired 同 noise/timestep，按 block 均值）\n\n")
        md.append("Δx = 该 block 注入记忆后加到 x 的贡献（= gate*out）；off 模式 Δx=0。\n\n")
        md.append("| timestep | \\|\\|Δx_oracle − Δx_off\\|\\| | \\|\\|Δx_oracle − Δx_wrong\\|\\| |\n")
        md.append("|---|---|---|\n")
        for ts_block in res["per_timestep"]:
            dx = ts_block["delta_x"]
            md.append(f"| {ts_block['req_timestep']} | "
                      f"{dx['oracle_vs_off']:.4f} | {dx['oracle_vs_wrong']:.4f} |\n")
        md.append("\n")

        # (d) verdict
        md.append("### (d) 自动判读 verdict（启发式）\n\n")
        for v in res["verdict"]:
            md.append(f"- {v}\n")
        md.append("\n")

    md_path = os.path.join(args.output_dir, "diag.md")
    with open(md_path, "w") as fh:
        fh.writelines(md)
    logger.info("Wrote diag.md: %s", md_path)


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().float().cpu().tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # 设置 oracle_injection 模块级全局（_build_oracle_memory_kv 读取分辨率/clip 长度）
    _oracle_mod._SIZE_HW = tuple(args.size.split("*"))   # ("480","832")
    _oracle_mod._ORACLE_CLIP_FRAMES = args.frame_num

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（前向会非常慢；注意力 eager 重算仍可跑）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))

    logger.info("Args: %s", vars(args))
    logger.info(
        "forward-only 诊断：每 (mode,timestep) 只 forward 一次；嫌疑 A=三模式 K/V 是否相同；"
        "嫌疑 B=注意力是否塌成均匀（熵≈ln(K)）；量化注入贡献比。")

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

    # ---- 加载 trainer + model（复用 eval_ablation = train_v4 load_models + ft 加载）----
    logger.info("Loading trainer + model (eval_ablation._build_trainer_and_model)...")
    trainer, model = _build_trainer_and_model(args, device)
    _log_memory_weight_sanity(model)

    # ---- 挂 hook 仪表化全部 memory_cross_attn ----
    instrument = MemoryDiagInstrument(model)

    # ---- timestep 整数 → 调度 (sigma, t)（确定映射，不随机采样）----
    ts_list = _resolve_timesteps(trainer.schedule, args.timesteps, args.model_type)
    if not ts_list:
        logger.error("无有效 timestep，退出。")
        instrument.remove()
        return

    results: Dict[str, Dict] = {}
    ln_K_by_point: Dict[str, float] = {}

    try:
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

            # 解码 video + VAE encode（oracle/wrong V + target clip latent 都要用）
            try:
                frames = _decode_episode_video(ep, height=height, width=width)
                latents_full = _vae_encode_batched(trainer.vae, frames, device=device,
                                                   batch_frames=args.vae_batch)
                latents_per_frame = _expand_latents_to_frames(latents_full, T)
                del latents_full
            except Exception as exc:  # noqa: BLE001
                logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
                continue

            for pt in points:
                # 单点抗崩：任何环节失败只记录 + continue，不中断整轮
                try:
                    key = f"{ep_id}::q{pt.query_frame}"

                    # ---- 构造 oracle / wrong K/V（复用 oracle_injection，不重写）----
                    oracle_indices = _pick_oracle_indices(pt, args.num_oracle_frames, T)
                    oracle_kv2 = _build_oracle_memory_kv(
                        model, None, ep, latents_per_frame, oracle_indices, device)
                    wrong_indices = _pick_wrong_indices(pt, args.num_oracle_frames, T, rng)
                    wrong_kv2 = _build_oracle_memory_kv(
                        model, None, ep, latents_per_frame, wrong_indices, device)

                    # 嫌疑 A：跨模式 K/V 张量 sanity
                    sanity = _kv_tensor_sanity(oracle_kv2, wrong_kv2)
                    K_for_lnK = max(sanity["K_oracle"], sanity["K_wrong"], 1)
                    ln_K = float(math.log(K_for_lnK)) if K_for_lnK > 0 else float("nan")
                    ln_K_by_point[key] = ln_K

                    # oracle_only：tier_ids 全标 long=2（与 oracle_injection 一致）
                    def _with_tier(kv):
                        if kv is None:
                            return None
                        k, v = kv
                        tids = torch.full((k.shape[0],), 2, dtype=torch.long)
                        return (k, v, tids)
                    oracle_kv = _with_tier(oracle_kv2)
                    wrong_kv = _with_tier(wrong_kv2)

                    # ---- 共享前向输入（三模式 × 多 timestep 复用）----
                    (video_latent, context, y, dit_cond_dict_target, seq_len
                     ) = _prepare_forward_inputs(
                        trainer, model, ep, pt, frames, args, device)

                    per_timestep: List[Dict] = []
                    for (req_ts, sigma, t) in ts_list:
                        # 三模式共享同一 noise（paired）
                        torch.manual_seed(args.seed + req_ts)
                        noise = torch.randn_like(video_latent)

                        mode_aggs: Dict[str, Dict] = {}
                        mode_probes: Dict[str, Dict[int, _BlockProbe]] = {}
                        for mode, kv in (("oracle", oracle_kv),
                                         ("off", None),
                                         ("wrong", wrong_kv)):
                            probes = _forward_once(
                                trainer, model, instrument, ep, pt, frames,
                                video_latent, context, y, dit_cond_dict_target,
                                seq_len, kv, sigma, t, noise, device,
                            )
                            mode_probes[mode] = probes
                            mode_aggs[mode] = _aggregate_probes(probes)

                        # (c) Δx 跨模式差异（paired 同 noise/timestep）
                        delta_x = {
                            "oracle_vs_off": _delta_x_between_modes(
                                mode_probes["oracle"], mode_probes["off"]),
                            "oracle_vs_wrong": _delta_x_between_modes(
                                mode_probes["oracle"], mode_probes["wrong"]),
                        }
                        per_timestep.append({
                            "req_timestep": req_ts,
                            "sigma": sigma,
                            "modes": mode_aggs,
                            "delta_x": delta_x,
                        })
                        logger.info(
                            "ep=%s q=%d t=%d | oracle: gate=%.4f 贡献比=%.4f 熵=%.4f | "
                            "Δx(o-off)=%.4f Δx(o-wrong)=%.4f",
                            ep_id, pt.query_frame, req_ts,
                            mode_aggs["oracle"]["gate_mean"],
                            mode_aggs["oracle"]["contrib_ratio_mean"],
                            mode_aggs["oracle"]["attn_entropy_mean"],
                            delta_x["oracle_vs_off"], delta_x["oracle_vs_wrong"],
                        )

                    # verdict 用第一个 timestep 的 oracle 聚合（贡献比/熵在不同 t 趋势一致）
                    agg_for_verdict = (per_timestep[0]["modes"]["oracle"]
                                       if per_timestep else {})
                    verdict = _make_verdict(sanity, agg_for_verdict, ln_K)

                    results[key] = {
                        "episode_id": ep_id,
                        "query_frame": int(pt.query_frame),
                        "first_visit_frame": int(pt.first_visit_frame),
                        "oracle_indices": [int(x) for x in oracle_indices],
                        "wrong_indices": [int(x) for x in wrong_indices],
                        "sanity": sanity,
                        "per_timestep": per_timestep,
                        "verdict": verdict,
                    }
                    del video_latent, context, y, dit_cond_dict_target
                except Exception as exc:  # noqa: BLE001
                    logger.exception("重访点诊断失败 ep=%s q=%d: %s",
                                     ep_id, pt.query_frame, exc)
                    continue

            del frames, latents_per_frame
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        instrument.remove()

    if not results:
        logger.error("无任何重访点产出诊断结果；退出。")
        return

    _write_outputs(args, results, ln_K_by_point)
    logger.info("Done. 输出目录: %s", args.output_dir)


if __name__ == "__main__":
    main()
