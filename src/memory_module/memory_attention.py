"""
memory_attention.py — Memory Cross-Attention Module

结构参考 lingbot-world 的 WanCrossAttention，Query 来自当前帧特征，
Key/Value 来自 MemoryBank 检索到的历史帧 pose_emb。

与 WanCrossAttention 的区别：
  - Key/Value 是 memory_states [B, K, dim]，而非文本嵌入
  - 不使用 RoPE（与 WanCrossAttention 一致）
  - 自带 RMSNorm，不依赖 lingbot-world 内部类

参考：
  - lingbot-world: wan/modules/model.py WanCrossAttention（接口风格）
  - lingbot-world: wan/modules/attention.py flash_attention（底层计算）
"""

import logging
import os
import sys

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---- sys.path（供 forward 内懒加载 flash_attention 使用）----
_LINGBOT_WORLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'refs', 'lingbot-world'
)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

# flash_attention 已移至 MemoryCrossAttention.forward() 内懒加载，
# 避免模块导入时触发 wan/__init__.py → T5EncoderModel → torch.cuda.current_device()

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """轻量 RMSNorm，不依赖 lingbot-world 内部类。

    优先使用 F.rms_norm（PyTorch ≥ 2.4 fused kernel，不在 global memory 中
    实体化 float32 张量），避免 gradient checkpoint recomputation 时 OOM。
    fallback 实现去掉命名 float32 张量，最终乘法在 bfloat16 完成以节省内存。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._use_fused = hasattr(F, 'rms_norm')

    def forward(self, x: Tensor) -> Tensor:
        if self._use_fused:
            # fused kernel：不在 global memory 中实体化 float32 张量
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        # fallback：norm scalar 在 float32 计算后立即投影回 x.dtype，
        # 最终乘法在 bfloat16（160 MiB）而非 float32（320 MiB）完成，
        # 且不命名 float32 临时张量，Python 可在 .mean() 后提前释放。
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm.to(x.dtype) * self.weight


class MemoryCrossAttention(nn.Module):
    """历史帧 Memory Cross-Attention。

    Query 来自当前帧的隐藏状态 x，Key/Value 来自 Memory Bank 检索到的历史帧。
    结构与 WanCrossAttention 保持一致（无 RoPE，支持 Flash Attention）。

    Args:
        dim:       模型隐藏维度（A14B 配置为 5120）
        num_heads: 注意力头数（A14B 配置为 40）
        qk_norm:   是否对 Q/K 做 RMSNorm，默认 True（与 lingbot-world 一致）
        eps:       归一化 epsilon
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        out_rms_cap: float = 1.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        # 注入幅度硬上界常数（非参数，不可训练）：forward 里把分支输出 per-token RMS
        # 截断到 <= out_rms_cap。见 forward 中 clip-by-RMS 说明。
        self.out_rms_cap = float(out_rms_cap)

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        # Q/K 归一化（与 WanSelfAttention 保持一致）
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # NOTE: 不对 V 做 norm —— 之前加的 norm_v 会把 value 钉成单位 RMS，抹掉「强/弱
        #   记忆」的幅度信息，可能伤注入有效性。把 V 拉进分布内的事改由后续单独工作用
        #   保幅度的方式做，这里不误伤目标③（注入有效性）。
        # NOTE: 注入幅度的硬上界不再用可训练 RMSNorm（其 weight 可被训练推大 → 假上界，
        #   且训练脚本对整个 memory 模块 requires_grad_(True) 会解冻它），改为 forward 里
        #   无参数的 clip-by-RMS 硬截断（见 forward）。

        # Flamingo 式 tanh(α) gate，α（=self.gate）为 pre-tanh logit，init=0
        # → 有效 gate = tanh(0) = 0 = 恒等（注入分支起步对冻结骨干零贡献），
        #   再由训练把 α 慢慢推大，有效 gate 在 [-1, 1] 内有界增长。
        # 动机（回链 F-25）：裸标量 init=1.0 + 无 squash 会让冻结骨干从第 1 步就被满
        #   强度注入；配合扩散 exposure bias（单步训练 loss 看不见 70 步采样里的误差
        #   累积），gate 卡在 ~1.0、推理塌成噪点。WorldMem / Flamingo / ControlNet /
        #   IP-Adapter 等能 work 的注入都让分支从恒等(≈0)起步再慢慢长大
        #   （zero-init gate / tanh(α=0) / zero-conv）。
        #
        # ⚠️ checkpoint 格式变更：本参数语义由「裸标量 = raw 乘子」改为「pre-tanh
        #   logit」。旧 checkpoint（epoch_1/2/3）的 gate 值与新模块语义不兼容（旧
        #   gate=0.99 ≠ 新 tanh(0.99)=0.76），须重训 fresh；评测旧 checkpoint 必须
        #   用对应旧 git 版本代码。
        #
        # NOTE: 参数属性名保留为 self.gate（不改名），使所有读 .gate / _last_gate_value
        #   的诊断/日志代码（wandb_utils.py / memory_injection_diag.py /
        #   train_v4_stage1_dual.py / eval_ablation.py）无需连带改动；但语义已变为
        #   logit（见上）。
        self.gate = nn.Parameter(torch.zeros(1))

        # Innovation 10：Tier Embedding — 告知模型检索帧来自 Short/Medium/Long 层
        # 0=Short（连续性锚点），1=Medium（动态事件），2=Long（稳定场景）
        # 作用于 K，不影响 Q/V；tier_ids=None 时跳过（向后兼容 v3）
        # 默认随机初始化（zero-init 不适合 Embedding）
        self.tier_emb = nn.Embedding(3, dim)

        # 运行时诊断指标（非参数，供 WandBLogger._collect_memory_diagnostics 采集）
        self._last_attn_out_norm: float = 0.0
        self._last_gate_value: float = 0.0

    def forward(
        self,
        x: Tensor,
        memory_key_states: Tensor,
        memory_value_states: Optional[Tensor] = None,
        memory_lens: Tensor = None,
        tier_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        签名从 forward(x, memory_states, memory_lens) 改为
              forward(x, memory_key_states, memory_value_states=None, memory_lens=None)

        Innovation 10（Tier Embedding）新增 tier_ids 参数（2026-04-20）。

        Args:
            x:                    [B, L, dim]  当前帧序列（Query 来源）
            memory_key_states:    [B, K, dim]  Memory Bank 检索到的历史帧 pose_emb（用于投影 K，FOV 路由）
            memory_value_states:  [B, K, dim]  Memory Bank 检索到的历史帧 visual_emb（用于投影 V，视觉内容）；
                                               若 None 则退化为 memory_key_states（向后兼容）
            memory_lens:          [B]          每个样本实际有效的 memory 帧数（用于 padding mask）
                                               若所有样本 memory 数相同可传 None
            tier_ids:             [K] int64    每帧所属层的 ID（0=Short/1=Medium/2=Long）；
                                               None 时跳过 tier embedding 叠加（向后兼容 v3）
                                               (Innovation 10: Tier Embedding)

        Returns:
            out: [B, L, dim]  memory cross-attention 的输出（残差加法前）
        """
        from wan.modules.attention import flash_attention  # lazy import：仅在 forward 调用时加载 wan

        # MODIFIED: F-03/F5 fix — 若 value_states 未提供，退化为 key_states（向后兼容）
        if memory_value_states is None:
            memory_value_states = memory_key_states

        # dtype 对齐：直接读 Linear weight dtype，避免 next(parameters()) 在 ZeRO-3 下不可靠
        target_dtype = self.q.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if memory_key_states is not None and memory_key_states.dtype != target_dtype:
            memory_key_states = memory_key_states.to(target_dtype)
        if memory_value_states is not None and memory_value_states.dtype != target_dtype:
            memory_value_states = memory_value_states.to(target_dtype)

        B, L, _ = x.shape
        if memory_key_states is None:
            return x.new_zeros(B, L, self.dim)
        K = memory_key_states.shape[1]

        # Projection + QK-norm（在 view 之前 norm，与 WanCrossAttention 一致）
        # K 来自 pose_embs（FOV 路由），V 来自 visual_embs（视觉内容）
        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        q = self.norm_q(self.q(x)).view(B, L, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(memory_key_states)).view(B, K, self.num_heads, self.head_dim)
        # V 不做 norm（保留「强/弱记忆」幅度信息，避免误伤注入有效性 = 目标③）。
        v = self.v(memory_value_states).view(B, K, self.num_heads, self.head_dim)

        # Innovation 10：Tier Embedding — 在 K 上叠加 tier 嵌入，告知模型帧来自哪一层
        # tier_ids=None 时跳过，保持向后兼容（v3 训练/推理路径不受影响）
        if tier_ids is not None:
            _tier_emb = self.tier_emb(tier_ids.to(x.device))           # [K, dim]
            _tier_emb = _tier_emb.unsqueeze(0).expand(B, -1, -1)        # [B, K, dim]
            k = k + _tier_emb.view(B, K, self.num_heads, self.head_dim)  # 叠加到 K

        # Flash Attention: [B, L, num_heads, head_dim]
        out = flash_attention(q, k, v, k_lens=memory_lens)

        # Merge heads: [B, L, dim]
        out = self.o(out.flatten(2))

        # 注入幅度硬上界（无可训练参数 → 训练对 memory 模块 requires_grad_(True) 解冻后依然生效，
        # 区别于可训练 weight 的 RMSNorm「软上界」）。把分支输出 per-token RMS 截断到 <= out_rms_cap：
        # 超过上限才按比例缩小，低于上限原样保留 —— 保留 attention 对「该注入多少」的调幅能力，
        # 不像 RMSNorm 把注入幅度钉成常数。动机回链 F-25（注入幅度/分布外是采样塌噪主因）。
        # NOTE: 不做 .float() 转换，避免 gradient checkpointing 重计算时产生额外 640MB 显存峰值。
        with torch.no_grad():
            # 诊断记录 clip 前的真实 attention 信号范数（会随训练/输入变化，
            # 用于判断注入分支是否真在学到内容；clip 后近常数，无诊断价值）。
            self._last_attn_out_norm = out.detach().norm().item()
            self._last_gate_value = torch.tanh(self.gate).item()
        _rms = out.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()   # [B, L, 1]
        out = out * (self.out_rms_cap / _rms).clamp(max=1.0)
        # Gate 缩放：有效 gate = tanh(self.gate)，self.gate 为 pre-tanh logit，
        # init=0 → 有效 gate=0（恒等起步，Flamingo α=0 原理，回链 F-25），训练中长大。
        # 真·硬上界：tanh(gate)∈[-1,1] 乘以 per-token RMS ≤ out_rms_cap 的 out，
        #   无可训练参数可把上界推大 → 注入吹不爆冻结骨干残差流。
        return torch.tanh(self.gate) * out
