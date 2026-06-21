"""
memory_encoder.py — v5 in-context KV 记忆方案的唯一可训练件（experiment_design Step 40 / S-V1）

把检索到的过去帧 VAE latent 编码成骨干 hidden 空间的 memory token，供下游
（MemorySelfAttention，下一个组件）走骨干**冻结**的 self.k / self.v 当只读 K/V 源。

设计依据（已锁定）：
  - decisions.md「讨论 8」+ open_problems.md「OP-5」：
    v5 = in-context KV、无 gate、方案A（检索帧 VAE latent → 骨干 hidden token，尽量在分布内）、
    只读、全冻骨干；**唯一可训练件 = 本 MemoryEncoder**。
  - 20260621_v5_backbone_interface_modeA.md：
    §2 做法(b) + patch_embedding 前 z_dim 通道切片初始化（分布内起步 + 可训练）；
    §3 每帧空间下采样到 g×g（g=8→64 token/帧，6 帧≈384 memory token，落在 DiT-Mem 量级）。

本文件只负责 latent → memory token。**不放 gate、不做注入**（注入在 MemorySelfAttention 里）。

== 容量取舍说明 ==
输入投影若只用单层 Linear(z_dim→dim)，表达力不足：它是 v5 唯一的适配面，要扛起把
检索 latent 投到骨干 hidden 分布、并喂进**冻结** W_k / W_v 的全部工作。研究侧 DiT-Mem 的
memory encoder ~150M 参数。这里在输入投影后接 N 层带残差的 Transformer 块
（每块 = LayerNorm + 多头 self-attn 残差 + LayerNorm + MLP(dim→4dim→dim) 残差），
让 memory token 之间能交换空间上下文，并提供足够容量对齐 hidden 空间。
层数可配：层越多容量越大但越易在冷启动扰动冻结骨干（无 gate）。**默认 depth=1（精简起步，
grid=16 时约 315M 参数，接近 DiT-Mem ~150M 锚点的合理上限）**——Orchestrator 裁决：数据有限
+ revisit 稀疏，先精简、欠拟合再加层；depth 保持可配，可上调到 2~4 增容量。

== 产出契约（供下游组件 / project_map）==
  forward(latents [K, z_dim, h, w]) -> memory_tokens [K, grid*grid, dim]
  init_from_patch_embedding(patch_embed_weight [dim, in_dim, 1, ph, pw]) -> None（原地初始化输入投影）

调用方 model_with_memory_v5（下一个组件）在 from_wan_model 时调用
init_from_patch_embedding(backbone.patch_embedding.weight)，再把 retrieve 出的 latent
喂进 forward，输出当只读 memory K/V 源。
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from torch import Tensor

logger = logging.getLogger(__name__)


def _build_2d_sincos_pos_emb(grid_size: int, dim: int) -> Tensor:
    """构造固定（非学习）2D sin-cos 位置编码，形状 [grid_size*grid_size, dim]。

    仿照 model_with_memory.py 的同名函数（保持行/列编码约定一致）：
    行向量 [:dim/2] 编码 h（行），[dim/2:] 编码 w（列）。要求 dim % 4 == 0。

    用 register_buffer（非 nn.Parameter）持有：DeepSpeed ZeRO-3 会把裸 nn.Parameter
    分片到各 rank（forward 之外访问时 .data 为 size-0），而 buffer 在所有 rank 复制、
    永不分片，可安全访问。固定 sin-cos 仍给每个 patch 唯一位置，告知模型 token 的空间布局。
    """
    assert dim % 4 == 0, f"_build_2d_sincos_pos_emb 需要 dim 能被 4 整除，got {dim}"

    def _1d(d: int, pos: Tensor) -> Tensor:
        # d 为该轴输出维度（偶数）；pos: [M] → 返回 [M, d]
        omega = torch.arange(d // 2, dtype=torch.float32) / (d / 2.0)
        omega = 1.0 / (10000.0 ** omega)                        # [d/2]
        out = torch.einsum("m,k->mk", pos.reshape(-1), omega)   # [M, d/2]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [M, d]

    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(coords, coords, indexing="ij")  # 行=h, 列=w
    emb_h = _1d(dim // 2, grid_h)   # [g*g, dim/2]
    emb_w = _1d(dim // 2, grid_w)   # [g*g, dim/2]
    return torch.cat([emb_h, emb_w], dim=1)  # [g*g, dim]


class _ResidualTransformerBlock(nn.Module):
    """一层 pre-norm Transformer 块（self-attn 残差 + MLP 残差）。

    pre-norm（LayerNorm 在子层之前）便于深层稳定训练。MLP 隐藏维 4*dim 为标准扩张比。
    self-attn 让同一帧的 g*g 个空间 token 互相交换上下文；batch_first=True 直接吃
    [K, grid*grid, dim]。

    == 恒等起步（无 gate 下的冷启动护栏）==
    v5 设计前提是**无 gate**，冷启动唯一的护栏是 in_proj 的「分布内切片初始化」。但若
    残差块随机初始化，两条残差分支会在 init 时立刻往输出里注入随机量，稀释掉 in_proj
    的分布内起步、扰动冻结骨干（ReviewAgent WARN）。
    解法：把两条残差分支的**输出投影零初始化**——
      - self-attn 的 `attn.out_proj`（weight + bias）置 0；
      - MLP 末层 Linear（weight + bias）置 0。
    于是 init 时 `attn_out ≡ 0`、`mlp(...) ≡ 0`，整块 forward 退化为 `x → x`（恒等）。
    encoder 起步输出 == in_proj(切片初始化) + pos_emb，纯分布内；训练中两条分支从 0
    自然长大（ReZero / zero-conv 原理），无需引入 gate。
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

        # 恒等起步：两条残差分支的输出投影零初始化（见类 docstring）。
        # 不改参数量；训练中分支自 0 长大。
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        _mlp_last = self.mlp[-1]  # MLP 末层 Linear(mlp_ratio*dim → dim)
        nn.init.zeros_(_mlp_last.weight)
        if _mlp_last.bias is not None:
            nn.init.zeros_(_mlp_last.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MemoryEncoder(nn.Module):
    """检索帧 VAE latent → 骨干 hidden 空间的 memory token（v5 唯一可训练件）。

    Pipeline（forward）：
      latents [K, z_dim, h, w]
        --(adaptive_avg_pool2d to grid×grid)--> [K, z_dim, grid, grid]   # 空间下采样
        --(flatten spatial)--> [K, grid*grid, z_dim]
        --(in_proj Linear z_dim→dim)--> [K, grid*grid, dim]              # 唯一分布内起步面
        --(+ 2D sin-cos pos emb, 可选)--> [K, grid*grid, dim]
        --(N×ResidualTransformerBlock)--> [K, grid*grid, dim] = memory_tokens

    Args:
        z_dim:       VAE latent 通道数（默认 16）。
        dim:         骨干 hidden 维（默认 5120）。
        grid:        每帧池化到的空间网格边长，输出 grid*grid token/帧
                     （默认 16→256；OOM fallback 用 8→64）。
        depth:       残差 Transformer 块层数（默认 1，精简起步；可上调到 2~4 增容量，见模块 docstring）。
        num_heads:   编码头 self-attn 头数（默认 8）。
        add_pos_emb: 是否叠加 grid×grid 的 2D 位置编码（默认 True）。
        eps:         LayerNorm epsilon。
    """

    def __init__(
        self,
        z_dim: int = 16,
        dim: int = 5120,
        grid: int = 16,
        depth: int = 1,
        num_heads: int = 8,
        add_pos_emb: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.grid = grid
        self.add_pos_emb = add_pos_emb

        # 输入投影：唯一分布内起步面，由 init_from_patch_embedding 用骨干切片初始化。
        # bias=False 与骨干 patch_embedding 行为不完全相同（Conv3d 默认有 bias），但
        # 切片初始化只取 weight；bias 设为 0 可学，避免引入未对齐的常量偏置。
        self.in_proj = nn.Linear(z_dim, dim, bias=False)

        # 编码头：N 层带残差的 Transformer 块（容量主体）
        self.blocks = nn.ModuleList(
            [_ResidualTransformerBlock(dim, num_heads=num_heads, eps=eps) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(dim, eps=eps)

        # 2D 位置编码：register_buffer（非 Parameter），ZeRO-3 安全（见 _build_2d_sincos_pos_emb）。
        # persistent=False：由 (grid, dim) 完全确定，构造时重算，不写入 checkpoint。
        if add_pos_emb:
            self.register_buffer(
                "pos_emb",
                _build_2d_sincos_pos_emb(grid, dim),  # [grid*grid, dim]
                persistent=False,
            )
        else:
            self.pos_emb = None

        logger.info(
            "MemoryEncoder: z_dim=%d, dim=%d, grid=%d (%d token/frame), depth=%d, "
            "num_heads=%d, add_pos_emb=%s",
            z_dim, dim, grid, grid * grid, depth, num_heads, add_pos_emb,
        )

    @torch.no_grad()
    def init_from_patch_embedding(self, patch_embed_weight: Tensor) -> None:
        """用骨干 patch_embedding 的前 z_dim 输入通道切片初始化输入投影（分布内起步）。

        见 20260621_v5_backbone_interface_modeA.md §2 做法(b)：骨干
          patch_embedding = nn.Conv3d(in_dim, dim, kernel=(1,ph,pw), stride=(1,ph,pw))
          weight shape = [dim, in_dim, 1, ph, pw]
        取前 z_dim 个输入通道 `[:, :z_dim]` → [dim, z_dim, 1, ph, pw]，再对时间(1)与空间
        (ph,pw) 核维度求均值，压成 [dim, z_dim] 给 in_proj.weight。

        语义：等价于「干净 latent 走 noisy 通道槽、其余条件/掩码槽置 0」的投影，且把
        Conv3d 在 patch 内的空间核做平均（本编码器先 adaptive_avg_pool2d 池化，已无 patch
        内空间结构，故对核取均值是与池化一致的折叠）。这让 memory token 起步即落在骨干
        hidden 分布内，缓解无 gate 冷启动对冻结骨干的扰动（报告 R1）。

        调用方：model_with_memory_v5 在 from_wan_model 时调用：
            mem_enc.init_from_patch_embedding(backbone.patch_embedding.weight)

        Args:
            patch_embed_weight: 骨干 Conv3d 权重 [dim, in_dim, 1, ph, pw]。
                                要求 in_dim >= z_dim、dim == self.dim。
        """
        w = patch_embed_weight
        assert w.dim() == 5, (
            f"init_from_patch_embedding 期望 5D Conv3d 权重 [dim, in_dim, 1, ph, pw]，"
            f"got shape {tuple(w.shape)}"
        )
        out_dim, in_dim = w.shape[0], w.shape[1]
        assert out_dim == self.dim, (
            f"patch_embedding out_dim={out_dim} 与 MemoryEncoder.dim={self.dim} 不一致"
        )
        assert in_dim >= self.z_dim, (
            f"patch_embedding in_dim={in_dim} < z_dim={self.z_dim}，无法切片前 z_dim 通道"
        )
        # [dim, z_dim, 1, ph, pw] → 对 (time, ph, pw) 核维度求均值 → [dim, z_dim]
        sliced = w[:, : self.z_dim]                 # [dim, z_dim, 1, ph, pw]
        init_w = sliced.mean(dim=(2, 3, 4))         # [dim, z_dim]
        init_w = init_w.to(self.in_proj.weight.dtype).to(self.in_proj.weight.device)
        self.in_proj.weight.copy_(init_w)
        logger.info(
            "MemoryEncoder.init_from_patch_embedding: in_proj.weight 已用 "
            "patch_embedding[:, :%d] 核均值初始化（[%d, %d]）。",
            self.z_dim, self.dim, self.z_dim,
        )

    def forward(self, latents: Tensor) -> Tensor:
        """latent → memory token。

        Args:
            latents: [K, z_dim, h, w]，K 个检索帧的 VAE latent（h/w 为 latent 空间分辨率）。

        Returns:
            memory_tokens: [K, grid*grid, dim]，骨干 hidden 空间的 memory token。
        """
        assert latents.dim() == 4, (
            f"MemoryEncoder.forward 期望 latents [K, z_dim, h, w]，got {tuple(latents.shape)}"
        )
        assert latents.shape[1] == self.z_dim, (
            f"latent 通道 {latents.shape[1]} 与 z_dim={self.z_dim} 不一致"
        )

        # dtype/device 对齐到模块权重：直接读 in_proj.weight（ZeRO-3 下 next(parameters())
        # 不可靠，见 code_standards / 报告说明）。
        w = self.in_proj.weight
        latents = latents.to(dtype=w.dtype, device=w.device)

        # 1. 空间下采样到 grid×grid（仿 get_projected_latent_emb 的 adaptive_avg_pool2d）
        pooled = torch_F.adaptive_avg_pool2d(latents, (self.grid, self.grid))  # [K, z_dim, g, g]

        # 2. 展平空间 → [K, g*g, z_dim]
        K = pooled.shape[0]
        gg = self.grid * self.grid
        tokens = pooled.reshape(K, self.z_dim, gg).transpose(1, 2)  # [K, g*g, z_dim]

        # 3. 输入投影 → [K, g*g, dim]
        x = self.in_proj(tokens)  # [K, g*g, dim]

        # 4. 叠加 2D 位置编码（告知 token 空间布局）
        if self.add_pos_emb and self.pos_emb is not None:
            x = x + self.pos_emb.to(dtype=x.dtype, device=x.device).unsqueeze(0)  # [1, g*g, dim] 广播

        # 5. 编码头（带残差 Transformer 块）
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)

        return x  # [K, g*g, dim]
