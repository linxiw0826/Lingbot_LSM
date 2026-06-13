"""
model_with_memory.py — WanModel with Surprise-Driven Memory

在不修改 lingbot-world 原始代码的前提下，通过继承和包装引入 Memory 机制：
  - MemoryBlockWrapper: 包裹现有 WanAttentionBlock，在其输出后追加 Memory Cross-Attention
  - WanModelWithMemory: 继承 WanModel，将指定 blocks 替换为 MemoryBlockWrapper，
                        并添加 NFPHead；通过 dit_cond_dict 传递 memory_states

插入位置：每个被包裹的 WanAttentionBlock 完整输出之后（含 FFN），
          追加 memory_norm → MemoryCrossAttention → 残差连接。

此处对应 experiment_design.md Step 2（memory_attention.py）和 Step 3（model.py 修改）。

参考：
  - lingbot-world: wan/modules/model.py  WanModel, WanAttentionBlock, WanLayerNorm
  - WorldMem:      algorithms/worldmem/models/dit.py  memory attention 插入方式（残差+gate）
"""

import logging
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from torch import Tensor
from einops import rearrange

# ---- 引入 lingbot-world 模块 ----
_LINGBOT_WORLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'refs', 'lingbot-world'
)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

from wan.modules.model import WanModel, WanAttentionBlock, WanLayerNorm  # noqa: E402

from .memory_attention import MemoryCrossAttention  # noqa: E402
from .nfp_head import NFPHead  # noqa: E402

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# dit_cond_dict 中 memory_states 的 key（避免字符串 typo）
_MEMORY_STATES_KEY = "memory_states"
# MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
# dit_cond_dict 中 memory value states 的 key（visual_emb 投影到模型空间，用作 cross-attn V）
_MEMORY_VALUE_KEY = "__memory_value_states__"
# Innovation 10: Tier Embedding — dit_cond_dict 中 tier_ids 的 key
# tier_ids [K] int64，0=Short / 1=Medium / 2=Long；None 时 v3 向后兼容
_TIER_IDS_KEY = "__tier_ids__"


# ---------------------------------------------------------------------------
# MemoryBlockWrapper
# ---------------------------------------------------------------------------

class MemoryBlockWrapper(nn.Module):
    """包裹一个 WanAttentionBlock，在其输出后追加 Memory Cross-Attention。

    Forward 流程：
        1. block(x, **kwargs)          原始 Self-Attn + Camera FiLM + Text Cross-Attn + FFN
        2. memory_norm(x)              LayerNorm
        3. memory_cross_attn(x, M)     Memory Cross-Attention（Query=x，KV=memory_states M）
        4. x = x + output              残差连接

    当 dit_cond_dict 中不含 'memory_states' 时，跳过步骤 2-4，行为与原始 block 完全一致。

    Args:
        block:      待包裹的 WanAttentionBlock 实例（直接复用，不复制）
        dim:        模型隐藏维度
        num_heads:  注意力头数
        qk_norm:    是否对 Q/K 做 RMSNorm
        eps:        归一化 epsilon
    """

    def __init__(
        self,
        block: WanAttentionBlock,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.block = block
        self.memory_norm = WanLayerNorm(dim, eps)
        self.memory_cross_attn = MemoryCrossAttention(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, eps=eps
        )

    # N-01 fix: 将 **kwargs 改为显式参数，确保 torch.checkpoint 以位置参数调用时不崩溃
    def forward(self, x: Tensor, e, seq_lens, grid_sizes, freqs,
                context, context_lens, dit_cond_dict=None) -> Tensor:
        """
        Args:
            x:      [B, L, dim]
            e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict:
                传递给内部 WanAttentionBlock 的参数（与 WanAttentionBlock.forward 签名一致）

        Returns:
            x: [B, L, dim]  含记忆注入后的隐藏状态
        """
        # Step 1: 原始 block
        x = self.block(x, e=e, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs,
                       context=context, context_lens=context_lens, dit_cond_dict=dit_cond_dict)

        # Step 2-4: Memory Cross-Attention（仅当 memory_states 存在时）
        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        # K = memory_key_states (pose_emb，FOV 路由)，V = memory_value_states (visual_emb，视觉内容)
        if dit_cond_dict is not None and _MEMORY_STATES_KEY in dit_cond_dict:
            memory_key_states = dit_cond_dict[_MEMORY_STATES_KEY]        # [B, K, dim]，pose_emb
            memory_value_states = dit_cond_dict.get(_MEMORY_VALUE_KEY)   # [B, K, dim]，visual_emb；可 None
            tier_ids = dit_cond_dict.get(_TIER_IDS_KEY)  # None 时 v3 向后兼容（Innovation 10）
            x = x + self.memory_cross_attn(
                self.memory_norm(x), memory_key_states, memory_value_states, tier_ids=tier_ids
            )

        return x


# ---------------------------------------------------------------------------
# WanModelWithMemory
# ---------------------------------------------------------------------------

class WanModelWithMemory(WanModel):
    """继承 WanModel，为指定 blocks 添加 Memory Cross-Attention，并增加 NFPHead。

    使用方式：
        # 从预训练权重加载原始模型
        base_model = WanModel.from_pretrained(ckpt_dir)

        # 转换为带记忆的版本（新增参数随机初始化）
        model = WanModelWithMemory.from_wan_model(
            base_model,
            memory_layers=None,   # None = 全部 blocks
            max_memory_size=8,
        )

        # 推理时传入 memory_states
        output = model(
            x, t, context, seq_len,
            dit_cond_dict={"c2ws_plucker_emb": ...},
            memory_states=memory_bank_states,  # [1, K, dim]
        )

    Args:
        memory_layers:   要插入 Memory Cross-Attention 的 block 索引列表。
                         None = 全部 blocks（默认）。
                         建议从后半段 blocks 开始，例如 range(20, 40)。
        max_memory_size: Memory Bank 最大容量 K，用于初始化文档，不影响模型权重。
        其余参数与 WanModel 完全相同。
    """

    def __init__(
        self,
        *args,
        memory_layers: Optional[List[int]] = None,
        max_memory_size: int = 8,
        spatial_v_grid: Optional[int] = None,
        **kwargs,
    ):
        # 先调用 WanModel.__init__，创建原始 blocks
        super().__init__(*args, **kwargs)

        # 确定要包裹的 block 索引
        if memory_layers is None:
            memory_layers = list(range(len(self.blocks)))
        self._memory_layers = memory_layers
        self._max_memory_size = max_memory_size

        # Exp2 spatial-V：memory token 从帧级（1 token/帧）升到 patch 级（g*g token/帧）
        #   None = 旧行为（均值池化、帧级，所有现有诊断/v3 路径不变）
        #   g    = 新 patch 行为（g×g 网格，每帧展开成 g*g 个 patch token）
        self.spatial_v_grid = spatial_v_grid
        if spatial_v_grid is not None:
            # 可学习 2D 位置编码：零初始化，形状 [g*g, dim]
            # 加在 K 的组装侧（每帧 pose_emb 重复 g*g 次 + patch_pos_emb），告知模型 patch 的空间位置
            self.patch_pos_emb = nn.Parameter(
                torch.zeros(spatial_v_grid * spatial_v_grid, self.dim)
            )

        # 将指定 blocks 替换为 MemoryBlockWrapper
        for i in memory_layers:
            self.blocks[i] = MemoryBlockWrapper(
                block=self.blocks[i],
                dim=self.dim,
                num_heads=self.num_heads,
                qk_norm=self.qk_norm,
                eps=self.eps,
            )

        # NFPHead：预测下一帧 latent，用于 Surprise score 计算
        self.nfp_head = NFPHead(dim=self.dim, z_dim=self.out_dim)

        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        # latent_proj：将 VAE latent 的空间均值（z_dim=16）投影到模型空间（dim=5120），
        # 作为 cross-attention 的 V（视觉内容嵌入）
        self.latent_proj = nn.Linear(self.out_dim, self.dim, bias=False)

        # Innovation 9: Visual Feature Fusion — Semantic Key 融合视觉特征
        # 将 visual_emb (latent_proj 输出, [dim]) 投影到 K 投影空间，与 pose_key 加权融合
        self.visual_key_proj = nn.Linear(self.dim, self.dim, bias=False)

        logger.info(
            "WanModelWithMemory: wrapped %d blocks with MemoryBlockWrapper "
            "(layers=%s), added NFPHead(dim=%d, z_dim=%d), latent_proj(%d→%d), "
            "visual_key_proj(%d→%d).",
            len(memory_layers), memory_layers, self.dim, self.out_dim,
            self.out_dim, self.dim,
            self.dim, self.dim,
        )

    # ------------------------------------------------------------------
    # forward：注入 memory_states
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        memory_states: Optional[Tensor] = None,
        memory_value_states: Optional[Tensor] = None,  # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
    ):
        """在原始 WanModel.forward 基础上支持 memory_states 和 memory_value_states 注入。

        MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        新增 memory_value_states 参数，对应 cross-attention 的 V（视觉内容 visual_emb）。

        Args:
            memory_states:       [B, K, dim]  Memory Bank 检索到的历史帧 pose_emb（K）。
                                 若为 None，行为与原始 WanModel 完全一致。
            memory_value_states: [B, K, dim]  Memory Bank 检索到的历史帧 visual_emb（V）。
                                 若为 None 则 cross-attention 的 V 退化为 pose_emb（向后兼容）。
            其余参数见 WanModel.forward 文档。

        Returns:
            与 WanModel.forward 相同：List[Tensor]，每项 [C_out, F, H/8, W/8]
        """
        if memory_states is not None:
            # 注入到 dit_cond_dict，MemoryBlockWrapper 会从中取出
            dit_cond_dict = dict(dit_cond_dict) if dit_cond_dict is not None else {}
            dit_cond_dict[_MEMORY_STATES_KEY] = memory_states
            # MODIFIED: F-03/F5 fix — 同时注入 visual_emb 作为 V
            if memory_value_states is not None:
                dit_cond_dict[_MEMORY_VALUE_KEY] = memory_value_states

        return super().forward(
            x, t, context, seq_len, y=y, dit_cond_dict=dit_cond_dict
        )

    # ------------------------------------------------------------------
    # 工厂方法：从预训练 WanModel 转换
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 推理辅助：提取 per-frame visual embedding（F-03/F5 新增）
    # ------------------------------------------------------------------

    def get_projected_latent_emb(self, latent: Tensor) -> Tensor:
        """将单帧 VAE latent 投影到模型空间，返回 visual embedding。

        MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        用于计算 cross-attention 的 V（视觉内容），与 K（pose_emb）配合使用。

        Exp2 spatial-V：当 self.spatial_v_grid 非 None，返回 patch 级 visual emb
        [g*g, dim]（保留空间布局），否则保持帧级 [dim]（均值池化、旧行为）。

        Args:
            latent: [z_dim=16, lat_h, lat_w]  单帧 VAE latent（无 batch 维度）

        Returns:
            visual_emb:
              spatial_v_grid is None → [dim=5120]（帧级，均值池化）
              spatial_v_grid = g     → [g*g, dim=5120]（patch 级，g×g 网格）
        """
        latent = latent.to(self.latent_proj.weight.dtype)
        if self.spatial_v_grid is None:
            # 旧路径：空间均值池化 [z_dim, lat_h, lat_w] → [z_dim] → [dim]
            feat = latent.mean(dim=[-2, -1])  # [z_dim=16]
            return self.latent_proj(feat)  # [dim=5120]

        # 新路径：自适应池化到 g×g 网格，逐 patch 套用 latent_proj（复用旧权重，不改 shape）
        g = self.spatial_v_grid
        pooled = torch_F.adaptive_avg_pool2d(latent[None], (g, g))[0]  # [z_dim, g, g]
        patches = pooled.reshape(pooled.shape[0], g * g).transpose(0, 1)  # [g*g, z_dim]
        return self.latent_proj(patches)  # [g*g, dim=5120]

    def build_memory_kv(
        self,
        pose_embs: Tensor,
        visual_embs: Tensor,
        tier_ids: Optional[Tensor] = None,
    ):
        """Exp2 spatial-V：把 retrieve 出的帧级 (pose_embs, visual_embs, tier_ids)
        组装成送入 cross-attn 的 (memory_states K, memory_value_states V, tier_ids)，
        均带 batch 维度 [1, K', dim]（K' 与 tier_ids 长度一致）。

        不变量：K/V token 数必须一致。

        - spatial_v_grid is None（旧路径）：
            K = pose_embs [k, dim] → [1, k, dim]
            V = visual_embs [k, dim] → [1, k, dim]
            tier_ids 原样（[k] 或 None）
          与现有 pipeline 的 unsqueeze(0) 行为完全等价（向后兼容）。

        - spatial_v_grid = g（新路径），要求 visual_embs 为 [k, g*g, dim]：
            V = visual_embs 展平成 [k*g*g, dim] → [1, k*g*g, dim]
            K = 每帧 pose_emb 重复 g*g 次 + patch_pos_emb[g*g, dim]，
                堆叠成 [k*g*g, dim] → [1, k*g*g, dim]
            tier_ids 每帧重复 g*g → [k*g*g]

        Args:
            pose_embs:   [k, dim]            每个被检索帧的 pose_emb（K 源）
            visual_embs: [k, dim] 或         旧路径帧级 visual_emb
                         [k, g*g, dim]       新路径 patch 级 visual_emb
            tier_ids:    [k] int64 或 None   每帧所属层 id（0/1/2）

        Returns:
            (memory_states, memory_value_states, tier_ids_out)：
              memory_states       [1, K', dim]
              memory_value_states [1, K', dim]
              tier_ids_out        [K'] int64 或 None
        """
        if self.spatial_v_grid is None:
            # 旧路径：帧级，直接加 batch 维（等价于现有 pipeline 的 unsqueeze(0)）
            memory_states = pose_embs.unsqueeze(0)             # [1, k, dim]
            memory_value_states = visual_embs.unsqueeze(0)     # [1, k, dim]
            return memory_states, memory_value_states, tier_ids

        g = self.spatial_v_grid
        gg = g * g
        k = pose_embs.shape[0]
        assert visual_embs.dim() == 3 and visual_embs.shape[1] == gg, (
            f"build_memory_kv: spatial_v_grid={g} expects visual_embs [k, {gg}, dim], "
            f"got {tuple(visual_embs.shape)}"
        )
        dim = pose_embs.shape[-1]
        dev = pose_embs.device

        # V：[k, g*g, dim] → [k*g*g, dim] → [1, k*g*g, dim]
        memory_value_states = visual_embs.reshape(k * gg, dim).unsqueeze(0)

        # K：每帧 pose 重复 g*g 次 + patch_pos_emb（2D 位置编码）
        pos = self.patch_pos_emb.to(device=dev, dtype=pose_embs.dtype)  # [g*g, dim]
        # pose_embs[:, None, :] [k,1,dim] 广播 + pos[None] [1,g*g,dim] → [k, g*g, dim]
        key_full = pose_embs.unsqueeze(1) + pos.unsqueeze(0)            # [k, g*g, dim]
        memory_states = key_full.reshape(k * gg, dim).unsqueeze(0)      # [1, k*g*g, dim]

        # tier_ids：每帧重复 g*g 次
        tier_ids_out = None
        if tier_ids is not None:
            tier_ids_out = tier_ids.to(dev).repeat_interleave(gg)       # [k*g*g]

        return memory_states, memory_value_states, tier_ids_out

    def get_semantic_key(
        self,
        pose_emb: Tensor,
        visual_emb: Optional[Tensor] = None,
        alpha: float = 0.7,
    ) -> Tensor:
        """计算 pose_emb 在 K 投影空间的语义键（用于 LongTermBank 语义相似度）。

        使用所有 memory 层的 K 投影平均，不同层捕捉不同抽象层次的场景特征，
        平均后对 novelty check 和检索更鲁棒（借鉴 HyDRA 思路，本工作独立设计）。

        若提供 visual_emb，则融合视觉特征（Innovation 9: Visual Feature Fusion）：
          semantic_key = alpha * normalize(pose_key) + (1-alpha) * normalize(vis_key)
        此设计解决"相机方向相同但视觉不同的走廊"被 LongTermBank 误判为重复的问题。

        Args:
            pose_emb:   [dim=5120]，单帧的 camera pose embedding（来自 get_projected_frame_embs 的某一帧）
            visual_emb: [dim=5120]（可选）VAE latent 投影到模型空间的视觉嵌入
                        （即 get_projected_latent_emb() 的输出）；
                        若提供则融合（Innovation 9 Visual Feature Fusion）；
                        None 时退化为纯 pose_key（与 v3 行为完全一致）
            alpha:      pose_key 权重（默认 0.7），1-alpha 为 visual_key 权重

        Returns:
            semantic_key: [dim=5120]，已 detach
        """
        keys = []
        for idx in self._memory_layers:
            ca = self.blocks[idx].memory_cross_attn
            # ca.k: Linear(dim → dim)，ca.norm_k: RMSNorm
            keys.append(ca.norm_k(ca.k(pose_emb)))
        pose_key = torch.stack(keys).mean(dim=0)  # [dim=5120]

        if visual_emb is not None:
            # Exp2 spatial-V：若传入 patch 级 visual_emb [g*g, dim]，先均值还原为帧级 [dim]，
            # 保证 novelty / 检索特征与帧级行为完全一致（语义 key 不引入空间布局）
            if visual_emb.dim() == 2:
                visual_emb = visual_emb.mean(dim=0)  # [g*g, dim] → [dim]
            # Innovation 9: Visual Feature Fusion — 融合视觉特征
            vis_key = torch_F.normalize(self.visual_key_proj(visual_emb), dim=-1)
            key = alpha * torch_F.normalize(pose_key, dim=-1) + (1.0 - alpha) * vis_key
            return key.detach()
        return pose_key.detach()  # [dim=5120]

    # ------------------------------------------------------------------
    # 推理辅助：提取 per-frame pose embedding（用于 MemoryBank 存储与检索）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_projected_frame_embs(self, c2ws_plucker_emb: Tensor) -> Tensor:
        """将 raw plucker embedding 投影到模型空间，返回 per-frame 嵌入。

        镜像 WanModel.forward() 内部的 camera 处理逻辑，返回的嵌入与
        传入 WanAttentionBlock 的 c2ws_plucker_emb 保持同一向量空间，
        可直接存入 MemoryBank 并用于 cosine similarity 检索。

        Args:
            c2ws_plucker_emb: [1, C, lat_f, lat_h, lat_w]
                              与 dit_cond_dict["c2ws_plucker_emb"] 进 model.forward() 之前的
                              原始张量相同（chunk 之前）

        Returns:
            frame_embs: [lat_f, dim]  每帧经过均值池化的模型空间 pose 嵌入
        """
        _, _C, lat_f, lat_h, lat_w = c2ws_plucker_emb.shape
        p_t, p_h, p_w = self.patch_size  # (1, 2, 2)
        h_p = lat_h // p_h              # spatial patch 格数（height）
        w_p = lat_w // p_w              # spatial patch 格数（width）

        # 与 WanModel.forward() 完全相同的 rearrange
        x = rearrange(
            c2ws_plucker_emb,
            '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
            c1=p_t, c2=p_h, c3=p_w,
        )  # [1, lat_f * h_p * w_p, raw_dim]

        x = self.patch_embedding_wancamctrl(x)  # [1, L, dim]
        hidden = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(x))
        )
        projected = x + hidden  # [1, L, dim]

        # Mean-pool over spatial patches per frame → [lat_f, dim]
        projected = projected.view(1, lat_f, h_p * w_p, self.dim)
        frame_embs = projected.mean(dim=2).squeeze(0)
        return frame_embs

    @classmethod
    def from_wan_model(
        cls,
        base_model: WanModel,
        memory_layers: Optional[List[int]] = None,
        max_memory_size: int = 8,
        skip_to_device: bool = False,
        spatial_v_grid: Optional[int] = None,
    ) -> "WanModelWithMemory":
        """从已加载的预训练 WanModel 转换为 WanModelWithMemory。

        新增的参数（MemoryBlockWrapper 中的 memory_cross_attn 和 nfp_head）
        将随机初始化，需要通过微调来学习。

        Args:
            base_model:      已加载预训练权重的 WanModel 实例
            memory_layers:   要插入记忆注意力的 block 索引，None = 全部
            max_memory_size: Memory Bank 容量 K
            skip_to_device:  True 时仅做 dtype 转换（保持 CPU），由调用方负责
                             搬迁到目标设备。用于大模型转换时避免旧模型未释放
                             导致 GPU OOM。

        Returns:
            WanModelWithMemory 实例，原有权重不变，新增参数随机初始化
        """
        cfg = base_model.config
        # NOTE: WanModel.ignore_for_config = ['patch_size', 'cross_attn_norm',
        #   'qk_norm', 'text_dim', 'window_size']
        # These five fields are NOT stored in base_model.config by ConfigMixin,
        # so we read them directly from instance attributes instead.
        model = cls(
            model_type=cfg['model_type'],
            control_type=cfg.get('control_type', 'cam'),
            patch_size=base_model.patch_size,
            text_len=cfg['text_len'],
            in_dim=cfg['in_dim'],
            dim=cfg['dim'],
            ffn_dim=cfg['ffn_dim'],
            freq_dim=cfg['freq_dim'],
            text_dim=base_model.text_dim,
            out_dim=cfg['out_dim'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            window_size=base_model.window_size,
            qk_norm=base_model.qk_norm,
            cross_attn_norm=base_model.cross_attn_norm,
            eps=cfg['eps'],
            memory_layers=memory_layers,
            max_memory_size=max_memory_size,
            spatial_v_grid=spatial_v_grid,
        )

        # 将 base_model 的 state_dict key 重映射，使 memory_layers 中的
        # blocks.{i}.xxx → blocks.{i}.block.xxx，以匹配 MemoryBlockWrapper 的结构
        memory_layer_set = set(
            memory_layers if memory_layers is not None
            else list(range(cfg['num_layers']))
        )
        remapped_sd = {}
        for key, value in base_model.state_dict().items():
            new_key = key
            for i in memory_layer_set:
                prefix = f"blocks.{i}."
                if key.startswith(prefix):
                    new_key = f"blocks.{i}.block." + key[len(prefix):]
                    break
            remapped_sd[new_key] = value

        # 加载重映射后的权重（MemoryBlockWrapper 内的 block 权重正确匹配）
        missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
        if unexpected:
            logger.warning("from_wan_model: unexpected keys: %s", unexpected)
        if missing:
            logger.info(
                "from_wan_model: %d keys not in base model "
                "(expected: memory_cross_attn + nfp_head + latent_proj + visual_key_proj + tier_emb): %s",
                len(missing), missing[:5],
            )

        # 将新模型移到与 base_model 相同的设备和 dtype（通常 bfloat16）
        dtype = next(base_model.parameters()).dtype
        if skip_to_device:
            # 仅做 dtype 转换，保持 CPU；调用方负责 del 旧模型后再搬到 GPU
            model = model.to(dtype=dtype)
        else:
            device = next(base_model.parameters()).device
            model = model.to(device=device, dtype=dtype)

        return model
