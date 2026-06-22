"""
model_with_memory_v5.py — WanModel + v5 in-context KV 记忆（experiment_design Step 40 / S-V1）

`WanModelWithMemoryV5(WanModel)`：把指定 blocks 的 self_attn 替换为 MemorySelfAttention
（保留骨干权重），并挂一个唯一可训练件 MemoryEncoder。forward 时把检索帧 latent 编码成
memory token，逐层 set_memory，跑骨干 super().forward，再在 finally 里统一 clear_memory。

设计依据（已锁定）：
  - decisions.md「讨论 8」+ open_problems.md「OP-5」：v5 锁定方案（in-context KV、无 gate、
    方案A、只读、全冻骨干，唯一可训练件 = MemoryEncoder）。
  - 20260621_v5_backbone_interface_modeA.md：
    R4（from_wan_model 绝不加 v4 的 `blocks.{i}.block.` 前缀重映射；v5 是替换 self_attn、
        键名不变，骨干 state_dict 直接 load，硬断言 missing 仅 memory_encoder、unexpected 空）；
    R6（memory stash → read → clear 链路，clear 在 finally）。

== 与 v4 model_with_memory.py 的根本区别（反面参照）==
  - v4 用 MemoryBlockWrapper 包裹 block → state_dict key 多一层 `block.`，from_wan_model
    需把 `blocks.{i}.xxx` 重映射为 `blocks.{i}.block.xxx`。
  - **v5 不包 block、不重映射**：直接把 `blocks.{i}.self_attn` 实例替换为
    MemorySelfAttention（同名参数 q/k/v/o/norm_q/norm_k），键名完全不变。
    **本文件刻意不出现任何 `.block.` 前缀重映射**（防 R4 复发）。

== 已知限制 ==
  - 注入路径按 **B=1** 设计：memory token reshape 成 [1, K*gg, dim]（batch 维=1）。
    B>1 时 memory 需按样本对齐，本期作已知限制。
  - padded-batch 不支持（继承 MemorySelfAttention 的限制，见 injection.py）。
"""

import logging
import os
import sys
from typing import List, Optional

import torch
from torch import Tensor

# ---- 引入 lingbot-world 骨干（只继承/复用，不改 refs）----
_LINGBOT_WORLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'refs', 'lingbot-world'
)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

from wan.modules.model import WanModel, WanLayerNorm, WanRMSNorm  # noqa: E402

# ---- v5_incontext 同级组件 ----
from .injection import MemorySelfAttention  # noqa: E402
from .memory_encoder import MemoryEncoder  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 全局 dtype-robust 修法（不依赖 autocast，让骨干 forward 类型自洽）
# ----------------------------------------------------------------------
#
# 背景（v5 全冻骨干 + 探针 no_grad → autocast 整体失效）：
#   - refs 的 WanLayerNorm/WanRMSNorm 设计在 float32 跑（`x.float()` + `type_as`），
#     需要 float32 权重；blanket `.to(bf16)` 把它们压成 bf16 →
#     `F.layer_norm(float32, bf16)` 崩。
#   - 骨干 block 的 modulation e 故意是 float32（refs model.py:489/244 autocast(float32)
#     + assert），故 block 算出的激活是 float32，流进 cross_attn/ffn 的 bf16 Linear → 崩。
#   - time_embedding/time_projection 必须保持 float32 产出 e（探针已验证不崩）→ 不动它们。
#
# 本函数做两件事（详见 (A)/(B)）：
#   (A) 所有 norm 模块权重 → float32（让 `x.float()` 路径成立）；
#   (B) 给骨干「会吃 float32 激活的 bf16 矩阵 op」(Linear/Conv3d) 挂 forward_pre_hook，
#       把输入 cast 到自己权重 dtype，从而不依赖 autocast。

def _cast_pre_hook(module, args):
    """forward_pre_hook：把 input[0] cast 到 module.weight.dtype（已同 dtype 则原样）。"""
    if not args:
        return args
    x = args[0]
    if isinstance(x, torch.Tensor) and x.dtype != module.weight.dtype:
        x = x.to(module.weight.dtype)
        return (x,) + args[1:]
    return args


def _make_dtype_robust(model: "WanModelWithMemoryV5"):
    """让骨干 forward 不依赖 autocast 也类型自洽（在 model.to(dtype) 之后调用）。

    (A) 所有 norm 模块（WanLayerNorm/WanRMSNorm/torch.nn.LayerNorm）权重 → float32。
        memory_encoder 内部的 LayerNorm 也会被覆盖到——无害（float32 norm 更稳），
        但只转 norm 类模块，memory_encoder 的非-norm 权重（Linear/in_proj）不被误转。
    (B) 对骨干的 Linear/Conv3d（满足排除条件的）挂 forward_pre_hook，把输入 cast 到
        自己权重 dtype。排除：time_embedding / time_projection（需保持 float32 产出 e，
        挂了会把 e 变 bf16 触发 assert）、memory_encoder（自身 forward 已对齐 dtype）。
        norm 类模块本就不是 Linear/Conv3d，不会被选中。

    返回挂上的 hook 句柄列表（也存在 model._dtype_robust_hooks 上，确保不被 GC）。
    """
    # ---- (A) norm 权重 → float32 ----
    n_norm = 0
    for m in model.modules():
        if isinstance(m, (WanRMSNorm, WanLayerNorm, torch.nn.LayerNorm)):
            m.float()  # 只把 norm 类模块的 param/buffer cast 到 float32
            n_norm += 1

    # ---- (B) Linear/Conv3d 挂 forward_pre_hook（带排除）----
    handles = []
    n_hook = 0
    for name, m in model.named_modules():
        if not isinstance(m, (torch.nn.Linear, torch.nn.Conv3d)):
            continue
        # 排除：time_embedding/time_projection（保持 float32 产出 e）、memory_encoder（自洽）
        if "time_embedding" in name or "time_projection" in name:
            continue
        if "memory_encoder" in name:
            continue
        handles.append(m.register_forward_pre_hook(_cast_pre_hook))
        n_hook += 1

    # 存住句柄，防 GC（模型生命周期内一直有效）
    model._dtype_robust_hooks = handles

    logger.info(
        "_make_dtype_robust: norm→float32 %d 个（WanLayerNorm/WanRMSNorm/LayerNorm，"
        "含 memory_encoder 内部 LayerNorm，无害）；挂 forward_pre_hook %d 个"
        "（骨干 Linear/Conv3d；已排除 time_embedding/time_projection/memory_encoder）。",
        n_norm, n_hook,
    )
    return handles


class WanModelWithMemoryV5(WanModel):
    """WanModel + v5 in-context KV 记忆（只读、无 gate、全冻骨干）。

    唯一可训练件 = self.memory_encoder。被注入的 blocks 的 self_attn 替换为
    MemorySelfAttention（保留骨干权重，不新增 Parameter）。

    用法：
        base = WanModel.from_pretrained(ckpt)
        model = WanModelWithMemoryV5.from_wan_model(base, memory_layers=None, grid=16)
        out = model(x, t, context, seq_len, y=y, dit_cond_dict=...,
                    memory_latents=retrieved_latents)  # [K, 16, h, w] 或 None
    """

    def __init__(
        self,
        *args,
        memory_layers: Optional[List[int]] = None,
        grid: int = 16,
        encoder_depth: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 确定注入层（默认全部 blocks）
        if memory_layers is None:
            memory_layers = list(range(len(self.blocks)))
        self._memory_layers = list(memory_layers)
        self._grid = grid
        self._encoder_depth = encoder_depth

        # 将指定 blocks 的 self_attn 替换为 MemorySelfAttention（保留权重）。
        # 做法：新建同配置的 MemorySelfAttention，load_state_dict(old.state_dict())，
        #       再赋值替换。保证 q/k/v/o/norm_q/norm_k 权重零丢失。
        for i in self._memory_layers:
            old_attn = self.blocks[i].self_attn
            new_attn = MemorySelfAttention(
                dim=old_attn.dim,
                num_heads=old_attn.num_heads,
                window_size=old_attn.window_size,
                qk_norm=old_attn.qk_norm,
                eps=old_attn.eps,
            )
            new_attn.load_state_dict(old_attn.state_dict())
            # 对齐 dtype/device 到原 self_attn 权重（替换前后保持一致）
            ref = old_attn.q.weight
            new_attn = new_attn.to(dtype=ref.dtype, device=ref.device)
            self.blocks[i].self_attn = new_attn

        # 唯一可训练件：MemoryEncoder（latent → memory token）
        # z_dim 用骨干 out_dim（VAE latent C，通常 16）
        self.memory_encoder = MemoryEncoder(
            z_dim=self.out_dim,
            dim=self.dim,
            grid=grid,
            depth=encoder_depth,
        )

        logger.info(
            "WanModelWithMemoryV5: 替换 %d 个 block 的 self_attn 为 MemorySelfAttention "
            "(layers=%s)，挂载 MemoryEncoder(z_dim=%d, dim=%d, grid=%d, depth=%d)。",
            len(self._memory_layers), self._memory_layers,
            self.out_dim, self.dim, grid, encoder_depth,
        )

    # ------------------------------------------------------------------
    # forward：注入 memory（R6 stash → read → clear）
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        memory_latents: Optional[Tensor] = None,
    ):
        """在 WanModel.forward 基础上支持 memory_latents 注入。

        Args:
            memory_latents: [K, z_dim, h, w]（检索帧 VAE latent）或 None。
                            None 时行为与原始 WanModel 完全一致。
            其余参数见 WanModel.forward。

        Returns:
            与 WanModel.forward 相同：List[Tensor]，每项 [C_out, F, H/8, W/8]
        """
        injected = False
        if memory_latents is not None:
            # latent → memory token：[K, gg, dim]
            mem = self.memory_encoder(memory_latents)          # [K, gg, dim]
            K, gg, dim = mem.shape
            # reshape 成 [1, K*gg, dim]（注入路径按 B=1 设计）
            mem = mem.reshape(1, K * gg, dim)
            for i in self._memory_layers:
                self.blocks[i].self_attn.set_memory(mem)
            injected = True

        try:
            out = super().forward(
                x, t, context, seq_len, y=y, dit_cond_dict=dit_cond_dict
            )
        finally:
            # R6：无论是否异常，对所有注入层统一 clear（杜绝串味、保护梯度检查点重算）
            if injected:
                for i in self._memory_layers:
                    self.blocks[i].self_attn.clear_memory()

        return out

    # ------------------------------------------------------------------
    # 工厂方法：从预训练 WanModel 转换（R4 闸）
    # ------------------------------------------------------------------

    @classmethod
    def from_wan_model(
        cls,
        base_model: WanModel,
        memory_layers: Optional[List[int]] = None,
        grid: int = 16,
        encoder_depth: int = 1,
        skip_to_device: bool = False,
    ) -> "WanModelWithMemoryV5":
        """从已加载的预训练 WanModel 转换为 WanModelWithMemoryV5。

        **R4 闸**（防骨干被破坏）：base_model.state_dict() **直接** load 进新模型，
        **绝不加 v4 的 `blocks.{i}.block.` 前缀重映射**（v5 是替换 self_attn、键名不变）。
        硬断言：missing 仅 memory_encoder.*、unexpected 为空，否则 raise RuntimeError。

        Args:
            base_model:     已加载预训练权重的 WanModel 实例
            memory_layers:  注入层索引，None = 全部
            grid:           MemoryEncoder 每帧 grid×grid token
            encoder_depth:  MemoryEncoder Transformer 块层数
            skip_to_device: True 时仅做 dtype 转换（保持 CPU），调用方负责搬到 GPU

        Returns:
            WanModelWithMemoryV5 实例，骨干权重原样，memory_encoder 随机初始化（再切片初始化）
        """
        cfg = base_model.config
        # WanModel.ignore_for_config = ['patch_size','cross_attn_norm','qk_norm',
        #   'text_dim','window_size'] → 这五项不在 config，从实例属性读（照 v4 写法）。
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
            grid=grid,
            encoder_depth=encoder_depth,
        )

        # R4 闸：base state_dict 直接 load，键名不变（self_attn 替换为同名参数子类）。
        # **刻意不做任何 key 重映射**（防 R4 复发：grep 该文件无 `.block.`）。
        base_sd = base_model.state_dict()
        missing, unexpected = model.load_state_dict(base_sd, strict=False)

        # 硬断言：骨干 0 缺失（missing 全是 memory_encoder.*），unexpected 必须为空。
        bad_missing = [k for k in missing if not k.startswith("memory_encoder")]
        if bad_missing or unexpected:
            raise RuntimeError(
                "from_wan_model R4 闸失败：骨干权重加载不干净。\n"
                f"  非 memory_encoder 的 missing keys（骨干缺失，{len(bad_missing)}）："
                f"{bad_missing}\n"
                f"  unexpected keys（{len(unexpected)}）：{unexpected}\n"
                "可能原因：v5 不应做任何 key 重映射；若引入 `blocks.{i}.block.` 前缀即 R4 复发。"
            )
        logger.info(
            "from_wan_model R4 闸通过：骨干 0 缺失，unexpected 空；"
            "memory_encoder missing %d 个（随机初始化，待切片初始化）。",
            len(missing),
        )

        # memory_encoder 用骨干 patch_embedding 前 z_dim 通道切片初始化（分布内起步）
        model.memory_encoder.init_from_patch_embedding(model.patch_embedding.weight)

        # 冻结骨干：除 memory_encoder.* 外全部 requires_grad_(False)
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("memory_encoder"))
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "from_wan_model: 冻结骨干完成。可训练参数 %.1fM（应全在 memory_encoder），"
            "总参数 %.1fM。",
            n_train / 1e6, n_total / 1e6,
        )

        # dtype/device 处理（照 v4 写法）
        dtype = next(base_model.parameters()).dtype
        if skip_to_device:
            model = model.to(dtype=dtype)
        else:
            device = next(base_model.parameters()).device
            model = model.to(device=device, dtype=dtype)

        # 全局 dtype-robust 修法：在 model.to(dtype) **之后** 调用——
        #   (A) 把 norm 权重转回 float32（blanket .to(bf16) 已把它们压成 bf16），
        #   (B) 给骨干 Linear/Conv3d 挂 forward_pre_hook 对齐输入 dtype。
        # 让骨干 forward 不依赖 autocast 也类型自洽（终结 v5 dtype 连环崩）。
        _make_dtype_robust(model)

        return model
