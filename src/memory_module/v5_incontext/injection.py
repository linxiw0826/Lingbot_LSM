"""
injection.py — v5 in-context KV 记忆注入的只读注意力子类（experiment_design Step 40 / S-V1）

`MemorySelfAttention(WanSelfAttention)`：WorldMem MemFullAttention 形态的「只读 KV-concat」
self-attention。memory token 只进 K/V（不进 Q），输出长度与当前帧序列 L 完全一致，
unpatchify 天然安全（memory 不污染输出）。

设计依据（已锁定）：
  - decisions.md「讨论 8」+ open_problems.md「OP-5」：
    v5 = in-context KV、无 gate、方案A、只读、全冻骨干；唯一可训练件 = MemoryEncoder。
  - 20260621_v5_backbone_interface_modeA.md §4（路线1）：
    在 qkv_fn 之后、flash_attention 之前 concat k_mem/v_mem；
    memory key 复用冻结 self.k + norm_k（尺度对齐），memory value 复用冻结 self.v（不 norm）；
    memory 不施 rope（无合法 (f,h,w) 网格坐标，先例：WanCrossAttention 的文本 K/V 也不加 rope）；
    k_lens = seq_lens + M；q 仍 L 个 → 输出长度不变。

**R4 关键**：本类**不新增任何 nn.Parameter**——继承父类 q/k/v/o/norm_q/norm_k，
骨干权重通过 load_state_dict 原样加载（键名不变，无需重映射）。

**R6 关键（stash → read → clear 链路）**：
  - set_memory(mem) 由 model.forward 在 super().forward() **之前**调用，暂存 memory token；
  - forward 读取 self._mem_tokens 但**读后不清除**（梯度检查点会重算 forward，
    若在 attn 内 read 后立即 clear，重算时 memory 已为 None，记忆丢失）；
  - clear_memory() 由 model.forward 在 super().forward() 之后的 **finally** 里统一清除，
    保证无论是否异常都不串味（杜绝下一次 forward 误用上一次的 memory）。

== 已知限制（本期不支持，写进契约）==
  - **padded-batch 不支持**：flash_attention 变长语义下，仅当「当前帧无 padding
    （seq_lens 全 == L）」时 `k_lens = seq_lens + M` 才正确——因为 k/v 的 memory 段
    紧跟在每个样本的有效当前 token 之后，但本实现把 memory concat 在 padded 的 L 维末尾，
    只有 seq_lens==L（无 pad）时 memory 才落在每个样本有效区的正确位置。
    forward 在 memory 注入路径会 `assert (seq_lens == L).all()`。
    padded-batch（变长样本同 batch）需要把 memory 插到每个样本有效段之后再重新 pad，
    本期不实现。
  - **B>1**：注入路径按 B=1 设计（model.forward 注入路径 batch=1），B>1 时 memory token
    需按样本对齐，本期作已知限制（assert 不限制 B，但调用方应保证 B=1 注入）。
"""

import os
import sys
from typing import Optional

import torch
from torch import Tensor

# ---- 引入 lingbot-world 骨干（只继承/复用，不改 refs）----
_LINGBOT_WORLD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'refs', 'lingbot-world'
)
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

from wan.modules.attention import flash_attention  # noqa: E402
from wan.modules.model import WanSelfAttention, rope_apply  # noqa: E402


class MemorySelfAttention(WanSelfAttention):
    """只读 KV-concat self-attention（v5 注入核心，不新增 Parameter）。

    继承 WanSelfAttention 的全部投影（q/k/v/o/norm_q/norm_k）。除多了一个普通属性
    `_mem_tokens`（非 buffer/param），结构与父类完全一致，故骨干权重原样加载（R4）。

    forward 与父类 WanSelfAttention.forward 行为一致，除非 `self._mem_tokens is not None`：
    此时把 memory token 经冻结 self.k/self.v 投影后 concat 到当前帧 k/v 上（memory 不施 rope），
    k_lens += M，q 仍 L 个。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 普通属性，非 buffer/非 param（不进 state_dict、不被 to()/ZeRO 影响），R4 安全。
        self._mem_tokens: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # 暂存接口（R6：由 model.forward 管理生命周期）
    # ------------------------------------------------------------------

    def set_memory(self, mem_tokens: Optional[Tensor]) -> None:
        """暂存 memory token，[B, M, dim] 或 None。

        由 WanModelWithMemoryV5.forward 在 super().forward() **之前**调用。
        forward 读取后**不**清除（梯度检查点重算保护）；统一由 model.forward 的 finally
        调 clear_memory() 清除。
        """
        self._mem_tokens = mem_tokens

    def clear_memory(self) -> None:
        """清除暂存的 memory token（置 None）。

        由 WanModelWithMemoryV5.forward 在 super().forward() 之后的 finally 里统一调用，
        保证无论是否异常都不串味。
        """
        self._mem_tokens = None

    # ------------------------------------------------------------------
    # forward：照搬父类 + 可选 memory KV-concat
    # ------------------------------------------------------------------

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor):          [B, L, dim]
            seq_lens(Tensor):   [B]，每样本当前帧有效 token 数
            grid_sizes(Tensor): [B, 3]，(F, H, W)
            freqs(Tensor):      rope freqs [1024, dim/num_heads/2]

        Returns:
            [B, L, dim]（memory 只读，输出仍 L 个，unpatchify 安全）
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # dtype 对齐（不依赖 autocast）：WanModel 的 modulation e/e0 故意走 float32
        # （refs model.py:489 autocast(float32)+assert e.dtype==float32），故 block 算出的
        # norm1(x)*(1+e)+e 会是 float32 传进本 self-attn。v4 靠外层 autocast 把 self.q 输入
        # cast 回 bf16；但 v5 全冻骨干下本替换子模块疑似未吃到 autocast（ZeRO-3 对无可训练
        # 参数的冻结子模块走不同路径），会崩在 self.q(float vs bf16)。这里显式对齐到权重
        # dtype，保证 self.q/k/v 输入与权重同 dtype，不依赖 autocast（OFF + memory-ON 两条
        # 路径都经 qkv_fn(x)，故覆盖二者）。
        _w_dtype = self.q.weight.dtype
        if x.dtype != _w_dtype:
            x = x.to(_w_dtype)

        # query, key, value function（照搬父类 qkv_fn）
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # 当前帧 q/k 施 rope（与父类一致）
        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        mem = self._mem_tokens
        if mem is not None and mem.dtype != _w_dtype:
            # memory token 来自 memory_encoder 应已是 bf16，但显式对齐更稳，
            # 防 self.k(mem)/self.v(mem) 再撞 dtype（同 _w_dtype = self.q.weight.dtype）。
            mem = mem.to(_w_dtype)
        if mem is None:
            # 无 memory：行为与父类 WanSelfAttention.forward 完全一致
            x = flash_attention(
                q=q_rope,
                k=k_rope,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
            )
        else:
            # ---- memory 注入路径 ----
            # padded-batch 不支持（见模块 docstring）：仅当当前帧无 padding 时
            # k_lens = seq_lens + M 才正确。
            L = x.shape[1]
            assert bool((seq_lens == L).all()), (
                "MemorySelfAttention: memory 注入路径要求 seq_lens 全 == L（当前帧无 "
                f"padding）；got seq_lens={seq_lens.tolist()}, L={L}。"
                "padded-batch 本期不支持（见 injection.py 模块 docstring）。"
            )

            # dtype 对齐到 self.k.weight.dtype（mem 来自 memory_encoder，可能 dtype 不同）
            mem = mem.to(dtype=self.k.weight.dtype, device=self.k.weight.device)
            M = mem.shape[1]

            # memory key：复用冻结 self.k + norm_k 做尺度对齐；memory 不施 rope。
            k_mem = self.norm_k(self.k(mem)).view(mem.shape[0], M, n, d)
            # memory value：复用冻结 self.v，不 norm（与当前帧 v 一致）。
            v_mem = self.v(mem).view(mem.shape[0], M, n, d)

            # concat 到当前帧 k/v（memory 段紧跟当前 token，不加 rope）
            k_all = torch.cat([k_rope, k_mem.to(k_rope.dtype)], dim=1)  # [B, L+M, n, d]
            v_all = torch.cat([v, v_mem.to(v.dtype)], dim=1)           # [B, L+M, n, d]

            # k_lens = seq_lens + M（含 memory）；q 仍 L 个 → 输出长度不变
            k_lens = seq_lens + M

            x = flash_attention(
                q=q_rope,
                k=k_all,
                v=v_all,
                k_lens=k_lens,
                window_size=self.window_size,
            )

        # output（照搬父类）
        x = x.flatten(2)
        x = self.o(x)
        return x
