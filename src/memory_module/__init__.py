"""
memory_module — Surprise-Driven Memory Bank for LingBot-World

新增模块，不修改 lingbot-world 原始代码。

主要组件：
    MemoryBank          — 存储 / 检索历史帧（按 Surprise score 筛选）
    MemoryFrame         — 单帧记忆数据结构
    MemoryCrossAttention — 历史帧 Cross-Attention（Query=当前帧，KV=历史帧）
    NFPHead             — Next Frame Prediction Head（Surprise score 来源）
    WanModelWithMemory  — 带记忆机制的 WanModel（继承，不修改原始代码）
    MemoryBlockWrapper  — 单 block 包裹器（WanAttentionBlock + MemoryCrossAttention）
"""

from .memory_bank import MemoryBank, MemoryFrame
from .nfp_head import NFPHead
from .causal_memory_adapter import (
    CausalMemoryAdapter,
    CausalMemoryAdapterConfig,
    WanCausalMemoryAdapterHooks,
    expected_trainable_inventory,
    tensor_module_fingerprint,
)

__all__ = [
    'MemoryBank', 'MemoryFrame', 'NFPHead',
    'CausalMemoryAdapter', 'CausalMemoryAdapterConfig',
    'WanCausalMemoryAdapterHooks',
    'expected_trainable_inventory',
    'tensor_module_fingerprint',
]

try:
    from .memory_attention import MemoryCrossAttention
    from .model_with_memory import MemoryBlockWrapper, WanModelWithMemory
    __all__ += ['MemoryCrossAttention', 'MemoryBlockWrapper', 'WanModelWithMemory']
except (ImportError, RuntimeError):
    # CUDA 不可用时（如登录节点）跳过，按需直接 import 对应子模块
    pass
