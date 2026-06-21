"""
train_v3_stage2_dual.py — LingBot-World Memory Enhancement 训练脚本
v3 数据 + Stage2 + 双模型全参数解冻联合微调（对应实验配置 ⑧，最终目标）

数据：v3（8ch action：WASD + jump / crouch / fire / walk）
训练阶段：Stage2 — 解冻全部参数，联合微调

双模型架构（与 train_v3_stage1_dual.py 一致）：
  - low_noise_model：WanModelWithMemory（t < 0.947）
  - high_noise_model：WanModelWithMemory（t >= 0.947）
  - 两个模型同时参与训练（非交替 epoch 策略）

学习率分组（同 train_v2_stage2.py）：
  - DiT blocks（low + high）：lr_dit（默认 1e-5）
  - 记忆模块（MemoryCrossAttention + NFPHead + memory_norm）：lr（默认 1e-4）
  - VAE 和 T5：保持冻结

v3 vs v2 关键差异：
  - action.npy shape：[81, 8]（v2 为 [81, 4]）

# PENDING[D-03]：Stage2 起点权重待定（选项A: 原始预训练权重；选项B: 对方 CSGO-DiT 权重）

状态：PENDING — 依赖条件：① train_v3_stage1_dual.py 实现完成 + ② v3 数据就绪 + ③ D-03 解除
"""

raise NotImplementedError(
    "train_v3_stage2_dual.py（双模型 Stage2）尚未实现。\n"
    "依赖：train_v3_stage1_dual.py 完成 + v3 数据（8ch action）就绪 + D-03 决策解除。\n"
    "# PENDING[D-03]：Stage2 起点权重待定"
)
