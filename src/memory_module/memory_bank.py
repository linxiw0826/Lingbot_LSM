"""
memory_bank.py — Memory Bank 模块

当前实现（v3，已完成）：
  1. 单层 MemoryBank（原有，向后兼容）
     - MemoryFrame dataclass：单帧记忆条目
     - MemoryBank：固定容量 K，surprise-driven 写入（evict 最低 surprise 帧）
     - retrieve()：query_pose_emb 与存储帧做 cosine similarity，返回 (pose_embs [k,5120], visual_embs [k,5120])

  2. 三层 ThreeTierMemoryBank（新增，Orchestrator 2026-04-15 授权）
     ┌─────────────────────────────────────────────────────────────────────┐
     │  ShortTermBank   │ FIFO，容量 1，强制存最近帧，保证 chunk 连续性          │
     │  MediumTermBank  │ 容量 8，高 surprise 帧，age decay eviction           │
     │  LongTermBank    │ 容量 32，stable（低 surprise）且 novel（语义新颖）帧  │
     └─────────────────────────────────────────────────────────────────────┘

     MemoryFrame 新增字段（Orchestrator 2026-04-15 授权）：
       semantic_key [5120]：LongTermBank 写入/检索时使用，
                            = mean(norm_k_i(k_i(pose_emb)) for all memory_layers)
       tier str：标记所属层（"short"/"medium"/"long"），便于调试

     混合检索预算（Hybrid Retrieval Budget）：Short 1 + Medium top-3 + Long top-2 = 6 帧
     retrieve() 返回：(key_states [≤6,5120], value_states [≤6,5120])，去重后可能少于 6 帧

依赖：PyTorch（无 lingbot-world 依赖，独立可测试）
设计参考：
  - WorldMem：Memory Bank 结构 + FOV pose 检索
  - Cambrian-S：NFPHead Surprise 机制（见 nfp_head.py）
  - HyDRA（Out of Sight but Not Out of Mind）：semantic_key 借鉴 K 投影特征提取器思路
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryFrame:
    """单帧记忆条目。

    Attributes:
        pose_emb:   当前帧的 camera pose 在模型空间的嵌入，[dim]，
                    由 WanModel 的 c2ws_plucker_emb 经过 mean-pool 得到。
                    用作 MemoryCrossAttention 的 Key（FOV 路由）。
        latent:     VAE encoded latent，[z_dim, h, w]，
                    用于 NFPHead 的预测目标。
        surprise_score: NFPHead 计算的 cosine distance（越大越"意外"）。
        timestep:   原始视频帧索引，用于 temporal ordering。
        visual_emb: VAE latent 投影到模型空间的视觉嵌入，用作 MemoryCrossAttention 的 Value。
                    可为 [dim=5120]（帧级，旧行为）或 [g*g, dim]（patch 级，Exp2 spatial-V）。
                    存储时原样保存，不在存储时池化。
                    若 None 则 retrieve() 退化为 pose_emb 做 V（向后兼容）。
                    # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
    """
    pose_emb: Tensor                      # [dim]
    latent: Tensor                         # [z_dim, h, w]
    surprise_score: float
    timestep: int
    visual_emb: Optional[Tensor] = None   # [dim=5120]，MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
    chunk_id: int = 0    # 所属 chunk 编号，授权新增（Orchestrator 2026-04-02）
    age: int = 0         # 自写入以来经历的 chunk 数，授权新增（Orchestrator 2026-04-02）
    # 新增字段（Orchestrator 2026-04-15 授权）
    semantic_key: Optional[Tensor] = None  # [dim=5120]，LongTermBank 写入/检索时使用，= mean(norm_k_i(k_i(pose_emb)) for all memory_layers)
    tier: str = ""                          # 标记所属层（"short"/"medium"/"long"），便于调试
    # OP-2 Bug2 修复（重访检索）：绝对 c2w 平移向量 [3]，用于地点重访检索（按位置 L2）
    # 默认 None 以向后兼容——现有所有构造点不传 location 也能工作。
    location: Optional[Tensor] = None       # [3]，绝对 c2w 平移向量（世界位置）


class MemoryBank:
    """Surprise-Driven Memory Bank。

    容量为 max_size，存满后替换 surprise_score 最低的帧。
    检索时用 query 的 pose_emb 与所有存储帧的 pose_emb 做 cosine similarity，
    返回 top-k 帧的 pose_emb 作为 MemoryCrossAttention 的 Key/Value 输入。

    使用方式（推理循环）：
        bank = MemoryBank(max_size=8)
        # 每帧生成后：
        bank.update(pose_emb, latent, surprise_score, timestep, visual_emb=visual_emb)
        # 下一帧生成前（MODIFIED: F-03/F5 fix — 返回 tuple）：
        retrieved = bank.retrieve(query_pose_emb, top_k=4)  # (pose_embs [k,dim], visual_embs [k,dim])
        if retrieved is not None:
            key_states, value_states = retrieved
    """

    def __init__(self, max_size: int = 8):
        """
        Args:
            max_size: Memory Bank 的最大容量 K，默认 8（WorldMem reference_length=8）
        """
        self.max_size = max_size
        self.frames: List[MemoryFrame] = []
        # 操作统计（用于 W&B 日志）
        self.store_count: int = 0
        self.reject_count: int = 0
        self.evict_count: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        pose_emb: Tensor,
        latent: Tensor,
        surprise_score: float,
        timestep: int,
        visual_emb: Optional[Tensor] = None,  # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        chunk_id: int = 0,   # 新增，默认 0（向后兼容）
    ) -> None:
        """存入一帧。若已满，替换 surprise_score 最低的帧。

        Args:
            pose_emb:       [dim] 当前帧的 pose embedding（已 mean-pool 到 1D）
            latent:         [z_dim, h, w] VAE latent
            surprise_score: NFPHead 计算的 per-frame surprise（0~2 之间，越大越值得存）
            timestep:       当前帧在原始视频中的帧索引
            visual_emb:     [dim=5120] VAE latent 投影到模型空间的视觉嵌入（可选）；
                            None 时 retrieve() 退化为 pose_emb 做 V（向后兼容）
                            # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
            chunk_id:       所属 chunk 编号（Feature 3 新增，默认 0）
        """
        # MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        new_frame = MemoryFrame(
            pose_emb=pose_emb.detach().cpu(),
            latent=latent.detach().cpu(),
            surprise_score=float(surprise_score),
            timestep=int(timestep),
            visual_emb=visual_emb.detach().cpu() if visual_emb is not None else None,
            chunk_id=int(chunk_id),
            age=0,
        )

        if len(self.frames) < self.max_size:
            self.frames.append(new_frame)
            self.store_count += 1
            logger.debug(
                "MemoryBank: added frame t=%d, surprise=%.4f, size=%d/%d",
                timestep, surprise_score, len(self.frames), self.max_size,
            )
        else:
            # 替换 surprise_score 最低的帧
            min_idx = min(range(len(self.frames)),
                          key=lambda i: self.frames[i].surprise_score)
            evicted = self.frames[min_idx]
            if surprise_score > evicted.surprise_score:
                self.frames[min_idx] = new_frame
                self.evict_count += 1
                self.store_count += 1
                logger.debug(
                    "MemoryBank: replaced t=%d(s=%.4f) with t=%d(s=%.4f)",
                    evicted.timestep, evicted.surprise_score,
                    timestep, surprise_score,
                )
            else:
                self.reject_count += 1
                logger.debug(
                    "MemoryBank: frame t=%d(s=%.4f) not stored (min stored=%.4f)",
                    timestep, surprise_score, evicted.surprise_score,
                )

    def increment_age(self) -> None:
        """在每个新 chunk 生成前调用，所有已存储帧 age +1。"""
        for frame in self.frames:
            frame.age += 1

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_pose_emb: Tensor,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """按 pose cosine similarity 检索最相关的 top-k 帧。

        MODIFIED: F-03/F5 fix, authorized by Orchestrator 2026-04-02
        返回类型从 Optional[Tensor] 改为 Optional[Tuple[Tensor, Tensor]]：
          - (pose_embs [k, dim], visual_embs [k, dim])
          - pose_embs 用作 cross-attention 的 K（FOV 路由）
          - visual_embs 用作 cross-attention 的 V（视觉内容）
          - 若所有帧 visual_emb is None，则 visual_embs = pose_embs（退化，向后兼容）

        Args:
            query_pose_emb: [dim] 当前帧的 pose embedding，用于检索
            top_k:          返回帧数，None 表示返回全部
            device:         输出 tensor 放置的设备，None 时跟随 query

        Returns:
            (pose_embs, visual_embs): 各 [k, dim]，k = min(top_k, len(self.frames))
            若 bank 为空返回 None
        """
        if not self.frames:
            return None

        device = device or query_pose_emb.device
        k = min(top_k, len(self.frames)) if top_k is not None else len(self.frames)

        all_pose_embs = torch.stack(
            [f.pose_emb for f in self.frames]
        ).to(device)  # [K, dim]

        # cosine similarity: [K]
        sims = F.cosine_similarity(
            query_pose_emb.unsqueeze(0).to(device),  # [1, dim]
            all_pose_embs,                            # [K, dim]
            dim=-1,
        )

        _, indices = torch.topk(sims, k=k)
        idx_list = indices.tolist()

        # MODIFIED: F-03/F5 fix — K = pose_embs, V = visual_embs
        pose_embs = torch.stack(
            [self.frames[i].pose_emb for i in idx_list]
        ).to(device)   # [k, dim]

        # 若任何帧有 visual_emb，则为每帧填充（None 退化为 pose_emb）
        # Exp2 spatial-V：若 visual_emb 是 [g*g, dim]，stack 后为 [k, g*g, dim]（patch 级）；
        #   帧级 [dim] 时为 [k, dim]（旧行为）。上层 build_memory_kv 据维度区分。
        if any(self.frames[i].visual_emb is not None for i in idx_list):
            visual_embs = torch.stack([
                self.frames[i].visual_emb if self.frames[i].visual_emb is not None
                else self.frames[i].pose_emb
                for i in idx_list
            ]).to(device)   # [k, dim] 或 [k, g*g, dim]
        else:
            visual_embs = pose_embs   # 退化路径（向后兼容）

        return pose_embs, visual_embs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_states(
        self, device: Optional[torch.device] = None
    ) -> Optional[Tensor]:
        """返回所有存储帧的 pose_emb，[K, dim]。

        适用于不需要检索筛选、直接把全部 memory 传入 cross-attention 的场景。
        若 bank 为空返回 None。
        """
        if not self.frames:
            return None
        device = device or self.frames[0].pose_emb.device
        return torch.stack([f.pose_emb for f in self.frames]).to(device)

    def size(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        """清空 Memory Bank（换 episode 时调用）。"""
        self.frames.clear()
        self.store_count = 0
        self.reject_count = 0
        self.evict_count = 0
        logger.info("MemoryBank: cleared.")

    def get_stats(self) -> dict:
        """返回 W&B 可记录的统计字典。"""
        surprises = [f.surprise_score for f in self.frames]
        ages = [f.age for f in self.frames]
        return {
            "memory/bank_size": float(len(self.frames)),
            "memory/store_count": float(self.store_count),
            "memory/reject_count": float(self.reject_count),
            "memory/evict_count": float(self.evict_count),
            "memory/surprise_mean": float(sum(surprises) / max(len(surprises), 1)),
            "memory/surprise_max": float(max(surprises)) if surprises else 0.0,
            "memory/surprise_min": float(min(surprises)) if surprises else 0.0,
            "memory/age_mean": float(sum(ages) / max(len(ages), 1)),
        }

    def __repr__(self) -> str:
        scores = [f"{f.surprise_score:.3f}" for f in self.frames]
        return (
            f"MemoryBank(size={len(self.frames)}/{self.max_size}, "
            f"surprise=[{', '.join(scores)}])"
        )


# ===========================================================================
# 三层 Memory Bank 实现（Orchestrator 2026-04-15 授权）
# ===========================================================================


class ShortTermBank:
    """短期记忆银行：FIFO，容量固定，无条件接受所有帧，保证 chunk 间连续性。

    设计意图：
      - 无论 surprise 高低，每帧都强制存入
      - 容量 1（默认），始终保留最新的 1 帧
      - 检索时全部返回，提供 chunk 间衔接的近期上下文
    """

    def __init__(self, cap: int = 1):
        """
        Args:
            cap: 容量（FIFO 队列长度），默认 1
        """
        self.cap = cap
        self.frames: List[MemoryFrame] = []

    def update(self, frame: MemoryFrame) -> None:
        """存入一帧（FIFO）。

        若未满直接 append；若已满弹出最旧帧（pop(0)）后 append。
        无条件接受，不检查 surprise。

        Args:
            frame: 待存入的 MemoryFrame
        """
        frame.tier = "short"
        if len(self.frames) >= self.cap:
            self.frames.pop(0)  # 弹出最旧帧
        self.frames.append(frame)

    def retrieve_all(self, device=None) -> List[MemoryFrame]:
        """返回全部帧（顺序：从旧到新）。

        Args:
            device: 保留参数（MemoryFrame 存 CPU tensor，调用方负责 .to(device)）

        Returns:
            所有存储帧的列表（浅拷贝）
        """
        return list(self.frames)

    def clear(self) -> None:
        """清空 ShortTermBank。"""
        self.frames.clear()

    def size(self) -> int:
        return len(self.frames)

    def get_stats(self) -> dict:
        """返回 W&B 可记录的统计字典。"""
        return {
            "memory/short_bank_size": float(len(self.frames)),
        }

    def __repr__(self) -> str:
        return f"ShortTermBank(size={len(self.frames)}/{self.cap})"


class MediumTermBank:
    """中期记忆银行：存高 surprise 帧（动态事件），age decay eviction。

    设计意图：
      - 写入条件：surprise > surprise_threshold（"意外"帧）
      - Eviction 策略：替换 effective_score 最低帧
        effective_score = surprise * (0.5 ** (age / half_life))
      - 检索策略：pose_emb cosine similarity，top-k
      - 目的：记住近期出现的场景变化（动态事件）
    """

    def __init__(
        self,
        cap: int = 8,
        surprise_threshold: float = 0.4,
        half_life: float = 10.0,
    ):
        """
        Args:
            cap:                最大容量
            surprise_threshold: 写入下限（帧 surprise > 此值才写入）
            half_life:          age decay 半衰期（单位：chunk 数），
                                effective_score = surprise * 0.5^(age / half_life)
        """
        self.cap = cap
        self.surprise_threshold = surprise_threshold
        self.half_life = half_life
        self.frames: List[MemoryFrame] = []
        self.store_count: int = 0
        self.reject_count: int = 0
        self.evict_count: int = 0

    def _effective_score(self, frame: MemoryFrame) -> float:
        """计算帧的有效 surprise 分数（含 age decay）。

        effective_score = surprise_score * (0.5 ** (age / half_life))
        """
        return frame.surprise_score * (0.5 ** (frame.age / self.half_life))

    def update(self, frame: MemoryFrame) -> None:
        """存入一帧（仅当 surprise > surprise_threshold）。

        若写入条件不满足 → reject_count += 1，直接返回。
        若满足且未满 → append，store_count += 1。
        若满足且已满：
          min_idx = argmin(effective_score)
          if frame.surprise > evicted.surprise → 替换，evict_count += 1，store_count += 1
          else → reject_count += 1

        Args:
            frame: 待存入的 MemoryFrame
        """
        if frame.surprise_score <= self.surprise_threshold:
            self.reject_count += 1
            return

        frame.tier = "medium"

        if len(self.frames) < self.cap:
            self.frames.append(frame)
            self.store_count += 1
            logger.debug(
                "MediumTermBank: added frame t=%d, surprise=%.4f, size=%d/%d",
                frame.timestep, frame.surprise_score, len(self.frames), self.cap,
            )
        else:
            # 找 effective_score 最低帧，无条件替换（新帧已通过 surprise_threshold 门槛）
            min_idx = min(
                range(len(self.frames)),
                key=lambda i: self._effective_score(self.frames[i]),
            )
            evicted = self.frames[min_idx]
            self.frames[min_idx] = frame
            self.evict_count += 1
            self.store_count += 1
            logger.debug(
                "MediumTermBank: replaced t=%d(eff=%.4f) with t=%d(s=%.4f)",
                evicted.timestep, self._effective_score(evicted),
                frame.timestep, frame.surprise_score,
            )

    def increment_age(self) -> None:
        """所有存储帧 age += 1。（只有 MediumTermBank 需要 age decay，每 chunk 调用一次）"""
        for frame in self.frames:
            frame.age += 1

    def retrieve(
        self,
        query_pose_emb: Tensor,
        top_k: int = 3,
        device=None,
    ) -> List[MemoryFrame]:
        """按 pose_emb cosine similarity 检索 top-k 帧。

        若存储帧数 < top_k，返回全部帧（不补空）。

        Args:
            query_pose_emb: [dim] 当前帧的 pose embedding，用于检索
            top_k:          返回帧数上限
            device:         保留参数（frames 存 CPU tensor，调用方负责 .to(device)）

        Returns:
            List[MemoryFrame]，按相似度由高到低排序，长度 ≤ top_k
        """
        if not self.frames:
            return []

        k = min(top_k, len(self.frames))
        query = query_pose_emb.float().cpu()
        all_pose_embs = torch.stack([f.pose_emb.float() for f in self.frames])  # [N, dim]
        sims = F.cosine_similarity(query.unsqueeze(0), all_pose_embs, dim=-1)  # [N]
        _, indices = torch.topk(sims, k=k)
        return [self.frames[i] for i in indices.tolist()]

    def clear(self) -> None:
        """清空 MediumTermBank。"""
        self.frames.clear()
        self.store_count = 0
        self.reject_count = 0
        self.evict_count = 0

    def size(self) -> int:
        return len(self.frames)

    def get_stats(self) -> dict:
        """返回 W&B 可记录的统计字典。"""
        surprises = [f.surprise_score for f in self.frames]
        ages = [f.age for f in self.frames]
        return {
            "memory/medium_bank_size": float(len(self.frames)),
            "memory/medium_surprise_mean": float(sum(surprises) / max(len(surprises), 1)),
            "memory/medium_store_count": float(self.store_count),
            "memory/medium_reject_count": float(self.reject_count),
            "memory/medium_evict_count": float(self.evict_count),
            "memory/medium_age_mean": float(sum(ages) / max(len(ages), 1)),
        }

    def __repr__(self) -> str:
        return (
            f"MediumTermBank(size={len(self.frames)}/{self.cap}, "
            f"threshold={self.surprise_threshold})"
        )


class LongTermBank:
    """长期记忆银行：存 stable（低 surprise）且 novel（语义新颖）的帧，支持场景重访。

    设计意图：
      - 写入条件：stable AND novel（同时满足）
        stable：surprise < stability_threshold（低惊讶度，稳定场景）
        novel：max cosine_sim(semantic_key, stored_keys) < novelty_threshold
               （semantic_key is None 时跳过 novelty check，退化路径）
      - Eviction 策略：移除 semantic redundancy 最高帧（max sim 最大者）
      - 检索策略：semantic_key cosine similarity（退化为 pose_emb cosine sim）
      - 目的：记住稳定的地图区域，支持"离开→回来"的场景重访一致性
    """

    def __init__(
        self,
        cap: int = 32,
        stability_threshold: float = 0.2,
        novelty_threshold: float = 0.7,
    ):
        """
        Args:
            cap:                  最大容量（默认 32）
            stability_threshold:  写入上限（帧 surprise < 此值认为 stable）
            novelty_threshold:    写入上限（新帧与已存储帧 max cosine_sim < 此值认为 novel）
        """
        self.cap = cap
        self.stability_threshold = stability_threshold
        self.novelty_threshold = novelty_threshold
        self.frames: List[MemoryFrame] = []
        self.store_count: int = 0
        self.reject_count: int = 0
        self.evict_count: int = 0

    def _novelty_check(self, semantic_key: Tensor) -> bool:
        """检查 semantic_key 是否与已存储帧语义不同（返回 True 表示 novel）。

        若 bank 为空，直接返回 True。
        若所有存储帧 semantic_key is None，返回 True（退化路径）。
        否则：sims = cosine_similarity(semantic_key, stored_keys)，
             return float(sims.max()) < novelty_threshold

        Args:
            semantic_key: [dim] 待检查帧的语义 key

        Returns:
            True 表示 novel（可写入），False 表示冗余（不写入）
        """
        if not self.frames:
            return True

        stored_keys_list = [
            f.semantic_key for f in self.frames if f.semantic_key is not None
        ]
        if not stored_keys_list:
            return True  # 所有存储帧 semantic_key is None，退化路径

        stored_keys = torch.stack([k.float() for k in stored_keys_list])  # [N, dim]
        sims = F.cosine_similarity(
            semantic_key.float().unsqueeze(0), stored_keys, dim=-1
        )  # [N]
        return float(sims.max()) < self.novelty_threshold

    def _find_most_redundant_idx(self) -> int:
        """找出与其他帧 semantic_key 最相似（redundancy 最高）的帧索引。

        对每帧 i：max_sim_i = max(cosine_sim(semantic_key_i, semantic_key_j) for j != i)
        返回 max_sim_i 最大的帧索引。

        若所有 semantic_key is None，fallback：返回 surprise_score 最低帧的索引。

        Returns:
            待淘汰帧的索引
        """
        keys = [f.semantic_key for f in self.frames]
        valid_indices = [i for i, k in enumerate(keys) if k is not None]

        if not valid_indices:
            # 退化 fallback：移除 surprise 最低帧
            return min(range(len(self.frames)),
                       key=lambda i: self.frames[i].surprise_score)

        # 为所有 valid 帧构建 key matrix
        valid_keys = torch.stack(
            [keys[i].float() for i in valid_indices]
        )  # [M, dim]

        # 对每个 valid 帧，计算与其他所有 valid 帧的最大相似度
        max_sims = []
        for vi, i in enumerate(valid_indices):
            if len(valid_indices) == 1:
                # 只有一帧，max_sim = 0（无比较对象）
                max_sims.append(0.0)
            else:
                # 排除自身
                other_keys = torch.cat(
                    [valid_keys[:vi], valid_keys[vi + 1:]], dim=0
                )  # [M-1, dim]
                sims = F.cosine_similarity(
                    valid_keys[vi].unsqueeze(0), other_keys, dim=-1
                )  # [M-1]
                max_sims.append(float(sims.max()))

        # 找 max_sim 最大的 valid 帧
        most_redundant_valid_pos = max_sims.index(max(max_sims))
        return valid_indices[most_redundant_valid_pos]

    def update(self, frame: MemoryFrame) -> None:
        """存入一帧（仅当 stable AND novel）。

        写入条件：
          1. stable：frame.surprise_score < stability_threshold
          2. novel：若 frame.semantic_key is not None 则做 novelty_check；
                    若 is None 则跳过 novelty check（退化路径）

        若两个条件不同时满足 → reject_count += 1，直接返回。
        若满足且未满 → append，store_count += 1。
        若满足且已满：
          evict_idx = _find_most_redundant_idx()
          替换该位置，evict_count += 1，store_count += 1

        Args:
            frame: 待存入的 MemoryFrame
        """
        # 条件1：stable 检查
        if frame.surprise_score >= self.stability_threshold:
            self.reject_count += 1
            return

        # 条件2：novel 检查（semantic_key is None 时跳过）
        if frame.semantic_key is not None:
            if not self._novelty_check(frame.semantic_key):
                self.reject_count += 1
                logger.debug(
                    "LongTermBank: frame t=%d rejected (not novel)",
                    frame.timestep,
                )
                return

        frame.tier = "long"

        if len(self.frames) < self.cap:
            self.frames.append(frame)
            self.store_count += 1
            logger.debug(
                "LongTermBank: added frame t=%d, surprise=%.4f, size=%d/%d",
                frame.timestep, frame.surprise_score, len(self.frames), self.cap,
            )
        else:
            evict_idx = self._find_most_redundant_idx()
            evicted = self.frames[evict_idx]
            self.frames[evict_idx] = frame
            self.evict_count += 1
            self.store_count += 1
            logger.debug(
                "LongTermBank: evicted t=%d(s=%.4f) → t=%d(s=%.4f)",
                evicted.timestep, evicted.surprise_score,
                frame.timestep, frame.surprise_score,
            )

    def retrieve(
        self,
        query_semantic_key: Optional[Tensor],
        query_pose_emb: Tensor,
        top_k: int = 2,
        device=None,
    ) -> List[MemoryFrame]:
        """按 semantic_key cosine similarity 检索 top-k 帧（退化为 pose_emb）。

        若 query_semantic_key is not None 且 bank 中有非 None semantic_key 帧：
          使用 semantic_key cosine_sim 检索。
        否则（退化路径）：使用 pose_emb cosine_sim。

        Args:
            query_semantic_key: [dim] 检索用语义 key（可为 None）
            query_pose_emb:     [dim] 当前帧 pose embedding（退化路径使用）
            top_k:              返回帧数上限
            device:             保留参数

        Returns:
            List[MemoryFrame]，按相似度由高到低排序，长度 ≤ top_k
        """
        if not self.frames:
            return []

        k = min(top_k, len(self.frames))

        # 判断是否可以用 semantic_key 检索
        use_semantic = (
            query_semantic_key is not None
            and any(f.semantic_key is not None for f in self.frames)
        )

        if use_semantic:
            query_vec = query_semantic_key.float().cpu()
            # 对有 semantic_key 的帧用 semantic_key，其余退化为 pose_emb
            embs = torch.stack([
                f.semantic_key.float() if f.semantic_key is not None
                else f.pose_emb.float()
                for f in self.frames
            ])  # [N, dim]
        else:
            query_vec = query_pose_emb.float().cpu()
            embs = torch.stack([f.pose_emb.float() for f in self.frames])  # [N, dim]

        sims = F.cosine_similarity(query_vec.unsqueeze(0), embs, dim=-1)  # [N]
        _, indices = torch.topk(sims, k=k)
        return [self.frames[i] for i in indices.tolist()]

    def retrieve_by_location(
        self,
        query_location: Tensor,       # [3] 当前帧绝对位置
        query_timestep: int,          # 当前帧 timestep
        top_k: int = 5,
        min_gap_frames: int = 0,      # 排除 timestep 距离 < 此值的近邻帧
        device=None,
    ) -> List[MemoryFrame]:
        """按绝对位置 L2 距离最近检索 top-k（Bug2），排除最近 min_gap_frames 帧（Bug3）。

        OP-2 重访检索路径（全部新增、与现有 retrieve 并存，不替换）：
          候选 = {f : f.location is not None and query_timestep - f.timestep >= min_gap_frames}
          按 L2(f.location, query_location) 升序，返回最近 top_k。

        与 retrieval_probe._retrieve_pose_abs_gap 的逻辑（L2 + 排除近邻 + largest=False）
        保持一致，便于对照诊断。

        Args:
            query_location:  [3] 当前帧绝对 c2w 平移向量（世界位置）
            query_timestep:  当前帧 timestep
            top_k:           返回帧数上限
            min_gap_frames:  排除 timestep 距离 < 此值的近邻帧（Bug3 近邻污染修复）
            device:          保留参数（frames 存 CPU tensor，调用方负责 .to(device)）

        Returns:
            List[MemoryFrame]，按 L2 距离升序（最近优先），长度 ≤ top_k；
            无满足条件的候选时返回 []。
        """
        cands = [
            f for f in self.frames
            if f.location is not None and (query_timestep - f.timestep) >= min_gap_frames
        ]
        if not cands:
            return []
        q = query_location.float().cpu().view(-1)
        locs = torch.stack([f.location.float().cpu().view(-1) for f in cands])  # [M,3]
        dists = torch.norm(locs - q.unsqueeze(0), dim=-1)  # [M]
        k = min(top_k, len(cands))
        _, idx = torch.topk(dists, k=k, largest=False)  # 最近
        return [cands[i] for i in idx.tolist()]

    def clear(self) -> None:
        """清空 LongTermBank。"""
        self.frames.clear()
        self.store_count = 0
        self.reject_count = 0
        self.evict_count = 0

    def size(self) -> int:
        return len(self.frames)

    def get_stats(self) -> dict:
        """返回 W&B 可记录的统计字典。"""
        surprises = [f.surprise_score for f in self.frames]
        return {
            "memory/long_bank_size": float(len(self.frames)),
            "memory/long_surprise_mean": float(sum(surprises) / max(len(surprises), 1)),
            "memory/long_store_count": float(self.store_count),
            "memory/long_reject_count": float(self.reject_count),
            "memory/long_evict_count": float(self.evict_count),
        }

    def __repr__(self) -> str:
        return (
            f"LongTermBank(size={len(self.frames)}/{self.cap}, "
            f"stability={self.stability_threshold}, novelty={self.novelty_threshold})"
        )


class ThreeTierMemoryBank:
    """三层 Memory Bank 主类（Orchestrator 2026-04-15 授权）。

    组合 ShortTermBank / MediumTermBank / LongTermBank，实现差异化存储和混合检索。

    路由规则：
      - ShortTermBank：无条件接受每帧（保证连续性）
      - MediumTermBank：surprise > surprise_threshold 时写入（动态事件）
      - LongTermBank：stable AND novel 时写入（稳定场景，支持重访）

    混合检索预算（Hybrid Retrieval Budget）：
      Short 1 + Medium top-3 + Long top-2 = 最多 6 帧（去重后可能更少）

    Cross-tier Dedup：
      合并后移除 pose_emb cosine_sim > dup_threshold 的冗余帧。
      Short 帧先选，自然享有最高优先级。

    用法（推理循环）：
        bank = ThreeTierMemoryBank(short_cap=1, medium_cap=8, long_cap=32)
        # 每 chunk 生成后：
        bank.update(pose_emb, latent, surprise, timestep,
                    visual_emb=visual_emb, chunk_id=clip_idx,
                    semantic_key=semantic_key)
        bank.increment_age()  # 新 chunk 开始前调用
        # 生成前检索：
        retrieved = bank.retrieve(query_pose_emb, query_semantic_key)
        if retrieved is not None:
            key_states, value_states = retrieved  # 各 [≤6, 5120]
    """

    def __init__(
        self,
        short_cap: int = 1,
        medium_cap: int = 8,
        long_cap: int = 32,
        surprise_threshold: float = 0.4,
        stability_threshold: float = 0.2,
        novelty_threshold: float = 0.7,
        half_life: float = 10.0,
        dup_threshold: float = 0.95,
    ):
        """
        Args:
            short_cap:            ShortTermBank 容量（默认 1）
            medium_cap:           MediumTermBank 容量（默认 8）
            long_cap:             LongTermBank 容量（默认 32）
            surprise_threshold:   MediumTermBank 写入下限（默认 0.4）
            stability_threshold:  LongTermBank stable 写入上限（默认 0.2）
            novelty_threshold:    LongTermBank novelty 写入上限（默认 0.7）
            half_life:            MediumTermBank age decay 半衰期（默认 10.0 chunks）
            dup_threshold:        Cross-tier dedup 阈值（pose_emb cosine_sim > 此值认为冗余，默认 0.95）
        """
        self.short = ShortTermBank(cap=short_cap)
        self.medium = MediumTermBank(
            cap=medium_cap,
            surprise_threshold=surprise_threshold,
            half_life=half_life,
        )
        self.long = LongTermBank(
            cap=long_cap,
            stability_threshold=stability_threshold,
            novelty_threshold=novelty_threshold,
        )
        self.dup_threshold = dup_threshold

    def update(
        self,
        pose_emb: Tensor,
        latent: Tensor,
        surprise_score: float,
        timestep: int,
        visual_emb: Optional[Tensor] = None,
        chunk_id: int = 0,
        semantic_key: Optional[Tensor] = None,
        location: Optional[Tensor] = None,
    ) -> None:
        """存入一帧（按路由规则分发到各层）。

        所有 tensor 在存入前 .detach().cpu()（与 MemoryBank.update 一致）。

        Args:
            pose_emb:       [dim=5120] 当前帧的 pose embedding
            latent:         [z_dim, h, w] VAE latent
            surprise_score: NFPHead 计算的 per-frame surprise（越大越"意外"）
            timestep:       当前帧在原始视频中的帧索引
            visual_emb:     [dim=5120] VAE latent 投影到模型空间的视觉嵌入（可选）
            chunk_id:       所属 chunk 编号
            semantic_key:   [dim=5120] 由 model_with_memory.get_semantic_key() 计算
                            （待 model_with_memory.py 实现 get_semantic_key() 后传入）
            location:       [3] 绝对 c2w 平移向量（世界位置），用于 retrieve_revisit
                            地点重访检索（OP-2 Bug2 修复）；None 时该帧不参与重访检索（向后兼容）
        """
        frame = MemoryFrame(
            pose_emb=pose_emb.detach().cpu(),
            latent=latent.detach().cpu(),
            surprise_score=float(surprise_score),
            timestep=int(timestep),
            visual_emb=visual_emb.detach().cpu() if visual_emb is not None else None,
            chunk_id=int(chunk_id),
            age=0,
            semantic_key=semantic_key.detach().cpu() if semantic_key is not None else None,
            location=location.detach().cpu() if location is not None else None,
        )

        # 路由规则：各层独立写入（各层 update() 会写 frame.tier，需传入独立副本避免互相覆盖）
        # Short：无条件接受每帧
        self.short.update(dataclasses.replace(frame))
        # Medium：surprise > surprise_threshold 时写入
        self.medium.update(dataclasses.replace(frame))
        # Long：stable AND novel 时写入
        self.long.update(dataclasses.replace(frame))

    def retrieve(
        self,
        query_pose_emb: Tensor,
        query_semantic_key: Optional[Tensor] = None,
        short_n: int = 1,
        medium_k: int = 3,
        long_k: int = 2,
        device: Optional[torch.device] = None,
        return_tier_ids: bool = False,
        return_latents: bool = False,
    ) -> Optional[Tuple[Tensor, ...]]:
        """混合检索，返回去重后的 (key_states, value_states)，可选返回 tier_ids / latents。

        检索预算：Short short_n 帧 + Medium top-medium_k + Long top-long_k。
        Cross-tier Dedup：移除 pose_emb cosine_sim > dup_threshold 的冗余帧。
        优先级：Short > Medium > Long（Short 先选，自然优先）。

        Args:
            query_pose_emb:     [dim=5120] 当前帧的 pose embedding
            query_semantic_key: [dim=5120] 语义检索 key。
                                为 None 时：LongTermBank 退化为 pose_emb cosine_sim 检索；
                                LongTermBank.update() 跳过 novelty check，仅做 stability 检查。
                                bank.size()==0 时无需计算 query_semantic_key（不会调用 retrieve）。
            short_n:            从 ShortTermBank 取的帧数上限（默认 1，即全部）
            medium_k:           从 MediumTermBank 取的帧数（默认 3）
            long_k:             从 LongTermBank 取的帧数（默认 2）
            device:             输出 tensor 放置的设备（None 则跟随 query_pose_emb.device）
            return_tier_ids:    若 True，额外返回第三个张量 tier_ids [K] int64，
                                其中 0=Short / 1=Medium / 2=Long（Innovation 10: Tier Embedding）；
                                tier="" 时映射到 0。
                                若 False（默认），行为与修改前完全一致，返回 Tuple[Tensor, Tensor]
                                （向后兼容约束：v3 的 key_states, value_states = bank.retrieve(...) 不 break）
            return_latents:     若 True，**额外**在返回元组**末尾**追加 latents
                                [K, z_dim, h, w]（所选帧 latent stack，供 v5 MemoryEncoder 用）。
                                False（默认）时行为与修改前**逐字节一致**（v3/v4 既有调用零破坏）。
                                可与 return_tier_ids 组合（latents 始终是最后一个元素）。

        Returns:
            （以下按 (return_tier_ids, return_latents) 组合，latents 恒为末尾元素）
            (False, False)（默认）: (pose_embs, visual_embs)，各 [K, 5120]
            (True,  False):        (pose_embs, visual_embs, tier_ids)
            (False, True):         (pose_embs, visual_embs, latents)
            (True,  True):         (pose_embs, visual_embs, tier_ids, latents)
            其中 latents: [K, z_dim, h, w]（所选帧 latent stack，各帧同分辨率可 stack）。
            若合并后为空（bank 全空）返回 None
        """
        # 1. 收集各层帧
        short_frames = self.short.retrieve_all(device=None)[:short_n]
        medium_frames = self.medium.retrieve(query_pose_emb, top_k=medium_k, device=None)
        long_frames = self.long.retrieve(
            query_semantic_key, query_pose_emb, top_k=long_k, device=None
        )

        # 2. 按优先级拼接：Short 优先 > Medium > Long
        all_frames = short_frames + medium_frames + long_frames

        if not all_frames:
            return None

        # 3. Cross-tier dedup：移除 pose_emb cosine_sim > dup_threshold 的冗余帧
        # Short 帧先选，自然享有优先级
        selected_frames: List[MemoryFrame] = []
        selected_pose_embs: List[Tensor] = []

        for frame in all_frames:
            if not selected_frames:
                selected_frames.append(frame)
                selected_pose_embs.append(frame.pose_emb.float())
                continue
            stacked = torch.stack(selected_pose_embs)  # [N, dim]
            sims = F.cosine_similarity(
                frame.pose_emb.float().unsqueeze(0), stacked, dim=-1
            )  # [N]
            if float(sims.max()) < self.dup_threshold:
                selected_frames.append(frame)
                selected_pose_embs.append(frame.pose_emb.float())

        if not selected_frames:
            return None

        # 4. 构建返回张量（与 MemoryBank.retrieve 格式一致：Tuple[pose_embs, visual_embs]）
        device = device or query_pose_emb.device
        pose_embs = torch.stack(
            [f.pose_emb for f in selected_frames]
        ).to(device)  # [K, dim]

        # Exp2 spatial-V：visual_emb 为 [g*g, dim] 时 stack 成 [K, g*g, dim]（patch 级）；
        #   帧级 [dim] 时为 [K, dim]（旧行为）。上层 build_memory_kv 据维度区分。
        if any(f.visual_emb is not None for f in selected_frames):
            visual_embs = torch.stack([
                f.visual_emb if f.visual_emb is not None else f.pose_emb
                for f in selected_frames
            ]).to(device)  # [K, dim] 或 [K, g*g, dim]
        else:
            visual_embs = pose_embs  # 退化路径

        logger.debug(
            "ThreeTierMemoryBank.retrieve: short=%d medium=%d long=%d → selected=%d (after dedup)",
            len(short_frames), len(medium_frames), len(long_frames), len(selected_frames),
        )

        # v5：可选 latents（恒为末尾元素，供 MemoryEncoder 用）。selected_frames 非空
        # （前面已 return None 处理空情形），各帧 latent 同分辨率可 stack。
        latents = None
        if return_latents:
            latents = torch.stack([f.latent for f in selected_frames]).to(device)  # [K, z_dim, h, w]

        if not return_tier_ids:
            # 向后兼容：v3 路径，不返回 tier_ids
            if return_latents:
                return pose_embs, visual_embs, latents
            return pose_embs, visual_embs

        # Innovation 10：构建 tier_ids 张量（int64），对应 selected_frames 顺序
        # 映射规则：short→0, medium→1, long→2, ""→0（默认）
        _TIER_MAP = {"short": 0, "medium": 1, "long": 2}
        tier_id_list = [_TIER_MAP.get(f.tier, 0) for f in selected_frames]
        tier_ids = torch.tensor(tier_id_list, dtype=torch.long, device=device)  # [K]
        if return_latents:
            return pose_embs, visual_embs, tier_ids, latents
        return pose_embs, visual_embs, tier_ids

    def retrieve_revisit(
        self,
        query_location: Tensor,
        query_timestep: int,
        top_k: int = 5,
        min_gap_frames: int = 0,
        device=None,
        return_latents: bool = False,
    ):
        """地点重访检索：只走 Long tier（Bug3：避开 short/medium 的近邻帧）+ 位置 L2（Bug2）。

        OP-2 重访检索路径（全部新增、与现有 retrieve 并存，不替换）。
        只查 self.long（LongTermBank），避免 short/medium tier 装满最近帧时
        在合并检索里挤掉 Long tier 的老重访帧（Bug3 近邻污染）。

        Args:
            query_location:  [3] 当前帧绝对 c2w 平移向量（世界位置）
            query_timestep:  当前帧 timestep
            top_k:           返回帧数上限
            min_gap_frames:  排除 timestep 距离 < 此值的近邻帧
            device:          保留参数
            return_latents:  若 True，**额外**返回所选帧的 latent stack（供 v5 MemoryEncoder 用）。
                             False（默认）时行为与修改前**逐字节一致**（只返回 List[MemoryFrame]，
                             v4 既有调用零破坏）。

        Returns:
            return_latents=False（默认）:
                List[MemoryFrame]，按位置 L2 距离升序，长度 ≤ top_k（与修改前完全一致）。
            return_latents=True:
                (frames, latents)：
                  frames  = List[MemoryFrame]（同上）
                  latents = [K, z_dim, h, w]（各帧 latent stack，K=len(frames)）；
                            frames 为空时 latents 为 None。
                            各帧 latent 同分辨率（同 episode 同 VAE），可直接 stack。
        """
        frames = self.long.retrieve_by_location(
            query_location=query_location,
            query_timestep=query_timestep,
            top_k=top_k,
            min_gap_frames=min_gap_frames,
            device=device,
        )
        if not return_latents:
            # 向后兼容：行为与修改前逐字节一致
            return frames

        # return_latents 路径：额外 stack 所选帧的 latent（MemoryFrame.latent: [z_dim, h, w]）
        if not frames:
            latents = None
        else:
            latents = torch.stack([f.latent for f in frames])  # [K, z_dim, h, w]
            if device is not None:
                latents = latents.to(device)
        return frames, latents

    def increment_age(self) -> None:
        """只对 MediumTermBank 帧 +1（Long/Short 不依赖 age decay）。

        在每个新 chunk 开始前调用。
        """
        self.medium.increment_age()

    def clear(self) -> None:
        """清空所有层。"""
        self.short.clear()
        self.medium.clear()
        self.long.clear()
        logger.info("ThreeTierMemoryBank: cleared.")

    def size(self) -> int:
        """返回三层帧总数。"""
        return self.short.size() + self.medium.size() + self.long.size()

    def get_stats(self) -> dict:
        """合并三层统计，返回 W&B 可记录的统计字典。"""
        stats = {}
        stats.update(self.short.get_stats())
        stats.update(self.medium.get_stats())
        stats.update(self.long.get_stats())
        stats["memory/three_tier_total_size"] = float(self.size())
        return stats

    def __repr__(self) -> str:
        return (
            f"ThreeTierMemoryBank("
            f"short={self.short}, "
            f"medium={self.medium}, "
            f"long={self.long}, "
            f"dup_threshold={self.dup_threshold})"
        )
