"""oracle_injection.py — Exp2 Oracle 记忆注入测试（v4 ThreeTierMemoryBank / 冻结 DiT）

目的（D-06 / experiment_design.md Exp 2）
-----------------------------------------
剥离检索，单独测「冻结 DiT 能否利用注入的记忆」。在重访 target clip 处直接注入
GT 历史帧（oracle memory）——绕过 bank 检索/NFP surprise/存储逻辑，直接构造注入用
的 memory K/V——并弱化首帧 image condition（逼模型靠记忆），对照 oracle / off /
wrong-memory，看生成的重访帧是否更一致。

⚠️ 重要前提（结果有效性依赖）
----------------------------
本脚本的有意义结果**依赖 memory_cross_attn 已训练**（OP-1 / Bug1 已修 + 重训）。
在 epoch_4（memory_cross_attn 随机初始化，见 OP-1 / F-12：use_reentrant=True 梯度
检查点 + Stage1 冻结 backbone 导致 memory_cross_attn 全程零梯度）上跑，只能作为
**负对照基线**——此时 gate=0.1 固定 + 投影随机，注入任何 K/V 都只是加结构化噪声，
oracle / off / wrong 的差别**不能**用来肯定或否定 idea。
主结果须用 **Exp0 重训后的 checkpoint**（memory 模块真正训练过）跑。

三档 Tier 对照（D-06）
---------------------
| --tier_config | 含义 | 测什么 |
|---------------|------|-------|
| full         | 弱化首帧 + 注入 **bank Short+Medium+Long 检索帧** | Memory Bank 整体能不能用（最宽松） |
| medium_long  | 弱化首帧 + 注入 **bank Medium+Long 检索帧** | Medium+Long 能不能用（idea 主打） |
| oracle_only  | 弱化首帧 + 三层全关，**只注入 oracle GT 帧**（绕过 bank） | 注入本身有没有被用到（最严格） |

full / medium_long 档（memory_mode=oracle）现在真正消费 bank：先 populate bank over
episode（query clip 之前的帧，逐帧 bank.update，口径同 retrieval_probe._eval_episode），
再从启用的 tier 检索（Long 用位置检索 retrieve_revisit / Medium 用 medium.retrieve /
Short 用 short.retrieve_all），把检索帧构造成 memory K/V 注入。检索方式与 Exp1 探针同源。

注入内容对照（--memory_mode）
-----------------------------
- oracle : 注入该重访点首次访问时的 GT 历史帧（正确记忆）
- off    : 不注入记忆（memory_states=None，baseline）
- wrong  : 注入同 episode 远距离位置的 GT 帧（错误场景）——检验"是不是任何注入都改变输出"
- random : 从**同 episode 历史帧随机抽一帧**注入（与 oracle 同候选池，排除 oracle 帧 +
           近邻）——confound 对照「任何注入都涨」，判据 oracle ≫ {random ≈ off}

注入帧 tier label（--inject_tier，D-06 配套）
--------------------------------------------
oracle / random / wrong 三臂注入帧的 tier_ids 统一用 --inject_tier 指定值
（short=0/medium=1/long=2），默认 **medium**。修「Long tier embedding 从未训练」坑：
Stage 2 训练日志 Long tier 全程 0l/32 从未填过 → tier_emb 的 tier_id=2(Long) 那一行
从未被训练、仍随机初始化；旧版硬编码 long=2 等于在 K 上叠未训练的随机 tier 向量给注入
掺噪、压低 oracle 效果。medium(=1) 层训练时一直有内容、tier_emb 训练过。
（bank 检索路径 full/medium_long 的 tier_ids 仍按检索帧真实来源标，不受此参数影响。）

评测指标（exp2_redesign_draft.md「Exp2 评测指标」）
-------------------------------------------------
一遍过帧同时算两个核心指标：
- 像素 SSIM   : revisit_consistency_{max,mean,last}
- 语义 DINO   : revisit_consistency_dino_{max,mean,last}（DINOv2 dinov2_vits14 cosine，
                lazy-load 全局缓存；加载失败 graceful 跳过 dino 字段，不让 eval 崩）

首帧弱化（--weaken_first_frame，D-06）
-------------------------------------
砍掉首帧 image condition 这条强信号通道，制造"模型不靠记忆就答不出"的清洁实验条件。
- zero  : 首帧置零（中性灰，**默认**）——温和锚点，给记忆留贡献空间又不毁场景
- none  : 不弱化（首帧条件保留）
- noise : 随机 RGB 替代首帧（仅消融）——F-18：随机 RGB 会摧毁 i2v 场景锚点，
          三臂（off/oracle/wrong）都生成噪点/无关视频、revisit 指标全部地板化、
          oracle-vs-wrong 无法对比，故不再作默认。
首帧弱化在 WanI2V.generate() 的 `img` 入口处替换 PIL 图像实现——不修改 image2video.py。

依赖与约束
----------
- 复用 infer_v4.py 的 load/convert/patch pipeline（import，不重写）
- 复用 retrieval_probe.py 的 episode 加载 + 绝对位置 GT 重访判定（import）
- 不修改 infer_v4.py / train_v4 / memory_bank.py / model_with_memory.py
- 代码风格与 infer_v4 / retrieval_probe 一致（中文注释 / type hints / docstring）
- 单 GPU 即可运行（不强制多卡；本脚本不支持 Ulysses SP，逻辑更清晰）

跨模块数据契约（oracle 帧注入 / bank 检索帧注入，同一契约）
-----------------------------------------------------------
- memory_states (K)       : [1, K, 5120]，pose_emb（get_projected_frame_embs）
                            oracle_only=oracle GT 帧；full/medium_long=bank 检索帧的 frame.pose_emb
- memory_value_states (V) : [1, K, 5120]，visual_emb（get_projected_latent_emb）
- tier_ids                : [K] int64 或 None
                            oracle_only / random / wrong：全标 --inject_tier（默认 medium=1）；
                            full/medium_long：按检索帧来源标 Short=0/Medium=1/Long=2
  消费方：_patch_pipeline_memory → MemoryCrossAttention.forward（两条注入路径同接口）
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# sys.path 设置（与 infer_v4.py / retrieval_probe.py 一致）
# ---------------------------------------------------------------------------

_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # → src/pipeline/
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Exp2 Oracle 记忆注入测试 — 冻结 DiT，注入 GT 历史帧 + 弱化首帧，"
            "对照 oracle/off/wrong × 三档 tier_config。"
        )
    )
    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_full_train.csv")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录（视频 + summary.md）")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")

    # ---- 模型权重（与 infer_v4 一致）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="基础模型目录（lingbot-world checkpoint，同 infer_v4）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="重训后的 v4 low_noise_model checkpoint 目录"
                        "（memory 模块训练好的）；缺失则用 base 权重（memory 随机，仅负对照）")
    p.add_argument("--ft_high_model_dir", type=str, default=None,
                   help="dual 训练 high_noise_model 目录（可选）")

    # ---- 三档 Tier 对照（D-06）----
    p.add_argument("--tier_config", type=str, default="oracle_only",
                   choices=["full", "medium_long", "oracle_only"],
                   help="full=Short/Medium/Long 全保留；medium_long=关 Short；"
                        "oracle_only=三层全关只注入 oracle GT 帧（默认）")
    # ---- 注入内容对照 ----
    p.add_argument("--memory_mode", type=str, default="oracle",
                   choices=["oracle", "off", "wrong", "random"],
                   help="oracle=注入 GT 历史帧 / off=不注入 / wrong=注入错误场景帧 / "
                        "random=从同 episode 历史帧随机抽一帧注入（confound 对照：判据 "
                        "oracle ≫ {random ≈ off}，排除'任何注入都涨'）")
    # ---- gate 强制覆写（诊断用：解耦"接口有效性"与"训练是否开门"）----
    p.add_argument("--gate_override", type=float, default=None,
                   help="诊断用——强制把所有 block 的 memory cross-attn 有效 gate"
                        "（post-tanh 乘子）设为该值（建议 0.1/0.3/1.0），用于解耦"
                        "\"接口有效性\"与\"训练是否开门\"：fresh 重训中 gate 卡在 ~1e-3"
                        "（exposure bias + gate 同时门控前/反向自饿），训出来的 gate ≈ 0"
                        "→ 注入≈恒等 → oracle/off/wrong/random 四臂塌成几乎一样、无法判别"
                        "\"接口是否有用\"。强制开门后：oracle ≫ {random≈off} → 接口有用、"
                        "问题在训练；oracle≈off 或出噪点 → 接口真有问题（触发 #3）。"
                        "默认 None=用 checkpoint 训出来的 gate（零改动路径，行为不变）。")
    # ---- 注入帧 tier label（D-06 配套：修 tier embedding 未训练坑）----
    p.add_argument("--inject_tier", type=str, default="medium",
                   choices=["short", "medium", "long"],
                   help="oracle/random/wrong 三种注入帧的 tier label 统一取此值"
                        "（short=0/medium=1/long=2，叠到 MemoryCrossAttention 的 K 上）。"
                        "默认 medium：Stage 2 训练日志显示 Long tier 全程 0l/32 从未填过，"
                        "tier_emb 的 tier_id=2(Long) 那一行从未被训练、仍是随机初始化；"
                        "而 Medium(=1) 层训练时一直有内容、tier_emb 训练过。旧版硬编码 long=2 "
                        "等于在 K 上叠未训练的随机 tier 向量给注入掺噪、压低 oracle 效果、污染判读。"
                        "保留 long 选项做敏感性对照。")
    # ---- 首帧弱化（D-06）----
    p.add_argument("--weaken_first_frame", type=str, default="zero",
                   choices=["noise", "zero", "none"],
                   help="zero=置零中性灰（默认，温和锚点）/ none=不弱化 / "
                        "noise=随机 RGB 替首帧——已知会摧毁 i2v 场景锚点、使 revisit "
                        "指标全部地板化（F-18），仅作消融用。")

    # ---- 重访点判定（复用 retrieval_probe 口径）----
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
    p.add_argument("--max_revisit_points", type=int, default=2,
                   help="每 episode 最多取多少个重访点（控制生成耗时）")
    p.add_argument("--num_oracle_frames", type=int, default=2,
                   help="每个重访点注入多少 oracle/wrong GT 帧（K）")

    # ---- 生成参数 ----
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=70,
                   help="diffusion 采样步数（infer_v4 中名为 sample_steps）")
    p.add_argument("--sample_shift", type=float, default=10.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（min_time_gap_sec→帧数换算 + 保存）")

    # ---- Bank 超参数（full / medium_long 模式构建 bank 时使用，与 v4 默认对齐）----
    p.add_argument("--medium_cap", type=int, default=8)
    p.add_argument("--long_cap", type=int, default=32)
    p.add_argument("--surprise_threshold", type=float, default=0.4)
    p.add_argument("--stability_threshold", type=float, default=0.2)
    p.add_argument("--novelty_threshold", type=float, default=0.7)
    p.add_argument("--half_life", type=float, default=10.0)
    p.add_argument("--dup_threshold", type=float, default=0.95)
    p.add_argument("--visual_fusion_alpha", type=float, default=0.7)

    return p.parse_args()


# tier 名称 → MemoryCrossAttention.tier_emb 索引（Short=0/Medium=1/Long=2，
# 与 memory_attention.py:116 nn.Embedding(3, dim) 的行序一致）
_TIER_NAME_TO_ID = {"short": 0, "medium": 1, "long": 2}


def _apply_gate_override(model, gate_override: float) -> None:
    """诊断用——强制覆写所有 memory cross-attn 的有效 gate（与训练真实行为解耦）。

    背景：fresh 重训中 memory cross-attn 的 gate 卡在 ~1e-3 且往 0 缩（exposure bias：
    单步 diffusion loss 不奖励开门；gate 同时门控前/反向 → gate≈0 时 q/k/v/o 几乎拿不到
    梯度、模块自饿）。用这种 checkpoint 跑 4 臂 eval，注入 ≈ 恒等 → oracle/off/wrong/
    random 四臂塌成几乎一样，无法判别"接口是否有用"。本旋钮把"接口有效性"与"训练是否
    开门"解耦：强制开门后若 oracle ≫ {random≈off} → 接口有用、问题在训练；若 oracle≈off
    或出噪点 → 接口真有问题。

    实现：MemoryCrossAttention.forward 末尾是 `torch.tanh(self.gate) * out`，要让 post-tanh
    有效乘子 = g，需把 self.gate.data 设为 atanh(g)。对 g 做 [-0.9999,0.9999] clamp 防 atanh→inf。
    **不动 out_rms_cap 的 clip-by-RMS 硬上界**（#2 幅度上界始终生效，强制开门也不会炸成噪点
    —— 这正是 #2 的价值）。

    ⚠️ 此结果不能等同于"训练后真实行为"，仅用于接口判别。

    Args:
        model:         WanModelWithMemory（pipeline.low_noise_model）
        gate_override: 目标 post-tanh 有效 gate 值（如 0.1/0.3/1.0）
    """
    from memory_module.memory_attention import MemoryCrossAttention

    # 命中模块：优先按类名 isinstance；退而求其次按 (gate, out_rms_cap) 双属性匹配，
    # 避免误伤其它带 gate 的模块（如 WanAttentionBlock）。
    def _is_mem_attn(m) -> bool:
        return isinstance(m, MemoryCrossAttention) or (
            hasattr(m, "gate") and hasattr(m, "out_rms_cap")
        )

    modules = [m for m in model.modules() if _is_mem_attn(m)]

    # 覆写前先采集 trained tanh(gate) 的跨 block 均值（用于对照）
    trained_vals: List[float] = []
    for m in modules:
        try:
            trained_vals.append(float(torch.tanh(m.gate.detach()).mean().item()))
        except Exception:  # noqa: BLE001
            continue
    mean_trained = (sum(trained_vals) / len(trained_vals)) if trained_vals else float("nan")

    # g clamp 防 atanh→inf；atanh 的 dtype/device 与原 param 一致
    g_clamped = min(max(float(gate_override), -0.9999), 0.9999)
    n_overridden = 0
    with torch.no_grad():
        for m in modules:
            target = torch.atanh(
                torch.tensor(g_clamped, dtype=m.gate.dtype, device=m.gate.device)
            )
            m.gate.data.fill_(target.item())
            n_overridden += 1

    logger.warning(
        "⚠️ [诊断模式] gate 强制覆写：所有 memory cross-attn 有效 gate（post-tanh）"
        "被强制设为 %.4f（覆写前 mean trained tanh(gate)=%.6f）；命中并覆写 %d 个 module。"
        "out_rms_cap 幅度硬上界不动（强制开门也不会炸成噪点）。"
        "⚠️ 此结果不等同于\"训练后真实行为\"，仅用于解耦\"接口有效性\"与\"训练是否开门\"。",
        g_clamped, mean_trained, n_overridden,
    )


# ---------------------------------------------------------------------------
# 重访点数据结构
# ---------------------------------------------------------------------------

@dataclass
class RevisitPoint:
    """一个重访点：query 帧 q 重新到达了首访帧 first_visit_frame 所在的地点。"""
    episode_id: str
    query_frame: int                 # 全局帧索引（episode 拼接后）
    first_visit_frame: int           # 该地点首次访问的 GT 历史帧（全局索引）
    gt_past_frames: List[int]        # 全部命中的过去帧（用于诊断）


# ---------------------------------------------------------------------------
# 复用 retrieval_probe 的 episode 加载 + GT 重访判定（import，不重写）
# ---------------------------------------------------------------------------

from pipeline.eval.retrieval_probe import (  # noqa: E402
    EpisodeData,
    load_episode_clips,
    build_episode_data,
    compute_gt_revisit,
    _decode_episode_video,
    _vae_encode_batched,
    _expand_latents_to_frames,
    _compute_pose_embs_episode,
    _compute_visual_embs_from_latents,
    _compute_surprise_visual_cosine,
)

# 复用 infer_v4 的 pipeline 装载/转换/注入（import，不重写）
from pipeline.v4.infer_v4 import (  # noqa: E402
    _load_ft_model_and_prepare_ckpt,
    _convert_pipeline_to_memory,
    _patch_pipeline_memory,
    _unpatch_pipeline_memory,
)


def _find_revisit_points(
    ep: EpisodeData,
    args,
    min_time_gap_frames: int,
) -> List[RevisitPoint]:
    """对 episode 找重访点：复用 retrieval_probe.compute_gt_revisit 的绝对位置 +
    yaw + time_gap + 中间分离判定。

    对每个 query 帧 q（有 GT 过去帧），取最早的过去帧作为 first_visit_frame
    （"该地点首次访问"语义）。按 query_frame 升序，取前 max_revisit_points 个。
    """
    gt = compute_gt_revisit(
        ep,
        hit_dist=args.hit_dist,
        hit_yaw=args.hit_yaw,
        intermediate_separation=args.intermediate_separation,
        min_time_gap_frames=min_time_gap_frames,
    )
    points: List[RevisitPoint] = []
    for q in sorted(gt.keys()):
        past = sorted(gt[q])
        if not past:
            continue
        points.append(
            RevisitPoint(
                episode_id=ep.episode_id,
                query_frame=int(q),
                first_visit_frame=int(past[0]),   # 最早访问 = 首访
                gt_past_frames=[int(x) for x in past],
            )
        )
    if args.max_revisit_points > 0:
        points = points[:args.max_revisit_points]
    return points


# ---------------------------------------------------------------------------
# 帧索引 → clip 切片（用于取 query clip 的 pose/action/intrinsics 与首帧图像）
# ---------------------------------------------------------------------------

def _frame_to_clip_slice(
    ep: EpisodeData,
    center_frame: int,
    frame_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """从 episode 拼接后的全序列里，截取以 center_frame 起始的 frame_num 帧
    pose/action/intrinsics（不足时回退到末尾对齐）。

    Returns:
        (poses[frame_num,4,4], actions[frame_num,4], intrinsics[frame_num,4], start)
    """
    T = ep.poses.shape[0]
    start = center_frame
    end = start + frame_num
    if end > T:
        start = max(0, T - frame_num)
        end = start + frame_num
    poses = ep.poses[start:end]
    actions = ep.actions[start:end]
    intr = ep.intrinsics[start:end]
    # 不足 frame_num 时用末帧 pad（与 infer_v4 fallback 思路一致）
    if poses.shape[0] < frame_num:
        pad_n = frame_num - poses.shape[0]
        poses = np.concatenate([poses, np.tile(poses[-1:], (pad_n, 1, 1))], axis=0)
        actions = np.concatenate([actions, np.tile(actions[-1:], (pad_n, 1))], axis=0)
        intr = np.concatenate([intr, np.tile(intr[-1:], (pad_n, 1))], axis=0)
    return poses, actions, intr, start


def _weaken_image(img: Image.Image, mode: str, rng: np.random.Generator) -> Image.Image:
    """按 mode 弱化首帧 image condition（砍掉 I2V 强信号通道，D-06）。

    在 PIL 入口替换图像，避免修改 image2video.py。

    Args:
        img:  原始首帧 PIL 图像
        mode: noise（随机噪声）/ zero（中性灰）/ none（不弱化）
        rng:  numpy Generator（保证可复现）

    Returns:
        弱化后的 PIL 图像（mode=none 时原样返回）
    """
    if mode == "none":
        return img
    w, h = img.size
    if mode == "noise":
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if mode == "zero":
        # 置零 = [-1,1] 空间的 0 → 像素 127.5（中性灰），避免纯黑被 VAE 当强信号
        arr = np.full((h, w, 3), 128, dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    return img


# ---------------------------------------------------------------------------
# Oracle 帧 → memory K/V 构造（绕过 bank 检索 / NFP surprise / 存储逻辑）
# ---------------------------------------------------------------------------

def _build_oracle_memory_kv(
    model,
    pipeline,
    ep: EpisodeData,
    latents_per_frame: torch.Tensor,    # [T, z_dim, lat_h, lat_w]，CPU
    frame_indices: List[int],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """直接构造注入用 memory K/V（最干净做法，task 注意点 2）。

    对每个 GT 帧 i：
      K (pose_emb) ：用该帧所在 clip 的 pose 经 get_projected_frame_embs 取对应帧
      V (visual_emb)：用该帧的 VAE latent 经 get_projected_latent_emb

    这条路径绕过 bank.retrieve / NFP surprise / 三层存储，把 GT 历史帧直接 VAE
    encode 后作为 memory_states 传给 memory_cross_attn（与 bank 检索路径同 shape 契约）。

    Args:
        model:             WanModelWithMemory（pipeline.low_noise_model）
        pipeline:          WanI2V
        ep:                EpisodeData（提供 poses/actions/intrinsics）
        latents_per_frame: episode 全帧 VAE latent（用于 V）
        frame_indices:     要注入的 GT 帧全局索引列表
        device:            目标设备

    Returns:
        (key_states [K,5120], value_states [K,5120])，CPU；无有效帧时 None
    """
    from memory_module.model_with_memory import WanModelWithMemory
    from pipeline.common.dataloader import build_dit_cond_dict

    if not isinstance(model, WanModelWithMemory):
        logger.warning("model 非 WanModelWithMemory，无法构造 oracle K/V")
        return None
    if not frame_indices:
        return None

    T = ep.poses.shape[0]
    _h, _w = (int(x) for x in _SIZE_HW)
    vae_stride_t = 4

    # 确保相关投影层在 device（offload 安全）
    for _attr in ("patch_embedding_wancamctrl", "c2ws_hidden_states_layer1",
                  "c2ws_hidden_states_layer2"):
        if hasattr(model, _attr):
            getattr(model, _attr).to(device)
    if hasattr(model, "latent_proj"):
        model.latent_proj.to(device)

    key_list: List[torch.Tensor] = []
    val_list: List[torch.Tensor] = []

    for fi in frame_indices:
        if fi < 0 or fi >= T:
            continue
        # --- K：该帧的 pose_emb ---
        # 取以 fi 为中心的 clip pose 段，算 plucker → projected frame embs，
        # 取段内对应该帧的 latent index。
        poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(ep, fi, _ORACLE_CLIP_FRAMES)
        try:
            cond = build_dit_cond_dict(
                poses=torch.from_numpy(poses_c).float(),
                actions=torch.from_numpy(acts_c).float(),
                intrinsics=torch.from_numpy(intr_c).float(),
                height=_h, width=_w,
            )
            plucker = cond["c2ws_plucker_emb"][0].to(device)  # [1,448,lat_f,lat_h,lat_w]
            with torch.no_grad():
                frame_embs = model.get_projected_frame_embs(plucker)  # [lat_f,5120]
            local = fi - seg_start
            lat_idx = min(max(local // vae_stride_t, 0), frame_embs.shape[0] - 1)
            pose_emb = frame_embs[lat_idx].float().cpu()  # [5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("oracle K 计算失败 fi=%d: %s；跳过该帧", fi, exc)
            continue

        # --- V：该帧的 visual_emb ---
        try:
            lat = latents_per_frame[fi].to(device)  # [z_dim,lat_h,lat_w]
            with torch.no_grad():
                visual_emb = model.get_projected_latent_emb(lat).float().cpu()  # [5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("oracle V 计算失败 fi=%d: %s；跳过该帧", fi, exc)
            continue

        key_list.append(pose_emb)
        val_list.append(visual_emb)

    if not key_list:
        return None
    key_states = torch.stack(key_list)   # [K,5120]
    value_states = torch.stack(val_list)  # [K,5120]
    return key_states, value_states


# 全局占位（main 中设置，供 _build_oracle_memory_kv 读取分辨率 / clip 长度）
_SIZE_HW: Tuple[str, str] = ("480", "832")
_ORACLE_CLIP_FRAMES: int = 81


# ---------------------------------------------------------------------------
# Bank populate + 检索（full / medium_long 档：让 bank 真正被消费，
# 复用 retrieval_probe._eval_episode 的逐帧 update + retrieve 口径）
# ---------------------------------------------------------------------------

def _semantic_key_for_frame(
    pose_emb: torch.Tensor,
    visual_emb: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """逐帧 semantic_key（口径与 retrieval_probe._eval_episode._semantic_key 完全一致）。

    semantic_key = alpha * normalize(pose_emb) + (1-alpha) * normalize(visual_emb)
    （visual_emb is None 时退化为 normalize(pose_emb)）。

    注：retrieval_probe 同样用 raw normalize（而非走 cross-attn K 投影），原因见
    retrieval_probe summary 偏差说明 1（K 投影层在 epoch_4 前零梯度，raw normalize 更稳）。
    """
    pk = F.normalize(pose_emb.float().unsqueeze(0), dim=-1).squeeze(0)
    if visual_emb is None:
        return pk
    vk = F.normalize(visual_emb.float().unsqueeze(0), dim=-1).squeeze(0)
    return alpha * pk + (1.0 - alpha) * vk


def _populate_bank(
    bank,
    ep: EpisodeData,
    query_clip_start: int,
    pose_embs: torch.Tensor,        # [T, 5120] CPU
    visual_embs: Optional[torch.Tensor],   # [T, 5120] CPU 或 None
    surprise: torch.Tensor,         # [T] CPU
    latents_per_frame: torch.Tensor,  # [T, z_dim, lat_h, lat_w] CPU
    abs_translations: torch.Tensor,  # [T, 3] CPU 绝对位置
    args,
) -> None:
    """对 query clip 之前的所有帧 [0, query_clip_start) 逐帧 bank.update（populate bank）。

    逐帧量计算方式与 retrieval_probe._eval_episode 完全一致：
      - pose_emb / visual_emb：传入的 [T,5120]（_compute_pose_embs_episode /
        _compute_visual_embs_from_latents 产出，与探针同源）
      - surprise：传入的 [T]
      - semantic_key：_semantic_key_for_frame（= alpha*norm(pose)+(1-alpha)*norm(visual)）
      - location：abs_translations[t]（= ep.poses[:, :3, 3]）
      - timestep：t；chunk_id：t // 21；increment_age：每 21 帧一次（clip 边界）

    Args:
        bank:              ThreeTierMemoryBank（_build_bank_for_config 构建）
        ep:                EpisodeData
        query_clip_start:  query clip 起始帧（全局索引）；只填 [0, query_clip_start)
        pose_embs/visual_embs/surprise/latents_per_frame/abs_translations: episode 全帧量
        args:              CLI args（提供 visual_fusion_alpha）
    """
    T = pose_embs.shape[0]
    end = max(0, min(query_clip_start, T))
    for t in range(end):
        try:
            frame_visual = visual_embs[t] if visual_embs is not None else None
            sk = _semantic_key_for_frame(
                pose_embs[t], frame_visual, args.visual_fusion_alpha
            )
            bank.update(
                pose_emb=pose_embs[t].float(),
                latent=latents_per_frame[t].float() if latents_per_frame is not None
                else torch.zeros(1),
                surprise_score=float(surprise[t].item()),
                timestep=int(t),
                visual_emb=frame_visual.float() if frame_visual is not None else None,
                chunk_id=int(t // 21),   # 21 latent 帧 ≈ 1 clip（与 retrieval_probe 一致）
                semantic_key=sk,
                location=abs_translations[t].float(),  # [3] 绝对位置（OP-2 Bug2 口径）
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("populate bank.update failed at ep=%s t=%d: %s",
                           ep.episode_id, t, exc)
            continue
        # age increment 按 clip 边界（每 21 frame），与 retrieval_probe._eval_episode 一致
        if t > 0 and (t % 21) == 0:
            bank.increment_age()


def _retrieve_bank_kv(
    bank,
    query_location: torch.Tensor,    # [3] CPU
    query_pose_emb: torch.Tensor,    # [5120] CPU
    query_semantic_key: torch.Tensor,  # [5120] CPU
    query_timestep: int,
    tier_config: str,
    model,
    latents_per_frame: torch.Tensor,  # [T, z_dim, lat_h, lat_w] CPU
    args,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """按 tier_config 从启用的 tier 检索 MemoryFrame，构造注入用 K/V + tier_ids。

    检索（口径与 retrieval_probe._eval_episode 同源）：
      - Long（所有非 oracle_only 档）：bank.long.retrieve_by_location(
            query_location, query_timestep, top_k=long_cap, min_gap_frames)
        min_gap_frames = max(1, round(fps * min_time_gap_sec))（与 GT / 探针同口径）
      - Medium（medium_long / full）：bank.medium.retrieve(query_pose_emb, top_k=medium_cap)
      - Short（仅 full）：bank.short.retrieve_all()[:short_cap]

    每帧 → K=get_projected_frame_embs（取该帧 pose_emb 已在 frame.pose_emb 中）/
    V=get_projected_latent_emb（用 latents_per_frame[frame.timestep] 重算，
    与 _build_oracle_memory_kv 同样的 V 构造方式）。

    tier_ids：Short=0 / Medium=1 / Long=2（按帧来源赋，Innovation 10 同口径）。

    Returns:
        (memory_states_kv, memory_value_kv, tier_ids)：
          memory_states_kv  [K,5120] CPU（K，pose_emb）
          memory_value_kv   [K,5120] CPU（V，visual_emb）
          tier_ids          [K] int64 CPU
        无任何检索结果时返回 None。
    """
    from memory_module.model_with_memory import WanModelWithMemory

    if not isinstance(model, WanModelWithMemory):
        logger.warning("model 非 WanModelWithMemory，无法构造 bank 检索 K/V")
        return None

    T = latents_per_frame.shape[0] if latents_per_frame is not None else 0

    # min_gap_frames：与 GT / retrieval_probe._retrieve_pose_abs_gap 同口径
    min_gap_frames = max(1, int(round(args.fps * args.min_time_gap_sec)))

    # 按 tier_config 收集 (MemoryFrame, tier_id)
    collected: List[Tuple["object", int]] = []

    # Long：所有非 oracle_only 档都启用（位置检索，F-14）
    long_frames = bank.retrieve_revisit(
        query_location=query_location,
        query_timestep=query_timestep,
        top_k=args.long_cap,
        min_gap_frames=min_gap_frames,
    )
    for f in long_frames:
        collected.append((f, 2))

    # Medium：medium_long / full 启用
    if tier_config in ("medium_long", "full"):
        medium_frames = bank.medium.retrieve(
            query_pose_emb.float(), top_k=args.medium_cap, device=None
        )
        for f in medium_frames:
            collected.append((f, 1))

    # Short：仅 full 启用
    if tier_config == "full":
        short_cap = bank.short.cap if bank.short.cap > 0 else 1
        short_frames = bank.short.retrieve_all(device=None)[:short_cap]
        for f in short_frames:
            collected.append((f, 0))

    if not collected:
        return None

    # 确保相关投影层在 device（offload 安全，与 _build_oracle_memory_kv 一致）
    if hasattr(model, "latent_proj"):
        model.latent_proj.to(device)

    key_list: List[torch.Tensor] = []
    val_list: List[torch.Tensor] = []
    tier_id_list: List[int] = []

    for frame, tier_id in collected:
        # --- K：该帧的 pose_emb（populate 时已存入 frame.pose_emb，与
        # _build_oracle_memory_kv 经 get_projected_frame_embs 算出的 pose_emb 同源）---
        try:
            pose_emb = frame.pose_emb.float().cpu()  # [5120]
        except Exception as exc:  # noqa: BLE001
            logger.warning("bank K 取用失败 t=%s: %s；跳过该帧",
                           getattr(frame, "timestep", "?"), exc)
            continue

        # --- V：该帧的 visual_emb（用 latents_per_frame[frame.timestep] 经
        # get_projected_latent_emb 重算，与 _build_oracle_memory_kv 同样构造）---
        fi = int(getattr(frame, "timestep", -1))
        if 0 <= fi < T and latents_per_frame is not None:
            try:
                lat = latents_per_frame[fi].to(device)  # [z_dim,lat_h,lat_w]
                with torch.no_grad():
                    visual_emb = model.get_projected_latent_emb(lat).float().cpu()  # [5120]
            except Exception as exc:  # noqa: BLE001
                logger.warning("bank V 计算失败 t=%d: %s；回退 frame.visual_emb", fi, exc)
                visual_emb = (frame.visual_emb.float().cpu()
                              if getattr(frame, "visual_emb", None) is not None
                              else pose_emb)
        else:
            # timestep 越界或无 latent：回退到 populate 时存入的 visual_emb（向后兼容）
            visual_emb = (frame.visual_emb.float().cpu()
                          if getattr(frame, "visual_emb", None) is not None
                          else pose_emb)

        key_list.append(pose_emb)
        val_list.append(visual_emb)
        tier_id_list.append(tier_id)

    if not key_list:
        return None

    key_states = torch.stack(key_list)    # [K,5120]
    value_states = torch.stack(val_list)  # [K,5120]
    tier_ids = torch.tensor(tier_id_list, dtype=torch.long)  # [K]
    return key_states, value_states, tier_ids


# ---------------------------------------------------------------------------
# 轻量一致性指标（D-06：第一轮用最简单可跑的）
# ---------------------------------------------------------------------------

def _to_gray_uint8(frame_chw: np.ndarray) -> np.ndarray:
    """[3,H,W] in [-1,1] → 灰度 [H,W] uint8。"""
    hwc = ((frame_chw.transpose(1, 2, 0) * 127.5 + 127.5)
           .clip(0, 255).astype(np.uint8))
    # RGB → 灰度（Rec.601）
    gray = (0.299 * hwc[..., 0] + 0.587 * hwc[..., 1]
            + 0.114 * hwc[..., 2])
    return gray.astype(np.float32)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """全局单窗口 SSIM（灰度，[H,W] float32）。

    简单可跑实现（D-06 已定：第一轮指标用现有/最简可跑指标，不追求完美；
    若定性模糊再补 Pose-Paired SSIM / DINO cosine / LPIPS。此处选 SSIM 是其中最简实现）。
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# DINO 语义指标（exp2_redesign_draft.md「Exp2 评测指标」节规格）
# ---------------------------------------------------------------------------

# DINO 模型 / device 全局单例缓存（只加载一次；加载失败缓存哨兵 False）。
# 不放模块顶层 import：off/oracle 等不需要 DINO 的路径不应被 torch.hub 加载拖累。
_DINO_MODEL = None          # 已加载的 dinov2_vits14 module，或 False（加载失败哨兵）
_DINO_DEVICE: Optional[torch.device] = None

# ImageNet 归一化常数（DINOv2 预训练统计量）
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# DINOv2 ViT-S/14 patch size = 14，输入须 resize 到 14 的整数倍
_DINO_PATCH = 14
_DINO_SIDE = 14 * 16        # 224，标准 DINOv2 输入边长（14 的整数倍）


def _get_dino_model(device: torch.device):
    """lazy-load 且全局缓存 dinov2_vits14（只加载一次，单例）。

    用 torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')，eval()，搬到 device。
    加载失败 graceful：缓存哨兵 False + warn，调用方据此跳过 dino 字段（不让 eval 崩）。

    Returns:
        已加载并 eval 的 module；加载失败返回 None。
    """
    global _DINO_MODEL, _DINO_DEVICE
    if _DINO_MODEL is False:
        return None              # 之前已失败，直接跳过（不重复尝试）
    if _DINO_MODEL is not None and _DINO_DEVICE == device:
        return _DINO_MODEL
    try:
        import torch.hub  # noqa: F401（确保 hub 可用）
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = model.to(device).eval()
        _DINO_MODEL = model
        _DINO_DEVICE = device
        logger.info("DINOv2 (dinov2_vits14) 已加载 → device=%s", device)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("DINOv2 加载失败（torch.hub.load）: %s；本轮跳过 dino 语义指标", exc)
        _DINO_MODEL = False
        return None


def _dino_feat(frame_chw: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    """单帧 → DINOv2 CLS/pooled 特征向量（exp2_redesign_draft.md 规格）。

    输入 [3,H,W] in [-1,1] → resize 到 14 的整数倍（224×224）→ ImageNet 归一化 →
    DINOv2 forward 取 pooled（CLS）特征。eval + no_grad，与主模型同 device。

    Args:
        frame_chw: [3,H,W] float in [-1,1]
        device:    目标设备（与主模型同 device）

    Returns:
        [feat_dim] float32 CPU 特征向量；DINO 不可用时返回 None。
    """
    model = _get_dino_model(device)
    if model is None:
        return None
    try:
        # [-1,1] → [0,1]
        x = (torch.from_numpy(np.ascontiguousarray(frame_chw)).float() + 1.0) / 2.0
        x = x.clamp(0.0, 1.0).unsqueeze(0).to(device)  # [1,3,H,W]
        # resize 到 14 的整数倍（224×224，双线性）
        x = F.interpolate(x, size=(_DINO_SIDE, _DINO_SIDE),
                          mode="bilinear", align_corners=False)
        # ImageNet 归一化
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            # DINOv2 默认 forward 返回 CLS/pooled 特征 [B, feat_dim]
            feat = model(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return feat.squeeze(0).float().cpu()  # [feat_dim]
    except Exception as exc:  # noqa: BLE001
        logger.warning("DINO 特征计算失败: %s；跳过该帧 dino", exc)
        return None


def _revisit_consistency(
    gen_video: np.ndarray,        # [3, F, H, W] in [-1,1]（生成的 query clip）
    gt_first_visit_frame: np.ndarray,  # [3, H, W] in [-1,1]（首访 GT 帧）
    device: Optional[torch.device] = None,  # DINO 计算用 device（None 则按 CUDA 可用回退）
) -> Dict[str, float]:
    """计算生成的重访帧 vs 首访 GT 帧的相似度（像素 SSIM + 语义 DINO，一遍过帧）。

    revisit_consistency = max_t SSIM(gen_video[:,t], gt_first_visit_frame)
      —— 取整段 query clip 里与首访帧最相似的一帧（重访发生的精确帧不确定，取 max）。
    同时返回 mean / last 供诊断。

    语义 DINO（exp2_redesign_draft.md「Exp2 评测指标」节）：在同一帧循环里追加
    revisit_consistency_dino_{max,mean,last} = gen 帧与 GT 首访帧在 DINOv2 特征空间的
    cosine 相似度（GT 首访帧特征只算一次）。聚合方式（max/mean/last）与 SSIM 一致。
    DINO 不可用（加载失败/特征算不出）时**不写** dino 字段（graceful，不让 eval 崩）。
    """
    F_ = gen_video.shape[1]
    gt_gray = _to_gray_uint8(gt_first_visit_frame)  # [H_gt,W_gt] float32 [0,255]

    # P0 修复：WanI2V.generate() 内部会把请求的 H（如 480）调整为 patch/VAE 整除的
    # 实际输出高度（如 464），而 GT 首访帧按 --size（480×832）加载，两者 H 不一致会让
    # _ssim 逐元素相乘 broadcast 失败。这里以 gen 输出为准（它才是模型真实输出分辨率），
    # 把 GT 灰度帧 resize 到 gen 帧的 (H_gen,W_gen)。gen 各帧 H,W 相同，故循环外只 resize 一次。
    if F_ > 0:
        h_gen, w_gen = gen_video.shape[2], gen_video.shape[3]
        if gt_gray.shape != (h_gen, w_gen):
            # 灰度是 float32 [0,255]：用 mode='F'（32-bit float）Image 双线性 resize，
            # 避免 uint8 量化损失。PIL resize 的 size 参数顺序为 (W,H)。
            gt_img = Image.fromarray(gt_gray, mode="F")
            gt_img = gt_img.resize((w_gen, h_gen), resample=Image.BILINEAR)
            gt_gray = np.asarray(gt_img, dtype=np.float32)

    # DINO device：默认与主模型同 device（CUDA 可用则用 cuda:0，否则 CPU）
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # GT 首访帧 DINO 特征只算一次（DINO 自带 resize，无需先对齐分辨率）
    gt_dino = _dino_feat(gt_first_visit_frame, device) if F_ > 0 else None

    ssims: List[float] = []
    dino_cos: List[float] = []
    for t in range(F_):
        ssims.append(_ssim(_to_gray_uint8(gen_video[:, t]), gt_gray))
        if gt_dino is not None:
            g = _dino_feat(gen_video[:, t], device)
            if g is not None:
                cos = F.cosine_similarity(g.unsqueeze(0), gt_dino.unsqueeze(0), dim=-1)
                dino_cos.append(float(cos.item()))

    ssims_np = np.asarray(ssims, dtype=np.float32)
    out: Dict[str, float] = {
        "revisit_consistency_max": float(ssims_np.max()) if F_ > 0 else 0.0,
        "revisit_consistency_mean": float(ssims_np.mean()) if F_ > 0 else 0.0,
        "revisit_consistency_last": float(ssims_np[-1]) if F_ > 0 else 0.0,
    }
    # DINO 字段：仅当成功算出（与 SSIM 同口径 max/mean/last）才追加；否则 graceful 跳过
    if dino_cos:
        dino_np = np.asarray(dino_cos, dtype=np.float32)
        out["revisit_consistency_dino_max"] = float(dino_np.max())
        out["revisit_consistency_dino_mean"] = float(dino_np.mean())
        out["revisit_consistency_dino_last"] = float(dino_np[-1])
    return out


def _record_point(
    all_records: List[Dict],
    args,
    ep_id: str,
    ep_out_dir: str,
    point: "RevisitPoint",
    video: np.ndarray,
    gt_first: np.ndarray,
    n_oracle_frames: int,
    mp4_path: str,
    device: Optional[torch.device] = None,
) -> None:
    """算指标 + 构造 record + append all_records + 增量写 per_window.csv（P1）。

    生成路径与 mp4 读回路径（P2）共用本函数，保证两条路径的 record 字段一致。
    n_oracle_frames=-1 表示该点来自 mp4 读回（未重算 oracle 帧），仅作标记。
    device：DINO 语义指标计算用 device（与主模型同 device）。
    """
    metrics = _revisit_consistency(video, gt_first, device=device)
    logger.info("ep=%s q=%d [%s/%s] %s",
                ep_id, point.query_frame, args.tier_config,
                args.memory_mode, metrics)
    record = {
        "episode_id": ep_id,
        "query_frame": point.query_frame,
        "first_visit_frame": point.first_visit_frame,
        "tier_config": args.tier_config,
        "memory_mode": args.memory_mode,
        "inject_tier": args.inject_tier,
        "weaken_first_frame": args.weaken_first_frame,
        "n_oracle_frames": n_oracle_frames,
        "video_path": mp4_path,
        "gt_first_visit_png": os.path.join(
            ep_out_dir, f"q{point.query_frame}_gt_first_visit.png"),
        # DINO 字段先占位 None（保证 per_window.csv schema 跨行稳定：DINO 不可用的点
        # 也有相同列），下方 **metrics 用实际算出的值覆盖（算出才有 dino_* key）。
        "revisit_consistency_dino_max": None,
        "revisit_consistency_dino_mean": None,
        "revisit_consistency_dino_last": None,
        **metrics,
    }
    # 结果可溯源：把本次 --gate_override 取值写进 record（→ per_window.csv 末列）。
    # None（未覆写，用 checkpoint 训出来的 gate）写 "trained"，便于事后区分不同 run。
    # 追加到 record 末尾，不破坏既有列顺序（**metrics 之后）。
    record["gate_override"] = (
        "trained" if args.gate_override is None else args.gate_override
    )
    all_records.append(record)
    # P1：逐点增量落盘，中途崩溃已完成点不丢失
    _append_per_window_csv(args.output_dir, record)


# ---------------------------------------------------------------------------
# 单个重访点的一次生成（给定 tier_config × memory_mode）
# ---------------------------------------------------------------------------

def _generate_for_point(
    wan_i2v,
    bank_kv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    oracle_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    wrong_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    random_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    point: RevisitPoint,
    ep: EpisodeData,
    base_img: Image.Image,
    args,
    device: torch.device,
    rng: np.random.Generator,
    tmp_action_dir: str,
) -> Optional[np.ndarray]:
    """对一个重访点跑一次 diffusion 生成，返回生成视频 [3,F,H,W]。

    流程：
      a. 弱化首帧条件（按 --weaken_first_frame）
      b. 配置注入 K/V（四臂统一语义）：
         - memory_mode=off：memory_states=None（不注入，所有档 baseline）
         - memory_mode=wrong：注入 wrong_kv（错误帧控制组，所有档通用）
         - memory_mode=random：注入 random_kv（同 episode 历史帧随机抽，confound 对照）
         - memory_mode=oracle：注入"该档配置的记忆源"——
             · tier_config=oracle_only：注入 oracle GT 帧（oracle_kv，绕过 bank，保持现状）
             · tier_config=full / medium_long：注入 bank_kv（bank 检索帧 + tier_ids），
               由外层 _retrieve_bank_kv 按启用 tier 检索构造（full=Short+Medium+Long，
               medium_long=Medium+Long）
      c. 跑生成（复用 infer_v4 的 _patch_pipeline_memory + wan_i2v.generate）

    tier label（D-06 配套）：oracle / random / wrong 三臂注入帧的 tier_ids 统一用
    --inject_tier 指定值（不再硬编码 long=2，修「Long tier embedding 从未训练」坑）。
    bank 检索路径（full/medium_long）的 tier_ids 仍按检索帧真实来源标，不受此参数影响。
    """
    from memory_module.model_with_memory import WanModelWithMemory
    from wan.configs import MAX_AREA_CONFIGS

    # --- 首帧图像：base_img = query clip 首帧 GT 图（外层从解码帧构造并传入），按需弱化 ---
    # query clip 起始帧 = 以 query_frame 对齐的 clip 段起点
    poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(
        ep, point.query_frame, args.frame_num
    )
    img = _weaken_image(base_img, args.weaken_first_frame, rng)

    # --- 写 query clip 的 action 切片到临时目录（供 generate action_path 读取）---
    np.save(os.path.join(tmp_action_dir, "poses.npy"), poses_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "action.npy"), acts_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "intrinsics.npy"), intr_c.astype(np.float32))

    # --- 选择注入 K/V（四臂统一语义，见 docstring b）---
    memory_states = None
    memory_value_states = None
    tier_ids = None

    # 注入帧 tier label：oracle / random / wrong 三臂统一用 --inject_tier（D-06 配套，
    # 替代旧版硬编码 long=2）。修「Long tier embedding 从未被训练（Stage 2 0l/32）」坑。
    inject_tier_id = _TIER_NAME_TO_ID[args.inject_tier]

    if args.memory_mode == "off":
        # 所有档：不注入（baseline）
        memory_states = None
    elif args.memory_mode == "wrong":
        # 所有档：注入 wrong_kv（错误帧控制组），tier_ids 标 --inject_tier
        if wrong_kv is not None:
            key_states, value_states = wrong_kv
            memory_states = key_states.unsqueeze(0).to(device)         # [1,K,5120]
            memory_value_states = value_states.unsqueeze(0).to(device)  # [1,K,5120]
            tier_ids = torch.full(
                (key_states.shape[0],), inject_tier_id, dtype=torch.long, device=device
            )
    elif args.memory_mode == "random":
        # 所有档：注入 random_kv（同 episode 历史帧随机抽，confound 对照），tier_ids 标 --inject_tier
        if random_kv is not None:
            key_states, value_states = random_kv
            memory_states = key_states.unsqueeze(0).to(device)         # [1,K,5120]
            memory_value_states = value_states.unsqueeze(0).to(device)  # [1,K,5120]
            tier_ids = torch.full(
                (key_states.shape[0],), inject_tier_id, dtype=torch.long, device=device
            )
    else:  # memory_mode == "oracle"：注入"该档配置的记忆源"
        if args.tier_config == "oracle_only":
            # oracle_only：注入 oracle GT 帧（绕过 bank，保持现状不动），tier_ids 标 --inject_tier
            if oracle_kv is not None:
                key_states, value_states = oracle_kv
                memory_states = key_states.unsqueeze(0).to(device)         # [1,K,5120]
                memory_value_states = value_states.unsqueeze(0).to(device)  # [1,K,5120]
                tier_ids = torch.full(
                    (key_states.shape[0],), inject_tier_id, dtype=torch.long, device=device
                )
        else:
            # full / medium_long：注入 bank 检索帧（含来源 tier_ids）
            if bank_kv is not None:
                key_states, value_states, bank_tier_ids = bank_kv
                memory_states = key_states.unsqueeze(0).to(device)         # [1,K,5120]
                memory_value_states = value_states.unsqueeze(0).to(device)  # [1,K,5120]
                tier_ids = bank_tier_ids.to(device)                        # [K]，Short0/Medium1/Long2

    max_area = MAX_AREA_CONFIGS[args.size]

    if memory_states is not None:
        _patch_pipeline_memory(wan_i2v, memory_states, memory_value_states, tier_ids=tier_ids)
    try:
        video = wan_i2v.generate(
            args.prompt,
            img,
            action_path=tmp_action_dir,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.num_inference_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            offload_model=True,
        )
    finally:
        if memory_states is not None:
            _unpatch_pipeline_memory(wan_i2v)

    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    return video  # [3,F,H,W]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    if args.weaken_first_frame == "noise":
        logger.warning(
            "⚠️ F-18: --weaken_first_frame=noise 会用随机 RGB 替换首帧、摧毁 i2v "
            "场景锚点 → 三臂都生成噪点、revisit 指标全部地板化、oracle/wrong 无法对比。"
            "revisit 一致性评测请用 zero(中性灰温和锚点);noise 仅作消融。")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    global _SIZE_HW, _ORACLE_CLIP_FRAMES
    _SIZE_HW = tuple(args.size.split("*"))  # ("480","832")
    _ORACLE_CLIP_FRAMES = args.frame_num

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))

    logger.info("Args: %s", vars(args))
    logger.warning(
        "⚠️ Exp2 有效性前提：memory_cross_attn 须已训练（OP-1 修复 + 重训后的 "
        "--ft_model_dir）。epoch_4 随机 memory 上结果仅作负对照。"
    )

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

    # ---- 加载 WanI2V pipeline（复用 infer_v4，转换为 WanModelWithMemory）----
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS

    # 全参微调模式：准备 tmp ckpt_dir（base 权重，ft 权重由 _convert 注入）
    ckpt_dir = args.ckpt_dir
    if args.ft_model_dir:
        _fake = argparse.Namespace(ckpt_dir=args.ckpt_dir)
        ckpt_dir = _load_ft_model_and_prepare_ckpt(_fake)

    cfg = WAN_CONFIGS["i2v-A14B"]
    local_rank = device.index if device.type == "cuda" and device.index is not None else 0
    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=local_rank,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
    )
    logger.info("转换 pipeline → WanModelWithMemory ...")
    wan_i2v = _convert_pipeline_to_memory(
        wan_i2v,
        memory_ckpt_path=None,
        high_model_dir=args.ft_high_model_dir,
        low_model_dir=args.ft_model_dir,
    )
    model = wan_i2v.low_noise_model

    # ---- gate 强制覆写（诊断用，default=None 时零改动跳过）----
    # 必须在进入推理循环之前做：覆写所有 memory cross-attn 的有效 gate（post-tanh 乘子）。
    # 与训练真实行为解耦 —— 仅用于判别"接口是否有用"，不改 out_rms_cap 幅度硬上界。
    if args.gate_override is not None:
        _apply_gate_override(model, args.gate_override)

    # ---- 逐 episode → 找重访点 → 三注入对照生成 ----
    import tempfile

    all_records: List[Dict] = []

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

        # 解码 video + VAE encode（oracle V + 首帧图像 + GT 一致性参照都要用）
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W] in [-1,1]
            latents_full = _vae_encode_batched(wan_i2v.vae, frames, device=device,
                                               batch_frames=8)
            latents_per_frame = _expand_latents_to_frames(latents_full, T)
            del latents_full
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
            continue

        # full / medium_long + oracle 档：预计算 bank populate 所需逐帧量
        # （pose_emb / visual_emb / surprise / 绝对位置），口径与 retrieval_probe 同源。
        # oracle_only 档及 off/wrong 模式不需要 bank → 不计算（保持现状路径不动）。
        need_bank = (args.tier_config != "oracle_only" and args.memory_mode == "oracle")
        ep_pose_embs = None
        ep_visual_embs = None
        ep_surprise = None
        ep_abs_translations = None
        if need_bank:
            try:
                ep_pose_embs = _compute_pose_embs_episode(
                    ep, model, device, height=height, width=width, fps=args.fps,
                )  # [T,5120]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Episode %s pose_embs 计算失败: %s；跳过 bank populate",
                               ep_id, exc)
                need_bank = False
            if need_bank:
                try:
                    ep_visual_embs = _compute_visual_embs_from_latents(
                        latents_per_frame, model, device,
                    )  # [T,5120]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Episode %s visual_embs 计算失败: %s；continue with None",
                                   ep_id, exc)
                    ep_visual_embs = None
                # surprise：用 visual_cosine（自包含，无 NFP 依赖；与 retrieval_probe fallback 同函数）
                if ep_visual_embs is not None:
                    ep_surprise = _compute_surprise_visual_cosine(ep_visual_embs)
                else:
                    ep_surprise = torch.zeros(T, dtype=torch.float32)
                # 绝对世界位置（c2w 平移），与 retrieval_probe._eval_episode 同口径
                ep_abs_translations = torch.from_numpy(ep.poses[:, :3, 3]).float()  # [T,3]

        ep_out_dir = os.path.join(args.output_dir, ep_id)
        os.makedirs(ep_out_dir, exist_ok=True)

        for pt in points:
            # P1 抗崩：单个重访点整段（生成→保存→算指标→append record）包 try/except，
            # 单点失败只 logger.exception + continue，不让整轮（可能 8h）崩掉。
            try:
                # 首访 GT 帧（一致性参照 + 人工定性对比）
                gt_first = frames[pt.first_visit_frame]  # [3,H,W]
                # 保存首访 GT 帧供人工对比
                _save_frame_png(gt_first,
                                os.path.join(ep_out_dir,
                                             f"q{pt.query_frame}_gt_first_visit.png"))

                mp4_name = f"q{pt.query_frame}_{args.tier_config}_{args.memory_mode}.mp4"
                mp4_path = os.path.join(ep_out_dir, mp4_name)

                # P2 可续/止损：该点目标 mp4 已存在 → 不重跑 generate（省 ~49min），
                # 从 mp4 读回视频帧照常算指标。此分支不重算 oracle_indices
                # （record 里 n_oracle_frames 标 -1 表示"来自读回，未重算"）。
                oracle_indices: List[int] = []
                if os.path.exists(mp4_path):
                    video = _read_video_back(mp4_path)
                    if video is not None:
                        logger.info(
                            "ep=%s q=%d [%s/%s]：mp4 已存在 → 读回重算指标（跳过生成）",
                            ep_id, pt.query_frame, args.tier_config, args.memory_mode)
                        _record_point(
                            all_records, args, ep_id, ep_out_dir, pt, video,
                            gt_first, n_oracle_frames=-1, mp4_path=mp4_path,
                            device=device)
                        continue
                    logger.warning(
                        "ep=%s q=%d：mp4 存在但读回失败 → 重新生成", ep_id, pt.query_frame)

                # query clip 首帧 GT 图像（弱化前的 base，作为 generate 的 img 入口）
                _pc, _ac, _ic, seg_start = _frame_to_clip_slice(
                    ep, pt.query_frame, args.frame_num)
                query_first_np = frames[seg_start]  # [3,H,W]
                base_img = _frame_to_pil(query_first_np)

                # 构造 oracle K/V（绕过 bank）：注入首访点附近的 GT 历史帧
                oracle_indices = _pick_oracle_indices(pt, args.num_oracle_frames, T)
                oracle_kv = _build_oracle_memory_kv(
                    model, wan_i2v, ep, latents_per_frame, oracle_indices, device)
                # wrong K/V：同 episode 远距离位置的 GT 帧
                wrong_indices = _pick_wrong_indices(pt, args.num_oracle_frames, T, rng)
                wrong_kv = _build_oracle_memory_kv(
                    model, wan_i2v, ep, latents_per_frame, wrong_indices, device)
                # random K/V：同 episode 历史帧随机抽（confound 对照），与 oracle 同候选池
                random_indices = _pick_random_indices(
                    pt, args.num_oracle_frames, T, rng)
                random_kv = _build_oracle_memory_kv(
                    model, wan_i2v, ep, latents_per_frame, random_indices, device)

                # full / medium_long + oracle 档：构建 bank → populate [0, query_clip_start)
                # → 从启用的 tier 检索 → 构造注入 K/V（bank_kv）。oracle_only / off / wrong 路径不走此处。
                bank_kv = None
                if need_bank and ep_pose_embs is not None:
                    bank = _build_bank_for_config(args)
                    # query clip 起始帧（与 _generate_for_point 内 _frame_to_clip_slice 对齐）
                    _qpc, _qac, _qic, q_clip_start = _frame_to_clip_slice(
                        ep, pt.query_frame, args.frame_num)
                    _populate_bank(
                        bank, ep, q_clip_start,
                        ep_pose_embs, ep_visual_embs, ep_surprise,
                        latents_per_frame, ep_abs_translations, args,
                    )
                    # query 侧逐帧量（location / pose_emb / semantic_key），口径同 populate
                    q_idx = min(max(pt.query_frame, 0), T - 1)
                    q_visual = ep_visual_embs[q_idx] if ep_visual_embs is not None else None
                    q_semantic_key = _semantic_key_for_frame(
                        ep_pose_embs[q_idx], q_visual, args.visual_fusion_alpha)
                    bank_kv = _retrieve_bank_kv(
                        bank,
                        query_location=ep_abs_translations[q_idx],
                        query_pose_emb=ep_pose_embs[q_idx],
                        query_semantic_key=q_semantic_key,
                        query_timestep=int(pt.query_frame),
                        tier_config=args.tier_config,
                        model=model,
                        latents_per_frame=latents_per_frame,
                        args=args,
                        device=device,
                    )
                    if bank_kv is None:
                        logger.warning(
                            "ep=%s q=%d [%s]：bank 检索为空（无满足条件的记忆帧）→ "
                            "本次 oracle 注入退化为不注入",
                            ep_id, pt.query_frame, args.tier_config)

                _tmp_action = tempfile.mkdtemp(
                    prefix=f"oracle_inj_{ep_id}_q{pt.query_frame}_")
                try:
                    video = _generate_for_point(
                        wan_i2v, bank_kv, oracle_kv, wrong_kv, random_kv, pt, ep,
                        base_img, args, device, rng, _tmp_action)
                finally:
                    import shutil
                    shutil.rmtree(_tmp_action, ignore_errors=True)

                if video is None:
                    logger.warning("ep=%s q=%d：生成返回 None，跳过该点",
                                   ep_id, pt.query_frame)
                    continue

                # 保存生成视频
                _save_video(video, mp4_path, fps=args.fps)

                _record_point(
                    all_records, args, ep_id, ep_out_dir, pt, video,
                    gt_first, n_oracle_frames=len(oracle_indices), mp4_path=mp4_path,
                    device=device)
            except Exception as exc:  # noqa: BLE001
                # P1：单点任何环节失败 → 记录并继续下一个点，不中断整轮
                logger.exception("重访点处理失败 ep=%s q=%d: %s",
                                 ep_id, pt.query_frame, exc)
                continue

        del frames, latents_per_frame
        if ep_pose_embs is not None:
            del ep_pose_embs
        if ep_visual_embs is not None:
            del ep_visual_embs
        if ep_abs_translations is not None:
            del ep_abs_translations
        torch.cuda.empty_cache()

    _write_summary(args, all_records)
    logger.info("Done. 输出目录: %s", args.output_dir)


# ---------------------------------------------------------------------------
# oracle / wrong 帧选择
# ---------------------------------------------------------------------------

def _pick_oracle_indices(pt: RevisitPoint, n: int, T: int) -> List[int]:
    """从 GT 过去帧里取 n 个作为 oracle 注入帧（优先首访帧及其邻近）。"""
    cand = sorted(set(pt.gt_past_frames))
    if not cand:
        return []
    # 取最早 n 个（首访点附近，最贴"该地点首次访问的真实历史帧"语义）
    return cand[:max(1, n)]


# oracle 注入帧的近邻排除半径（帧）。random 臂从「同 oracle 的候选池」里随机选，
# 必须排除 oracle 帧本身 + 其近邻（与 oracle 同一候选语义、只是随机抽），
# 避免 random 抽到几乎就是 oracle 的帧、稀释「oracle ≫ random」的对照力度。
_ORACLE_NEIGHBOR_RADIUS: int = 2


def _pick_random_indices(pt: RevisitPoint, n: int, T: int,
                         rng: np.random.Generator) -> List[int]:
    """从**同一 episode 的可用历史帧**里随机抽 n 帧作为 random-in-bank 注入帧
    （memory_mode=random，confound 对照：判据 oracle ≫ {random ≈ off}）。

    候选池与 oracle 同源（query 之前的历史帧 [0, query_frame)），但：
      - 排除 oracle 注入帧本身（_pick_oracle_indices 选出的帧）
      - 排除 oracle 帧的近邻（±_ORACLE_NEIGHBOR_RADIUS，复用 oracle 的近邻排除语义），
        否则 random 抽到的帧可能几乎就是 oracle，对照失去意义
    在剩余历史帧里用传入的 rng（固定 seed）随机抽，保证可复现。
    若历史帧不足/排除后为空，回退到不带近邻排除的历史池。

    签名/返回结构对齐 _pick_wrong_indices（返回升序 List[int]）；注入路径完全复用
    oracle/wrong 那条（仅换选帧）。
    """
    oracle_idx = _pick_oracle_indices(pt, n, T)
    # oracle 帧 + 其近邻构成排除区
    forbidden = set()
    for fi in oracle_idx:
        for d in range(-_ORACLE_NEIGHBOR_RADIUS, _ORACLE_NEIGHBOR_RADIUS + 1):
            forbidden.add(fi + d)
    # 候选池 = query 之前的历史帧（与 oracle「可用历史」同口径）
    hist_end = min(max(int(pt.query_frame), 0), T)
    pool = [i for i in range(hist_end) if i not in forbidden]
    if not pool:
        # 排除后为空：退到全历史池（仅排除 oracle 帧本身），保证 random 臂仍能注入
        pool = [i for i in range(hist_end) if i not in set(oracle_idx)]
    if not pool:
        return []
    rng.shuffle(pool)
    return sorted(pool[:max(1, n)])


def _pick_wrong_indices(pt: RevisitPoint, n: int, T: int,
                        rng: np.random.Generator) -> List[int]:
    """取同 episode 远离 query / 首访点的随机帧作为 wrong-memory 注入帧。"""
    forbidden = set(pt.gt_past_frames) | {pt.query_frame}
    # 远离 query_frame：要求 |i - query_frame| > T//4，避免误命中真实重访
    margin = max(1, T // 4)
    pool = [i for i in range(T)
            if i not in forbidden and abs(i - pt.query_frame) > margin
            and abs(i - pt.first_visit_frame) > margin]
    if not pool:
        pool = [i for i in range(T) if i not in forbidden]
    if not pool:
        return []
    rng.shuffle(pool)
    return sorted(pool[:max(1, n)])


def _build_bank_for_config(args):
    """按 tier_config 构建 ThreeTierMemoryBank。

    - full        : Short/Medium/Long 全保留，检索时三层都取
    - medium_long : Short 仍 populate（cap=1）但**检索时不取 Short**（_retrieve_bank_kv
                    仅在 tier_config=="full" 时读 Short），等价"关 Short"
    （oracle_only 不构建 bank，main 中传 None）

    注：Short cap 固定为 1（不用 cap=0）——ShortTermBank(cap=0).update() 会在空队列上
    pop(0) 抛 IndexError，且 ThreeTierMemoryBank.update 先调 short.update，会连带阻断
    Medium/Long 写入。"关 Short" 改由检索侧 tier 选择实现，更安全且不动 memory_bank.py。
    """
    from memory_module.memory_bank import ThreeTierMemoryBank
    short_cap = 1
    return ThreeTierMemoryBank(
        short_cap=short_cap,
        medium_cap=args.medium_cap,
        long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life,
        dup_threshold=args.dup_threshold,
    )


# ---------------------------------------------------------------------------
# 图像 / 视频 IO
# ---------------------------------------------------------------------------

def _frame_to_pil(frame_chw: np.ndarray) -> Image.Image:
    """[3,H,W] in [-1,1] → PIL RGB。"""
    hwc = ((frame_chw.transpose(1, 2, 0) * 127.5 + 127.5)
           .clip(0, 255).astype(np.uint8))
    return Image.fromarray(hwc, mode="RGB")


def _save_frame_png(frame_chw: np.ndarray, path: str) -> None:
    _frame_to_pil(frame_chw).save(path)


def _read_video_back(path: str) -> Optional[np.ndarray]:
    """读回已存在的 mp4 为 [3,F,H,W] float32 in [-1,1]（与生成路径值域一致）。

    P2 可续/止损：该重访点 mp4 已存在时不重跑 generate（省 ~49min），改为从 mp4 读回
    视频帧、照常算指标。用 cv2 顺序解码（cv2 已在依赖中，与 retrieval_probe._decode_episode_video
    同栈）。⚠️ mp4 经 libx264 有损压缩，读回值与原始生成张量略有差异；本读回仅用于一致性
    相对比较（oracle/off/wrong 同等受压缩影响），可接受此近似。

    Returns:
        [3,F,H,W] float32 in [-1,1]；读取失败或无帧时返回 None。
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.warning("读回 mp4 失败（无法打开）：%s", path)
        return None
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # 归一到 [-1,1]，与生成路径（_save_video value_range=(-1,1)）值域一致
            arr = rgb.astype(np.float32) / 127.5 - 1.0
            frames.append(arr.transpose(2, 0, 1))  # [3,H,W]
    finally:
        cap.release()
    if not frames:
        logger.warning("读回 mp4 失败（无有效帧）：%s", path)
        return None
    return np.stack(frames, axis=1)  # [3,F,H,W]


def _save_video(video: np.ndarray, path: str, fps: int) -> None:
    """保存 [3,F,H,W] in [-1,1] 视频，复用 wan.utils.save_video。"""
    from wan.utils.utils import save_video
    t = torch.from_numpy(video) if isinstance(video, np.ndarray) else video
    save_video(
        tensor=t[None],   # [1,3,F,H,W]
        save_file=path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info("保存视频 → %s", path)


# ---------------------------------------------------------------------------
# Summary 输出
# ---------------------------------------------------------------------------

def _append_per_window_csv(output_dir: str, record: Dict) -> None:
    """逐点增量写 per_window.csv（P1 抗崩：算完一个点立即 append 一行）。

    这样长跑（可能 8h）中途崩溃时，已完成点的指标不丢失。首次写入 header，
    之后续写。字段顺序以 record 的 key 为准（与 all_records 的 dict 一致）。
    """
    csv_path = os.path.join(output_dir, "per_window.csv")
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as exc:  # noqa: BLE001
        # 增量落盘失败不应影响主流程（指标已在内存 all_records 中，summary 仍会写）
        logger.warning("写 per_window.csv 失败: %s", exc)


def _write_summary(args, records: List[Dict]) -> None:
    """输出 summary.md（一致性数值表 + 视频路径）+ summary.json。"""
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "records": records,
        }, fh, indent=2)

    md = []
    md.append("# Exp2 Oracle Injection Summary\n\n")
    md.append(f"- timestamp: {datetime.now().isoformat()}\n")
    md.append(f"- tier_config: **{args.tier_config}** | "
              f"memory_mode: **{args.memory_mode}** | "
              f"inject_tier: **{args.inject_tier}** | "
              f"weaken_first_frame: **{args.weaken_first_frame}**\n")
    # 结果可溯源：记录本次 gate_override 取值（None=用 checkpoint 训出来的 gate）。
    _gate_ov = "trained (None, 用 checkpoint 训出来的 gate)" if args.gate_override is None \
        else f"**{args.gate_override}** (诊断模式：强制覆写有效 gate，与训练真实行为解耦)"
    md.append(f"- gate_override: {_gate_ov}\n")
    md.append(f"- ft_model_dir: {args.ft_model_dir}\n\n")
    md.append("> ⚠️ 有效性前提：memory_cross_attn 须已训练（OP-1 修复 + 重训）。"
              "epoch_4 随机 memory 上仅作负对照。\n\n")
    md.append("> 单次 run 跑一种 (tier_config × memory_mode)。完整对照需多次 run "
              "（不同 --tier_config / --memory_mode），事后对比各 summary。\n\n")
    md.append("> 判据（exp2_redesign_draft）：oracle ≫ {random ≈ off}；DINO 为主判据，"
              "SSIM 为辅，两者背离以 DINO 为准 + 人工复核。dino 列为空 = DINO 未加载成功。\n\n")
    md.append("## 一致性数值（SSIM rc_* + DINO cosine rc_dino_*，均 = agg_t(gen[:,t] vs 首访GT帧)）\n\n")
    md.append("| episode | query_frame | tier | mode | "
              "rc_max | rc_mean | rc_last | "
              "rc_dino_max | rc_dino_mean | rc_dino_last | video |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|\n")

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

    for r in records:
        md.append(
            f"| {r['episode_id']} | {r['query_frame']} | {r['tier_config']} | "
            f"{r['memory_mode']} | {r['revisit_consistency_max']:.4f} | "
            f"{r['revisit_consistency_mean']:.4f} | "
            f"{r['revisit_consistency_last']:.4f} | "
            f"{_fmt(r.get('revisit_consistency_dino_max'))} | "
            f"{_fmt(r.get('revisit_consistency_dino_mean'))} | "
            f"{_fmt(r.get('revisit_consistency_dino_last'))} | "
            f"`{os.path.basename(r['video_path'])}` |\n"
        )
    md.append("\n## 人工定性对比\n\n")
    md.append("每个重访点已保存：\n")
    md.append("- `q<frame>_<tier>_<mode>.mp4`：生成的 query clip\n")
    md.append("- `q<frame>_gt_first_visit.png`：该地点首访 GT 帧（参照）\n\n")
    md.append("判读：oracle 是否优于 off / random（rc 更高）？wrong 是否变差"
              "（rc 更低 / 接近 off）？random 是否 ≈ off（confound 排除）？\n")

    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as fh:
        fh.writelines(md)
    logger.info("写 summary: %s", md_path)


if __name__ == "__main__":
    main()
