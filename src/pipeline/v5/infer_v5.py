"""
infer_v5.py — v5 in-context KV 记忆的多 clip 长视频自然画质生成（experiment_design Step 41 / S-V2 第三块）
============================================================================================================

**目的**：给人眼看 memory 在长程一致性上有没有用的 demo。
与 eval_v5.py 的区别：**不做 weaken=zero、不算 DINO、不三臂**；就是真实 bank
检索 → in-context KV 注入 → 多 clip 连续生成 → 存一条长视频。

数据格式（严格对齐 v4 infer_v4.py）：
  - --image           首帧图像（PNG/JPG）
  - --action_path     目录，含 poses.npy[T,4,4] / action.npy / intrinsics.npy
                      （整条 GT 动作轨迹，是控制信号；模型据此渲染视频）
  - --num_clips       默认 12
  - --frame_num       默认 81
  - --save_file       输出 mp4
  - 模型参数：--ckpt_dir / --memory_encoder_ckpt（必填）/ --grid /
    --encoder_depth / --memory_layers（三者默认走 _resolve_model_config 从
    training_metadata.json 采纳）/ --inject_high（默认 False，W2 low-only）
  - bank 参数默认值对齐 train_v5

核心流程（主循环，**复用 eval_v5 而非重写**）：
  1. 模型装载：直接 `from pipeline.v5.eval_v5 import _load_v5_pipeline`，拿到
     wan_i2v（low_noise_model 已是 WanModelWithMemoryV5 + memory_encoder.pth
     已载，W1/W1b 断言已过）。
  2. bank = ThreeTierMemoryBank(...)（from memory_module.memory_bank import）。
  3. 多 clip 循环 for clip_idx in range(num_clips):
       a. 切本 clip 的 poses/actions/intrinsics（帧区间 [c*frame_num:(c+1)*frame_num]）。
       b. 检索记忆：仅当 bank.size()>0（即 clip_idx>0）时，query_location = 本 clip
          第一个 latent 帧的绝对位置；调用复刻的 _retrieve_memory_latents →
          memory_latents [K,16,h,w] 或 None。clip_idx==0 时 memory_latents=None
          （纯 i2v 起步）。
       c. 生成：本 clip 的 poses/actions/intrinsics 存到临时 action 目录 →
          _patch_memory_latents(wan_i2v, memory_latents) →
          wan_i2v.generate(...) → _unpatch_memory_latents(wan_i2v)（try/finally
          保证还原）。img：clip_idx==0 用 --image；之后用上一 clip 生成视频的
          最后一帧（autoregressive 链接，对齐 v4 infer 的 current_img）。
       d. bank 更新（surprise-independent，对齐 train_v5 context-clip 循环）：
          VAE encode 本 clip 生成视频 → latents [16,lat_f,lat_h,lat_w]；
          locations = _per_latent_abs_locations(poses, lat_f) → [lat_f,3]；
          逐 latent 帧 bank.update(pose_emb=torch.zeros(dim), latent=latents[:,t],
          surprise_score=0.0, timestep=global_t, semantic_key=None,
          location=locations[t])；global_t+=1。dim=wan_i2v.low_noise_model.dim。
          然后 bank.increment_age()。
       e. 把本 clip 的视频帧 append 到 all_videos。
  4. 拼接 all_videos → 存 --save_file（对齐 v4 infer 的拼接+save_video）。

复刻而非 import 的小 helper（放本文件内，注明"对齐 train_v5，避免 import 拉
accelerate/deepspeed 重依赖"）：
  - _per_latent_abs_locations(poses, lat_f)：照搬 train_v5 同名函数。
  - _retrieve_memory_latents(bank, query_location, query_timestep, top_k,
    min_gap_frames, device)：照搬 train_v5:652-672。

本地无 torch/CUDA 真跑不动；--help / py_compile 走通即可（真跑待服务器）。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from os.path import abspath, dirname, join
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path（与 eval_v5.py / train_v5.py 一致）
# ---------------------------------------------------------------------------
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                            # → src/
_PROJECT_ROOT = dirname(_SRC_DIR)                            # → Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, "refs", "lingbot-world")

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 复用 eval_v5.py（不重写、不改 v5 其他文件）
#   - _load_v5_pipeline / _resolve_model_config：模型装载（WanI2V + low/high
#     转 V5 + memory_encoder.pth 载入 + W1/W1b 断言）
#   - _patch_memory_latents / _unpatch_memory_latents：把 memory_latents 绑进
#     generate 期间的 noise_model.forward（try/finally 还原）
#   - _v5_injectable_models：只 patch 已转 V5 的模型（W2 low-only 默认）
# 复用 paths.py：eval/infer 产出布局 + config 快照 + default_run_name
# ---------------------------------------------------------------------------
from pipeline.v5.eval_v5 import (  # noqa: E402
    _load_v5_pipeline,
    _patch_memory_latents,
    _unpatch_memory_latents,
)
from pipeline.common.paths import (  # noqa: E402
    infer_run_dir,
    snapshot_config,
    default_run_name,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "v5 in-context KV 记忆的多 clip 长视频自然画质生成：真实 bank 检索 → "
            "in-context KV 注入 → 多 clip 连续生成 → 存一条长视频（demo 用）。"
        )
    )

    # ---- 数据（严格对齐 v4 infer_v4.py）----
    p.add_argument("--image", type=str, required=True,
                   help="首帧图像路径（PNG/JPG）")
    p.add_argument("--action_path", type=str, required=True,
                   help="目录，含 poses.npy[T,4,4] / action.npy / intrinsics.npy"
                        "（整条 GT 动作轨迹，是控制信号）")

    # ---- 模型权重 ----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="lingbot-world 预训练权重目录（含 low_noise_model / high_noise_model / VAE / T5）")
    p.add_argument("--memory_encoder_ckpt", type=str, required=True,
                   help="训练好的 memory_encoder.pth（train_v5 save_memory_encoder 产出，"
                        "只含 memory_encoder.* 权重）")

    # ---- v5 模型超参（须与训练一致；默认走 _resolve_model_config 从 metadata 采纳）----
    p.add_argument("--grid", type=int, default=16,
                   help="MemoryEncoder 每帧 grid×grid token（须与训练一致，默认 16）")
    p.add_argument("--encoder_depth", type=int, default=1,
                   help="MemoryEncoder 残差块层数（须与训练一致，默认 1）")
    p.add_argument("--memory_layers", type=str, default=None,
                   help="注入层索引逗号分隔（如 '0,10,20,39'）；None/空=全部 block"
                        "（默认以 training_metadata.json 为准；显式传且与训练冲突会 raise）")
    p.add_argument("--inject_high", action="store_true", default=False,
                   help="默认 False=只把 low_noise_model 转 V5 + 注入（对齐训练，只训了 low 的 "
                        "memory_encoder；high 用未训练对齐的 encoder 仅消融，会污染主判读，慎用）。")

    # ---- 生成参数 ----
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay",
                   help="文本描述")
    p.add_argument("--size", type=str, default="480*832",
                   help="分辨率 H*W")
    p.add_argument("--frame_num", type=int, default=81,
                   help="每个 clip 的帧数（默认 81）")
    p.add_argument("--num_clips", type=int, default=12,
                   help="生成的 clip 数量（默认 12，目标 12 clip 连续生成）")
    p.add_argument("--sample_steps", type=int, default=70,
                   help="采样步数（默认 70）")
    p.add_argument("--sample_shift", type=float, default=10.0,
                   help="sigma shift（默认 10.0）")
    p.add_argument("--guide_scale", type=float, default=5.0,
                   help="CFG scale（默认 5.0）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（默认 42；每个 clip 用 seed + clip_idx 保证多样性）")
    p.add_argument("--fps", type=int, default=16,
                   help="输出视频 fps（默认 16）")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="目标设备（默认 cuda:0）")

    # ---- bank 参数（默认值对齐 train_v5）----
    p.add_argument("--short_cap", type=int, default=1,
                   help="ShortTermBank 容量（默认 1，对齐 train_v5）")
    p.add_argument("--medium_cap", type=int, default=8,
                   help="MediumTermBank 容量（默认 8，对齐 train_v5）")
    p.add_argument("--long_cap", type=int, default=256,
                   help="LongTermBank 容量（默认 256，对齐 train_v5 长视频 demo）")
    p.add_argument("--surprise_threshold", type=float, default=0.4,
                   help="MediumTermBank 写入下限（默认 0.4，对齐 train_v5）")
    p.add_argument("--stability_threshold", type=float, default=0.2,
                   help="LongTermBank stable 写入上限（默认 0.2，对齐 train_v5）")
    p.add_argument("--novelty_threshold", type=float, default=0.7,
                   help="LongTermBank novelty 写入上限（默认 0.7，对齐 train_v5）")
    p.add_argument("--half_life", type=float, default=10.0,
                   help="MediumTermBank age decay 半衰期（单位 chunk，默认 10.0，对齐 train_v5）")
    p.add_argument("--dup_threshold", type=float, default=0.95,
                   help="Cross-tier dedup 阈值（pose_emb cosine_sim > 此值认为冗余，默认 0.95，对齐 train_v5）")
    p.add_argument("--revisit_top_k", type=int, default=5,
                   help="retrieve_revisit 取回帧数上限（默认 5，对齐 train_v5）")
    p.add_argument("--revisit_min_gap_frames", type=int, default=0,
                   help="retrieve_revisit 排除 timestep 距离 < 此值的近邻帧（默认 0，对齐 train_v5）")

    # ---- 产出（走 paths.py；OUTPUT_ROOT 环境变量可覆盖根）----
    p.add_argument("--save_file", type=str, default=None,
                   help="输出 mp4 路径。默认落在 infer_run_dir 内（OUTPUT_ROOT/v5/infer/<run_name>/<tag>/long_video.mp4）")
    p.add_argument("--run_name", type=str, default=None,
                   help="infer run 名（默认 default_run_name('v5_infer')）")
    p.add_argument("--tag", type=str, default="long_video",
                   help="infer 场景 tag（默认 long_video）")

    return p.parse_args()


# ===========================================================================
# 复刻自 train_v5.py 的小 helper（注明：对齐 train_v5，避免 import 拉
# accelerate/deepspeed 重依赖——train_v5.py 顶部 import 了 accelerate 等
# 训练侧重依赖，推理脚本不应触发。）
# ===========================================================================

def _per_latent_abs_locations(poses: torch.Tensor, lat_f: int) -> torch.Tensor:
    """从 clip 的 [F,4,4] 位姿取每个 latent 帧的绝对 c2w 平移向量。

    与 prepare_control_signal 的 latent-帧子采样口径一致：每 4 个原始帧 → 1 个 latent 帧
    （poses[::4]），再裁/补到 lat_f 个。

    对齐 train_v5.py:471-490（逐字照搬，避免 import train_v5 拉 accelerate/deepspeed）。

    Args:
        poses:  [F, 4, 4] 该 clip 的绝对 c2w 位姿（CPU/GPU 均可）。
        lat_f:  该 clip 的 latent 帧数。

    Returns:
        locations: [lat_f, 3] 绝对位置（CPU float）。
    """
    abs_trans = poses[::4, :3, 3].float().cpu()  # [~F/4, 3]
    if abs_trans.shape[0] >= lat_f:
        abs_trans = abs_trans[:lat_f]
    else:
        pad = abs_trans[-1:].repeat(lat_f - abs_trans.shape[0], 1)
        abs_trans = torch.cat([abs_trans, pad], dim=0)
    return abs_trans  # [lat_f, 3]


def _retrieve_memory_latents(
    bank,
    query_location: torch.Tensor,
    query_timestep: int,
    top_k: int,
    min_gap_frames: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """surprise-independent revisit 检索，返回记忆帧 latent [K,16,h,w] 或 None。

    对齐 train_v5.py:652-672（照搬，参数从 args 解包传入以保持本文件无 args 依赖）。

    Args:
        bank:            ThreeTierMemoryBank 实例。
        query_location:  [3] 当前 clip 第一个 latent 帧的绝对 c2w 平移。
        query_timestep:  当前 clip 第一 latent 帧的全局 t。
        top_k / min_gap_frames: bank.retrieve_revisit 参数。
        device:          输出 tensor 放置设备。

    Returns:
        latents [K,16,h,w]（已 .to(device)）或 None（bank 空或检索空）。
    """
    if bank.size() == 0:
        return None
    frames, latents = bank.retrieve_revisit(
        query_location=query_location,
        query_timestep=query_timestep,
        top_k=top_k,
        min_gap_frames=min_gap_frames,
        device=device,
        return_latents=True,
    )
    if not frames or latents is None:
        return None
    return latents  # [K, 16, h, w]


# ===========================================================================
# bank 更新（surprise-independent，对齐 train_v5 context-clip 循环）
# ===========================================================================

def _update_bank_from_clip(
    bank,
    video: np.ndarray,
    wan_i2v,
    device: torch.device,
    poses_clip: torch.Tensor,
    global_t_start: int,
    dim: int,
    clip_idx: int,
) -> int:
    """VAE encode 本 clip 生成视频 → 逐 latent 帧 update bank（surprise-independent）。

    对齐 train_v5.multi_clip_training_step_v5 的 context-clip bank 填充口径
    （train_v5.py:540-568）：surprise_score=0.0（永远 stable，全进 long），
    semantic_key=None（LongTermBank.update 跳过 novelty check），pose_emb 仅占位
    （retrieve_revisit 不用 pose_emb，只用 location）。

    **train/infer bank update 口径对称不变量**：
      - 三元组 (latent[:,t], surprise=0.0, semantic_key=None, location=locations[t])
        逐 latent 帧写入，与 train_v5 一致；
      - global_t 跨 clip 累加，与 train_v5 _global_t 一致；
      - 写入后由调用方调 bank.increment_age()（对齐 train_v5 context-clip 间隙）。

    Args:
        bank:          ThreeTierMemoryBank 实例。
        video:         本 clip 生成视频，[3, F, H, W]（float，范围 [-1,1] 或 [0,255]，
                       VAE 接受前会做归一化处理）。
        wan_i2v:       WanI2V 管道（含 vae）。
        device:        目标设备。
        poses_clip:    本 clip 的 [F,4,4] c2w 位姿（CPU/GPU 均可）。
        global_t_start: 本 clip 第一个 latent 帧的全局 t（写入前的累加值）。
        dim:           骨干 dim（= wan_i2v.low_noise_model.dim），pose_emb 占位维度。
        clip_idx:      当前 clip 编号（写入 chunk_id）。

    Returns:
        更新后的 global_t（= global_t_start + lat_f）。
    """
    # FIX[B-02]（对齐 infer_v4）：offload_model=True 时 VAE 可能已在 CPU，需先移回 device
    vae_device = next(wan_i2v.vae.model.parameters()).device
    if vae_device != device:
        wan_i2v.vae.model.to(device)

    # video [3,F,H,W] numpy → tensor [C=3, T=F, H, W]（VAE 期望 [C,T,H,W]）
    if isinstance(video, np.ndarray):
        video_t = torch.from_numpy(video.copy())
    else:
        video_t = video
    # 归一化到 [-1,1]（VAE 期望）：若数据范围疑似 [0,255] 则转换
    _vmax = float(video_t.max()) if video_t.numel() > 0 else 0.0
    if _vmax > 1.5:  # [0,255] 范围
        video_t = video_t.float() / 127.5 - 1.0
    video_t = video_t.to(device)

    with torch.no_grad():
        latent = wan_i2v.vae.encode([video_t])[0]  # [z_dim=16, lat_f, lat_h, lat_w]

    lat_f = latent.shape[1]
    locations = _per_latent_abs_locations(poses_clip, lat_f)  # [lat_f, 3]

    _pose_placeholder = torch.zeros(dim)
    global_t = global_t_start
    for t_idx in range(lat_f):
        latent_frame = latent[:, t_idx]  # [16, lat_h, lat_w]
        bank.update(
            pose_emb=_pose_placeholder,
            latent=latent_frame,
            surprise_score=0.0,
            timestep=global_t,
            visual_emb=None,
            chunk_id=clip_idx,
            semantic_key=None,
            location=locations[t_idx],
        )
        global_t += 1

    logger.info(
        "Clip %d: bank updated with %d latent frames (long tier %d, total %d).",
        clip_idx, lat_f, bank.long.size(), bank.size(),
    )
    return global_t


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    args = _parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 产出目录（paths.py 新布局）----
    run_name = args.run_name or default_run_name("v5_infer")
    run_dir = infer_run_dir("v5", run_name, args.tag)

    # 文件日志（落 run 目录）
    log_path = os.path.join(str(run_dir), "infer.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    snapshot_config(run_dir, {k: v for k, v in vars(args).items()
                              if not k.startswith("_")})
    logger.info("v5 infer run_dir = %s", run_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 默认 save_file 落 run_dir
    if args.save_file is None:
        args.save_file = os.path.join(str(run_dir), "long_video.mp4")

    height, width = (int(x) for x in args.size.split("*"))
    logger.info("Args: %s", vars(args))

    # ---- 加载整条动作轨迹（与 v4 infer_v4 主循环口径一致）----
    if not (args.action_path and os.path.isdir(args.action_path)):
        logger.error("action_path 不存在或非目录：%s；退出。", args.action_path)
        return
    try:
        poses_np = np.load(os.path.join(args.action_path, "poses.npy"))
        actions_np = np.load(os.path.join(args.action_path, "action.npy"))
        intrinsics_np = np.load(os.path.join(args.action_path, "intrinsics.npy"))
        logger.info(
            "Loaded pose data: poses=%s actions=%s intrinsics=%s",
            poses_np.shape, actions_np.shape, intrinsics_np.shape,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("加载 action_path=%s 失败：%s；退出。", args.action_path, exc)
        return

    # ---- 加载 v5 pipeline（复用 eval_v5._load_v5_pipeline）----
    #   内部完成：WanI2V 构造 + low 转 WanModelWithMemoryV5 + memory_encoder.pth
    #   载入（W1/W1b 断言已过）。W2：默认 low-only（high 保持原始 WanModel）。
    logger.info("Loading v5 pipeline (low_noise_model → WanModelWithMemoryV5) ...")
    wan_i2v = _load_v5_pipeline(args, device)

    # 骨干 dim（pose_emb 占位维度，对齐 train_v5 用 model.dim）
    dim = wan_i2v.low_noise_model.dim
    logger.info("Backbone dim = %d（pose_emb 占位维度）", dim)

    # ---- bank 初始化（默认值对齐 train_v5）----
    from memory_module.memory_bank import ThreeTierMemoryBank
    bank = ThreeTierMemoryBank(
        short_cap=args.short_cap,
        medium_cap=args.medium_cap,
        long_cap=args.long_cap,
        surprise_threshold=args.surprise_threshold,
        stability_threshold=args.stability_threshold,
        novelty_threshold=args.novelty_threshold,
        half_life=args.half_life,
        dup_threshold=args.dup_threshold,
    )
    logger.info("ThreeTierMemoryBank created: %s", bank)

    # ---- 多 clip 连续生成 ----
    from wan.configs import MAX_AREA_CONFIGS
    max_area = MAX_AREA_CONFIGS[args.size]

    all_videos: List[torch.Tensor] = []
    current_img = Image.open(args.image).convert("RGB")

    global_t = 0  # 跨 clip 递增的 latent-帧 timestep（对齐 train_v5 _global_t）

    for clip_idx in range(args.num_clips):
        logger.info("Generating clip %d/%d ...", clip_idx + 1, args.num_clips)

        # 新 clip 开始前，已存储帧 age +1（MediumTermBank age decay；
        # 对齐 v4 infer_v4 / train_v5 context-clip 间隙）
        if clip_idx > 0:
            bank.increment_age()

        # a. 切本 clip 的 poses/actions/intrinsics（帧区间 [c*frame_num:(c+1)*frame_num]）
        clip_start = clip_idx * args.frame_num
        clip_end = clip_start + args.frame_num
        if clip_end <= len(poses_np):
            clip_poses = poses_np[clip_start:clip_end]
            clip_actions = actions_np[clip_start:clip_end]
            clip_intrinsics = intrinsics_np[clip_start:clip_end]
        else:
            # 数据不足 frame_num 时取最后 frame_num 帧（对齐 v4 infer_v4 fallback）
            clip_poses = (poses_np[-args.frame_num:]
                          if len(poses_np) >= args.frame_num else poses_np)
            clip_actions = (actions_np[-args.frame_num:]
                            if len(actions_np) >= args.frame_num else actions_np)
            clip_intrinsics = (intrinsics_np[-args.frame_num:]
                               if len(intrinsics_np) >= args.frame_num else intrinsics_np)
            logger.warning(
                "Clip %d: action data shorter than expected (%d < %d), using last %d frames.",
                clip_idx + 1, len(poses_np), clip_end, len(clip_poses),
            )
        poses_tensor = torch.from_numpy(clip_poses).float()

        # b. 检索记忆（仅 clip_idx>0，bank 非空时）
        memory_latents: Optional[torch.Tensor] = None
        if clip_idx > 0:
            # query_location = 本 clip 第一个 latent 帧的绝对位置（与 train_v5 target 口径一致）
            #   train_v5 用 lat_f（=video_latent.shape[1]）计算 locations；此处用 frame_num 估
            #   的 lat_f 上界（每 4 帧一 latent），取 locations[0] 即可（首帧位置不依赖 lat_f）。
            _est_lat_f = max(1, len(clip_poses) // 4)
            _locations_est = _per_latent_abs_locations(poses_tensor, _est_lat_f)
            query_location = _locations_est[0]  # [3]
            query_timestep = global_t           # 本 clip 第一 latent 帧的全局 t
            memory_latents = _retrieve_memory_latents(
                bank, query_location, query_timestep,
                top_k=args.revisit_top_k,
                min_gap_frames=args.revisit_min_gap_frames,
                device=device,
            )
            _k = 0 if memory_latents is None else memory_latents.shape[0]
            logger.info(
                "Clip %d: retrieved %d memory frames (revisit_top_k=%d, min_gap=%d).",
                clip_idx + 1, _k, args.revisit_top_k, args.revisit_min_gap_frames,
            )
        else:
            logger.info("Clip 1: pure i2v startup (memory_latents=None, bank empty).")

        # c. 生成：本 clip 的 poses/actions/intrinsics 存到临时 action 目录
        tmp_action_dir = tempfile.mkdtemp(prefix=f"v5_infer_clip{clip_idx}_")
        np.save(os.path.join(tmp_action_dir, "poses.npy"),
                clip_poses.astype(np.float32))
        np.save(os.path.join(tmp_action_dir, "action.npy"),
                clip_actions.astype(np.float32))
        np.save(os.path.join(tmp_action_dir, "intrinsics.npy"),
                clip_intrinsics.astype(np.float32))

        _patch_memory_latents(wan_i2v, memory_latents)
        try:
            video = wan_i2v.generate(
                args.prompt,
                current_img,
                action_path=tmp_action_dir,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale,
                seed=args.seed + clip_idx,
                offload_model=True,
            )
        finally:
            _unpatch_memory_latents(wan_i2v)
            shutil.rmtree(tmp_action_dir, ignore_errors=True)

        if video is None:
            logger.error("Clip %d: generate() 返回 None，终止。", clip_idx + 1)
            break

        # 归一化为 numpy [3,F,H,W] float（generate 可能返回 tensor 或 ndarray）
        if isinstance(video, torch.Tensor):
            video_np = video.detach().cpu().float().numpy()
        else:
            video_np = np.asarray(video, dtype=np.float32)
        # 存入 all_videos（torch.Tensor，对齐 v4 infer_v4 拼接口径）
        all_videos.append(torch.from_numpy(video_np.copy()))

        # d. bank 更新（surprise-independent，对齐 train_v5 context-clip 循环）
        global_t = _update_bank_from_clip(
            bank=bank,
            video=video_np,
            wan_i2v=wan_i2v,
            device=device,
            poses_clip=poses_tensor,
            global_t_start=global_t,
            dim=dim,
            clip_idx=clip_idx,
        )

        # e. 最后一帧作为下一 clip 的初始帧（autoregressive 链接，对齐 v4 current_img）
        last_frame_chw = video_np[:, -1]  # [C=3, H, W]
        last_frame_hwc = last_frame_chw.transpose(1, 2, 0)  # [H, W, 3]
        last_frame_uint8 = (last_frame_hwc * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        current_img = Image.fromarray(last_frame_uint8)

        logger.info(
            "Clip %d done. bank.size=%d, bank.long=%d, global_t=%d.",
            clip_idx + 1, bank.size(), bank.long.size(), global_t,
        )

    # ---- 拼接所有 clips + 存 mp4（对齐 v4 infer_v4 的拼接+save_video）----
    if not all_videos:
        logger.error("无视频生成，退出。")
        return

    full_video = torch.cat(all_videos, dim=1)  # [C, T*N, H, W]，沿时间维拼接

    try:
        # 优先复用 wan.utils.utils.save_video（与 v4 infer_v4 同路径）
        from wan.utils.utils import save_video
        _cfg_fps = getattr(wan_i2v, "config", None)
        _save_fps = getattr(_cfg_fps, "sample_fps", args.fps) if _cfg_fps else args.fps
        save_video(
            tensor=full_video[None],
            save_file=args.save_file,
            fps=_save_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logger.info("Saved video → %s (fps=%d)", args.save_file, _save_fps)
    except Exception as exc:  # noqa: BLE001
        # 回退：imageio / torchvision 写 mp4（fps=args.fps）
        logger.warning(
            "wan.utils.utils.save_video 不可用 (%s)；回退到 imageio/torchvision。", exc)
        _saved = False
        # imageio
        try:
            import imageio  # type: ignore
            _frames = full_video.permute(1, 2, 3, 0).cpu().float()  # [T,H,W,C]
            _frames = (_frames * 127.5 + 127.5).clip(0, 255).to(torch.uint8).numpy()
            imageio.mimsave(args.save_file, list(_frames), fps=args.fps,
                            codec="libx264", quality=8)
            logger.info("Saved video → %s (imageio, fps=%d)", args.save_file, args.fps)
            _saved = True
        except Exception as exc2:  # noqa: BLE001
            logger.warning("imageio 写视频失败：%s", exc2)
        # torchvision fallback
        if not _saved:
            try:
                import torchvision.io as tio  # type: ignore
                _frames = full_video.permute(1, 2, 3, 0).cpu().float()  # [T,H,W,C]
                _frames = (_frames * 127.5 + 127.5).clip(0, 255).to(torch.uint8)
                tio.write_video(args.save_file, _frames, fps=args.fps)
                logger.info("Saved video → %s (torchvision, fps=%d)",
                            args.save_file, args.fps)
                _saved = True
            except Exception as exc3:  # noqa: BLE001
                logger.error("torchvision 写视频也失败：%s；未保存 mp4。", exc3)
                raise

    logger.info("Done. 输出目录: %s | 视频: %s", run_dir, args.save_file)


if __name__ == "__main__":
    main()
