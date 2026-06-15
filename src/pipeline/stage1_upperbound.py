"""stage1_upperbound.py — Stage1 上界实验：冻结骨干的「记忆帧」天花板（无训练）

科学目的（experiment_design.md Exp2 的"上界"前置；OP-1/F-12 背景）
================================================================
我们的创新点是"三层记忆 + memory cross-attention 注入历史帧"。forward-only 诊断
（memory_injection_diag.py / diag.md）已确诊**我们当前的 memory 模块**注入无效，机制：
  (1) V 被 latent.mean(dim=[-2,-1]) 池化成单一无空间结构向量；
  (2) 检索 Key（pose_emb）在帧间近常量 → attention 被逼成均匀（零检索选择性）；
  (3) gate≈0.095 × out 本就只占 x 的 ~8-18%，记忆贡献仅 ~1%，70 步去噪 + 冻结
      i2v 强先验下被冲刷 → 生成 oracle≈off≈wrong。

在投入"改架构 + 训练"之前，本脚本用**无训练**实验回答一个 yes/no 问题：
  **"在这个冻结骨干上，把过去某帧作为记忆喂进去，到底能不能改善重访一致性？"**
（idea 的上界 / 天花板）

设计原则（关键）
----------------
要测 idea 的**上界**，必须**绕开我们当前坏掉的 memory 模块**（Key 常量、V 池化、
gate 关死、W_q/W_k/W_v/W_o 未充分训练），改用**冻结骨干自己已充分训练的原生 i2v
条件注入路径**：把 GT 历史帧当作一个额外的 clean anchor 条件帧塞进生成 context，让
骨干自身的 **self-attention** 去用它（FramePack / Context-as-Memory 思路）。
本实验**全程关闭** memory cross-attn（memory_states=None）。

三个 arm（每个都做真实生成：完整采样 + VAE decode）
---------------------------------------------------
对每个重访点 pt：
- **off**    ：标准 i2v 生成 query clip（首帧=query clip 首帧），无任何额外 anchor，
               memory 模块关闭。基线。直接调用 stock WanI2V.generate()。
- **oracle** ：与 off 相同的 query 生成，但**额外**把 GT 首访帧
               （pt.first_visit_frame）VAE encode 成 clean latent，作为一个 clean
               anchor 条件帧注入骨干 context，anchor 的相机位姿/plucker 设为该首访帧
               的真实位姿。memory 模块仍关闭。
- **wrong**  ：与 oracle 相同，但 anchor 换成远离 query/首访点的帧（_pick_wrong_indices）。

指标：revisit_consistency = max_t SSIM(gen_query_frame[:,t], GT_first_visit_frame)。

================================================================================
ANCHOR 注入机制（可行性闸 = 可行并已实现；anchor 如何进 msk/y/plucker，时间对齐）
================================================================================
背景：WanI2V 的原生 i2v 条件（image2video.py:302-401）沿 latent 时间维 lat_f =
(F-1)//4+1 耦合四个张量：
  - noise         : [16, lat_f, lat_h, lat_w]
  - y = concat([msk(4ch), vae_encode([first_frame, zeros(F-1)])(16ch)])
                    : [20, lat_f, lat_h, lat_w]；msk 第 0 个 latent 槽=1（clean），其余=0
  - c2ws_plucker  : [1, C, lat_f, lat_h, lat_w]，per-frame pose
  - RoPE grid_sizes: 由 patchify 后 x 的 shape 推出（model.py:476-477），
                     rope_apply 按 grid_sizes 的 f 维迭代（model.py:48-59），
                     因此**时间维扩展会被 RoPE 自动、干净地跟随**。
模型在 forward 里 `x = cat([x_noise, y], dim=0)`（通道维，model.py:472），要求 noise
与 y 的时间维一致。

**为什么不修改 generate() / 不污染 query 帧（BLOCK-1/2 修复：anchor append 到尾部）**：
不去 monkey-patch image2video.py 内部张量契约（那会动冻结骨干的数值契约，违反可行性闸
的"高风险改动"红线）。而是在本脚本内自包含实现 `_generate_with_anchor()`，**逐行镜像**
generate() 的张量构造，在 latent 时间维**尾部 append** `n_anchor` 个 clean 时间槽
（原 prepend 会把 query_0 挤出 VAE 因果首槽 + 相对位姿参考首槽，污染 query 段；Wan DiT
self-attention 在 flatten 时空 token 上全连接/非因果，anchor 放尾部对 query token 仍完全
可见，只有 VAE 是因果的）。core invariant：query 占 slots [0..lat_f_query-1]，anchor 占
尾部 [lat_f_query..lat_f_query+n_anchor-1]：

  1) **msk**：query 段原构造（首槽 1、其余 0）；anchor 槽全 1（clean）append 到尾部。
  2) **y（latent 部分）**：query cond latent = [query首帧 clean, zeros] VAE encode（与
     generate 逐元素一致）；anchor 单帧 clean latent **append 到尾部**。
     y = concat([msk, query_cond_latent ⊕ anchor_latent])。
  3) **noise**：query 噪声 [16,lat_f_query,...] 用固定 seed 先采（三臂 byte-identical），
     anchor 噪声用**独立** RNG（seed+1）单独采后 append 到尾部 → 不消耗 query 的 RNG、
     不改变 query 噪声（WARN-1 修复）。x 时间维 = lat_f_query+n_anchor，与 y 对齐。
  4) **plucker（彻底解耦，BLOCK-2 修复）**：query 段 plucker 完全按 off 臂方式计算
     （compute_relative_poses 参考帧=query[0]，输入不含 anchor）→ 与 off 逐元素一致；
     anchor 段单独把每帧位姿算成"相对 query[0]"（构造 [query_ref, anchor] 2 帧序列走同一
     framewise relative，取 frame[1]）→ append 到尾部。act 模式 only_rays_d 只用旋转，
     平移不进 plucker，两段各自 normalize 互不耦合（避免 cam 模式 max_norm 漂移）。
  5) **seq_len / grid_sizes / RoPE**：max_seq_len 按 lat_f_total=lat_f_query+n_anchor 重算；
     grid_sizes 由 patchify 自动推出；RoPE 自动跟随，无需手动扩展位置编码。

**时间对齐（vae_stride_t=4）如何处理**：每个 anchor 以**单帧 latent**形式占 1 个 latent
槽（[3,1,h,w] → [16,1,lat_h,lat_w]），不引入 pixel-frame 的 1+4k 约束。query 段时间结构
100% 不变 → query 帧的去噪/RoPE/条件/噪声与 off 基线**逐元素一致**，只是尾部多了可被
self-attn 注意的 clean anchor 槽。

**unpatchify / decode（BLOCK-1 修复）**：因果 VAE 下 query 段 lat_f_query 个 latent →
query_0(因果首槽)=1 帧 + 其余每 latent 帧 4 帧 = 1+4*(lat_f_query-1) = F_pix 帧；anchor
不在因果首槽 → 每 anchor latent 解 4 pixel 帧，位于**尾部**。故 decode 后丢弃**尾部**
vae_stride_t*n_anchor 帧、保留前 F_pix 帧（带断言 off/oracle query 段帧数相等），指标只
在 query clip 上算。

跨模块数据契约（新增，供 project_map.md）
-----------------------------------------
- [产出] stage1_upperbound.py:_encode_anchor_latent → anchor_latent [16, n_anchor, lat_h, lat_w]
  → [消费] stage1_upperbound.py:_generate_with_anchor 的 y 构造（**append 到尾部** cond latent）
- [产出] stage1_upperbound.py:_build_conditioning_with_anchor → [1, C, lat_f_query+n_anchor, lat_h, lat_w]
  → [消费] WanModel.forward dit_cond_dict["c2ws_plucker_emb"]（time 维 = lat_f_query+n_anchor，
    query 段在前/anchor 段在尾；query 段与 off 臂逐元素一致）
- 不引入任何训练/推理对称要求（本文件是无训练的纯推理探针；memory_states 全程 None，
  不经过 memory_cross_attn / bank / NFP 路径）。

依赖与约束
----------
- 复用 oracle_injection.py：_find_revisit_points / RevisitPoint / _frame_to_clip_slice /
  _SIZE_HW / _ORACLE_CLIP_FRAMES / _revisit_consistency / _ssim / _pick_wrong_indices /
  _to_gray_uint8 / _frame_to_pil / _save_frame_png / _save_video / _build_bank_for_config(未用)
- 复用 retrieval_probe.py：episode 加载 + GT 重访判定 + 解码 + VAE encode（import）
- 复用 infer_v4.py：load/convert pipeline（import，用于把 pipeline 对象建好；
  memory 模块虽被转换但本实验不注入 memory_states）
- 不修改任何已有文件（只新增本文件）；forward/生成全程 torch.no_grad；不训练。
- 单 GPU 即可运行（不支持 Ulysses SP，逻辑更清晰）。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import tempfile
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# sys.path 设置（与 oracle_injection.py / infer_v4.py / retrieval_probe.py 一致）
# ---------------------------------------------------------------------------

_PIPELINE_DIR = dirname(abspath(__file__))          # → src/pipeline/
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
# 复用 oracle_injection.py（重访点检测 / 帧切片 / 指标 / 帧选择 / IO，import 不重写）
# ---------------------------------------------------------------------------

from pipeline.oracle_injection import (  # noqa: E402
    RevisitPoint,
    _find_revisit_points,
    _frame_to_clip_slice,
    _revisit_consistency,
    _pick_wrong_indices,
    _pick_oracle_indices,
    _frame_to_pil,
    _save_frame_png,
    _save_video,
    _read_video_back,
)
import pipeline.oracle_injection as _oracle_inj  # noqa: E402  （用于设置 _SIZE_HW/_ORACLE_CLIP_FRAMES 全局）

# 复用 retrieval_probe.py（episode 加载 + GT 重访判定 + 解码 + VAE encode）
from pipeline.retrieval_probe import (  # noqa: E402
    EpisodeData,
    load_episode_clips,
    build_episode_data,
)

# 复用 infer_v4.py 的 pipeline 装载/转换（import，不重写）
from pipeline.infer_v4 import (  # noqa: E402
    _load_ft_model_and_prepare_ckpt,
    _convert_pipeline_to_memory,
)


# ---------------------------------------------------------------------------
# CLI（参数风格对齐 memory_injection_diag.py / oracle_injection.py）
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage1 上界实验：冻结骨干的记忆帧天花板（无训练）。"
            "对每个重访点跑 off / oracle / wrong 三 arm 真实生成，"
            "anchor 经原生 i2v 条件路径注入（memory cross-attn 全程关闭）。"
        )
    )
    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_full_train.csv")
    p.add_argument("--output_dir", type=str, required=True,
                   help="输出目录（视频 + stage1.md + stage1.json + run.log）")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")

    # ---- 模型权重（与 infer_v4 / oracle_injection 一致）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="基础模型目录（lingbot-world checkpoint，同 infer_v4）")
    p.add_argument("--ft_model_dir", type=str, default=None,
                   help="（可选）v4 low_noise_model checkpoint 目录。"
                        "注意：本实验只用原生 i2v 条件路径，memory 模块不参与；"
                        "ft 权重仅影响 DiT 主干（若 ft 是全参微调过的 CSGO-DiT）")
    p.add_argument("--ft_high_model_dir", type=str, default=None,
                   help="（可选）dual 训练 high_noise_model 目录")

    # ---- 重访点判定（复用 oracle_injection / retrieval_probe 口径）----
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
    p.add_argument("--num_anchor_frames", type=int, default=1,
                   help="注入多少个 anchor 帧（oracle/wrong arm）。"
                        "每个 anchor 占 1 个 latent 时间槽。默认 1（最干净）")

    # ---- 生成参数（与 oracle_injection 对齐）----
    p.add_argument("--frame_num", type=int, default=81,
                   help="query clip 帧数（4n+1）")
    p.add_argument("--num_inference_steps", type=int, default=70,
                   help="diffusion 采样步数")
    p.add_argument("--sample_shift", type=float, default=10.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16,
                   help="视频帧率（min_time_gap_sec→帧数换算 + 保存）")

    # ---- verdict 阈值 ----
    p.add_argument("--gap_threshold", type=float, default=0.05,
                   help="判定上界成立的 (oracle-off) 最小提升阈值（默认 +0.05）")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Anchor latent 编码（单帧 GT → clean VAE latent，单 latent 槽）
# ---------------------------------------------------------------------------

def _encode_anchor_latent(
    vae,
    anchor_frame_chw: np.ndarray,   # [3, H, W] in [-1, 1]
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    """把单帧 anchor GT 图像 VAE encode 成单帧 clean latent。

    VAE encode 接受 [3, F, H, W]；单帧（F=1）时间 stride=4 自然产出 1 个 latent 帧。
    这样 anchor 在 latent 空间正好占 **1 个时间槽**，无需 1+4k pixel 对齐。

    Args:
        vae:              Wan2_1_VAE
        anchor_frame_chw: [3, H_src, W_src] in [-1,1]
        h, w:             目标像素分辨率（与 query 生成一致，VAE 兼容）
        device:           目标设备

    Returns:
        anchor_latent: [16, 1, lat_h, lat_w]，clean（与 generate 的 y latent 同空间）
    """
    frame = torch.from_numpy(anchor_frame_chw).float()  # [3,H_src,W_src] in [-1,1]
    # resize 到目标分辨率（与 generate 内 img interpolate 到 (h,w) 一致语义）
    frame = torch.nn.functional.interpolate(
        frame[None], size=(h, w), mode="bicubic"
    )[0]  # [3,h,w]
    vid = frame[:, None, :, :].to(device)  # [3,1,h,w]
    with torch.no_grad():
        latent = vae.encode([vid])[0]  # [16, 1, lat_h, lat_w]（VAE 时间 stride 对单帧→1）
    return latent.float()


# ---------------------------------------------------------------------------
# Anchor plucker 构造（query 段按 off 计算；anchor c2w 位姿 append 到尾部）
# ---------------------------------------------------------------------------

def _build_conditioning_with_anchor(
    anchor_poses: Optional[np.ndarray],   # [n_anchor, 4, 4] 或 None（off arm）
    query_poses: np.ndarray,              # [frame_num, 4, 4]
    query_actions: np.ndarray,            # [frame_num, 4]
    query_intrinsics: np.ndarray,         # [frame_num, 4]
    h: int,
    w: int,
    lat_h: int,
    lat_w: int,
    control_type: str,
    param_dtype: torch.dtype,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """构造 c2ws_plucker_emb，anchor **append 到时间维尾部**（BLOCK-1/2 修复）。

    核心不变量（review 要求）：query 段 plucker 与 off 臂**逐元素一致** → query 占
    latent slots [0..lat_f_query-1]（query_0 仍是因果首槽 + 相对位姿参考首槽），
    anchor 占 [lat_f_query..lat_f_query+n_anchor-1]。Wan DiT 的 self-attention 在
    flatten 时空 token 上全连接（非因果），anchor 放尾部对 query token 仍完全可见。

    plucker 彻底解耦（BLOCK-2 修复）：
      - **query 段**：完全按 off 臂方式计算（compute_relative_poses 参考帧=query[0]，
        输入不含 anchor）→ 与 off 逐元素一致。
      - **anchor 段**：单独把每个 anchor 的位姿算成"相对 query[0]"——构造 2 帧序列
        [query_ref_c2w, anchor_c2w] 走同一 compute_relative_poses(framewise=True)，
        取 frame[1]（anchor 相对 query 参考帧的位姿）。act 模式 only_rays_d 只用旋转，
        平移不进 plucker，故 normalize_trans 的整段 max_norm 不影响 query 段（query
        段单独 normalize，anchor 段单独 normalize，互不耦合）。
      - 两段 plucker 在 frame 维 concat（query 在前，anchor 在尾）。

    时间对齐：每个 anchor 占 1 个 latent 槽（无时间插值）。最终 plucker 时间维 =
    lat_f_query + n_anchor。

    Args:
        anchor_poses:  [n_anchor,4,4] anchor 帧真实 c2w（None=off arm，不拼 anchor）
        query_*:       query clip 的 pose/action/intrinsics（[frame_num, ...]）
        h,w,lat_h,lat_w,control_type,param_dtype,patch_size: 来自 pipeline
        device:        目标设备

    Returns:
        (c2ws_plucker_emb [1, C, lat_f_query+n_anchor, lat_h, lat_w], n_anchor)
    """
    from einops import rearrange
    from wan.utils.cam_utils import (
        compute_relative_poses,
        interpolate_camera_poses,
        get_plucker_embeddings,
        get_Ks_transformed,
    )

    n_anchor = 0 if anchor_poses is None else int(anchor_poses.shape[0])
    c1 = int(h // lat_h)
    c2 = int(w // lat_w)

    # ---- Ks（与 generate() 完全一致）----
    Ks = torch.from_numpy(query_intrinsics).float()
    Ks = get_Ks_transformed(
        Ks, height_org=480, width_org=832,
        height_resize=h, width_resize=w, height_final=h, width_final=w,
    )
    Ks = Ks[0]  # [4]

    # ---- query 段相对位姿：完全按 off 臂方式（输入不含 anchor）----
    c2ws = query_poses.astype(np.float32)
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    c2ws = c2ws[:len_c2ws]
    lat_f_query = (len_c2ws - 1) // 4 + 1

    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
    )  # → [lat_f_query, 4, 4]（torch）；query 参考帧 = 该序列 frame[0]
    if not isinstance(c2ws_infer, torch.Tensor):
        c2ws_infer = torch.as_tensor(np.asarray(c2ws_infer)).float()
    query_ref_c2w = c2ws_infer[0:1].clone()  # [1,4,4]，anchor 段相对位姿的参考帧
    c2ws_query_rel = compute_relative_poses(c2ws_infer, framewise=True)  # 与 off 完全一致

    only_rays_d = (control_type == "act")

    def _plucker_for(c2ws_rel: torch.Tensor) -> torch.Tensor:
        """给定相对位姿序列 [f,4,4] → plucker [1, C_pose, f, lat_h, lat_w]（param_dtype）。"""
        f = c2ws_rel.shape[0]
        Ks_f = Ks.repeat(f, 1).to(device)
        emb = get_plucker_embeddings(
            c2ws_rel.to(device), Ks_f, h, w, only_rays_d=only_rays_d
        )
        emb = rearrange(emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=c1, c2=c2)
        emb = emb[None, ...]
        emb = rearrange(emb, "b (f h w) c -> b c f h w", f=f, h=lat_h, w=lat_w)
        return emb.to(param_dtype)

    query_plucker = _plucker_for(c2ws_query_rel)  # [1,C_pose,lat_f_query,lat_h,lat_w]

    # ---- anchor 段相对位姿：每个 anchor 相对 query 参考帧（2 帧序列取 frame[1]）----
    anchor_pluckers: List[torch.Tensor] = []
    if n_anchor > 0:
        for ai in range(n_anchor):
            anchor_c2w = torch.from_numpy(
                anchor_poses[ai:ai + 1].astype(np.float32)).to(query_ref_c2w.dtype)  # [1,4,4]
            pair = torch.cat([query_ref_c2w, anchor_c2w], dim=0)  # [2,4,4]
            pair_rel = compute_relative_poses(pair, framewise=True)  # [2,4,4]
            anchor_rel = pair_rel[1:2]  # [1,4,4]：anchor 相对 query 参考帧的 framewise 位姿
            anchor_pluckers.append(_plucker_for(anchor_rel))  # [1,C_pose,1,lat_h,lat_w]

    # ---- pose plucker：query 在前，anchor 在尾 concat（frame 维 dim=2）----
    if anchor_pluckers:
        c2ws_plucker_emb = torch.cat([query_plucker] + anchor_pluckers, dim=2)
    else:
        c2ws_plucker_emb = query_plucker
    lat_f_total = lat_f_query + n_anchor

    # ---- action（act 模式）：query [::4] 在前，anchor 零动作在尾 ----
    if control_type == "act":
        wasd_q = torch.from_numpy(
            query_actions.astype(np.float32)[:len_c2ws][::4]).float()  # [lat_f_query, A]
        if n_anchor > 0:
            zero_act = torch.zeros(n_anchor, wasd_q.shape[1], dtype=wasd_q.dtype)
            wasd = torch.cat([wasd_q, zero_act], dim=0)  # query 在前，anchor 零动作在尾
        else:
            wasd = wasd_q
        wasd = wasd.to(device)

        wasd_tensor = wasd[:, None, None, :].repeat(1, h, w, 1)
        wasd_tensor = rearrange(
            wasd_tensor, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=c1, c2=c2)
        wasd_tensor = wasd_tensor[None, ...]
        wasd_tensor = rearrange(
            wasd_tensor, "b (f h w) c -> b c f h w",
            f=lat_f_total, h=lat_h, w=lat_w).to(param_dtype)
        c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_tensor], dim=1)

    return c2ws_plucker_emb, n_anchor


# ---------------------------------------------------------------------------
# 核心：带 anchor 的真实生成（自包含镜像 WanI2V.generate，anchor append 到尾部）
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_with_anchor(
    wan_i2v,
    query_first_img: Image.Image,
    anchor_latent: Optional[torch.Tensor],   # [16, n_anchor, lat_h, lat_w] 或 None
    anchor_poses: Optional[np.ndarray],       # [n_anchor, 4, 4] 或 None
    query_poses: np.ndarray,
    query_actions: np.ndarray,
    query_intrinsics: np.ndarray,
    args,
    device: torch.device,
) -> Optional[np.ndarray]:
    """对一个重访点跑一次真实生成，返回 query 段视频 [3, F_query, H, W]。

    逐行镜像 WanI2V.generate()（image2video.py:217-511），唯一差异：在 latent 时间维
    **尾部 append** n_anchor 个 clean anchor 槽（BLOCK-1/2 修复，原 prepend 会污染
    query 因果首槽 + 相对位姿参考帧）。anchor 槽 = msk=1（clean）+ anchor 干净 latent +
    anchor 位姿 plucker（相对 query[0]）+ anchor 独立噪声。

    核心不变量：query 占 latent slots [0..lat_f_query-1]（query_0 仍是 VAE 因果首槽 +
    相对位姿参考首槽），anchor 占尾部 [lat_f_query..lat_f_total-1]。query 段的
    **noise / plucker / cond_latent / msk 与 off 基线逐元素一致**（off arm anchor=None）；
    Wan DiT self-attention 在 flatten 时空 token 上全连接（非因果），尾部 anchor 对 query
    token 仍完全可见。decode 后丢弃**尾部** vae_stride_t*n_anchor 个 pixel 帧（anchor 不在
    因果首槽，每 anchor latent 帧解 4 pixel 帧），保留前段 query 像素。
    memory_states 全程 None（memory cross-attn 关闭）。

    anchor_latent / anchor_poses 同时为 None → 退化为 off（标准 query 生成）。
    """
    import torch.distributed as dist
    from contextlib import contextmanager
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    n_anchor = 0 if anchor_latent is None else int(anchor_latent.shape[1])
    max_area = MAX_AREA_CONFIGS[args.size]
    vae_stride = wan_i2v.vae_stride          # (4, 8, 8)
    patch_size = wan_i2v.patch_size
    param_dtype = wan_i2v.param_dtype

    # ---- 分辨率推导（与 generate 一致）----
    img = TF.to_tensor(query_first_img).sub_(0.5).div_(0.5).to(device)  # [3,H,W] in [-1,1]
    F_pix = args.frame_num
    h0, w0 = img.shape[1:]
    aspect = h0 / w0
    lat_h = round(np.sqrt(max_area * aspect) // vae_stride[1]
                  // patch_size[1] * patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect) // vae_stride[2]
                  // patch_size[2] * patch_size[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    lat_f_query = (F_pix - 1) // vae_stride[0] + 1
    lat_f_total = lat_f_query + n_anchor

    max_seq_len = lat_f_total * lat_h * lat_w // (patch_size[1] * patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / wan_i2v.sp_size)) * wan_i2v.sp_size

    # ---- 噪声（WARN-1：query 噪声必须跨 off/oracle/wrong 逐元素一致）----
    # 先用固定 seed 采 query 噪声 [16, lat_f_query, ...]（三臂 byte-identical），
    # 再单独采 anchor 噪声 append 到尾部 → anchor 槽不提前消耗 RNG、不改变 query 噪声。
    seed = args.seed if args.seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    query_noise = torch.randn(
        16, lat_f_query, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device,
    )
    if n_anchor > 0:
        anchor_g = torch.Generator(device=device)
        anchor_g.manual_seed(seed + 1)  # 独立 RNG，不影响 query_noise
        anchor_noise = torch.randn(
            16, n_anchor, lat_h, lat_w,
            dtype=torch.float32, generator=anchor_g, device=device,
        )
        noise = torch.cat([query_noise, anchor_noise], dim=1)  # query 前 / anchor 尾
    else:
        noise = query_noise

    # ---- msk（pixel-frame 空间构造 query 段，再在 latent 空间 append anchor clean 槽）----
    # query 段 msk：首帧 clean、其余 0（与 generate image2video.py:311-318 一致）
    msk_q = torch.ones(1, F_pix, lat_h, lat_w, device=device)
    msk_q[:, 1:] = 0
    msk_q = torch.concat(
        [torch.repeat_interleave(msk_q[:, 0:1], repeats=4, dim=1), msk_q[:, 1:]],
        dim=1,
    )
    msk_q = msk_q.view(1, msk_q.shape[1] // 4, 4, lat_h, lat_w)
    msk_q = msk_q.transpose(1, 2)[0]   # [4, lat_f_query, lat_h, lat_w]
    if n_anchor > 0:
        # anchor 槽全 clean（=1），append 到尾部
        msk_anchor = torch.ones(4, n_anchor, lat_h, lat_w, device=device)
        msk = torch.cat([msk_q, msk_anchor], dim=1)   # [4, lat_f_total, lat_h, lat_w]
    else:
        msk = msk_q

    # ---- 文本编码（与 generate 一致）----
    n_prompt = wan_i2v.sample_neg_prompt
    if not wan_i2v.t5_cpu:
        wan_i2v.text_encoder.model.to(device)
        context = wan_i2v.text_encoder([args.prompt], device)
        context_null = wan_i2v.text_encoder([n_prompt], device)
        wan_i2v.text_encoder.model.cpu()
    else:
        context = wan_i2v.text_encoder([args.prompt], torch.device("cpu"))
        context_null = wan_i2v.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]

    # ---- cam / plucker（含 anchor 段）----
    c2ws_plucker_emb, _ = _build_conditioning_with_anchor(
        anchor_poses=anchor_poses,
        query_poses=query_poses,
        query_actions=query_actions,
        query_intrinsics=query_intrinsics,
        h=h, w=w, lat_h=lat_h, lat_w=lat_w,
        control_type=wan_i2v.control_type,
        param_dtype=param_dtype, patch_size=patch_size, device=device,
    )
    # 时间维一致性校验：plucker 时间维必须 = lat_f_total（防 anchor/query 对齐错位）
    assert c2ws_plucker_emb.shape[2] == lat_f_total, (
        f"plucker time dim {c2ws_plucker_emb.shape[2]} != lat_f_total {lat_f_total}"
    )
    # BLOCK-2 自检（debug）：oracle/wrong 臂的 query 段 plucker 必须与 off 臂逐元素一致。
    # off 臂的 plucker = 同一函数 anchor_poses=None 的输出；取 query 前 lat_f_query 段比对。
    if n_anchor > 0 and os.environ.get("STAGE1_SELFCHECK", "0") == "1":
        off_plucker, _ = _build_conditioning_with_anchor(
            anchor_poses=None, query_poses=query_poses, query_actions=query_actions,
            query_intrinsics=query_intrinsics, h=h, w=w, lat_h=lat_h, lat_w=lat_w,
            control_type=wan_i2v.control_type, param_dtype=param_dtype,
            patch_size=patch_size, device=device)
        q_seg = c2ws_plucker_emb[:, :, :lat_f_query]
        assert torch.allclose(q_seg, off_plucker, atol=1e-5), (
            "query 段 plucker 与 off 臂不一致（BLOCK-2 回归）")
        logger.info("[selfcheck] query 段 plucker == off 臂（allclose 通过）")

    dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

    # ---- y = concat([msk, cond_latent]）；anchor clean latent **append 到尾部** ----
    # query cond latent：[query首帧 clean, zeros(F_pix-1)] VAE encode（与 generate 逐元素一致）
    query_cond_latent = wan_i2v.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
            torch.zeros(3, F_pix - 1, h, w),
        ], dim=1).to(device)
    ])[0]   # [16, lat_f_query, lat_h, lat_w]

    if n_anchor > 0:
        # anchor clean latent append 到 query cond latent 之后（时间维尾部）
        anchor_lat = anchor_latent.to(query_cond_latent.dtype).to(device)  # [16,n_anchor,...]
        cond_latent = torch.cat([query_cond_latent, anchor_lat], dim=1)    # [16, lat_f_total, ...]
    else:
        cond_latent = query_cond_latent
    y = torch.concat([msk, cond_latent])   # [20, lat_f_total, lat_h, lat_w]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(wan_i2v.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high = getattr(wan_i2v.high_noise_model, "no_sync", noop_no_sync)

    with (
        torch.amp.autocast("cuda", dtype=param_dtype),
        torch.no_grad(),
        no_sync_low(),
        no_sync_high(),
    ):
        boundary = wan_i2v.boundary * wan_i2v.num_train_timesteps

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=wan_i2v.num_train_timesteps,
            shift=1, use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            args.num_inference_steps, device=device, shift=args.sample_shift)
        timesteps = sample_scheduler.timesteps

        latent = noise
        arg_c = {
            "context": [context[0]],
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }
        arg_null = {
            "context": context_null,
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }

        torch.cuda.empty_cache()
        from tqdm import tqdm
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(device)]
            timestep = torch.stack([t]).to(device)

            model = wan_i2v._prepare_model_for_timestep(t, boundary, offload_model=True)
            sample_guide_scale = (
                args.guide_scale if t.item() < boundary else args.guide_scale)

            # memory_states 全程 None：WanModelWithMemory.forward 退化为 base WanModel
            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            torch.cuda.empty_cache()
            noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
            torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + sample_guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            x0 = [latent]
            del latent_model_input, timestep

        wan_i2v.low_noise_model.cpu()
        wan_i2v.high_noise_model.cpu()
        torch.cuda.empty_cache()

        videos = wan_i2v.vae.decode(x0)   # [3, F_total_pix, H, W]

    del noise, latent, x0, sample_scheduler
    import gc
    gc.collect()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    video = videos[0]   # [3, F_total_pix, H, W]
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()

    # ---- 丢弃 anchor 对应的**尾部** pixel 帧，只留前段 query（BLOCK-1 修复）----
    # 因果 VAE：query 段 lat_f_query 个 latent → query_0(因果首槽)=1 帧 + 其余每帧 4 帧
    #          = 1 + 4*(lat_f_query-1) = F_pix；anchor 不在因果首槽 → 每 anchor latent
    # 解 4 pixel 帧，位于尾部 → 丢尾部 4*n_anchor 帧，保留前 F_pix 帧。
    if n_anchor > 0:
        anchor_pix = vae_stride[0] * n_anchor
        video = video[:, :video.shape[1] - anchor_pix, :, :]
    # 断言：off 与 oracle 的 query 段像素帧数相等（= F_pix；anchor 不改变 query 帧数）
    assert video.shape[1] == F_pix, (
        f"query 段像素帧数 {video.shape[1]} != F_pix {F_pix}（anchor 尾部丢弃错位）")
    return video   # [3, F_query=F_pix, H, W]


# ---------------------------------------------------------------------------
# anchor 帧索引选择（oracle/wrong arm）
# ---------------------------------------------------------------------------

def _build_anchor_inputs(
    ep: EpisodeData,
    frames: np.ndarray,            # [T, 3, H, W] in [-1,1]
    latents_per_frame: Optional[torch.Tensor],   # 未用（anchor 单帧重 encode 保证 clean 单槽）
    vae,
    frame_indices: List[int],
    h: int,
    w: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """为给定 anchor 帧索引构造 (anchor_latent [16,n_anchor,lat_h,lat_w], anchor_poses [n_anchor,4,4])。

    每个 anchor 帧：
      - latent：单帧 GT 图 VAE encode（_encode_anchor_latent，保证 1 latent 槽 / 帧）
      - pose  ：该帧的真实 c2w（ep.poses[fi]）
    """
    T = ep.poses.shape[0]
    lat_list: List[torch.Tensor] = []
    pose_list: List[np.ndarray] = []
    for fi in frame_indices:
        if fi < 0 or fi >= T:
            continue
        try:
            lat = _encode_anchor_latent(vae, frames[fi], h, w, device)  # [16,1,lat_h,lat_w]
        except Exception as exc:  # noqa: BLE001
            logger.warning("anchor latent 编码失败 fi=%d: %s；跳过", fi, exc)
            continue
        lat_list.append(lat)
        pose_list.append(ep.poses[fi])
    if not lat_list:
        return None
    anchor_latent = torch.cat(lat_list, dim=1)   # [16, n_anchor, lat_h, lat_w]
    anchor_poses = np.stack(pose_list, axis=0)   # [n_anchor, 4, 4]
    return anchor_latent, anchor_poses


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def _per_query_sign_stats(
    per_point: List[Dict],
    off_key: str,
    oracle_key: str,
    wrong_key: str,
) -> Dict:
    """per-query 符号口径（experiment_design「Exp 2 本轮执行口径」/ [[F-20]] 教训）。

    不只看 episode/全局均值（单点卡阈值会被均值掩盖），而是逐重访查询点看符号：
      - oracle−off 在各点**多数为正**（记忆被因果使用 → 提升一致性）
      - wrong−off 在各点**多数为负**（注入错误锚反而拉低）
    返回各点 gap 的符号统计（正占比 / 负占比 / 样本数），SSIM、DINO 各调用一次。
    某点 off 或对照值缺失（None）→ 该点该 gap 跳过，不计入分母。
    """
    n_oracle_pos = n_oracle = 0
    n_wrong_neg = n_wrong = 0
    for r in per_point:
        off = r.get(off_key)
        ora = r.get(oracle_key)
        wro = r.get(wrong_key)
        if off is not None and ora is not None:
            n_oracle += 1
            if (ora - off) > 0:
                n_oracle_pos += 1
        if off is not None and wro is not None:
            n_wrong += 1
            if (wro - off) < 0:
                n_wrong_neg += 1
    return {
        "n_oracle_points": n_oracle,
        "oracle_minus_off_pos_count": n_oracle_pos,
        "oracle_minus_off_pos_frac": (n_oracle_pos / n_oracle) if n_oracle else None,
        "n_wrong_points": n_wrong,
        "wrong_minus_off_neg_count": n_wrong_neg,
        "wrong_minus_off_neg_frac": (n_wrong_neg / n_wrong) if n_wrong else None,
    }


def _compute_verdict(
    per_point: List[Dict],
    gap_threshold: float,
) -> Dict:
    """聚合 off/oracle/wrong 的 rc_max，给自动 verdict（SSIM 与 DINO 两套并列）。

    **DINO 为主判据**（experiment_design「Exp2 评测指标补强」/ [[F-21]]），SSIM 为辅、
    向后兼容（N=2 旧结果仍可比）。两套各算：
      (1) episode 均值口径：(oracle−off) >= +gap_threshold 且 oracle > wrong → 上界成立。
      (2) per-query 符号口径（[[F-20]] 教训，主看）：各重访点 oracle−off 多数为正
          + wrong−off 多数为负 → 方向性接口裁决（结果 A）。
    DINO 全缺失（加载失败）时 DINO 均值/符号统计为 None，自动回退 SSIM 主判。
    """

    def _mean_gap_holds(off_key, oracle_key, wrong_key):
        offs = [r[off_key] for r in per_point if r.get(off_key) is not None]
        oracles = [r[oracle_key] for r in per_point if r.get(oracle_key) is not None]
        wrongs = [r[wrong_key] for r in per_point if r.get(wrong_key) is not None]

        def _mean(xs):
            return float(np.mean(xs)) if xs else None

        off_m, oracle_m, wrong_m = _mean(offs), _mean(oracles), _mean(wrongs)
        gap_oracle = (oracle_m - off_m) if (oracle_m is not None and off_m is not None) else None
        gap_wrong = (wrong_m - off_m) if (wrong_m is not None and off_m is not None) else None
        holds = False
        if (gap_oracle is not None and oracle_m is not None and wrong_m is not None):
            holds = (gap_oracle >= gap_threshold) and (oracle_m > wrong_m)
        return off_m, oracle_m, wrong_m, gap_oracle, gap_wrong, holds

    # ---- SSIM（向后兼容字段名）----
    (off_m, oracle_m, wrong_m,
     gap_oracle, gap_wrong, holds) = _mean_gap_holds("off", "oracle", "wrong")
    ssim_sign = _per_query_sign_stats(per_point, "off", "oracle", "wrong")

    # ---- DINO（主判据；rc_dino_max 字段 off_dino/oracle_dino/wrong_dino）----
    (off_dm, oracle_dm, wrong_dm,
     gap_oracle_d, gap_wrong_d, holds_d) = _mean_gap_holds(
        "off_dino", "oracle_dino", "wrong_dino")
    dino_sign = _per_query_sign_stats(per_point, "off_dino", "oracle_dino", "wrong_dino")
    dino_available = (off_dm is not None and oracle_dm is not None)

    # 主判据：DINO 优先（可用时），否则回退 SSIM
    primary = "dino" if dino_available else "ssim"
    holds_primary = holds_d if dino_available else holds

    if holds_primary:
        verdict = (
            f"idea 上界成立（主判据={primary.upper()}）：冻结骨干能利用记忆帧，"
            "病在我们模块实现 → 进 Stage 2 改 V 表征+修 Key+放带宽"
        )
    else:
        verdict = (
            f"idea 上界不成立（主判据={primary.upper()}）：冻结骨干用不了外部记忆帧 → "
            "需换干预（解冻部分 block）"
        )
    return {
        # ---- SSIM（向后兼容，旧字段名不变）----
        "off_mean": off_m,
        "oracle_mean": oracle_m,
        "wrong_mean": wrong_m,
        "gap_oracle_minus_off": gap_oracle,
        "gap_wrong_minus_off": gap_wrong,
        "gap_threshold": gap_threshold,
        "upper_bound_holds": holds,
        "ssim_per_query_sign": ssim_sign,
        # ---- DINO（主判据，新增）----
        "off_mean_dino": off_dm,
        "oracle_mean_dino": oracle_dm,
        "wrong_mean_dino": wrong_dm,
        "gap_oracle_minus_off_dino": gap_oracle_d,
        "gap_wrong_minus_off_dino": gap_wrong_d,
        "upper_bound_holds_dino": holds_d,
        "dino_per_query_sign": dino_sign,
        "dino_available": dino_available,
        # ---- 主判据标注 ----
        "primary_metric": primary,
        "upper_bound_holds_primary": holds_primary,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Summary 输出（stage1.md + stage1.json）
# ---------------------------------------------------------------------------

def _append_per_point_csv(output_dir: str, record: Dict) -> None:
    """逐点增量写 per_point.csv（P1 抗崩）。"""
    csv_path = os.path.join(output_dir, "per_point.csv")
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 per_point.csv 失败: %s", exc)


def _write_summary(args, per_point: List[Dict], verdict: Dict) -> None:
    """输出 stage1.md（人类可读）+ stage1.json。"""
    json_path = os.path.join(args.output_dir, "stage1.json")
    with open(json_path, "w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "per_point": per_point,
            "verdict": verdict,
        }, fh, indent=2)

    md: List[str] = []
    md.append("# Stage1 上界实验 Summary（冻结骨干的记忆帧天花板，无训练）\n\n")
    md.append(f"- timestamp: {datetime.now().isoformat()}\n")
    md.append(f"- ckpt_dir: {args.ckpt_dir}\n")
    md.append(f"- ft_model_dir: {args.ft_model_dir}\n")
    md.append(f"- num_anchor_frames: {args.num_anchor_frames} | "
              f"sample_steps: {args.num_inference_steps} | "
              f"shift: {args.sample_shift} | guide: {args.guide_scale}\n\n")
    md.append("> Anchor 经**原生 i2v 条件路径**注入（msk/y/plucker 尾部 append clean 槽，"
              "query 段与 off 逐元素一致），"
              "**memory cross-attn 全程关闭**（memory_states=None）。\n")
    md.append("> 指标 revisit_consistency = max_t SSIM(gen_query[:,t], 首访 GT 帧)。\n\n")

    def _f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    md.append("## 每个重访点 × 三 arm（SSIM rc_max + DINO rc_dino_max 并列）\n\n")
    md.append("| episode | query_frame | first_visit | off | oracle | wrong | "
              "oracle−off | wrong−off | "
              "off_dino | oracle_dino | wrong_dino | "
              "oracle−off (dino) | wrong−off (dino) |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in per_point:
        g_o = (r["oracle"] - r["off"]) if (r.get("oracle") is not None and r.get("off") is not None) else None
        g_w = (r["wrong"] - r["off"]) if (r.get("wrong") is not None and r.get("off") is not None) else None
        g_od = (r["oracle_dino"] - r["off_dino"]) if (r.get("oracle_dino") is not None and r.get("off_dino") is not None) else None
        g_wd = (r["wrong_dino"] - r["off_dino"]) if (r.get("wrong_dino") is not None and r.get("off_dino") is not None) else None
        md.append(
            f"| {r['episode_id']} | {r['query_frame']} | {r['first_visit_frame']} | "
            f"{_f(r.get('off'))} | {_f(r.get('oracle'))} | {_f(r.get('wrong'))} | "
            f"{_f(g_o)} | {_f(g_w)} | "
            f"{_f(r.get('off_dino'))} | {_f(r.get('oracle_dino'))} | {_f(r.get('wrong_dino'))} | "
            f"{_f(g_od)} | {_f(g_wd)} |\n"
        )

    md.append("\n## 自动 Verdict（**DINO 为主判据**，SSIM 为辅 / 向后兼容）\n\n")
    md.append(f"- **主判据：{verdict['primary_metric'].upper()}**"
              f"（dino_available={verdict['dino_available']}；"
              f"DINO 全缺失时自动回退 SSIM）\n\n")

    md.append("### DINO（主判据）\n\n")
    md.append(f"- off_mean_dino   = {verdict['off_mean_dino']}\n")
    md.append(f"- oracle_mean_dino= {verdict['oracle_mean_dino']}\n")
    md.append(f"- wrong_mean_dino = {verdict['wrong_mean_dino']}\n")
    md.append(f"- gap (oracle−off, dino) = {verdict['gap_oracle_minus_off_dino']} "
              f"(阈值 ≥ {verdict['gap_threshold']})\n")
    md.append(f"- gap (wrong−off, dino)  = {verdict['gap_wrong_minus_off_dino']}\n")
    _ds = verdict["dino_per_query_sign"]
    md.append(f"- per-query 符号（[[F-20]] 教训，主看）："
              f"oracle−off 正占比 = {_f(_ds['oracle_minus_off_pos_frac'])} "
              f"({_ds['oracle_minus_off_pos_count']}/{_ds['n_oracle_points']})；"
              f"wrong−off 负占比 = {_f(_ds['wrong_minus_off_neg_frac'])} "
              f"({_ds['wrong_minus_off_neg_count']}/{_ds['n_wrong_points']})\n")
    md.append(f"- DINO 均值口径上界成立 = {verdict['upper_bound_holds_dino']}\n\n")

    md.append("### SSIM（辅助 / 向后兼容）\n\n")
    md.append(f"- off_mean   = {verdict['off_mean']}\n")
    md.append(f"- oracle_mean= {verdict['oracle_mean']}\n")
    md.append(f"- wrong_mean = {verdict['wrong_mean']}\n")
    md.append(f"- gap (oracle−off) = {verdict['gap_oracle_minus_off']} "
              f"(阈值 ≥ {verdict['gap_threshold']})\n")
    md.append(f"- gap (wrong−off)  = {verdict['gap_wrong_minus_off']}\n")
    _ss = verdict["ssim_per_query_sign"]
    md.append(f"- per-query 符号：oracle−off 正占比 = {_f(_ss['oracle_minus_off_pos_frac'])} "
              f"({_ss['oracle_minus_off_pos_count']}/{_ss['n_oracle_points']})；"
              f"wrong−off 负占比 = {_f(_ss['wrong_minus_off_neg_frac'])} "
              f"({_ss['wrong_minus_off_neg_count']}/{_ss['n_wrong_points']})\n")
    md.append(f"- SSIM 均值口径上界成立 = {verdict['upper_bound_holds']}\n\n")

    md.append(f"**结论：{verdict['verdict']}**\n\n")
    md.append("判读规则（主判据 DINO）：(oracle−off) ≥ +{:.2f} 且 oracle > wrong → 上界成立；"
              "且 per-query 符号（oracle−off 多数为正 + wrong−off 多数为负）做方向性裁决；"
              "否则上界不成立。SSIM 同口径并列、向后兼容。\n".format(verdict["gap_threshold"]))

    md.append("\n## 人工定性对比\n\n")
    md.append("每个重访点已保存：\n")
    md.append("- `q<frame>_off.mp4` / `q<frame>_oracle.mp4` / `q<frame>_wrong.mp4`\n")
    md.append("- `q<frame>_gt_first_visit.png`：该地点首访 GT 帧（参照）\n")

    md_path = os.path.join(args.output_dir, "stage1.md")
    with open(md_path, "w") as fh:
        fh.writelines(md)
    logger.info("写 summary: %s", md_path)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # oracle_injection 内的全局（_frame_to_clip_slice 等不依赖，但 _build_* 读它；本脚本不调那些）
    _oracle_inj._SIZE_HW = tuple(args.size.split("*"))
    _oracle_inj._ORACLE_CLIP_FRAMES = args.frame_num

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
    logger.info(
        "Stage1 上界实验：anchor 经原生 i2v 条件路径注入，memory cross-attn 全程关闭。"
        "off=stock query 生成；oracle=注入首访 GT 帧；wrong=注入远距离帧。"
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

    # ---- 加载 WanI2V pipeline（复用 infer_v4，转换为 WanModelWithMemory；但本实验不注入 memory）----
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS
    from pipeline.retrieval_probe import (
        _decode_episode_video,
    )

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
    logger.info("转换 pipeline → WanModelWithMemory（memory 不参与本实验，仅复用 DiT 主干）...")
    wan_i2v = _convert_pipeline_to_memory(
        wan_i2v,
        memory_ckpt_path=None,
        high_model_dir=args.ft_high_model_dir,
        low_model_dir=args.ft_model_dir,
    )

    all_per_point: List[Dict] = []

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

        # 解码 episode 全帧（anchor 帧 + 首访 GT 参照 + query 首帧都从这里取）
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码失败: %s；跳过", ep_id, exc)
            continue

        ep_out_dir = os.path.join(args.output_dir, ep_id)
        os.makedirs(ep_out_dir, exist_ok=True)

        for pt in points:
            try:
                # 首访 GT 帧（一致性参照 + 人工定性）
                gt_first = frames[pt.first_visit_frame]   # [3,H,W]
                _save_frame_png(
                    gt_first,
                    os.path.join(ep_out_dir, f"q{pt.query_frame}_gt_first_visit.png"))

                # query clip 段（首帧图 + pose/action/intrinsics 切片）
                poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(
                    ep, pt.query_frame, args.frame_num)
                base_img = _frame_to_pil(frames[seg_start])

                # anchor 帧索引（oracle = 首访点附近 GT；wrong = 远距离 GT）
                oracle_idx = _pick_oracle_indices(pt, args.num_anchor_frames, T)
                wrong_idx = _pick_wrong_indices(pt, args.num_anchor_frames, T, rng)
                oracle_in = _build_anchor_inputs(
                    ep, frames, None, wan_i2v.vae, oracle_idx, _resize_h(height, width, args),
                    _resize_w(height, width, args), device)
                wrong_in = _build_anchor_inputs(
                    ep, frames, None, wan_i2v.vae, wrong_idx, _resize_h(height, width, args),
                    _resize_w(height, width, args), device)

                point_record: Dict = {
                    "episode_id": ep_id,
                    "query_frame": pt.query_frame,
                    "first_visit_frame": pt.first_visit_frame,
                    # SSIM rc_max（向后兼容，N=2 旧结果仍可比）
                    "off": None, "oracle": None, "wrong": None,
                    # DINO rc_dino_max（主判据；与 SSIM 字段并列，DINO 失败时占位 None）
                    "off_dino": None, "oracle_dino": None, "wrong_dino": None,
                }

                # ---- 三 arm ----
                arms: List[Tuple[str, Optional[Tuple[torch.Tensor, np.ndarray]]]] = [
                    ("off", None),
                    ("oracle", oracle_in),
                    ("wrong", wrong_in),
                ]
                for arm_name, anchor_in in arms:
                    mp4_path = os.path.join(ep_out_dir,
                                            f"q{pt.query_frame}_{arm_name}.mp4")
                    # P2 可续：mp4 已存在 → 读回算指标，跳过生成
                    if os.path.exists(mp4_path):
                        video = _read_video_back(mp4_path)
                        if video is not None:
                            metrics = _revisit_consistency(video, gt_first)
                            point_record[arm_name] = metrics["revisit_consistency_max"]
                            # DINO 主判据：_revisit_consistency 内部已算（DINO 加载失败时
                            # 返回 dict 无 dino_* key → .get() 得 None，占位稳定）
                            point_record[f"{arm_name}_dino"] = metrics.get(
                                "revisit_consistency_dino_max")
                            logger.info("ep=%s q=%d [%s] (读回) rc_max=%.4f rc_dino_max=%s",
                                        ep_id, pt.query_frame, arm_name,
                                        metrics["revisit_consistency_max"],
                                        metrics.get("revisit_consistency_dino_max"))
                            continue

                    anchor_latent = anchor_in[0] if anchor_in is not None else None
                    anchor_poses = anchor_in[1] if anchor_in is not None else None
                    if arm_name != "off" and anchor_in is None:
                        logger.warning(
                            "ep=%s q=%d [%s]：anchor 构造为空 → 退化为 off（不注入）",
                            ep_id, pt.query_frame, arm_name)

                    video = _generate_with_anchor(
                        wan_i2v, base_img, anchor_latent, anchor_poses,
                        poses_c, acts_c, intr_c, args, device)
                    if video is None:
                        logger.warning("ep=%s q=%d [%s]：生成返回 None，跳过该 arm",
                                       ep_id, pt.query_frame, arm_name)
                        continue
                    _save_video(video, mp4_path, fps=args.fps)
                    metrics = _revisit_consistency(video, gt_first)
                    point_record[arm_name] = metrics["revisit_consistency_max"]
                    # DINO 主判据：与 SSIM 字段并列记录（DINO 加载失败 → None 占位）
                    point_record[f"{arm_name}_dino"] = metrics.get(
                        "revisit_consistency_dino_max")
                    logger.info("ep=%s q=%d [%s] rc_max=%.4f rc_mean=%.4f rc_dino_max=%s",
                                ep_id, pt.query_frame, arm_name,
                                metrics["revisit_consistency_max"],
                                metrics["revisit_consistency_mean"],
                                metrics.get("revisit_consistency_dino_max"))
                    torch.cuda.empty_cache()

                all_per_point.append(point_record)
                _append_per_point_csv(args.output_dir, point_record)
            except Exception as exc:  # noqa: BLE001
                logger.exception("重访点处理失败 ep=%s q=%d: %s",
                                 ep_id, pt.query_frame, exc)
                continue

        del frames
        torch.cuda.empty_cache()

    verdict = _compute_verdict(all_per_point, args.gap_threshold)
    _write_summary(args, all_per_point, verdict)
    logger.info("Verdict: %s", verdict["verdict"])
    logger.info("Done. 输出目录: %s", args.output_dir)


# ---------------------------------------------------------------------------
# 分辨率换算辅助（与 generate 的 lat_h/lat_w → h/w 推导一致，供 anchor encode 用）
# ---------------------------------------------------------------------------

def _resize_h(height: int, width: int, args) -> int:
    """复算 generate 内部的实际像素高度 h（anchor VAE encode 须与 query 同 h/w）。"""
    return _resize_hw(height, width, args)[0]


def _resize_w(height: int, width: int, args) -> int:
    return _resize_hw(height, width, args)[1]


def _resize_hw(height: int, width: int, args) -> Tuple[int, int]:
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    max_area = MAX_AREA_CONFIGS[args.size]
    cfg = WAN_CONFIGS["i2v-A14B"]
    vae_stride = cfg.vae_stride
    patch_size = cfg.patch_size
    aspect = height / width
    lat_h = round(np.sqrt(max_area * aspect) // vae_stride[1]
                  // patch_size[1] * patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect) // vae_stride[2]
                  // patch_size[2] * patch_size[2])
    return lat_h * vae_stride[1], lat_w * vae_stride[2]


if __name__ == "__main__":
    main()
