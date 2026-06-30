"""
ideal_inject_diag.py — v5-KV 理想注入诊断（encoder-free ideal injection；experiment_design Step 43 / S-V4）
==========================================================================================================

**目的**（decisions.md 讨论 9/10 + open_problems OP-5）：v5 in-context KV 两层 NO-GO 已坐实
根因 = memory_encoder 惰性（输出跨输入 cos≈1.0000）。所以「冻结骨干能否用 in-context KV」这个
**通道本身从未被公平测试**——所有测试都经过惰性 encoder。本脚本 = **绕过 encoder，把「理想内容」
直接灌进 in-context KV 槽（mem_tokens → 冻结 W_k/W_v），看冻结骨干生成的重访 clip 是否优于
memory-off**。这是「修 encoder vs pivot latent-concat / 跳阶梯 ② LoRA-q」的 go/no-go 闸门。

**三臂定量**（每个 revisit case）：
  · off       : 无 memory（baseline，纯 i2v）。
  · ideal_A   : 对的 GT 首访帧逐层真实 KV-cache（冻结骨干过一遍记忆帧，forward hook 抓每个注入层
                self_attn 的输入 hidden，逐层不同地注入 → 经该层冻结 self.k/self.v 投影成真·KV）。
                这是骨干用 in-context KV 的**理论上界**。
  · random_A  : 错帧（非首访随机历史帧）逐层真实 KV-cache（内容对照，排除「随便注入都扰动」）。
可选第 4 臂：
  · ideal_B   : mem = pool(patch_embedding(zero-padded GT 首访帧 latent))，所有层共享
                （encoder 被设计去逼近的分布内投影的理想版；--arms 里显式加 ideal_B 才跑）。

**判据（GO/NO-GO）**：复用既有 frame-aligned DINO 口径（_revisit_consistency：生成的重访 clip
逐帧 vs GT 首访帧的 DINOv2 cosine，取 mean）。
  GO  ⇔ ideal_A 的 DINO 均值 > off + margin（默认 0.01）**且** ideal_A > random_A + margin
        → 通道+骨干能用 = 情况（乙）→ **修 encoder**（理想 KV 当蒸馏目标），不上 LoRA-q。
  NO-GO ⇔ 否则 → 通道真用不了 = 情况（甲）→ 进 OP-5 阶梯 ② backbone q(/k) LoRA / pivot latent-concat。

**最大化复用**：模型装载 / 三臂生成的 patch 机制走 eval_v5；revisit 点查找 / 首帧弱化 / 采样 +
VAE decode 的 IO / DINO 打分走 oracle_injection（不改 v4 / refs）。档 A 的「逐层捕获 + 逐层注入」
经 model_with_memory_v5 的 `set_mem_source('ideal_A') + set_layer_kv_cache(cache)` 开关实现
（默认 'encoder' 行为逐位不变）。

服务器跑前置：`export TMPDIR=/tmp`（否则 kill 掉的 run 会在仓库留 pymp-* 孤儿）。
本地无 torch/CUDA/DINO 真跑不动；--help / py_compile 走通即可（真跑待服务器）。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from os.path import abspath, dirname, join
from typing import Dict, List, Optional

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
# 复用 oracle_injection.py（v4 eval）的纯脚手架（import，不重写、不改 v4）
# ---------------------------------------------------------------------------
from pipeline.eval.oracle_injection import (  # noqa: E402
    RevisitPoint,
    _find_revisit_points,
    _frame_to_clip_slice,
    _weaken_image,
    _revisit_consistency,
    _frame_to_pil,
    _save_frame_png,
    _save_video,
    _read_video_back,
)
from pipeline.eval.retrieval_probe import (  # noqa: E402
    load_episode_clips,
    build_episode_data,
    _decode_episode_video,
    _vae_encode_batched,
    _expand_latents_to_frames,
)
# 复用 eval_v5 的模型装载 + 三臂注入 patch 机制（档 B 用 _patch_memory_latents）
from pipeline.v5.eval_v5 import (  # noqa: E402
    _load_v5_pipeline,
    _patch_memory_latents,
    _unpatch_memory_latents,
    _pick_random_hist_frame,
)
from pipeline.common.paths import (  # noqa: E402
    eval_run_dir,
    snapshot_config,
    default_run_name,
)


# 臂顺序（off 在前作 baseline）。ideal_B 默认不跑，--arms 显式加才跑。
ALL_ARMS = ("off", "ideal_A", "random_A", "ideal_B")
DEFAULT_ARMS = ("off", "ideal_A", "random_A")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "v5-KV 理想注入诊断（encoder-free）：off / ideal_A（逐层真实 KV-cache）/ "
            "random_A（错帧对照）[/ ideal_B] × frame-aligned DINO。"
            "服务器跑前置：export TMPDIR=/tmp。"
        )
    )
    # ---- 模型权重（与 eval_v5 对齐）----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="lingbot-world 预训练权重目录（含 low/high noise_model / VAE / T5）")
    p.add_argument("--memory_encoder_ckpt", type=str, required=True,
                   help="训练好的 memory_encoder.pth（仅用于重建 V5 模型壳；ideal_A/B 不经 encoder，"
                        "但 _load_v5_pipeline 的确载断言需要它）")
    # ---- v5 模型超参（默认走 _resolve_model_config 从 training_metadata.json 采纳）----
    p.add_argument("--grid", type=int, default=16,
                   help="MemoryEncoder 每帧 grid×grid token（须与训练一致，默认 16；档 B 池化也用它）")
    p.add_argument("--encoder_depth", type=int, default=1)
    p.add_argument("--memory_layers", type=str, default=None,
                   help="注入层索引逗号分隔（如 '0,10,20,39'）；None/空=全部 block")
    p.add_argument("--inject_high", action="store_true", default=False,
                   help="默认 False=只 low 转 V5 + 注入（对齐训练）。开启时 high 也转 V5（消融）。")

    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")
    p.add_argument("--max_cases", type=int, default=5,
                   help="总 revisit case 上限（默认 5，对齐 Step 43「5 个 revisit case」）")

    # ---- 臂选择 ----
    p.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS),
                   help="逗号分隔的臂子集（默认 off,ideal_A,random_A；可加 ideal_B）")

    # ---- 判据 ----
    p.add_argument("--go_margin", type=float, default=0.01,
                   help="GO 判据 margin：ideal_A 的 DINO 均值须 > off 且 > random_A 至少这么多（默认 0.01）")

    # ---- 档 A 捕获 ----
    p.add_argument("--capture_steps", type=int, default=70,
                   help="档 A 捕获时用来确定「最小噪声 timestep」的采样步数（取该调度的最小 t；默认 70，"
                        "与生成 num_inference_steps 同源）")

    # ---- 首帧弱化（F-18 护栏，默认 zero）----
    p.add_argument("--weaken_first_frame", type=str, default="zero",
                   choices=["noise", "zero", "none"],
                   help="zero=置零中性灰（默认，温和锚点）/ none=不弱化 / noise=随机 RGB（摧毁锚点，仅消融）")

    # ---- 重访点判定（复用 oracle_injection 口径）----
    p.add_argument("--hit_dist", type=float, default=40.0)
    p.add_argument("--hit_yaw", type=float, default=30.0)
    p.add_argument("--intermediate_separation", type=float, default=100.0)
    p.add_argument("--min_time_gap_sec", type=float, default=5.0)
    p.add_argument("--clip_overlap_frames", type=int, default=0)
    p.add_argument("--max_revisit_points", type=int, default=2)

    # ---- 生成参数 ----
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=70)
    p.add_argument("--sample_shift", type=float, default=10.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--size", type=str, default="480*832", help="分辨率 H*W")
    p.add_argument("--prompt", type=str,
                   default="First-person view of CS:GO competitive gameplay")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)

    # ---- 产出 ----
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--tag", type=str, default="ideal_inject",
                   help="eval 场景 tag（INDEX 区分用，默认 ideal_inject）")

    # ---- 定性渲染（S-V4 / Step 43 定性部分：完整多 clip 自回归长视频，off vs ideal_A）----
    p.add_argument("--render_qual", action="store_true", default=False,
                   help="定性渲染开关（向后兼容 additive）：开启时走【定性渲染路径】——对每个 "
                        "revisit case 渲完整多 clip 自回归长视频（off vs ideal_A），不算 DINO / "
                        "不判 GO/NO-GO；纯出视频供眼看。**不开时定量三臂行为逐位不变。** "
                        "定量与定性可分别独立跑（同一脚本，靠本开关切路径）。")
    p.add_argument("--num_clips_qual", type=int, default=5,
                   help="定性长视频的 clip 数（默认 5；5 clips × frame_num=81 @ fps=16 ≈ 25s/405 帧）。"
                        "帧数复用 --frame_num（默认 81）。")
    p.add_argument("--qual_arms", type=str, default="off,ideal_A",
                   help="定性渲染的臂子集（逗号分隔，默认 off,ideal_A；可加 random_A，省算力默认不含）。")

    # ---- 分片（additive：shard_count 默认 1 → 逐位与改前一致）----
    # 多卡并行诊断用：与 eval_v5 语义一致，但**切分轴是 revisit case**（非 episode），
    # 因为本诊断的全局上限是 --max_cases（case 数）而非 episode 数。先在所有 episode 上
    # 跑 revisit case 检测得到**完整 case 列表**（按 episode 升序、episode 内 query 升序），
    # 并**在分片切分之前**应用 --max_cases 全局上限，保证各分片看到同一全集；然后本分片
    # 只处理 `case_global_index % shard_count == shard_index` 的 case。各分片用各自 GPU
    # （脚本层 CUDA_VISIBLE_DEVICES 指定，代码内 --device 仍用 cuda:0），写各自 run_dir
    # （脚本传不同 --tag），跑完用 merge_diag_shards.py 合并 per_window.csv + 重算判决。
    # 单卡场景不传这两个参数即可（shard_count=1 → 走原始路径，逐位不变）。
    p.add_argument("--shard_index", type=int, default=0,
                   help="当前分片索引（0-based），多卡并行诊断用。默认 0。")
    p.add_argument("--shard_count", type=int, default=1,
                   help="总分片数。默认 1=不分片（走原始路径，逐位与单进程一致）；"
                        ">1 时按 case 全局序号取模 [case_idx %% shard_count == shard_index] 分片。")

    return p.parse_args()


# ===========================================================================
# 档 A：逐层真实 KV-cache 捕获
#   让冻结骨干对「记忆帧」（GT 首访帧 / 错帧）跑一次 forward，在固定低噪 timestep 下
#   用 forward_pre_hook 抓每个注入层 self_attn 的输入 hidden（= 经 norm1+modulation 后、
#   self.k/self.v 期望的那个 hidden），存成 cache[layer_i]。
#   注入时（ideal_A）对每层 set_memory(cache[layer_i])，经该层冻结 self.k/self.v 投影成
#   真·KV——与骨干对当前帧 token 的 K/V 处理逐字一致，故是「骨干用 in-context KV 的理论上界」。
# ===========================================================================

def _build_capture_inputs(wan_i2v, img, clip_frames, poses_c, acts_c, intr_c,
                          args, device):
    """复刻 WanI2V.generate 的输入准备（image2video.py:281-401），为「记忆帧所在 clip」
    构造一次低噪 forward 的全部输入。**不改 refs**，逐行对齐 generate。

    与 generate 的唯一区别：generate 从纯噪声起步采样，本函数用**记忆帧 clip 的干净 GT
    latent** x0（VAE encode clip_frames）在**最小噪声 timestep** t_min 下加极少噪声 →
    x_t≈干净帧，让骨干输出「接近干净的记忆帧 hidden」（Step 43：固定低噪 timestep / 接近干净帧）。

    Returns:
        dict(x_t=[16,lat_f,lat_h,lat_w], t_min=Tensor[1], context=Tensor[L,C],
             y=Tensor[C',lat_f,lat_h,lat_w], dit_cond_dict, max_seq_len,
             lat_f, tpf)  —— 直接喂 WanModelWithMemoryV5.forward。
    """
    import math
    import numpy as _np
    import torchvision.transforms.functional as TF
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.cam_utils import (
        compute_relative_poses, interpolate_camera_poses,
        get_plucker_embeddings, get_Ks_transformed,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    max_area = MAX_AREA_CONFIGS[args.size]
    vae_stride = wan_i2v.vae_stride
    patch_size = wan_i2v.patch_size

    img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)  # [3,H,W] in [-1,1]
    F = args.frame_num
    h_img, w_img = img_t.shape[1:]
    aspect_ratio = h_img / w_img
    lat_h = round(_np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(_np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    lat_f = (F - 1) // vae_stride[0] + 1
    max_seq_len = lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / wan_i2v.sp_size)) * wan_i2v.sp_size
    tpf = (lat_h // patch_size[1]) * (lat_w // patch_size[2])  # tokens / latent frame

    # ---- msk + y（i2v 条件；逐行对齐 generate:311-401）----
    msk = torch.ones(1, F, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    # ---- 文本 context ----
    wan_i2v.text_encoder.model.to(device)
    context = wan_i2v.text_encoder([args.prompt], device)

    # ---- cam preparation（plucker；best-effort，失败回退 dit_cond_dict=None）----
    dit_cond_dict = None
    try:
        c2ws = poses_c.astype(_np.float32)
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        eff_F = min(F, len_c2ws)
        c2ws = c2ws[:eff_F]
        wasd_action = acts_c.astype(_np.float32)[:eff_F] if wan_i2v.control_type == 'act' else None

        Ks = torch.from_numpy(intr_c.astype(_np.float32)).float()
        Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                                height_resize=h, width_resize=w, height_final=h, width_final=w)
        Ks = Ks[0]
        len_c2ws = len(c2ws)
        c2ws_infer = interpolate_camera_poses(
            src_indices=_np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=_np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)
        c2ws_infer = c2ws_infer.to(device)
        Ks = Ks.to(device)
        if wasd_action is not None:
            wasd_action = torch.from_numpy(wasd_action[::4]).float().to(device)
        only_rays_d = wasd_action is not None
        from einops import rearrange
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h), c2=int(w // lat_w))
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w).to(wan_i2v.param_dtype)
        if wasd_action is not None:
            wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1)
            wasd_action_tensor = rearrange(
                wasd_action_tensor, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w))
            wasd_action_tensor = wasd_action_tensor[None, ...]
            wasd_action_tensor = rearrange(
                wasd_action_tensor, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w).to(wan_i2v.param_dtype)
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)
        dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("档 A 捕获：plucker 构造失败（%s）→ 退回 dit_cond_dict=None（无 cam 注入，"
                       "略损忠实度但不致命）。", exc)
        dit_cond_dict = None

    # ---- i2v 条件 latent y（generate:392-401）----
    y = wan_i2v.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(img_t[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, h, w),
        ], dim=1).to(device)
    ])[0]
    y = torch.concat([msk, y])

    # ---- 干净记忆帧 clip latent x0（VAE encode GT clip 帧）----
    cf = torch.from_numpy(_np.ascontiguousarray(clip_frames)).float()  # [F,3,H,W] in [-1,1]
    cf = cf.permute(1, 0, 2, 3)                                         # [3,F,H,W]
    cf = torch.nn.functional.interpolate(
        cf.unsqueeze(0), size=(cf.shape[1], h, w), mode='trilinear', align_corners=False
    ).squeeze(0) if (cf.shape[2] != h or cf.shape[3] != w) else cf
    x0 = wan_i2v.vae.encode([cf.to(device)])[0]                        # [16,lat_f,lat_h,lat_w]

    # ---- 最小噪声 timestep（采样调度里最小的 t）+ flow-matching 加极少噪声 ----
    # 理由（Step 43）：选采样调度的最小 t（接近干净帧），flow-matching x_t=(1-σ)x0+σ·noise，
    # σ=t_min/num_train_timesteps；t_min 最小 → σ≈0 → x_t≈干净记忆帧 → hidden 反映干净内容。
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=wan_i2v.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.capture_steps, device=device, shift=args.sample_shift)
    t_min = scheduler.timesteps[-1]                                    # 调度里最小 t
    sigma = float(t_min) / float(wan_i2v.num_train_timesteps)
    noise = torch.randn(x0.shape, generator=torch.Generator(device=device).manual_seed(args.seed),
                        device=device, dtype=torch.float32)
    x_t = (1.0 - sigma) * x0.float() + sigma * noise                  # [16,lat_f,lat_h,lat_w]

    return dict(
        x_t=x_t, t_min=torch.stack([t_min]).to(device),
        context=[context[0]], y=[y], dit_cond_dict=dit_cond_dict,
        max_seq_len=max_seq_len, lat_f=lat_f, tpf=tpf,
    )


def _capture_layer_kv(wan_i2v, model, frames, ep, mem_frame: int,
                      args, device, memory_layers: List[int]) -> Dict[int, torch.Tensor]:
    """对「记忆帧 mem_frame」跑一次低噪 forward，逐注入层捕获 self_attn 输入 hidden。

    Returns:
        cache: {layer_index: [1, tpf, dim]}（CPU，省 GPU 显存；注入时由 injection.py 对齐 device）。
    """
    # clip 切片（以 mem_frame 起始；seg_start 通常 == mem_frame，近末尾时回退对齐）
    poses_c, acts_c, intr_c, seg_start = _frame_to_clip_slice(ep, mem_frame, args.frame_num)
    img = _frame_to_pil(frames[seg_start])  # 记忆帧 clip 首帧（i2v 条件 = 记忆帧本身）
    T = frames.shape[0]
    clip_frames = frames[seg_start:seg_start + args.frame_num]  # [F',3,H,W]
    # 不足 frame_num 时用末帧 pad（与 _frame_to_clip_slice 的 pose pad 思路一致）
    if clip_frames.shape[0] < args.frame_num:
        pad_n = args.frame_num - clip_frames.shape[0]
        clip_frames = np.concatenate(
            [clip_frames, np.tile(clip_frames[-1:], (pad_n, 1, 1, 1))], axis=0)

    inp = _build_capture_inputs(wan_i2v, img, clip_frames, poses_c, acts_c, intr_c, args, device)
    lat_f, tpf = inp["lat_f"], inp["tpf"]
    # 记忆帧在 clip 内的 latent-帧索引（vae 时间步长子采样）
    mem_lat_idx = (mem_frame - seg_start) // wan_i2v.vae_stride[0]
    mem_lat_idx = int(max(0, min(mem_lat_idx, lat_f - 1)))
    tok_lo, tok_hi = mem_lat_idx * tpf, (mem_lat_idx + 1) * tpf

    # ---- 注册 forward_pre_hook：抓每个注入层 self_attn 的输入 x（args[0]）----
    cache: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_i: int):
        def _hook(module, hook_args):
            x_in = hook_args[0]                       # [1, max_seq_len, dim]（modulated hidden）
            # 只取记忆帧那一 latent-帧的 token 块 → [1, tpf, dim]
            cache[layer_i] = x_in[:, tok_lo:tok_hi, :].detach().to("cpu").contiguous()
            return None
        return _hook

    for i in memory_layers:
        handles.append(model.blocks[i].self_attn.register_forward_pre_hook(_make_hook(i)))

    # ---- 一次低噪 forward（mem_source=encoder + memory_latents=None → 所有层 set_memory(None)，
    #      即纯骨干 forward；hook 抓真·hidden）----
    prev_src = model._mem_source
    model.set_mem_source("encoder")
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=wan_i2v.param_dtype):
            model(
                [inp["x_t"]], t=inp["t_min"], context=inp["context"],
                seq_len=inp["max_seq_len"], y=inp["y"],
                dit_cond_dict=inp["dit_cond_dict"], memory_latents=None,
            )
    finally:
        for hd in handles:
            hd.remove()
        model.set_mem_source(prev_src)

    if not cache:
        raise RuntimeError("档 A 捕获失败：未捕获到任何层的 hidden（hook 未触发？）。")
    any_shape = tuple(next(iter(cache.values())).shape)
    logger.info("档 A 捕获 mem_frame=%d（clip seg_start=%d, lat_idx=%d）：%d 层 KV-cache，"
                "每层 shape=%s", mem_frame, seg_start, mem_lat_idx, len(cache), any_shape)
    return cache


# ===========================================================================
# 注入上下文（按臂切来源开关；finally 一律复位 encoder，杜绝串味）
# ===========================================================================

def _high_v5_model(wan_i2v, model, inject_high: bool):
    """返回需与 low 同步配置 mem_source 的 high V5 模型；否则 None。

    W-1：仅当 `--inject_high` 且 high_noise_model 已转 V5（WanModelWithMemoryV5、与 low 同有
    set_mem_source / set_layer_kv_cache 开关）时返回 high；否则（默认 low-only，或 high 仍是原始
    WanModel）返回 None。**inject_high=False 时恒返回 None**——保证默认 low-only 路径逐位不变。
    """
    if not inject_high:
        return None
    from memory_module.v5_incontext.model_with_memory_v5 import WanModelWithMemoryV5
    high = getattr(wan_i2v, "high_noise_model", None)
    if high is not None and high is not model and isinstance(high, WanModelWithMemoryV5):
        return high
    return None


@contextmanager
def _arm_injection(wan_i2v, model, arm: str,
                   cache_ideal: Optional[Dict[int, torch.Tensor]],
                   cache_random: Optional[Dict[int, torch.Tensor]],
                   oracle_latent: Optional[torch.Tensor],
                   inject_high: bool = False):
    """generate 期间按臂设置 memory 来源；退出时复位为 'encoder' + 清缓存/还原 patch。

    W-1：`--inject_high` 下 high 也转了 V5，且 `_patch_memory_latents` 会把 oracle_latent 绑进
    **所有**已转 V5 模型（含 high）的 forward。若不同步切 high 的来源，high 会对注入的内容走惰性
    encoder 分支，污染该臂贡献。故任一 ideal_* 臂在 inject_high 下都把 high 切到**同一来源**，
    确保没有任何模型悄悄走 encoder 路径；finally 把 low/high 双双复位 encoder + 清 cache。
    **inject_high=False 时 `high is None`，全部 high 分支跳过，行为与现状逐位等价。**
    """
    if arm == "off":
        # 不注入：保持 encoder + 不 patch memory_latents → forward 所有层 set_memory(None)
        yield
        return
    high = _high_v5_model(wan_i2v, model, inject_high)
    if arm in ("ideal_A", "random_A"):
        cache = cache_ideal if arm == "ideal_A" else cache_random
        model.set_layer_kv_cache(cache)
        model.set_mem_source("ideal_A")
        # W-1：high 未捕获逐层 cache（cache 只对 low 捕获）→ 给 high 设 cache=None 并切 ideal_A 源，
        #      使其 forward 走 ideal_A 分支对各层 set_memory(None)（不注入、不经 encoder），而非
        #      对 None memory_latents 走 encoder 分支。
        if high is not None:
            high.set_layer_kv_cache(None)
            high.set_mem_source("ideal_A")
        try:
            yield
        finally:
            model.set_mem_source("encoder")
            model.set_layer_kv_cache(None)
            if high is not None:
                high.set_mem_source("encoder")
                high.set_layer_kv_cache(None)
        return
    if arm == "ideal_B":
        # 档 B：所有层共享 pool(patch_embedding(zero-padded latent))，经 _patch_memory_latents
        # 把 oracle_latent 绑进 forward 的 memory_latents，src=ideal_B 触发 _encode_ideal_B。
        model.set_mem_source("ideal_B")
        # W-1：high 也被 _patch_memory_latents 绑进 oracle_latent → 必须把 high 也切到 ideal_B，
        #      让它走 _encode_ideal_B(oracle_latent) 而非惰性 encoder 路径。
        if high is not None:
            high.set_mem_source("ideal_B")
        _patch_memory_latents(wan_i2v, oracle_latent)
        try:
            yield
        finally:
            _unpatch_memory_latents(wan_i2v)
            model.set_mem_source("encoder")
            if high is not None:
                high.set_mem_source("encoder")
        return
    raise ValueError(f"未知臂 {arm!r}")


def _generate_arm(wan_i2v, model, arm, cache_ideal, cache_random, oracle_latent,
                  pt: RevisitPoint, ep, base_img, args, device, rng,
                  tmp_action_dir: str) -> Optional[np.ndarray]:
    """对单个 (case, 臂) 跑一次 diffusion 生成，返回 [3,F,H,W] 或 None。"""
    from wan.configs import MAX_AREA_CONFIGS

    poses_c, acts_c, intr_c, _seg = _frame_to_clip_slice(ep, pt.query_frame, args.frame_num)
    img = _weaken_image(base_img, args.weaken_first_frame, rng)
    np.save(os.path.join(tmp_action_dir, "poses.npy"), poses_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "action.npy"), acts_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "intrinsics.npy"), intr_c.astype(np.float32))
    max_area = MAX_AREA_CONFIGS[args.size]

    with _arm_injection(wan_i2v, model, arm, cache_ideal, cache_random, oracle_latent,
                        inject_high=args.inject_high):
        video = wan_i2v.generate(
            args.prompt, img, action_path=tmp_action_dir, max_area=max_area,
            frame_num=args.frame_num, shift=args.sample_shift, sample_solver="unipc",
            sampling_steps=args.num_inference_steps, guide_scale=args.guide_scale,
            seed=args.seed, offload_model=True,
        )
    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    return video  # [3,F,H,W]


# ===========================================================================
# per_window.csv
# ===========================================================================

_CSV_FIELDS = [
    "episode_id", "query_frame", "first_visit_frame", "arm",
    "weaken_first_frame", "video_path", "gt_first_visit_png",
    "dino_max", "dino_mean", "dino_last",      # DINO（主判据）
    "ssim_max", "ssim_mean", "ssim_last",      # SSIM（对照）
]


def _append_csv(run_dir: str, record: Dict) -> None:
    import csv
    csv_path = os.path.join(run_dir, "per_window.csv")
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 per_window.csv 失败: %s", exc)


def _score_and_record(all_records, run_dir, args, ep_id, pt, arm, video, gt_first,
                      mp4_path, gt_png_path, device) -> Optional[float]:
    """算 DINO + SSIM → record → append + 落盘。返回 dino_mean（判据用）。"""
    metrics = _revisit_consistency(video, gt_first, device=device)
    record = {
        "episode_id": ep_id, "query_frame": pt.query_frame,
        "first_visit_frame": pt.first_visit_frame, "arm": arm,
        "weaken_first_frame": args.weaken_first_frame,
        "video_path": mp4_path, "gt_first_visit_png": gt_png_path,
        "dino_max": metrics.get("revisit_consistency_dino_max"),
        "dino_mean": metrics.get("revisit_consistency_dino_mean"),
        "dino_last": metrics.get("revisit_consistency_dino_last"),
        "ssim_max": metrics.get("revisit_consistency_max"),
        "ssim_mean": metrics.get("revisit_consistency_mean"),
        "ssim_last": metrics.get("revisit_consistency_last"),
    }
    logger.info("ep=%s q=%d [%s] dino_mean=%s ssim_mean=%s",
                ep_id, pt.query_frame, arm, record["dino_mean"], record["ssim_mean"])
    all_records.append(record)
    _append_csv(run_dir, record)
    dm = record["dino_mean"]
    return float(dm) if dm is not None else None


# ===========================================================================
# 判据 / summary
# ===========================================================================

def _verdict(all_records: List[Dict], arms: List[str], margin: float,
             run_dir: str) -> str:
    """逐 case 三臂表 + 均值 + GO/NO-GO 判决，打印并写 summary.md。"""
    # 按 case (ep, q) 聚合每臂 dino_mean
    cases: Dict[tuple, Dict[str, float]] = {}
    for r in all_records:
        key = (r["episode_id"], r["query_frame"])
        dm = r["dino_mean"]
        if dm is None:
            continue
        cases.setdefault(key, {})[r["arm"]] = float(dm)

    lines: List[str] = []
    lines.append("# v5-KV 理想注入诊断（S-V4 / Step 43）—— GO/NO-GO\n")
    lines.append(f"判据: GO ⇔ ideal_A 的 DINO 均值 > off + {margin} **且** > random_A + {margin}\n")
    lines.append("（GO = 情况乙 通道+骨干能用 → 修 encoder，不上 LoRA-q；"
                 "NO-GO = 情况甲 通道用不了 → 阶梯 ② LoRA-q / pivot latent-concat）\n")

    header = "| episode | query | " + " | ".join(arms) + " | ideal_A−off | ideal_A−random_A |"
    sep = "|" + "---|" * (len(arms) + 4)
    lines.append("\n## 逐 case（frame-aligned DINO mean）\n")
    lines.append(header)
    lines.append(sep)

    means: Dict[str, List[float]] = {a: [] for a in arms}
    for key in sorted(cases.keys()):
        row = cases[key]
        cells = []
        for a in arms:
            v = row.get(a)
            cells.append("nan" if v is None else f"{v:.4f}")
            if v is not None:
                means[a].append(v)
        d_off = (row.get("ideal_A", float("nan")) - row.get("off", float("nan")))
        d_rnd = (row.get("ideal_A", float("nan")) - row.get("random_A", float("nan")))
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) +
                     f" | {d_off:+.4f} | {d_rnd:+.4f} |")

    mean_off = float(np.mean(means["off"])) if means.get("off") else float("nan")
    mean_ideal = float(np.mean(means["ideal_A"])) if means.get("ideal_A") else float("nan")
    mean_rnd = float(np.mean(means["random_A"])) if means.get("random_A") else float("nan")

    lines.append("\n## 均值\n")
    for a in arms:
        mv = float(np.mean(means[a])) if means.get(a) else float("nan")
        lines.append(f"- {a}: {mv:.4f}  (n={len(means.get(a, []))})")

    go = (
        ("ideal_A" in arms and "off" in arms and "random_A" in arms)
        and (mean_ideal == mean_ideal)  # not nan
        and (mean_off == mean_off) and (mean_rnd == mean_rnd)
        and (mean_ideal > mean_off + margin)
        and (mean_ideal > mean_rnd + margin)
    )
    verdict = "GO" if go else "NO-GO"
    route = ("通道+骨干能用（情况乙）→ 修 encoder（理想 KV 当蒸馏目标 / 改训练目标），不上 LoRA-q"
             if go else
             "通道用不了（情况甲）→ 进 OP-5 阶梯 ② backbone q(/k) LoRA / pivot latent-concat")
    lines.append("\n## 判决\n")
    lines.append(f"**{verdict}** — ideal_A={mean_ideal:.4f} vs off={mean_off:.4f} "
                 f"(Δ={mean_ideal - mean_off:+.4f}) vs random_A={mean_rnd:.4f} "
                 f"(Δ={mean_ideal - mean_rnd:+.4f})，margin={margin}")
    lines.append(f"\n**路由**：{route}")

    summary = "\n".join(lines) + "\n"
    try:
        with open(os.path.join(run_dir, "summary.md"), "w") as fh:
            fh.write(summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 summary.md 失败: %s", exc)
    print("\n" + summary)
    logger.info("verdict=%s", verdict)
    return verdict


# ===========================================================================
# 定性渲染（S-V4 / Step 43 定性部分）：完整多 clip 自回归长视频，off vs ideal_A
#
#   每个 revisit case 渲一条 ~25s 长视频（默认 5 clips × frame_num=81 @ 16fps = 405 帧），
#   自回归（上一 clip 末帧 → 下一 clip current_img），镜像 infer_v5 的多 clip 循环
#   （seed=seed+clip_idx、sample_steps=num_inference_steps、sample_shift、guide_scale、size、
#    prompt 同 diag 默认）。两臂并排眼看「回到旧地」的重访 clip（最后一个 clip）：
#     · off     ：纯自回归无注入（mem_source=encoder + memory_latents=None → 各层 set_memory(None)）。
#     · ideal_A ：对该 case 的 GT 首访帧（= 定量 ideal_A 同一帧）用 _capture_layer_kv 捕获一次
#                 逐层真实 KV（固定低噪步，复用定量现成逻辑），贯穿全部 clip 注入。
#   纯出视频，不算 DINO / 不判 GO/NO-GO。复用定量的模型加载 / _capture_layer_kv / _arm_injection /
#   _decode_episode_video / _enumerate_cases / _save_video，不重造轮子。
# ===========================================================================

def _render_case_qual(wan_i2v, model, memory_layers, ep, ep_id, pt, frames,
                      args, device, videos_root, qual_arms, num_clips):
    """对单个 (episode, revisit case) 渲完整多 clip 自回归长视频（每臂一条 long_video.mp4）。

    产出：videos/<ep_id>/q<query_frame>/long_video_<arm>.mp4（405 帧 @ args.fps）。

    长视频窗口锚定：使重访 query_frame 落在**最后一个 clip**——这样并排观看时，最后一个 clip
    （局部帧 (num_clips-1)*frame_num : num_clips*frame_num，5×81 时即帧 324-404）正是模型「回到
    旧地」的重访段，便于眼看 ideal_A 是否比 off 更忠实于 GT 首访场景。
    """
    from wan.configs import MAX_AREA_CONFIGS
    max_area = MAX_AREA_CONFIGS[args.size]
    T = ep.poses.shape[0]
    frame_num = args.frame_num
    total_frames = num_clips * frame_num

    q_dir = os.path.join(videos_root, ep_id, f"q{pt.query_frame}")
    os.makedirs(q_dir, exist_ok=True)

    # 留 GT 首访帧 png 供并排对照（与定量 _process_case 同口径）
    try:
        _save_frame_png(frames[int(pt.first_visit_frame)],
                        os.path.join(q_dir, "gt_first_visit.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("定性 ep=%s q=%d：存 gt_first_visit.png 失败（非致命）: %s",
                       ep_id, pt.query_frame, exc)

    # ---- 锚定 5-clip 长视频窗口：使 query_frame 落在最后一个 clip ----
    win_start = pt.query_frame - (num_clips - 1) * frame_num
    win_start = int(max(0, min(win_start, max(0, T - total_frames))))
    logger.info(
        "定性 ep=%s q=%d：长视频窗口 win_start=%d，%d clips × %d 帧 = %d 帧（first_visit=%d）",
        ep_id, pt.query_frame, win_start, num_clips, frame_num, total_frames,
        pt.first_visit_frame,
    )

    # ---- ideal_A：对 GT 首访帧捕获一次逐层真实 KV（与定量 ideal_A 同一帧、同低噪步）----
    cache_ideal = None
    if "ideal_A" in qual_arms:
        # 捕获前确保 model / vae 在 device（generate 的 offload 可能把它们搬到 CPU）
        model.to(device)
        wan_i2v.vae.model.to(device)
        fi = int(pt.first_visit_frame)
        if 0 <= fi < T:
            cache_ideal = _capture_layer_kv(
                wan_i2v, model, frames, ep, fi, args, device, memory_layers)
        else:
            logger.warning("定性 ep=%s q=%d：first_visit_frame=%d 越界 → 跳过 ideal_A 臂",
                           ep_id, pt.query_frame, fi)

    for arm in qual_arms:
        if arm == "ideal_A" and cache_ideal is None:
            continue
        out_path = os.path.join(q_dir, f"long_video_{arm}.mp4")
        if os.path.exists(out_path):
            logger.info("定性 ep=%s q=%d [%s]：long_video 已存在 → 跳过", ep_id, pt.query_frame, arm)
            continue

        # 起始首帧 = 窗口首帧（自然画质 demo，不弱化；镜像 infer_v5 episode 模式 frames[0]）
        current_img = _frame_to_pil(frames[win_start])
        clip_videos: List[np.ndarray] = []

        for clip_idx in range(num_clips):
            clip_start = win_start + clip_idx * frame_num
            clip_end = clip_start + frame_num
            if clip_end <= T:
                poses_c = ep.poses[clip_start:clip_end].astype(np.float32)
                acts_c = ep.actions[clip_start:clip_end].astype(np.float32)
                intr_c = ep.intrinsics[clip_start:clip_end].astype(np.float32)
            else:
                # 数据不足：取末尾 frame_num 帧 + 末帧 pad（对齐 infer_v5 / _frame_to_clip_slice fallback）
                poses_c, acts_c, intr_c, _ = _frame_to_clip_slice(ep, clip_start, frame_num)
                poses_c = poses_c.astype(np.float32)
                acts_c = acts_c.astype(np.float32)
                intr_c = intr_c.astype(np.float32)

            tmp_dir = tempfile.mkdtemp(
                prefix=f"v5_qual_{ep_id}_q{pt.query_frame}_{arm}_c{clip_idx}_")
            np.save(os.path.join(tmp_dir, "poses.npy"), poses_c)
            np.save(os.path.join(tmp_dir, "action.npy"), acts_c)
            np.save(os.path.join(tmp_dir, "intrinsics.npy"), intr_c)

            # 注入排期：ideal_A **贯穿全部 clip 注入**（"can't hurt, might help"——冻结骨干能忽略
            # 无关记忆；非 OPEN 决策，无 PENDING 依赖）。如日后要改成「只在重访 clip 注入」，把下方
            # _arm_injection 的 arm 在 clip_idx < num_clips-1 时传 "off" 即可（off 臂不注入）。
            try:
                with _arm_injection(wan_i2v, model, arm, cache_ideal, None, None,
                                    inject_high=args.inject_high):
                    video = wan_i2v.generate(
                        args.prompt, current_img, action_path=tmp_dir, max_area=max_area,
                        frame_num=frame_num, shift=args.sample_shift, sample_solver="unipc",
                        sampling_steps=args.num_inference_steps, guide_scale=args.guide_scale,
                        seed=args.seed + clip_idx, offload_model=True,
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if video is None:
                logger.warning("定性 ep=%s q=%d [%s] clip %d/%d：生成 None，终止该臂",
                               ep_id, pt.query_frame, arm, clip_idx + 1, num_clips)
                break
            if isinstance(video, torch.Tensor):
                video = video.detach().cpu().float().numpy()
            video = np.asarray(video, dtype=np.float32)  # [3,F,H,W] in [-1,1]
            clip_videos.append(video)

            # 自回归链接：本 clip 末帧 → 下一 clip 首帧（镜像 infer_v5）
            last_chw = video[:, -1]                                    # [3,H,W]
            last_hwc = (last_chw.transpose(1, 2, 0) * 127.5 + 127.5
                        ).clip(0, 255).astype(np.uint8)               # [H,W,3] uint8
            current_img = Image.fromarray(last_hwc)
            logger.info("定性 ep=%s q=%d [%s]：clip %d/%d done", ep_id, pt.query_frame,
                        arm, clip_idx + 1, num_clips)

        if not clip_videos:
            logger.warning("定性 ep=%s q=%d [%s]：无 clip 生成，跳过保存",
                           ep_id, pt.query_frame, arm)
            continue
        full_video = np.concatenate(clip_videos, axis=1)              # [3, total_F, H, W]
        _save_video(full_video, out_path, fps=args.fps)
        logger.info("定性 ep=%s q=%d [%s]：长视频已存 → %s（%d 帧 @ %dfps）",
                    ep_id, pt.query_frame, arm, out_path, full_video.shape[1], args.fps)


def _render_qualitative(wan_i2v, model, memory_layers, ep_groups, ep_ids, args,
                        device, height, width, videos_root, min_time_gap_frames):
    """定性渲染主驱动：枚举 revisit case（复用 _enumerate_cases）→ 分片 → 逐 case 渲长视频。

    分片口径与定量完全一致（按 case 全局序号取模），让定性也能 6 卡并行；
    shard_count<=1 时处理全部 case。每个 episode 只解码一次（_decode_episode_video，
    定性不需要 VAE encode 全 episode，故不调 _decode_episode）。
    """
    _requested = [a.strip() for a in args.qual_arms.split(",") if a.strip()]
    if "random_A" in _requested:
        # 定性暂不渲 random_A：本路径未捕获错帧 KV-cache（省算力，task 默认关）；若直接放行会
        # 用 None cache 注入 → 产出与 off 等价却标 random_A 的误导视频。故显式剔除并告警。
        logger.warning("定性渲染暂不支持 random_A（未捕获错帧 KV-cache）→ 已从 qual_arms 剔除。")
    qual_arms = [a for a in _requested if a in ("off", "ideal_A")]
    if not qual_arms:
        qual_arms = ["off", "ideal_A"]
    # off 在前（off 臂依赖复位态为 encoder；ideal_A 的 _arm_injection finally 会复位 encoder）
    _order = ("off", "ideal_A")
    qual_arms = [a for a in _order if a in qual_arms]
    num_clips = max(1, args.num_clips_qual)
    logger.info("定性渲染：qual_arms=%s num_clips=%d frame_num=%d fps=%d",
                qual_arms, num_clips, args.frame_num, args.fps)

    ordered_cases, ep_cache = _enumerate_cases(ep_ids, ep_groups, args, min_time_gap_frames)
    if not ordered_cases:
        logger.error("定性渲染：无 revisit case，退出。")
        return

    if args.shard_count > 1:
        sel = [(gi, e, p) for gi, (e, p) in enumerate(ordered_cases)
               if gi % args.shard_count == args.shard_index]
    else:
        sel = [(gi, e, p) for gi, (e, p) in enumerate(ordered_cases)]
    logger.info("定性渲染：case 全集 %d，本分片处理 %d（shard %d/%d，max_cases=%d 已在切分前应用）",
                len(ordered_cases), len(sel), args.shard_index, args.shard_count, args.max_cases)
    if not sel:
        logger.error("定性渲染：shard %d/%d 分到 0 个 case，退出。",
                     args.shard_index, args.shard_count)
        return

    from collections import OrderedDict
    by_ep: "OrderedDict[str, List]" = OrderedDict()
    for _gi, ep_id, pt in sel:
        by_ep.setdefault(ep_id, []).append(pt)

    for ep_id, pts in by_ep.items():
        ep = ep_cache[ep_id]
        logger.info("定性 Episode %s：T=%d，本分片 %d 个 case", ep_id, ep.poses.shape[0], len(pts))
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
        except Exception as exc:  # noqa: BLE001
            logger.warning("定性 episode %s 解码失败: %s；跳过", ep_id, exc)
            continue
        for pt in pts:
            try:
                _render_case_qual(wan_i2v, model, memory_layers, ep, ep_id, pt,
                                  frames, args, device, videos_root, qual_arms, num_clips)
            except Exception as exc:  # noqa: BLE001
                logger.exception("定性 case 渲染失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                continue
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ===========================================================================
# episode 解码 + 单 case 处理（从 main 抽出，供单卡 / 分片两条路径共用）
# ===========================================================================

def _decode_episode(wan_i2v, ep, args, height, width, device):
    """解码 episode video + VAE encode → (frames[T,3,H,W], latents_per_frame[T,z,h,w])。

    与原 main 内联逻辑逐字一致；抽成函数供单卡 / 分片路径共用（行为不变）。
    """
    T = ep.poses.shape[0]
    frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
    latents_full = _vae_encode_batched(wan_i2v.vae, frames, device=device, batch_frames=8)
    latents_per_frame = _expand_latents_to_frames(latents_full, T)  # [T,z,h,w]
    del latents_full
    return frames, latents_per_frame


def _process_case(wan_i2v, model, memory_layers, ep, ep_id, pt, frames,
                  latents_per_frame, args, device, rng, videos_root, run_dir,
                  all_records, arms, need_ideal):
    """处理单个 (episode, revisit case)：档 A 捕获 + 逐臂生成 + 打分落盘。

    与原 main 内层 `for pt in points` 的 try 块体逐字一致（抽出以供单卡 / 分片共用），
    成功则向 all_records 追加各臂记录；异常由调用方捕获（保留原 case 级 try/except 语义）。
    """
    T = ep.poses.shape[0]
    q_dir = os.path.join(videos_root, ep_id, f"q{pt.query_frame}")
    os.makedirs(q_dir, exist_ok=True)
    gt_first = frames[pt.first_visit_frame]  # [3,H,W]
    gt_png_path = os.path.join(q_dir, "gt_first_visit.png")
    _save_frame_png(gt_first, gt_png_path)

    _pc, _ac, _ic, seg_start = _frame_to_clip_slice(ep, pt.query_frame, args.frame_num)
    base_img = _frame_to_pil(frames[seg_start])

    # ---- 档 A 捕获（ideal/random 各一份逐层 KV-cache）----
    cache_ideal = cache_random = None
    oracle_latent = None
    if need_ideal:
        # 捕获前确保 low_noise_model / vae 在 device（generate 的 offload 可能搬走）
        model.to(device)
        wan_i2v.vae.model.to(device)
        fi = int(pt.first_visit_frame)
        if 0 <= fi < T and "ideal_A" in arms:
            cache_ideal = _capture_layer_kv(
                wan_i2v, model, frames, ep, fi, args, device, memory_layers)
        if "random_A" in arms:
            rfi = _pick_random_hist_frame(pt, T, rng)
            if rfi is not None:
                cache_random = _capture_layer_kv(
                    wan_i2v, model, frames, ep, int(rfi), args, device, memory_layers)
            else:
                logger.warning("ep=%s q=%d：random_A 取不到错帧 → 跳过该臂",
                               ep_id, pt.query_frame)
    if "ideal_B" in arms:
        fi = int(pt.first_visit_frame)
        if 0 <= fi < T:
            oracle_latent = latents_per_frame[fi].unsqueeze(0).contiguous()  # [1,z,h,w]

    # ---- 逐臂生成 + 打分 ----
    for arm in arms:
        if arm == "ideal_A" and cache_ideal is None:
            continue
        if arm == "random_A" and cache_random is None:
            continue
        if arm == "ideal_B" and oracle_latent is None:
            continue
        mp4_path = os.path.join(q_dir, f"{arm}.mp4")
        if os.path.exists(mp4_path):
            video = _read_video_back(mp4_path)
            if video is not None:
                logger.info("ep=%s q=%d [%s]：mp4 已存在 → 读回重算指标",
                            ep_id, pt.query_frame, arm)
                _score_and_record(all_records, run_dir, args, ep_id, pt, arm,
                                  video, gt_first, mp4_path, gt_png_path, device)
                continue
        _tmp = tempfile.mkdtemp(prefix=f"v5_ideal_{ep_id}_q{pt.query_frame}_{arm}_")
        try:
            video = _generate_arm(
                wan_i2v, model, arm, cache_ideal, cache_random, oracle_latent,
                pt, ep, base_img, args, device, rng, _tmp)
        finally:
            import shutil
            shutil.rmtree(_tmp, ignore_errors=True)
        if video is None:
            logger.warning("ep=%s q=%d [%s]：生成 None，跳过", ep_id, pt.query_frame, arm)
            continue
        _save_video(video, mp4_path, fps=args.fps)
        _score_and_record(all_records, run_dir, args, ep_id, pt, arm,
                          video, gt_first, mp4_path, gt_png_path, device)


def _enumerate_cases(ep_ids, ep_groups, args, min_time_gap_frames):
    """分片用：枚举**完整** revisit case 列表（不解码视频、不消耗 rng）。

    顺序与单卡路径完全一致：episode 按 ep_ids 顺序、episode 内 query 升序
    （_find_revisit_points 已按 query_frame 升序）。**在分片切分之前**应用 --max_cases
    全局上限，保证各分片看到同一全集再切。

    Returns:
        (ordered_cases, ep_cache)
        ordered_cases: List[(ep_id, RevisitPoint)]，按全局处理序，已截断到 max_cases。
        ep_cache:      {ep_id: EpisodeData}，避免处理阶段重复 build。
    """
    ordered_cases: List = []
    ep_cache: Dict = {}
    for ep_id in ep_ids:
        if args.max_cases > 0 and len(ordered_cases) >= args.max_cases:
            break
        ep = build_episode_data(ep_id, ep_groups[ep_id],
                                clip_overlap_frames=args.clip_overlap_frames)
        if ep is None:
            continue
        points = _find_revisit_points(ep, args, min_time_gap_frames)
        if not points:
            logger.warning("Episode %s 无重访点；跳过", ep_id)
            continue
        ep_cache[ep_id] = ep
        for pt in points:
            if args.max_cases > 0 and len(ordered_cases) >= args.max_cases:
                break
            ordered_cases.append((ep_id, pt))
    return ordered_cases, ep_cache


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    args = _parse_args()
    if args.weaken_first_frame == "noise":
        logger.warning("⚠️ F-18: --weaken_first_frame=noise 会摧毁 i2v 锚点 → 各臂指标地板化；"
                       "revisit 评测请用 zero。")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ALL_ARMS]
    if not arms:
        arms = list(DEFAULT_ARMS)
    # 保证臂顺序稳定（off 在前）
    arms = [a for a in ALL_ARMS if a in arms]
    need_ideal = ("ideal_A" in arms) or ("random_A" in arms)

    # ---- 产出目录 ----
    run_name = args.run_name or default_run_name("v5_ideal_diag")
    run_dir = eval_run_dir("v5", run_name, args.tag)
    videos_root = os.path.join(str(run_dir), "videos")
    os.makedirs(videos_root, exist_ok=True)
    log_path = os.path.join(str(run_dir), "diag.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    snapshot_config(run_dir, {k: v for k, v in vars(args).items() if not k.startswith("_")})
    logger.info("ideal_inject_diag run_dir=%s | arms=%s", run_dir, arms)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))
    logger.info("Args: %s", vars(args))

    # ---- episode CSV ----
    ep_filter = None
    if args.episode_ids:
        ep_filter = [s.strip() for s in args.episode_ids.split(",") if s.strip()]
    ep_groups = load_episode_clips(args.dataset_dir, args.metadata, episode_ids_filter=ep_filter)
    ep_ids = list(ep_groups.keys())
    if args.max_episodes > 0 and len(ep_ids) > args.max_episodes:
        ep_ids = ep_ids[:args.max_episodes]
    if not ep_ids:
        logger.error("无 episode 可处理，退出。")
        return

    # ---- 加载 v5 pipeline（复用 eval_v5）----
    wan_i2v = _load_v5_pipeline(args, device)
    model = wan_i2v.low_noise_model  # WanModelWithMemoryV5（W2：只 low 转 V5）
    memory_layers = list(model._memory_layers)

    # ---- 定性渲染路径（--render_qual）：只渲长视频，不算 DINO / 不判 GO/NO-GO，提前返回。----
    #      不开 --render_qual 时此分支跳过，下方定量三臂行为逐位不变（向后兼容硬要求）。
    if args.render_qual:
        _render_qualitative(wan_i2v, model, memory_layers, ep_groups, ep_ids, args,
                            device, height, width, videos_root, min_time_gap_frames)
        logger.info("定性渲染完成。输出目录: %s", run_dir)
        return

    all_records: List[Dict] = []
    run_dir_str = str(run_dir)

    if args.shard_count <= 1:
        # ---- 单卡原始路径（shard_count<=1 → 逐位与改前一致）----
        n_cases = 0
        for ep_id in ep_ids:
            if n_cases >= args.max_cases > 0:
                break
            ep = build_episode_data(ep_id, ep_groups[ep_id],
                                    clip_overlap_frames=args.clip_overlap_frames)
            if ep is None:
                continue
            T = ep.poses.shape[0]
            points = _find_revisit_points(ep, args, min_time_gap_frames)
            if not points:
                logger.warning("Episode %s 无重访点；跳过", ep_id)
                continue
            logger.info("Episode %s: T=%d, 重访点 %d 个", ep_id, T, len(points))

            try:
                frames, latents_per_frame = _decode_episode(
                    wan_i2v, ep, args, height, width, device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
                continue

            for pt in points:
                if n_cases >= args.max_cases > 0:
                    break
                try:
                    _process_case(wan_i2v, model, memory_layers, ep, ep_id, pt, frames,
                                  latents_per_frame, args, device, rng, videos_root,
                                  run_dir_str, all_records, arms, need_ideal)
                    n_cases += 1
                except Exception as exc:  # noqa: BLE001
                    logger.exception("case 处理失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                    continue

            del frames, latents_per_frame
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # ---- 分片路径：先枚举完整 case 全集（max_cases 在切分前应用），再按 case 全局序号取模 ----
        ordered_cases, ep_cache = _enumerate_cases(
            ep_ids, ep_groups, args, min_time_gap_frames)
        logger.info("分片 %d/%d：case 全集 %d 个（max_cases=%d 已在切分前应用）",
                    args.shard_index, args.shard_count, len(ordered_cases), args.max_cases)
        shard_cases = [(gi, ep_id, pt) for gi, (ep_id, pt) in enumerate(ordered_cases)
                       if gi % args.shard_count == args.shard_index]
        logger.info("分片 %d/%d：本分片处理 %d 个 case（全局序号 %% %d == %d）",
                    args.shard_index, args.shard_count, len(shard_cases),
                    args.shard_count, args.shard_index)
        if not shard_cases:
            logger.error("shard %d/%d 分到 0 个 case（全集太小？），退出。",
                         args.shard_index, args.shard_count)
            return

        # 按 episode 分组（保持全局序）→ 每个 episode 只解码一次
        from collections import OrderedDict
        by_ep: "OrderedDict[str, List]" = OrderedDict()
        for _gi, ep_id, pt in shard_cases:
            by_ep.setdefault(ep_id, []).append(pt)

        for ep_id, pts in by_ep.items():
            ep = ep_cache[ep_id]
            T = ep.poses.shape[0]
            logger.info("Episode %s: T=%d, 本分片 %d 个 case", ep_id, T, len(pts))
            try:
                frames, latents_per_frame = _decode_episode(
                    wan_i2v, ep, args, height, width, device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
                continue

            for pt in pts:
                try:
                    _process_case(wan_i2v, model, memory_layers, ep, ep_id, pt, frames,
                                  latents_per_frame, args, device, rng, videos_root,
                                  run_dir_str, all_records, arms, need_ideal)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("case 处理失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                    continue

            del frames, latents_per_frame
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_records:
        logger.error("无任何记录（无 case / 全失败），退出。")
        return

    _verdict(all_records, arms, args.go_margin, str(run_dir))

    logger.info("Done. 输出目录: %s", run_dir)


if __name__ == "__main__":
    main()
