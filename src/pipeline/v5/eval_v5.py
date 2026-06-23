"""
eval_v5.py — v5 in-context KV 记忆注入评测（experiment_design Step 41 / S-V2 第二块）
====================================================================================

复用 `pipeline/eval/oracle_injection.py` 的评测骨架（episode/revisit 点查找、
weaken_first_frame、采样 + VAE decode、DINO cosine 到 GT 首访 PNG、per_window.csv 列结构），
改造为 **v5 + 方案A latent 注入 + 新产出布局**：

  - 模型：`WanModelWithMemoryV5.from_wan_model(base, memory_layers, grid, ...)`，再
    `load_state_dict(memory_encoder.pth, strict=False)`（骨干来自 base 已就位，见
    train_v5「checkpoint 存取约定」）。**默认只把 low_noise_model 转 V5 + 注入**
    （W2，对齐训练：训练只训了 low 的 memory_encoder；high 用未训练对齐的 encoder 注入
    会污染判读，也对齐 v4 oracle_injection 的 low-only 口径）。high_noise_model 默认保持
    原始 WanModel，forward 走原始路径（不 patch、不注入）；仅 --inject_high 显式开启时
    才把 high 也转 V5 并注入。
    （注：grid/encoder_depth/memory_layers 默认以 ckpt 同目录 training_metadata.json
     为准重建，CLI 显式冲突会 raise，见 _resolve_model_config。）
  - **三臂**（每个 revisit query 默认全跑）：
      · off    : memory_latents=None（baseline，纯 i2v）。
      · oracle : 该 query **GT 首访帧的 VAE latent** `[1,16,h,w]` 作 memory_latents
                 （方案A latent 路径，不走 v4 的 pose_emb/visual_emb K/V）。
      · random : 同 episode **非首访** 的随机历史帧 latent 作 memory_latents
                 （confound 对照，排除"注入任意帧都变好"）。
  - **weaken_first_frame 默认 zero**（沿用 F-18 护栏；noise 会摧毁 i2v 场景锚点）。
  - **指标**：复用 oracle_injection 的 `revisit_consistency_dino_{max,mean,last}`（对 GT
    首访 PNG 的 DINOv2 cosine，主判据）+ 非 DINO 三列（SSIM，保留作对照）。DINO 模型
    **只加载一次**（oracle_injection 的全局单例缓存，本脚本直接复用其 `_revisit_consistency`）。
  - **产出布局（新，人性化）**：`eval_run_dir('v5', run_name, tag)`；**三臂视频同夹**
        videos/<episode>/<query>/{off,oracle,random}.mp4 + gt_first_visit.png
    （开一个文件夹即可对比）；`per_window.csv`、`config.yaml`、`eval.log` 落 run 目录。
  - 跑完自动调用 `summarize_eval`（import）生成 summary.md + 追加 INDEX.md。

== 与 oracle_injection.py（v4）的关系 ==
  - **不改 v4 / refs**：本文件 import 复用 v4/eval 的纯函数（episode 加载、GT 重访、
    VAE encode、weaken、DINO/SSIM 指标、视频 IO），但模型装载与注入路径全部用 v5 的。
  - 注入方式：v5 模型 forward 直接吃 `memory_latents`；generate 不传该参数，故用一个
    monkey-patch 把 memory_latents 绑进 low/high noise_model 的 forward（generate 前 set、
    finally clear），等价 v4 的 `_patch_pipeline_memory` 思路但走 latent 路径。

本地无 torch/CUDA/DINO 真跑不动；--help / py_compile 走通即可（真跑待服务器）。
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
import tempfile
from os.path import abspath, dirname, join
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# sys.path（与 oracle_injection.py / train_v5.py 一致）
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
# 复用 oracle_injection.py（v4 eval）的纯脚手架函数（import，不重写、不改 v4）
#   - 重访点结构 + 查找
#   - clip 切片 / 首帧弱化
#   - SSIM + DINO 一致性（DINO 全局单例，只加载一次）
#   - 图像 / 视频 IO
# 复用 retrieval_probe 的 episode 加载 + VAE encode（oracle_injection 也从这里 import）
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
from pipeline.common.paths import (  # noqa: E402
    eval_run_dir,
    snapshot_config,
    default_run_name,
)


# 三臂顺序（off 在前作 baseline；oracle / random 紧随，便于人工对比）
MEMORY_MODES = ("off", "oracle", "random")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "v5 in-context KV 记忆注入评测：三臂 off/oracle/random × 方案A latent，"
            "对 GT 首访帧的 DINO cosine 为主判据。"
        )
    )
    # ---- 模型权重 ----
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="lingbot-world 预训练权重目录（含 low_noise_model / high_noise_model / VAE / T5）")
    p.add_argument("--memory_encoder_ckpt", type=str, required=True,
                   help="训练好的 memory_encoder.pth（train_v5 save_memory_encoder 产出，"
                        "只含 memory_encoder.* 权重）")
    # ---- v5 模型超参（须与训练一致）----
    p.add_argument("--grid", type=int, default=16,
                   help="MemoryEncoder 每帧 grid×grid token（须与训练一致，默认 16）")
    p.add_argument("--encoder_depth", type=int, default=1,
                   help="MemoryEncoder 残差块层数（须与训练一致，默认 1）")
    p.add_argument("--memory_layers", type=str, default=None,
                   help="注入层索引逗号分隔（如 '0,10,20,39'）；None/空=全部 block"
                        "（默认以 training_metadata.json 为准；显式传且与训练冲突会 raise）")
    p.add_argument("--inject_high", action="store_true", default=False,
                   help="默认 False=只把 low_noise_model 转 V5 + 注入（对齐训练，只训了 low 的 "
                        "memory_encoder）。开启时 high_noise_model 也转 V5 并注入（high 用未训练"
                        "对齐的 encoder，仅作消融，会污染主判读，慎用）。")

    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_exp_train.csv")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")

    # ---- 产出（走 paths.py；OUTPUT_ROOT 环境变量可覆盖根）----
    p.add_argument("--run_name", type=str, default=None,
                   help="eval run 名（默认 default_run_name('v5_eval')）")
    p.add_argument("--tag", type=str, default="bank_revisit",
                   help="eval 场景 tag（INDEX 区分用，默认 bank_revisit）")

    # ---- 三臂选择（默认全跑）----
    p.add_argument("--modes", type=str, default=",".join(MEMORY_MODES),
                   help="逗号分隔的注入臂子集（默认 off,oracle,random 全跑）")

    # ---- 首帧弱化（F-18 护栏，默认 zero）----
    p.add_argument("--weaken_first_frame", type=str, default="zero",
                   choices=["noise", "zero", "none"],
                   help="zero=置零中性灰（默认，温和锚点）/ none=不弱化 / "
                        "noise=随机 RGB（已知摧毁 i2v 锚点 → 指标地板化，仅消融）")

    # ---- 重访点判定（复用 retrieval_probe / oracle_injection 口径）----
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

    # ---- 分片（additive：shard_count 默认 1 → 逐字节与改前一致）----
    # 多卡并行 eval 用：每个 shard 跑不同子集，各自写独立 run_dir（不同 --tag），
    # 跑完用 merge_eval_shards.py 合并 per_window.csv。单卡场景不传这两个参数即可。
    p.add_argument("--shard_index", type=int, default=0,
                   help="当前分片索引（0-based），多卡并行 eval 用。默认 0。")
    p.add_argument("--shard_count", type=int, default=1,
                   help="总分片数。默认 1=不分片（逐字节与单进程一致）；"
                        ">1 时本进程只处理 ep_ids[shard_index::shard_count]。")

    return p.parse_args()


# ---------------------------------------------------------------------------
# v5 模型装载：WanI2V + low/high noise_model → WanModelWithMemoryV5 + memory_encoder
# ---------------------------------------------------------------------------

def _convert_to_v5(base_model, memory_layers, grid, encoder_depth, mem_sd, tag: str):
    """单个 WanModel → WanModelWithMemoryV5 + load memory_encoder（strict=False）。

    Args:
        base_model:     已加载预训练权重的 WanModel（low 或 high noise_model）。
        memory_layers:  注入层索引 list 或 None（全部）。
        grid/encoder_depth: 须与训练一致。
        mem_sd:         memory_encoder.pth 的 state_dict（只含 memory_encoder.*）。
        tag:            日志标识（"low"/"high"）。

    Returns:
        WanModelWithMemoryV5（骨干来自 base，memory_encoder 已 load）。
    """
    from memory_module.v5_incontext.model_with_memory_v5 import WanModelWithMemoryV5

    _dev = next(base_model.parameters()).device
    _dtype = next(base_model.parameters()).dtype
    model = WanModelWithMemoryV5.from_wan_model(
        base_model,
        memory_layers=memory_layers,
        grid=grid,
        encoder_depth=encoder_depth,
        skip_to_device=True,   # 保 CPU，下方统一 .to
    )
    # 模型侧 memory_encoder 参数张量全集（断言基准）
    model_me_keys = {n for n, _ in model.named_parameters()
                     if n.startswith("memory_encoder")}
    n_model_me = len(model_me_keys)

    # load memory_encoder（骨干来自 base 已就位）：strict=False，只关心 memory_encoder.* 是否齐
    missing, unexpected = model.load_state_dict(mem_sd, strict=False)
    me_missing = [k for k in missing if k.startswith("memory_encoder")]
    me_unexpected = [k for k in unexpected if k.startswith("memory_encoder")]

    # ---- W1a: 硬断言「确载」----
    #   ckpt 里 memory_encoder.* 键集合 ∩ 模型参数键 = 实际成功载入的张量。
    #   要求：成功载入数 > 0，且 == 模型 memory_encoder 参数张量数（无任何 me_missing）。
    #   任一不满足（全 missing / 空 ckpt / 前缀错）→ raise（绝不只 warning，防假阴性）。
    sd_me_keys = {k for k in mem_sd if k.startswith("memory_encoder")}
    loaded_me_keys = sd_me_keys & model_me_keys
    n_loaded = len(loaded_me_keys)
    if n_loaded == 0:
        raise RuntimeError(
            f"[{tag}] memory_encoder 确载断言失败：成功载入 0 个 memory_encoder.* 张量。"
            f"\n  ckpt 内 memory_encoder.* 键数={len(sd_me_keys)}（ckpt 总键数={len(mem_sd)}）；"
            f"模型 memory_encoder 参数张量数={n_model_me}。"
            "\n  可能原因：ckpt 为空 / 全部键前缀不匹配（如缺 'memory_encoder.' 前缀或键名错位）。"
        )
    if n_loaded != n_model_me or me_missing:
        raise RuntimeError(
            f"[{tag}] memory_encoder 确载断言失败：成功载入 {n_loaded} 个张量，"
            f"但模型 memory_encoder 需要 {n_model_me} 个（缺失 {len(me_missing)} 个）。"
            f"\n  缺失键（前 10）：{me_missing[:10]}"
            "\n  shape/键名与训练不一致，拒绝带病评测。"
        )
    if me_unexpected:
        # ckpt 里有模型不认识的 memory_encoder.* 键 → 结构不符，同样致命。
        raise RuntimeError(
            f"[{tag}] memory_encoder 确载断言失败：ckpt 含 {len(me_unexpected)} 个"
            f" 模型未知的 memory_encoder.* 键（前 10）：{me_unexpected[:10]}。"
            "\n  ckpt 与当前模型结构不一致（grid/depth/层数错配？），拒绝评测。"
        )
    logger.info(
        "[%s] WanModelWithMemoryV5 ready；memory_encoder 确载通过 "
        "(loaded=%d == model_me=%d, missing_me=0, unexpected_me=0).",
        tag, n_loaded, n_model_me,
    )
    model = model.to(device=_dev, dtype=_dtype)
    model.eval().requires_grad_(False)
    return model


def _cli_explicitly_passed(flag: str) -> bool:
    """判断某 CLI flag 是否被用户在命令行**显式**传入（区分 default vs 显式）。

    用于 W1b 配置一致性：仅当用户显式传了某项且与 metadata 冲突时才 raise；
    若用户未传（用默认值）而 metadata 不同，则静默以 metadata 为准（不算冲突）。
    """
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def _resolve_model_config(args):
    """W1b：决定重建模型用的 grid / encoder_depth / memory_layers（list 或 None）。

    读 memory_encoder_ckpt 同目录的 training_metadata.json：
      - 存在 → **以 metadata 的 grid/encoder_depth/memory_layers 为准**重建（覆盖 CLI
        默认，并 log "adopted from training metadata"）；
        若用户 CLI **显式**传了某项且与 metadata 冲突 → raise（防 shape 静默错位）。
      - 缺失 → 用 CLI 值并打 WARN（无法校验，风险自负）。

    Returns:
        (grid, encoder_depth, memory_layers)
    """
    # CLI 侧的 memory_layers（统一成 list 或 None，便于与 metadata 比较）
    cli_memory_layers = None
    if args.memory_layers:
        cli_memory_layers = [int(x) for x in args.memory_layers.split(",") if x != ""]

    meta_path = os.path.join(os.path.dirname(os.path.abspath(args.memory_encoder_ckpt)),
                             "training_metadata.json")
    if not os.path.exists(meta_path):
        logger.warning(
            "W1b: 未找到 training_metadata.json（%s）→ 无法校验配置一致性，"
            "使用 CLI 值 grid=%s encoder_depth=%s memory_layers=%s（风险自负，"
            "若与训练不一致会因 strict=False 静默错位）。",
            meta_path, args.grid, args.encoder_depth, cli_memory_layers,
        )
        return args.grid, args.encoder_depth, cli_memory_layers

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    def _resolve_one(name, cli_val, meta_val):
        # metadata 缺该字段（老 ckpt，metadata 不含 grid/depth/layers）→ 退回 CLI 值并 WARN。
        # 注意：memory_layers=None 是合法值（全部 block），故必须用 'name in meta' 判存在，
        #       不能用 meta_val is None 判缺失。
        if name not in meta:
            logger.warning(
                "W1b: training_metadata.json 缺字段 '%s' → 用 CLI 值 %s（无法校验）。",
                name, cli_val)
            return cli_val
        # 用户显式传了且与 metadata 冲突 → raise
        if _cli_explicitly_passed("--" + name) and cli_val != meta_val:
            raise RuntimeError(
                f"W1b 配置冲突：CLI 显式传 --{name}={cli_val}，但 training_metadata.json "
                f"记录训练时 {name}={meta_val}。两者不一致会导致 pos_emb/每帧 token 数静默"
                f"错位（strict=False 不会报错）→ 假阴性。请移除 --{name} 或改为与训练一致。"
            )
        # 以 metadata 为准
        if cli_val != meta_val:
            logger.info("W1b: %s adopted from training metadata: %s (CLI default was %s)",
                        name, meta_val, cli_val)
        else:
            logger.info("W1b: %s = %s (CLI 与 metadata 一致)", name, meta_val)
        return meta_val

    grid = _resolve_one("grid", args.grid, meta.get("grid"))
    encoder_depth = _resolve_one("encoder_depth", args.encoder_depth,
                                 meta.get("encoder_depth"))
    # memory_layers：metadata 存的是 list 或 None
    memory_layers = _resolve_one("memory_layers", cli_memory_layers,
                                 meta.get("memory_layers"))
    return grid, encoder_depth, memory_layers


def _load_v5_pipeline(args, device):
    """构建 WanI2V，把 noise_model 替换为 WanModelWithMemoryV5 + memory_encoder。

    与 oracle_injection 不同：不调 v4 的 _convert_pipeline_to_memory（那是 v4 K/V 路径），
    直接用 from_wan_model 转换并加载 memory_encoder.pth。

    W2（与训练对齐）：训练只对齐 low_noise_model 的 memory_encoder。因此 **默认只把
    low_noise_model 转 V5 + 注入**；high_noise_model 保持原始 WanModel（forward 走原始
    路径，不 patch、不注入未训练对齐的 encoder，避免污染判读，对齐 v4 oracle_injection
    口径）。仅当 --inject_high 显式开启时才把 high 也转 V5 并注入。

    W1b：重建用的 grid/encoder_depth/memory_layers 由 _resolve_model_config 以训练
    metadata 为准决定（CLI 显式冲突会 raise）。
    """
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS

    grid, encoder_depth, memory_layers = _resolve_model_config(args)

    # memory_encoder 权重（只含 memory_encoder.*）
    mem_sd = torch.load(args.memory_encoder_ckpt, map_location="cpu", weights_only=True)

    cfg = WAN_CONFIGS["i2v-A14B"]
    local_rank = device.index if device.type == "cuda" and device.index is not None else 0
    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
    )

    logger.info("转换 low_noise_model → WanModelWithMemoryV5 ...")
    wan_i2v.low_noise_model = _convert_to_v5(
        wan_i2v.low_noise_model, memory_layers, grid, encoder_depth,
        mem_sd, tag="low")

    if args.inject_high:
        logger.info("--inject_high 开启：转换 high_noise_model → WanModelWithMemoryV5 ...")
        wan_i2v.high_noise_model = _convert_to_v5(
            wan_i2v.high_noise_model, memory_layers, grid, encoder_depth,
            mem_sd, tag="high")
    else:
        logger.info(
            "W2: 默认 low-only 注入（--inject_high 关闭）→ high_noise_model 保持原始 "
            "WanModel，不转 V5、不注入（与训练对齐：只训了 low 的 memory_encoder）。")
    return wan_i2v


# ---------------------------------------------------------------------------
# memory_latents 注入：monkey-patch low/high noise_model 的 forward
#   （generate 不传 memory_latents，故 set/clear 绑进 forward；等价 v4 _patch_pipeline_memory）
# ---------------------------------------------------------------------------

def _v5_injectable_models(wan_i2v):
    """返回应被注入 memory 的 noise_model 列表。

    W2：默认只有 low_noise_model 转了 V5（high 保持原始 WanModel）。仅对**已转 V5**
    （forward 接受 memory_latents 的 WanModelWithMemoryV5）的模型打 patch；用 isinstance
    判定，避免对未转换的原始 high 误打 patch（其 forward 不接受 memory_latents）。
    """
    from memory_module.v5_incontext.model_with_memory_v5 import WanModelWithMemoryV5
    out = []
    for m in (wan_i2v.low_noise_model, wan_i2v.high_noise_model):
        if isinstance(m, WanModelWithMemoryV5):
            out.append(m)
    return out


def _patch_memory_latents(wan_i2v, memory_latents: Optional[torch.Tensor]):
    """把 memory_latents 绑进**已转 V5** 的 noise_model 的 forward（generate 期间生效）。

    W2：默认只 low 转了 V5，故默认只 patch low；high 未转 V5 时不 patch（forward 走原始
    路径）。仅 --inject_high 时 high 也被 patch。

    memory_latents=None 时不打 patch（等价 off 臂；forward 行为与原 WanModelWithMemoryV5
    无 memory 注入完全一致）。
    """
    if memory_latents is None:
        return
    for m in _v5_injectable_models(wan_i2v):
        if getattr(m, "_v5_orig_forward", None) is None:
            m._v5_orig_forward = m.forward

        def _make(model, mem):
            @functools.wraps(model._v5_orig_forward)
            def _patched(x, t, context, seq_len, y=None, dit_cond_dict=None,
                         memory_latents=None):
                _dev = next(model.parameters()).device
                return model._v5_orig_forward(
                    x, t, context, seq_len, y=y, dit_cond_dict=dit_cond_dict,
                    memory_latents=mem.to(_dev),
                )
            return _patched

        m.forward = _make(m, memory_latents)


def _unpatch_memory_latents(wan_i2v):
    """还原 _patch_memory_latents 的 forward 替换（仅已转 V5 的 noise_model）。"""
    for m in _v5_injectable_models(wan_i2v):
        if getattr(m, "_v5_orig_forward", None) is not None:
            m.forward = m._v5_orig_forward
            m._v5_orig_forward = None


# ---------------------------------------------------------------------------
# 三臂 memory_latents 取用（方案A latent 路径）
# ---------------------------------------------------------------------------

def _pick_random_hist_frame(pt: RevisitPoint, T: int,
                            rng: np.random.Generator) -> Optional[int]:
    """random 臂：从同 episode **非首访** 的随机历史帧里抽一帧（confound 对照）。

    候选池 = query 之前的历史帧 [0, query_frame)，排除全部 GT 过去帧（含首访帧）；
    池为空时退回到仅排除首访帧。返回单帧索引或 None。
    """
    hist_end = min(max(int(pt.query_frame), 0), T)
    forbidden = set(pt.gt_past_frames) | {pt.first_visit_frame}
    pool = [i for i in range(hist_end) if i not in forbidden]
    if not pool:
        pool = [i for i in range(hist_end) if i != pt.first_visit_frame]
    if not pool:
        return None
    return int(rng.choice(pool))


def _memory_latents_for_mode(
    mode: str,
    pt: RevisitPoint,
    latents_per_frame: torch.Tensor,   # [T, z_dim, lat_h, lat_w] CPU
    T: int,
    rng: np.random.Generator,
) -> Optional[torch.Tensor]:
    """按臂取 memory_latents [1, z_dim, h, w]（或 None=off）。

    - off    : None（不注入）。
    - oracle : GT 首访帧 latent = latents_per_frame[first_visit_frame] → [1,z,h,w]。
    - random : 非首访随机历史帧 latent → [1,z,h,w]。
    """
    if mode == "off":
        return None
    if mode == "oracle":
        fi = int(pt.first_visit_frame)
        if fi < 0 or fi >= T:
            return None
        return latents_per_frame[fi].unsqueeze(0).contiguous()  # [1,z,h,w]
    if mode == "random":
        fi = _pick_random_hist_frame(pt, T, rng)
        if fi is None:
            return None
        return latents_per_frame[fi].unsqueeze(0).contiguous()
    return None


# ---------------------------------------------------------------------------
# 单臂生成
# ---------------------------------------------------------------------------

def _generate_one_arm(
    wan_i2v,
    mode: str,
    memory_latents: Optional[torch.Tensor],
    pt: RevisitPoint,
    ep,
    base_img,
    args,
    device,
    rng: np.random.Generator,
    tmp_action_dir: str,
) -> Optional[np.ndarray]:
    """对单个 (revisit query, 臂) 跑一次 diffusion 生成，返回 [3,F,H,W] 或 None。"""
    from wan.configs import MAX_AREA_CONFIGS

    poses_c, acts_c, intr_c, _seg_start = _frame_to_clip_slice(
        ep, pt.query_frame, args.frame_num)
    img = _weaken_image(base_img, args.weaken_first_frame, rng)

    np.save(os.path.join(tmp_action_dir, "poses.npy"), poses_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "action.npy"), acts_c.astype(np.float32))
    np.save(os.path.join(tmp_action_dir, "intrinsics.npy"), intr_c.astype(np.float32))

    max_area = MAX_AREA_CONFIGS[args.size]

    _patch_memory_latents(wan_i2v, memory_latents)
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
        _unpatch_memory_latents(wan_i2v)

    if video is None:
        return None
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    return video  # [3,F,H,W]


# ---------------------------------------------------------------------------
# per_window.csv（列对齐 v4，便于历史对比）
# ---------------------------------------------------------------------------

# per_window.csv 列序（与 oracle_injection 对齐 + v5 三臂布局所需列）。
_CSV_FIELDS = [
    "episode_id", "query_frame", "first_visit_frame", "memory_mode",
    "weaken_first_frame", "video_path", "gt_first_visit_png",
    # DINO（主判据）
    "dino_max", "dino_mean", "dino_last",
    # SSIM（对照）
    "max", "mean", "last",
]


def _append_per_window_csv(run_dir: str, record: Dict) -> None:
    """逐臂增量写 per_window.csv（长跑抗崩：算完一个臂立即 append 一行）。"""
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


def _record_arm(
    all_records: List[Dict],
    run_dir: str,
    args,
    ep_id: str,
    pt: RevisitPoint,
    mode: str,
    video: np.ndarray,
    gt_first: np.ndarray,
    mp4_path: str,
    gt_png_path: str,
    device,
) -> None:
    """算指标（SSIM + DINO）→ 构造 record（列对齐 v4）→ append + 增量落盘。"""
    metrics = _revisit_consistency(video, gt_first, device=device)
    record: Dict = {
        "episode_id": ep_id,
        "query_frame": pt.query_frame,
        "first_visit_frame": pt.first_visit_frame,
        "memory_mode": mode,
        "weaken_first_frame": args.weaken_first_frame,
        "video_path": mp4_path,
        "gt_first_visit_png": gt_png_path,
        # SSIM（对照）— oracle_injection metrics key 为 revisit_consistency_{max,mean,last}
        "max": metrics.get("revisit_consistency_max"),
        "mean": metrics.get("revisit_consistency_mean"),
        "last": metrics.get("revisit_consistency_last"),
        # DINO（主判据）— 可能因 DINO 不可用而缺失（graceful）
        "dino_max": metrics.get("revisit_consistency_dino_max"),
        "dino_mean": metrics.get("revisit_consistency_dino_mean"),
        "dino_last": metrics.get("revisit_consistency_dino_last"),
    }
    logger.info("ep=%s q=%d [%s] dino_mean=%s ssim_mean=%s",
                ep_id, pt.query_frame, mode,
                record["dino_mean"], record["mean"])
    all_records.append(record)
    _append_per_window_csv(run_dir, record)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    if args.weaken_first_frame == "noise":
        logger.warning(
            "⚠️ F-18: --weaken_first_frame=noise 会摧毁 i2v 场景锚点 → 三臂指标地板化、"
            "oracle/random/off 无法对比。revisit 评测请用 zero；noise 仅作消融。")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MEMORY_MODES]
    if not modes:
        modes = list(MEMORY_MODES)

    # ---- 产出目录（paths.py 新布局）----
    run_name = args.run_name or default_run_name("v5_eval")
    run_dir = eval_run_dir("v5", run_name, args.tag)
    videos_root = os.path.join(str(run_dir), "videos")
    os.makedirs(videos_root, exist_ok=True)

    # 文件日志（落 run 目录）
    log_path = os.path.join(str(run_dir), "eval.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    snapshot_config(run_dir, {k: v for k, v in vars(args).items()
                              if not k.startswith("_")})
    logger.info("v5 eval run_dir = %s | modes=%s", run_dir, modes)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退 CPU（生成会非常慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))

    logger.info("Args: %s", vars(args))

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

    # ---- 分片（additive：shard_count 默认 1 → ep_ids 不变，逐字节兼容）----
    # 多卡并行 eval：先按 max_episodes 截断全集，再 [shard_index::shard_count] 切片，
    # 保证各 shard 合起来正好是全集的前 max_episodes 个（无重叠、无遗漏）。
    if getattr(args, "shard_count", 1) > 1:
        ep_ids = ep_ids[args.shard_index :: args.shard_count]
        logger.info("eval shard %d/%d: 本分片处理 %d 个 episode",
                    args.shard_index, args.shard_count, len(ep_ids))
        if not ep_ids:
            logger.error("shard %d/%d 分到 0 个 episode（全集太小？），退出。",
                         args.shard_index, args.shard_count)
            return

    # ---- 加载 v5 pipeline ----
    wan_i2v = _load_v5_pipeline(args, device)

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

        # 解码 video + VAE encode（oracle/random latent + 首帧图像 + GT 参照都要用）
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
            latents_full = _vae_encode_batched(wan_i2v.vae, frames, device=device,
                                               batch_frames=8)
            latents_per_frame = _expand_latents_to_frames(latents_full, T)  # [T,z,h,w]
            del latents_full
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码/encode 失败: %s；跳过", ep_id, exc)
            continue

        for pt in points:
            # 单 query 整段（三臂）包 try：单点失败不中断整轮
            try:
                # 三臂同夹：videos/<episode>/<query>/
                q_dir = os.path.join(videos_root, ep_id, f"q{pt.query_frame}")
                os.makedirs(q_dir, exist_ok=True)

                # GT 首访帧（参照 + 人工对比），三臂共享同一张
                gt_first = frames[pt.first_visit_frame]  # [3,H,W]
                gt_png_path = os.path.join(q_dir, "gt_first_visit.png")
                _save_frame_png(gt_first, gt_png_path)

                # query clip 首帧 GT 图（弱化前 base，作 generate 的 img 入口）
                _pc, _ac, _ic, seg_start = _frame_to_clip_slice(
                    ep, pt.query_frame, args.frame_num)
                base_img = _frame_to_pil(frames[seg_start])

                for mode in modes:
                    mp4_path = os.path.join(q_dir, f"{mode}.mp4")

                    # 可续/止损：该臂 mp4 已存在 → 读回重算指标（跳过生成）
                    if os.path.exists(mp4_path):
                        video = _read_video_back(mp4_path)
                        if video is not None:
                            logger.info(
                                "ep=%s q=%d [%s]：mp4 已存在 → 读回重算指标（跳过生成）",
                                ep_id, pt.query_frame, mode)
                            _record_arm(all_records, str(run_dir), args, ep_id, pt,
                                        mode, video, gt_first, mp4_path, gt_png_path,
                                        device)
                            continue
                        logger.warning("ep=%s q=%d [%s]：mp4 存在但读回失败 → 重新生成",
                                       ep_id, pt.query_frame, mode)

                    memory_latents = _memory_latents_for_mode(
                        mode, pt, latents_per_frame, T, rng)
                    if mode != "off" and memory_latents is None:
                        logger.warning(
                            "ep=%s q=%d [%s]：取不到 memory_latents（候选池空）→ "
                            "退化为不注入", ep_id, pt.query_frame, mode)

                    _tmp_action = tempfile.mkdtemp(
                        prefix=f"v5_eval_{ep_id}_q{pt.query_frame}_{mode}_")
                    try:
                        video = _generate_one_arm(
                            wan_i2v, mode, memory_latents, pt, ep, base_img,
                            args, device, rng, _tmp_action)
                    finally:
                        import shutil
                        shutil.rmtree(_tmp_action, ignore_errors=True)

                    if video is None:
                        logger.warning("ep=%s q=%d [%s]：生成返回 None，跳过",
                                       ep_id, pt.query_frame, mode)
                        continue

                    _save_video(video, mp4_path, fps=args.fps)
                    _record_arm(all_records, str(run_dir), args, ep_id, pt,
                                mode, video, gt_first, mp4_path, gt_png_path,
                                device)
            except Exception as exc:  # noqa: BLE001
                logger.exception("重访点处理失败 ep=%s q=%d: %s",
                                 ep_id, pt.query_frame, exc)
                continue

        del frames, latents_per_frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("生成 + 指标完成；调用 summarize_eval ...")
    # ---- 自动 summarize（import；产出 summary.md + 追加 INDEX.md）----
    try:
        from pipeline.v5.summarize_eval import summarize_run
        verdict = summarize_run(str(run_dir), run_name=run_name, tag=args.tag)
        logger.info("summarize verdict: %s", verdict)
    except Exception as exc:  # noqa: BLE001
        logger.warning("summarize_eval 调用失败（非致命）: %s", exc)

    logger.info("Done. 输出目录: %s", run_dir)


if __name__ == "__main__":
    main()
