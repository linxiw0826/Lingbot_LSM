"""revisit_metric_sanity.py — revisit DINO 指标 headroom / 重访真实性 sanity 检查（零生成）
========================================================================================

**目的**（承 tier-A in-context KV NO-GO：off/ideal_A/random_A 的 revisit DINO ≈ 0.11–0.20，
差异 ±0.01）：担心 revisit DINO 指标本身在 CS:GO 动态场景下**饱和/无 headroom**——同一地点
首访 vs 重访时刻画面本就因不同玩家/动作而不同，GT-vs-GT DINO 可能就很低。若如此，tier-A 的
NO-GO 及后续 v6 闸门的数都不可信。本脚本**先量出指标天花板**，回答两问：

  Q1（指标有没有 headroom）：DINO(GT 首访, GT 重访) 是否显著高于 DINO(GT 首访, 随机无关帧)？
  Q2（重访是否真实）：并排 PNG（首访 / 重访 / 随机）供人眼确认自动检测出的重访 case 确是同地点。

**零生成、不加载 Wan 骨干、不用 VAE**：只做 (a) 复用 _enumerate_cases 枚举 revisit case（同一套
hit_dist/hit_yaw/time_gap 参数）；(b) 解码 episode GT 帧；(c) 对每 case 取三真值帧
（GT 首访 / GT 重访 / 随机历史帧）算 DINO 余弦 + SSIM；(d) 存并排 PNG + sanity.csv + summary.md。
全程无任何 diffusion 生成、无 WanI2V 装载——快、轻，CPU/单卡即可。

**三个量（每 case）**：
  · dino_first_revisit : DINO(GT 首访, GT 重访)  = 指标天花板（同地点不同时刻真值一致性）
  · dino_first_random  : DINO(GT 首访, 随机无关帧) = 随机对照天花板（复用 _pick_random_hist_frame）
  · ssim_first_revisit : SSIM(GT 首访, GT 重访)  = 像素对照（参考）

**判读（summary.md 末尾自动给出）**：
  · mean dino_first_revisit 显著 > mean dino_first_random（默认高出 --headroom_margin=0.1+）
      → 指标能分辨"同地点" vs "随机帧"，**有 headroom**：off/ideal 的低值是"该低"，
        记忆有发挥空间，tier-A NO-GO 可信。
  · dino_first_revisit ≈ dino_first_random（都低，Δ < margin）
      → **指标饱和/失效**：revisit-vs-首访帧这把尺在动态 CS:GO 上无区分度，需换度量
        （与 GT 重访 clip 逐帧比 / pose-paired / MEt3R 几何一致性）。

**复用（import，不重写、不改既有 src）**：
  · pipeline.v5.ideal_inject_diag._enumerate_cases  — revisit case 枚举（内部用 build_episode_data
    + _find_revisit_points，与 ideal_diag / oracle_injection 同一套 hit_dist/hit_yaw/time_gap）
  · pipeline.eval.oracle_injection._dino_feat / _ssim / _to_gray_uint8 / _save_frame_png / RevisitPoint
  · pipeline.eval.retrieval_probe.load_episode_clips / _decode_episode_video
  · pipeline.v5.eval_v5._pick_random_hist_frame  — 随机历史帧（confound 对照，与 ideal_random 同源）

上述 import 均**不在模块顶层加载 wan / memory_module / VAE**（这些在各文件的函数体内才 import）；
本脚本也不调用任何加载骨干的函数 → 确认零生成、不加载骨干。

服务器跑前置：`export TMPDIR=/tmp`（否则被 kill 的 run 会在仓库留 pymp-* 孤儿）。
本地即便无 CUDA 也能跑（DINO 走 CPU；仅慢，不崩）。DINO 加载失败时 dino_* 字段留空（graceful）。
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# sys.path（与 ideal_inject_diag / oracle_injection 一致）
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
# 复用（import，不重写）——均不在模块顶层加载 wan / 骨干 / VAE
# ---------------------------------------------------------------------------
from pipeline.eval.oracle_injection import (  # noqa: E402
    RevisitPoint,
    _dino_feat,
    _ssim,
    _to_gray_uint8,
    _save_frame_png,
)
from pipeline.eval.retrieval_probe import (  # noqa: E402
    load_episode_clips,
    _decode_episode_video,
)
from pipeline.v5.eval_v5 import _pick_random_hist_frame  # noqa: E402
# _enumerate_cases 内部用 build_episode_data + _find_revisit_points，与 ideal_diag 同一套判定
from pipeline.v5.ideal_inject_diag import _enumerate_cases  # noqa: E402

from pipeline.common.paths import (  # noqa: E402
    eval_run_dir,
    snapshot_config,
    default_run_name,
)


# ---------------------------------------------------------------------------
# CLI（对齐 latentconcat_ideal_diag：数据 / 重访点判定 / 分片）
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "revisit DINO 指标 headroom / 重访真实性 sanity 检查（零生成、不加载骨干）："
            "对每个自动检测的 revisit case 取 GT 首访/重访/随机帧，算 DINO+SSIM 天花板，"
            "判读指标是否有 headroom。服务器跑前置：export TMPDIR=/tmp。"
        )
    )
    # ---- 数据 ----
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="含重访的数据集根目录（含 metadata CSV 和 clips/）")
    p.add_argument("--metadata", type=str, required=True,
                   help="相对 dataset_dir 的 CSV 路径，如 metadata_verify_train.csv")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="仅跑这些 episode（逗号分隔），默认跑 CSV 全集")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="0=不限；>0 时取前 N 个 episode")
    p.add_argument("--max_cases", type=int, default=20,
                   help="总 revisit case 上限（默认 20；在切分前应用，各分片看同一全集）")

    # ---- 重访点判定（与 oracle_injection / ideal_diag 同口径）----
    p.add_argument("--hit_dist", type=float, default=40.0)
    p.add_argument("--hit_yaw", type=float, default=30.0)
    p.add_argument("--intermediate_separation", type=float, default=100.0)
    p.add_argument("--min_time_gap_sec", type=float, default=5.0)
    p.add_argument("--clip_overlap_frames", type=int, default=0)
    p.add_argument("--max_revisit_points", type=int, default=2)

    # ---- 帧解码 / 判读 ----
    p.add_argument("--size", type=str, default="480*832", help="解码分辨率 H*W")
    p.add_argument("--fps", type=int, default=16,
                   help="帧率（min_time_gap_sec→帧数换算，与重访判定一致）")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="DINO 计算 device；CUDA 不可用自动回退 CPU")
    p.add_argument("--seed", type=int, default=42,
                   help="随机帧选取 rng 种子（可复现）")
    p.add_argument("--headroom_margin", type=float, default=0.1,
                   help="判读 margin：mean DINO(首访,重访) 须比 mean DINO(首访,随机) 高出至少"
                        "这么多，才判'指标有 headroom'（默认 0.1）")

    # ---- 产出 ----
    p.add_argument("--out", type=str, default=None,
                   help="输出根目录（存 sanity.csv/summary.md/并排 PNG）；缺省用 eval_run_dir")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--tag", type=str, default="revisit_metric_sanity",
                   help="eval 场景 tag（INDEX 区分用）")

    # ---- 分片（additive：shard_count 默认 1 → 逐位与单进程一致；单卡够快，无生成）----
    p.add_argument("--shard_index", type=int, default=0,
                   help="当前分片索引（0-based）。默认 0。")
    p.add_argument("--shard_count", type=int, default=1,
                   help="总分片数。默认 1=不分片；>1 时按 case 全局序号取模分片。")

    return p.parse_args()


# ---------------------------------------------------------------------------
# 单帧 DINO 余弦（复用 _dino_feat；两帧特征均算不出时返回 None）
# ---------------------------------------------------------------------------

def _dino_cosine(frame_a: np.ndarray, frame_b: np.ndarray,
                 device: torch.device) -> Optional[float]:
    """DINO(frame_a, frame_b) 余弦相似度。任一帧特征算不出（DINO 不可用）→ None。"""
    fa = _dino_feat(frame_a, device)
    fb = _dino_feat(frame_b, device)
    if fa is None or fb is None:
        return None
    cos = F.cosine_similarity(fa.unsqueeze(0), fb.unsqueeze(0), dim=-1)
    return float(cos.item())


def _ssim_frames(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """SSIM(frame_a, frame_b)（灰度全局单窗口，复用 oracle_injection._ssim）。

    两帧同为 GT 解码帧、同分辨率，无需 resize 对齐。
    """
    return _ssim(_to_gray_uint8(frame_a), _to_gray_uint8(frame_b))


# ---------------------------------------------------------------------------
# sanity.csv 增量写（抗崩 / 抗分片）
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "episode_id", "query_frame", "first_visit_frame", "random_frame",
    "dino_first_revisit", "dino_first_random",
    "ssim_first_revisit", "ssim_first_random",
    "gt_first_visit_png", "gt_revisit_png", "gt_random_png",
]


def _append_csv(out_dir: str, record: Dict) -> None:
    csv_path = os.path.join(out_dir, "sanity.csv")
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 sanity.csv 失败: %s", exc)


# ---------------------------------------------------------------------------
# 单 case 处理（零生成：只取三真值帧 + DINO/SSIM + 存并排 PNG）
# ---------------------------------------------------------------------------

def _process_case(ep, ep_id: str, pt: RevisitPoint, frames: np.ndarray,
                  out_dir: str, device: torch.device, rng: np.random.Generator,
                  all_records: List[Dict]) -> None:
    """处理单个 (episode, revisit case)：取 GT 首访/重访/随机帧，算 DINO+SSIM，存并排 PNG。"""
    T = frames.shape[0]
    fi = int(pt.first_visit_frame)
    qi = int(pt.query_frame)
    if not (0 <= fi < T and 0 <= qi < T):
        logger.warning("ep=%s q=%d：first_visit=%d / query=%d 越界（T=%d），跳过",
                       ep_id, qi, fi, qi, T)
        return

    gt_first = frames[fi]     # [3,H,W] in [-1,1]（GT 首访帧）
    gt_revisit = frames[qi]   # [3,H,W]（GT 重访帧）

    # 随机对照帧（非首访随机历史帧，复用 ideal_random 同源选取）
    ri = _pick_random_hist_frame(pt, T, rng)
    gt_random = frames[int(ri)] if ri is not None else None

    # ---- 并排 PNG（人眼确认首访/重访是否同地点）----
    q_dir = os.path.join(out_dir, ep_id, f"q{qi}")
    os.makedirs(q_dir, exist_ok=True)
    first_png = os.path.join(q_dir, "gt_first_visit.png")
    revisit_png = os.path.join(q_dir, "gt_revisit.png")
    random_png = os.path.join(q_dir, "gt_random.png")
    _save_frame_png(gt_first, first_png)
    _save_frame_png(gt_revisit, revisit_png)
    if gt_random is not None:
        _save_frame_png(gt_random, random_png)
    else:
        random_png = ""

    # ---- DINO 天花板 + 随机对照 + SSIM ----
    dino_first_revisit = _dino_cosine(gt_first, gt_revisit, device)
    dino_first_random = (_dino_cosine(gt_first, gt_random, device)
                         if gt_random is not None else None)
    ssim_first_revisit = _ssim_frames(gt_first, gt_revisit)
    ssim_first_random = (_ssim_frames(gt_first, gt_random)
                         if gt_random is not None else None)

    record = {
        "episode_id": ep_id,
        "query_frame": qi,
        "first_visit_frame": fi,
        "random_frame": int(ri) if ri is not None else -1,
        "dino_first_revisit": dino_first_revisit,
        "dino_first_random": dino_first_random,
        "ssim_first_revisit": ssim_first_revisit,
        "ssim_first_random": ssim_first_random,
        "gt_first_visit_png": first_png,
        "gt_revisit_png": revisit_png,
        "gt_random_png": random_png,
    }
    logger.info("ep=%s q=%d | DINO(首访,重访)=%s DINO(首访,随机)=%s SSIM(首访,重访)=%.4f",
                ep_id, qi, dino_first_revisit, dino_first_random, ssim_first_revisit)
    all_records.append(record)
    _append_csv(out_dir, record)


# ---------------------------------------------------------------------------
# 判读 / summary.md
# ---------------------------------------------------------------------------

def _mean_or_nan(vals: List[float]) -> float:
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else float("nan")


def _write_summary(all_records: List[Dict], margin: float, out_dir: str) -> str:
    """逐 case 表 + 均值 + headroom 判读，打印并写 summary.md。返回判读结论字符串。"""
    lines: List[str] = []
    lines.append("# revisit DINO 指标 headroom / 重访真实性 sanity（零生成）\n")
    lines.append(f"- timestamp: {datetime.now().isoformat()}")
    lines.append(f"- n_cases: {len(all_records)}")
    lines.append(f"- headroom_margin: {margin}\n")
    lines.append("说明：DINO(首访,重访)=指标天花板（同地点不同时刻真值一致性）；"
                 "DINO(首访,随机)=随机对照天花板。若前者显著 > 后者 → 指标有 headroom。\n")

    header = ("| episode | query | first | random | DINO(首访,重访) | DINO(首访,随机) | "
              "Δ(重访−随机) | SSIM(首访,重访) |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append("## 逐 case\n")
    lines.append(header)
    lines.append(sep)

    dvr, dvran, svr = [], [], []
    for r in sorted(all_records, key=lambda x: (x["episode_id"], x["query_frame"])):
        dr = r["dino_first_revisit"]
        drn = r["dino_first_random"]
        sr = r["ssim_first_revisit"]
        dvr.append(dr)
        dvran.append(drn)
        svr.append(sr)
        delta = (dr - drn) if (dr is not None and drn is not None) else float("nan")
        f_dr = "nan" if dr is None else f"{dr:.4f}"
        f_drn = "nan" if drn is None else f"{drn:.4f}"
        f_sr = "nan" if sr is None else f"{sr:.4f}"
        f_delta = "nan" if delta != delta else f"{delta:+.4f}"
        lines.append(f"| {r['episode_id']} | {r['query_frame']} | {r['first_visit_frame']} | "
                     f"{r['random_frame']} | {f_dr} | {f_drn} | {f_delta} | {f_sr} |")

    mean_dvr = _mean_or_nan(dvr)
    mean_dvran = _mean_or_nan(dvran)
    mean_svr = _mean_or_nan(svr)
    delta_mean = mean_dvr - mean_dvran

    lines.append("\n## 均值\n")
    lines.append(f"- mean DINO(首访,重访) = {mean_dvr:.4f}  (指标天花板)")
    lines.append(f"- mean DINO(首访,随机) = {mean_dvran:.4f}  (随机对照)")
    lines.append(f"- Δ(重访−随机)        = {delta_mean:+.4f}  (margin={margin})")
    lines.append(f"- mean SSIM(首访,重访) = {mean_svr:.4f}  (像素参考)")

    # 判读
    has_headroom = (mean_dvr == mean_dvr) and (mean_dvran == mean_dvran) \
        and (delta_mean >= margin)
    if has_headroom:
        verdict = "HAS_HEADROOM"
        interp = (
            f"指标能分辨'同地点'(DINO={mean_dvr:.4f}) 与'随机帧'(DINO={mean_dvran:.4f})，"
            f"Δ={delta_mean:+.4f} ≥ margin={margin} → **revisit DINO 有 headroom**："
            "off/ideal 的低值是'该低'，记忆有发挥空间，tier-A NO-GO 及 v6 闸门的数可信。")
    else:
        verdict = "METRIC_SATURATED"
        interp = (
            f"DINO(首访,重访)={mean_dvr:.4f} ≈ DINO(首访,随机)={mean_dvran:.4f}"
            f"（Δ={delta_mean:+.4f} < margin={margin}）→ **指标饱和/失效**："
            "revisit-vs-首访帧这把尺在动态 CS:GO 上无区分度，tier-A NO-GO 与 v6 闸门的数"
            "不可信，需换度量（与 GT 重访 clip 逐帧比 / pose-paired / MEt3R 几何一致性）。")

    lines.append("\n## 判读\n")
    lines.append(f"**{verdict}** — {interp}")

    summary = "\n".join(lines) + "\n"
    try:
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(summary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("写 summary.md 失败: %s", exc)
    print("\n" + summary)
    logger.info("verdict=%s | mean DINO(首访,重访)=%.4f vs DINO(首访,随机)=%.4f (Δ=%+.4f)",
                verdict, mean_dvr, mean_dvran, delta_mean)
    return verdict


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- 输出目录 ----
    if args.out:
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
    else:
        run_name = args.run_name or default_run_name("v6_revisit_sanity")
        out_dir = str(eval_run_dir("v6", run_name, args.tag))
    log_path = os.path.join(out_dir, "sanity.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    try:
        snapshot_config(__import__("pathlib").Path(out_dir),
                        {k: v for k, v in vars(args).items() if not k.startswith("_")})
    except Exception as exc:  # noqa: BLE001
        logger.warning("snapshot_config 失败（非致命）: %s", exc)
    logger.info("revisit_metric_sanity out_dir=%s（零生成、不加载骨干）", out_dir)
    logger.info("Args: %s", vars(args))

    # ---- device（DINO 计算用；CUDA 不可用回退 CPU，仅慢不崩）----
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，DINO 走 CPU（较慢）")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    height, width = (int(x) for x in args.size.split("*"))
    min_time_gap_frames = max(1, int(round(args.min_time_gap_sec * args.fps)))

    # ---- episode CSV ----
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

    # ---- 枚举 revisit case 全集（max_cases 在切分前应用）→ 单卡 or 分片 ----
    ordered_cases, ep_cache = _enumerate_cases(ep_ids, ep_groups, args, min_time_gap_frames)
    if not ordered_cases:
        logger.error("无 revisit case，退出。")
        return
    if args.shard_count > 1:
        sel = [(gi, e, p) for gi, (e, p) in enumerate(ordered_cases)
               if gi % args.shard_count == args.shard_index]
        logger.info("分片 %d/%d：case 全集 %d，本分片处理 %d",
                    args.shard_index, args.shard_count, len(ordered_cases), len(sel))
    else:
        sel = [(gi, e, p) for gi, (e, p) in enumerate(ordered_cases)]
        logger.info("单卡路径：处理全部 %d 个 case", len(sel))
    if not sel:
        logger.error("shard %d/%d 分到 0 个 case，退出。",
                     args.shard_index, args.shard_count)
        return

    by_ep: "OrderedDict[str, List]" = OrderedDict()
    for _gi, ep_id, pt in sel:
        by_ep.setdefault(ep_id, []).append(pt)

    all_records: List[Dict] = []
    for ep_id, pts in by_ep.items():
        ep = ep_cache[ep_id]
        T = ep.poses.shape[0]
        logger.info("Episode %s: T=%d, 本分片 %d 个 case", ep_id, T, len(pts))
        try:
            frames = _decode_episode_video(ep, height=height, width=width)  # [T,3,H,W]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episode %s 解码失败: %s；跳过", ep_id, exc)
            continue
        for pt in pts:
            try:
                _process_case(ep, ep_id, pt, frames, out_dir, device, rng, all_records)
            except Exception as exc:  # noqa: BLE001
                logger.exception("case 处理失败 ep=%s q=%d: %s", ep_id, pt.query_frame, exc)
                continue
        del frames

    if not all_records:
        logger.error("无任何记录（无 case / 全失败），退出。")
        return

    _write_summary(all_records, args.headroom_margin, out_dir)
    logger.info("Done. 输出目录: %s", out_dir)


if __name__ == "__main__":
    main()
