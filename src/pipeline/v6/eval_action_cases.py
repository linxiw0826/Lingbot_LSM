"""
eval_action_cases.py — v6 action_path 推理生成视频的 frame-aligned DINO/SSIM 定量打分
============================================================================================================

**目的**（工作流第 2 步；承 latentconcat_infer.py action_path 推理）：`latentconcat_infer.py` 用
action_path 模式为每个手工 case、每条臂（trained / base）生成一条 `long_video.mp4`（405 帧 = 5 clip
= 25s@16fps），存到 `<infer_root>/<arm>/<case>/long_video.mp4`。本脚本读这些生成视频 + 每个 case 的
ground truth，**逐 clip** 算 DINO/SSIM，输出 CSV + 汇总表。

为什么要独立脚本（不用 infer 的 `--score`）：latentconcat_infer 的内置 `--score` 只走
oracle_injection 的重访点检测（episode 模式，需机器可读的首访/重访标注），action_path 手工 case
没有这类标注、`--score` 路径不打分。本脚本改用 **label-free frame-aligned** 口径：动作驱动相机 →
gen 帧 t 与 GT 帧 t 同一动作，同下标对比合理；逐 clip 逐帧 cos / SSIM 取均值，不需要重访标注。

**指标（label-free，frame-aligned）**，对每个 (case, arm)：
  1. gen = _read_video_back(<infer_root>/<arm>/<case>/long_video.mp4) → [3,Fg,H,W] in [-1,1]。
  2. gt  = _read_video_back(<cases_root>/<case>/ground_truth_full.mp4) → [3,Fgt,H,W] in [-1,1]。
  3. F = min(Fg, Fgt)；按 frame_num（默认 81）切 clip（clip i = 帧 [i*81:(i+1)*81]，超出 F 截断）。
  4. 逐 clip：clip 内每帧 t 算 cos(_dino_feat(gen[:,t]), _dino_feat(gt[:,t]))（frame-aligned 同下标）；
     该 clip 的 dino = 逐帧 cos 均值。SSIM 同理逐帧 mean（_ssim 灰度全局单窗）。GT 帧 DINO 特征缓存
     避免重算（同 case 跨 arm 复用）。
  5. 某帧 DINO 算不出（返回 None）→ 跳过该帧，不崩（graceful，记 warning）。

**复用 oracle_injection（不重写 DINO/解码/SSIM）**：
  - _dino_feat / _get_dino_model : DINOv2 dinov2_vits14 cosine 特征（lazy 单例 loader）。
  - _read_video_back            : mp4 → [3,F,H,W] in [-1,1]（cv2 顺序解码）。
  - _ssim / _to_gray_uint8      : 灰度全局单窗 SSIM（与 oracle_injection._revisit_consistency 同源）。

**输出**：
  - 逐行 CSV（--out_csv，默认 <infer_root>/action_eval_scores.csv）：
    列 = case, arm, clip_idx, n_frames, dino_mean, ssim_mean。
  - stdout 汇总表：每 arm「全 clip 平均 dino」+「各 clip 平均 dino（跨 case）」并排对比各 arm
    （方便看 trained vs base 在哪个 clip 差异大——重访 clip 是重点）。
  - --revisit_clip N（可选）：额外单独打印「各 arm 在 clip N 的平均 dino」。

约束：DINO/torch 相关 import 放函数内 / lazy（别拖垮 --help）；只读输入，不改其它文件；缺失文件
（某 arm/case 的 long_video.mp4 不存在）→ warning 跳过，不崩。
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from os.path import abspath, dirname, join
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path（与 latentconcat_infer / latentconcat_ideal_diag / infer_v5 一致）
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "v6 action_path 推理生成视频的 frame-aligned DINO/SSIM 打分（对照手工 case ground truth）。"
            "逐 clip 输出 CSV + 各 arm 汇总对比。"
        )
    )
    p.add_argument("--infer_root", type=str, required=True,
                   help="推理输出根目录（如 <OUT>/v6/infer/v6_eval_ep027）；"
                        "每 arm/case 的生成视频在 <infer_root>/<arm>/<case>/long_video.mp4。")
    p.add_argument("--cases_root", type=str, required=True,
                   help="手工 case 根目录（如 .../revisit_ep027_manual_v2_5clip_selected）；"
                        "每 case 含 ground_truth_full.mp4 + case_meta.json。")
    p.add_argument("--arms", type=str, default="trained,base",
                   help="逗号分隔臂名（默认 trained,base；对应 <infer_root>/<arm>/）。")
    p.add_argument("--cases", type=str, default=None,
                   help="逗号分隔 case 名；不给则自动扫描 <infer_root>/<第一个 arm>/ 下的子目录。")
    p.add_argument("--frame_num", type=int, default=81,
                   help="每 clip 帧数（默认 81；切 clip 用 [i*frame_num:(i+1)*frame_num]）。")
    p.add_argument("--revisit_clip", type=int, default=None,
                   help="（可选）重访 clip 下标；给了则额外单独打印各 arm 在该 clip 的平均 dino。")
    p.add_argument("--out_csv", type=str, default=None,
                   help="逐行 CSV 输出路径（默认 <infer_root>/action_eval_scores.csv）。")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="DINO 计算 device（默认 cuda:0；不可用回退 cpu）。")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 打分核心（复用 oracle_injection 的 DINO / 解码 / SSIM，不重写）
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str):
    """按请求 device 解析 torch.device，CUDA 不可用则回退 cpu（graceful）。"""
    import torch
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用 → device 回退 cpu（原请求 %s）", device_str)
        return torch.device("cpu")
    return torch.device(device_str)


def _score_case_arm(
    gen_path: str,
    gt_video,                     # [3,Fgt,H,W] in [-1,1]（已解码，同 case 跨 arm 复用）
    gt_dino_cache: List,          # 长度 Fgt 的列表；元素 = GT 帧 t 的 DINO 特征 / None / 未算（用哨兵）
    frame_num: int,
    device,
) -> List[Dict]:
    """对单个 (case, arm) 逐 clip 算 frame-aligned dino_mean / ssim_mean。

    Args:
        gen_path:      <infer_root>/<arm>/<case>/long_video.mp4
        gt_video:      已解码 GT full 视频 [3,Fgt,H,W]
        gt_dino_cache: GT 每帧 DINO 特征缓存（in/out；跨 arm 复用，避免重算）
        frame_num:     每 clip 帧数
        device:        DINO device

    Returns:
        每 clip 一个 dict：{clip_idx, n_frames, dino_mean, ssim_mean}；
        gen 缺失 / 读不回时返回空 list（调用方 warning 跳过）。
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from pipeline.eval.oracle_injection import (
        _dino_feat, _ssim, _to_gray_uint8,
    )
    from pipeline.eval.oracle_injection import _read_video_back

    if not os.path.isfile(gen_path):
        logger.warning("生成视频不存在 → 跳过：%s", gen_path)
        return []
    gen_video = _read_video_back(gen_path)
    if gen_video is None:
        logger.warning("生成视频读回失败 → 跳过：%s", gen_path)
        return []

    fg = gen_video.shape[1]
    fgt = gt_video.shape[1]
    n_common = min(fg, fgt)
    if n_common <= 0:
        logger.warning("生成 / GT 无可比帧（Fg=%d Fgt=%d）→ 跳过：%s", fg, fgt, gen_path)
        return []
    if fg != fgt:
        logger.warning("生成 / GT 帧数不等（Fg=%d Fgt=%d）→ 取前 %d 帧对齐：%s",
                       fg, fgt, n_common, gen_path)

    n_clips = (n_common + frame_num - 1) // frame_num   # 向上取整，末 clip 可能不足 frame_num
    rows: List[Dict] = []

    # gen 各帧 H,W 相同（同一次生成）；GT 灰度若分辨率不同，resize 到 gen 分辨率（同
    # oracle_injection._revisit_consistency 口径：以 gen 输出为准，float32 双线性）。
    h_gen, w_gen = gen_video.shape[2], gen_video.shape[3]

    def _gt_gray_aligned(t: int):
        """GT 帧 t → 灰度 [h_gen,w_gen] float32（分辨率对齐 gen 用于 _ssim）。"""
        g = _to_gray_uint8(gt_video[:, t])            # [H_gt,W_gt] float32 [0,255]
        if g.shape != (h_gen, w_gen):
            img = Image.fromarray(g, mode="F").resize((w_gen, h_gen), resample=Image.BILINEAR)
            g = np.asarray(img, dtype=np.float32)
        return g

    def _gt_dino(t: int):
        """GT 帧 t 的 DINO 特征（缓存；_UNSET 哨兵表示未算）。DINO 自带 resize，无需对齐分辨率。"""
        cached = gt_dino_cache[t]
        if cached is not _UNSET:
            return cached
        feat = _dino_feat(gt_video[:, t], device)     # None（算不出）也缓存，避免重试
        gt_dino_cache[t] = feat
        return feat

    for c in range(n_clips):
        f0 = c * frame_num
        f1 = min((c + 1) * frame_num, n_common)
        dino_cos: List[float] = []
        ssims: List[float] = []
        for t in range(f0, f1):
            # SSIM（frame-aligned，灰度全局单窗）
            ssims.append(_ssim(_to_gray_uint8(gen_video[:, t]), _gt_gray_aligned(t)))
            # DINO cosine（frame-aligned；任一帧特征算不出则跳过该帧的 dino，不崩）
            g_feat = _dino_feat(gen_video[:, t], device)
            gt_feat = _gt_dino(t)
            if g_feat is not None and gt_feat is not None:
                cos = F.cosine_similarity(g_feat.unsqueeze(0), gt_feat.unsqueeze(0), dim=-1)
                dino_cos.append(float(cos.item()))
            else:
                logger.warning("DINO 特征缺失 → 跳过帧 t=%d（clip %d）：%s", t, c, gen_path)

        dino_arr = np.asarray(dino_cos, dtype=np.float32)
        ssim_arr = np.asarray(ssims, dtype=np.float32)
        rows.append({
            "clip_idx": c,
            "n_frames": int(f1 - f0),
            # dino_mean：跨该 clip 成功算出的帧取均值；全无 → NaN（写 CSV 为空串等价）
            "dino_mean": float(dino_arr.mean()) if dino_arr.size else float("nan"),
            "ssim_mean": float(ssim_arr.mean()) if ssim_arr.size else float("nan"),
        })
    return rows


# GT DINO 缓存哨兵（区分「未算」与「算过但为 None」）
_UNSET = object()


def _load_gt_and_meta(cases_root: str, case: str):
    """解码 case 的 ground_truth_full.mp4 + 读 case_meta.json 的 frame_num（若有）。

    Returns:
        (gt_video[3,Fgt,H,W] 或 None, meta_frame_num 或 None)
    """
    import json
    from pipeline.eval.oracle_injection import _read_video_back

    gt_path = join(cases_root, case, "ground_truth_full.mp4")
    if not os.path.isfile(gt_path):
        logger.warning("GT 视频不存在 → 跳过 case %s：%s", case, gt_path)
        return None, None
    gt_video = _read_video_back(gt_path)
    if gt_video is None:
        logger.warning("GT 视频读回失败 → 跳过 case %s：%s", case, gt_path)
        return None, None

    meta_frame_num = None
    meta_path = join(cases_root, case, "case_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            meta_frame_num = meta.get("frame_num")
        except Exception as exc:  # noqa: BLE001
            logger.warning("读 case_meta.json 失败（用 --frame_num 兜底）：%s（%s）", meta_path, exc)
    return gt_video, meta_frame_num


# ---------------------------------------------------------------------------
# 汇总输出
# ---------------------------------------------------------------------------

def _print_summary(records: List[Dict], arms: List[str], revisit_clip: Optional[int]) -> None:
    """stdout 汇总表：各 arm「全 clip 平均 dino」+「各 clip 平均 dino（跨 case）」并排对比。

    records: 逐 (case,arm,clip) 行 dict（含 arm/clip_idx/dino_mean）。
    """
    import numpy as np

    # 收集所有 clip 下标（并集，稳态排序）
    clip_ids = sorted({r["clip_idx"] for r in records})
    if not clip_ids:
        print("\n[summary] 无有效记录（所有 arm/case 均缺失或读回失败）。")
        return

    def _mean_dino(rows: List[Dict]) -> float:
        vals = [r["dino_mean"] for r in rows if not np.isnan(r["dino_mean"])]
        return float(np.mean(vals)) if vals else float("nan")

    def _fmt(x: float) -> str:
        return "  nan " if np.isnan(x) else f"{x:6.4f}"

    print("\n" + "=" * 72)
    print("汇总：各 arm 的 frame-aligned DINO（跨 case 平均；重访 clip 是重点）")
    print("=" * 72)

    # 表头：arm | overall | clip0 clip1 ...
    header = f"{'arm':<12}{'overall':>9} | " + " ".join(f"clip{c:<2d}" for c in clip_ids)
    print(header)
    print("-" * len(header))
    for arm in arms:
        arm_rows = [r for r in records if r["arm"] == arm]
        overall = _mean_dino(arm_rows)
        per_clip = []
        for c in clip_ids:
            per_clip.append(_mean_dino([r for r in arm_rows if r["clip_idx"] == c]))
        line = f"{arm:<12}{_fmt(overall):>9} | " + " ".join(_fmt(v) for v in per_clip)
        print(line)

    # 可选：单独打印 revisit_clip 各 arm 平均 dino
    if revisit_clip is not None:
        print("\n" + "-" * 40)
        print(f"[revisit_clip={revisit_clip}] 各 arm 平均 dino（跨 case）")
        print("-" * 40)
        for arm in arms:
            v = _mean_dino([r for r in records
                            if r["arm"] == arm and r["clip_idx"] == revisit_clip])
            print(f"  {arm:<12} {_fmt(v)}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("--arms 解析为空")

    # case 列表：显式 --cases 或自动扫描 <infer_root>/<第一个 arm>/ 子目录
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        scan_dir = join(args.infer_root, arms[0])
        if not os.path.isdir(scan_dir):
            raise SystemExit(f"自动扫描失败：{scan_dir} 不是目录（用 --cases 显式指定）")
        cases = sorted(
            d for d in os.listdir(scan_dir)
            if os.path.isdir(join(scan_dir, d))
        )
        logger.info("自动扫描到 %d 个 case（from %s）", len(cases), scan_dir)
    if not cases:
        raise SystemExit("case 列表为空（--cases 未给且自动扫描无结果）")

    device = _resolve_device(args.device)
    out_csv = args.out_csv or join(args.infer_root, "action_eval_scores.csv")

    records: List[Dict] = []            # 逐 (case,arm,clip) 行（含全部 CSV 列）
    for case in cases:
        gt_video, meta_frame_num = _load_gt_and_meta(args.cases_root, case)
        if gt_video is None:
            continue                    # GT 缺失 → 该 case 所有 arm 跳过
        # frame_num 口径：优先 case_meta.json 的 frame_num，回退 --frame_num
        frame_num = meta_frame_num if meta_frame_num else args.frame_num
        if meta_frame_num and meta_frame_num != args.frame_num:
            logger.info("case %s：用 case_meta.frame_num=%d（覆盖 --frame_num=%d）",
                        case, meta_frame_num, args.frame_num)

        # GT DINO 特征缓存（同 case 跨 arm 复用；哨兵 _UNSET = 未算）
        gt_dino_cache = [_UNSET] * gt_video.shape[1]

        for arm in arms:
            gen_path = join(args.infer_root, arm, case, "long_video.mp4")
            clip_rows = _score_case_arm(
                gen_path, gt_video, gt_dino_cache, frame_num, device,
            )
            for row in clip_rows:
                records.append({
                    "case": case,
                    "arm": arm,
                    "clip_idx": row["clip_idx"],
                    "n_frames": row["n_frames"],
                    "dino_mean": row["dino_mean"],
                    "ssim_mean": row["ssim_mean"],
                })
                logger.info("case=%s arm=%s clip=%d n=%d dino=%.4f ssim=%.4f",
                            case, arm, row["clip_idx"], row["n_frames"],
                            row["dino_mean"], row["ssim_mean"])

    # 逐行 CSV
    os.makedirs(dirname(abspath(out_csv)), exist_ok=True)
    fieldnames = ["case", "arm", "clip_idx", "n_frames", "dino_mean", "ssim_mean"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    logger.info("逐行分数写出 → %s（%d 行）", out_csv, len(records))

    # stdout 汇总表
    _print_summary(records, arms, args.revisit_clip)


if __name__ == "__main__":
    main()
