"""
covis_census.py — 共视 anchor 挖掘器（covisibility.py）在**真训练数据**上的标定普查
================================================================================

背景（decisions.md 讨论 13 / D-09'，[[F-33]]）
------------------------------------------------
`covisibility.select_covisible_anchor_frames` 的阈值（`COVIS_POS_TEMP_M=4.0` /
`COVIS_YAW_TEMP_DEG=35` / `DEFAULT_MIN_COVIS_SCORE=0.3`）与 `POSE_UNITS_PER_METER=40`
都是**未在真数据上标定的首猜值**。若 `POSE_UNITS_PER_METER` 与真实 pose 单位不符，所有距离
判据会静默失灵 → 挖掘器永远找不到共视帧 → `train_v6._select_anchor` 退化成 all-empty anchor，
训练悄悄降级。本脚本在**训练同源数据 / 同窗构造**下把挖掘器**跑一遍**，输出标定统计，让人在
花训练算力前判断阈值/单位是否合理。

它做什么（不改 covisibility.py / train_v6.py，纯诊断，只写 stdout / 可选 --out_json）
-----------------------------------------------------------------------------
1. **复用 train_v6 的数据路径**：直接实例化 `pipeline.v4.train_v4_stage1_dual.CSGOMultiClipDataset`
   （train_v6.main 用的同一个类），拿到 `dataset.samples`（= 训练时逐窗的 clip 行窗口）。为省时/纯
   CPU，普查**只加载 poses(+intrinsics).npy**（不解码 video.mp4），路径约定与 `__getitem__` 一致
   （`clips/{ep_id}/{ep_id}_clip{idx:02d}/poses.npy`），并用同款 `_pad_or_truncate` 补齐到 num_frames。
2. **逐窗镜像 `_select_anchor` 的共视路径**（不含 dropout 随机）：n_ctx ~ Uniform(2, max_context_clips)
   （与 train 循环同分布，seeded）；context = window[:n_ctx]，target = window[n_ctx]；
   ref_c2w = target frame-0 绝对 c2w；调 `select_covisible_anchor_frames(context_clips, ref_c2w,
   num_frames=1, min_score, rng, exclude_adjacent_frames=4, use_fov)`。同时**独立**对每个候选
   context 帧算 covis_score + 原生/米 XZ 位置差 + yaw 差，得到阈值前的原始分布。
3. **报告**：命中率 / 分布(min/median/p90/max) / 单位 sanity(帧间&窗内位移) / 命中窗抽样 dump /
   阈值扫描（min_score × pos_temp）。

无真实数据时可跑 `--synthetic` 合成 smoke：证明普查逻辑端到端可跑（不需服务器数据）。

运行（服务器，真数据）：
    python src/pipeline/v6/covis_census.py \
        --dataset_dir /home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3 \
        --phase verify --max_windows 500

    合成 smoke（本地无数据）：python src/pipeline/v6/covis_census.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# sys.path（与 train_v6 对齐：把 src/ 放上去，pipeline.* 可导入）
# ---------------------------------------------------------------------------
_V6_DIR = dirname(abspath(__file__))          # src/pipeline/v6/
_PIPELINE_DIR = dirname(_V6_DIR)              # src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)             # src/
for _p in (_SRC_DIR, _PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 共视挖掘器（被普查对象）+ 其复用的相机几何工具（口径必须一致）
from pipeline.v6.covisibility import (  # noqa: E402
    select_covisible_anchor_frames,
    covisibility_score,
    DEFAULT_MIN_COVIS_SCORE,
    COVIS_POS_TEMP_M,
    COVIS_YAW_TEMP_DEG,
    _to_poses_np,
    _to_c2w_np,
    _xz_pos,
)
from pipeline.data.revisit_data_spec import (  # noqa: E402
    POSE_UNITS_PER_METER,
    yaw_deg_from_c2w,
    angular_diff_deg,
)

# train_v6._select_anchor 挖掘时排除紧邻 query 的帧数（= train_v6._ANCHOR_EXCLUDE_ADJACENT_FRAMES）
_EXCLUDE_ADJACENT_FRAMES = 4


# ═══════════════════════════════════════════════════════════════════════════
# 候选帧原始几何量（与挖掘器口径一致：XZ 平面、原生单位、yaw=atan2(R[0,2],R[2,2])）
# ═══════════════════════════════════════════════════════════════════════════

def _candidate_record(ref_c2w: np.ndarray, cand_c2w: np.ndarray,
                      cand_intr, args) -> dict:
    """对单个候选 context 帧算原始几何量 + covis_score（默认温度，与 select 一致）。"""
    dist_native = float(np.linalg.norm(_xz_pos(ref_c2w) - _xz_pos(cand_c2w)))
    dist_m = dist_native / max(1e-9, POSE_UNITS_PER_METER)
    dyaw = float(angular_diff_deg(yaw_deg_from_c2w(ref_c2w), yaw_deg_from_c2w(cand_c2w)))
    score = covisibility_score(
        ref_c2w, cand_c2w, None, cand_intr,
        use_fov=args.use_fov,
    )
    return {"dist_native": dist_native, "dist_m": dist_m, "yaw_deg": dyaw, "score": score}


def _enumerate_candidates(context_clips: List[dict], ref_c2w: np.ndarray, args):
    """枚举挖掘器会打分的**全部**候选 (ci, fi)（含 exclude_adjacent 排除逻辑），逐帧算 record。

    与 select_covisible_anchor_frames 内层遍历口径严格一致：仅最后一个 context clip 的末尾
    _EXCLUDE_ADJACENT_FRAMES 帧被排除。
    """
    last_idx = len(context_clips) - 1
    records: List[dict] = []
    for ci, clip in enumerate(context_clips):
        poses = _to_poses_np(clip["poses"])          # [F,4,4]
        n_frames = poses.shape[0]
        hi = n_frames
        if ci == last_idx and _EXCLUDE_ADJACENT_FRAMES > 0:
            hi = max(0, n_frames - _EXCLUDE_ADJACENT_FRAMES)
        clip_intr = None
        if args.use_fov and clip.get("intrinsics") is not None:
            clip_intr = np.squeeze(np.asarray(clip["intrinsics"], dtype=np.float32))
        for fi in range(hi):
            cand_intr = None
            if clip_intr is not None:
                cand_intr = clip_intr[fi] if clip_intr.ndim == 2 else clip_intr
            rec = _candidate_record(ref_c2w, poses[fi], cand_intr, args)
            rec["ci"], rec["fi"] = ci, fi
            records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 普查核心：逐窗跑挖掘器 + 累积统计
# ═══════════════════════════════════════════════════════════════════════════

def _pct(vals: List[float]):
    """min / median / p90 / max（空列表 → 全 None）。"""
    if not vals:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def run_census(windows, args) -> dict:
    """对一批窗口跑挖掘器普查。

    windows: 可迭代，逐项 yield dict：
        {"episode": str, "target_clip_idx": int, "num_frames": int,
         "context_clips": List[{"poses": [1,F,4,4] tensor/np, "intrinsics": ...}],
         "ref_c2w": [4,4] tensor/np}
    返回 summary dict（同时打印到 stdout）。
    """
    n_windows = 0
    n_hit = 0
    all_cand = {"dist_native": [], "dist_m": [], "yaw_deg": [], "score": []}
    sel_cand = {"dist_native": [], "dist_m": [], "yaw_deg": [], "score": []}
    interframe_native: List[float] = []    # 帧间（同 clip 连续帧）位移
    intrawindow_native: List[float] = []   # 窗内（首帧 vs 各 clip 首帧）位移
    hit_spotchecks: List[dict] = []

    # 阈值扫描累加器
    sweep_min = {s: 0 for s in args.sweep_min_scores}
    sweep_pos = {p: 0 for p in args.sweep_pos_temps}

    for w in windows:
        n_windows += 1
        context_clips = w["context_clips"]
        ref_c2w = _to_c2w_np(w["ref_c2w"])
        rng = random.Random(args.seed * 1_000_003 + n_windows)

        # ---- 挖掘器主调用（镜像 _select_anchor 的共视路径，num_frames=1）----
        picks = select_covisible_anchor_frames(
            context_clips, ref_c2w,
            num_frames=1,
            min_score=args.min_covis_score,
            rng=rng,
            exclude_adjacent_frames=_EXCLUDE_ADJACENT_FRAMES,
            use_fov=args.use_fov,
        )
        hit = len(picks) >= 1
        n_hit += int(hit)

        # ---- 全候选原始分布 ----
        records = _enumerate_candidates(context_clips, ref_c2w, args)
        by_cifi = {(r["ci"], r["fi"]): r for r in records}
        for r in records:
            for k in all_cand:
                all_cand[k].append(r[k])

        # ---- 选中 anchor 的量 ----
        for ci, fi in picks:
            r = by_cifi.get((ci, fi))
            if r is not None:
                for k in sel_cand:
                    sel_cand[k].append(r[k])

        # ---- 单位 sanity：帧间 & 窗内位移（原生单位）----
        for clip in context_clips:
            poses = _to_poses_np(clip["poses"])
            xz = poses[:, [0, 2], 3]                       # [F,2]
            if xz.shape[0] >= 2:
                d = np.linalg.norm(np.diff(xz, axis=0), axis=1)
                interframe_native.extend(d.tolist())
        ref_xz = _xz_pos(ref_c2w)
        for clip in context_clips:
            poses = _to_poses_np(clip["poses"])
            intrawindow_native.append(
                float(np.linalg.norm(poses[0, [0, 2], 3] - ref_xz)))

        # ---- 阈值扫描 ----
        for s in args.sweep_min_scores:
            p = select_covisible_anchor_frames(
                context_clips, ref_c2w, num_frames=1, min_score=s,
                rng=random.Random(args.seed * 7 + n_windows),
                exclude_adjacent_frames=_EXCLUDE_ADJACENT_FRAMES, use_fov=args.use_fov)
            sweep_min[s] += int(len(p) >= 1)
        for pt in args.sweep_pos_temps:
            p = select_covisible_anchor_frames(
                context_clips, ref_c2w, num_frames=1, min_score=args.min_covis_score,
                rng=random.Random(args.seed * 11 + n_windows),
                exclude_adjacent_frames=_EXCLUDE_ADJACENT_FRAMES, use_fov=args.use_fov,
                pos_temp_m=pt)
            sweep_pos[pt] += int(len(p) >= 1)

        # ---- 命中窗抽样 dump（收集全部命中，末尾随机取 ~10）----
        if hit:
            ci, fi = picks[0]
            r = by_cifi.get((ci, fi), {})
            hit_spotchecks.append({
                "episode": w["episode"],
                "target_frame_global_idx": int(w["target_clip_idx"]) * int(w["num_frames"]),
                "picked_clip_idx": ci,
                "picked_frame_idx": fi,
                "pos_dist_m": round(r.get("dist_m", float("nan")), 3),
                "yaw_diff_deg": round(r.get("yaw_deg", float("nan")), 1),
                "covis_score": round(r.get("score", float("nan")), 3),
            })

    hit_rate = n_hit / max(1, n_windows)

    # 抽样 dump：从命中窗里随机取 ~10（seeded）
    spot_rng = random.Random(args.seed)
    spot = list(hit_spotchecks)
    spot_rng.shuffle(spot)
    spot = spot[:10]

    summary = {
        "config": {
            "dataset_dir": getattr(args, "dataset_dir", None),
            "phase": getattr(args, "phase", None),
            "split": getattr(args, "split", None),
            "max_context_clips": args.max_context_clips,
            "min_covis_score": args.min_covis_score,
            "use_fov": args.use_fov,
            "seed": args.seed,
            "POSE_UNITS_PER_METER": POSE_UNITS_PER_METER,
            "COVIS_POS_TEMP_M": COVIS_POS_TEMP_M,
            "COVIS_YAW_TEMP_DEG": COVIS_YAW_TEMP_DEG,
            "exclude_adjacent_frames": _EXCLUDE_ADJACENT_FRAMES,
        },
        "n_windows": n_windows,
        "n_hit": n_hit,
        "hit_rate": hit_rate,
        "dist_all_candidates": {
            "native": _pct(all_cand["dist_native"]),
            "meters": _pct(all_cand["dist_m"]),
        },
        "yaw_all_candidates_deg": _pct(all_cand["yaw_deg"]),
        "score_all_candidates": _pct(all_cand["score"]),
        "dist_selected_meters": _pct(sel_cand["dist_m"]),
        "yaw_selected_deg": _pct(sel_cand["yaw_deg"]),
        "score_selected": _pct(sel_cand["score"]),
        "unit_sanity": {
            "interframe_disp_native": _pct(interframe_native),
            "interframe_disp_m": _pct([x / POSE_UNITS_PER_METER for x in interframe_native]),
            "intrawindow_disp_native": _pct(intrawindow_native),
            "intrawindow_disp_m": _pct([x / POSE_UNITS_PER_METER for x in intrawindow_native]),
        },
        "sweep_min_score": {str(s): sweep_min[s] / max(1, n_windows)
                            for s in args.sweep_min_scores},
        "sweep_pos_temp_m": {str(p): sweep_pos[p] / max(1, n_windows)
                             for p in args.sweep_pos_temps},
        "spot_checks": spot,
    }
    _print_summary(summary)
    return summary


def _fmt(d: dict) -> str:
    if d.get("n", 0) == 0:
        return "  (无候选)"
    return (f"  n={d['n']:<7d} min={d['min']:.4g}  median={d['median']:.4g}  "
            f"p90={d['p90']:.4g}  max={d['max']:.4g}")


def _print_summary(s: dict) -> None:
    c = s["config"]
    print("=" * 78)
    print("COVIS CENSUS — 共视 anchor 挖掘器真数据标定普查")
    print("=" * 78)
    print(f"dataset_dir = {c['dataset_dir']}  phase={c['phase']} split={c['split']}")
    print(f"POSE_UNITS_PER_METER = {c['POSE_UNITS_PER_METER']}  "
          f"COVIS_POS_TEMP_M={c['COVIS_POS_TEMP_M']}  COVIS_YAW_TEMP_DEG={c['COVIS_YAW_TEMP_DEG']}")
    print(f"min_covis_score={c['min_covis_score']}  use_fov={c['use_fov']}  "
          f"exclude_adjacent={c['exclude_adjacent_frames']}  seed={c['seed']}  "
          f"max_context_clips={c['max_context_clips']}")
    print("-" * 78)
    print(f">>> HIT RATE = {s['hit_rate']*100:.1f}%  "
          f"({s['n_hit']}/{s['n_windows']} 窗至少挖出 1 个共视 anchor)  <<<")
    print("    (近 0 → 阈值/单位很可能错；近 100 → 可能太松)")
    print("-" * 78)
    print("[全候选帧] XZ 位置差（原生单位）:");   print(_fmt(s["dist_all_candidates"]["native"]))
    print("[全候选帧] XZ 位置差（米）:");          print(_fmt(s["dist_all_candidates"]["meters"]))
    print("[全候选帧] yaw 差（度）:");             print(_fmt(s["yaw_all_candidates_deg"]))
    print("[全候选帧] covis_score:");             print(_fmt(s["score_all_candidates"]))
    print("[选中 anchor] XZ 位置差（米）:");       print(_fmt(s["dist_selected_meters"]))
    print("[选中 anchor] yaw 差（度）:");          print(_fmt(s["yaw_selected_deg"]))
    print("[选中 anchor] covis_score:");          print(_fmt(s["score_selected"]))
    print("-" * 78)
    print("[单位 sanity] 帧间位移（同 clip 连续帧）:")
    print("  native:"); print(_fmt(s["unit_sanity"]["interframe_disp_native"]))
    print("  meters:"); print(_fmt(s["unit_sanity"]["interframe_disp_m"]))
    print("[单位 sanity] 窗内位移（各 clip 首帧 vs ref）:")
    print("  native:"); print(_fmt(s["unit_sanity"]["intrawindow_disp_native"]))
    print("  meters:"); print(_fmt(s["unit_sanity"]["intrawindow_disp_m"]))
    print("  (人判：玩家在一个 clip / 一个窗内走过的米数是否合理？不合理 → 单位设错)")
    print("-" * 78)
    print("[阈值扫描] hit-rate @ min_score:")
    for k, v in s["sweep_min_score"].items():
        print(f"    min_score={k:<5}: {v*100:.1f}%")
    print("[阈值扫描] hit-rate @ pos_temp(m) (min_score 固定):")
    for k, v in s["sweep_pos_temp_m"].items():
        print(f"    pos_temp={k:<5}m: {v*100:.1f}%")
    print("-" * 78)
    print(f"[命中窗抽样] (随机 {len(s['spot_checks'])} 个命中窗；人眼判是否同地点):")
    print(f"    {'episode':<22} {'tgt_gidx':>9} {'pick(ci,fi)':>12} "
          f"{'dist_m':>8} {'yaw°':>7} {'score':>7}")
    for sc in s["spot_checks"]:
        print(f"    {str(sc['episode']):<22} {sc['target_frame_global_idx']:>9} "
              f"{'('+str(sc['picked_clip_idx'])+','+str(sc['picked_frame_idx'])+')':>12} "
              f"{sc['pos_dist_m']:>8} {sc['yaw_diff_deg']:>7} {sc['covis_score']:>7}")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════
# 真数据 window 生成（复用 CSGOMultiClipDataset 的窗口构造，只加载 poses/intrinsics）
# ═══════════════════════════════════════════════════════════════════════════

def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """与 train_v4_stage1_dual._pad_or_truncate 同款（末帧复制补齐）。"""
    if len(arr) >= target_len:
        return arr[:target_len]
    rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
    pad = np.tile(arr[-1:], rep_shape)
    return np.concatenate([arr, pad], axis=0)


def _load_clip_geom(dataset_dir: str, row: dict, num_frames: int, want_intr: bool):
    """按 __getitem__ 的路径约定加载单 clip 的 poses(+intrinsics)，返回 collate 风格 dict
    （tensor 带 batch 维 [1,F,4,4]，这里用 np，covisibility 的 _to_* 容忍）。"""
    ep_id = row["episode_id"]
    clip_idx = int(row["clip_idx"])
    clip_dir = join(dataset_dir, f"clips/{ep_id}/{ep_id}_clip{clip_idx:02d}")
    poses = np.load(join(clip_dir, "poses.npy"))
    poses = _pad_or_truncate(poses, num_frames).astype(np.float32)
    out = {"poses": poses[None]}                          # [1,F,4,4]（模拟 collate batch 维）
    if want_intr:
        intr = np.load(join(clip_dir, "intrinsics.npy"))
        out["intrinsics"] = _pad_or_truncate(intr, num_frames).astype(np.float32)[None]
    return out, ep_id, clip_idx


def real_windows(args):
    """用真 CSGOMultiClipDataset 的窗口构造逐窗 yield（只加载几何，不解码 video）。"""
    from pipeline.v4.train_v4_stage1_dual import CSGOMultiClipDataset

    dataset = CSGOMultiClipDataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        phase=args.phase,
        max_context_clips=args.max_context_clips,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        repeat=1,
    )
    windows = dataset.samples                              # List[List[row]]，训练时的窗口
    print(f"[real] CSGOMultiClipDataset: {len(windows)} 个窗口可用，"
          f"普查前 {min(args.max_windows, len(windows))} 个。")

    order = list(range(len(windows)))
    random.Random(args.seed).shuffle(order)               # 随机抽样窗口（seeded）
    order = order[:args.max_windows]

    for wi, idx in enumerate(order):
        window = windows[idx]                             # List[row]，长度 = max_context_clips+1
        n_ctx = random.Random(args.seed * 131 + wi).randint(2, args.max_context_clips)
        # context = window[:n_ctx]，target = window[n_ctx]（与 train 循环一致）
        ctx_rows = window[:n_ctx]
        target_row = window[n_ctx]
        try:
            context_clips = []
            for r in ctx_rows:
                clip, _, _ = _load_clip_geom(args.dataset_dir, r, args.num_frames, args.use_fov)
                context_clips.append(clip)
            tgt_clip, ep_id, tgt_clip_idx = _load_clip_geom(
                args.dataset_dir, target_row, args.num_frames, args.use_fov)
        except Exception as e:                            # 缺文件 → 跳过该窗（诊断不因单窗炸）
            print(f"[real] skip window {idx}: {e}")
            continue
        ref_c2w = tgt_clip["poses"][0][0]                 # target frame-0 绝对 c2w
        yield {
            "episode": ep_id,
            "target_clip_idx": tgt_clip_idx,
            "num_frames": args.num_frames,
            "context_clips": context_clips,
            "ref_c2w": ref_c2w,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 合成 smoke（不需真数据，证明普查逻辑端到端可跑）
# ═══════════════════════════════════════════════════════════════════════════

def _make_c2w(x: float, z: float, yaw_deg: float) -> np.ndarray:
    """构造 c2w，使 yaw_deg_from_c2w(pose)==yaw_deg（与 covisibility._make_c2w 同款）。"""
    pose = np.eye(4, dtype=np.float32)
    yaw = math.radians(yaw_deg)
    pose[0, 2] = math.sin(yaw)
    pose[2, 2] = math.cos(yaw)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def synthetic_windows(args):
    """造几个合成窗口：部分 context clip 与 ref 共视（同地点微移），部分远处，覆盖命中/未命中。"""
    upm = POSE_UNITS_PER_METER
    F = args.num_frames
    n_win = 6
    for wi in range(n_win):
        wr = random.Random(1000 + wi)
        ref = _make_c2w(0.0, 0.0, 0.0)
        # 造 3 个 context clip：clip0 共视（近 ref，缓慢移动），clip1 远处，clip2 中等
        near = np.stack([_make_c2w((0.2 + 0.02 * k) * upm, 0.05 * k * upm, float(k % 20))
                         for k in range(F)])
        far = np.stack([_make_c2w(0.0, (25.0 + 0.1 * k) * upm, 180.0) for k in range(F)])
        mid = np.stack([_make_c2w((3.0 + 0.03 * k) * upm, 2.0 * upm, 20.0 + 0.1 * k)
                        for k in range(F)])
        # 部分窗把 near 换成远处 → 制造未命中窗，让命中率 < 100%
        if wi % 3 == 0:
            near = far.copy()
        clips = [{"poses": near[None]}, {"poses": far[None]}, {"poses": mid[None]}]
        wr.shuffle(clips)
        yield {
            "episode": f"synth_ep{wi:02d}",
            "target_clip_idx": wi + 1,
            "num_frames": F,
            "context_clips": clips,
            "ref_c2w": ref,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="共视 anchor 挖掘器真数据标定普查（covisibility.py 上线前必跑）")
    p.add_argument("--dataset_dir", type=str,
                   default="/home/nvme02/Memory-dataset/v4_dynamic_481e3d7795739da3",
                   help="CSGO 预处理数据集根目录（含 metadata_{phase}_{split}.csv）；同 run_train_v6.sh 默认")
    p.add_argument("--metadata", type=str, default=None,
                   help="可选：显式 metadata CSV 路径（形如 .../metadata_{phase}_{split}.csv）；"
                        "给出则从其 basename 解析 phase/split 并覆盖，dirname 覆盖 dataset_dir")
    p.add_argument("--phase", type=str, default="verify",
                   choices=["exp", "full", "verify"],
                   help="数据集 phase（同 train_v6 默认 verify）")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_windows", type=int, default=500, help="普查采样的窗口数上限")
    p.add_argument("--max_context_clips", type=int, default=6,
                   help="同 train_v6：n_ctx~Uniform(2,max_context_clips)，window_size=+1")
    p.add_argument("--min_covis_score", type=float, default=DEFAULT_MIN_COVIS_SCORE,
                   help="共视 anchor 最低分（挖掘器主调用用；默认 DEFAULT_MIN_COVIS_SCORE）")
    p.add_argument("--use_fov", action="store_true", default=False,
                   help="共视打分叠加 FOV 精修（需 clip['intrinsics']）")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", default=False,
                   help="合成 smoke 模式：不读真数据，证明普查逻辑端到端可跑")
    p.add_argument("--out_json", type=str, default=None, help="可选：把 summary dump 成 JSON")
    # 扫描网格（可覆盖）
    p.add_argument("--sweep_min_scores", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--sweep_pos_temps", type=float, nargs="+", default=[2.0, 4.0, 8.0])

    args = p.parse_args()
    if args.metadata:
        base = os.path.basename(args.metadata)
        stem = base[:-4] if base.endswith(".csv") else base
        parts = stem.split("_")           # metadata_{phase}_{split}
        if len(parts) >= 3 and parts[0] == "metadata":
            args.phase, args.split = parts[1], parts[2]
        args.dataset_dir = os.path.dirname(args.metadata) or args.dataset_dir
    return args


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.synthetic:
        print("[synthetic] 合成 smoke 模式（不读真数据）")
        windows = synthetic_windows(args)
    else:
        windows = real_windows(args)

    summary = run_census(windows, args)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[out_json] summary 已写入 {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
