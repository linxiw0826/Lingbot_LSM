"""
covisibility.py — 共视（co-visible）较早帧挖掘，用作 v6 latent-concat 记忆 anchor
================================================================================

背景（decisions.md 讨论 13 / D-09'，findings F-33）
--------------------------------------------------
v6 = 冻结 Wan2.2-i2v + r128 LoRA + latent-concat 记忆（把一个 clean anchor 帧 latent 拼到
时间维尾部）。当前 `train_v6._select_anchor` **随机**选 anchor（`random.choice(context_clips)`
+ `random.randint(frame)`）。Context-as-Memory（arXiv 2506.03141）Table 3 消融显示随机选是
**最差臂**（17.70 PSNR），而 FOV-相关的同场景选帧到 20.1–20.2（+2.4 PSNR）。

本模块的修复：**用相机位姿挑一个「共视」（同一地点、视野重叠）的较早帧**当 anchor。
关键 insight：挖 *共视*（任意较早帧的相机视野与目标视野重叠），**不是严格重访**——
共视远比重访丰富，天然化解 revisit 稀疏（D-07）的数据问题。

与 `revisit_data_spec.py` 的关系
--------------------------------
- **复用**其相机几何工具，口径严格一致（不重新发明 yaw / 位置 / 角差公式）：
    * 世界位置 = `c2w[:3, 3]`；BEV 距离取 (x, z) 两维（`revisit_data_spec._xz` 同款 [0,2]）。
    * yaw = `revisit_data_spec.yaw_deg_from_c2w`（atan2(R[0,2], R[2,2])）。
    * 角差 = `revisit_data_spec.angular_diff_deg`（周期 360°，落 [0,180]）。
    * 单位 = `revisit_data_spec.POSE_UNITS_PER_METER`（米→数据集原生单位换算，见其顶部警告）。
- **不同点**：revisit 要求「离开过再回来 + 严格 1m/30° + 时间窗」；共视更宽松，只问
  「两帧相机是否看着大致同一片地方」，因此位置 / yaw 用**软衰减打分**（非硬阈值），
  阈值也更松（见下方默认常量）。

设计约束（与本任务一致）
------------------------
- 纯 numpy + 标准库，可选 torch（仅用于把 clip["poses"] 张量转 numpy）。无模型 / VAE / IO。
- **不使用全局 `random` / `np.random`**：采样必须传入 `rng`（`random.Random` 或
  `np.random.Generator`），保证确定性。本模块只用 `rng.random()`（两类 rng 都有）。
- 打分向量化到一个 clip 的 F 帧；候选池小（几个 clip × 81 帧），清晰 > 微优化。

API
---
- `covisibility_score(ref_c2w, cand_c2w, ref_intr=None, cand_intr=None, *, use_fov=False, ...) -> float`
    位姿邻近打分（默认，WorldMem 式），可选 FOV 重叠精修。返回 [0,1]。
- `select_covisible_anchor_frames(context_clips, ref_c2w, num_frames, min_score, rng, ...) -> List[(clip_idx, frame_idx)]`
    对所有较早 context 帧打分，保留 ≥ min_score 者，按分数加权采样 num_frames 个。无候选返回 []。

    可 `python src/pipeline/v6/covisibility.py` 跑合成 smoke test。
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 复用 revisit_data_spec 的相机几何工具（口径必须一致）。
# import 路径与 train_v6 对齐：把 src/ 放上 sys.path 后 `pipeline.data.*` 可导入。
# ---------------------------------------------------------------------------
_V6_DIR = os.path.dirname(os.path.abspath(__file__))          # src/pipeline/v6/
_PIPELINE_DIR = os.path.dirname(_V6_DIR)                       # src/pipeline/
_SRC_DIR = os.path.dirname(_PIPELINE_DIR)                      # src/
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pipeline.data.revisit_data_spec import (  # noqa: E402
    POSE_UNITS_PER_METER,
    angular_diff_deg,
    yaw_deg_from_c2w,
)

# ═══════════════════════════════════════════════════════════════════════════
# 模块级默认常量（首猜值 —— 真数据上须重新标定，见每条注释）
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠️ 这些默认是**未在真数据上标定的首猜值**。上线前请先对训练集跑一遍 census
#    （把每对 (ref, 较早帧) 的 pos 距离 / yaw 差 / covis_score 汇总成分布），
#    再据分布定 temperature 和 min_score，别盲信默认。共视本应丰富（这是本 idea 的前提），
#    若默认下几乎选不出候选，多半是 temperature 太紧或 POSE_UNITS_PER_METER 设错。

# 位置软衰减温度（**米**）：位置差达该值时位置项 ≈ 0.607（高斯 exp(-0.5)）。
# 比 revisit 的硬阈值 HIT_DIST_M=1m 松——共视允许「附近不同站位」也算看同一片地方。
COVIS_POS_TEMP_M = 4.0

# yaw 软衰减温度（**度**）：yaw 差达该值时朝向项 ≈ 0.607。取 ~35°（任务建议 30–45°），
# 与 revisit HIT_YAW_DEG=30° 同量级但略松，容忍视野边缘重叠。
COVIS_YAW_TEMP_DEG = 35.0

# 采样默认最低分：低于此的候选帧不参与加权采样。占位保守值，须随分布调。
DEFAULT_MIN_COVIS_SCORE = 0.30

# —— 可选 FOV 精修（use_fov=True 时启用，Context-as-Memory §3.3 四射线扇 / 视锥重叠）——
# 无 intrinsics 时退化的水平半 FOV（度）；90° 全 FOV → 半角 45°。
FOV_DEFAULT_HALF_DEG = 45.0
# 视锥在 XZ 平面的最大有效视距（**米**）——扇形远端裁剪，避免无限远误判重叠。
FOV_MAX_RANGE_M = 8.0
# FOV 只作**精修**：最终分 = pose_score * (FOV_FLOOR + (1-FOV_FLOOR) * overlap)，
# 使 FOV 不能把一个好的位姿匹配彻底清零，只做加权。
FOV_FLOOR = 0.25
# 采样 ref 视锥内点数（深度 × 横向），用于点在 cand 视锥内的占比估计。
_FOV_N_DEPTH = 6
_FOV_N_LATERAL = 5


# ═══════════════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════════════

def _to_c2w_np(pose) -> np.ndarray:
    """把单个 c2w 转成 numpy [4,4] float32；容忍 torch 张量 / 前导 singleton 维。

    caller（train_v6._select_anchor）里 clip["poses"] 是 torch，且真实存储常带 batch 维
    [1,F,4,4]（train_v6 用 .squeeze(0) 去掉）。本函数对**单个** pose 做 squeeze 到 [4,4]。
    """
    if hasattr(pose, "detach"):          # torch.Tensor
        pose = pose.detach().cpu().numpy()
    arr = np.asarray(pose, dtype=np.float32)
    arr = np.squeeze(arr)                # 去掉前导/尾随 singleton
    if arr.shape != (4, 4):
        raise ValueError(f"c2w 必须可 squeeze 到 [4,4]，got {arr.shape}")
    return arr


def _to_poses_np(clip_poses) -> np.ndarray:
    """把一个 clip 的 poses 转成 numpy [F,4,4]；容忍 torch / 前导 singleton（[1,F,4,4]）。"""
    if hasattr(clip_poses, "detach"):
        clip_poses = clip_poses.detach().cpu().numpy()
    arr = np.asarray(clip_poses, dtype=np.float32)
    # 去掉多余前导 singleton，直到 3 维 [F,4,4]
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"clip poses 必须可整理成 [F,4,4]，got {arr.shape}")
    return arr


def _xz_pos(c2w: np.ndarray) -> np.ndarray:
    """世界位置的 (x, z) 两维（与 revisit_data_spec._xz 口径一致：取平移的 [0,2]）。"""
    return c2w[[0, 2], 3]


def _forward_xz(c2w: np.ndarray) -> np.ndarray:
    """相机前向在 XZ 平面的单位向量，与 yaw_deg_from_c2w 同源：
    yaw = atan2(R[0,2], R[2,2]) → 前向 ∝ (R[0,2], R[2,2])（即相机 z 轴投到 XZ）。"""
    R = c2w[:3, :3]
    f = np.array([R[0, 2], R[2, 2]], dtype=np.float32)
    n = float(np.linalg.norm(f))
    if n < 1e-8:
        return np.array([0.0, 1.0], dtype=np.float32)
    return f / n


def _half_fov_deg(intr: Optional[Sequence[float]]) -> float:
    """从 intrinsics [fx, fy, cx, cy] 估水平半 FOV（度）：half = atan(cx / fx)。
    intr 为 None 时退化到 FOV_DEFAULT_HALF_DEG。"""
    if intr is None:
        return FOV_DEFAULT_HALF_DEG
    a = np.asarray(intr, dtype=np.float32).reshape(-1)
    if a.shape[0] < 3:
        return FOV_DEFAULT_HALF_DEG
    fx, cx = float(a[0]), float(a[2])
    if fx <= 1e-6 or cx <= 0.0:
        return FOV_DEFAULT_HALF_DEG
    return math.degrees(math.atan2(cx, fx))


def _pose_proximity_score(
    ref_c2w: np.ndarray,
    cand_c2w: np.ndarray,
    *,
    pos_temp_m: float,
    yaw_temp_deg: float,
    pose_units_per_meter: float,
) -> float:
    """WorldMem 式位姿邻近分 ∈ [0,1]：位置近 且 yaw 近时高，各按高斯软衰减，取乘积
    （两者都要满足才高分）。距离在 XZ 平面、数据集原生单位；温度由米换算到原生单位。"""
    pos_temp_native = max(1e-6, pos_temp_m * pose_units_per_meter)
    dist = float(np.linalg.norm(_xz_pos(ref_c2w) - _xz_pos(cand_c2w)))  # 原生单位
    dyaw = angular_diff_deg(yaw_deg_from_c2w(ref_c2w), yaw_deg_from_c2w(cand_c2w))  # 度
    pos_term = math.exp(-0.5 * (dist / pos_temp_native) ** 2)
    yaw_term = math.exp(-0.5 * (dyaw / max(1e-6, yaw_temp_deg)) ** 2)
    return float(pos_term * yaw_term)


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """向量化点在三角形内判定（含边）。p:[N,2], a/b/c:[2] → bool[N]。符号一致法。"""
    def cross(o, u, v):
        return (u[..., 0] - o[0]) * (v[1] - o[1]) - (u[..., 1] - o[1]) * (v[0] - o[0])
    d1 = cross(a, p, b)
    d2 = cross(b, p, c)
    d3 = cross(c, p, a)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _fov_overlap_xz(
    ref_c2w: np.ndarray,
    cand_c2w: np.ndarray,
    ref_intr: Optional[Sequence[float]],
    cand_intr: Optional[Sequence[float]],
    *,
    pose_units_per_meter: float,
) -> float:
    """轻量 FOV 重叠精修（Context-as-Memory §3.3 视锥扇思路，简化到 XZ 平面）∈ [0,1]。

    把每个相机视野建成 XZ 平面上的三角扇（apex=相机位置，两条边射线 = 前向 ±half_fov，
    延伸到 FOV_MAX_RANGE_M）。在 ref 视锥内均匀采点，返回落进 cand 视锥的点占比
    = 「ref 看到的地方有多少也被 cand 看到」的方向性重叠代理。纯几何、无外部依赖。
    """
    rng_m = FOV_MAX_RANGE_M * pose_units_per_meter
    p_ref, p_cand = _xz_pos(ref_c2w), _xz_pos(cand_c2w)
    f_ref, f_cand = _forward_xz(ref_c2w), _forward_xz(cand_c2w)
    half_ref = math.radians(_half_fov_deg(ref_intr))
    half_cand = math.radians(_half_fov_deg(cand_intr))

    def _tri(p, f, half):
        # 边射线 = 前向绕 XZ 旋转 ±half；三角形 = apex + 两远端角点
        cs, sn = math.cos(half), math.sin(half)
        left = np.array([f[0] * cs - f[1] * sn, f[0] * sn + f[1] * cs], dtype=np.float32)
        right = np.array([f[0] * cs + f[1] * sn, -f[0] * sn + f[1] * cs], dtype=np.float32)
        return p, p + left * rng_m, p + right * rng_m

    a_r, b_r, c_r = _tri(p_ref, f_ref, half_ref)
    a_c, b_c, c_c = _tri(p_cand, f_cand, half_cand)

    # 在 ref 视锥内采点（深度 × 横向，barycentric-ish：沿前向不同深度 + 横向张开）
    perp = np.array([f_ref[1], -f_ref[0]], dtype=np.float32)
    tan_h = math.tan(half_ref)
    depths = np.linspace(0.15, 1.0, _FOV_N_DEPTH, dtype=np.float32) * rng_m
    lats = np.linspace(-1.0, 1.0, _FOV_N_LATERAL, dtype=np.float32)
    pts = []
    for d in depths:
        base = p_ref + f_ref * d
        for t in lats:
            pts.append(base + perp * (t * d * tan_h))
    pts = np.asarray(pts, dtype=np.float32)                       # [N,2]

    inside = _point_in_triangle(pts, a_c, b_c, c_c)
    return float(inside.mean()) if pts.shape[0] else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════════

def covisibility_score(
    ref_c2w,
    cand_c2w,
    ref_intr=None,
    cand_intr=None,
    *,
    use_fov: bool = False,
    pos_temp_m: float = COVIS_POS_TEMP_M,
    yaw_temp_deg: float = COVIS_YAW_TEMP_DEG,
    pose_units_per_meter: float = POSE_UNITS_PER_METER,
) -> float:
    """给一个候选帧相机相对参考帧相机的**共视度**打分，返回 [0,1]（越高越同视野）。

    必选路径 = 位姿邻近（WorldMem 式）：位置差（XZ 原生单位）小 且 yaw 差小 → 高分，
    各按高斯软衰减取乘积。可选 `use_fov=True` 时用 intrinsics 做 FOV 视锥重叠精修
    （见 `_fov_overlap_xz`），最终分 = pose_score * (FOV_FLOOR + (1-FOV_FLOOR)*overlap)。

    Args:
        ref_c2w:   参考帧 camera-to-world，可 [4,4] / torch / 带前导 singleton。
                   在 caller 里 = query clip 第 0 帧的绝对 c2w。
        cand_c2w:  候选（较早）帧 c2w，同上。
        ref_intr / cand_intr: intrinsics [fx, fy, cx, cy]，仅 use_fov=True 时使用；None 退化默认 FOV。
        use_fov:   是否叠加 FOV 精修（默认 False——位姿邻近是必选、已足够鲁棒的路径）。
        pos_temp_m / yaw_temp_deg: 软衰减温度（米 / 度），默认见模块常量（首猜，须标定）。
        pose_units_per_meter: 平移单位换算（复用 revisit_data_spec.POSE_UNITS_PER_METER）。

    Returns:
        float ∈ [0,1]。1 ≈ 同地点同朝向（几乎同一视野）；0 ≈ 远处 / 背对。
    """
    ref = _to_c2w_np(ref_c2w)
    cand = _to_c2w_np(cand_c2w)
    score = _pose_proximity_score(
        ref, cand,
        pos_temp_m=pos_temp_m, yaw_temp_deg=yaw_temp_deg,
        pose_units_per_meter=pose_units_per_meter,
    )
    if use_fov:
        overlap = _fov_overlap_xz(
            ref, cand, ref_intr, cand_intr,
            pose_units_per_meter=pose_units_per_meter,
        )
        score *= FOV_FLOOR + (1.0 - FOV_FLOOR) * overlap
    return float(score)


def _weighted_sample_without_replacement(
    n_items: int, weights: np.ndarray, k: int, rng
) -> List[int]:
    """按 weights 无放回加权抽 k 个下标。Efraimidis–Spirakis：key = u**(1/w)，取 top-k。
    只用 rng.random()（random.Random 与 np.random.Generator 都支持）→ 确定性、无全局 random。
    k >= n_items 时返回全部下标（按 key 降序，稳定确定）。"""
    k = min(k, n_items)
    keys = np.empty(n_items, dtype=np.float64)
    for i in range(n_items):
        u = float(rng.random())
        w = float(weights[i])
        if w <= 0.0:
            keys[i] = -np.inf
        else:
            # u**(1/w) 单调等价于 (1/w)*ln(u)，用后者数值稳定
            keys[i] = math.log(max(u, 1e-300)) / w
        # 注：keys 越大越优先（ln(u)<0，w 越大 key 越接近 0 越大）
    order = np.argsort(-keys)          # 降序
    return [int(idx) for idx in order[:k]]


def select_covisible_anchor_frames(
    context_clips: List[dict],
    ref_c2w,
    num_frames: int,
    min_score: float,
    rng,
    *,
    ref_intr=None,
    use_fov: bool = False,
    exclude_adjacent_frames: int = 0,
    pos_temp_m: float = COVIS_POS_TEMP_M,
    yaw_temp_deg: float = COVIS_YAW_TEMP_DEG,
    pose_units_per_meter: float = POSE_UNITS_PER_METER,
) -> List[Tuple[int, int]]:
    """在所有较早 context clip 的帧里，挑与 `ref_c2w`（query 参考位姿）**共视**的帧作 anchor。

    流程：对每个 clip 的每帧算 covisibility_score → 保留 score ≥ min_score 的候选 →
    按 score 无放回加权采样 num_frames 个（用传入 rng，确定性）。无候选过阈值 → 返回 []
    （caller 据其 dropout 策略回退到空 / 随机）。

    Args:
        context_clips: 早于 query 的 context clip 列表，**假设按时间升序**（列表末尾 = 最靠近
            query）。每个 clip 是 dict，至少含 `clip["poses"]`（torch/np，可整理成 [F,4,4]）。
            use_fov=True 时可含 `clip["intrinsics"]`（[F,4] 或单条 [4]）。
        ref_c2w:  query 参考位姿（caller = query clip 第 0 帧绝对 c2w）。
        num_frames: 要选的 anchor 帧数（= args.num_anchor_frames）。<=0 或空列表 → []。
        min_score: 候选最低共视分（见 DEFAULT_MIN_COVIS_SCORE，首猜须标定）。
        rng:      `random.Random` 或 `np.random.Generator`；仅用其 .random()，不碰全局 random。
        ref_intr: query 参考 intrinsics，仅 use_fov=True 用。
        use_fov:  是否叠加 FOV 精修。
        exclude_adjacent_frames: 排除**时间上紧邻 query 的帧**——即**最后一个** context clip 的
            末尾这么多帧（它们是 query 的平凡续帧，不算「记忆」）。默认 0 = 不排除。

    Returns:
        List[(clip_idx, frame_idx)]，长度 ≤ num_frames；无合格候选返回 []。
        clip_idx 是 context_clips 的下标，frame_idx 是该 clip 内帧号。
    """
    if not context_clips or num_frames <= 0:
        return []

    last_clip_idx = len(context_clips) - 1
    cand_clip_ids: List[int] = []
    cand_frame_ids: List[int] = []
    cand_scores: List[float] = []

    for ci, clip in enumerate(context_clips):
        poses = _to_poses_np(clip["poses"])        # [F,4,4]
        n_frames = poses.shape[0]

        # 排除紧邻 query 的帧：仅最后一个 context clip 的末尾 exclude_adjacent_frames 帧
        hi = n_frames
        if ci == last_clip_idx and exclude_adjacent_frames > 0:
            hi = max(0, n_frames - exclude_adjacent_frames)

        # 可选取该 clip 的 intrinsics（use_fov 才需要）
        clip_intr = None
        if use_fov and ("intrinsics" in clip) and (clip["intrinsics"] is not None):
            intr_arr = clip["intrinsics"]
            if hasattr(intr_arr, "detach"):
                intr_arr = intr_arr.detach().cpu().numpy()
            clip_intr = np.squeeze(np.asarray(intr_arr, dtype=np.float32))

        for fi in range(hi):
            cand_intr = None
            if clip_intr is not None:
                cand_intr = clip_intr[fi] if clip_intr.ndim == 2 else clip_intr
            s = covisibility_score(
                ref_c2w, poses[fi], ref_intr, cand_intr,
                use_fov=use_fov,
                pos_temp_m=pos_temp_m, yaw_temp_deg=yaw_temp_deg,
                pose_units_per_meter=pose_units_per_meter,
            )
            if s >= min_score:
                cand_clip_ids.append(ci)
                cand_frame_ids.append(fi)
                cand_scores.append(s)

    if not cand_scores:
        return []

    weights = np.asarray(cand_scores, dtype=np.float64)
    picks = _weighted_sample_without_replacement(len(cand_scores), weights, num_frames, rng)
    return [(cand_clip_ids[i], cand_frame_ids[i]) for i in picks]


# ═══════════════════════════════════════════════════════════════════════════
# smoke test（合成位姿，不需真实数据）：run `python src/pipeline/v6/covisibility.py`
# ═══════════════════════════════════════════════════════════════════════════

def _make_c2w(x: float, z: float, yaw_deg: float) -> np.ndarray:
    """构造 c2w：平移 (x,_,z) + 绕 y 轴 yaw，使 yaw_deg_from_c2w(pose)==yaw_deg。
    与 revisit_data_spec._yaw_to_c2w 同款（R[0,2]=sin, R[2,2]=cos）。"""
    pose = np.eye(4, dtype=np.float32)
    yaw = math.radians(yaw_deg)
    pose[0, 2] = math.sin(yaw)
    pose[2, 2] = math.cos(yaw)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def _smoke_test() -> int:
    import random as _random_mod

    upm = POSE_UNITS_PER_METER
    failures: List[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            failures.append(msg)

    # 参考帧：原点，yaw 0
    ref = _make_c2w(0.0, 0.0, 0.0)
    # 同地点同朝向：略微偏移 0.5m，yaw 5°
    same_place = _make_c2w(0.5 * upm, 0.0, 5.0)
    # 远处（20m 外），同朝向
    far_away = _make_c2w(0.0, 20.0 * upm, 0.0)
    # 原地但转身 90°（背对侧）
    rotated = _make_c2w(0.0, 0.0, 90.0)

    s_same = covisibility_score(ref, same_place)
    s_far = covisibility_score(ref, far_away)
    s_rot = covisibility_score(ref, rotated)
    print(f"[pose-prox]  same_place={s_same:.4f}  far_away={s_far:.4f}  rotated={s_rot:.4f}")

    check(s_same > s_far, f"same_place({s_same:.3f}) 应 > far_away({s_far:.3f})")
    check(s_same > s_rot, f"same_place({s_same:.3f}) 应 > rotated({s_rot:.3f})")
    check(s_same > 0.6, f"same_place 应高分(>0.6)，got {s_same:.3f}")
    check(s_far < 0.05, f"far_away 应低分(<0.05)，got {s_far:.3f}")
    check(s_rot < 0.3, f"rotated 应偏低(<0.3)，got {s_rot:.3f}")

    # self（ref vs ref）应 ≈ 1
    check(abs(covisibility_score(ref, ref) - 1.0) < 1e-6, "自比 covis 应 ≈ 1")

    # FOV 精修路径可跑通且不抬高远处帧
    intr = np.array([320.0, 320.0, 416.0, 240.0], dtype=np.float32)  # ~ 832x480
    s_same_fov = covisibility_score(ref, same_place, intr, intr, use_fov=True)
    s_far_fov = covisibility_score(ref, far_away, intr, intr, use_fov=True)
    print(f"[+FOV]       same_place={s_same_fov:.4f}  far_away={s_far_fov:.4f}")
    check(s_same_fov > s_far_fov, "use_fov 下 same_place 仍应 > far_away")
    check(0.0 <= s_same_fov <= 1.0, "FOV 分应在 [0,1]")

    # select_covisible_anchor_frames：clip0 全共视帧、clip1 全远处帧 → 应只选 clip0
    F = 8
    clip0_poses = np.stack([_make_c2w(0.3 * upm, 0.0, float(k)) for k in range(F)])   # 近参考
    clip1_poses = np.stack([_make_c2w(0.0, 30.0 * upm, 0.0) for _ in range(F)])       # 很远
    context_clips = [
        {"poses": clip0_poses},
        {"poses": clip1_poses},
    ]

    # 两种 rng 都测：np.random.Generator + random.Random
    for rng in (np.random.default_rng(0), _random_mod.Random(0)):
        picks = select_covisible_anchor_frames(
            context_clips, ref, num_frames=3, min_score=DEFAULT_MIN_COVIS_SCORE, rng=rng,
        )
        rng_name = type(rng).__name__
        check(len(picks) == 3, f"[{rng_name}] 应选 3 帧，got {len(picks)}")
        check(all(ci == 0 for ci, _ in picks),
              f"[{rng_name}] 应只从共视 clip0 选，got {picks}")

    # 确定性：同种子两次结果一致
    p1 = select_covisible_anchor_frames(
        context_clips, ref, 3, DEFAULT_MIN_COVIS_SCORE, np.random.default_rng(42))
    p2 = select_covisible_anchor_frames(
        context_clips, ref, 3, DEFAULT_MIN_COVIS_SCORE, np.random.default_rng(42))
    check(p1 == p2, f"同种子应确定性一致，got {p1} vs {p2}")

    # 无候选过阈值 → []（把 ref 挪到极远，所有 context 都非共视）
    far_ref = _make_c2w(0.0, 1000.0 * upm, 0.0)
    empty = select_covisible_anchor_frames(
        context_clips, far_ref, 3, DEFAULT_MIN_COVIS_SCORE, np.random.default_rng(0))
    check(empty == [], f"无合格候选应返回 []，got {empty}")

    # exclude_adjacent_frames：排除最后一个 clip 末尾帧（这里两个 clip 都近以便观测）
    ctx_near = [{"poses": clip0_poses}, {"poses": clip0_poses.copy()}]
    picks_excl = select_covisible_anchor_frames(
        ctx_near, ref, num_frames=50, min_score=DEFAULT_MIN_COVIS_SCORE,
        rng=np.random.default_rng(0), exclude_adjacent_frames=F,  # 排掉整个末尾 clip
    )
    check(all(ci == 0 for ci, _ in picks_excl),
          f"exclude_adjacent 应排掉末尾 clip1 全部帧，got {picks_excl}")

    # torch 张量 / [1,F,4,4] 形状容忍
    try:
        import torch
        t_poses = torch.from_numpy(clip0_poses[None])   # [1,F,4,4]
        t_ref = torch.from_numpy(ref[None, None])       # [1,1,4,4]
        picks_t = select_covisible_anchor_frames(
            [{"poses": t_poses}], t_ref, 2, DEFAULT_MIN_COVIS_SCORE, np.random.default_rng(0))
        check(len(picks_t) == 2, f"torch/[1,F,4,4] 输入应正常，got {picks_t}")
        print("[torch]      张量 + 前导 singleton 维输入 OK")
    except ImportError:
        print("[torch]      未装 torch，跳过张量形状测试（非硬失败）")

    if failures:
        print("\nSMOKE_TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nSMOKE_TEST PASSED (all checks ok)")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
