"""
用途
-----
本模块**不是完整 preprocess 脚本**

  Part A —— 目标 v4 训练格式的规格
    - 常量：CLIP_FRAMES / TARGET_FPS / HEIGHT / WIDTH / CLIP_FILES /
      REQUIRED_METADATA_COLUMNS
    - validate_clip_dir(clip_dir)：切完一个 clip 自检是否符合 v4 格式
    - write_metadata_csv(rows, out_csv)：写 metadata CSV，强制校验必需列 +
      clip_idx/episode_idx 可转 int

  Part B —— 重访数据筛选判据（与 src/pipeline/data/explore_data.py 完全一致的判定逻辑）
    - yaw_deg_from_c2w / angular_diff_deg：几何工具（公式逐一对齐 explore_data.py）
    - is_position_revisit：判定一对帧 (i,j) 是否构成"位置重访"
    - episode_revisit_stats：扫一个 episode 统计重访密度
    - episode_qualifies：是否保留该 episode

设计约束
---------
- 纯 numpy + 标准库（csv/os/argparse/math），**不依赖 torch / 训练代码**。
- 可被 import，也可 `python revisit_data_spec.py --self_test` 跑合成数据自测。
- 重访判定逻辑必须与 explore_data.py 严格一致（dist=XZ 平面 L2、
  yaw=atan2(R[0,2],R[2,2])、中间分离=_compute_max_intermediate_separation、
  时间间隔判定），这样钰淇切完后用 explore_data.py 验收时口径对得上。

⚠️ 单位（最重要）
-----------------
原始 pose 平移单位若是 CSGO game units（≈ inches），则 1 米 ≈ 40 units；
若已经是米则把 POSE_UNITS_PER_METER 设为 1.0。
**钰淇必须按实际原始数据单位设置 POSE_UNITS_PER_METER，否则所有距离阈值失效**
（HIT_DIST_M / INTERMEDIATE_SEP_M 都以米表达，函数内会乘 POSE_UNITS_PER_METER 换算
到数据集原生单位）。我们之前踩过这个坑：默认按 game units(40)，若你的数据已是米务必改成 1.0。

═══════════════════════════════════════════════════════════════════════════
怎么用
═══════════════════════════════════════════════════════════════════════════

切完一个 episode 的所有 clip 后：

    import numpy as np
    import revisit_data_spec as spec

    # 1) 先按你的实际原始单位确认这个常量（极其重要！）
    #    若 pose 平移是 game units → 40.0；若是米 → 1.0
    units_per_m = spec.POSE_UNITS_PER_METER   # 必要时改之

    # 2) 把该 episode 的所有 clip 的 poses 按 clip_idx 升序拼接成全帧轨迹
    #    每个 clip 的 poses.npy 是 [81,4,4]，拼成 [T,4,4]
    poses_list = [np.load(p) for p in sorted_clip_poses_paths]  # 按 clip_idx 排好序
    poses_all = np.concatenate(poses_list, axis=0)              # [T,4,4]

    # 3) 算 translations / yaws（与 explore_data.py 口径一致）
    translations = poses_all[:, :3, 3]                          # [T,3]
    yaws = np.array([spec.yaw_deg_from_c2w(poses_all[t]) for t in range(len(poses_all))])

    # 4) 统计这个 episode 的重访结构
    stats = spec.episode_revisit_stats(
        translations, yaws,
        frames_per_clip=spec.CLIP_FRAMES,
        pose_units_per_meter=units_per_m,
    )

    # 5) 决定是否保留该 episode（默认阈值保守，先 census 看分布再调）
    keep = spec.episode_qualifies(stats)

    # 6) 写 metadata 时确保每行带 clip_idx（episode 内整数序号）+ episode_idx（全局整数）
    rows = [
        {"clip_path": "...", "episode_id": "player01_ep01",
         "clip_idx": 0, "episode_idx": 1, "prompt": "..."},
        ...
    ]
    spec.write_metadata_csv(rows, "metadata_all.csv")

    # 7) 切完每个 clip 自检格式
    problems = spec.validate_clip_dir(clip_dir)
    assert not problems, problems

校验对齐：切完后可直接跑
    python src/pipeline/data/explore_data.py --dataset_dir <你的输出> --metadata metadata_all.csv ...
口径应与本模块一致（distance_eps=HIT_DIST_M*units_per_m，yaw_eps=HIT_YAW_DEG，
require_intermediate_separation=INTERMEDIATE_SEP_M*units_per_m）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Part A —— 目标 v4 训练格式规格
# ═══════════════════════════════════════════════════════════════════════════
#
# 来源确认（train_v4_stage1_dual.py CSGOMultiClipDataset + dataloader）：
#   - _load_single_clip 读 video.mp4 / poses.npy / action.npy / intrinsics.npy，
#     按 num_frames=81、resize 到 (width=832, height=480) 解码视频。
#   - 视频按 target_fps=16 切（旧 preprocess_csgo_v3.py 默认 target_fps=16）。
#   - clip 目录由 clips/{episode_id}/{episode_id}_clip{clip_idx:02d} 定位
#     （train_v4_stage1_dual.py:343）；所以 metadata 必须有 episode_id + clip_idx。

CLIP_FRAMES = 81           # 每个 clip 帧数（必须 4n+1；v4 = 81）
TARGET_FPS = 16            # 切片目标帧率
HEIGHT, WIDTH = 480, 832   # 视频分辨率（dataloader resize 到此尺寸）

# 每个 clip 目录须含的 6 个文件：
CLIP_FILES = ["video.mp4", "poses.npy", "action.npy", "intrinsics.npy",
              "image.jpg", "prompt.txt"]

# 各 .npy 的期望 shape（用于 validate_clip_dir 校验）：
#   poses.npy      [81, 4, 4] float32  camera-to-world (c2w)，OpenCV 约定 x右/y下/z前
#   action.npy     [81, C] (C>=4)      v4 模型只用前 4 维 WASD（0/1）
#   intrinsics.npy [81, 4] float32     [fx, fy, cx, cy] 像素
EXPECTED_POSES_SHAPE = (CLIP_FRAMES, 4, 4)
EXPECTED_INTRINSICS_SHAPE = (CLIP_FRAMES, 4)
ACTION_MIN_DIM = 4         # action.npy 第二维至少 4（WASD）

# metadata CSV 必需列：
#   clip_path    相对/绝对 clip 目录路径
#   episode_id   字符串 episode 标识（如 player01_ep01），用于分组 + 定位
#   clip_idx     整数，clip 在 episode 内的序号（旧 preprocess 缺，必须补）
#   episode_idx  整数，episode 的全局编号（旧 preprocess 缺，必须补；
#                供 prepare_v4_splits.py 按 phase 过滤 exp/full）
REQUIRED_METADATA_COLUMNS = ["clip_path", "episode_id", "clip_idx", "episode_idx"]


def validate_clip_dir(clip_dir: str) -> List[str]:
    """检查一个 clip 目录是否符合 v4 训练格式。

    校验项：
      1. CLIP_FILES 中 6 个文件是否都存在；
      2. poses.npy shape == [81,4,4]；
      3. intrinsics.npy shape == [81,4]；
      4. action.npy shape == [81, C] 且 C >= 4；
      5. video.mp4 帧数 == 81（用 opencv 读 CAP_PROP_FRAME_COUNT；
         若环境无 opencv 则跳过该项并加一条提示，不算硬错误）。

    Args:
        clip_dir: clip 目录路径（绝对或相对均可）。

    Returns:
        问题描述字符串列表。**空列表 = 合格**；非空则每条描述一个问题。
        供钰淇切完一个 clip 后自检，例如：
            problems = validate_clip_dir(clip_dir)
            assert not problems, problems
    """
    problems: List[str] = []

    if not os.path.isdir(clip_dir):
        return [f"clip_dir 不是目录或不存在: {clip_dir}"]

    # 1. 6 个文件存在性
    for fname in CLIP_FILES:
        if not os.path.isfile(os.path.join(clip_dir, fname)):
            problems.append(f"缺少文件: {fname}")

    # 2/3/4. npy shape 校验
    def _check_npy(name: str, expected_shape, min_last_dim: Optional[int] = None):
        path = os.path.join(clip_dir, name)
        if not os.path.isfile(path):
            return  # 缺文件已在上面记录
        try:
            arr = np.load(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} 无法 np.load: {exc}")
            return
        if min_last_dim is not None:
            # action.npy: [81, C], C >= min_last_dim
            if arr.ndim != 2 or arr.shape[0] != CLIP_FRAMES:
                problems.append(
                    f"{name} shape={arr.shape} 期望 [{CLIP_FRAMES}, C>={min_last_dim}]")
            elif arr.shape[1] < min_last_dim:
                problems.append(
                    f"{name} 第二维 C={arr.shape[1]} < {min_last_dim}（v4 需 WASD 前4维）")
        else:
            if tuple(arr.shape) != tuple(expected_shape):
                problems.append(
                    f"{name} shape={arr.shape} 期望 {tuple(expected_shape)}")

    _check_npy("poses.npy", EXPECTED_POSES_SHAPE)
    _check_npy("intrinsics.npy", EXPECTED_INTRINSICS_SHAPE)
    _check_npy("action.npy", None, min_last_dim=ACTION_MIN_DIM)

    # 5. video 帧数 == 81（依赖 opencv，缺则软提示）
    video_path = os.path.join(clip_dir, "video.mp4")
    if os.path.isfile(video_path):
        try:
            import cv2  # 局部导入：本模块核心不依赖 cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                problems.append("video.mp4 无法打开（cv2.VideoCapture 失败）")
            else:
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if n != CLIP_FRAMES:
                    problems.append(
                        f"video.mp4 帧数={n} 期望 {CLIP_FRAMES}")
            cap.release()
        except ImportError:
            problems.append(
                "提示: 未安装 opencv，跳过 video.mp4 帧数校验（非硬错误，请手动确认 81 帧）")

    return problems


def write_metadata_csv(rows: List[dict], out_csv: str) -> None:
    """写 metadata CSV，强制校验每行含 REQUIRED_METADATA_COLUMNS 且整数列可转 int。

    校验规则（任一不满足直接 raise，并提示哪行哪列）：
      - 每行必须含 REQUIRED_METADATA_COLUMNS 全部键；
      - clip_idx / episode_idx 必须能转成 int（旧 preprocess 缺这两列，
        是本次交接最容易漏的点）。

    写出列顺序：REQUIRED_METADATA_COLUMNS 在前，其余键（如 prompt/map/stem）
    按首次出现顺序追加在后；clip_idx/episode_idx 会被规范化为 int 写出。

    Args:
        rows:    每个 clip 一个 dict。
        out_csv: 输出 CSV 路径（父目录自动创建）。

    Raises:
        ValueError: 缺必需列，或 clip_idx/episode_idx 无法转 int。
    """
    if not rows:
        raise ValueError("rows 为空，无法写 metadata CSV")

    int_cols = ["clip_idx", "episode_idx"]

    # 收集所有出现过的列（必需列在前，附加列按首次出现顺序）
    extra_cols: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in REQUIRED_METADATA_COLUMNS and k not in extra_cols:
                extra_cols.append(k)
    fieldnames = list(REQUIRED_METADATA_COLUMNS) + extra_cols

    # 逐行校验 + 规范化 int 列
    normalized: List[dict] = []
    for i, row in enumerate(rows):
        missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in row]
        if missing:
            raise ValueError(
                f"第 {i} 行缺少必需列 {missing}；"
                f"REQUIRED_METADATA_COLUMNS={REQUIRED_METADATA_COLUMNS}。"
                "（旧 preprocess_csgo_v3.py 缺 clip_idx/episode_idx，记得补上）")
        new_row = dict(row)
        for c in int_cols:
            try:
                new_row[c] = int(row[c])
            except (TypeError, ValueError):
                raise ValueError(
                    f"第 {i} 行的 {c}={row[c]!r} 无法转 int；"
                    f"{c} 必须是整数（clip_idx=episode内序号，episode_idx=全局编号）")
        normalized.append(new_row)

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for new_row in normalized:
            writer.writerow(new_row)


# ═══════════════════════════════════════════════════════════════════════════
# Part B —— 重访筛选判据（复用 explore_data.py 逻辑）
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠️ 单位常量（见模块顶部 docstring 的醒目警告）：
# 原始 pose 平移单位若是 CSGO game units（≈ inches），1 米 ≈ 40 units；
# 若已是米则把它改成 1.0。钰淇必须按实际原始数据单位设置，否则所有距离阈值失效。
POSE_UNITS_PER_METER = 40.0

# 阈值常量（以米表达；函数内乘 POSE_UNITS_PER_METER 换算到数据集原生单位）。
# 与 explore_data.py 的默认参数对齐：
#   distance_eps=40 units≈1m, yaw_eps=30°,
#   require_intermediate_separation=100 units≈2.5m。
HIT_DIST_M = 1.0           # 同地点：平移距离 < 1m
HIT_YAW_DEG = 30.0         # 同朝向：yaw 差 < 30°
INTERMEDIATE_SEP_M = 2.5   # 中间分离：agent 真离开过（中间帧距端点 > 2.5m）
GAP_MIN_CLIPS = 2          # 时间间隔下限（clip 数）——超出眼前 context 才需记忆
GAP_MAX_CLIPS = 6          # 时间间隔上限（clip 数）——训练窗口 max_context_clips+1=7，留在窗口内

# episode_revisit_stats 内部对长序列稀疏采样的默认帧数上限（与 explore_data.py
# 的 max_pairs_dim 默认值 500 一致），避免 O(T^2) 爆炸。
MAX_PAIRS_DIM = 500


def yaw_deg_from_c2w(pose_4x4: np.ndarray) -> float:
    """从单个 c2w 矩阵 [4,4] 计算 yaw（度）。

    与 explore_data.py extract_xz_and_yaw 完全一致的公式：
        yaw = atan2(R[0,2], R[2,2])  （弧度），再转度。
    其中 R = pose[:3,:3] 是 c2w 旋转部分，yaw 衡量相机绕 y 轴方向（BEV 平面）。

    Args:
        pose_4x4: camera-to-world 矩阵，shape [4,4]。

    Returns:
        yaw 角，单位度，范围 (-180, 180]。
    """
    pose = np.asarray(pose_4x4, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"pose 必须是 [4,4]，got {pose.shape}")
    R = pose[:3, :3]
    yaw_rad = math.atan2(float(R[0, 2]), float(R[2, 2]))
    return math.degrees(yaw_rad)


def angular_diff_deg(a: float, b: float) -> float:
    """两个角度（度）之间的周期 360° 最小角差，结果在 [0, 180]。

    与 explore_data.py angular_diff_deg 公式一致：
        diff = abs(a - b) % 360; if diff > 180: 360 - diff。

    Args:
        a, b: 角度，单位度。

    Returns:
        最小角差，单位度，范围 [0, 180]。
    """
    diff = abs(float(a) - float(b)) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def _xz(translations: np.ndarray) -> np.ndarray:
    """取平移的 (x, z) 两维做 BEV 距离（与 explore_data.py 一致：poses[:,[0,2],3]）。

    Args:
        translations: [T,3] 平移分量（pose[:3,3]）。

    Returns:
        [T,2] 的 (x, z)。
    """
    t = np.asarray(translations, dtype=np.float32)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"translations 必须是 [T,3]，got {t.shape}")
    return t[:, [0, 2]]


def _max_intermediate_separation(xz: np.ndarray, i_a: int, i_b: int) -> float:
    """中间分离：max_{k ∈ (i_a, i_b)} max(dist(k, i_a), dist(k, i_b))。

    与 explore_data.py _compute_max_intermediate_separation 完全一致：
    扫描原始全部中间帧（非稀疏采样），判断 agent 是否真的"离开过"端点附近。
    原地静止 + yaw 来回时本值很小；真的绕一圈再回来则很大。

    Args:
        xz:  [T,2] 的 BEV 平移。
        i_a, i_b: 全局帧号，要求 i_a < i_b。

    Returns:
        中间分离值（数据集原生单位，与 xz 同单位）。i_b - i_a <= 1 时返回 0.0。
    """
    if i_b - i_a <= 1:
        return 0.0
    seg = xz[i_a + 1:i_b]                              # [m,2]
    d_a = np.linalg.norm(seg - xz[i_a], axis=1)        # [m]
    d_b = np.linalg.norm(seg - xz[i_b], axis=1)        # [m]
    per_frame_max = np.maximum(d_a, d_b)
    return float(per_frame_max.max())


def is_position_revisit(
    episode_translations: np.ndarray,
    episode_yaws: np.ndarray,
    i: int,
    j: int,
    frames_per_clip: int,
    *,
    pose_units_per_meter: float = POSE_UNITS_PER_METER,
    hit_dist_m: float = HIT_DIST_M,
    hit_yaw_deg: float = HIT_YAW_DEG,
    intermediate_sep_m: float = INTERMEDIATE_SEP_M,
    gap_min_clips: int = GAP_MIN_CLIPS,
    gap_max_clips: int = GAP_MAX_CLIPS,
) -> bool:
    """判定一对帧 (i, j)（要求 i < j）是否构成"位置重访"。

    判定条件（全部满足才 True），逻辑与 explore_data.py 的 location_revisit 一致，
    距离阈值统一在米→原生单位换算（乘 pose_units_per_meter）：
      1. 同地点：XZ 平面 L2 距离 < hit_dist_m * pose_units_per_meter；
      2. 同朝向：angular_diff_deg(yaw_i, yaw_j) < hit_yaw_deg；
      3. 中间分离：_max_intermediate_separation(xz, i, j) > intermediate_sep_m * pose_units_per_meter
         （agent 真的离开过端点附近，排除原地静止 lookaround）；
      4. 时间间隔：(j - i) 落在 [gap_min_clips, gap_max_clips] * frames_per_clip 帧之间
         （下限：超出眼前 context 才需记忆；上限：留在训练窗口 max_context_clips+1 内）。

    Args:
        episode_translations: 一个 episode 全帧拼接后的平移 [T,3]（pose[:3,3]）。
        episode_yaws:         对应每帧 yaw（度）[T]，由 yaw_deg_from_c2w 算出。
        i, j:                 两个全局帧号（拼接后），要求 i < j。
        frames_per_clip:      每个 clip 的帧数（v4 = CLIP_FRAMES = 81）。
        pose_units_per_meter: ⚠️ 平移单位换算（game units≈40；米=1.0）。
        hit_dist_m / hit_yaw_deg / intermediate_sep_m: 阈值（米 / 度 / 米）。
        gap_min_clips / gap_max_clips: 时间间隔下/上限（clip 数）。

    Returns:
        True 表示 (i, j) 是一对位置重访。
    """
    if i >= j:
        return False
    xz = _xz(episode_translations)
    yaws = np.asarray(episode_yaws, dtype=np.float32)
    T = xz.shape[0]
    if not (0 <= i < T and 0 <= j < T):
        raise ValueError(f"帧号越界: i={i}, j={j}, T={T}")
    if yaws.shape[0] != T:
        raise ValueError(f"yaws 长度 {yaws.shape[0]} 与 translations T={T} 不一致")

    dist_eps = hit_dist_m * pose_units_per_meter
    sep_eps = intermediate_sep_m * pose_units_per_meter

    # 1. 同地点
    dist = float(np.linalg.norm(xz[i] - xz[j]))
    if dist >= dist_eps:
        return False
    # 2. 同朝向
    if angular_diff_deg(yaws[i], yaws[j]) >= hit_yaw_deg:
        return False
    # 4. 时间间隔（clip 数 → 帧数窗口）
    gap_frames = j - i
    if not (gap_min_clips * frames_per_clip <= gap_frames
            <= gap_max_clips * frames_per_clip):
        return False
    # 3. 中间分离
    if _max_intermediate_separation(xz, i, j) <= sep_eps:
        return False
    return True


def _sample_frame_indices(total_frames: int, max_pairs_dim: int = MAX_PAIRS_DIM) -> np.ndarray:
    """长序列稀疏采样（与 explore_data.py _sample_frame_indices 一致）。

    T <= max_pairs_dim 时返回全部帧号；否则按 step 抽样，避免 O(T^2) 爆炸。
    """
    if total_frames <= max_pairs_dim:
        return np.arange(total_frames, dtype=np.int64)
    step = max(1, total_frames // max_pairs_dim)
    return np.arange(0, total_frames, step, dtype=np.int64)


def episode_revisit_stats(
    episode_translations: np.ndarray,
    episode_yaws: np.ndarray,
    frames_per_clip: int,
    *,
    pose_units_per_meter: float = POSE_UNITS_PER_METER,
    hit_dist_m: float = HIT_DIST_M,
    hit_yaw_deg: float = HIT_YAW_DEG,
    intermediate_sep_m: float = INTERMEDIATE_SEP_M,
    gap_min_clips: int = GAP_MIN_CLIPS,
    gap_max_clips: int = GAP_MAX_CLIPS,
    max_pairs_dim: int = MAX_PAIRS_DIM,
) -> dict:
    """扫一个 episode 的帧对，统计位置重访对数与密度。

    对长序列做稀疏采样（与 explore_data.py 同思路，max_pairs_dim 默认 500），
    在采样后的帧号上枚举 i<j 帧对，用 is_position_revisit 逐对判定。
    中间分离始终在原始全部帧上计算（_max_intermediate_separation 内部已保证）。

    Args:
        episode_translations: 全帧平移 [T,3]。
        episode_yaws:         全帧 yaw（度）[T]。
        frames_per_clip:      每 clip 帧数（v4 = 81）。
        其余:                 阈值 / 单位 / 采样上限（含义同 is_position_revisit）。

    Returns:
        dict，含：
          - n_frames:          原始帧数 T
          - n_clips_est:       估计 clip 数 = T // frames_per_clip
          - n_sampled_frames:  稀疏采样后参与枚举的帧数
          - n_pairs_examined:  实际枚举的帧对总数
          - n_revisit_pairs:   判定为位置重访的帧对数
          - revisit_density:   n_revisit_pairs / max(1, n_pairs_examined)
          - pose_units_per_meter / 各阈值: 回显，便于核对口径
    """
    xz = _xz(episode_translations)
    yaws = np.asarray(episode_yaws, dtype=np.float32)
    T = xz.shape[0]
    if yaws.shape[0] != T:
        raise ValueError(f"yaws 长度 {yaws.shape[0]} 与 translations T={T} 不一致")

    sample_idx = _sample_frame_indices(T, max_pairs_dim=max_pairs_dim)
    n = sample_idx.shape[0]

    n_pairs_examined = 0
    n_revisit = 0
    for a in range(n):
        ia = int(sample_idx[a])
        for b in range(a + 1, n):
            jb = int(sample_idx[b])
            n_pairs_examined += 1
            if is_position_revisit(
                episode_translations, yaws, ia, jb, frames_per_clip,
                pose_units_per_meter=pose_units_per_meter,
                hit_dist_m=hit_dist_m,
                hit_yaw_deg=hit_yaw_deg,
                intermediate_sep_m=intermediate_sep_m,
                gap_min_clips=gap_min_clips,
                gap_max_clips=gap_max_clips,
            ):
                n_revisit += 1

    return {
        "n_frames": int(T),
        "n_clips_est": int(T // frames_per_clip),
        "n_sampled_frames": int(n),
        "n_pairs_examined": int(n_pairs_examined),
        "n_revisit_pairs": int(n_revisit),
        "revisit_density": n_revisit / float(max(1, n_pairs_examined)),
        "pose_units_per_meter": float(pose_units_per_meter),
        "hit_dist_m": float(hit_dist_m),
        "hit_yaw_deg": float(hit_yaw_deg),
        "intermediate_sep_m": float(intermediate_sep_m),
        "gap_min_clips": int(gap_min_clips),
        "gap_max_clips": int(gap_max_clips),
    }


def episode_qualifies(stats: dict, *, min_revisit_pairs: int = 3) -> bool:
    """根据 episode_revisit_stats 的结果判定是否保留该 episode。

    Args:
        stats:             episode_revisit_stats 的返回 dict。
        min_revisit_pairs: 保留所需的最小位置重访对数。
            ⚠️ 默认 3 是一个**保守占位值**——请先对全数据集跑一遍 census
            （把每个 episode 的 n_revisit_pairs / revisit_density 汇总成分布），
            再依据分布定阈值，**别盲信这个默认**。重访本就稀疏（见 D-07），
            阈值定太高会把大部分 episode 筛掉。

    Returns:
        True 表示保留该 episode。
    """
    return int(stats.get("n_revisit_pairs", 0)) >= int(min_revisit_pairs)


# ═══════════════════════════════════════════════════════════════════════════
# 自测（合成数据，不需要真实数据集）
# ═══════════════════════════════════════════════════════════════════════════

def _yaw_to_c2w(x: float, z: float, yaw_deg: float) -> np.ndarray:
    """构造一个 c2w：平移 (x, _, z) + 绕 y 轴 yaw，使 yaw_deg_from_c2w 能反解出 yaw_deg。

    需满足 atan2(R[0,2], R[2,2]) == yaw_deg。取
        R[0,2] = sin(yaw), R[2,2] = cos(yaw)。
    """
    pose = np.eye(4, dtype=np.float32)
    yaw = math.radians(yaw_deg)
    pose[0, 2] = math.sin(yaw)
    pose[2, 2] = math.cos(yaw)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def _self_test() -> int:
    """合成数据自测：验证 is_position_revisit / 几何工具 / validate_clip_dir /
    write_metadata_csv 的核心逻辑。返回 0=全过，1=有失败。"""
    import tempfile

    failures: List[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            failures.append(msg)

    # --- 几何工具 ---
    p = _yaw_to_c2w(0.0, 0.0, 37.0)
    check(abs(yaw_deg_from_c2w(p) - 37.0) < 1e-3, "yaw_deg_from_c2w 反解失败")
    check(abs(angular_diff_deg(350.0, 10.0) - 20.0) < 1e-6, "angular_diff_deg 周期处理错")
    check(abs(angular_diff_deg(10.0, 200.0) - 170.0) < 1e-6, "angular_diff_deg 取最小角错")

    # --- is_position_revisit：构造一个"走开再绕回"的合成 episode ---
    fpc = 4  # 用小 frames_per_clip 便于构造（仅自测，真实是 81）
    upm = 40.0
    # gap = 3 个 clip = 12 帧，落在 [2,6]*4=[8,24] 内
    T = 13
    trans = np.zeros((T, 3), dtype=np.float32)
    yaws = np.zeros(T, dtype=np.float32)
    # 帧 0：起点 (0,0)，yaw 0
    # 中间帧 1..11：走到远处（z 拉到 200 units = 5m > 中间分离 2.5m*40=100）再走回
    for t in range(T):
        # 三角形轨迹：先离开到 z=200，再回来
        if t <= 6:
            trans[t] = [0.0, 0.0, (200.0 / 6.0) * t]
        else:
            trans[t] = [0.0, 0.0, 200.0 - (200.0 / 6.0) * (t - 6)]
        yaws[t] = 0.0
    # 帧 0 与帧 12：都在 (0,0)，yaw 0 → 同地点同朝向；gap=12 帧=3 clip ∈ [8,24]
    check(
        is_position_revisit(trans, yaws, 0, 12, fpc, pose_units_per_meter=upm),
        "is_position_revisit 应判定 (0,12) 为重访（走开再回）",
    )
    # 朝向不同：改帧 12 的 yaw 为 90° → 应 False
    yaws_diff = yaws.copy()
    yaws_diff[12] = 90.0
    check(
        not is_position_revisit(trans, yaws_diff, 0, 12, fpc, pose_units_per_meter=upm),
        "yaw 差 90° 应不算重访",
    )
    # 原地静止（中间帧不离开）：中间分离不够 → False
    trans_static = np.zeros((T, 3), dtype=np.float32)  # 全在原点
    check(
        not is_position_revisit(trans_static, yaws, 0, 12, fpc, pose_units_per_meter=upm),
        "原地静止（无中间分离）应不算重访",
    )
    # gap 太小：(0,4) gap=4 帧=1 clip < gap_min_clips*fpc=8 → False
    check(
        not is_position_revisit(trans, yaws, 0, 4, fpc, pose_units_per_meter=upm),
        "gap 太小（1 clip < 2）应不算重访",
    )
    # 单位坑：若误把米数据当 units（dist 阈值放大 40x）会误判——这里验证换算生效
    # 端点距离设 1.5m=60units，dist_eps=1m*40=40 → 应 False（>阈值）
    trans_far = trans.copy()
    trans_far[12] = [0.0, 0.0, 60.0]
    # 注意帧12移到60处后中间分离仍足够，但 dist(0,12)=60>40 → False
    check(
        not is_position_revisit(trans_far, yaws, 0, 12, fpc, pose_units_per_meter=upm),
        "端点距离 60units > 40units 阈值应不算重访",
    )

    # --- episode_revisit_stats / episode_qualifies ---
    stats = episode_revisit_stats(trans, yaws, fpc, pose_units_per_meter=upm)
    check(stats["n_revisit_pairs"] >= 1, "episode_revisit_stats 应至少找到 1 对重访")
    check(episode_qualifies(stats, min_revisit_pairs=1), "episode_qualifies(min=1) 应为 True")
    check(not episode_qualifies(stats, min_revisit_pairs=10**6),
          "episode_qualifies(min=巨大) 应为 False")

    # --- validate_clip_dir：构造一个合成合格 clip 目录（不含 video 校验硬依赖）---
    with tempfile.TemporaryDirectory() as td:
        clip_dir = os.path.join(td, "clip00")
        os.makedirs(clip_dir, exist_ok=True)
        np.save(os.path.join(clip_dir, "poses.npy"),
                np.zeros(EXPECTED_POSES_SHAPE, dtype=np.float32))
        np.save(os.path.join(clip_dir, "intrinsics.npy"),
                np.zeros(EXPECTED_INTRINSICS_SHAPE, dtype=np.float32))
        np.save(os.path.join(clip_dir, "action.npy"),
                np.zeros((CLIP_FRAMES, 4), dtype=np.int32))
        # 造占位文件（video.mp4 不是真视频，cv2 校验会报问题，但其它项应过）
        for fn in ["video.mp4", "image.jpg", "prompt.txt"]:
            with open(os.path.join(clip_dir, fn), "w") as fh:
                fh.write("x")
        problems = validate_clip_dir(clip_dir)
        # 只允许 video 相关问题（占位 mp4 非真视频）；npy/文件缺失类问题不应出现
        non_video = [p for p in problems if "video.mp4" not in p and "opencv" not in p]
        check(not non_video, f"validate_clip_dir 不应有非 video 问题: {non_video}")

        # 缺一个文件 + 错 shape 应被检出
        bad_dir = os.path.join(td, "clip_bad")
        os.makedirs(bad_dir, exist_ok=True)
        np.save(os.path.join(bad_dir, "poses.npy"),
                np.zeros((10, 4, 4), dtype=np.float32))  # 错帧数
        bad_problems = validate_clip_dir(bad_dir)
        check(any("poses.npy" in p for p in bad_problems), "应检出 poses.npy 错 shape")
        check(any("action.npy" in p for p in bad_problems), "应检出缺 action.npy")

    # --- write_metadata_csv ---
    with tempfile.TemporaryDirectory() as td:
        out_csv = os.path.join(td, "metadata_all.csv")
        rows = [
            {"clip_path": "clips/ep01/ep01_clip00", "episode_id": "ep01",
             "clip_idx": 0, "episode_idx": 1, "prompt": "hello"},
            {"clip_path": "clips/ep01/ep01_clip01", "episode_id": "ep01",
             "clip_idx": "1", "episode_idx": "1", "prompt": "world"},  # 字符串可转 int
        ]
        write_metadata_csv(rows, out_csv)
        with open(out_csv, newline="", encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
        check(len(got) == 2, "write_metadata_csv 应写 2 行")
        check(set(REQUIRED_METADATA_COLUMNS).issubset(got[0].keys()),
              "CSV 应含全部必需列")
        check(got[0]["clip_idx"] == "0" and got[1]["clip_idx"] == "1",
              "clip_idx 应被规范化写出")

        # 缺列应 raise
        try:
            write_metadata_csv([{"clip_path": "x", "episode_id": "ep"}], out_csv)
            check(False, "缺 clip_idx/episode_idx 应 raise")
        except ValueError:
            pass
        # 不可转 int 应 raise
        try:
            write_metadata_csv([{"clip_path": "x", "episode_id": "ep",
                                 "clip_idx": "abc", "episode_idx": 1}], out_csv)
            check(False, "clip_idx 非整数应 raise")
        except ValueError:
            pass

    if failures:
        print("SELF_TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SELF_TEST PASSED (all checks ok)")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v4 数据格式规格 + 重访筛选判据（交接模块）。"
                    "可 import 调用，或 --self_test 跑合成数据自测。")
    parser.add_argument("--self_test", action="store_true",
                        help="用合成数据验证核心逻辑，不需要真实数据集")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.self_test:
        sys.exit(_self_test())
    print(__doc__)
    print("提示：运行 `python revisit_data_spec.py --self_test` 跑自测。")
    sys.exit(0)
