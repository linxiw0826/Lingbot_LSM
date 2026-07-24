"""Deterministic 81-frame window planner and aligned modality slicing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowPlan:
    window_index: int
    source_frame_index: Tuple[int, ...]
    is_pad: Tuple[bool, ...]
    owned_start: int
    owned_end: int
    support: bool
    seam_left: int
    seam_right: int

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["source_frame_index"] = list(self.source_frame_index)
        value["is_pad"] = list(self.is_pad)
        return value


def _segments(total: int, support: Tuple[int, int]) -> List[Tuple[int, int, bool]]:
    start, end = support
    out = []
    if start > 0:
        out.append((0, start, False))
    out.append((start, end, True))
    if end < total:
        out.append((end, total, False))
    return out


def plan_windows(
    total_frames: int,
    support: Tuple[int, int],
    *,
    context_frames: int = 81,
    seam_buffer: int = 8,
) -> List[WindowPlan]:
    """Plan windows whose owned intervals partition every original frame once.

    Each support/non-support segment is chunked independently. A window is
    centred around its owned chunk where possible and edge-repeat padded only
    when the episode itself is shorter than the model context.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if context_frames <= 0 or (context_frames - 1) % 4:
        raise ValueError("context_frames must satisfy 4n+1")
    if seam_buffer < 0 or 2 * seam_buffer >= context_frames:
        raise ValueError("seam_buffer must be >=0 and less than half the context")
    start, end = support
    if not 0 <= start < end <= total_frames:
        raise ValueError(f"invalid support [{start},{end})")
    owned_capacity = context_frames - 2 * seam_buffer
    plans: List[WindowPlan] = []
    for seg_start, seg_end, in_support in _segments(total_frames, support):
        cursor = seg_start
        while cursor < seg_end:
            owned_end = min(seg_end, cursor + owned_capacity)
            desired_start = cursor - seam_buffer
            max_start = max(0, total_frames - context_frames)
            source_start = min(max(0, desired_start), max_start)
            raw = list(range(source_start, min(total_frames, source_start + context_frames)))
            if not raw:
                raw = [0]
            source = raw + [raw[-1]] * (context_frames - len(raw))
            is_pad = [False] * len(raw) + [True] * (context_frames - len(raw))
            left = cursor - source_start
            right = source_start + context_frames - owned_end
            plans.append(WindowPlan(
                window_index=len(plans),
                source_frame_index=tuple(source),
                is_pad=tuple(is_pad),
                owned_start=cursor,
                owned_end=owned_end,
                support=in_support,
                seam_left=max(0, left),
                seam_right=max(0, right),
            ))
            cursor = owned_end
    assert_owned_partition(plans, total_frames)
    return plans


def assert_owned_partition(plans: Sequence[WindowPlan], total_frames: int) -> None:
    owners = np.zeros(total_frames, dtype=np.int64)
    for plan in plans:
        owners[plan.owned_start:plan.owned_end] += 1
    if not np.all(owners == 1):
        bad = np.flatnonzero(owners != 1).tolist()
        raise AssertionError(f"owned_output must cover each frame exactly once; bad={bad[:20]}")


def slice_modalities(
    modalities: Mapping[str, np.ndarray], plan: WindowPlan
) -> Dict[str, np.ndarray]:
    """Apply the exact same source-frame mapping to RGB/pose/action/intrinsics."""
    indices = np.asarray(plan.source_frame_index, dtype=np.int64)
    out: Dict[str, np.ndarray] = {}
    for name in ("rgb", "pose", "action", "intrinsics"):
        if name not in modalities:
            raise KeyError(f"missing modality: {name}")
        array = np.asarray(modalities[name])
        if array.shape[0] <= int(indices.max()):
            raise ValueError(f"{name} has {array.shape[0]} frames, needs {indices.max()+1}")
        out[name] = array[indices]
    return out


def stitch_owned(
    plans: Sequence[WindowPlan],
    window_outputs: Sequence[np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Write only owned frames back; no blending or duplicate ownership."""
    if len(plans) != len(window_outputs):
        raise ValueError("plan/output count mismatch")
    result = None
    written = np.zeros(total_frames, dtype=bool)
    for plan, output in zip(plans, window_outputs):
        arr = np.asarray(output)
        if arr.shape[0] != len(plan.source_frame_index):
            raise ValueError(f"window {plan.window_index}: expected frame-first output")
        positions = {
            source: local for local, source in enumerate(plan.source_frame_index)
            if not plan.is_pad[local]
        }
        locals_ = [positions[source] for source in range(plan.owned_start, plan.owned_end)]
        owned = arr[np.asarray(locals_, dtype=np.int64)]
        if result is None:
            result = np.empty((total_frames,) + arr.shape[1:], dtype=arr.dtype)
        if written[plan.owned_start:plan.owned_end].any():
            raise AssertionError("duplicate owned-output write")
        result[plan.owned_start:plan.owned_end] = owned
        written[plan.owned_start:plan.owned_end] = True
    if result is None or not written.all():
        raise AssertionError("stitch did not produce every original frame")
    return result
