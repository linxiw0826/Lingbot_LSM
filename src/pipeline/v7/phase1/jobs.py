"""Matched four-arm job construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .manifest import manifest_seeds

from .planner import WindowPlan, plan_windows

PRIMARY_ARMS = ("off", "global", "correct_local", "wrong_local")


@dataclass(frozen=True)
class Phase1Job:
    case_id: str
    event_id: str
    arm: str
    seed: int
    support: Tuple[int, int]
    anchor_source_case: Optional[str]
    anchor_frame_indices: Tuple[int, ...]
    wrong_match_verified: bool
    windows: Tuple[WindowPlan, ...]


def build_matched_jobs(
    manifest: Mapping[str, Any],
    seeds: Optional[Iterable[int]] = None,
    *,
    context_frames: int = 81,
    seam_buffer: int = 8,
) -> List[Phase1Job]:
    registered_seeds = manifest_seeds(manifest, override=seeds, min_count=3)
    jobs: List[Phase1Job] = []
    case_id = str(manifest["case_id"])
    total = int(manifest["total_frames"])
    for event in manifest["revisit_events"]:
        support = (int(event["query_start"]), int(event["query_end"]))
        plans = tuple(plan_windows(
            total, support, context_frames=context_frames, seam_buffer=seam_buffer))
        correct = tuple(int(x) for x in event["memory_frame_indices"])
        wrong = event.get("wrong_anchor")
        for seed in registered_seeds:
            for arm in PRIMARY_ARMS:
                source_case: Optional[str] = None
                frames: Tuple[int, ...] = ()
                verified = True
                if arm in {"global", "correct_local"}:
                    source_case, frames = case_id, correct
                elif arm == "wrong_local":
                    if wrong is None:
                        verified = False
                    else:
                        source_case = str(wrong["source_case_id"])
                        frames = tuple(int(x) for x in wrong["frame_indices"])
                        verified = wrong.get("match_verified") is True
                jobs.append(Phase1Job(
                    case_id=case_id,
                    event_id=str(event["event_id"]),
                    arm=arm,
                    seed=int(seed),
                    support=support,
                    anchor_source_case=source_case,
                    anchor_frame_indices=frames,
                    wrong_match_verified=verified,
                    windows=plans,
                ))
    return jobs


def anchor_enabled(job: Phase1Job, window: WindowPlan) -> bool:
    if job.arm == "off":
        return False
    if job.arm == "global":
        return True
    if job.arm in {"correct_local", "wrong_local"}:
        return window.support
    raise ValueError(f"unknown primary arm: {job.arm}")
