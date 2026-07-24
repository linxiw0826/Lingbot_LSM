"""Strict per-frame RAFT anti-freeze evidence contract."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import validate_matched_run_invariants

RAFT_COLUMNS = (
    "case_id",
    "event_id",
    "seed",
    "arm",
    "frame",
    "raft_gated_anti_freeze",
    "raft_model",
    "raft_weights",
    "metric_version",
    "generated_video",
)


def read_raft_scores(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"RAFT scores CSV missing: {target}")
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(RAFT_COLUMNS):
            raise ValueError(
                f"RAFT CSV schema mismatch: expected={RAFT_COLUMNS}, got={reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("RAFT scores CSV is empty")
    return rows


def validate_raft_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    runs: Sequence[Mapping[str, Any]],
    total_frames: int,
) -> dict[tuple[str, str, int, str, int], float]:
    """Require one finite score for every matched run/frame, with source identity."""
    validate_matched_run_invariants(list(runs), require_complete=True)
    expected_runs = {
        (
            str(run["case_id"]),
            str(run["event_id"]),
            int(run["seed"]),
            str(run["arm"]),
        ): run
        for run in runs
    }
    expected = {
        (*key, frame) for key in expected_runs for frame in range(total_frames)
    }
    scores: dict[tuple[str, str, int, str, int], float] = {}
    for row in rows:
        key4 = (
            str(row["case_id"]),
            str(row["event_id"]),
            int(row["seed"]),
            str(row["arm"]),
        )
        frame = int(row["frame"])
        key = (*key4, frame)
        if key not in expected:
            raise ValueError(f"RAFT row references unexpected tuple/frame: {key}")
        if key in scores:
            raise ValueError(f"duplicate RAFT tuple/frame: {key}")
        run = expected_runs[key4]
        if str(row["generated_video"]) != str(run["video"]):
            raise ValueError(f"RAFT generated_video mismatch for {key}")
        for field in ("raft_model", "raft_weights", "metric_version"):
            if not str(row[field]).strip():
                raise ValueError(f"RAFT {field} must be non-empty for {key}")
        value = float(row["raft_gated_anti_freeze"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"RAFT score must be finite in [0,1] for {key}")
        scores[key] = value
    missing = sorted(expected - set(scores))
    if missing:
        raise ValueError(
            f"RAFT evidence missing {len(missing)} tuple/frame rows; first={missing[:3]}")
    return scores


def attach_raft_scores(
    records: list[dict[str, Any]],
    scores: Mapping[tuple[str, str, int, str, int], float],
) -> None:
    seen = set()
    for record in records:
        key = (
            str(record["case_id"]),
            str(record["event_id"]),
            int(record["seed"]),
            str(record["arm"]),
            int(record["frame"]),
        )
        if key not in scores:
            raise ValueError(f"no RAFT evidence for scored frame: {key}")
        gate = float(scores[key])
        record["raft_gated_anti_freeze"] = gate
        record["raft_gated_masked_dino"] = float(record["masked_dino"]) * gate
        seen.add(key)
    extra = set(scores) - seen
    if extra:
        raise ValueError(f"RAFT evidence has rows absent from score output: {sorted(extra)[:3]}")
