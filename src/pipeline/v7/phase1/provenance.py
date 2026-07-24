"""Phase 1 provenance writer and completeness validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REQUIRED = {
    "phase", "arm", "case_id", "event_id", "query_support",
    "anchor_source_case", "anchor_frame_indices", "seed", "commit_sha",
    "checkpoint", "config", "video", "backend", "windows", "actual_output_frames",
    "peak_memory_slots", "peak_memory_tokens", "tokens_per_anchor_frame",
    "cumulative_memory_exposure_token_frames",
    "cumulative_anchor_frame_uses",
    "failure_reason",
    "invariant_evidence", "invariant_fingerprints",
}

MATCHED_INVARIANTS = (
    "commit_sha",
    "checkpoint",
    "config",
    "backend",
    "query_support",
    "planner_windows",
    "source_frame_mapping",
    "prompt",
    "trajectory",
    "planned_anchor_budget",
    "actual_output_frames",
)


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_invariant_fingerprints(evidence: Mapping[str, Any]) -> dict[str, str]:
    missing = sorted(set(MATCHED_INVARIANTS) - set(evidence))
    if missing:
        raise ValueError(f"invariant_evidence missing fields: {missing}")
    return {key: stable_fingerprint(evidence[key]) for key in MATCHED_INVARIANTS}


def validate_provenance(value: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED - set(value))
    if missing:
        raise ValueError(f"provenance missing fields: {missing}")
    if value["phase"] != "phase1":
        raise ValueError("provenance.phase must be phase1")
    if value["arm"] not in {"off", "global", "correct_local", "wrong_local"}:
        raise ValueError(f"invalid provenance.arm: {value['arm']!r}")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        raise ValueError("provenance.seed must be an integer")
    if not isinstance(value["actual_output_frames"], int) or value["actual_output_frames"] <= 0:
        raise ValueError("provenance.actual_output_frames must be a positive integer")
    if not isinstance(value["commit_sha"], str) or not value["commit_sha"]:
        raise ValueError("provenance.commit_sha must be non-empty")
    evidence = value["invariant_evidence"]
    fingerprints = value["invariant_fingerprints"]
    if not isinstance(evidence, Mapping) or not isinstance(fingerprints, Mapping):
        raise ValueError("provenance invariant evidence/fingerprints must be objects")
    expected = build_invariant_fingerprints(evidence)
    if dict(fingerprints) != expected:
        raise ValueError("provenance invariant_fingerprints do not match invariant_evidence")
    mirrored = {
        "commit_sha": value["commit_sha"],
        "checkpoint": value["checkpoint"],
        "config": value["config"],
        "backend": value["backend"],
        "query_support": value["query_support"],
        "planner_windows": value["windows"],
        "actual_output_frames": value["actual_output_frames"],
    }
    for key, top_level in mirrored.items():
        if evidence[key] != top_level:
            raise ValueError(
                f"provenance invariant_evidence.{key} does not match top-level evidence")


def validate_run_index_entry(
    entry: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Cross-check the three formal evidence sources, never trusting filenames."""
    validate_provenance(provenance)
    required = {
        "case_id", "event_id", "seed", "arm", "video", "provenance",
        "commit_sha", "checkpoint", "config", "actual_output_frames",
        "invariant_fingerprints",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError(f"run index missing fields: {missing}")
    for key in (
        "case_id", "event_id", "seed", "arm", "video", "commit_sha", "checkpoint",
        "config", "actual_output_frames",
        "invariant_fingerprints",
    ):
        if entry[key] != provenance[key]:
            raise ValueError(
                f"run index/provenance mismatch for {key}: "
                f"{entry[key]!r} != {provenance[key]!r}")
    if not isinstance(entry["video"], str) or not entry["video"]:
        raise ValueError("run index video must be a non-empty path")
    if manifest is not None:
        if entry["case_id"] != manifest["case_id"]:
            raise ValueError("run index case_id does not match manifest")
        events = {event["event_id"]: event for event in manifest["revisit_events"]}
        if entry["event_id"] not in events:
            raise ValueError("run index event_id does not exist in manifest")
        event = events[entry["event_id"]]
        if list(provenance["query_support"]) != [
            event["query_start"], event["query_end"]
        ]:
            raise ValueError("provenance query_support does not match manifest event")


def validate_matched_run_invariants(
    entries: list[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> None:
    """Reject cross-arm confounds before scoring or aggregation."""
    groups: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    for entry in entries:
        key = (str(entry["case_id"]), str(entry["event_id"]), int(entry["seed"]))
        arm = str(entry["arm"])
        arm_entries = groups.setdefault(key, {})
        if arm in arm_entries:
            raise ValueError(f"duplicate matched arm for {key}: {arm}")
        arm_entries[arm] = entry
    expected_arms = {"off", "global", "correct_local", "wrong_local"}
    for key, arms in groups.items():
        if require_complete and set(arms) != expected_arms:
            raise ValueError(
                f"incomplete matched four-arm evidence for {key}: "
                f"have={sorted(arms)}, need={sorted(expected_arms)}")
        reference_arm = sorted(arms)[0]
        reference = arms[reference_arm]["invariant_fingerprints"]
        for arm, entry in arms.items():
            current = entry["invariant_fingerprints"]
            for invariant in MATCHED_INVARIANTS:
                if current.get(invariant) != reference.get(invariant):
                    raise ValueError(
                        f"matched four-arm invariant mismatch for {key}: "
                        f"{invariant} differs between {reference_arm} and {arm}")


def write_provenance(path: str | Path, value: Mapping[str, Any]) -> None:
    validate_provenance(value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
