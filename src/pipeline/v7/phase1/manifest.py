"""Strict event-manifest contract for the Phase 1 oracle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

SCHEMA_VERSION = 1
REQUIRED_TOP = {
    "schema_version", "case_id", "fps", "total_frames", "first_visit",
    "revisit_events", "evaluation_seeds", "human_review",
}
REQUIRED_EVENT = {
    "event_id", "query_start", "query_end", "memory_frame_indices", "notes",
}


class ManifestError(ValueError):
    """Raised when a manifest cannot be used for a formal Phase 1 run."""


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{label} must be an integer, got {value!r}")
    return value


def _required(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ManifestError(f"{label} missing required fields: {missing}")


def manifest_seeds(
    data: Mapping[str, Any],
    *,
    override: Iterable[int] | None = None,
    min_count: int = 1,
) -> tuple[int, ...]:
    """Return the authoritative preregistered seeds.

    An optional CLI/environment override is only a consistency assertion.  It
    can never add, remove, or post-hoc select seeds.
    """
    raw = data.get("evaluation_seeds")
    if not isinstance(raw, list) or not raw:
        raise ManifestError("evaluation_seeds must be a non-empty list of preregistered integers")
    seeds = tuple(_integer(value, "evaluation_seeds") for value in raw)
    if len(set(seeds)) != len(seeds):
        raise ManifestError("evaluation_seeds must contain unique integers")
    if len(seeds) < min_count:
        raise ManifestError(
            f"evaluation_seeds requires at least {min_count} values, got {len(seeds)}")
    if override is not None:
        asserted = tuple(_integer(value, "seed override") for value in override)
        if len(set(asserted)) != len(asserted):
            raise ManifestError("seed override contains duplicates")
        if set(asserted) != set(seeds):
            raise ManifestError(
                "seed override must equal the manifest evaluation_seeds set exactly; "
                f"manifest={sorted(seeds)}, override={sorted(asserted)}")
    return seeds


def validate_manifest(data: Mapping[str, Any], *, require_review: bool = True) -> Dict[str, Any]:
    """Validate and return a JSON-roundtrippable manifest.

    Intervals are half-open. Events must be sorted and non-overlapping because
    Phase 1 assigns every primary job exactly one event anchor.
    """
    if not isinstance(data, Mapping):
        raise ManifestError("manifest root must be an object")
    _required(data, REQUIRED_TOP, "manifest")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported schema_version={data['schema_version']!r}; expected {SCHEMA_VERSION}")
    case_id = data["case_id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ManifestError("case_id must be a non-empty string")
    fps = _integer(data["fps"], "fps")
    total = _integer(data["total_frames"], "total_frames")
    if fps <= 0 or total <= 0:
        raise ManifestError("fps and total_frames must be positive")
    manifest_seeds(data, min_count=3 if require_review else 1)

    review = data["human_review"]
    if not isinstance(review, Mapping):
        raise ManifestError("human_review must be an object")
    _required(review, {"status", "reviewer", "reviewed_at"}, "human_review")
    if require_review:
        if review["status"] != "approved":
            raise ManifestError(
                "formal run requires human_review.status='approved'; TODO templates are not runnable")
        if not review["reviewer"] or not review["reviewed_at"]:
            raise ManifestError("approved manifest requires reviewer and reviewed_at")

    first = data["first_visit"]
    if not isinstance(first, Mapping):
        raise ManifestError("first_visit must be an object")
    _required(first, {"start", "end", "notes"}, "first_visit")
    fs = _integer(first["start"], "first_visit.start")
    fe = _integer(first["end"], "first_visit.end")
    if not 0 <= fs < fe <= total:
        raise ManifestError(f"invalid first_visit interval [{fs},{fe}) for total={total}")

    events = data["revisit_events"]
    if not isinstance(events, list) or not events:
        raise ManifestError("revisit_events must be a non-empty list")
    ids = set()
    previous_end = -1
    normalized: List[Dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ManifestError(f"revisit_events[{index}] must be an object")
        _required(event, REQUIRED_EVENT, f"revisit_events[{index}]")
        eid = event["event_id"]
        if not isinstance(eid, str) or not eid.strip() or eid in ids:
            raise ManifestError(f"event_id must be non-empty and unique: {eid!r}")
        ids.add(eid)
        start = _integer(event["query_start"], f"{eid}.query_start")
        end = _integer(event["query_end"], f"{eid}.query_end")
        if not 0 <= start < end <= total:
            raise ManifestError(f"{eid}: invalid half-open interval [{start},{end})")
        if start < previous_end:
            raise ManifestError(f"{eid}: events are unsorted or overlap at frame {start}")
        previous_end = end
        memory = event["memory_frame_indices"]
        if not isinstance(memory, list) or not memory:
            raise ManifestError(f"{eid}: memory_frame_indices must be non-empty")
        if len(set(memory)) != len(memory):
            raise ManifestError(f"{eid}: duplicate memory_frame_indices")
        for frame in memory:
            frame = _integer(frame, f"{eid}.memory_frame_indices")
            if not 0 <= frame < start:
                raise ManifestError(
                    f"{eid}: memory frame {frame} must be in [0, query_start={start})")
            if not fs <= frame < fe:
                raise ManifestError(
                    f"{eid}: memory frame {frame} is outside first_visit [{fs},{fe})")
        wrong = event.get("wrong_anchor")
        if wrong is not None:
            if not isinstance(wrong, Mapping):
                raise ManifestError(f"{eid}.wrong_anchor must be object or null")
            _required(
                wrong,
                {"source_case_id", "frame_indices", "surface_id", "match_verified", "notes"},
                f"{eid}.wrong_anchor",
            )
            frames = wrong["frame_indices"]
            if not isinstance(frames, list) or len(frames) != len(memory):
                raise ManifestError(
                    f"{eid}: wrong anchor budget must equal correct anchor budget ({len(memory)})")
            if wrong["source_case_id"] == case_id and frames == memory:
                raise ManifestError(f"{eid}: wrong anchor equals correct anchor")
            for frame in frames:
                frame = _integer(frame, f"{eid}.wrong_anchor.frame_indices")
                if frame < 0:
                    raise ManifestError(f"{eid}: wrong anchor frame must be non-negative")
                if wrong["source_case_id"] == case_id and frame >= start:
                    raise ManifestError(
                        f"{eid}: same-case wrong anchor frame {frame} must precede query")
            target_surface = event.get("target_surface_id")
            if (
                target_surface is not None
                and wrong.get("surface_id") is not None
                and wrong["surface_id"] == target_surface
            ):
                raise ManifestError(f"{eid}: wrong anchor must use a different surface_id")
            if require_review and wrong["match_verified"] is not True:
                raise ManifestError(f"{eid}: formal wrong_local requires match_verified=true")
        normalized.append(dict(event))
    return json.loads(json.dumps(dict(data)))


def load_manifest(path: str | Path, *, require_review: bool = True) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    return validate_manifest(data, require_review=require_review)
