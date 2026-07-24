"""Evidence-derived, preregistered Phase 1 secondary guardrails.

Threshold JSON is configuration only.  Values and pass/fail decisions are
always recomputed here from identity-linked per-frame evidence.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import manifest_seeds
from .planner import plan_windows
from .provenance import MATCHED_INVARIANTS, stable_fingerprint

SCHEMA_VERSION = "phase1_guardrails_v1"
GUARDRAIL_SPECS = {
    "raft_gated_anti_freeze": {
        "metric": "correct_local_raft_drop_vs_global",
        "direction": "max",
        "source": "scored_rows",
        "higher_is_better": True,
    },
    "seam": {
        "metric": "seam_band_masked_dino_drop_v1",
        "direction": "max",
        "source": "external",
        "higher_is_better": False,
    },
    "non_support_quality": {
        "metric": "non_support_masked_dino_drop_vs_global",
        "direction": "max",
        "source": "scored_rows",
        "higher_is_better": True,
    },
    "action_following": {
        "metric": "action_flow_error_v1",
        "direction": "max",
        "source": "external",
        "higher_is_better": False,
    },
    "copy_leakage": {
        "metric": "nonsupport_anchor_copy_similarity_v1",
        "direction": "max",
        "source": "external",
        "higher_is_better": False,
    },
}
EVIDENCE_COLUMNS = (
    "guardrail",
    "metric",
    "case_id",
    "event_id",
    "seed",
    "arm",
    "frame",
    "value",
    "run_fingerprint",
    "manifest_digest",
    "input_video_digest",
    "reference_digest",
    "mask_digest",
    "mask_provenance_digest",
    "producer",
    "producer_version",
)


class GuardrailError(ValueError):
    """Raised when guardrail configuration or evidence is not trustworthy."""


def load_guardrail_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise GuardrailError(f"guardrail threshold config missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "thresholds"}:
        raise GuardrailError(
            "guardrail config must contain exactly schema_version and thresholds; "
            "value/pass/metric claims are forbidden")
    if value["schema_version"] != SCHEMA_VERSION:
        raise GuardrailError(f"unsupported guardrail schema: {value['schema_version']!r}")
    thresholds = value["thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(GUARDRAIL_SPECS):
        raise GuardrailError(
            f"threshold keys must be exactly {sorted(GUARDRAIL_SPECS)}")
    normalized = {}
    for name in GUARDRAIL_SPECS:
        item = thresholds[name]
        if not isinstance(item, Mapping) or set(item) != {"threshold"}:
            raise GuardrailError(
                f"{name} must contain only preregistered field 'threshold'")
        try:
            threshold = float(item["threshold"])
        except (TypeError, ValueError) as exc:
            raise GuardrailError(f"{name}.threshold must be numeric") from exc
        if not math.isfinite(threshold):
            raise GuardrailError(f"{name}.threshold must be finite")
        normalized[name] = {"threshold": threshold}
    return {"schema_version": SCHEMA_VERSION, "thresholds": normalized}


def guardrail_config_fingerprint(config: Mapping[str, Any]) -> str:
    return stable_fingerprint(config)


def build_expected_region_universe(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
    min_seeds: int = 3,
    seam_buffer: int = 8,
) -> dict[str, set[tuple[str, str, int, str, int]]]:
    """Build formal frame identities without consulting produced score rows."""
    asserted = tuple(int(seed) for seed in seeds) if seeds is not None else None
    regions = {"all": set(), "support": set(), "non_support": set(), "seam": set()}
    for case_id, manifest in manifests.items():
        registered = manifest_seeds(
            manifest, override=asserted, min_count=min_seeds)
        total = int(manifest["total_frames"])
        for event in manifest["revisit_events"]:
            event_id = str(event["event_id"])
            start, end = int(event["query_start"]), int(event["query_end"])
            plans = plan_windows(
                total, (start, end), seam_buffer=seam_buffer)
            boundaries = {
                int(plan.owned_end)
                for index, plan in enumerate(plans[:-1])
                if int(plan.owned_end) == int(plans[index + 1].owned_start)
            }
            seam_frames = {
                frame
                for boundary in boundaries
                for frame in range(
                    max(0, boundary - seam_buffer),
                    min(total, boundary + seam_buffer),
                )
            }
            for seed in registered:
                for arm in ("off", "global", "correct_local", "wrong_local"):
                    for frame in range(total):
                        key = (str(case_id), event_id, int(seed), arm, frame)
                        regions["all"].add(key)
                        regions[
                            "support" if start <= frame < end else "non_support"
                        ].add(key)
                        if frame in seam_frames:
                            regions["seam"].add(key)
    return regions


def _validate_formal_coverage(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    expected_regions: Mapping[str, set[tuple[str, str, int, str, int]]],
) -> dict[tuple[str, str, int, str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int, str, int], Mapping[str, Any]] = {}
    for row in scored_rows:
        key = (
            str(row["case_id"]), str(row["event_id"]), int(row["seed"]),
            str(row["arm"]), int(row["frame"]),
        )
        if key not in expected_regions["all"]:
            raise GuardrailError(f"scored row has unexpected formal identity: {key}")
        if key in indexed:
            raise GuardrailError(f"scored rows duplicate formal identity: {key}")
        declared = str(row.get("region", ""))
        expected_region = (
            "support" if key in expected_regions["support"] else "non_support")
        if declared != expected_region:
            raise GuardrailError(
                f"scored row region mismatch for {key}: "
                f"expected={expected_region}, got={declared!r}")
        indexed[key] = row
    missing = expected_regions["all"] - set(indexed)
    if missing:
        raise GuardrailError(
            f"scored evidence missing {len(missing)} formal frames; "
            f"first={sorted(missing)[:3]}")
    return indexed


def _validate_guardrail_fingerprint_chain(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    expected_fp: str,
    runs: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
) -> None:
    """Bind config to standalone CSV, embedded invariant, and run provenance."""
    for row in scored_rows:
        key4 = (
            str(row["case_id"]), str(row["event_id"]), int(row["seed"]),
            str(row["arm"]),
        )
        run = runs.get(key4)
        if run is None:
            raise GuardrailError(f"no authoritative run provenance for {key4}")
        raw = row.get("invariant_fingerprints")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GuardrailError(
                    f"invalid invariant_fingerprints JSON for {key4}") from exc
        if not isinstance(raw, Mapping):
            raise GuardrailError(f"missing embedded invariant_fingerprints for {key4}")
        run_raw = run.get("invariant_fingerprints")
        if not isinstance(run_raw, Mapping):
            raise GuardrailError(f"run provenance missing invariant_fingerprints for {key4}")
        for invariant in MATCHED_INVARIANTS:
            if str(raw.get(invariant, "")) != str(run_raw.get(invariant, "")):
                raise GuardrailError(
                    f"score/run provenance invariant mismatch for {key4}: {invariant}")
        chain = {
            "standalone score column": str(
                row.get("guardrail_config_fingerprint", "")),
            "score embedded invariant": str(raw.get("guardrail_config", "")),
            "run provenance invariant": str(run_raw.get("guardrail_config", "")),
            "current canonical config": expected_fp,
        }
        if any(not value for value in chain.values()) or len(set(chain.values())) != 1:
            raise GuardrailError(
                f"guardrail config fingerprint chain mismatch for {key4}: {chain}")


def _validate_run_region_specs(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    runs: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    seeds: Sequence[int] | None,
    min_seeds: int,
    seam_buffer: int,
) -> None:
    """Prove that the reconstructed region universe is the generation planner."""
    asserted = tuple(int(seed) for seed in seeds) if seeds is not None else None
    for case_id, manifest in manifests.items():
        registered = manifest_seeds(
            manifest, override=asserted, min_count=min_seeds)
        total = int(manifest["total_frames"])
        for event in manifest["revisit_events"]:
            event_id = str(event["event_id"])
            support = [int(event["query_start"]), int(event["query_end"])]
            planner = [
                plan.to_dict() for plan in plan_windows(
                    total, tuple(support), seam_buffer=seam_buffer)
            ]
            expected_fingerprints = {
                "query_support": stable_fingerprint(support),
                "planner_windows": stable_fingerprint(planner),
                "actual_output_frames": stable_fingerprint(total),
            }
            for seed in registered:
                for arm in ("off", "global", "correct_local", "wrong_local"):
                    key = (str(case_id), event_id, int(seed), arm)
                    fingerprints = runs[key].get("invariant_fingerprints")
                    if not isinstance(fingerprints, Mapping):
                        raise GuardrailError(
                            f"run provenance missing invariant_fingerprints for {key}")
                    for name, expected in expected_fingerprints.items():
                        if str(fingerprints.get(name, "")) != expected:
                            raise GuardrailError(
                                f"run provenance {name} does not match formal "
                                f"manifest/planner region spec for {key}")


def _validate_scored_metric_coverage(
    scored_index: Mapping[
        tuple[str, str, int, str, int], Mapping[str, Any]],
    expected_keys: set[tuple[str, str, int, str, int]],
    metric: str,
) -> None:
    for key in sorted(expected_keys):
        try:
            value = float(scored_index[key][metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise GuardrailError(
                f"{metric} missing/non-numeric for required frame {key}") from exc
        if not math.isfinite(value):
            raise GuardrailError(f"{metric} non-finite for required frame {key}")


def _hierarchical_arm_means(
    rows: Sequence[Mapping[str, Any]], metric: str, *, region: str | None,
) -> dict[str, dict[str, float]]:
    event: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if region is not None and str(row.get("region")) != region:
            continue
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        event[(
            str(row["case_id"]), str(row["event_id"]), int(row["seed"]), str(row["arm"])
        )].append(value)
    seed: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (case, _event, seed_id, arm), values in event.items():
        seed[(case, seed_id, arm)].append(float(np.mean(values)))
    case: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (case_id, _seed, arm), values in seed.items():
        case[(case_id, arm)].append(float(np.mean(values)))
    return {
        case_id: {
            arm: float(np.mean(values))
            for (case, arm), values in case.items() if case == case_id
        }
        for case_id in sorted({key[0] for key in case})
    }


def _paired_drop(
    rows: Sequence[Mapping[str, Any]], metric: str, *, region: str | None,
    higher_is_better: bool,
) -> tuple[float, dict[str, float]]:
    scores = _hierarchical_arm_means(rows, metric, region=region)
    deltas = {
        case: (
            arms["global"] - arms["correct_local"]
            if higher_is_better else arms["correct_local"] - arms["global"]
        )
        for case, arms in scores.items()
        if {"global", "correct_local"}.issubset(arms)
    }
    if len(deltas) != 5:
        raise GuardrailError(
            f"{metric} requires paired correct_local/global evidence for 5 cases; "
            f"got {len(deltas)}")
    return float(np.mean(list(deltas.values()))), deltas


def read_external_evidence(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        raise GuardrailError(f"guardrail evidence missing: {target}")
    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EVIDENCE_COLUMNS):
            raise GuardrailError(
                f"evidence schema mismatch: expected={EVIDENCE_COLUMNS}, got={reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise GuardrailError(f"guardrail evidence is empty: {target}")
    return rows


def _validate_external(
    name: str,
    evidence: Sequence[Mapping[str, Any]],
    scored_index: Mapping[
        tuple[str, str, int, str, int], Mapping[str, Any]],
    expected_keys: set[tuple[str, str, int, str, int]],
) -> tuple[float, dict[str, float]]:
    spec = GUARDRAIL_SPECS[name]
    values: dict[tuple[str, str, int, str, int], float] = {}
    linked_rows = []
    for row in evidence:
        if row["guardrail"] != name or row["metric"] != spec["metric"]:
            raise GuardrailError(
                f"{name} evidence must use canonical metric {spec['metric']!r}")
        key = (
            str(row["case_id"]), str(row["event_id"]), int(row["seed"]),
            str(row["arm"]), int(row["frame"]),
        )
        if key not in expected_keys or key in values:
            raise GuardrailError(f"{name} evidence has unexpected/duplicate identity: {key}")
        source = scored_index[key]
        for field in (
            "run_fingerprint", "manifest_digest", "input_video_digest",
            "reference_digest", "mask_digest",
            "mask_provenance_digest",
        ):
            if not str(row[field]).strip() or str(row[field]) != str(source.get(field, "")):
                raise GuardrailError(f"{name} detached artifact: {field} mismatch for {key}")
        for field in ("producer", "producer_version"):
            if not str(row[field]).strip():
                raise GuardrailError(f"{name} evidence missing {field} for {key}")
        value = float(row["value"])
        if not math.isfinite(value):
            raise GuardrailError(f"{name} evidence has non-finite value for {key}")
        values[key] = value
        linked_rows.append({**source, "_external_value": value})
    missing = expected_keys - set(values)
    if missing:
        raise GuardrailError(
            f"{name} evidence missing {len(missing)} scored frames; first={sorted(missing)[:3]}")
    return _paired_drop(
        linked_rows, "_external_value", region=None,
        higher_is_better=bool(spec["higher_is_better"]))


def derive_guardrails(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    runs: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    seeds: Sequence[int] | None = None,
    min_seeds: int = 3,
    seam_buffer: int = 8,
    external_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute every guardrail; incomplete evidence can never become GO."""
    expected_fp = guardrail_config_fingerprint(config)
    expected_regions = build_expected_region_universe(
        manifests, seeds=seeds, min_seeds=min_seeds, seam_buffer=seam_buffer)
    expected_runs = {key[:4] for key in expected_regions["all"]}
    if set(runs) != expected_runs:
        missing = sorted(expected_runs - set(runs))
        extra = sorted(set(runs) - expected_runs)
        raise GuardrailError(
            f"authoritative run provenance coverage mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}")
    _validate_run_region_specs(
        manifests,
        runs=runs,
        seeds=seeds,
        min_seeds=min_seeds,
        seam_buffer=seam_buffer,
    )
    scored_index = _validate_formal_coverage(
        scored_rows, expected_regions=expected_regions)
    _validate_guardrail_fingerprint_chain(
        scored_rows, expected_fp=expected_fp, runs=runs)
    external_evidence = external_evidence or {}
    results: dict[str, Any] = {}
    missing = []
    for name, spec in GUARDRAIL_SPECS.items():
        try:
            if name == "raft_gated_anti_freeze":
                _validate_scored_metric_coverage(
                    scored_index, expected_regions["all"],
                    "raft_gated_anti_freeze")
                value, case_values = _paired_drop(
                    scored_rows, "raft_gated_anti_freeze", region=None,
                    higher_is_better=True)
            elif name == "non_support_quality":
                _validate_scored_metric_coverage(
                    scored_index, expected_regions["non_support"], "masked_dino")
                value, case_values = _paired_drop(
                    scored_rows, "masked_dino", region="non_support",
                    higher_is_better=True)
            else:
                evidence = external_evidence.get(name)
                if evidence is None:
                    raise GuardrailError("required per-frame evidence artifact not supplied")
                region = {
                    "seam": "seam",
                    "action_following": "all",
                    "copy_leakage": "non_support",
                }[name]
                value, case_values = _validate_external(
                    name, evidence, scored_index, expected_regions[region])
        except (GuardrailError, KeyError, TypeError, ValueError) as exc:
            missing.append(name)
            results[name] = {
                "status": "INCONCLUSIVE",
                "metric": spec["metric"],
                "reason": str(exc),
            }
            continue
        threshold = float(config["thresholds"][name]["threshold"])
        passed = value <= threshold if spec["direction"] == "max" else value >= threshold
        results[name] = {
            "status": "PASS" if passed else "FAIL",
            "metric": spec["metric"],
            "direction": spec["direction"],
            "aggregation": "frame->event->seed->case; paired global-correct_local; case mean",
            "value": value,
            "threshold": threshold,
            "case_values": case_values,
            "passed": passed,
        }
    return {
        "complete": not missing,
        "passed": not missing and all(item.get("passed") for item in results.values()),
        "missing": missing,
        "config_fingerprint": expected_fp,
        "results": results,
    }
