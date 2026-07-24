"""Strict paired Phase 1 aggregation and case-cluster bootstrap."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .manifest import manifest_seeds
from .provenance import MATCHED_INVARIANTS

PRIMARY_ARMS = ("off", "global", "correct_local", "wrong_local")
REQUIRED_GUARDRAILS = (
    "raft_gated_anti_freeze",
    "seam",
    "non_support_quality",
    "action_following",
    "copy_leakage",
)


class EvaluationError(ValueError):
    """Raised when formal primary evaluation evidence is incomplete."""


def validate_score_invariants(rows: Sequence[Mapping[str, Any]]) -> None:
    """Recheck cross-arm fingerprints at the final CSV aggregation boundary."""
    groups: Dict[Tuple[str, str, int], Dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("arm") not in PRIMARY_ARMS:
            continue
        raw = row.get("invariant_fingerprints")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise EvaluationError("invalid invariant_fingerprints JSON in score row") from exc
        if not isinstance(raw, Mapping):
            raise EvaluationError("score row missing invariant_fingerprints evidence")
        if set(MATCHED_INVARIANTS) - set(raw):
            raise EvaluationError("score row invariant_fingerprints is incomplete")
        key = (str(row["case_id"]), str(row["event_id"]), int(row["seed"]))
        arm = str(row["arm"])
        prior = groups[key].get(arm)
        normalized = {name: str(raw[name]) for name in MATCHED_INVARIANTS}
        if prior is not None and dict(prior) != normalized:
            raise EvaluationError(f"invariant fingerprint changes within {key}/{arm}")
        groups[key][arm] = normalized
    for key, arms in groups.items():
        if set(arms) != set(PRIMARY_ARMS):
            continue  # aggregate_primary reports the missing matched tuple.
        reference_arm = PRIMARY_ARMS[0]
        reference = arms[reference_arm]
        for arm in PRIMARY_ARMS[1:]:
            for name in MATCHED_INVARIANTS:
                if arms[arm][name] != reference[name]:
                    raise EvaluationError(
                        f"score evidence invariant mismatch for {key}: "
                        f"{name} differs between {reference_arm} and {arm}")


def validate_shared_mask(mask: np.ndarray, total_frames: int, provenance: Mapping[str, Any]) -> None:
    if mask.ndim != 3 or mask.shape[0] != total_frames:
        raise EvaluationError(
            f"static mask must be [T,H,W] with T={total_frames}, got {mask.shape}")
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise EvaluationError("static mask must be finite in [0,1]")
    for key in (
        "source", "model_or_engine", "version", "config",
        "estimated_from_generated_arm", "excluded_regions",
    ):
        if not provenance.get(key):
            if key == "estimated_from_generated_arm" and key in provenance:
                continue
            raise EvaluationError(f"mask provenance missing {key}")
    if provenance.get("estimated_from_generated_arm"):
        raise EvaluationError("static mask cannot be estimated separately from generated arms")
    excluded = {str(item).lower() for item in provenance["excluded_regions"]}
    required = {"hud", "weapon", "dynamic_foreground"}
    if not required.issubset(excluded):
        raise EvaluationError(
            f"static mask must exclude {sorted(required)}; got {sorted(excluded)}")


def aggregate_primary(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int] | None = None,
    expected_cases: int = 5,
    min_seeds: int = 3,
) -> Dict[str, Any]:
    """Strict join against manifests × preregistered seeds × four arms."""
    validate_score_invariants(rows)
    if len(manifests) != expected_cases:
        raise EvaluationError(
            f"formal Gate requires exactly {expected_cases} manifests, got {len(manifests)}")
    asserted_seeds = tuple(int(seed) for seed in seeds) if seeds is not None else None
    seeds_by_case = {}
    for case, manifest in manifests.items():
        try:
            seeds_by_case[case] = manifest_seeds(
                manifest, override=asserted_seeds, min_count=min_seeds)
        except ValueError as exc:
            raise EvaluationError(f"{case}: invalid preregistered seeds: {exc}") from exc
    seed_sets = {tuple(sorted(values)) for values in seeds_by_case.values()}
    if len(seed_sets) != 1:
        raise EvaluationError(
            f"formal manifests have inconsistent evaluation_seeds: {sorted(seed_sets)}")
    registered_seeds = tuple(sorted({seed for values in seeds_by_case.values() for seed in values}))
    expected = {
        (case, str(event["event_id"]), seed)
        for case, manifest in manifests.items()
        for event in manifest["revisit_events"]
        for seed in seeds_by_case[case]
    }
    expected_events = {
        (case, str(event["event_id"])): event
        for case, manifest in manifests.items()
        for event in manifest["revisit_events"]
    }
    by_tuple: Dict[Tuple[str, str, int], Dict[str, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for row in rows:
        if str(row.get("region", "support")) != "support":
            continue
        arm = str(row["arm"])
        if arm not in PRIMARY_ARMS:
            continue
        key = (str(row["case_id"]), str(row["event_id"]), int(row["seed"]))
        if key not in expected:
            raise EvaluationError(f"row references non-preregistered tuple: {key}")
        frame = int(row["frame"])
        value = float(row["masked_dino"])
        if not math.isfinite(value):
            raise EvaluationError(f"non-finite masked_dino for {key}/{arm}")
        if frame in by_tuple[key][arm]:
            raise EvaluationError(f"duplicate primary frame for {key}/{arm}/frame={frame}")
        by_tuple[key][arm][frame] = value

    missing = []
    event_scores: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for key in sorted(expected):
        arms = by_tuple.get(key, {})
        absent = sorted(set(PRIMARY_ARMS) - set(arms))
        if absent:
            missing.append({"tuple": list(key), "missing_arms": absent})
            continue
        event = expected_events[(key[0], key[1])]
        expected_frames = set(range(event["query_start"], event["query_end"]))
        coverage_error = False
        for arm in PRIMARY_ARMS:
            actual = set(arms[arm])
            if actual != expected_frames:
                missing.append({
                    "tuple": list(key),
                    "arm": arm,
                    "missing_frames": sorted(expected_frames - actual),
                    "extra_frames": sorted(actual - expected_frames),
                })
                coverage_error = True
        if coverage_error:
            continue
        event_scores[key] = {
            arm: float(np.mean(list(arms[arm].values()))) for arm in PRIMARY_ARMS
        }

    seed_events: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for (case, _event, seed), scores in event_scores.items():
        for arm, value in scores.items():
            seed_events[(case, seed)][arm].append(value)
    seed_scores = {
        key: {arm: float(np.mean(values)) for arm, values in arms.items()}
        for key, arms in seed_events.items()
    }

    case_seeds: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (case, _seed), scores in seed_scores.items():
        for arm, value in scores.items():
            case_seeds[case][arm].append(value)
    case_scores: Dict[str, Dict[str, float]] = {}
    insufficient = {}
    for case, arms in sorted(case_seeds.items()):
        counts = {arm: len(values) for arm, values in arms.items()}
        if set(arms) != set(PRIMARY_ARMS) or min(counts.values(), default=0) < min_seeds:
            insufficient[case] = counts
            continue
        case_scores[case] = {
            arm: float(np.mean(values)) for arm, values in arms.items()
        }
    comparisons = {}
    for reference in ("global", "off", "wrong_local"):
        deltas = {
            case: scores["correct_local"] - scores[reference]
            for case, scores in case_scores.items()
        }
        comparisons[f"correct_local-{reference}"] = {
            "case_deltas": deltas,
            "mean": float(np.mean(list(deltas.values()))),
            "positive_cases": int(sum(v > 0 for v in deltas.values())),
        }
    return {
        "expected_tuple_count": len(expected),
        "preregistered_seeds": list(registered_seeds),
        "preregistered_seeds_by_case": {
            case: list(values) for case, values in sorted(seeds_by_case.items())
        },
        "missing_tuples": missing,
        "insufficient_cases": insufficient,
        "absent_complete_cases": sorted(set(manifests) - set(case_scores)),
        "event_scores": _stringify_keys(event_scores),
        "seed_scores": _stringify_keys(seed_scores),
        "case_scores": case_scores,
        "comparisons": comparisons,
    }


def _stringify_keys(value: Mapping[Tuple[Any, ...], Any]) -> Dict[str, Any]:
    return {"|".join(str(part) for part in key): item for key, item in value.items()}


def summarize_regions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Hierarchical support/non-support summaries for existing metrics."""
    result: Dict[str, Any] = {}
    for region in ("support", "non_support"):
        result[region] = {}
        for metric in ("masked_dino", "raft_gated_masked_dino", "full_dino", "ssim"):
            event_values: Dict[Tuple[str, str, int, str], List[float]] = defaultdict(list)
            for row in rows:
                if row.get("region") != region or row.get("arm") not in PRIMARY_ARMS:
                    continue
                try:
                    value = float(row.get(metric))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    event_values[(
                        str(row["case_id"]), str(row["event_id"]),
                        int(row["seed"]), str(row["arm"]),
                    )].append(value)
            seed_values: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
            for (case, _event, seed, arm), values in event_values.items():
                seed_values[(case, seed, arm)].append(float(np.mean(values)))
            case_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for (case, _seed, arm), values in seed_values.items():
                case_values[(case, arm)].append(float(np.mean(values)))
            cases = sorted({case for case, _arm in case_values})
            case_scores = {
                case: {
                    arm: float(np.mean(case_values[(case, arm)]))
                    for arm in PRIMARY_ARMS if (case, arm) in case_values
                }
                for case in cases
            }
            arm_means = {
                arm: float(np.mean([scores[arm] for scores in case_scores.values()
                                    if arm in scores]))
                for arm in PRIMARY_ARMS
                if any(arm in scores for scores in case_scores.values())
            }
            contrasts = {}
            for reference in ("global", "off", "wrong_local"):
                deltas = {
                    case: scores["correct_local"] - scores[reference]
                    for case, scores in case_scores.items()
                    if "correct_local" in scores and reference in scores
                }
                contrasts[f"correct_local-{reference}"] = {
                    "case_deltas": deltas,
                    "mean": float(np.mean(list(deltas.values()))) if deltas else None,
                }
            result[region][metric] = {
                "case_scores": case_scores,
                "arm_means": arm_means,
                "contrasts": contrasts,
            }
    raft_values = []
    for row in rows:
        try:
            value = float(row.get("raft_gated_anti_freeze"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            raft_values.append(value)
    result["raft_gated_anti_freeze"] = {
        "status": "available" if raft_values else "unavailable",
        "guardrail": "requires_review" if raft_values else "INCONCLUSIVE",
        "reason": None if raft_values else "No finite RAFT-gated anti-freeze values supplied.",
        "n_values": len(raft_values),
    }
    return result


def case_cluster_bootstrap(
    case_deltas: Mapping[str, float],
    *,
    iterations: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Resample cases only; frames/events/seeds never become independent units."""
    values = np.asarray(list(case_deltas.values()), dtype=np.float64)
    if values.size != 5:
        raise EvaluationError(f"case bootstrap requires 5 deltas, got {values.size}")
    if iterations < 1000:
        raise EvaluationError("bootstrap iterations must be >=1000")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def validate_guardrails(guardrails: Mapping[str, Any] | None) -> Dict[str, Any]:
    if guardrails is None:
        return {"complete": False, "passed": False, "missing": list(REQUIRED_GUARDRAILS)}
    missing = [name for name in REQUIRED_GUARDRAILS if name not in guardrails]
    if missing:
        return {"complete": False, "passed": False, "missing": missing}
    normalized = {}
    for name in REQUIRED_GUARDRAILS:
        item = guardrails[name]
        if not isinstance(item, Mapping):
            raise EvaluationError(f"guardrail {name} must be an object")
        for field in (
            "provided", "passed", "metric", "value", "threshold", "direction",
        ):
            if field not in item:
                raise EvaluationError(f"guardrail {name} missing {field}")
        if item["provided"] is not True:
            return {"complete": False, "passed": False, "missing": [name]}
        if not isinstance(item["passed"], bool):
            raise EvaluationError(f"guardrail {name}.passed must be boolean")
        if not isinstance(item["metric"], str) or not item["metric"]:
            raise EvaluationError(f"guardrail {name}.metric must be non-empty")
        try:
            value = float(item["value"])
            threshold = float(item["threshold"])
        except (TypeError, ValueError) as exc:
            raise EvaluationError(
                f"guardrail {name} value/threshold must be numeric") from exc
        if not math.isfinite(value) or not math.isfinite(threshold):
            raise EvaluationError(f"guardrail {name} value/threshold must be finite")
        direction = item["direction"]
        if direction not in {"min", "max"}:
            raise EvaluationError(f"guardrail {name}.direction must be min or max")
        computed_pass = value >= threshold if direction == "min" else value <= threshold
        if item["passed"] is not computed_pass:
            raise EvaluationError(
                f"guardrail {name}.passed contradicts value/threshold/direction")
        normalized[name] = dict(item)
    return {
        "complete": True,
        "passed": all(item["passed"] for item in normalized.values()),
        "missing": [],
        "results": normalized,
    }


def gate_verdict(
    aggregate: Mapping[str, Any],
    *,
    bootstrap_seed: int = 0,
    guardrails: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    outcomes: Dict[str, Any] = {}
    required = ("correct_local-global", "correct_local-wrong_local")
    for name in required:
        comparison = aggregate["comparisons"][name]
        if len(comparison["case_deltas"]) == 5:
            bootstrap = case_cluster_bootstrap(
                comparison["case_deltas"], seed=bootstrap_seed)
            passed = comparison["positive_cases"] >= 4 and bootstrap["ci95_lower"] > 0
        else:
            bootstrap = {"status": "unavailable", "reason": "requires exactly 5 case deltas"}
            passed = False
        outcomes[name] = {**comparison, "bootstrap": bootstrap, "passed": passed}
    incomplete = bool(
        aggregate.get("missing_tuples")
        or aggregate.get("insufficient_cases")
        or aggregate.get("absent_complete_cases")
    )
    statistical_pass = all(outcomes[name]["passed"] for name in required)
    guardrail_result = validate_guardrails(guardrails)
    raft_unavailable = (
        aggregate["region_summaries"]["raft_gated_anti_freeze"]["status"] != "available"
    )
    if incomplete or raft_unavailable:
        status = "INCONCLUSIVE"
    elif not statistical_pass:
        status = "NO-GO"
    elif not guardrail_result["complete"]:
        status = "STATISTICAL_PASS"
    elif guardrail_result["passed"]:
        status = "GO"
    else:
        status = "NO-GO"
    return {
        "status": status,
        "primary_status": "GO" if statistical_pass else "NO-GO",
        "primary": outcomes,
        "guardrails": guardrail_result,
        "note": "GO requires both preregistered statistics and all required guardrails.",
    }


def read_per_frame_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_aggregate(
    rows: Sequence[Mapping[str, Any]],
    output_json: str | Path,
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int] | None = None,
    expected_cases: int = 5,
    min_seeds: int = 3,
    bootstrap_seed: int = 0,
    guardrails: Mapping[str, Any] | None = None,
    output_csv: str | Path | None = None,
) -> Dict[str, Any]:
    aggregate = aggregate_primary(
        rows, manifests=manifests, seeds=seeds,
        expected_cases=expected_cases, min_seeds=min_seeds)
    aggregate["region_summaries"] = summarize_regions(rows)
    aggregate["gate"] = gate_verdict(
        aggregate, bootstrap_seed=bootstrap_seed, guardrails=guardrails)
    target = Path(output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if output_csv is not None:
        _write_summary_csv(aggregate, output_csv)
    return aggregate


def _write_summary_csv(aggregate: Mapping[str, Any], path: str | Path) -> None:
    fields = ("region", "metric", "record_type", "name", "case_id", "value", "status")
    records = []
    for region in ("support", "non_support"):
        for metric, summary in aggregate["region_summaries"][region].items():
            for arm, value in summary["arm_means"].items():
                records.append((region, metric, "arm_mean", arm, "", value, "available"))
            for name, contrast in summary["contrasts"].items():
                status = "available" if contrast["mean"] is not None else "unavailable"
                records.append((region, metric, "contrast_mean", name, "", contrast["mean"], status))
                for case, value in contrast["case_deltas"].items():
                    records.append((region, metric, "case_delta", name, case, value, "available"))
    raft = aggregate["region_summaries"]["raft_gated_anti_freeze"]
    records.append(("all", "raft_gated_anti_freeze", "guardrail",
                    "raft_gated_anti_freeze", "", "", raft["status"]))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(dict(zip(fields, record)))
