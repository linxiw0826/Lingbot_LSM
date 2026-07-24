"""CPU-only contract tests for the V7 Phase 1 oracle."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.v7.phase1.evaluation import (
    EvaluationError,
    aggregate_primary,
    case_cluster_bootstrap,
    gate_verdict,
    summarize_regions,
    validate_shared_mask,
)
from pipeline.v7.phase1.guardrails import (
    EVIDENCE_COLUMNS,
    GUARDRAIL_SPECS,
    GuardrailError,
    build_expected_region_universe,
    derive_guardrails,
    guardrail_config_fingerprint,
    load_guardrail_config,
)
from pipeline.v7.phase1.collect import collect_indexes
from pipeline.v7.phase1.jobs import PRIMARY_ARMS, anchor_enabled, build_matched_jobs
from pipeline.v7.phase1.manifest import (
    ManifestError, load_manifest, manifest_seeds, validate_manifest,
)
from pipeline.v7.phase1.planner import (
    plan_windows,
    slice_modalities,
    stitch_owned,
)
from pipeline.v7.phase1.provenance import (
    build_invariant_fingerprints,
    stable_fingerprint,
    validate_matched_run_invariants,
    validate_provenance,
    validate_run_index_entry,
)
from pipeline.v7.phase1.raft import attach_raft_scores, validate_raft_scores
from pipeline.v7.phase1.run import tokens_per_anchor_frame


def _manifest(case: str = "case0", total: int = 173):
    return {
        "schema_version": 1,
        "case_id": case,
        "fps": 16,
        "total_frames": total,
        "evaluation_seeds": [42, 43, 44],
        "first_visit": {"start": 2, "end": 20, "notes": "reviewed"},
        "revisit_events": [{
            "event_id": "event0",
            "query_start": 70,
            "query_end": 103,
            "memory_frame_indices": [5],
            "target_surface_id": "surface-A",
            "wrong_anchor": {
                "source_case_id": "case-wrong",
                "frame_indices": [6],
                "surface_id": "surface-B",
                "match_verified": True,
                "notes": "pose/FOV/age/quality matched",
            },
            "notes": "reviewed",
        }],
        "human_review": {
            "status": "approved",
            "reviewer": "tester",
            "reviewed_at": "2026-07-24T00:00:00Z",
        },
    }


def test_manifest_valid_and_file_roundtrip(tmp_path):
    value = validate_manifest(_manifest())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_manifest(path)["case_id"] == "case0"


@pytest.mark.parametrize("mutation", [
    lambda d: d["revisit_events"][0].update(query_end=999),
    lambda d: d["revisit_events"][0].update(query_end=70),
    lambda d: d["revisit_events"][0].update(memory_frame_indices=[80]),
    lambda d: d["revisit_events"].append({
        "event_id": "event1", "query_start": 80, "query_end": 110,
        "memory_frame_indices": [5], "notes": "overlap",
    }),
    lambda d: d["revisit_events"].append({
        "event_id": "event0", "query_start": 120, "query_end": 130,
        "memory_frame_indices": [5], "notes": "duplicate id",
    }),
])
def test_manifest_rejects_bounds_order_overlap_and_late_memory(mutation):
    value = _manifest()
    mutation(value)
    with pytest.raises(ManifestError):
        validate_manifest(value)


def test_missing_manifest_and_todo_template_fail_fast(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "missing.json")
    template = (
        Path(__file__).parents[1]
        / "pipeline/v7/phase1/manifests/Ep000027_p0001_head20s_lookaround.json"
    )
    with pytest.raises(ManifestError):
        load_manifest(template)


def test_manifest_seeds_are_nonempty_unique_and_formal_has_three():
    duplicate = _manifest()
    duplicate["evaluation_seeds"] = [42, 42, 44]
    with pytest.raises(ManifestError, match="unique"):
        validate_manifest(duplicate)
    too_few = _manifest()
    too_few["evaluation_seeds"] = [42]
    with pytest.raises(ManifestError, match="at least 3"):
        validate_manifest(too_few)
    assert manifest_seeds(_manifest(), min_count=3) == (42, 43, 44)
    with pytest.raises(ManifestError, match="must equal"):
        manifest_seeds(_manifest(), override=[42, 43], min_count=3)
    with pytest.raises(ValueError, match="must equal"):
        build_matched_jobs(_manifest(), seeds=[42, 43])


def test_planner_4n_plus_1_padding_partition_and_stitch():
    plans = plan_windows(43, (12, 27), context_frames=81, seam_buffer=8)
    assert all(len(plan.source_frame_index) == 81 for plan in plans)
    assert all((len(plan.source_frame_index) - 1) % 4 == 0 for plan in plans)
    assert all(plan.source_frame_index[-1] == 42 for plan in plans)
    assert any(plan.is_pad for plan in plans)
    outputs = [
        np.asarray(plan.source_frame_index, dtype=np.int64)[:, None]
        for plan in plans
    ]
    stitched = stitch_owned(plans, outputs, 43)
    np.testing.assert_array_equal(stitched[:, 0], np.arange(43))


def test_four_modalities_share_exact_source_mapping():
    total = 121
    plans = plan_windows(total, (45, 82), seam_buffer=8)
    modalities = {
        "rgb": np.arange(total)[:, None] + 1000,
        "pose": np.arange(total)[:, None] + 2000,
        "action": np.arange(total)[:, None] + 3000,
        "intrinsics": np.arange(total)[:, None] + 4000,
    }
    for plan in plans:
        sliced = slice_modalities(modalities, plan)
        source = np.asarray(plan.source_frame_index)
        np.testing.assert_array_equal(sliced["rgb"][:, 0] - 1000, source)
        np.testing.assert_array_equal(sliced["pose"][:, 0] - 2000, source)
        np.testing.assert_array_equal(sliced["action"][:, 0] - 3000, source)
        np.testing.assert_array_equal(sliced["intrinsics"][:, 0] - 4000, source)


def test_matched_four_arms_have_same_planner_support_and_budget():
    jobs = [job for job in build_matched_jobs(_manifest()) if job.seed == 42]
    assert tuple(job.arm for job in jobs) == PRIMARY_ARMS
    assert len({job.windows for job in jobs}) == 1
    correct = next(job for job in jobs if job.arm == "correct_local")
    wrong = next(job for job in jobs if job.arm == "wrong_local")
    assert correct.support == wrong.support
    assert correct.anchor_frame_indices != wrong.anchor_frame_indices
    assert len(correct.anchor_frame_indices) == len(wrong.anchor_frame_indices)
    for job in jobs:
        for window in job.windows:
            if not window.support:
                assert not anchor_enabled(job, window) or job.arm == "global"


def _invariant_evidence():
    return {
        "commit_sha": "abc",
        "checkpoint": {"base": "ckpt"},
        "config": {"size": "64*96"},
        "backend": "subclip",
        "query_support": [1, 2],
        "planner_windows": [],
        "source_frame_mapping": [],
        "prompt": "prompt-hash",
        "trajectory": {"pose": "p", "action": "a", "intrinsics": "i"},
        "planned_anchor_budget": {"anchor_frames": 1, "tokens_per_anchor_frame": 10},
        "actual_output_frames": 10,
        "guardrail_config": _guardrail_config(),
    }


def _guardrail_config(threshold=1.0):
    return {
        "schema_version": "phase1_guardrails_v1",
        "thresholds": {
            name: {"threshold": threshold} for name in GUARDRAIL_SPECS
        },
    }


def _provenance(arm="off"):
    evidence = _invariant_evidence()
    value = {
        "phase": "phase1", "arm": arm, "case_id": "c", "event_id": "e",
        "query_support": [1, 2], "anchor_source_case": None,
        "anchor_frame_indices": [], "seed": 42, "commit_sha": "abc",
        "checkpoint": {"base": "ckpt"}, "config": {"size": "64*96"}, "video": "video.mp4",
        "backend": "subclip", "windows": [],
        "actual_output_frames": 10, "peak_memory_slots": 0,
        "peak_memory_tokens": 0, "tokens_per_anchor_frame": 0,
        "cumulative_memory_exposure_token_frames": 0,
        "cumulative_anchor_frame_uses": 0,
        "failure_reason": None,
        "invariant_evidence": evidence,
        "invariant_fingerprints": build_invariant_fingerprints(evidence),
    }
    return value


def test_provenance_complete():
    value = _provenance()
    validate_provenance(value)
    del value["seed"]
    with pytest.raises(ValueError):
        validate_provenance(value)


def test_provenance_cannot_fingerprint_evidence_different_from_top_level():
    value = _provenance()
    value["checkpoint"] = {"base": "different"}
    with pytest.raises(ValueError, match="checkpoint"):
        validate_provenance(value)


def test_static_mask_is_shared_and_traceable():
    mask = np.ones((10, 4, 5), dtype=np.float32)
    provenance = {
        "source": "GT reference", "model_or_engine": "engine label",
        "version": "v1", "config": {"exclude": ["HUD", "weapon", "dynamic"]},
        "estimated_from_generated_arm": False,
        "excluded_regions": ["hud", "weapon", "dynamic_foreground"],
    }
    validate_shared_mask(mask, 10, provenance)
    bad = dict(provenance, estimated_from_generated_arm=True)
    with pytest.raises(EvaluationError):
        validate_shared_mask(mask, 10, bad)


def _evaluation_rows(cases=5, seeds=3):
    rows = []
    evidence = _invariant_evidence()
    evidence["query_support"] = [0, 2]
    evidence["planner_windows"] = [
        plan.to_dict() for plan in plan_windows(4, (0, 2), seam_buffer=1)
    ]
    evidence["actual_output_frames"] = 4
    fingerprints = json.dumps(build_invariant_fingerprints(evidence))
    config_fp = guardrail_config_fingerprint(_guardrail_config())
    for ci in range(cases):
        for event in ("e0", "e1"):
            for seed in range(seeds):
                values = {
                    "off": 0.50, "global": 0.55,
                    "correct_local": 0.70 + ci * 0.001, "wrong_local": 0.45,
                }
                for arm, value in values.items():
                    for frame in range(4):
                        rows.append({
                            "case_id": f"c{ci}", "event_id": event, "seed": seed,
                            "arm": arm, "frame": frame,
                            "region": "support" if frame < 2 else "non_support",
                            "masked_dino": value, "full_dino": value - 0.01,
                            "ssim": value - 0.02, "raft_gated_anti_freeze": "",
                            "invariant_fingerprints": fingerprints,
                            "guardrail_config_fingerprint": config_fp,
                        })
    return rows


def test_primary_aggregation_is_frame_event_seed_case_and_case_bootstrap():
    aggregate = aggregate_primary(
        _evaluation_rows(), manifests=_evaluation_manifests(), seeds=[0, 1, 2],
        min_seeds=3)
    assert len(aggregate["event_scores"]) == 5 * 2 * 3
    assert len(aggregate["seed_scores"]) == 5 * 3
    assert len(aggregate["case_scores"]) == 5
    deltas = aggregate["comparisons"]["correct_local-global"]["case_deltas"]
    boot = case_cluster_bootstrap(deltas, iterations=2000, seed=7)
    assert boot["ci95_lower"] > 0


def test_primary_eval_excludes_and_reports_missing_tuple():
    rows = _evaluation_rows()
    rows = [
        row for row in rows
        if not (row["case_id"] == "c0" and row["event_id"] == "e0"
                and row["seed"] == 0 and row["arm"] == "wrong_local")
    ]
    aggregate = aggregate_primary(
        rows, manifests=_evaluation_manifests(), seeds=[0, 1, 2],
        min_seeds=3)
    assert aggregate["missing_tuples"] == [{
        "tuple": ["c0", "e0", 0], "missing_arms": ["wrong_local"]
    }]


def test_primary_eval_reports_completely_absent_preregistered_tuple():
    rows = [
        row for row in _evaluation_rows()
        if not (row["case_id"] == "c0" and row["event_id"] == "e0" and row["seed"] == 0)
    ]
    aggregate = aggregate_primary(
        rows, manifests=_evaluation_manifests(), seeds=[0, 1, 2], min_seeds=3)
    absent = next(
        item for item in aggregate["missing_tuples"]
        if item["tuple"] == ["c0", "e0", 0])
    assert set(absent["missing_arms"]) == set(PRIMARY_ARMS)
    aggregate["region_summaries"] = summarize_regions(rows)
    assert gate_verdict(aggregate)["status"] == "INCONCLUSIVE"


def test_aggregate_rejects_inconsistent_manifest_seed_sets():
    manifests = _evaluation_manifests()
    manifests["c4"]["evaluation_seeds"] = [0, 1, 9]
    with pytest.raises(EvaluationError, match="inconsistent"):
        aggregate_primary(_evaluation_rows(), manifests=manifests)


def _evaluation_manifests():
    return {
        f"c{ci}": {
            "case_id": f"c{ci}",
            "total_frames": 4,
            "evaluation_seeds": [0, 1, 2],
            "revisit_events": [
                {"event_id": "e0", "query_start": 0, "query_end": 2},
                {"event_id": "e1", "query_start": 0, "query_end": 2},
            ],
        }
        for ci in range(5)
    }


def test_primary_manifest_coverage_rejects_missing_and_duplicate_frame():
    rows = _evaluation_rows()
    missing = [
        row for row in rows
        if not (
            row["case_id"] == "c0" and row["event_id"] == "e0"
            and row["seed"] == 0 and row["arm"] == "off" and row["frame"] == 1
        )
    ]
    incomplete = aggregate_primary(
        missing, manifests=_evaluation_manifests(), seeds=[0, 1, 2],
        min_seeds=3,
    )
    assert incomplete["missing_tuples"][0]["missing_frames"] == [1]
    duplicate = rows + [dict(rows[0])]
    with pytest.raises(EvaluationError, match="duplicate primary frame"):
        aggregate_primary(
            duplicate, manifests=_evaluation_manifests(), seeds=[0, 1, 2],
            min_seeds=3,
        )


def test_run_index_provenance_mismatch_is_rejected():
    provenance = _provenance()
    entry = {
        key: provenance[key] for key in (
            "case_id", "event_id", "seed", "arm", "commit_sha",
            "checkpoint", "config", "actual_output_frames",
            "invariant_fingerprints",
        )
    }
    entry.update(video="video.mp4", provenance="provenance.json")
    validate_run_index_entry(entry, provenance)
    entry["seed"] = 43
    with pytest.raises(ValueError, match="mismatch for seed"):
        validate_run_index_entry(entry, provenance)


def test_collect_rejects_provenance_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    video = run_dir / "long_video.mp4"
    video.write_bytes(b"placeholder")
    provenance = _provenance()
    provenance["video"] = str(video)
    provenance_path = run_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    entry = {
        key: provenance[key] for key in (
            "case_id", "event_id", "seed", "arm", "commit_sha",
            "checkpoint", "config", "actual_output_frames",
            "invariant_fingerprints",
        )
    }
    entry.update(
        video=str(video), provenance=str(provenance_path), commit_sha="wrong",
    )
    (run_dir / "run_index_entry.json").write_text(
        json.dumps(entry), encoding="utf-8")
    with pytest.raises(SystemExit, match="mismatch for commit_sha"):
        collect_indexes(tmp_path, tmp_path / "merged.json")


def test_statistical_pass_cannot_be_go_while_guardrails_pending():
    rows = _evaluation_rows()
    for row in rows:
        row["raft_gated_anti_freeze"] = 1.0
    aggregate = aggregate_primary(
        rows, manifests=_evaluation_manifests(), seeds=[0, 1, 2],
        min_seeds=3)
    aggregate["region_summaries"] = summarize_regions(rows)
    verdict = gate_verdict(aggregate, bootstrap_seed=7)
    assert verdict["status"] == "INCONCLUSIVE"
    assert verdict["guardrails"]["complete"] is False


def test_guardrail_config_rejects_metric_value_pass_and_unrelated_name(tmp_path):
    bad = _guardrail_config()
    bad["thresholds"]["seam"] = {
        "threshold": 1.0, "metric": "totally_unrelated", "value": 0.0, "passed": True,
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(GuardrailError, match="only preregistered"):
        load_guardrail_config(path)
    bad = _guardrail_config()
    bad["thresholds"]["totally_unrelated"] = {"threshold": 1.0}
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(GuardrailError, match="exactly"):
        load_guardrail_config(path)


def _guardrail_rows():
    rows = _evaluation_rows()
    for row in rows:
        row.update(
            raft_gated_anti_freeze=0.9,
            run_fingerprint=f"run-{row['case_id']}-{row['event_id']}-{row['seed']}-{row['arm']}",
            manifest_digest=f"manifest-{row['case_id']}",
            input_video_digest=f"video-{row['case_id']}-{row['event_id']}-{row['seed']}-{row['arm']}",
            reference_digest=f"reference-{row['case_id']}",
            mask_digest=f"mask-{row['case_id']}",
            mask_provenance_digest=f"mask-provenance-{row['case_id']}",
        )
        if row["region"] == "non_support":
            row["masked_dino"] = 0.8
    return rows


def _guardrail_runs():
    runs = {}
    for ci in range(5):
        for event in ("e0", "e1"):
            evidence = _invariant_evidence()
            evidence["query_support"] = [0, 2]
            evidence["planner_windows"] = [
                plan.to_dict() for plan in plan_windows(4, (0, 2), seam_buffer=1)
            ]
            evidence["actual_output_frames"] = 4
            fingerprints = build_invariant_fingerprints(evidence)
            for seed in range(3):
                for arm in PRIMARY_ARMS:
                    runs[(f"c{ci}", event, seed, arm)] = {
                        "invariant_fingerprints": dict(fingerprints),
                    }
    return runs


def _external_guardrail_evidence(name, rows):
    spec = GUARDRAIL_SPECS[name]
    universes = build_expected_region_universe(
        _evaluation_manifests(), seeds=[0, 1, 2], seam_buffer=1)
    region = {
        "seam": "seam",
        "action_following": "all",
        "copy_leakage": "non_support",
    }[name]
    expected = universes[region]
    evidence = []
    for row in rows:
        identity = (
            row["case_id"], row["event_id"], int(row["seed"]),
            row["arm"], int(row["frame"]),
        )
        if identity not in expected:
            continue
        value = 0.1 if row["arm"] == "correct_local" else 0.2
        evidence.append({
            "guardrail": name,
            "metric": spec["metric"],
            "case_id": row["case_id"],
            "event_id": row["event_id"],
            "seed": row["seed"],
            "arm": row["arm"],
            "frame": row["frame"],
            "value": value,
            "run_fingerprint": row["run_fingerprint"],
            "manifest_digest": row["manifest_digest"],
            "input_video_digest": row["input_video_digest"],
            "reference_digest": row["reference_digest"],
            "mask_digest": row["mask_digest"],
            "mask_provenance_digest": row["mask_provenance_digest"],
            "producer": "official-test-producer",
            "producer_version": "1",
        })
    assert tuple(evidence[0]) == EVIDENCE_COLUMNS
    return evidence


def _derive_guardrails(rows, *, config=None, external_evidence=None):
    return derive_guardrails(
        rows,
        config=config or _guardrail_config(),
        manifests=_evaluation_manifests(),
        runs=_guardrail_runs(),
        seeds=[0, 1, 2],
        seam_buffer=1,
        external_evidence=external_evidence,
    )


def test_complete_identity_linked_guardrail_evidence_is_computed_not_claimed():
    rows = _guardrail_rows()
    evidence = {
        name: _external_guardrail_evidence(name, rows)
        for name in ("seam", "action_following", "copy_leakage")
    }
    result = _derive_guardrails(rows, external_evidence=evidence)
    assert result["complete"] is True
    assert result["passed"] is True
    assert all(item["status"] == "PASS" for item in result["results"].values())
    aggregate = aggregate_primary(
        rows, manifests=_evaluation_manifests(), seeds=[0, 1, 2], min_seeds=3)
    aggregate["region_summaries"] = summarize_regions(rows)
    assert gate_verdict(aggregate, guardrails=result)["status"] == "GO"


def test_detached_external_guardrail_artifact_is_inconclusive():
    rows = _guardrail_rows()
    evidence = {
        name: _external_guardrail_evidence(name, rows)
        for name in ("seam", "action_following", "copy_leakage")
    }
    evidence["seam"][0]["run_fingerprint"] = "detached"
    result = _derive_guardrails(rows, external_evidence=evidence)
    assert result["complete"] is False
    assert result["results"]["seam"]["status"] == "INCONCLUSIVE"


def test_post_result_threshold_change_and_missing_external_evidence_are_inconclusive():
    rows = _guardrail_rows()
    changed = _guardrail_config(threshold=0.5)
    changed_fp = guardrail_config_fingerprint(changed)
    for row in rows:
        # Attacker rewrites both the threshold file and its standalone CSV
        # column, but cannot rewrite generation-time provenance/invariants.
        row["guardrail_config_fingerprint"] = changed_fp
    with pytest.raises(GuardrailError, match="fingerprint chain mismatch"):
        _derive_guardrails(rows, config=changed)
    for row in rows:
        row["guardrail_config_fingerprint"] = guardrail_config_fingerprint(
            _guardrail_config())
    result = _derive_guardrails(rows)
    assert result["complete"] is False
    assert set(result["missing"]) == {"seam", "action_following", "copy_leakage"}


def test_guardrail_formal_universe_rejects_missing_frame_and_entire_tuple():
    rows = _guardrail_rows()
    one_missing = rows[1:]
    with pytest.raises(GuardrailError, match="scored evidence missing"):
        _derive_guardrails(one_missing)
    tuple_missing = [
        row for row in rows
        if not (
            row["case_id"] == "c0" and row["event_id"] == "e0"
            and row["seed"] == 0 and row["arm"] == "off"
        )
    ]
    with pytest.raises(GuardrailError, match="scored evidence missing"):
        _derive_guardrails(tuple_missing)


@pytest.mark.parametrize(
    ("metric", "region", "guardrail"),
    [
        ("raft_gated_anti_freeze", "support", "raft_gated_anti_freeze"),
        ("masked_dino", "non_support", "non_support_quality"),
    ],
)
def test_guardrail_scored_metric_requires_every_expected_frame(
    metric, region, guardrail,
):
    rows = _guardrail_rows()
    target = next(row for row in rows if row["region"] == region)
    target[metric] = ""
    evidence = {
        name: _external_guardrail_evidence(name, rows)
        for name in ("seam", "action_following", "copy_leakage")
    }
    result = _derive_guardrails(rows, external_evidence=evidence)
    assert result["complete"] is False
    assert result["results"][guardrail]["status"] == "INCONCLUSIVE"
    assert "required frame" in result["results"][guardrail]["reason"]


def test_external_evidence_requires_frame_and_whole_region_coverage():
    rows = _guardrail_rows()
    evidence = {
        name: _external_guardrail_evidence(name, rows)
        for name in ("seam", "action_following", "copy_leakage")
    }
    missing_frame = copy.deepcopy(evidence)
    missing_frame["copy_leakage"] = missing_frame["copy_leakage"][:-1]
    result = _derive_guardrails(rows, external_evidence=missing_frame)
    assert result["results"]["copy_leakage"]["status"] == "INCONCLUSIVE"
    assert "evidence missing 1" in result["results"]["copy_leakage"]["reason"]

    missing_region = copy.deepcopy(evidence)
    missing_region["seam"] = []
    result = _derive_guardrails(rows, external_evidence=missing_region)
    assert result["results"]["seam"]["status"] == "INCONCLUSIVE"
    assert "evidence missing" in result["results"]["seam"]["reason"]


def test_raft_unavailable_is_explicit_and_makes_gate_inconclusive():
    rows = _evaluation_rows()
    aggregate = aggregate_primary(
        rows, manifests=_evaluation_manifests(), seeds=[0, 1, 2], min_seeds=3)
    aggregate["region_summaries"] = summarize_regions(rows)
    assert aggregate["region_summaries"]["raft_gated_anti_freeze"]["status"] == "unavailable"
    assert gate_verdict(aggregate)["status"] == "INCONCLUSIVE"


def test_runtime_tokens_per_anchor_uses_non_default_latent_and_patch_size():
    anchor = np.zeros((16, 1, 48, 80), dtype=np.float32)
    pipeline = SimpleNamespace(model=SimpleNamespace(patch_size=(1, 4, 5)))
    assert tokens_per_anchor_frame(anchor, pipeline) == 12 * 16


@pytest.mark.parametrize("field", [
    "commit_sha", "checkpoint", "config", "planner_windows",
    "source_frame_mapping", "actual_output_frames",
])
def test_matched_four_arm_invariant_mismatch_rejected(field):
    entries = []
    for arm in PRIMARY_ARMS:
        provenance = _provenance(arm)
        entry = {
            key: provenance[key] for key in (
                "case_id", "event_id", "seed", "arm", "video", "commit_sha",
                "checkpoint", "config", "actual_output_frames",
                "invariant_fingerprints",
            )
        }
        entry["provenance"] = f"{arm}.json"
        entries.append(entry)
    changed = dict(entries[-1]["invariant_fingerprints"])
    changed[field] = "different"
    entries[-1]["invariant_fingerprints"] = changed
    with pytest.raises(ValueError, match="invariant mismatch"):
        validate_matched_run_invariants(entries)


def test_raft_evidence_requires_exact_tuple_frame_and_attaches_gated_score():
    runs = []
    rows = []
    for arm in PRIMARY_ARMS:
        provenance = _provenance(arm)
        entry = {
            key: provenance[key] for key in (
                "case_id", "event_id", "seed", "arm", "video", "commit_sha",
                "checkpoint", "config", "actual_output_frames",
                "invariant_fingerprints",
            )
        }
        entry["provenance"] = f"{arm}.json"
        entry["run_fingerprint"] = stable_fingerprint(entry)
        entry["manifest_digest"] = "manifest"
        entry["input_video_digest"] = "video-digest"
        entry["reference_digest"] = "reference"
        entry["mask_digest"] = "mask"
        entry["mask_provenance_digest"] = "mask-provenance"
        runs.append(entry)
        for frame in range(10):
            rows.append({
                "case_id": "c", "event_id": "e", "seed": 42, "arm": arm,
                "frame": frame, "raft_gated_anti_freeze": 0.5,
                "raft_model": "raft", "raft_weights": "w", "metric_version": "v1",
                "generated_video": "video.mp4",
                "run_fingerprint": entry["run_fingerprint"],
                "manifest_digest": "manifest",
                "input_video_digest": "video-digest",
                "reference_digest": "reference",
                "mask_digest": "mask",
                "mask_provenance_digest": "mask-provenance",
                "producer": "official",
                "producer_version": "1",
            })
    scores = validate_raft_scores(
        rows, runs=runs, total_frames=10,
        manifest_digest="manifest", reference_digest="reference",
        mask_digest="mask", mask_provenance_digest="mask-provenance")
    records = [{
        "case_id": key[0], "event_id": key[1], "seed": key[2],
        "arm": key[3], "frame": key[4], "masked_dino": 0.8,
    } for key in scores]
    attach_raft_scores(records, scores)
    assert all(row["raft_gated_masked_dino"] == pytest.approx(0.4) for row in records)
    with pytest.raises(ValueError, match="missing"):
        validate_raft_scores(
            rows[:-1], runs=runs, total_frames=10,
            manifest_digest="manifest", reference_digest="reference",
            mask_digest="mask", mask_provenance_digest="mask-provenance")
    contradictory = copy.deepcopy(rows)
    contradictory[0]["raft_gated_anti_freeze"] = 0.0
    altered = copy.deepcopy(contradictory)
    altered[0]["run_fingerprint"] = "forged-summary-link"
    with pytest.raises(ValueError, match="run_fingerprint"):
        validate_raft_scores(
            altered, runs=runs, total_frames=10,
            manifest_digest="manifest", reference_digest="reference",
            mask_digest="mask", mask_provenance_digest="mask-provenance")


def test_v6_default_entry_was_not_modified_for_phase1():
    source = Path(__file__).parents[1] / "pipeline/v6/latentconcat_infer.py"
    text = source.read_text(encoding="utf-8")
    assert 'default="bank"' in text
    assert "phase1" not in text.lower()
