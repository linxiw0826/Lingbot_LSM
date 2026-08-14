#!/usr/bin/env python3
"""Non-training A1 code preflight.  This entry point never optimizes a tensor."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/mnt/nas/wlx/Memory/projects/Lingbot_LSM")
    parser.add_argument("--fixture-output", default="/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811")
    parser.add_argument("--output", default="/mnt/nas/wlx/Memory/outputs/m2ba_a1_code_preflight_20260814")
    args = parser.parse_args()
    repo, fixture_root, output = Path(args.repo), Path(args.fixture_output), Path(args.output)
    sys.path.insert(0, str(repo / "src"))
    from pipeline.m2ba_a1_training import A1TrainConfig, compile_train_fixture, sha256_file

    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "m2ba_a1_code_preflight_v1", "scope": "A1_CODE_ONLY_NO_TRAINING"}
    try:
        frozen_path = fixture_root / "frozen_fixture_manifest.json"
        runtime_path = fixture_root / "runtime_contract.json"
        frozen = json.loads(frozen_path.read_text())
        runtime = json.loads(runtime_path.read_text())
        if frozen.get("status") != "FIXTURE_FREEZE_PASS" or runtime.get("status") != "PASS":
            raise RuntimeError("fixture/runtime evidence is not PASS")
        fixture = compile_train_fixture(frozen)
        route = fixture.build_route_mask(length=31668)
        config = A1TrainConfig()
        repo_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
        repo_status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
        if repo_status:
            raise RuntimeError("A1 code preflight requires a clean reviewed commit")
        cases_root = Path(frozen["cases_root"]).resolve()
        checkpoint_root = Path(frozen["checkpoint_root"]).resolve()
        if cases_root != Path("/mnt/h20/135/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected") or not cases_root.is_dir():
            raise RuntimeError("frozen cases root missing or drifted")
        if checkpoint_root != Path("/mnt/h20/135/lingbot-models/lingbot-world-base-act") or not checkpoint_root.is_dir():
            raise RuntimeError("frozen checkpoint root missing or drifted")
        helper_path = repo / "tools/preflight/m2ba_a01_fixture_freeze/probe_wan_runtime.py"
        spec = importlib.util.spec_from_file_location("_m2ba_a1_verified_runtime", helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load checkpoint fingerprint helper")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        actual_checkpoint_fingerprint = helper.checkpoint_inventory_fingerprint(checkpoint_root)
        if actual_checkpoint_fingerprint != frozen["checkpoint_fingerprint"]:
            raise RuntimeError("actual checkpoint inventory fingerprint drift")
        train_record = next(item for item in frozen["fixtures"] if item.get("role") == "TRAIN")
        snapshot = {
            "schema_version": "m2ba_a1_train_fixture_snapshot_v1",
            "status": "TRAIN_ONLY_SNAPSHOT",
            "source_fixture_manifest_sha256": sha256_file(frozen_path),
            "runtime_contract_sha256": sha256_file(runtime_path),
            "cases_root": frozen["cases_root"],
            "checkpoint_fingerprint": frozen["checkpoint_fingerprint"],
            "checkpoint_root": str(checkpoint_root),
            "a0_gate_evidence_commit": "0714801a7efa3109b574caf4af20aad1a7538c9e",
            "a1_expected_repo_commit": repo_commit,
            "repo_clean_at_snapshot": True,
            "fixtures": [train_record],
        }
        # This is the only fixture document the training executable consumes.
        snapshot_tmp = output / "train_fixture_snapshot.json.tmp"
        snapshot_tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        snapshot_tmp.replace(output / "train_fixture_snapshot.json")
        report.update({
            "status": "A1_CODE_PREFLIGHT_CANDIDATE",
            "training_started": False,
            "fixture_manifest_sha256": sha256_file(frozen_path),
            "runtime_contract_sha256": sha256_file(runtime_path),
            "fixture": {
                "case_id": fixture.case_id, "event_id": fixture.event_id,
                "memory_frames": list(fixture.memory_frames),
                "target_token_slice": list(fixture.target_token_slice),
                "route_tokens": int(route.sum()),
            },
            "config": config.canonical_dict(), "config_sha256": config.fingerprint(),
            "authorization_required": True,
            "authorization_artifact_written": False,
            "gates": {
                "TRAIN_FIXTURE_ONLY": fixture.case_id == config.train_case,
                "TARGET_1508": fixture.target_token_slice == (27144, 28652),
                "ROUTE_25636": int(route.sum()) == 25636,
                "NO_OPTIMIZER_STEP": True,
            },
        })
    except Exception as exc:
        report.update({"status": "BLOCKED_A1_CODE_PREFLIGHT", "error": f"{type(exc).__name__}: {exc}", "training_started": False})
    temporary = output / "a1_code_preflight.json.tmp"
    final = output / "a1_code_preflight.json"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(final)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
