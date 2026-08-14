from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pipeline.m2ba_a1_training import (
    A1Trainer,
    A1HealthMonitor,
    A1TrainConfig,
    HealthStop,
    SAMPLER_SCHEMA,
    validate_training_authorization,
    write_invalid_run_marker,
    assert_only_adapter_gradients,
    atomic_save_checkpoint,
    build_checkpoint_payload,
    compile_train_fixture,
    hidden_reconstruction_loss,
    load_checkpoint_strict,
    mask_support_query_tokens,
)


def frozen_fixture():
    groups = []
    for index in range(17):
        start = index * 1508
        groups.append({
            "window_id": 5, "latent_t": index, "token_start": start,
            "token_end": start + 1508, "full_frames": [280 + index],
        })
    # Frozen mapping has 17 temporal groups in target window: 17*1508=25636.
    return {
        "status": "FIXTURE_FREEZE_PASS",
        "fixtures": [{
            "role": "TRAIN",
            "case_id": "Ep000027_p0007_77s_86s_two_windows_revisit",
            "event_id": "side_alley_return",
            "support_full_half_open": [280, 405],
            "memory_selected_full_frames": [96, 128, 176],
            "frame_to_token_mapping": {
                "target": {
                    "full_frame": 342, "window_id": 5, "local_frame": 70,
                    "latent_t": 18, "patch_t": 18, "token_start": 27144,
                    "token_end": 28652, "token_count": 1508,
                },
                "deduplicated_many_to_one_groups": groups,
            },
        }, {
            "role": "UNTRAINED_DEV",
            "case_id": "Ep000027_p0007_26s_35s_fwd_back_two_windows",
            "event_id": "arch_return",
        }],
    }


def healthy_metrics(**overrides):
    value = {
        "loss_total": 1.0, "loss_hidden_recon": 1.0,
        "activation_finite": True, "embedding_finite": True,
        "delta_finite": True, "gradient_finite": True,
        "base_grad_nonzero_count": 0, "base_fingerprint_unchanged": True,
        "non_route_nonzero": 0, "empty_delta_nonzero": 0,
        "reject_delta_nonzero": 0, "shape_contract_pass": True,
        "adapter_grad_norm": 1.0, "embedding_variance": 1.0,
        "frame_pairwise_cosines": [0.1, 0.2, 0.3],
        "raw_delta_base_ratio": 0.1, "cap_saturation_fraction": 0.0,
    }
    value.update(overrides)
    return value


class SeventeenParamModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.ParameterList([nn.Parameter(torch.randn(())) for _ in range(17)])


def test_fixture_compiles_exact_route_and_target():
    fixture = compile_train_fixture(frozen_fixture())
    route = fixture.build_route_mask(length=31668)
    assert route.dtype == torch.bool
    assert int(route.sum()) == 25636
    assert fixture.target_token_slice == (27144, 28652)


def test_training_selection_cannot_peek_at_b():
    value = frozen_fixture()
    value["training_selection"] = {"case": "arch_return"}
    with pytest.raises(RuntimeError, match="B_NO_PEEK"):
        compile_train_fixture(value)


def test_query_mask_is_exact_zero_and_non_support_unchanged():
    query = torch.randn(1, 12, 4)
    original = query.clone()
    route = torch.zeros(1, 12, dtype=torch.bool)
    route[:, 3:8] = True
    masked = mask_support_query_tokens(query, route)
    assert torch.count_nonzero(masked[route]).item() == 0
    assert torch.equal(masked[~route], original[~route])
    assert torch.equal(query, original)


def test_loss_only_uses_1508_target_tokens_and_detaches_teacher():
    student = torch.zeros(1, 2000, 2, requires_grad=True)
    teacher = torch.zeros(1, 2000, 2, requires_grad=True)
    student.data[:, 100:1608] = 2
    student.data[:, :100] = 100
    loss = hidden_reconstruction_loss(student, teacher, slice(100, 1608))
    assert loss.item() == pytest.approx(4.0)
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[:, :100]).item() == 0
    assert torch.count_nonzero(student.grad[:, 100:1608]).item() > 0


def test_only_adapter_gets_gradients():
    adapter, base = nn.Linear(3, 2), nn.Linear(3, 2)
    base.requires_grad_(False)
    adapter(torch.ones(1, 3)).sum().backward()
    assert_only_adapter_gradients(adapter, base)
    base.weight.grad = torch.ones_like(base.weight)
    with pytest.raises(RuntimeError, match="base received gradients"):
        assert_only_adapter_gradients(adapter, base)


@pytest.mark.parametrize("field,value", [
    ("base_grad_nonzero_count", 1), ("non_route_nonzero", 1),
    ("empty_delta_nonzero", 1), ("reject_delta_nonzero", 1),
    ("base_fingerprint_unchanged", False), ("shape_contract_pass", False),
])
def test_immediate_health_stops(field, value):
    with pytest.raises(HealthStop, match="INVALID_HEALTH_STOP"):
        A1HealthMonitor().update(1, healthy_metrics(**{field: value}))


def test_health_streak_threshold_is_exact():
    monitor = A1HealthMonitor()
    monitor.update(1, healthy_metrics())
    for step in range(11, 30):
        monitor.update(step, healthy_metrics(adapter_grad_norm=0.0))
    with pytest.raises(HealthStop, match="grad_too_small"):
        monitor.update(30, healthy_metrics(adapter_grad_norm=0.0))


def test_atomic_checkpoint_roundtrip_restores_every_state(tmp_path: Path):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    adapter = SeventeenParamModule()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    health = A1HealthMonitor(); health.update(1, healthy_metrics())
    config = A1TrainConfig()
    payload = build_checkpoint_payload(
        adapter=adapter, optimizer=optimizer, completed_optimizer_step=0,
        config=config, fixture_manifest_sha="fixture", runtime_contract_sha="runtime",
        base_checkpoint_fingerprint="base", repo_commit="abc", repo_dirty=False,
        sampler_state={"schema_version": SAMPLER_SCHEMA, "kind": "fixed_A", "cursor": 0}, health=health, parent_checkpoint_sha=None,
        loss_log_tail=[{"step": 10, "loss": 1.0}],
    )
    path = tmp_path / "checkpoints" / "step_0000.pt"
    digest = atomic_save_checkpoint(path, payload)
    assert path.is_file() and not path.with_name(path.name + ".tmp").exists()
    assert len(digest) == 64
    before = copy.deepcopy(adapter.state_dict())
    with torch.no_grad():
        adapter.values[0].add_(1)
    restored_health = A1HealthMonitor()
    loaded = load_checkpoint_strict(
        path, adapter=adapter, optimizer=optimizer, config=config,
        expected_fixture_sha="fixture", expected_runtime_sha="runtime",
        expected_base_fingerprint="base", expected_repo_commit="abc",
        expected_repo_dirty=False, health=restored_health,
    )
    assert loaded["completed_optimizer_step"] == 0
    assert loaded["microstep"] == 0
    assert all(torch.equal(before[key], adapter.state_dict()[key]) for key in before)
    assert restored_health.loss0 == 1.0


def test_checkpoint_fingerprint_mismatch_fails_closed(tmp_path: Path):
    adapter = SeventeenParamModule()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    health = A1HealthMonitor(); health.update(1, healthy_metrics())
    config = A1TrainConfig()
    payload = build_checkpoint_payload(
        adapter=adapter, optimizer=optimizer, completed_optimizer_step=0,
        config=config, fixture_manifest_sha="fixture", runtime_contract_sha="runtime",
        base_checkpoint_fingerprint="base", repo_commit="abc", repo_dirty=False,
        sampler_state={"schema_version": SAMPLER_SCHEMA, "kind": "fixed_A", "cursor": 0}, health=health, parent_checkpoint_sha=None,
        loss_log_tail=[],
    )
    path = tmp_path / "checkpoints" / "step_0000.pt"; atomic_save_checkpoint(path, payload)
    with pytest.raises(RuntimeError, match="mismatch"):
        load_checkpoint_strict(
            path, adapter=adapter, optimizer=optimizer, config=config,
            expected_fixture_sha="WRONG", expected_runtime_sha="runtime",
            expected_base_fingerprint="base", expected_repo_commit="abc",
            expected_repo_dirty=False, health=A1HealthMonitor(),
        )


def test_checkpoint_sha_sidecar_and_invalid_run_fail_closed(tmp_path: Path):
    adapter = SeventeenParamModule(); optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    health = A1HealthMonitor(); health.update(1, healthy_metrics())
    config = A1TrainConfig()
    payload = build_checkpoint_payload(
        adapter=adapter, optimizer=optimizer, completed_optimizer_step=0,
        config=config, fixture_manifest_sha="f", runtime_contract_sha="r",
        base_checkpoint_fingerprint="b", repo_commit="c", repo_dirty=False,
        sampler_state={"schema_version": SAMPLER_SCHEMA, "kind": "fixed_A", "cursor": 0},
        health=health, parent_checkpoint_sha=None, loss_log_tail=[],
    )
    path = tmp_path / "checkpoints" / "step_0000.pt"; atomic_save_checkpoint(path, payload)
    sidecar = path.with_suffix(".pt.sha256.json")
    value = __import__("json").loads(sidecar.read_text()); value["sha256"] = "bad"
    sidecar.write_text(__import__("json").dumps(value))
    with pytest.raises(RuntimeError, match="mismatch"):
        load_checkpoint_strict(path, adapter=adapter, optimizer=optimizer, config=config,
            expected_fixture_sha="f", expected_runtime_sha="r", expected_base_fingerprint="b",
            expected_repo_commit="c", expected_repo_dirty=False, health=A1HealthMonitor())
    # Re-save valid files, then quarantine the whole run.
    atomic_save_checkpoint(path, payload)
    write_invalid_run_marker(tmp_path, reason="INVALID_HEALTH_STOP:test", attempted_step=1, last_completed_optimizer_step=0)
    marker = __import__("json").loads((tmp_path / "INVALID_RUN.json").read_text())
    assert marker["attempted_step"] == 1
    assert marker["last_completed_optimizer_step"] == 0
    assert "completed_optimizer_step" not in marker
    assert not path.exists()


@pytest.mark.parametrize("tamper", ["missing_rng", "future_parent", "non_direct_parent"])
def test_checkpoint_chain_and_required_state_tamper_fail_closed(tmp_path: Path, tamper: str):
    adapter = SeventeenParamModule()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    health = A1HealthMonitor(); health.update(1, healthy_metrics())
    config = A1TrainConfig()
    common = dict(adapter=adapter, optimizer=optimizer, config=config,
        fixture_manifest_sha="f", runtime_contract_sha="r", base_checkpoint_fingerprint="b",
        repo_commit="c", repo_dirty=False, health=health, loss_log_tail=[])
    payload0 = build_checkpoint_payload(completed_optimizer_step=0,
        sampler_state={"schema_version":SAMPLER_SCHEMA,"kind":"fixed_A","cursor":0},
        parent_checkpoint_sha=None, **common)
    path0 = tmp_path/"checkpoints"/"step_0000.pt"; sha0 = atomic_save_checkpoint(path0, payload0)
    if tamper == "missing_rng":
        payload0.pop("rng_state"); atomic_save_checkpoint(path0, payload0); target = path0
    elif tamper == "future_parent":
        payload50 = build_checkpoint_payload(completed_optimizer_step=50,
            sampler_state={"schema_version":SAMPLER_SCHEMA,"kind":"fixed_A","cursor":200},
            parent_checkpoint_sha="future-step-sha", **common)
        payload50["parent_completed_optimizer_step"] = 100
        target = tmp_path/"checkpoints"/"step_0050.pt"; atomic_save_checkpoint(target,payload50)
    else:
        payload100 = build_checkpoint_payload(completed_optimizer_step=100,
            sampler_state={"schema_version":SAMPLER_SCHEMA,"kind":"fixed_A","cursor":400},
            parent_checkpoint_sha=sha0, **common)
        target = tmp_path/"checkpoints"/"step_0100.pt"; atomic_save_checkpoint(target,payload100)
    with pytest.raises((RuntimeError, KeyError)):
        load_checkpoint_strict(target, adapter=adapter, optimizer=optimizer, config=config,
            expected_fixture_sha="f", expected_runtime_sha="r", expected_base_fingerprint="b",
            expected_repo_commit="c", expected_repo_dirty=False, health=A1HealthMonitor())


def test_authorization_is_hash_and_commit_bound(tmp_path: Path):
    import json
    path = tmp_path / "authorization.json"
    value = {
        "schema_version": "m2ba_a1_training_authorization_v1",
        "authorization": "A1_EXPLORATORY_200_AUTHORIZED",
        "repo_commit": "commit", "config_sha256": "config",
        "train_fixture_snapshot_sha256": "snapshot",
        "runtime_contract_sha256": "runtime",
        "checkpoint_inventory_fingerprint": "checkpoint",
        "authorized_by": "project_owner",
    }
    path.write_text(json.dumps(value))
    assert validate_training_authorization(path, repo_commit="commit", config_sha="config",
        snapshot_sha="snapshot", runtime_sha="runtime", checkpoint_fingerprint="checkpoint") == value
    value["repo_commit"] = "wrong"; path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="authorization mismatch"):
        validate_training_authorization(path, repo_commit="commit", config_sha="config",
            snapshot_sha="snapshot", runtime_sha="runtime", checkpoint_fingerprint="checkpoint")


def test_trainer_uses_only_hidden_recon_and_accumulates_four_microsteps():
    adapter = nn.Linear(2, 2, bias=False)
    base = nn.Linear(2, 2, bias=False).requires_grad_(False)
    teacher = torch.zeros(1, 1608, 2)
    calls = {"student": 0, "empty": 0}
    def student():
        calls["student"] += 1
        source = torch.ones(1, 1608, 2)
        return adapter(source), {}
    def empty():
        calls["empty"] += 1
        return torch.zeros_like(teacher)
    fingerprint = lambda: "frozen"
    config = A1TrainConfig(target_token_start=100, target_token_end=1608)
    trainer = A1Trainer(
        adapter=adapter, base=base, teacher_target=teacher,
        student_forward=student, empty_forward=empty, config=config,
        base_fingerprint=fingerprint,
        health_metrics=lambda _diag, _record: healthy_metrics(),
    )
    record, _ = trainer.optimizer_step(1)
    assert calls == {"student": 4, "empty": 0}
    assert record["loss_total"] == record["loss_hidden_recon"]
    assert record["base_grad_nonzero_count"] == 0
