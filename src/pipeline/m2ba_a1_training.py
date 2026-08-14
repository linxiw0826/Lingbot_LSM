"""Frozen M2-B-A A1 training contracts.

This module contains the deterministic, independently testable part of A1:
fixture compilation, the sole hidden-reconstruction objective, health stops,
and complete atomic checkpoints.  It deliberately has no DEV/B-fixture loader.
The real-Wan executable must supply tensors produced by the frozen A0 adapter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA_VERSION = "m2ba_a1_checkpoint_v1"
SAMPLER_SCHEMA = "m2ba_a1_fixed_a_sampler_v1"
RNG_SCHEMA = "m2ba_a1_rng_state_v1"
HEALTH_SCHEMA = "m2ba_a1_health_state_v1"
RUN_VALID = "VALID_A1_RUN"
TRAIN_CASE = "Ep000027_p0007_77s_86s_two_windows_revisit"
TRAIN_EVENT = "side_alley_return"
FORBIDDEN_DURING_TRAINING = (
    "Ep000027_p0007_26s_35s_fwd_back_two_windows",
    "arch_return",
)


@dataclasses.dataclass(frozen=True)
class A1TrainConfig:
    seed: int = 42
    optimizer_steps: int = 200
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    alpha: float = 0.05
    log_every: int = 10
    checkpoint_steps: tuple[int, ...] = (0, 50, 100, 200)
    target_token_start: int = 27144
    target_token_end: int = 28652
    expected_route_tokens: int = 25636
    train_case: str = TRAIN_CASE
    train_event: str = TRAIN_EVENT

    def canonical_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["checkpoint_steps"] = list(self.checkpoint_steps)
        return value

    def fingerprint(self) -> str:
        return sha256_json(self.canonical_dict())


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_b_reference(value: Any, location: str = "root") -> None:
    """Fail closed if a training input mentions the frozen B/DEV fixture."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_b_reference(key, f"{location}.key")
            _reject_b_reference(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_b_reference(item, f"{location}[{index}]")
    elif isinstance(value, (str, Path)):
        text = str(value)
        for forbidden in FORBIDDEN_DURING_TRAINING:
            if forbidden in text:
                raise RuntimeError(f"B_NO_PEEK_VIOLATION at {location}: {forbidden}")


@dataclasses.dataclass(frozen=True)
class CompiledA1Fixture:
    case_id: str
    event_id: str
    memory_frames: tuple[int, int, int]
    support_half_open: tuple[int, int]
    target_full_frame: int
    target_window_id: int
    target_token_slice: tuple[int, int]
    route_token_ranges: tuple[tuple[int, int], ...]

    def build_route_mask(self, *, length: int, device: torch.device | None = None) -> Tensor:
        route = torch.zeros(1, length, dtype=torch.bool, device=device)
        for start, end in self.route_token_ranges:
            if not (0 <= start < end <= length):
                raise ValueError(f"route range [{start},{end}) outside [0,{length})")
            route[:, start:end] = True
        return route

    def target_slice(self) -> slice:
        return slice(*self.target_token_slice)


def compile_train_fixture(frozen_manifest: Mapping[str, Any]) -> CompiledA1Fixture:
    _reject_b_reference({"training_selection": frozen_manifest.get("training_selection", {})})
    fixtures = frozen_manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise RuntimeError("frozen manifest fixtures must be a list")
    matches = [x for x in fixtures if x.get("role") == "TRAIN"]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one TRAIN fixture, got {len(matches)}")
    item = matches[0]
    if item.get("case_id") != TRAIN_CASE or item.get("event_id") != TRAIN_EVENT:
        raise RuntimeError("TRAIN fixture identity drift")
    if item.get("support_full_half_open") != [280, 405]:
        raise RuntimeError("TRAIN support drift")
    if item.get("memory_selected_full_frames") != [96, 128, 176]:
        raise RuntimeError("TRAIN memory drift")
    mapping = item.get("frame_to_token_mapping", {})
    target = mapping.get("target", {})
    expected_target = {
        "full_frame": 342, "window_id": 5, "local_frame": 70,
        "latent_t": 18, "patch_t": 18, "token_start": 27144,
        "token_end": 28652, "token_count": 1508,
    }
    if target != expected_target:
        raise RuntimeError("TRAIN target mapping drift")
    groups = [
        (int(x["token_start"]), int(x["token_end"]))
        for x in mapping.get("deduplicated_many_to_one_groups", [])
        if int(x["window_id"]) == 5
    ]
    if len(groups) != 17 or sum(end - start for start, end in groups) != 25636:
        raise RuntimeError("TRAIN route mapping must contain 17 groups / 25636 tokens")
    return CompiledA1Fixture(
        case_id=TRAIN_CASE, event_id=TRAIN_EVENT,
        memory_frames=(96, 128, 176), support_half_open=(280, 405),
        target_full_frame=342, target_window_id=5,
        target_token_slice=(27144, 28652), route_token_ranges=tuple(groups),
    )


def mask_support_query_tokens(query_tokens: Tensor, route_mask: Tensor) -> Tensor:
    """Return a new student query with every support visual token fixed to zero."""
    if query_tokens.ndim != 3 or route_mask.dtype != torch.bool:
        raise ValueError("query [B,L,D] and bool route mask [B,L] required")
    if tuple(query_tokens.shape[:2]) != tuple(route_mask.shape):
        raise ValueError("query/mask shape mismatch")
    masked = query_tokens.clone()
    masked[route_mask] = 0
    if torch.count_nonzero(masked[route_mask]).item() != 0:
        raise RuntimeError("query shortcut mask is not exact zero")
    return masked


def hidden_reconstruction_loss(
    student_pre_head: Tensor,
    teacher_pre_head: Tensor,
    target_slice: slice,
) -> Tensor:
    """The sole A1 objective: token-mean MSE on target342's 1508 tokens."""
    if student_pre_head.ndim != 3 or teacher_pre_head.ndim != 3:
        raise ValueError("teacher/student hidden must be [B,L,D]")
    prediction = student_pre_head[:, target_slice]
    if teacher_pre_head.shape == prediction.shape:
        target = teacher_pre_head.detach()
    elif teacher_pre_head.shape == student_pre_head.shape:
        target = teacher_pre_head[:, target_slice].detach()
    else:
        raise ValueError("teacher must be full hidden or exact target slice")
    if prediction.shape[1] != 1508:
        raise ValueError(f"target slice must contain 1508 tokens, got {prediction.shape[1]}")
    if teacher_pre_head.requires_grad:
        target = target.detach()
    return torch.mean((prediction.float() - target.float()) ** 2)


def trainable_inventory(module: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel()}
        for name, p in module.named_parameters() if p.requires_grad
    ]


def preclip_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


class HealthStop(RuntimeError):
    pass


class A1HealthMonitor:
    """Mechanical frozen health rules; no adaptive thresholds."""

    def __init__(self) -> None:
        self.loss0: float | None = None
        self.streaks: dict[str, int] = {}

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": HEALTH_SCHEMA, "loss0": self.loss0, "streaks": dict(self.streaks)}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != HEALTH_SCHEMA:
            raise RuntimeError("health state schema mismatch")
        self.loss0 = value.get("loss0")
        self.streaks = {str(k): int(v) for k, v in value.get("streaks", {}).items()}

    def _track(self, key: str, condition: bool, limit: int) -> None:
        self.streaks[key] = self.streaks.get(key, 0) + 1 if condition else 0
        if self.streaks[key] >= limit:
            raise HealthStop(f"INVALID_HEALTH_STOP:{key}")

    def update(self, step: int, metrics: Mapping[str, Any]) -> None:
        required = (
            "loss_total", "loss_hidden_recon", "activation_finite",
            "embedding_finite", "delta_finite", "gradient_finite",
            "base_grad_nonzero_count", "base_fingerprint_unchanged",
            "non_route_nonzero", "empty_delta_nonzero", "reject_delta_nonzero",
            "shape_contract_pass", "adapter_grad_norm", "embedding_variance",
            "frame_pairwise_cosines", "raw_delta_base_ratio", "cap_saturation_fraction",
        )
        missing = [key for key in required if key not in metrics]
        if missing:
            raise HealthStop(f"INVALID_HEALTH_STOP:missing_metrics:{missing}")
        loss = float(metrics["loss_total"])
        if self.loss0 is None:
            self.loss0 = loss
        finite_flags = (loss, float(metrics["loss_hidden_recon"]), float(metrics["adapter_grad_norm"]))
        if not all(np.isfinite(value) for value in finite_flags) or not all(
            bool(metrics[key]) for key in ("activation_finite", "embedding_finite", "delta_finite", "gradient_finite")
        ):
            raise HealthStop("INVALID_HEALTH_STOP:nonfinite")
        if loss != float(metrics["loss_hidden_recon"]):
            raise HealthStop("INVALID_HEALTH_STOP:loss_semantics_drift")
        immediate = {
            "base_grad": int(metrics["base_grad_nonzero_count"]) > 0,
            "base_mutation": not bool(metrics["base_fingerprint_unchanged"]),
            "non_route_delta": int(metrics["non_route_nonzero"]) != 0,
            "empty_delta": int(metrics["empty_delta_nonzero"]) != 0,
            "reject_delta": int(metrics["reject_delta_nonzero"]) != 0,
            "shape_contract": not bool(metrics["shape_contract_pass"]),
        }
        for key, failed in immediate.items():
            if failed:
                raise HealthStop(f"INVALID_HEALTH_STOP:{key}")
        if step <= 10:
            return
        cosines = tuple(float(x) for x in metrics["frame_pairwise_cosines"])
        if len(cosines) != 3:
            raise HealthStop("INVALID_HEALTH_STOP:frame_cosine_shape")
        self._track("grad_too_small", float(metrics["adapter_grad_norm"]) < 1e-8, 20)
        self._track("embedding_variance_collapse", float(metrics["embedding_variance"]) < 1e-8, 20)
        self._track("frame_cosine_collapse", all(x >= 0.999 for x in cosines), 20)
        self._track("raw_delta_explosion", float(metrics["raw_delta_base_ratio"]) > 100, 3)
        self._track("cap_saturation", float(metrics["cap_saturation_fraction"]) >= 0.95, 20)
        self._track("loss_explosion", loss > 10 * max(float(self.loss0), 1e-8), 10)
        self._track("grad_explosion", float(metrics["adapter_grad_norm"]) > 100, 3)


def capture_rng_state() -> dict[str, Any]:
    return {
        "schema_version": RNG_SCHEMA,
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if left_np[0] != right_np[0] or not np.array_equal(left_np[1], right_np[1]) or left_np[2:] != right_np[2:]:
        return False
    return torch.equal(left["torch_cpu"], right["torch_cpu"]) and len(left["torch_cuda_all"]) == len(right["torch_cuda_all"]) and all(
        torch.equal(a, b) for a, b in zip(left["torch_cuda_all"], right["torch_cuda_all"], strict=True)
    )


def restore_rng_state(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RNG_SCHEMA:
        raise RuntimeError("RNG state schema mismatch")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.random.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and value["torch_cuda_all"]:
        torch.cuda.set_rng_state_all(value["torch_cuda_all"])


def build_checkpoint_payload(
    *, adapter: nn.Module, optimizer: torch.optim.Optimizer,
    completed_optimizer_step: int, config: A1TrainConfig,
    fixture_manifest_sha: str, runtime_contract_sha: str,
    base_checkpoint_fingerprint: str, repo_commit: str, repo_dirty: bool,
    sampler_state: Mapping[str, Any], health: A1HealthMonitor,
    parent_checkpoint_sha: str | None, loss_log_tail: list[dict[str, Any]],
    lr_scheduler: Any = None,
) -> dict[str, Any]:
    if type(completed_optimizer_step) is not int or completed_optimizer_step < 0:
        raise ValueError("completed_optimizer_step must be a nonnegative integer")
    if completed_optimizer_step not in config.checkpoint_steps:
        raise ValueError("checkpoint step is not frozen/scheduled")
    expected_sampler = {
        "schema_version": SAMPLER_SCHEMA,
        "kind": "fixed_A", "cursor": completed_optimizer_step * config.gradient_accumulation,
    }
    if dict(sampler_state) != expected_sampler:
        raise ValueError(f"sampler state drift: {sampler_state} != {expected_sampler}")
    inventory = trainable_inventory(adapter)
    schedule = tuple(config.checkpoint_steps)
    schedule_index = schedule.index(completed_optimizer_step)
    expected_parent_step = None if schedule_index == 0 else schedule[schedule_index - 1]
    if expected_parent_step is None and parent_checkpoint_sha is not None:
        raise ValueError("step0 checkpoint cannot have a parent")
    if expected_parent_step is not None and not isinstance(parent_checkpoint_sha, str):
        raise ValueError("nonzero scheduled checkpoint requires direct-parent SHA")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_validity": RUN_VALID,
        "completed_optimizer_step": completed_optimizer_step,
        "microstep": 0,
        "adapter_state": adapter.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "lr_scheduler_config": {"kind": "constant"} if lr_scheduler is None else {"kind": type(lr_scheduler).__name__},
        "grad_scaler_state": None,
        "rng_state": capture_rng_state(),
        "sampler_state": dict(sampler_state),
        "gradient_accumulation": config.gradient_accumulation,
        "canonical_config": config.canonical_dict(),
        "config_sha256": config.fingerprint(),
        "fixture_manifest_sha256": fixture_manifest_sha,
        "runtime_contract_sha256": runtime_contract_sha,
        "base_checkpoint_inventory_fingerprint": base_checkpoint_fingerprint,
        "repo_commit": repo_commit, "repo_dirty": bool(repo_dirty),
        "trainable_inventory": inventory,
        "trainable_inventory_sha256": sha256_json(inventory),
        "parent_checkpoint_sha256": parent_checkpoint_sha,
        "parent_completed_optimizer_step": expected_parent_step,
        "utc": datetime.now(timezone.utc).isoformat(),
        "loss_log_tail": list(loss_log_tail[-20:]),
        "health_state": health.state_dict(),
    }


def atomic_save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = {
        "schema_version": "m2ba_a1_checkpoint_file_v1",
        "checkpoint": path.name, "sha256": digest,
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
        "parent_completed_optimizer_step": payload.get("parent_completed_optimizer_step"),
        "completed_optimizer_step": payload.get("completed_optimizer_step"),
        "run_validity": payload.get("run_validity"),
    }
    temporary_sidecar = path.with_suffix(path.suffix + ".sha256.json.tmp")
    final_sidecar = path.with_suffix(path.suffix + ".sha256.json")
    temporary_sidecar.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_sidecar, final_sidecar)
    return digest


def load_checkpoint_strict(
    path: Path, *, adapter: nn.Module, optimizer: torch.optim.Optimizer,
    config: A1TrainConfig, expected_fixture_sha: str,
    expected_runtime_sha: str, expected_base_fingerprint: str,
    expected_repo_commit: str, expected_repo_dirty: bool,
    health: A1HealthMonitor, map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    run_root = path.parent.parent
    invalid_marker = run_root / "INVALID_RUN.json"
    if invalid_marker.exists():
        raise RuntimeError(f"checkpoint belongs to quarantined invalid run: {invalid_marker}")
    sidecar_path = path.with_suffix(path.suffix + ".sha256.json")
    if not sidecar_path.is_file():
        raise RuntimeError("checkpoint SHA sidecar missing")
    sidecar = json.loads(sidecar_path.read_text())
    actual_file_sha = sha256_file(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    step = payload.get("completed_optimizer_step")
    parent_sha = payload.get("parent_checkpoint_sha256")
    schedule = tuple(config.checkpoint_steps)
    expected_parent_step = None
    if type(step) is int and step in schedule:
        index = schedule.index(step)
        expected_parent_step = None if index == 0 else schedule[index - 1]

    def validate_chain_node(node_path: Path, expected_step: int) -> bool:
        node_sidecar_path = node_path.with_suffix(node_path.suffix + ".sha256.json")
        if not node_path.is_file() or not node_sidecar_path.is_file():
            return False
        node_sidecar = json.loads(node_sidecar_path.read_text())
        node_payload = torch.load(node_path, map_location="cpu", weights_only=False)
        node_sha = sha256_file(node_path)
        idx = schedule.index(expected_step)
        predecessor = None if idx == 0 else schedule[idx - 1]
        valid = (
            node_sidecar.get("schema_version") == "m2ba_a1_checkpoint_file_v1"
            and node_sidecar.get("checkpoint") == node_path.name
            and node_sidecar.get("sha256") == node_sha
            and node_sidecar.get("completed_optimizer_step") == expected_step
            and node_sidecar.get("run_validity") == RUN_VALID
            and node_sidecar.get("parent_completed_optimizer_step") == predecessor
            and node_payload.get("schema_version") == SCHEMA_VERSION
            and node_payload.get("run_validity") == RUN_VALID
            and node_payload.get("completed_optimizer_step") == expected_step
            and node_payload.get("parent_completed_optimizer_step") == predecessor
            and node_payload.get("parent_checkpoint_sha256") == node_sidecar.get("parent_checkpoint_sha256")
        )
        if not valid:
            return False
        if predecessor is None:
            return node_payload.get("parent_checkpoint_sha256") is None
        parent_path = node_path.parent / f"step_{predecessor:04d}.pt"
        return (
            node_payload.get("parent_checkpoint_sha256") == sha256_file(parent_path)
            if parent_path.is_file() else False
        ) and validate_chain_node(parent_path, predecessor)

    parent_chain_ok = type(step) is int and step in schedule and validate_chain_node(path, step)
    expected_sampler = {
        "schema_version": SAMPLER_SCHEMA, "kind": "fixed_A",
        "cursor": step * config.gradient_accumulation if type(step) is int else None,
    }
    checks = {
        "schema": payload.get("schema_version") == SCHEMA_VERSION,
        "valid_run": payload.get("run_validity") == RUN_VALID,
        "file_sha": sidecar.get("sha256") == actual_file_sha,
        "sidecar_name": sidecar.get("checkpoint") == path.name,
        "sidecar_parent": sidecar.get("parent_checkpoint_sha256") == payload.get("parent_checkpoint_sha256"),
        "sidecar_parent_step": sidecar.get("parent_completed_optimizer_step") == expected_parent_step,
        "sidecar_schema": sidecar.get("schema_version") == "m2ba_a1_checkpoint_file_v1",
        "sidecar_validity": sidecar.get("run_validity") == RUN_VALID,
        "sidecar_step": sidecar.get("completed_optimizer_step") == step,
        "scheduled_step": type(step) is int and step in config.checkpoint_steps,
        "parent_chain": parent_chain_ok,
        "microstep": payload.get("microstep") == 0,
        "grad_accum": payload.get("gradient_accumulation") == config.gradient_accumulation,
        "config": payload.get("config_sha256") == config.fingerprint(),
        "fixture": payload.get("fixture_manifest_sha256") == expected_fixture_sha,
        "runtime": payload.get("runtime_contract_sha256") == expected_runtime_sha,
        "base": payload.get("base_checkpoint_inventory_fingerprint") == expected_base_fingerprint,
        "repo_commit": payload.get("repo_commit") == expected_repo_commit,
        "repo_dirty": payload.get("repo_dirty") is expected_repo_dirty,
        "inventory": payload.get("trainable_inventory") == trainable_inventory(adapter),
        "inventory_sha": payload.get("trainable_inventory_sha256") == sha256_json(payload.get("trainable_inventory")),
        "sampler": payload.get("sampler_state") == expected_sampler,
        "optimizer": isinstance(payload.get("optimizer_state"), dict)
            and set(payload["optimizer_state"]) == {"state", "param_groups"}
            and len(payload["optimizer_state"]["param_groups"]) == 1
            and len(payload["optimizer_state"]["param_groups"][0].get("params", [])) == 17
            and set(payload["optimizer_state"]["state"]).issubset(set(payload["optimizer_state"]["param_groups"][0]["params"]))
            and (step == 0 or len(payload["optimizer_state"]["state"]) == 17),
        "optimizer_hparams": isinstance(payload.get("optimizer_state"), dict)
            and payload["optimizer_state"]["param_groups"][0].get("lr") == config.learning_rate
            and payload["optimizer_state"]["param_groups"][0].get("weight_decay") == config.weight_decay
            and tuple(payload["optimizer_state"]["param_groups"][0].get("betas", ())) == (config.beta1, config.beta2),
        "scheduler_contract": payload.get("lr_scheduler_state") is None
            and payload.get("lr_scheduler_config") == {"kind": "constant"},
        "scaler_contract": "grad_scaler_state" in payload and payload.get("grad_scaler_state") is None,
        "rng_contract": isinstance(payload.get("rng_state"), dict)
            and set(payload["rng_state"]) == {"schema_version", "python", "numpy", "torch_cpu", "torch_cuda_all"}
            and payload["rng_state"].get("schema_version") == RNG_SCHEMA,
        "health_contract": isinstance(payload.get("health_state"), dict)
            and set(payload["health_state"]) == {"schema_version", "loss0", "streaks"}
            and payload["health_state"].get("schema_version") == HEALTH_SCHEMA
            and (payload["health_state"]["loss0"] is None or isinstance(payload["health_state"]["loss0"], float))
            and isinstance(payload["health_state"]["streaks"], dict)
            and all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in payload["health_state"]["streaks"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"checkpoint fingerprint/schema mismatch: {checks}")
    adapter.load_state_dict(payload["adapter_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    health.load_state_dict(payload["health_state"])
    restore_rng_state(payload["rng_state"])
    if not rng_states_equal(capture_rng_state(), payload["rng_state"]):
        raise RuntimeError("RNG restore verification failed")
    return payload


def write_invalid_run_marker(
    run_root: Path, *, reason: str, attempted_step: int | None,
    last_completed_optimizer_step: int | None,
) -> Path:
    marker = run_root / "INVALID_RUN.json"
    value = {
        "schema_version": "m2ba_a1_invalid_run_v1",
        "status": "INVALID_HEALTH_STOP", "reason": reason,
        "attempted_step": attempted_step,
        "last_completed_optimizer_step": last_completed_optimizer_step,
        "resume_forbidden": True,
        "evidence_eligible": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    quarantine = run_root / "QUARANTINED_CHECKPOINTS"
    quarantine.mkdir(exist_ok=True)
    for checkpoint in sorted((run_root / "checkpoints").glob("step_*.pt")) if (run_root / "checkpoints").exists() else []:
        os.replace(checkpoint, quarantine / checkpoint.name)
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")
        if sidecar.exists():
            os.replace(sidecar, quarantine / sidecar.name)
    return marker


def validate_training_authorization(
    path: Path, *, repo_commit: str, config_sha: str,
    snapshot_sha: str, runtime_sha: str, checkpoint_fingerprint: str,
    authorization: str = "A1_EXPLORATORY_200_AUTHORIZED",
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("A1 authorization artifact missing")
    value = json.loads(path.read_text())
    expected = {
        "schema_version": "m2ba_a1_training_authorization_v1",
        "authorization": authorization,
        "repo_commit": repo_commit, "config_sha256": config_sha,
        "train_fixture_snapshot_sha256": snapshot_sha,
        "runtime_contract_sha256": runtime_sha,
        "checkpoint_inventory_fingerprint": checkpoint_fingerprint,
    }
    checks = {key: value.get(key) == expected_value for key, expected_value in expected.items()}
    if not all(checks.values()):
        raise RuntimeError(f"A1 authorization mismatch: {checks}")
    if not isinstance(value.get("authorized_by"), str) or not value["authorized_by"].strip():
        raise RuntimeError("A1 authorization missing authorized_by")
    return value


def assert_only_adapter_gradients(adapter: nn.Module, base: nn.Module) -> None:
    actual = [name for name, p in adapter.named_parameters() if p.grad is not None]
    expected = [name for name, p in adapter.named_parameters() if p.requires_grad]
    if actual != expected:
        raise RuntimeError(f"adapter gradient inventory mismatch: {actual} != {expected}")
    leaked = [name for name, p in base.named_parameters() if p.grad is not None]
    if leaked:
        raise RuntimeError(f"base received gradients: {leaked[:8]}")


class A1Trainer:
    """Small deterministic optimizer driver around frozen Wan tensor callbacks.

    ``student_forward`` must return ``(fused_pre_head, diagnostics)`` for the
    exact same masked query on every microstep. ``empty_forward`` is evaluated
    only on log steps and must use the same masked query with physical bypass.
    No B/DEV callback exists by design.
    """

    def __init__(
        self, *, adapter: nn.Module, base: nn.Module, teacher_target: Tensor,
        student_forward: Callable[[], tuple[Tensor, Mapping[str, Any]]],
        empty_forward: Callable[[], Tensor], config: A1TrainConfig,
        base_fingerprint: Callable[[], str],
        health_metrics: Callable[[Mapping[str, Any], dict[str, Any]], Mapping[str, Any]],
    ) -> None:
        _reject_b_reference({"student": repr(student_forward), "empty": repr(empty_forward)})
        if teacher_target.requires_grad:
            raise ValueError("teacher target must be detached")
        self.adapter, self.base = adapter, base
        self.teacher_target = teacher_target.detach()
        self.student_forward, self.empty_forward = student_forward, empty_forward
        self.config, self.base_fingerprint = config, base_fingerprint
        self.health_metrics = health_metrics
        self.initial_base_fingerprint = base_fingerprint()
        self.optimizer = torch.optim.AdamW(
            [p for p in adapter.parameters() if p.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        self.health = A1HealthMonitor()

    def optimizer_step(self, step: int) -> tuple[dict[str, Any], Mapping[str, Any]]:
        self.optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        last_diag: Mapping[str, Any] = {}
        for _microstep in range(self.config.gradient_accumulation):
            prediction, last_diag = self.student_forward()
            loss = hidden_reconstruction_loss(
                prediction, self.teacher_target,
                slice(self.config.target_token_start, self.config.target_token_end),
            )
            (loss / self.config.gradient_accumulation).backward()
            accumulated += float(loss.detach())
        grad_norm = preclip_grad_norm(p for p in self.adapter.parameters() if p.requires_grad)
        assert_only_adapter_gradients(self.adapter, self.base)
        mean_loss = accumulated / self.config.gradient_accumulation
        record = {
            "step": step, "loss_total": mean_loss,
            "loss_hidden_recon": mean_loss, "adapter_grad_norm": grad_norm,
            "base_grad_nonzero_count": 0,
            "base_fingerprint_unchanged": self.base_fingerprint() == self.initial_base_fingerprint,
        }
        record.update(self.health_metrics(last_diag, record))
        # Frozen health is evaluated on pre-clip gradients and before any
        # optimizer mutation. A stopped run therefore cannot leak one bad step.
        self.health.update(step, record)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.adapter.parameters() if p.requires_grad], self.config.grad_clip
        )
        self.optimizer.step()
        return record, last_diag
