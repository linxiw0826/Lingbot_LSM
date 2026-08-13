#!/usr/bin/env python3
"""Fail-closed GPU probe for the frozen M2-B-A Wan runtime contract.

This program performs two identical clean-x0, t=0 direct forwards through the
real low-noise Wan model.  It never enters a scheduler loop, decodes a VAE
latent, writes a video, enables gradients, or updates model state.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


SCHEMA = "m2ba_a01_wan_runtime_contract_v1"
DETERMINISM_ATOL = 2.0e-3
DETERMINISM_RTOL = 2.0e-3
EXPECTED_X0 = (16, 21, 58, 104)
EXPECTED_Y = (20, 21, 58, 104)
EXPECTED_HIDDEN_DIM = 5120
EXPECTED_TOKENS_PER_LATENT_FRAME = 1508
SMALL_LIMIT = 64 * 1024 * 1024


class CleanT0Unsupported(RuntimeError):
    """Raised only while selecting or directly executing the t=0 expert."""


class HookContractError(RuntimeError):
    """Raised by hook shape/slice validation, not attributed to t=0 support."""


def raise_runtime_stage_error(stage: str, exc: Exception) -> None:
    """Classify runtime failures without blaming generic GPU/API errors on t=0."""
    import torch

    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in lowered:
        raise RuntimeError(f"BLOCKED_GPU_RUNTIME: {stage}: {text}") from exc
    if isinstance(exc, TypeError):
        raise RuntimeError(f"BLOCKED_MINIMAL_FORWARD_API_MISSING: {stage}: {text}") from exc
    timestep_markers = ("timestep", "time step", "t=0", "t tensor", "time embedding")
    if isinstance(exc, (ValueError, AssertionError)) and any(marker in lowered for marker in timestep_markers):
        raise CleanT0Unsupported(f"{stage}: {text}") from exc
    raise RuntimeError(f"BLOCKED_GPU_RUNTIME: {stage}: {text}") from exc


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_inventory_fingerprint(root: Path) -> str:
    inventory = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        row = {
            "path": str(path.relative_to(root)),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= SMALL_LIMIT and path.suffix.lower() in {".json", ".txt", ".yaml", ".yml", ".index"}:
            row["sha256"] = sha256(path)
        inventory.append(row)
    return stable_json_fingerprint(inventory)


def causal_latent_index(local_frame: int, temporal_stride: int, latent_frames: int) -> int:
    """Map a pixel frame to causal-Wan latent time: ceil(frame / stride)."""
    if local_frame < 0 or temporal_stride <= 0 or latent_frames <= 0:
        raise ValueError("invalid temporal mapping inputs")
    index = (local_frame + temporal_stride - 1) // temporal_stride
    if index >= latent_frames:
        raise ValueError(f"latent index {index} outside [0,{latent_frames})")
    return index


def token_slice_for_frame(
    local_frame: int,
    *,
    temporal_stride: int,
    latent_frames: int,
    latent_h: int,
    latent_w: int,
    patch_size: Iterable[int],
) -> dict[str, int]:
    patch = tuple(int(value) for value in patch_size)
    if len(patch) != 3 or patch[0] != 1:
        raise ValueError(f"probe requires temporal patch size 1, got {patch}")
    if latent_h % patch[1] or latent_w % patch[2]:
        raise ValueError("latent spatial size is not patch divisible")
    latent_t = causal_latent_index(local_frame, temporal_stride, latent_frames)
    tokens_per_t = (latent_h // patch[1]) * (latent_w // patch[2])
    return {
        "local_frame": local_frame,
        "latent_t": latent_t,
        "patch_t": latent_t,
        "token_start": latent_t * tokens_per_t,
        "token_end": (latent_t + 1) * tokens_per_t,
        "token_count": tokens_per_t,
    }


def map_fixture_frames(
    fixture: dict[str, Any], *, temporal_stride: int, latent_frames: int,
    latent_h: int, latent_w: int, patch_size: Iterable[int],
    include_support: bool,
) -> dict[str, Any]:
    """Map frozen full frames using their exact planner window/local positions.

    This is a causal receptive-field assignment derived from Wan's causal VAE
    convention, not an empirically perturbed attribution measurement.
    """
    windows = fixture.get("planner_windows") or []
    target = int(fixture["target_full_frame"])
    support = fixture.get("support_full_half_open")
    wanted = [target]
    if include_support:
        wanted = list(range(int(support[0]), int(support[1])))
    rows = []
    for full_frame in wanted:
        owners = [window for window in windows if window["owned_half_open"][0] <= full_frame < window["owned_half_open"][1]]
        if len(owners) != 1:
            raise ValueError(f"full frame {full_frame}: expected one owner, got {len(owners)}")
        window = owners[0]
        locals_ = [index for index, source in enumerate(window["source_frame_index"]) if source == full_frame]
        if len(locals_) != 1:
            raise ValueError(f"full frame {full_frame}: expected one local position, got {locals_}")
        mapping = token_slice_for_frame(
            locals_[0], temporal_stride=temporal_stride,
            latent_frames=latent_frames, latent_h=latent_h, latent_w=latent_w,
            patch_size=patch_size,
        )
        rows.append({"full_frame": full_frame, "window_id": window["window_index"], **mapping})
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for row in rows:
        key = (row["window_id"], row["latent_t"], row["token_start"], row["token_end"])
        groups.setdefault(key, []).append(row["full_frame"])
    deduplicated = [
        {"window_id": key[0], "latent_t": key[1], "token_start": key[2],
         "token_end": key[3], "full_frames": frames,
         "full_frame_half_open": [min(frames), max(frames) + 1]}
        for key, frames in sorted(groups.items())
    ]
    target_rows = [row for row in rows if row["full_frame"] == target]
    if len(target_rows) != 1:
        raise ValueError("target mapping is not unique")
    return {
        "status": "PASS",
        "semantics": "causal_receptive_field_assignment_not_empirical_attribution",
        "formula": "latent_t=ceil(window_local_frame/vae_temporal_stride); patch_t=latent_t because temporal_patch_size=1",
        "source": "Wan causal VAE temporal convention + frozen Phase1 planner source/owned mapping + runtime vae_stride/patch_size",
        "target": target_rows[0],
        "per_frame": rows if include_support else target_rows,
        "deduplicated_many_to_one_groups": deduplicated,
        "group_boundaries": [group["full_frame_half_open"] for group in deduplicated],
        "complete": len(rows) == len(wanted),
    }


def runtime_decision(static_pass: bool, evidence_pass: bool, blocker: str | None) -> str:
    if not static_pass:
        return "BLOCKED_STATIC_FACTS"
    if evidence_pass and blocker is None:
        return "FIXTURE_FREEZE_PASS"
    return blocker or "BLOCKED_GPU_RUNTIME"


def tensor_stats(value: Any) -> dict[str, Any]:
    import torch

    detached = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.isfinite(detached).all().item()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()),
        "rms": float(detached.square().mean().sqrt().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def tensor_digest(value: Any) -> str:
    import torch
    contiguous = value.detach().to("cpu").contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def state_inventory(module: Any) -> dict[str, Any]:
    """Low-cost mutation detector for a very large frozen model.

    Every parameter/buffer contributes identity, shape, dtype, device and its
    PyTorch version counter.  A deterministic first/middle/last content sample
    supplements the version counter.  This detects all ordinary in-place
    PyTorch writes without hashing roughly 28 GB twice; it is deliberately
    labelled a mutation detector, not a full cryptographic content hash.
    """
    import torch

    rows = []
    for kind, iterator in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, value in iterator:
            flat = value.detach().reshape(-1)
            sample = []
            if flat.numel():
                indexes = sorted({0, flat.numel() // 2, flat.numel() - 1})
                sample = [float(flat[index].float().cpu().item()) for index in indexes]
            rows.append({
                "kind": kind,
                "name": name,
                "object_id": id(value),
                "data_ptr": int(value.data_ptr()) if value.numel() else 0,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "version": int(getattr(value, "_version", -1)),
                "requires_grad": bool(getattr(value, "requires_grad", False)),
                "sample": sample,
            })
    # Wan intentionally does not register freqs as a buffer.  Include this
    # mutable plain tensor explicitly after expert selection/device placement.
    for name in ("freqs",):
        value = getattr(module, name, None)
        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1)
            indexes = sorted({0, flat.numel() // 2, flat.numel() - 1}) if flat.numel() else []
            rows.append({
                "kind": "plain_tensor_attribute", "name": name,
                "object_id": id(value), "data_ptr": int(value.data_ptr()) if value.numel() else 0,
                "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                "version": int(getattr(value, "_version", -1)), "requires_grad": False,
                "sample": [float(flat[index].float().cpu().item()) for index in indexes],
            })
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "method": "all_parameter_buffer_and_explicit_plain_tensor_attribute_identity_shape_dtype_device_version_counter_plus_first_middle_last_content",
        "strength": "complete for ordinary tracked in-place PyTorch mutation; sampled-content supplement is not a full cryptographic weight hash",
        "tensor_count": len(rows),
        "digest": hashlib.sha256(encoded).hexdigest(),
        "rows": rows,
    }


def build_conditioning_only(pipeline: Any, image: Any, window: dict[str, np.ndarray],
                            prompt: str, spatial: Any, device: Any) -> dict[str, Any]:
    """Construct exact Wan I2V conditioning without scheduler or random noise."""
    import torch
    import torch.nn.functional as torch_f
    import torchvision.transforms.functional as transform_f
    from einops import rearrange
    from wan.utils.cam_utils import (
        compute_relative_poses, get_Ks_transformed, get_plucker_embeddings,
        interpolate_camera_poses,
    )

    frame_num = int(window["rgb"].shape[0])
    latent_frames = (frame_num - 1) // int(pipeline.vae_stride[0]) + 1
    latent_h, latent_w = int(spatial.latent_h), int(spatial.latent_w)
    height, width = int(spatial.pixel_h), int(spatial.pixel_w)
    tokens_per_t = (latent_h // int(pipeline.patch_size[1])) * (latent_w // int(pipeline.patch_size[2]))
    max_seq_len = latent_frames * tokens_per_t
    max_seq_len = int(math.ceil(max_seq_len / pipeline.sp_size)) * pipeline.sp_size

    image_tensor = transform_f.to_tensor(image).sub_(0.5).div_(0.5).to(device)
    mask = torch.ones(1, frame_num, latent_h, latent_w, device=device)
    mask[:, 1:] = 0
    mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, mask.shape[1] // 4, 4, latent_h, latent_w).transpose(1, 2)[0]

    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([prompt], device)

    poses = window["pose"].astype(np.float32)
    effective = ((len(poses) - 1) // 4) * 4 + 1
    poses = poses[:effective]
    actions = window["action"].astype(np.float32)[:effective]
    intrinsics = torch.from_numpy(window["intrinsics"].astype(np.float32)).float()
    intrinsics = get_Ks_transformed(
        intrinsics, height_org=480, width_org=832,
        height_resize=height, width_resize=width,
        height_final=height, width_final=width,
    )[0]
    interpolated = interpolate_camera_poses(
        src_indices=np.linspace(0, len(poses) - 1, len(poses)),
        src_rot_mat=poses[:, :3, :3], src_trans_vec=poses[:, :3, 3],
        tgt_indices=np.linspace(0, len(poses) - 1, int((len(poses) - 1) // 4) + 1),
    )
    interpolated = compute_relative_poses(interpolated, framewise=True).to(device)
    intrinsics = intrinsics.repeat(len(interpolated), 1).to(device)
    sampled_actions = torch.from_numpy(actions[::4]).float().to(device) if pipeline.control_type == "act" else None
    plucker = get_plucker_embeddings(
        interpolated, intrinsics, height, width,
        only_rays_d=sampled_actions is not None,
    )
    plucker = rearrange(
        plucker, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=int(height // latent_h), c2=int(width // latent_w),
    )[None, ...]
    plucker = rearrange(
        plucker, "b (f h w) c -> b c f h w",
        f=latent_frames, h=latent_h, w=latent_w,
    ).to(pipeline.param_dtype)
    if sampled_actions is not None:
        action_tensor = sampled_actions[:, None, None, :].repeat(1, height, width, 1)
        action_tensor = rearrange(
            action_tensor, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(height // latent_h), c2=int(width // latent_w),
        )[None, ...]
        action_tensor = rearrange(
            action_tensor, "b (f h w) c -> b c f h w",
            f=latent_frames, h=latent_h, w=latent_w,
        ).to(pipeline.param_dtype)
        plucker = torch.cat([plucker, action_tensor], dim=1)

    first_condition = torch.concat([
        torch_f.interpolate(image_tensor[None].cpu(), size=(height, width), mode="bicubic").transpose(0, 1),
        torch.zeros(3, frame_num - 1, height, width),
    ], dim=1).to(device)
    condition_latent = pipeline.vae.encode([first_condition])[0]
    y = torch.concat([mask, condition_latent])
    return {
        "context": [context[0]], "y": y,
        "dit_cond_dict": {"c2ws_plucker_emb": plucker.chunk(1, dim=0)},
        "max_seq_len": max_seq_len,
        "construction": "conditioning_only_no_scheduler_no_rng_no_noise",
    }


def direct_forward_conditioning(prepared: dict[str, Any]) -> dict[str, Any]:
    """Validate conditioning shapes and apply Wan's batch-list contract once.

    ``build_conditioning_only`` intentionally returns the per-sample I2V
    conditioning tensor as ``[C, F, H, W]``.  Wan's direct forward accepts a
    batch as a Python list, so the list wrapper belongs here, not in the
    conditioning builder (and the tensor must never be indexed to add it).
    """
    y = prepared["y"]
    if getattr(y, "ndim", None) != 4:
        raise RuntimeError(
            f"conditioning y must be a single [C,F,H,W] tensor, got "
            f"{type(y).__name__} shape={getattr(y, 'shape', None)}"
        )
    context = prepared["context"]
    if not isinstance(context, list) or len(context) != 1:
        raise RuntimeError("conditioning context must already be a one-sample list")
    dit_cond_dict = prepared["dit_cond_dict"]
    if not isinstance(dit_cond_dict, dict):
        raise RuntimeError("dit_cond_dict must be a mapping")
    return {"context": context, "y": [y], "dit_cond_dict": dit_cond_dict}


def validate_static(frozen: dict[str, Any], fingerprints: dict[str, Any], repo: Path) -> dict[str, Any]:
    errors = []
    if frozen.get("status") != "BLOCKED_MINIMAL_FORWARD_API_MISSING":
        errors.append(f"unexpected frozen status: {frozen.get('status')}")
    if frozen.get("checkpoint_fingerprint") != fingerprints.get("checkpoint_fingerprint"):
        errors.append("checkpoint fingerprint chain mismatch")
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=False,
    )
    head = head_result.stdout.strip()
    if head != frozen.get("repo_commit"):
        errors.append(f"repo commit mismatch: runtime={head} frozen={frozen.get('repo_commit')}")
    train = [item for item in frozen.get("fixtures", []) if item.get("role") == "TRAIN"]
    if len(train) != 1:
        errors.append(f"expected one TRAIN fixture, got {len(train)}")
    for fixture in frozen.get("fixtures", []):
        manifest = Path(fixture.get("manifest_path", ""))
        if not manifest.is_file() or sha256(manifest) != fixture.get("manifest_sha256"):
            errors.append(f"manifest identity mismatch: {manifest}")
    for item in fingerprints.get("inputs", []):
        path = Path(item.get("path", ""))
        if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
            errors.append(f"frozen input identity mismatch: {path}")
    checkpoint_root = Path(frozen.get("checkpoint_root", ""))
    if not checkpoint_root.is_dir():
        errors.append(f"checkpoint root missing: {checkpoint_root}")
    elif checkpoint_inventory_fingerprint(checkpoint_root) != frozen.get("checkpoint_fingerprint"):
        errors.append("checkpoint inventory fingerprint changed since static freeze")
    return {"pass": not errors, "errors": errors, "repo_head": head, "train": train[0] if len(train) == 1 else None}


def _frame_to_pil(frame: np.ndarray):
    from PIL import Image

    hwc = np.transpose(frame, (1, 2, 0))
    pixels = np.clip(hwc * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as torch_f

    repo = Path(args.repo).resolve()
    src = repo / "src"
    ref = repo / "refs" / "lingbot-world"
    for path in (str(ref), str(src), str(repo)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from pipeline.eval.stage1_upperbound import runtime_spatial_plan
    from pipeline.v6.latentconcat_infer import _load_raw_pipeline
    from pipeline.v7.phase1.planner import plan_windows, slice_modalities
    from pipeline.v7.phase1.run import _load_case
    from wan.configs import MAX_AREA_CONFIGS

    output = Path(args.output)
    frozen_path = output / "frozen_fixture_manifest.json"
    fingerprint_path = output / "file_fingerprints.json"
    frozen = json.loads(frozen_path.read_text())
    fingerprints = json.loads(fingerprint_path.read_text())
    static = validate_static(frozen, fingerprints, repo)
    if not static["pass"]:
        raise RuntimeError("static evidence validation failed: " + "; ".join(static["errors"]))
    fixture = static["train"]

    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_GPU_RUNTIME: CUDA unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("BLOCKED_GPU_RUNTIME: --device must be CUDA")

    data = _load_case(frozen["cases_root"], fixture["case_id"], int(fixture["total_frames"]))
    plans = plan_windows(int(fixture["total_frames"]), tuple(fixture["support_full_half_open"]), context_frames=81, seam_buffer=8)
    plan = plans[int(fixture["target_window_id"])]
    window = slice_modalities(data, plan)
    target_local_values = [index for index, source in enumerate(plan.source_frame_index) if source == fixture["target_full_frame"]]
    if target_local_values != fixture["target_window_local_indices"] or len(target_local_values) != 1:
        raise RuntimeError(f"target local mapping drift: {target_local_values}")
    target_local = target_local_values[0]

    generation_args = SimpleNamespace(
        ckpt_dir=frozen["checkpoint_root"], ft_model_dir=None, ft_high_model_dir=None,
        lora_path=None, lora_rank=0, lora_alpha=0.0, lora_targets="",
        size="480*832", frame_num=81, num_inference_steps=40,
        sample_shift=10.0, guide_scale=5.0,
        prompt=data["prompt"], seed=42,
    )
    pipeline = _load_raw_pipeline(generation_args, device)
    configured_boundary = float(pipeline.boundary) * float(pipeline.num_train_timesteps)
    expected_boundary_observation = 0.947 * 1000.0
    boundary = configured_boundary
    selection_t0 = torch.tensor(0.0, device=device, dtype=torch.float32)
    try:
        model = pipeline._prepare_model_for_timestep(selection_t0, boundary, offload_model=True)
    except Exception as exc:
        raise_runtime_stage_error("t=0 expert selection", exc)
    if model is not pipeline.low_noise_model:
        raise CleanT0Unsupported("t=0 selected object is not pipeline.low_noise_model")
    model.eval().requires_grad_(False)
    # Wan's forward lazily moves this intentionally-unregistered tensor. Do the
    # documented device placement before the measured snapshots/forwards.
    model.freqs = model.freqs.to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base requires_grad freeze failed")

    input_h, input_w = map(int, window["rgb"].shape[-2:])
    spatial = runtime_spatial_plan(input_h, input_w, MAX_AREA_CONFIGS[generation_args.size], pipeline.vae_stride, pipeline.patch_size)
    image = _frame_to_pil(window["rgb"][0])
    try:
        prepared = build_conditioning_only(pipeline, image, window, data["prompt"], spatial, device)
    except Exception as exc:
        raise RuntimeError(f"BLOCKED_EXACT_CONDITIONING: {type(exc).__name__}: {exc}") from exc

    # Encode the exact clean query window.  This is the same encode operation
    # used inside _build_capture_inputs, without its low-noise perturbation.
    clean = torch.from_numpy(np.ascontiguousarray(window["rgb"])).float().permute(1, 0, 2, 3)
    if clean.shape[-2:] != (spatial.pixel_h, spatial.pixel_w):
        clean = torch_f.interpolate(
            clean.unsqueeze(0), size=(clean.shape[1], spatial.pixel_h, spatial.pixel_w),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
    x0 = pipeline.vae.encode([clean.to(device)])[0]
    forward_conditioning = direct_forward_conditioning(prepared)
    y = prepared["y"]
    if tuple(x0.shape) != EXPECTED_X0:
        raise RuntimeError(f"x0 shape drift: {tuple(x0.shape)} != {EXPECTED_X0}")
    if tuple(y.shape) != EXPECTED_Y:
        raise RuntimeError(f"y shape drift: {tuple(y.shape)} != {EXPECTED_Y}")
    if not torch.isfinite(x0.float()).all() or not torch.isfinite(y.float()).all():
        raise RuntimeError("non-finite x0 or y")

    fixture_mappings = {}
    for item in frozen["fixtures"]:
        fixture_mappings[item["case_id"]] = map_fixture_frames(
            item, temporal_stride=int(pipeline.vae_stride[0]),
            latent_frames=int(x0.shape[1]), latent_h=int(x0.shape[2]), latent_w=int(x0.shape[3]),
            patch_size=pipeline.patch_size, include_support=item["role"] == "TRAIN",
        )
    mapping = fixture_mappings[fixture["case_id"]]["target"]
    if not all(value.get("status") == "PASS" and value.get("complete") for value in fixture_mappings.values()):
        raise RuntimeError("fixture frame-to-token mappings incomplete")
    if mapping["token_count"] != EXPECTED_TOKENS_PER_LATENT_FRAME:
        raise RuntimeError(f"tokens-per-frame drift: {mapping}")
    expected_seq = int(x0.shape[1]) * mapping["token_count"]
    if prepared["max_seq_len"] != expected_seq:
        raise RuntimeError(f"max_seq_len drift: {prepared['max_seq_len']} != {expected_seq}")

    before = state_inventory(model)
    captures: list[dict[str, Any]] = []
    first_slices: dict[str, Any] = {}
    active: dict[str, Any] = {"repeat": -1}

    def make_hook(name: str):
        def hook(_module, hook_args):
            hidden = hook_args[0]
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != expected_seq or hidden.shape[2] != EXPECTED_HIDDEN_DIM:
                raise HookContractError(f"{name} hidden shape drift: {tuple(hidden.shape)}")
            selected = hidden[:, mapping["token_start"]:mapping["token_end"], :].detach()
            if selected.shape[1] != EXPECTED_TOKENS_PER_LATENT_FRAME:
                raise HookContractError(f"{name} target slice drift: {tuple(selected.shape)}")
            repeat = int(active["repeat"])
            record = {"repeat": repeat, "hook": name, "full_shape": list(hidden.shape), "target_slice": {**tensor_stats(selected), "sha256_raw_bytes": tensor_digest(selected)}}
            if repeat == 0:
                first_slices[name] = selected.to("cpu", dtype=torch.float32).contiguous()
            else:
                reference = first_slices[name]
                candidate = selected.to("cpu", dtype=torch.float32).contiguous()
                difference = (candidate - reference).abs()
                record["repeat_difference"] = {
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                    "allclose": bool(torch.allclose(candidate, reference, atol=DETERMINISM_ATOL, rtol=DETERMINISM_RTOL)),
                }
            captures.append(record)
        return hook

    handles = [
        model.blocks[0].register_forward_pre_hook(make_hook("low_noise_model.blocks.0")),
        model.head.register_forward_pre_hook(make_hook("low_noise_model.head")),
    ]
    outputs = []
    first_output = None
    t0 = torch.zeros(1, device=device, dtype=torch.float32)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
            for repeat in range(2):
                active["repeat"] = repeat
                try:
                    result = model(
                        [x0], t=t0, context=forward_conditioning["context"],
                        seq_len=prepared["max_seq_len"], y=forward_conditioning["y"],
                        dit_cond_dict=forward_conditioning["dit_cond_dict"],
                    )[0]
                except HookContractError:
                    raise
                except Exception as exc:
                    raise_runtime_stage_error("direct clean t=0 forward", exc)
                current = result.detach().to("cpu", dtype=torch.float32).contiguous()
                record = {**tensor_stats(result), "sha256_float32_bytes": tensor_digest(current)}
                if repeat == 0:
                    first_output = current
                else:
                    difference = (current - first_output).abs()
                    record["repeat_difference"] = {
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                        "allclose": bool(torch.allclose(current, first_output, atol=DETERMINISM_ATOL, rtol=DETERMINISM_RTOL)),
                        "exact_digest_match": tensor_digest(current) == tensor_digest(first_output),
                    }
                outputs.append(record)
    finally:
        for handle in handles:
            handle.remove()

    after = state_inventory(model)
    mutation_free = before["digest"] == after["digest"]
    determinism_rows = [row.get("repeat_difference") for row in captures if row["repeat"] == 1]
    output_deterministic = len(outputs) == 2 and outputs[1].get("repeat_difference", {}).get("allclose", False)
    deterministic = len(determinism_rows) == 2 and all(row and row["allclose"] for row in determinism_rows) and output_deterministic
    finite = all(row["target_slice"]["finite"] for row in captures) and all(row["finite"] for row in outputs)
    evidence_pass = deterministic and mutation_free and finite and len(captures) == 4
    if not evidence_pass:
        raise RuntimeError(
            f"runtime contract failed: deterministic={deterministic} mutation_free={mutation_free} finite={finite} captures={len(captures)}"
        )

    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "generated_at_utc": utc(),
        "repo_commit": static["repo_head"],
        "checkpoint_fingerprint": frozen["checkpoint_fingerprint"],
        "fixture": {"case_id": fixture["case_id"], "event_id": fixture["event_id"], "target_full_frame": fixture["target_full_frame"], "window_id": plan.window_index},
        "execution_contract": {
            "model": "pipeline.low_noise_model",
            "model_selection_api": "pipeline._prepare_model_for_timestep",
            "model_selection_timestep": 0.0,
            "model_selection_boundary": boundary,
            "pipeline_configured_boundary": configured_boundary,
            "expected_boundary_947_observation": expected_boundary_observation,
            "configured_boundary_matches_947_observation": math.isclose(
                configured_boundary, expected_boundary_observation,
                rel_tol=0.0, abs_tol=1.0e-3,
            ),
            "selected_object_is_low_noise_model": True,
            "selected_checkpoint_subfolder": "low_noise_model",
            "mode": "eval",
            "grad_enabled": False,
            "requires_grad_all_false": True,
            "forward_count": 2,
            "scheduler_sampling_steps": 0,
            "vae_decode_calls": 0,
            "video_writes": 0,
            "timestep": [0.0],
            "clean_x0_no_noise": True,
        },
        "geometry": {
            "input_pixel_hw": [input_h, input_w],
            "planned_pixel_hw": [spatial.pixel_h, spatial.pixel_w],
            "latent_hw": [spatial.latent_h, spatial.latent_w],
            "vae_stride": list(pipeline.vae_stride),
            "patch_size": list(pipeline.patch_size),
            "x0": tensor_stats(x0), "y": tensor_stats(y),
            "max_seq_len": prepared["max_seq_len"],
            "target_mapping": mapping,
            "all_fixture_mappings": fixture_mappings,
        },
        "hooks": captures,
        "forward_outputs": outputs,
        "determinism": {"scope": "both target-only block0/prehead slices and full direct-forward output latent", "atol": DETERMINISM_ATOL, "rtol": DETERMINISM_RTOL, "pass": deterministic},
        "model_state_mutation": {"pass": mutation_free, "before": {key: value for key, value in before.items() if key != "rows"}, "after": {key: value for key, value in after.items() if key != "rows"}},
        "finite_gate": finite,
        "WAN_RUNTIME_CONTRACT_GATE": "PASS",
    }


def update_outputs(output: Path, runtime: dict[str, Any]) -> None:
    atomic_json(output / "runtime_contract.json", runtime)
    frozen_path = output / "frozen_fixture_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    passed = runtime.get("status") == "PASS"
    if passed:
        mappings = runtime.get("geometry", {}).get("all_fixture_mappings", {})
        complete = len(mappings) == len(frozen.get("fixtures", [])) and all(
            value.get("status") == "PASS" and value.get("complete")
            for value in mappings.values()
        )
        if not complete:
            passed = False
            runtime["status"] = "BLOCKED_INCOMPLETE_FRAME_TOKEN_MAPPING"
            runtime["blocker"] = runtime["status"]
            runtime["WAN_RUNTIME_CONTRACT_GATE"] = "BLOCKED"
            atomic_json(output / "runtime_contract.json", runtime)
        else:
            for fixture in frozen["fixtures"]:
                fixture["frame_to_token_mapping"] = mappings[fixture["case_id"]]
    frozen["status"] = "FIXTURE_FREEZE_PASS" if passed else runtime.get("status", "BLOCKED_GPU_RUNTIME")
    frozen["blockers"] = [] if passed else [frozen["status"]]
    frozen["runtime_contract_path"] = str(output / "runtime_contract.json")
    atomic_json(frozen_path, frozen)
    summary_path = output / "probe_summary.md"
    summary = summary_path.read_text() if summary_path.is_file() else "# M2-B-A A0/A1 fixture freeze probe\n"
    lines = summary.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("- Final status:"):
            lines[index] = f"- Final status: `{frozen['status']}`"
        elif line.startswith("- WAN_RUNTIME_CONTRACT_GATE:"):
            lines[index] = f"- WAN_RUNTIME_CONTRACT_GATE: `{'PASS' if passed else 'BLOCKED'}`"
    summary = "\n".join(lines).rstrip() + "\n"
    summary += "\n## GPU runtime contract\n\n"
    if passed:
        mapping = runtime["geometry"]["target_mapping"]
        summary += f"- Direct forwards: 2 identical clean-x0 calls at t=0; target token slice=[{mapping['token_start']},{mapping['token_end']})\n"
        summary += f"- Determinism: PASS (atol={DETERMINISM_ATOL}, rtol={DETERMINISM_RTOL})\n"
        summary += "- No scheduler sampling loop, VAE decode, video write, gradient, or model-state mutation.\n"
    else:
        summary += f"- Blocker: `{runtime.get('blocker')}`\n"
        summary += f"- Error: `{runtime.get('error')}`\n"
    tmp = summary_path.with_suffix(".md.tmp")
    tmp.write_text(summary, encoding="utf-8")
    os.replace(tmp, summary_path)
    # The runtime stage atomically replaced three artifacts produced by the
    # static stage. Refresh their integrity chain while preserving its scope.
    generated_path = output / "generated_files.txt"
    if generated_path.is_file():
        names = [Path(line.strip()).name for line in generated_path.read_text().splitlines() if line.strip()]
        sums = []
        for name in names + ["generated_files.txt"]:
            path = output / name
            if path.is_file():
                sums.append(f"{sha256(path)}  {name}\n")
        sums_path = output / "SHA256SUMS"
        tmp_sums = sums_path.with_suffix(".tmp")
        tmp_sums.write_text("".join(sums), encoding="utf-8")
        os.replace(tmp_sums, sums_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("REPO", "/mnt/nas/wlx/Memory/projects/Lingbot_LSM"))
    parser.add_argument("--output", default=os.environ.get("PROBE_OUT", "/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811"))
    parser.add_argument("--device", default=os.environ.get("PROBE_DEVICE", "cuda:0"))
    args = parser.parse_args()
    output = Path(args.output)
    try:
        runtime = run(args)
        code = 0
    except Exception as exc:  # fail closed and preserve a machine-readable artifact
        text = f"{type(exc).__name__}: {exc}"
        blocker = "BLOCKED_GPU_RUNTIME"
        if isinstance(exc, CleanT0Unsupported):
            blocker = "BLOCKED_CLEAN_T0_UNSUPPORTED"
        for candidate in (
            "BLOCKED_STATIC_FACTS", "BLOCKED_EXACT_CONDITIONING",
            "BLOCKED_MINIMAL_FORWARD_API_MISSING", "BLOCKED_GPU_RUNTIME",
        ):
            if candidate in text:
                blocker = candidate
        runtime = {
            "schema_version": SCHEMA, "status": blocker,
            "generated_at_utc": utc(), "blocker": blocker,
            "error": text, "traceback": traceback.format_exc(),
            "WAN_RUNTIME_CONTRACT_GATE": "BLOCKED",
        }
        code = 2
    update_outputs(output, runtime)
    print(json.dumps(runtime, indent=2, sort_keys=True))
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
