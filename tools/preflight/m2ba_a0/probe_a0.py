#!/usr/bin/env python3
"""Real-Wan, frozen-fixture A0 parity/instrumentation probe (no training)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def load_fixture_probe_module(repo: Path):
    """Load the sibling helper by verified repo path, never top-level ``tools``."""
    expected = (
        repo / "tools" / "preflight" / "m2ba_a01_fixture_freeze"
        / "probe_wan_runtime.py"
    ).resolve()
    if not expected.is_file():
        raise RuntimeError(f"fixture runtime helper missing: {expected}")
    module_name = "_lingbot_lsm_m2ba_fixture_probe_wan_runtime"
    spec = importlib.util.spec_from_file_location(module_name, expected)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for fixture helper: {expected}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = Path(module.__file__).resolve()
    if loaded != expected:
        raise RuntimeError(f"fixture helper path mismatch: loaded={loaded} expected={expected}")
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().float().cpu().contiguous()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "finite": bool(torch.isfinite(cpu).all()),
        "rms": float(cpu.pow(2).mean().sqrt()),
        "sha256_float32": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "value": cpu,
    }


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "value"}


def comparison(left: dict[str, Any], right: dict[str, Any], atol=0.0, rtol=0.0) -> dict[str, Any]:
    delta = (left["value"] - right["value"]).abs()
    return {
        "shape_equal": left["shape"] == right["shape"],
        "dtype_equal": left["dtype"] == right["dtype"],
        "device_equal": left["device"] == right["device"],
        "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
        "exact": bool(torch.equal(left["value"], right["value"])),
        "allclose": bool(torch.allclose(left["value"], right["value"], atol=atol, rtol=rtol)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/mnt/nas/wlx/Memory/projects/Lingbot_LSM")
    parser.add_argument("--output", default="/mnt/nas/wlx/Memory/outputs/m2ba_a0_20260814")
    parser.add_argument("--fixture-output", default="/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811")
    parser.add_argument("--checkpoint", default="/mnt/h20/135/lingbot-models/lingbot-world-base-act")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-test-exit", type=int, required=True)
    args = parser.parse_args()
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in (repo / "src", repo / "refs" / "lingbot-world", repo):
        sys.path.insert(0, str(path))
    report: dict[str, Any] = {
        "schema_version": "m2ba_a0_parity_v2", "generated_at_utc": utc_now(),
        "status": "BLOCKED_GPU_RUNTIME", "repo": str(repo),
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True).stdout.strip(),
        "scope": "A0_ONLY_NO_TRAINING", "cpu_test_exit": args.cpu_test_exit,
    }
    try:
        if args.cpu_test_exit != 0:
            raise RuntimeError(f"BLOCKED_CPU_TESTS: exit={args.cpu_test_exit}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        from memory_module.causal_memory_adapter import CausalMemoryAdapter, CausalMemoryAdapterConfig, WanCausalMemoryAdapterHooks, expected_trainable_inventory, tensor_module_fingerprint
        from pipeline.eval.stage1_upperbound import runtime_spatial_plan
        from pipeline.v6.latentconcat_infer import _load_raw_pipeline
        from pipeline.v7.phase1.planner import plan_windows, slice_modalities
        from pipeline.v7.phase1.run import _load_case
        from wan.configs import MAX_AREA_CONFIGS

        fixture_probe = load_fixture_probe_module(repo)
        build_conditioning_only = fixture_probe.build_conditioning_only
        direct_forward_conditioning = fixture_probe.direct_forward_conditioning
        state_inventory = fixture_probe.state_inventory
        _frame_to_pil = fixture_probe._frame_to_pil

        frozen = json.loads((Path(args.fixture_output) / "frozen_fixture_manifest.json").read_text())
        runtime = json.loads((Path(args.fixture_output) / "runtime_contract.json").read_text())
        if frozen.get("status") != "FIXTURE_FREEZE_PASS" or runtime.get("status") != "PASS":
            raise RuntimeError("frozen TRAIN fixture/runtime contract is not PASS")
        fixture = next(item for item in frozen["fixtures"] if item["role"] == "TRAIN")
        mapping = fixture["frame_to_token_mapping"]
        device = torch.device(args.device)
        data = _load_case(frozen["cases_root"], fixture["case_id"], int(fixture["total_frames"]))
        plans = plan_windows(int(fixture["total_frames"]), tuple(fixture["support_full_half_open"]), context_frames=81, seam_buffer=8)
        plan = plans[int(fixture["target_window_id"])]
        window = slice_modalities(data, plan)
        generation_args = SimpleNamespace(
            ckpt_dir=args.checkpoint, ft_model_dir=None, ft_high_model_dir=None,
            lora_path=None, lora_rank=0, lora_alpha=0.0, lora_targets="",
            size="480*832", frame_num=81, num_inference_steps=40,
            sample_shift=10.0, guide_scale=5.0, prompt=data["prompt"], seed=42,
        )
        pipeline = _load_raw_pipeline(generation_args, device)
        boundary = float(pipeline.boundary) * float(pipeline.num_train_timesteps)
        model = pipeline._prepare_model_for_timestep(torch.tensor(0.0, device=device), boundary, offload_model=True)
        if model is not pipeline.low_noise_model:
            raise RuntimeError("t=0 did not select low_noise_model")
        model.eval().requires_grad_(False)
        model.freqs = model.freqs.to(device)
        input_h, input_w = map(int, window["rgb"].shape[-2:])
        spatial = runtime_spatial_plan(input_h, input_w, MAX_AREA_CONFIGS[generation_args.size], pipeline.vae_stride, pipeline.patch_size)
        prepared = build_conditioning_only(pipeline, _frame_to_pil(window["rgb"][0]), window, data["prompt"], spatial, device)
        forward_conditioning = direct_forward_conditioning(prepared)
        clean = torch.from_numpy(np.ascontiguousarray(window["rgb"])).float().permute(1, 0, 2, 3)
        if clean.shape[-2:] != (spatial.pixel_h, spatial.pixel_w):
            clean = torch.nn.functional.interpolate(clean.unsqueeze(0), size=(clean.shape[1], spatial.pixel_h, spatial.pixel_w), mode="trilinear", align_corners=False).squeeze(0)
        x0 = pipeline.vae.encode([clean.to(device)])[0]
        target = mapping["target"]
        token_slice = slice(int(target["token_start"]), int(target["token_end"]))

        selected_latents = []
        for frame_index in fixture["memory_selected_set"]:
            frame = torch.from_numpy(np.ascontiguousarray(data["rgb"][frame_index])).float().unsqueeze(1)
            if frame.shape[-2:] != (spatial.pixel_h, spatial.pixel_w):
                frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=(1, spatial.pixel_h, spatial.pixel_w), mode="trilinear", align_corners=False).squeeze(0)
            selected_latents.append(pipeline.vae.encode([frame.to(device)])[0])
        memory_latents = torch.cat(selected_latents, dim=1).unsqueeze(0)
        if tuple(memory_latents.shape) != (1, 16, 3, 58, 104):
            raise RuntimeError(f"memory latent shape drift: {tuple(memory_latents.shape)}")

        route = torch.zeros(1, prepared["max_seq_len"], dtype=torch.bool, device=device)
        window_groups = [
            group for group in mapping["deduplicated_many_to_one_groups"]
            if int(group["window_id"]) == int(plan.window_index)
        ]
        for group in window_groups:
                route[:, int(group["token_start"]):int(group["token_end"])] = True
        expected_nq = len(window_groups) * int(target["token_count"])
        if int(route.sum()) != expected_nq:
            raise RuntimeError(f"route token count drift: {int(route.sum())} != {expected_nq}")

        config = CausalMemoryAdapterConfig()
        base_before = state_inventory(model)
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        adapter = CausalMemoryAdapter.from_wan_self_attention(model.blocks[0].self_attn, config).to(device=device, dtype=torch.bfloat16).eval()
        rng_preserved = torch.equal(cpu_rng_before, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng_before, torch.cuda.get_rng_state_all(), strict=True))
        kvo_clone_pass = all(
            torch.equal(adapter.state_dict()[adapter_name], model.state_dict()[base_name])
            for adapter_name, base_name in (
                ("wk_mem.weight", "blocks.0.self_attn.k.weight"),
                ("wk_mem.bias", "blocks.0.self_attn.k.bias"),
                ("wv_mem.weight", "blocks.0.self_attn.v.weight"),
                ("wv_mem.bias", "blocks.0.self_attn.v.bias"),
                ("wo_mem.weight", "blocks.0.self_attn.o.weight"),
                ("wo_mem.bias", "blocks.0.self_attn.o.bias"),
                ("norm_k_mem.weight", "blocks.0.self_attn.norm_k.weight"),
            )
        )
        q_excluded = not any(
            "base_q" in name or "base_norm_q" in name for name in adapter.state_dict()
        )
        expected_inventory = expected_trainable_inventory(config, dtype="torch.bfloat16")
        actual_inventory = adapter.trainable_inventory()
        inventory_pass = actual_inventory == expected_inventory
        inventory_summary = {
            "expected_tensor_count": len(expected_inventory),
            "actual_tensor_count": len(actual_inventory),
            "expected_numel": sum(item["numel"] for item in expected_inventory),
            "actual_numel": sum(item["numel"] for item in actual_inventory),
            "exact_match": inventory_pass,
        }
        torch.cuda.reset_peak_memory_stats(device)
        t0 = torch.zeros(1, device=device, dtype=torch.float32)

        def direct_forward():
            return model([x0], t=t0, context=forward_conditioning["context"], seq_len=prepared["max_seq_len"], y=forward_conditioning["y"], dit_cond_dict=forward_conditioning["dit_cond_dict"])[0]

        def baseline_forward():
            captures = {}
            def capture_block(_module, _args, out):
                captures["block0"] = tensor_record(out[:, token_slice])

            def capture_head(_module, args_):
                captures["pre_head_original"] = tensor_record(args_[0][:, token_slice])

            handles = [
                model.blocks[0].register_forward_hook(capture_block),
                model.head.register_forward_pre_hook(capture_head),
            ]
            try:
                result = direct_forward()
            finally:
                for handle in handles:
                    handle.remove()
            captures["output"] = tensor_record(result)
            captures["pre_head_fused"] = captures["pre_head_original"]
            return captures

        def wrapped_forward(active_adapter=adapter, **kwargs):
            with WanCausalMemoryAdapterHooks(model, active_adapter, **kwargs) as hooks:
                result = direct_forward()
            if hooks.block0_output is None or hooks.pre_head_input is None or hooks.h_sa0 is None:
                raise RuntimeError("real Wan hook integration did not capture required tensors")
            return {
                "block0": tensor_record(hooks.block0_output[:, token_slice]),
                "pre_head_original": tensor_record(hooks.pre_head_input[:, token_slice]),
                "pre_head_fused": tensor_record(hooks.pre_head_fused[:, token_slice]),
                "query_source": tensor_record(hooks.h_sa0[:, token_slice]),
                "output": tensor_record(result), "diagnostics": hooks.adapter_diagnostics,
            }

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
            baseline = baseline_forward()
            bypass_runs = {
                "disabled": wrapped_forward(memory_latents=None, route_query_mask=None, adapter_enabled=False),
                "empty": wrapped_forward(memory_latents=None, route_query_mask=None),
                "reject": wrapped_forward(memory_latents=torch.ones(1, device=device), route_query_mask=None, rejected=True),
            }
            enabled = wrapped_forward(memory_latents=memory_latents, route_query_mask=route)

        bypass_comparisons = {
            name: {
                field: comparison(baseline[field], value[field])
                for field in ("block0", "pre_head_original", "pre_head_fused", "output")
            }
            for name, value in bypass_runs.items()
        }
        bypass_pass = all(item["exact"] for run in bypass_comparisons.values() for item in run.values())
        tensor_contract_pass = (
            baseline["block0"]["shape"] == [1, 1508, 5120]
            and baseline["pre_head_original"]["shape"] == [1, 1508, 5120]
            and enabled["query_source"]["shape"] == [1, 1508, 5120]
            and baseline["block0"]["dtype"] == "torch.bfloat16"
            and baseline["pre_head_original"]["dtype"] == "torch.float32"
            and all(
                record["device"].startswith("cuda") and record["finite"]
                for collection in (baseline, enabled)
                for key, record in collection.items() if key != "diagnostics"
            )
        )
        diagnostics = enabled["diagnostics"]
        route_only = torch.count_nonzero(diagnostics["fused_delta"][~route]).item() == 0 and torch.count_nonzero(diagnostics["scattered_memory_delta"][~route]).item() == 0
        enabled_finite = all(record["finite"] for key, record in enabled.items() if key != "diagnostics") and all(torch.isfinite(t.float()).all() for t in (diagnostics["memory_tokens"], diagnostics["fused_delta"]))
        adapter_fp = tensor_module_fingerprint(adapter)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            checkpoint = Path(tmp) / "adapter_state.pt"
            torch.save({
                "config": config.canonical_dict(),
                "config_fingerprint": config.fingerprint(),
                "trainable_inventory": actual_inventory,
                "state_dict": adapter.adapter_state_dict(),
            }, checkpoint)
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            restored_config = CausalMemoryAdapterConfig(**payload["config"])
            restored = CausalMemoryAdapter.from_wan_self_attention(model.blocks[0].self_attn, restored_config).to(device=device, dtype=torch.bfloat16).eval()
            restored.load_adapter_state_dict(payload["state_dict"])
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
                restored_disabled = wrapped_forward(
                    active_adapter=restored,
                    memory_latents=None,
                    route_query_mask=None,
                    adapter_enabled=False,
                )
                restored_enabled = wrapped_forward(
                    active_adapter=restored,
                    memory_latents=memory_latents,
                    route_query_mask=route,
                )
            reload_comparison = comparison(enabled["output"], restored_enabled["output"])
            reload_delta_comparison = comparison(
                tensor_record(diagnostics["fused_delta"]),
                tensor_record(restored_enabled["diagnostics"]["fused_delta"]),
            )
            reload_pass = (
                restored_config.fingerprint() == config.fingerprint()
                and payload["config_fingerprint"] == restored_config.fingerprint()
                and payload["trainable_inventory"] == expected_inventory
                and restored.trainable_inventory() == expected_inventory
                and tensor_module_fingerprint(restored) == adapter_fp
                and reload_comparison["exact"]
                and reload_delta_comparison["exact"]
                and comparison(baseline["output"], restored_disabled["output"])["exact"]
                and comparison(baseline["pre_head_fused"], restored_disabled["pre_head_fused"])["exact"]
            )
            restored_disabled_comparison = {
                "output": comparison(baseline["output"], restored_disabled["output"]),
                "pre_head_fused": comparison(
                    baseline["pre_head_fused"], restored_disabled["pre_head_fused"]
                ),
            }
            del restored_disabled, restored_enabled, restored, payload
        base_after = state_inventory(model)
        base_unchanged = base_before["digest"] == base_after["digest"]
        gates = {
            "CPU_TESTS": args.cpu_test_exit == 0, "REAL_WAN_HOOK_CAPTURE": True,
            "REAL_TENSOR_CONTRACT_FINITE": tensor_contract_pass,
            "PHYSICAL_BYPASS_REAL_FORWARD_EXACT": bypass_pass,
            "ENABLED_FINITE": enabled_finite, "ENABLED_ROUTE_ONLY_EXACT_ZERO": route_only,
            "BASE_STATE_EXACT_UNCHANGED": base_unchanged, "CONSTRUCTION_RNG_UNCHANGED": rng_preserved,
            "KVO_NORMK_EXACT_BASE_CLONES": kvo_clone_pass,
            "FROZEN_Q_EXCLUDED_FROM_ADAPTER_STATE": q_excluded,
            "TRAINABLE_INVENTORY_EXACT": inventory_pass,
            "ADAPTER_SAVE_LOAD_RELOAD_STATE_PARITY": reload_pass,
        }
        report.update({
            "status": "A0_GATE_CANDIDATE" if all(gates.values()) else "A0_GATE_NOT_CANDIDATE",
            "gates": gates, "fixture": {"case_id": fixture["case_id"], "event_id": fixture["event_id"], "window_id": plan.window_index, "memory_frames": fixture["memory_selected_set"], "route_query_tokens": int(route.sum())},
            "config": config.canonical_dict(), "config_fingerprint": config.fingerprint(),
            "adapter_state_fingerprint": adapter_fp, "trainable_inventory": adapter.trainable_inventory(),
            "expected_trainable_inventory": expected_inventory,
            "trainable_inventory_summary": inventory_summary,
            "reload_enabled_output_comparison": reload_comparison,
            "reload_enabled_delta_comparison": reload_delta_comparison,
            "reload_disabled_comparison": restored_disabled_comparison,
            "bypass_hook_policy": {
                "production": "needs_integration_hooks=False; do not install hooks for disabled/empty/reject",
                "probe": "A0_INSTRUMENTATION_EXCEPTION: hooks intentionally installed only to capture real parity evidence",
            },
            "baseline": {key: public_record(value) for key, value in baseline.items()},
            "bypass_comparisons": bypass_comparisons,
            "enabled": {key: public_record(value) for key, value in enabled.items() if key != "diagnostics"},
            "enabled_diagnostics": {"physical_bypass": diagnostics["physical_bypass"], "memory_tokens": list(diagnostics["memory_tokens"].shape), "fused_delta": list(diagnostics["fused_delta"].shape), "non_route_nonzero": int(torch.count_nonzero(diagnostics["fused_delta"][~route]))},
            "base_state_before_digest": base_before["digest"], "base_state_after_digest": base_after["digest"],
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "note": "Candidate evidence only; this probe never declares A0 Gate PASS.",
        })
    except Exception as exc:
        status = "BLOCKED_CPU_TESTS" if "BLOCKED_CPU_TESTS" in str(exc) else "BLOCKED_GPU_RUNTIME"
        report.update({"status": status, "error": f"{type(exc).__name__}: {exc}", "note": "No PASS inferred. Review and rerun."})
    dump(output / "a0_parity_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
