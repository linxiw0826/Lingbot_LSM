#!/usr/bin/env python3
"""Frozen A1 200-step exploratory runner. Never imports or reads fixture B."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verified helper {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("verified helper path mismatch")
    return module


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/mnt/nas/wlx/Memory/projects/Lingbot_LSM")
    parser.add_argument("--output", default="/mnt/nas/wlx/Memory/outputs/m2ba_a1_exploratory_20260814")
    parser.add_argument("--fixture-output", default="/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811")
    parser.add_argument("--train-fixture-snapshot", default="/mnt/nas/wlx/Memory/outputs/m2ba_a1_code_preflight_20260814/train_fixture_snapshot.json")
    parser.add_argument("--checkpoint", default="/mnt/h20/135/lingbot-models/lingbot-world-base-act")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-test-mode", action="store_true")
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    repo, output, fixture_root = Path(args.repo).resolve(), Path(args.output), Path(args.fixture_output)
    output.mkdir(parents=True, exist_ok=True)
    for path in (repo / "src", repo / "refs" / "lingbot-world", repo): sys.path.insert(0, str(path))
    report = {"schema_version": "m2ba_a1_exploratory_v1", "status": "BLOCKED", "scope": "A1_EXPLORATORY_ONLY_A_FIXTURE", "training_started": False}
    try:
        from memory_module.causal_memory_adapter import CausalMemoryAdapter, CausalMemoryAdapterConfig, WanA1MaskedTrainingHooks, expected_trainable_inventory
        from pipeline.m2ba_a1_training import SAMPLER_SCHEMA, A1TrainConfig, A1Trainer, atomic_save_checkpoint, build_checkpoint_payload, compile_train_fixture, load_checkpoint_strict, sha256_file, validate_training_authorization, write_invalid_run_marker
        from pipeline.eval.stage1_upperbound import runtime_spatial_plan
        from pipeline.v6.latentconcat_infer import _load_raw_pipeline
        from pipeline.v7.phase1.planner import plan_windows, slice_modalities
        from pipeline.v7.phase1.run import _load_case
        from wan.configs import MAX_AREA_CONFIGS

        if (output / "INVALID_RUN.json").exists():
            raise RuntimeError("output root is quarantined INVALID_RUN")
        if not args.resume and ((output / "training_log.jsonl").exists() or (output / "checkpoints").exists()):
            raise RuntimeError("fresh run refuses a non-empty prior output root")

        a0 = load_module("_m2ba_a0_verified", repo / "tools/preflight/m2ba_a0/probe_a0.py")
        runtime_helper = a0.load_fixture_probe_module(repo)
        snapshot_path, runtime_path = Path(args.train_fixture_snapshot), fixture_root / "runtime_contract.json"
        if snapshot_path.resolve() != Path("/mnt/nas/wlx/Memory/outputs/m2ba_a1_code_preflight_20260814/train_fixture_snapshot.json"):
            raise RuntimeError("TRAIN-only snapshot path drift")
        if fixture_root != Path("/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811"):
            raise RuntimeError("fixture/runtime evidence root drift")
        frozen, runtime = json.loads(snapshot_path.read_text()), json.loads(runtime_path.read_text())
        if frozen.get("status") != "TRAIN_ONLY_SNAPSHOT" or len(frozen.get("fixtures", [])) != 1:
            raise RuntimeError("training requires the preflight TRAIN-only snapshot")
        fixture = compile_train_fixture(frozen)
        config = A1TrainConfig(
            optimizer_steps=20, checkpoint_steps=(0, 10, 20)
        ) if args.resume_test_mode else A1TrainConfig()
        if repo != Path("/mnt/nas/wlx/Memory/projects/Lingbot_LSM"):
            raise RuntimeError("repository path is not the frozen execution path")
        if Path(frozen["cases_root"]).resolve() != Path("/mnt/h20/135/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected"):
            raise RuntimeError("cases path drift")
        if Path(frozen["checkpoint_root"]).resolve() != Path(args.checkpoint).resolve():
            raise RuntimeError("checkpoint path drift")
        repo_commit = subprocess.run(["git","rev-parse","HEAD"],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
        repo_status = subprocess.run(["git","status","--porcelain"],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
        if repo_status or repo_commit != frozen["a1_expected_repo_commit"]:
            raise RuntimeError("A1 training requires exact reviewed commit and clean tree")
        if frozen.get("a0_gate_evidence_commit") != "0714801a7efa3109b574caf4af20aad1a7538c9e":
            raise RuntimeError("A0 gate evidence commit drift")
        actual_checkpoint_fp = runtime_helper.checkpoint_inventory_fingerprint(Path(args.checkpoint))
        if actual_checkpoint_fp != frozen["checkpoint_fingerprint"]:
            raise RuntimeError("actual checkpoint inventory fingerprint drift")
        runtime_sha = sha256_file(runtime_path)
        snapshot_sha = sha256_file(snapshot_path)
        validate_training_authorization(
            Path(args.authorization), repo_commit=repo_commit,
            config_sha=config.fingerprint(), snapshot_sha=snapshot_sha,
            runtime_sha=runtime_sha, checkpoint_fingerprint=actual_checkpoint_fp,
            authorization="A1_RESUME_EQUIVALENCE_AUTHORIZED" if args.resume_test_mode else "A1_EXPLORATORY_200_AUTHORIZED",
        )
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        device = torch.device(args.device)
        data = _load_case(frozen["cases_root"], fixture.case_id, 405)
        plan = plan_windows(405, fixture.support_half_open, context_frames=81, seam_buffer=8)[5]
        window = slice_modalities(data, plan)
        generation_args = SimpleNamespace(ckpt_dir=args.checkpoint, ft_model_dir=None, ft_high_model_dir=None, lora_path=None, lora_rank=0, lora_alpha=0.0, lora_targets="", size="480*832", frame_num=81, num_inference_steps=40, sample_shift=10.0, guide_scale=5.0, prompt=data["prompt"], seed=42)
        pipeline = _load_raw_pipeline(generation_args, device)
        boundary = float(pipeline.boundary) * float(pipeline.num_train_timesteps)
        model = pipeline._prepare_model_for_timestep(torch.tensor(0.0, device=device), boundary, offload_model=True)
        if model is not pipeline.low_noise_model: raise RuntimeError("t=0 did not select low-noise teacher")
        model.eval().requires_grad_(False); model.freqs = model.freqs.to(device)
        spatial = runtime_spatial_plan(480, 832, MAX_AREA_CONFIGS[generation_args.size], pipeline.vae_stride, pipeline.patch_size)
        prepared = runtime_helper.build_conditioning_only(pipeline, runtime_helper._frame_to_pil(window["rgb"][0]), window, data["prompt"], spatial, device)
        conditions = runtime_helper.direct_forward_conditioning(prepared)
        clean = torch.from_numpy(np.ascontiguousarray(window["rgb"])).float().permute(1, 0, 2, 3)
        clean = F.interpolate(clean.unsqueeze(0), size=(81, spatial.pixel_h, spatial.pixel_w), mode="trilinear", align_corners=False).squeeze(0)
        with torch.no_grad(): x0 = pipeline.vae.encode([clean.to(device)])[0]
        memory_parts = []
        for index in fixture.memory_frames:
            frame = torch.from_numpy(np.ascontiguousarray(data["rgb"][index])).float().unsqueeze(1)
            frame = F.interpolate(frame.unsqueeze(0), size=(1, spatial.pixel_h, spatial.pixel_w), mode="trilinear", align_corners=False).squeeze(0)
            with torch.no_grad(): memory_parts.append(pipeline.vae.encode([frame.to(device)])[0])
        memory_latents = torch.cat(memory_parts, dim=1).unsqueeze(0)
        route = fixture.build_route_mask(length=prepared["max_seq_len"], device=device)
        if int(route.sum()) != 25636: raise RuntimeError("route count drift")
        t0 = torch.zeros(1, device=device, dtype=torch.float32)
        def forward_model():
            return model([x0], t=t0, context=conditions["context"], seq_len=prepared["max_seq_len"], y=conditions["y"], dit_cond_dict=conditions["dit_cond_dict"])[0]

        teacher_capture = {}
        def teacher_hook(_module, args_): teacher_capture["target"] = args_[0][:, 27144:28652].detach().clone()
        handle = model.head.register_forward_pre_hook(teacher_hook)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipeline.param_dtype): forward_model()
        handle.remove(); teacher = teacher_capture["target"]
        if tuple(teacher.shape) != (1, 1508, 5120): raise RuntimeError("teacher target shape drift")

        adapter = CausalMemoryAdapter.from_wan_self_attention(model.blocks[0].self_attn, CausalMemoryAdapterConfig()).to(device=device, dtype=torch.bfloat16).train()
        if adapter.trainable_inventory() != expected_trainable_inventory(adapter.config, dtype="torch.bfloat16"): raise RuntimeError("17-tensor inventory drift")
        def student_forward():
            with torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
                with WanA1MaskedTrainingHooks(model, adapter, memory_latents=memory_latents, route_query_mask=route) as hooks: forward_model()
            if hooks.pre_head_fused is None or hooks.adapter_diagnostics is None: raise RuntimeError("student hook capture missing")
            return hooks.pre_head_fused, hooks.adapter_diagnostics
        def bypass_forward(*, rejected: bool):
            supplied_memory = torch.ones(1, device=device) if rejected else None
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipeline.param_dtype):
                with WanA1MaskedTrainingHooks(
                    model, adapter, memory_latents=supplied_memory,
                    route_query_mask=route, adapter_enabled=not rejected,
                    rejected=rejected,
                ) as hooks: forward_model()
            if (
                hooks.adapter_diagnostics is None
                or hooks.adapter_diagnostics.get("physical_bypass") is not True
                or hooks.pre_head_fused is not hooks.pre_head_input
            ):
                raise RuntimeError("empty/reject path is not a physical exact bypass")
            delta_nonzero = int(torch.count_nonzero(hooks.pre_head_fused - hooks.pre_head_input))
            return hooks.pre_head_fused, delta_nonzero

        def empty_forward():
            return bypass_forward(rejected=False)[0]

        base_fp = lambda: runtime_helper.state_inventory(model)["digest"]
        stop_after = config.optimizer_steps if args.stop_after is None else args.stop_after
        if not (0 <= stop_after <= config.optimizer_steps):
            raise RuntimeError("stop-after outside configured optimizer steps")
        def metrics(diag, record):
            empty_hidden, empty_nonzero = bypass_forward(rejected=False)
            _, reject_nonzero = bypass_forward(rejected=True)
            tokens = diag["memory_tokens"].float().reshape(1, 3, 32, 5120)
            pooled = F.normalize(tokens.mean(2), dim=-1)
            cos = [float((pooled[:, i] * pooled[:, j]).sum(-1).mean()) for i, j in ((0,1),(0,2),(1,2))]
            fused, raw = diag["fused_delta"].float(), torch.cat([x["raw_delta"].float() for x in diag["bridge_records"]], 1)
            bases = torch.cat([x.float() for x in diag["routed_base"]], 1)
            scales = torch.cat([x["cap_scale"].float() for x in diag["bridge_records"]], 1)
            memory_delta = diag["scattered_memory_delta"].float()
            memory_rms = float(memory_delta.pow(2).mean().sqrt())
            fused_rms = float(fused.pow(2).mean().sqrt())
            base_rms = float(bases.pow(2).mean().sqrt())
            finite = bool(torch.isfinite(tokens).all() and torch.isfinite(fused).all() and torch.isfinite(raw).all())
            return {
                "activation_finite": finite,
                "embedding_finite": bool(torch.isfinite(tokens).all()),
                "delta_finite": bool(torch.isfinite(fused).all()),
                "gradient_finite": all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()),
                "embedding_variance": float(tokens.mean(-1).var(unbiased=False)),
                "embedding_pairwise_cosine": cos,
                "frame_pairwise_cosines": cos,
                "memory_delta_rms": memory_rms,
                "fused_delta_rms": fused_rms,
                "base_hidden_rms": base_rms,
                "delta/base_rms_ratio": fused_rms / (base_rms + 1e-6),
                "raw_delta_base_ratio": float(raw.pow(2).mean().sqrt() / (bases.pow(2).mean().sqrt() + 1e-6)),
                "cap_saturation_fraction": float((scales < 1).float().mean()),
                "non_route_nonzero": int(torch.count_nonzero(fused[~route])),
                "empty_delta_nonzero": empty_nonzero,
                "reject_delta_nonzero": reject_nonzero,
                "empty_recon_mse": float(((empty_hidden[:, 27144:28652].float() - teacher.float()) ** 2).mean()),
                "shape_contract_pass": tuple(teacher.shape) == (1, 1508, 5120),
                "nan_inf_count": 0 if finite else 1,
            }
        trainer = A1Trainer(adapter=adapter, base=model, teacher_target=teacher, student_forward=student_forward, empty_forward=empty_forward, config=config, base_fingerprint=base_fp, health_metrics=metrics)
        fixture_sha = frozen["source_fixture_manifest_sha256"]
        if runtime_sha != frozen["runtime_contract_sha256"]:
            raise RuntimeError("runtime contract SHA differs from TRAIN-only snapshot")
        repo_dirty = False
        logs=[]; parent_sha=None; start_step=0
        def save(step):
            nonlocal parent_sha
            payload=build_checkpoint_payload(adapter=adapter,optimizer=trainer.optimizer,completed_optimizer_step=step,config=config,fixture_manifest_sha=fixture_sha,runtime_contract_sha=runtime_sha,base_checkpoint_fingerprint=frozen["checkpoint_fingerprint"],repo_commit=repo_commit,repo_dirty=repo_dirty,sampler_state={"schema_version":SAMPLER_SCHEMA,"kind":"fixed_A","cursor":step*4},health=trainer.health,parent_checkpoint_sha=parent_sha,loss_log_tail=logs)
            path=output/"checkpoints"/f"step_{step:04d}.pt"; parent_sha=atomic_save_checkpoint(path,payload)
        if args.resume:
            resume_path = Path(args.resume)
            loaded = load_checkpoint_strict(
                resume_path, adapter=adapter, optimizer=trainer.optimizer,
                config=config, expected_fixture_sha=fixture_sha,
                expected_runtime_sha=runtime_sha,
                expected_base_fingerprint=frozen["checkpoint_fingerprint"],
                expected_repo_commit=repo_commit, expected_repo_dirty=False,
                health=trainer.health, map_location=device,
            )
            start_step = int(loaded["completed_optimizer_step"])
            if loaded["sampler_state"] != {"schema_version":SAMPLER_SCHEMA,"kind":"fixed_A", "cursor":start_step*4}:
                raise RuntimeError("resume sampler cursor mismatch")
            logs = list(loaded.get("loss_log_tail", []))
            parent_sha = sha256_file(resume_path)
        else:
            save(0)
        report["training_started"]=True
        for step in range(start_step + 1, stop_after + 1):
            started=time.time(); record, diag=trainer.optimizer_step(step); record["step_time_seconds"]=time.time()-started
            if args.resume_test_mode or step % 10 == 0:
                with torch.no_grad():
                    record["correct_recon_mse"]=record["loss_hidden_recon"]
                    record["gpu_memory_allocated_bytes"]=torch.cuda.memory_allocated(device); logs.append(record)
                    with (output/"training_log.jsonl").open("a") as f: f.write(json.dumps(record,sort_keys=True)+"\n")
            if step in config.checkpoint_steps: save(step)
        report.update({"status":"A1_RESUME_TEST_SEGMENT_COMPLETE" if args.resume_test_mode else "A1_EXPLORATORY_COMPLETE_CANDIDATE","completed_optimizer_step":stop_after,"training_started":True,"final_checkpoint_sha256":parent_sha,"base_fingerprint_unchanged":base_fp()==trainer.initial_base_fingerprint,"log_records":len(logs),"note":"Exploratory/resume evidence only; never declares A1 viability or authorizes confirmatory."})
    except Exception as exc:
        if "INVALID_HEALTH_STOP" in str(exc) and "write_invalid_run_marker" in locals():
            attempted = locals().get("step")
            write_invalid_run_marker(
                output, reason=str(exc), attempted_step=attempted,
                last_completed_optimizer_step=(attempted - 1) if isinstance(attempted, int) else locals().get("start_step"),
            )
            report["attempted_step"] = attempted
            report["last_completed_optimizer_step"] = (attempted - 1) if isinstance(attempted, int) else locals().get("start_step")
        report.update({"status":"INVALID_HEALTH_STOP" if "INVALID_HEALTH_STOP" in str(exc) else "BLOCKED_A1_EXPLORATORY","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()})
    atomic_json(output/"exploratory_report.json",report); print(json.dumps(report,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
