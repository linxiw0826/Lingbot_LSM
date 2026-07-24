"""Execution-server entry for the Phase 1 four-arm subclip oracle.

This module deliberately imports heavy Wan/v6 components only after `run` is
selected. Geometry validation and planning remain CPU/model-free.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.v7.phase1.jobs import (  # noqa: E402
    PRIMARY_ARMS,
    anchor_enabled,
    build_matched_jobs,
)
from pipeline.v7.phase1.guardrails import load_guardrail_config  # noqa: E402
from pipeline.v7.phase1.manifest import load_manifest, manifest_seeds  # noqa: E402
from pipeline.v7.phase1.planner import slice_modalities, stitch_owned  # noqa: E402
from pipeline.v7.phase1.provenance import (  # noqa: E402
    build_invariant_fingerprints,
    stable_fingerprint,
    write_provenance,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V7 Phase 1 time-local oracle")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", action="append", required=True)
    validate.add_argument("--allow_todo", action="store_true")

    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument(
        "--seed", type=int,
        help="optional assertion; must be preregistered in the manifest")
    plan.add_argument("--seam_buffer", type=int, default=8)
    plan.add_argument("--output", required=True)

    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--cases_root", required=True)
    run.add_argument("--case_id", required=True)
    run.add_argument("--event_id", required=True)
    run.add_argument("--arm", choices=PRIMARY_ARMS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output_dir", required=True)
    run.add_argument("--ckpt_dir", required=True)
    run.add_argument("--lora_path")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--size", default="480*832")
    run.add_argument("--frame_num", type=int, default=81)
    run.add_argument("--seam_buffer", type=int, default=8)
    run.add_argument("--num_inference_steps", type=int, default=40)
    run.add_argument("--sample_shift", type=float, default=10.0)
    run.add_argument("--guide_scale", type=float, default=5.0)
    run.add_argument("--fps", type=int, default=16)
    run.add_argument("--prompt")
    run.add_argument("--commit_sha")
    run.add_argument(
        "--guardrail_config", required=True,
        help="preregistered threshold-only config fingerprinted into every matched run")
    return parser


def _git_sha(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_SRC.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot determine commit SHA; pass --commit_sha: {exc}") from exc


def _case_dir(root: str, case_id: str) -> Path:
    path = Path(root) / case_id
    if not path.is_dir():
        raise SystemExit(f"case directory missing: {path}")
    return path


def _load_case(root: str, case_id: str, total_frames: int) -> Dict[str, Any]:
    from PIL import Image
    from pipeline.eval.oracle_injection import _read_video_back

    case = _case_dir(root, case_id)
    arrays = {}
    for key, filename in (
        ("pose", "poses.npy"),
        ("action", "action.npy"),
        ("intrinsics", "intrinsics.npy"),
    ):
        path = case / filename
        if not path.is_file():
            raise SystemExit(f"required modality missing: {path}")
        arrays[key] = np.load(path).astype(np.float32)
        if arrays[key].shape[0] != total_frames:
            raise SystemExit(
                f"{case_id}/{filename}: expected {total_frames} frames, got {arrays[key].shape[0]}")
    gt_path = case / "ground_truth_full.mp4"
    if not gt_path.is_file():
        raise SystemExit(f"required GT_full missing: {gt_path}")
    gt = _read_video_back(str(gt_path))
    if gt is None or gt.shape[1] != total_frames:
        raise SystemExit(f"{gt_path}: expected {total_frames} decoded frames")
    arrays["rgb"] = np.transpose(gt, (1, 0, 2, 3)).astype(np.float32)
    first_path = case / "image.jpg"
    if not first_path.is_file():
        first_path = case / "image.png"
    if not first_path.is_file():
        raise SystemExit(f"first-frame image missing in {case}")
    first_image = Image.open(first_path).convert("RGB")
    prompt_path = case / "prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.is_file() else None
    return {**arrays, "first_image": first_image, "prompt": prompt, "case_dir": case}


def _encode_anchor(vae, rgb: np.ndarray, frame_indices: List[int], size: str, device):
    import torch
    from pipeline.eval.stage1_upperbound import _encode_anchor_latent

    height, width = (int(x) for x in size.split("*"))
    pieces = [
        _encode_anchor_latent(vae, rgb[index], height, width, device)
        for index in frame_indices
    ]
    return torch.cat(pieces, dim=1)


def tokens_per_anchor_frame(anchor_latent: Any, pipeline: Any) -> int:
    """Return spatial DiT tokens per latent anchor frame from runtime geometry."""
    shape = getattr(anchor_latent, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(f"anchor latent must be [C,T,H,W], got {shape}")
    patch_size = getattr(getattr(pipeline, "model", None), "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(pipeline, "model", None), "patch", None)
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    elif isinstance(patch_size, (tuple, list)) and len(patch_size) >= 2:
        patch_h, patch_w = int(patch_size[-2]), int(patch_size[-1])
    else:
        raise ValueError(f"cannot determine pipeline spatial patch_size: {patch_size!r}")
    latent_h, latent_w = int(shape[-2]), int(shape[-1])
    if patch_h <= 0 or patch_w <= 0 or latent_h % patch_h or latent_w % patch_w:
        raise ValueError(
            f"latent {(latent_h, latent_w)} is not divisible by patch {(patch_h, patch_w)}")
    return (latent_h // patch_h) * (latent_w // patch_w)


def _run(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from pipeline.eval.oracle_injection import _save_video
    from pipeline.eval.stage1_upperbound import _generate_with_anchor
    from pipeline.v6.latentconcat_infer import _load_lora_into_backbone, _load_raw_pipeline

    if args.frame_num != 81:
        raise SystemExit("Phase 1 frozen planner requires --frame_num 81")
    manifest = load_manifest(args.manifest, require_review=True)
    guardrail_config = load_guardrail_config(args.guardrail_config)
    registered_seeds = manifest_seeds(manifest, min_count=3)
    if args.seed not in registered_seeds:
        raise SystemExit(
            f"--seed={args.seed} is not preregistered; manifest seeds={registered_seeds}")
    if manifest["case_id"] != args.case_id:
        raise SystemExit("--case_id does not match manifest")
    jobs = [
        job for job in build_matched_jobs(
            manifest, registered_seeds, context_frames=args.frame_num,
            seam_buffer=args.seam_buffer)
        if (
            job.event_id == args.event_id
            and job.arm == args.arm
            and job.seed == args.seed
        )
    ]
    if len(jobs) != 1:
        raise SystemExit(f"expected one job, got {len(jobs)}")
    job = jobs[0]
    if args.arm == "wrong_local" and not job.wrong_match_verified:
        raise SystemExit("wrong_local unavailable or not human-verified; mechanism Gate is INCONCLUSIVE")
    data = _load_case(args.cases_root, args.case_id, manifest["total_frames"])
    source_data = data
    if job.anchor_source_case and job.anchor_source_case != args.case_id:
        source_data = _load_case(
            args.cases_root, job.anchor_source_case, manifest["total_frames"])
    prompt = args.prompt or data["prompt"]
    if not prompt:
        raise SystemExit("prompt missing; provide case prompt.txt or --prompt")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal Phase 1 generation requires an available CUDA device")
    generation_args = SimpleNamespace(
        ckpt_dir=args.ckpt_dir,
        ft_model_dir=None,
        ft_high_model_dir=None,
        lora_path=args.lora_path,
        lora_rank=0,
        lora_alpha=0.0,
        lora_targets="",
        size=args.size,
        frame_num=args.frame_num,
        num_inference_steps=args.num_inference_steps,
        sample_shift=args.sample_shift,
        guide_scale=args.guide_scale,
        prompt=prompt,
        seed=args.seed,
    )
    pipeline = _load_raw_pipeline(generation_args, device)
    if args.lora_path:
        _load_lora_into_backbone(pipeline, args.lora_path, generation_args, device)

    event_spec = next(
        event for event in manifest["revisit_events"] if event["event_id"] == job.event_id
    )
    budget_indices = [int(value) for value in event_spec["memory_frame_indices"]]
    budget_anchor = _encode_anchor(
        pipeline.vae, data["rgb"], budget_indices, args.size, device)
    correct_anchor = None
    anchor_poses = None
    if job.anchor_frame_indices:
        if job.anchor_source_case == args.case_id and list(job.anchor_frame_indices) == budget_indices:
            correct_anchor = budget_anchor
        else:
            correct_anchor = _encode_anchor(
                pipeline.vae, source_data["rgb"], list(job.anchor_frame_indices), args.size, device)
        anchor_poses = source_data["pose"][list(job.anchor_frame_indices)]
    generated_cache: Dict[int, np.ndarray] = {}
    window_outputs = []
    cumulative_uses = 0
    for plan in job.windows:
        window = slice_modalities(data, plan)
        first_source = plan.source_frame_index[0]
        if plan.window_index == 0:
            query_first = data["first_image"]
        elif first_source in generated_cache:
            frame = generated_cache[first_source]
            hwc = (frame.transpose(1, 2, 0) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            query_first = Image.fromarray(hwc)
        else:
            raise RuntimeError(
                f"planner chaining failure: source frame {first_source} has no generated predecessor")
        enabled = anchor_enabled(job, plan)
        local_args = SimpleNamespace(**vars(generation_args))
        local_args.seed = args.seed + plan.window_index
        video = _generate_with_anchor(
            pipeline,
            query_first,
            correct_anchor if enabled else None,
            anchor_poses if enabled else None,
            window["pose"],
            window["action"],
            window["intrinsics"],
            local_args,
            device,
        )
        if video is None:
            raise RuntimeError(f"generation returned None for window {plan.window_index}")
        frame_first = np.transpose(np.asarray(video), (1, 0, 2, 3))
        window_outputs.append(frame_first)
        for local, source in enumerate(plan.source_frame_index):
            if not plan.is_pad[local]:
                generated_cache[source] = frame_first[local]
        if enabled:
            cumulative_uses += len(job.anchor_frame_indices) * len(plan.source_frame_index)

    stitched = stitch_owned(job.windows, window_outputs, manifest["total_frames"])
    chw = np.transpose(stitched, (1, 0, 2, 3))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "long_video.mp4"
    _save_video(chw, str(video_path), fps=args.fps)
    sha = _git_sha(args.commit_sha)
    tokens_per_anchor = tokens_per_anchor_frame(budget_anchor, pipeline)
    cumulative_token_frames = cumulative_uses * tokens_per_anchor
    invariant_evidence = {
        "commit_sha": sha,
        "checkpoint": {"base": args.ckpt_dir, "lora": args.lora_path},
        "config": vars(generation_args),
        "backend": "event_centered_subclip_81",
        "query_support": list(job.support),
        "planner_windows": [plan.to_dict() for plan in job.windows],
        "source_frame_mapping": [
            {
                "source_frame_index": list(plan.source_frame_index),
                "is_pad": list(plan.is_pad),
                "owned_output": [plan.owned_start, plan.owned_end],
            }
            for plan in job.windows
        ],
        "prompt": stable_fingerprint(prompt),
        "trajectory": {
            "pose": stable_fingerprint(np.asarray(data["pose"]).tolist()),
            "action": stable_fingerprint(np.asarray(data["action"]).tolist()),
            "intrinsics": stable_fingerprint(np.asarray(data["intrinsics"]).tolist()),
        },
        "planned_anchor_budget": {
            "anchor_frames": len(budget_indices),
            "tokens_per_anchor_frame": tokens_per_anchor,
        },
        "actual_output_frames": int(stitched.shape[0]),
        "guardrail_config": guardrail_config,
    }
    invariant_fingerprints = build_invariant_fingerprints(invariant_evidence)
    provenance = {
        "phase": "phase1",
        "arm": job.arm,
        "case_id": job.case_id,
        "event_id": job.event_id,
        "query_support": list(job.support),
        "anchor_source_case": job.anchor_source_case,
        "anchor_frame_indices": list(job.anchor_frame_indices),
        "seed": job.seed,
        "commit_sha": sha,
        "checkpoint": {"base": args.ckpt_dir, "lora": args.lora_path},
        "config": vars(generation_args),
        "video": str(video_path),
        "backend": "event_centered_subclip_81",
        "windows": [plan.to_dict() for plan in job.windows],
        "actual_output_frames": int(stitched.shape[0]),
        "peak_memory_slots": len(job.anchor_frame_indices),
        "peak_memory_tokens": len(job.anchor_frame_indices) * tokens_per_anchor,
        "tokens_per_anchor_frame": tokens_per_anchor,
        # Unit: memory tokens exposed per model query frame, summed over windows.
        "cumulative_memory_exposure_token_frames": cumulative_token_frames,
        "cumulative_anchor_frame_uses": cumulative_uses,
        "failure_reason": None,
        "stitch": {"blend": "none", "ownership": "unique", "seam_buffer": args.seam_buffer},
        "invariant_evidence": invariant_evidence,
        "invariant_fingerprints": invariant_fingerprints,
    }
    write_provenance(output / "provenance.json", provenance)
    with (output / "run_index_entry.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "case_id": job.case_id,
            "event_id": job.event_id,
            "seed": job.seed,
            "arm": job.arm,
            "video": str(video_path),
            "provenance": str(output / "provenance.json"),
            "commit_sha": sha,
            "checkpoint": provenance["checkpoint"],
            "config": provenance["config"],
            "actual_output_frames": provenance["actual_output_frames"],
            "invariant_fingerprints": invariant_fingerprints,
        }, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Phase 1 job complete: %s", output)


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "validate":
            for path in args.manifest:
                value = load_manifest(path, require_review=not args.allow_todo)
                LOGGER.info(
                    "valid manifest: case=%s events=%d",
                    value["case_id"], len(value["revisit_events"]))
        elif args.command == "plan":
            manifest = load_manifest(args.manifest, require_review=True)
            seeds = manifest_seeds(manifest, min_count=3)
            if args.seed is not None and args.seed not in seeds:
                raise ValueError(
                    f"--seed={args.seed} is not preregistered; manifest seeds={seeds}")
            jobs = build_matched_jobs(manifest, seam_buffer=args.seam_buffer)
            value = [
                {
                    "case_id": job.case_id,
                    "event_id": job.event_id,
                    "arm": job.arm,
                    "seed": job.seed,
                    "windows": [window.to_dict() for window in job.windows],
                }
                for job in jobs
            ]
            Path(args.output).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        else:
            _run(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Phase 1 validation/run failed: {exc}") from exc


if __name__ == "__main__":
    main()
