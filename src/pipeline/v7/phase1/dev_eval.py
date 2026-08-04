"""CPU-first, one-case Phase 1 development diagnostics.

This is deliberately separate from the formal five-case/four-arm evaluator.  It
never fabricates static masks, never downloads DINO weights, and never emits a
formal Phase 1 verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from pipeline.v7.phase1.manifest import load_manifest


ARMS = ("off", "global", "correct_local")
MARKERS = (
    "DEV_ONLY", "ONE_CASE", "ONE_SEED", "THREE_ARMS", "NO_STATIC_MASK",
    "NO_WRONG_LOCAL", "NOT_PHASE1_GO", "NOT_PAPER_EVIDENCE",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cases-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--evaluator-commit-sha", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dino-device", default="cpu")
    return parser


def _read_video(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames: {path}")
    return np.stack(frames), fps


def center_crop_to(frame_array: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, dict[str, int]]:
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(f"expected [F,H,W,3], got {frame_array.shape}")
    height, width = frame_array.shape[1:3]
    if target_h > height or target_w > width:
        raise ValueError(f"cannot crop {(height, width)} to {(target_h, target_w)}")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    crop = frame_array[:, top:top + target_h, left:left + target_w]
    return crop, {
        "top": top, "bottom": height - target_h - top,
        "left": left, "right": width - target_w - left,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(kept)) if kept else None


def _regions(frame: int, support: tuple[int, int], first_visit: tuple[int, int]) -> tuple[str, bool]:
    region = "support" if support[0] <= frame < support[1] else "non_support"
    return region, first_visit[0] <= frame < first_visit[1]


def _seam_boundaries(provenance: dict[str, Any]) -> list[int]:
    boundaries = set()
    for window in provenance["windows"]:
        boundary = int(window["owned_end"])
        if 0 < boundary < int(provenance["actual_output_frames"]):
            boundaries.add(boundary)
    return sorted(boundaries)


def _dino_cache_ready() -> tuple[bool, dict[str, Any]]:
    try:
        import torch
        hub = Path(torch.hub.get_dir())
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, {"status": "BLOCKED_DINO", "reason": f"torch unavailable: {exc}"}
    repo = hub / "facebookresearch_dinov2_main"
    candidates = sorted((hub / "checkpoints").glob("*dinov2*vits14*.pth"))
    if not repo.is_dir() or not candidates:
        return False, {
            "status": "BLOCKED_DINO",
            "reason": "offline DINO repository or dinov2_vits14 checkpoint missing; network disabled by policy",
            "hub_dir": str(hub), "repo_exists": repo.is_dir(),
            "checkpoint_candidates": [str(path) for path in candidates],
        }
    return True, {
        "status": "READY", "hub_dir": str(hub), "repo": str(repo),
        "checkpoint_candidates": [str(path) for path in candidates],
    }


def _dino_scores(gt: np.ndarray, generated: dict[str, np.ndarray], device_name: str) -> tuple[dict[str, list[float | None]], dict[str, Any]]:
    ready, status = _dino_cache_ready()
    scores = {arm: [None] * len(gt) for arm in ARMS}
    if not ready:
        return scores, status
    import torch
    from pipeline.eval.oracle_injection import _dino_feat
    device = torch.device(device_name)
    try:
        for frame in range(len(gt)):
            gt_chw = np.transpose(gt[frame].astype(np.float32) / 127.5 - 1.0, (2, 0, 1))
            gt_feat = _dino_feat(gt_chw, device)
            if gt_feat is None:
                raise RuntimeError(f"DINO returned None for GT frame {frame}")
            for arm in ARMS:
                gen_chw = np.transpose(generated[arm][frame].astype(np.float32) / 127.5 - 1.0, (2, 0, 1))
                feature = _dino_feat(gen_chw, device)
                if feature is None:
                    raise RuntimeError(f"DINO returned None for {arm} frame {frame}")
                scores[arm][frame] = float(torch.nn.functional.cosine_similarity(
                    gt_feat[None].float(), feature[None].float()).item())
    except Exception as exc:  # do not substitute another semantic metric
        return {arm: [None] * len(gt) for arm in ARMS}, {
            **status, "status": "BLOCKED_DINO", "reason": f"offline DINO execution failed: {exc}",
        }
    return scores, {**status, "status": "PASS", "device": str(device)}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    if args.repo_root.resolve() != Path.cwd().resolve():
        os.chdir(args.repo_root)
    manifest = load_manifest(args.manifest, require_review=True)
    if manifest["case_id"] != args.case_id:
        raise SystemExit("manifest/case mismatch")
    event = next((item for item in manifest["revisit_events"] if item["event_id"] == args.event_id), None)
    if event is None:
        raise SystemExit("event absent from manifest")
    support = (int(event["query_start"]), int(event["query_end"]))
    first_visit = (int(manifest["first_visit"]["start"]), int(manifest["first_visit"]["end"]))
    total = int(manifest["total_frames"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {}
    videos: dict[str, np.ndarray] = {}
    video_meta: dict[str, Any] = {}
    for arm in ARMS:
        run_dir = (args.input_root / "runs" / "phase1" / args.commit_sha / arm /
                   args.case_id / args.event_id / f"seed_{args.seed}")
        provenance_path = run_dir / "provenance.json"
        video_path = run_dir / "long_video.mp4"
        index_path = run_dir / "run_index_entry.json"
        for path in (provenance_path, video_path, index_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"missing run artifact: {path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (provenance["arm"], provenance["case_id"], provenance["event_id"],
                int(provenance["seed"]), provenance["commit_sha"],
                int(provenance["actual_output_frames"])) != (
                arm, args.case_id, args.event_id, args.seed, args.commit_sha, total):
            raise SystemExit(f"provenance identity mismatch: {provenance_path}")
        if int(provenance["config"]["num_inference_steps"]) != 40:
            raise SystemExit(f"not a 40-step run: {provenance_path}")
        video, fps = _read_video(video_path)
        if len(video) != total:
            raise SystemExit(f"decoded frame mismatch: {video_path}: {len(video)}")
        runs[arm] = provenance
        videos[arm] = video
        video_meta[arm] = {
            "path": str(video_path), "sha256": _sha256(video_path), "frames": len(video),
            "fps": fps, "height": int(video.shape[1]), "width": int(video.shape[2]),
        }
    shapes = {tuple(video.shape) for video in videos.values()}
    if len(shapes) != 1:
        raise SystemExit(f"generated arm shapes differ: {shapes}")

    gt_path = args.cases_root / args.case_id / "ground_truth_full.mp4"
    gt_raw, gt_fps = _read_video(gt_path)
    target_h, target_w = next(iter(videos.values())).shape[1:3]
    gt, crop = center_crop_to(gt_raw, target_h, target_w)
    if len(gt) != total:
        raise SystemExit(f"GT frame mismatch: {len(gt)} != {total}")
    alignment = {
        "policy": "deterministic_center_crop_gt_only_no_generated_resize",
        "gt_path": str(gt_path), "gt_original": list(gt_raw.shape), "gt_fps": gt_fps,
        "generated_shape": list(next(iter(videos.values())).shape), "crop": crop,
        "gt_aligned": list(gt.shape), "shared_for_arms": list(ARMS),
    }
    _json_dump(args.output_dir / "spatial_alignment.json", alignment)

    dino, dino_status = _dino_scores(gt, videos, args.dino_device)
    seam_sets = {arm: set(_seam_boundaries(runs[arm])) for arm in ARMS}
    if len({tuple(sorted(value)) for value in seam_sets.values()}) != 1:
        raise SystemExit("planner seam boundaries differ across arms")
    seams = seam_sets["off"]
    rows: list[dict[str, Any]] = []
    for frame in range(total):
        region, in_first = _regions(frame, support, first_visit)
        row: dict[str, Any] = {"frame": frame, "region": region, "first_visit": int(in_first), "is_seam_transition": int(frame in seams)}
        gt_float = gt[frame].astype(np.float32) / 255.0
        for arm in ARMS:
            gen_float = videos[arm][frame].astype(np.float32) / 255.0
            row[f"{arm}_dino_gt"] = dino[arm][frame]
            row[f"{arm}_l1_gt"] = float(np.mean(np.abs(gen_float - gt_float)))
            row[f"{arm}_motion_l1"] = (None if frame == 0 else float(np.mean(np.abs(
                gen_float - videos[arm][frame - 1].astype(np.float32) / 255.0))))
        for left, right in (("correct_local", "off"), ("correct_local", "global"), ("global", "off")):
            row[f"l1_{left}_vs_{right}"] = float(np.mean(np.abs(
                videos[left][frame].astype(np.float32) / 255.0 -
                videos[right][frame].astype(np.float32) / 255.0)))
        rows.append(row)
    fieldnames = list(rows[0])
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def subset(name: str) -> list[dict[str, Any]]:
        if name == "whole_video": return rows
        if name == "first_visit": return [row for row in rows if row["first_visit"]]
        return [row for row in rows if row["region"] == name]

    regions: dict[str, Any] = {}
    for name in ("support", "non_support", "first_visit", "whole_video"):
        selected = subset(name)
        regions[name] = {"frames": len(selected), "arms": {}}
        for arm in ARMS:
            motions = [row[f"{arm}_motion_l1"] for row in selected if row["frame"] > 0]
            normal = [row[f"{arm}_motion_l1"] for row in rows if row["frame"] > 0 and row["frame"] not in seams]
            seam_values = [rows[frame][f"{arm}_motion_l1"] for frame in sorted(seams)]
            normal_mean, seam_mean = _mean(normal), _mean(seam_values)
            regions[name]["arms"][arm] = {
                "dino_gt_mean": _mean(row[f"{arm}_dino_gt"] for row in selected),
                "l1_gt_mean": _mean(row[f"{arm}_l1_gt"] for row in selected),
                "motion_l1_mean": _mean(motions),
                "near_frozen_fraction_motion_l1_lt_1e-4": (float(np.mean([
                    float(value < 1e-4) for value in motions if value is not None])) if motions else None),
                "seam_motion_l1_mean": seam_mean,
                "ordinary_motion_l1_mean": normal_mean,
                "seam_to_ordinary_ratio": (seam_mean / normal_mean if seam_mean is not None and normal_mean else None),
            }
        regions[name]["pairwise_output_l1"] = {
            pair: _mean(row[pair] for row in selected)
            for pair in ("l1_correct_local_vs_off", "l1_correct_local_vs_global", "l1_global_vs_off")
        }

    def delta(region: str, left: str, right: str) -> float | None:
        a = regions[region]["arms"][left]["dino_gt_mean"]
        b = regions[region]["arms"][right]["dino_gt_mean"]
        return None if a is None or b is None else float(a - b)

    deltas = {region: {
        "correct_local_minus_off": delta(region, "correct_local", "off"),
        "correct_local_minus_global": delta(region, "correct_local", "global"),
        "global_minus_off": delta(region, "global", "off"),
    } for region in ("support", "non_support")}
    nonzero = all(regions["whole_video"]["pairwise_output_l1"][name] > 0 for name in (
        "l1_correct_local_vs_off", "l1_correct_local_vs_global", "l1_global_vs_off"))
    if dino_status["status"] != "PASS":
        verdict = "INCONCLUSIVE"
    else:
        comparisons = [deltas["support"]["correct_local_minus_off"],
                       deltas["support"]["correct_local_minus_global"],
                       deltas["non_support"]["correct_local_minus_global"]]
        if all(value is not None and value > 0 for value in comparisons): verdict = "DEV_POSITIVE"
        elif comparisons[0] is not None and comparisons[0] <= 0 and comparisons[1] is not None and comparisons[1] <= 0: verdict = "DEV_NEGATIVE"
        else: verdict = "DEV_MIXED"
    summary = {
        "markers": list(MARKERS), "verdict": verdict, "dino": dino_status,
        "identity": {"run_commit_sha": args.commit_sha,
                     "evaluator_commit_sha": args.evaluator_commit_sha,
                     "case_id": args.case_id,
                     "event_id": args.event_id, "seed": args.seed, "arms": list(ARMS)},
        "intervals_half_open": {"support": list(support), "first_visit": list(first_visit),
                                "non_support": [[0, support[0]], [support[1], total]]},
        "spatial_alignment": alignment, "videos": video_meta, "seam_boundaries": sorted(seams),
        "regions": regions, "paired_dino_deltas": deltas,
        "three_arms_nonzero_output_difference": nonzero,
    }
    _json_dump(args.output_dir / "summary.json", summary)
    _json_dump(args.output_dir / "provenance_snapshot.json", {arm: runs[arm] for arm in ARMS})
    lines = ["# Phase 1 one-case three-arm dev quantitative diagnostic", "",
             "> " + " · ".join(MARKERS), "", f"**Verdict: `{verdict}`**", "",
             f"- Identity: `{args.case_id}` / `{args.event_id}` / seed `{args.seed}` / run commit `{args.commit_sha}` / evaluator commit `{args.evaluator_commit_sha}`",
             f"- Intervals (half-open): support `{support}`, first_visit `{first_visit}`",
             f"- Alignment: GT {tuple(gt_raw.shape[1:3])} center-cropped by {crop} to {(target_h, target_w)}; generated not resized",
             f"- DINO: `{dino_status['status']}` — {dino_status.get('reason', 'offline cache used')}", "",
             "## Region summary", "", "| region | arm | DINO↑ | L1-to-GT↓ | motion L1 | near-frozen | seam ratio |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value: Any) -> str: return "NA" if value is None else f"{value:.6f}"
    for region in ("support", "non_support", "first_visit", "whole_video"):
        for arm in ARMS:
            value = regions[region]["arms"][arm]
            lines.append(f"| {region} | {arm} | {fmt(value['dino_gt_mean'])} | {fmt(value['l1_gt_mean'])} | {fmt(value['motion_l1_mean'])} | {fmt(value['near_frozen_fraction_motion_l1_lt_1e-4'])} | {fmt(value['seam_to_ordinary_ratio'])} |")
    lines += ["", "## Paired DINO deltas", "", "| region | correct_local-off | correct_local-global | global-off |", "|---|---:|---:|---:|"]
    for region in ("support", "non_support"):
        value = deltas[region]
        lines.append(f"| {region} | {fmt(value['correct_local_minus_off'])} | {fmt(value['correct_local_minus_global'])} | {fmt(value['global_minus_off'])} |")
    lines += ["", "## Required answers", "",
              f"- support correct_local > off: `{deltas['support']['correct_local_minus_off'] is not None and deltas['support']['correct_local_minus_off'] > 0}`",
              f"- support correct_local > global: `{deltas['support']['correct_local_minus_global'] is not None and deltas['support']['correct_local_minus_global'] > 0}`",
              f"- non-support correct_local > global: `{deltas['non_support']['correct_local_minus_global'] is not None and deltas['non_support']['correct_local_minus_global'] > 0}`",
              f"- all three pairwise output differences non-zero: `{nonzero}`",
              "- Freeze/seam values above are lightweight pixel diagnostics, not official RAFT guardrails.",
              "- This one-case/one-seed result cannot establish generalization or Phase 1 GO.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    files = sorted(str(path.resolve()) for path in args.output_dir.iterdir() if path.is_file())
    (args.output_dir / "generated_files.txt").write_text("\n".join(files + [str((args.output_dir / 'generated_files.txt').resolve())]) + "\n", encoding="utf-8")
    print(f"DINO_STATUS={dino_status['status']}")
    print(f"DEV_VERDICT={verdict}")
    print(f"SUMMARY={args.output_dir / 'summary.md'}")
    print("PHASE1_DEV_EVAL=PASS")


if __name__ == "__main__":
    main()
