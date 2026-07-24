"""Generate official per-frame RAFT anti-freeze evidence on the execution server."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.v7.phase1.manifest import load_manifest  # noqa: E402
from pipeline.v7.phase1.provenance import (  # noqa: E402
    file_digest,
    stable_fingerprint,
    validate_matched_run_invariants,
    validate_run_index_entry,
)
from pipeline.v7.phase1.raft import RAFT_COLUMNS  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 per-frame RAFT anti-freeze evidence")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt_full", required=True)
    parser.add_argument("--runs_index", required=True)
    parser.add_argument("--static_mask", required=True)
    parser.add_argument("--mask_provenance", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--weights_path",
        help="optional local RAFT-Large state_dict; otherwise torchvision DEFAULT cache is used")
    parser.add_argument("--motion_epsilon", type=float, default=0.25)
    return parser.parse_args()


def _load_model(device, weights_path: str | None):
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=None if weights_path else weights, progress=False)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        weights_id = str(Path(weights_path).resolve())
    else:
        weights_id = str(weights)
    model.eval().to(device)
    return model, weights.transforms(), weights_id


def _flow_magnitude(model, transforms, first, second, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    a = torch.from_numpy(first).permute(2, 0, 1)[None].float().to(device) / 255.0
    b = torch.from_numpy(second).permute(2, 0, 1)[None].float().to(device) / 255.0
    height = ((a.shape[-2] + 7) // 8) * 8
    width = ((a.shape[-1] + 7) // 8) * 8
    a = F.interpolate(a, (height, width), mode="bilinear", align_corners=False)
    b = F.interpolate(b, (height, width), mode="bilinear", align_corners=False)
    a, b = transforms(a, b)
    with torch.inference_mode():
        flow = model(a, b)[-1][0]
    return torch.linalg.vector_norm(flow, dim=0).cpu().numpy()


def main() -> None:
    import torch
    from pipeline.eval.oracle_injection import _read_video_back

    args = _args()
    manifest = load_manifest(args.manifest, require_review=True)
    runs = json.loads(Path(args.runs_index).read_text(encoding="utf-8"))
    events = {event["event_id"] for event in manifest["revisit_events"]}
    selected = []
    for run in runs:
        if run["case_id"] != manifest["case_id"]:
            continue
        if run["event_id"] not in events:
            raise SystemExit(f"unknown event in runs index: {run['event_id']}")
        provenance = json.loads(Path(run["provenance"]).read_text(encoding="utf-8"))
        validate_run_index_entry(run, provenance, manifest=manifest)
        selected.append(run)
    validate_matched_run_invariants(selected, require_complete=True)
    gt = _read_video_back(args.gt_full)
    if gt is None or gt.shape[1] != manifest["total_frames"]:
        raise SystemExit("GT_full decoded frame count does not match manifest")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("official RAFT evidence generation requires CUDA")
    model, transforms, weights_id = _load_model(device, args.weights_path)
    manifest_digest = file_digest(args.manifest)
    reference_digest = file_digest(args.gt_full)
    mask_digest = file_digest(args.static_mask)
    mask_provenance_digest = file_digest(args.mask_provenance)
    gt_hwc = np.transpose(gt, (1, 2, 3, 0))
    gt_motion = [0.0]
    for frame in range(1, manifest["total_frames"]):
        gt_motion.append(float(np.median(_flow_magnitude(
            model, transforms, gt_hwc[frame - 1], gt_hwc[frame], device))))
    rows = []
    for run in selected:
        generated = _read_video_back(run["video"])
        if generated is None or generated.shape[1] != manifest["total_frames"]:
            raise SystemExit(f"generated video frame mismatch: {run['video']}")
        gen_hwc = np.transpose(generated, (1, 2, 3, 0))
        gates = [1.0]
        for frame in range(1, manifest["total_frames"]):
            gt_mag = gt_motion[frame]
            gen_mag = float(np.median(_flow_magnitude(
                model, transforms, gen_hwc[frame - 1], gen_hwc[frame], device)))
            gate = 1.0 if gt_mag < args.motion_epsilon else min(1.0, gen_mag / gt_mag)
            gates.append(max(0.0, gate))
        for frame, gate in enumerate(gates):
            rows.append({
                "case_id": run["case_id"],
                "event_id": run["event_id"],
                "seed": run["seed"],
                "arm": run["arm"],
                "frame": frame,
                "raft_gated_anti_freeze": gate,
                "raft_model": "torchvision.models.optical_flow.raft_large",
                "raft_weights": weights_id,
                "metric_version": "phase1_raft_motion_ratio_v1",
                "generated_video": run["video"],
                "run_fingerprint": stable_fingerprint(run),
                "manifest_digest": manifest_digest,
                "input_video_digest": file_digest(run["video"]),
                "reference_digest": reference_digest,
                "mask_digest": mask_digest,
                "mask_provenance_digest": mask_provenance_digest,
                "producer": "pipeline.v7.phase1.raft_cli",
                "producer_version": "1",
            })
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAFT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
