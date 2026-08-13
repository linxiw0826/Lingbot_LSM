# M2-B-A A0/A1 fixture-freeze probe

Standalone, fail-closed evidence collector. It does not modify Lingbot_LSM,
datasets, manifests, checkpoints, or prior Phase 1 runs. It hashes authoritative
small inputs, inventories checkpoint shards, resolves three fixtures, replays the
Phase 1 planner contract, and records provenance candidates.

Manifest resolution follows this priority: repeated `--manifest-dir` arguments,
NAS pilot manifests, workspace pilot manifests, then repository dev manifests.
Multiple paths with identical SHA256 are recorded as aliases of one logical
manifest; differing contents remain a fail-closed ambiguity. Repository
`src/pipeline/v7/phase1/manifests` null/TODO templates are never candidates.
Positive TRAIN/DEV oracle fixtures require one selected memory frame. The
EMPTY/REJECT safety fixture records its ordered manifest candidates but consumes
no selected memory.

Run from the repository on the execution server:

```bash
bash /mnt/nas/wlx/Memory/projects/Lingbot_LSM/tools/preflight/m2ba_a01_fixture_freeze/run_probe.sh
```

The current version intentionally does **not** load Wan or sample a video. The
repository exposes a generation pipeline, but static inspection alone cannot
prove a minimal clean-`t=0` teacher forward, pre-head runtime shape, determinism,
or exact full-frame-to-token mapping. It therefore reports
`BLOCKED_MINIMAL_FORWARD_API_MISSING` after a successful static gate rather than
inventing runtime evidence. A separately reviewed runtime hook probe is required.
