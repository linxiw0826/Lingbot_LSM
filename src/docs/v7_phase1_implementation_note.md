# V7 Phase 1 implementation note

Status: implementation-ready on the code server; GPU generation and primary
evaluation have not been run here.

Component acceptance uses `pytest -q src/tests/test_v7_phase1.py` from the
repository root (no `PYTHONPATH` required), plus `pytest -q src/tests` where the
existing environment permits. A bare `pytest` is not this component's
acceptance command because pytest recursively discovers the user's unrelated,
untracked reference repositories under `refs/`; Phase 1 neither modifies nor
suppresses collection from those user-owned trees.

## Audit answers

### A. Why the V6 anchor affects the full 81-frame query

`pipeline/eval/stage1_upperbound.py::_generate_with_anchor` appends clean anchor
latents to the query latent time dimension. At 480×832, the query conditioning
has shape `[20,21,60,104]` for 81 pixel frames and one anchor adds one latent
slot. The corresponding anchor pose is appended to the Plücker conditioning.
The Wan DiT flattens spatiotemporal patches for self-attention; that attention
is not query-time causal. Consequently every query token can attend to the
appended anchor tokens. The VAE is causal, but only decodes the final latent;
the implementation removes the four anchor-derived tail pixels and asserts the
returned query still has exactly 81 frames.

### B. Arbitrary query-time × memory mask audit

The call chain is:

1. `v6/latentconcat_infer.py::_rollout_long_video`
2. `eval/stage1_upperbound.py::_generate_with_anchor`
3. `model(latent_model_input, context, seq_len, y, dit_cond_dict)`
4. Wan block self-attention over the complete flattened query+anchor sequence.

The V6 call passes no query×anchor visibility tensor. `y` has
`[20,T_latent,H_latent,W_latent]`; `c2ws_plucker_emb` is tuple-wrapped with a
matching latent-time dimension. Neither `_generate_with_anchor` nor the Wan
forward arguments expose an arbitrary `[query_token,memory_token]` mask.
Phase 1 therefore does not rewrite the attention kernel. It uses the approved
event-centred subclip oracle. A mask backend is not implemented.

### C. Subclip and alignment contract

`planner.py` partitions each event job into support/non-support segments. Every
model call has an 81-frame (`4n+1`) context. An owned interval has one unique
writer; its capacity is `81 - 2*seam_buffer`. Episode boundaries use only
edge-repeat padding. Each window records all 81 `source_frame_index` and
`is_pad` values. RGB, pose, action and intrinsics are indexed by the same vector.
Only owned frames are stitched back, with no blending and no duplicate write.
The final length must equal `manifest.total_frames`.

The GPU runner chains windows using an already generated overlapping source
frame. It does not use GT as a later-window query image. GT is used only to
construct the explicitly oracle anchor and later primary metric target.

### D. Existing five action cases

The historical V6 automation defines these cases:

- `Ep000027_p0001_75s_87s_lookback_path`
- `Ep000027_p0001_head20s_lookaround`
- `Ep000027_p0006_93s_105s_boxes_lookback`
- `Ep000027_p0007_26s_35s_fwd_back_two_windows`
- `Ep000027_p0007_77s_86s_two_windows_revisit`

Each case is expected to provide `image.jpg`, `prompt.txt`, `poses.npy`,
`action.npy`, `intrinsics.npy`, and `ground_truth_full.mp4`. Historical
generation uses five 81-frame clips: 405 frames at 16 FPS. Seeds were previously
CLI-controlled rather than part of a machine-readable event annotation. Phase
1 requires authoritative `evaluation_seeds`; committed TODO templates use
`null`, while a formal reviewed manifest requires at least three unique
integers.
Outputs were `<infer_root>/<arm>/<case>/long_video.mp4`. Phase 1 replaces that
ambiguous layout with
`phase1/<SHA>/<arm>/<case>/<event>/seed_<seed>/...`.

The repository contains no trustworthy event boundaries or matched-wrong
labels. The five committed manifests therefore contain explicit `null/TODO`
values and intentionally fail formal validation until a human reviews them.

## Added interfaces

- `manifest.py`: strict schema validation; no fallback to global.
- `evaluation_seeds` in each formal manifest is the sole seed authority.
  CLI/environment seed lists are optional exact-equality assertions only.
- `planner.py`: deterministic 81-frame plans, four-modality slicing and unique
  owned-output stitching.
- `jobs.py`: one event-anchor matched job for each arm and seed.
- `run.py`: model-free validate/plan commands and a CUDA generation command
  that reuses the V6 latent-concat generator.
- `provenance.py`: output trace including support, anchor source, seed,
  checkpoint/config, SHA, planner, peak slots/tokens and cumulative exposure.
- Every run preserves canonical fingerprints for commit, checkpoint, config,
  planner/window geometry, source mapping, prompt, trajectory, planned anchor
  budget and decoded output frames. Collection and scoring reject a matched
  `(case,event,seed)` unless all four arms agree; only arm and anchor
  source/exposure schedule may vary.
- `eval_cli.py`: shared GT/static-mask masked-DINO scoring.
- `raft_cli.py` / `raft.py`: official torchvision RAFT-Large evidence
  generation and strict external CSV ingestion. Exactly one finite `[0,1]`
  score is required for every `(case,event,seed,arm,frame)`, with model,
  weights, metric version and generated-video identity recorded.
- `evaluation.py`: strict frame→event→seed→case aggregation and case-only
  bootstrap. The expected universe is constructed from all formal manifest
  events × preregistered seeds × four arms, rather than inferred from rows.
- `collect.py`: duplicate-safe shard/index and CSV collection.

The historical `legacy_v6_global` is not a primary arm and is not used by these
entry points. Existing V6 files and defaults are unchanged.

## Execution-server sequence

From the repository root after `git pull`:

```bash
export CASES_ROOT=/path/to/five_cases
export CKPT_DIR=/path/to/lingbot_checkpoint
export PHASE1_OUTPUT_ROOT=/path/to/output
export STATIC_MASK_ROOT=/path/to/static_masks
export LORA_PATH=/path/to/lora_checkpoint   # optional
export PHASE1_GPUS=0,1,2,3,4
# Optional exact-equality assertion only; manifests remain authoritative.
export PHASE1_SEEDS=42,43,44

# 1. This must fail until all five TODO manifests are reviewed and completed.
bash src/scripts/v7/validate_phase1_manifests.sh

# 2. Single arm/single case smoke.
export CASE_ID=...
export EVENT_ID=...
export ARM=off
export SEED=42
bash src/scripts/v7/run_phase1_single.sh

# 3. Single case, four matched arms.
bash src/scripts/v7/run_phase1_case_four_arms.sh

# 4. All five cases, all events, >=3 seeds, all four arms; GPUs are pooled.
bash src/scripts/v7/run_phase1_all.sh

# 5. Before generation, copy and review the threshold-only preregistration.
# Do not fill thresholds after viewing Phase 1 outputs.
cp src/pipeline/v7/phase1/guardrail_thresholds.template.json /reviewed/path/guardrails.json
export PHASE1_GUARDRAIL_CONFIG=/reviewed/path/guardrails.json

# 6. Generate per-frame RAFT evidence, score, merge and primary evaluation.
# Optional when DEFAULT torchvision weights are not already cached:
export RAFT_WEIGHTS_PATH=/path/to/raft_large_state_dict.pth
bash src/scripts/v7/eval_phase1.sh

# To consume precomputed strict RAFT CSVs instead:
# export RAFT_SCORES_ROOT=/path/to/per_case_raft_csvs
# bash src/scripts/v7/eval_phase1.sh
```

Evaluation writes per-case RAFT CSVs plus `aggregate.json` and `aggregate.csv`,
including support/non-support arm and paired-contrast summaries for
masked-DINO, RAFT-gated masked-DINO, full-frame DINO and SSIM. Missing,
duplicate or mismatched RAFT tuple/frame evidence now fails before aggregation;
GO additionally requires the declared RAFT threshold and every other
guardrail to pass.

### Guardrail evidence contract

`PHASE1_GUARDRAIL_CONFIG` is threshold-only. Its exact canonical JSON is
fingerprinted into every generation run as a matched invariant. The loader
rejects `metric`, `value`, `passed`, unknown guardrail names, and any extra
field. Therefore changing a threshold after generation makes aggregation fail
instead of changing the verdict.

Aggregation also requires the original `runs_index.json` and reloads every
referenced `provenance.json`. For every score row, four fingerprints must be
present and identical: the standalone score column, the score row's embedded
`invariant_fingerprints.guardrail_config`, the validated generation provenance
invariant, and the canonical fingerprint of the threshold file currently being
used. Rewriting the threshold file and the standalone CSV column together is
therefore still rejected.

The five canonical guardrails are frozen in
`pipeline/v7/phase1/guardrails.py`:

- `raft_gated_anti_freeze` /
  `correct_local_raft_drop_vs_global` (maximum allowed degradation);
- `seam` / `seam_band_masked_dino_drop_v1` (maximum);
- `non_support_quality` /
  `non_support_masked_dino_drop_vs_global` (maximum);
- `action_following` / `action_flow_error_v1` (maximum);
- `copy_leakage` / `nonsupport_anchor_copy_similarity_v1` (maximum).

RAFT and non-support values are recomputed from the scored per-frame table.
RAFT CSV rows are additionally tied to the run fingerprint, manifest digest,
generated-video digest and GT-reference digest. A JSON summary cannot override
their value or pass/fail.

Seam, action-following and copy/leakage require official per-frame evidence
CSVs using the exact `EVIDENCE_COLUMNS` schema in `guardrails.py`. Their
expected identities are built independently from formal
`manifest × event × evaluation_seeds × four arms × region spec`: action follows
all frames, copy/leakage follows non-support frames, and seam follows the
deterministic planner boundary bands at `PHASE1_SEAM_BUFFER`. RAFT requires all
frames and non-support quality requires every non-support frame. The expected
sets are never inferred from whichever score/evidence rows happen to exist.
Every required identity must occur exactly once and match the run fingerprint
plus manifest/video/reference/mask digests. Deleting one frame, a whole tuple,
or a whole region makes the guardrail `INCONCLUSIVE` (or rejects aggregation);
producer name and version are mandatory. Supply the external artifacts as:

```bash
export PHASE1_GUARDRAIL_EVIDENCE_ROOT=/path/to/evidence
# contains seam.csv, action_following.csv, copy_leakage.csv
bash src/scripts/v7/eval_phase1.sh
```

Until all three complete identity-linked artifacts exist, the overall Gate is
`INCONCLUSIVE` even when the primary statistics pass. It can never be `GO`
from a handwritten summary JSON.

Formal evaluation additionally requires, per case:

- `STATIC_MASK_ROOT/<case>/static_mask.npy`, `[405,H,W]`, values in `[0,1]`;
- `STATIC_MASK_ROOT/<case>/mask_provenance.json`, containing `source`,
  `model_or_engine`, `version`, `config`, `estimated_from_generated_arm=false`,
  and `excluded_regions=["hud","weapon","dynamic_foreground"]`;
- a mask made once from GT/engine labels and shared by all arms. A mask
  separately estimated from generated output is rejected.

## Compatibility and limitations

- V6 source and defaults are untouched.
- No training or Phase 2 code is included.
- No attention-mask backend is included because the audited interface does not
  expose a safe arbitrary mask.
- Real weight loading, CUDA generation, video encoding, DINO loading, wall time
  and VRAM remain execution-server verification items.
- Static masks and event/wrong annotations are external prerequisites.
- Secondary SSIM/full-DINO/RAFT anti-freeze/action-following are retained as
  guardrail requirements. The primary implementation does not invent a new
  gap metric.
