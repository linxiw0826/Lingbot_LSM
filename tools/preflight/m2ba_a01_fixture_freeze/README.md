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
Positive TRAIN/DEV oracle fixtures freeze the complete, unique, deterministic
ordered `memory_frame_indices` set consumed by Phase 1. Every member must be an
integer inside `first_visit` and strictly before `query_start`. The EMPTY/REJECT
safety fixture records its ordered manifest candidates but consumes no selected
memory set.

Run from the repository on the execution server:

```bash
bash /mnt/nas/wlx/Memory/projects/Lingbot_LSM/tools/preflight/m2ba_a01_fixture_freeze/run_probe.sh
```

The shell runner is intentionally nonfatal and always returns `0`, including
when a diagnostic gate fails. It prints `STATIC_PROBE_PYTHON_EXIT=<code>` and
`RUNTIME_PROBE_PYTHON_EXIT=<code>` and records them in the corresponding exit
code logs; authoritative status remains in
`frozen_fixture_manifest.json`, `probe_summary.md`, and the logs. Setup, copy,
Python, and `tee` errors are printed rather than silently hidden.

After the static gate passes, `probe_wan_runtime.py` loads the real raw WanI2V
pipeline and performs exactly two identical direct calls to
`low_noise_model.forward` on the TRAIN fixture window. It uses clean VAE `x0`,
`t=torch.zeros(1)`, the real I2V `y`, text context, Plucker/action condition and
runtime spatial plan. It never enters a scheduler sampling loop, decodes a VAE
latent, writes a video, enables gradients, or trains.

The conditioning-only builder uses the lower-level Wan text/VAE/camera/action
components directly and never constructs a scheduler, RNG, random noise, or
low-noise surrogate. Expert selection is made through the real
`pipeline._prepare_model_for_timestep(t=0,
boundary=pipeline.boundary*pipeline.num_train_timesteps,
offload_model=True)` API and must return the identical `low_noise_model` object.
The historical `0.947*1000` value is recorded only as an observation comparison;
it is never used as runtime authority.

Forward-pre-hooks on `low_noise_model.blocks[0]` and `low_noise_model.head`
record full tensor metadata and only the target latent-frame's 1508-token slice.
The probe validates `[16,21,58,104]` x0, `[20,21,58,104]` y, `[1,L,5120]`
hidden layout, causal `latent_t=ceil(local_frame/4)`, finite values, repeated
bf16 determinism (`atol=rtol=2e-3`), and no base-state mutation. Model-state
checking inventories every parameter/buffer identity, shape, dtype, device and
PyTorch version counter, supplemented by deterministic first/middle/last content
samples. It also explicitly inventories Wan's unregistered plain tensor
`model.freqs`, after its expected device placement and before both measured
forwards. This avoids hashing roughly 28 GB twice and is explicitly not
described as a full cryptographic weight hash. Determinism covers both target
block-0/pre-head slices and the full direct-forward output latent.

On PASS, every frozen fixture target receives a full-frame → planner-window-local
→ causal-latent → token-slice mapping, and TRAIN additionally records every
support frame plus deduplicated many-to-one groups and boundaries. This mapping
is explicitly labelled a causal receptive-field assignment derived from the
planner, VAE stride and patch size—not empirical feature attribution.

Both diagnostic stages are nonfatal at the shell level. The runner always exits
zero and prints `STATIC_PROBE_PYTHON_EXIT` and `RUNTIME_PROBE_PYTHON_EXIT`; the
JSON status and logs remain authoritative.
