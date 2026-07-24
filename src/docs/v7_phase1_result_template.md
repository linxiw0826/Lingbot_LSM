# V7 Phase 1 execution result

- Commit SHA:
- Branch:
- Pull command:
- Execution server command(s):
- Checkpoint/config:
- Cases and event counts:
- Seeds per case:
- Manifest `evaluation_seeds` (authoritative):
- Manifest reviewer:
- Manifest review time:
- Per-case valid event counts:
- Missing matched-wrong cases/events:

## Artifacts

- Run root:
- `runs_index.json`:
- `per_frame_all.csv`:
- Per-case RAFT evidence CSVs:
- `aggregate.json`:
- `aggregate.csv`:
- Videos:
- Logs:

## Primary static masked-DINO

| case | correct_local-global | correct_local-off | correct_local-wrong_local |
|---|---:|---:|---:|
| case 1 | | | |
| case 2 | | | |
| case 3 | | | |
| case 4 | | | |
| case 5 | | | |

- correct_local-global positive cases:
- correct_local-global case-bootstrap 95% CI:
- correct_local-wrong_local positive cases:
- correct_local-wrong_local case-bootstrap 95% CI:
- Missing four-arm tuples:
- Cross-arm invariant fingerprint verification:
- Rejected SHA/checkpoint/config/window/source-mapping mismatches:
- Support summary:
- Non-support summary:

## Compute/exposure

- Peak memory slots/tokens by arm:
- Cumulative anchor-frame uses/token-frames by arm:

## Secondary guardrails

- SSIM:
- Full-frame DINO:
- RAFT-gated anti-freeze:
- RAFT-gated masked-DINO:
- RAFT tuple/frame coverage:
- RAFT model / weights / metric version:
- Guardrail result (`passed` / `failed` / `INCONCLUSIVE`):
- Seam observations/metric:
- Non-support quality:
- Copy/leakage:
- Action following:

## Decision

- Suggested status: `GO` / `INCONCLUSIVE` / `NO-GO`
- Evidence:
- Known confounds:
- Required rerun or next action:
