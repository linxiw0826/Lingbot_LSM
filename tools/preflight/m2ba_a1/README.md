# M2-B-A A1 code preflight

`run_code_preflight.sh` is deliberately non-training. It validates the frozen
TRAIN-only fixture, target token slice, support route mask, and CPU contracts.
It does not read the B/arch fixture, run input-swap, perform optimizer steps, or
authorize the 200-step exploratory run.

On the clean reviewed execution-server commit it writes a TRAIN-only snapshot
after independently rechecking the checkpoint inventory and frozen paths. The
training scripts remain fail-closed until a separate owner authorization JSON
is supplied. That artifact must bind the reviewed repo commit, train config,
TRAIN-only snapshot SHA, runtime-contract SHA, and measured checkpoint
fingerprint. An environment boolean is intentionally insufficient.

`run_exploratory.sh` is limited to the authorized 200-step A-only exploratory
run. `run_resume_equivalence.sh` requires its own authorization artifact and a
fresh output root. Neither entry point implements confirmatory, B/swap, or DEV.
