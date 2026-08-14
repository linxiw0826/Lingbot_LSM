# M2-B-A A0 probe

This is the non-training A0 candidate probe for the frozen single-depth memory
adapter. It checks CPU contracts first, then loads the frozen TRAIN fixture and
the real LingBot-World-Act low-noise Wan model. A temporary wrapper captures the
actual block-0 modulated self-attention input and actual pre-head input. It runs
one unwrapped baseline, disabled/empty/reject wrapped forwards, and one enabled
memory read/bridge. The probe checks exact bypass parity at the block-0,
pre-head, and final latent outputs; enabled finite and route-only exact-zero
behavior; RNG preservation; complete base-state non-mutation; and adapter
save/load/reload state parity. It records GPU peak allocated/reserved bytes.

Run on the execution server:

```bash
bash /mnt/nas/wlx/Memory/projects/Lingbot_LSM/tools/preflight/m2ba_a0/run_a0_probe.sh
```

The shell intentionally returns zero and passes the CPU-test exit code into the
authoritative report; a CPU failure can never produce a candidate status. Read the status in
`/mnt/nas/wlx/Memory/outputs/m2ba_a0_20260814/a0_parity_report.json`. A successful
probe says `A0_GATE_CANDIDATE`; it never declares the A0 Gate passed. A1 training
is outside this tool and remains forbidden until review and an explicit A0 Gate.
