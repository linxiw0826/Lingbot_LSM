#!/usr/bin/env bash
set +e

REPO=${REPO:-/mnt/nas/wlx/Memory/projects/Lingbot_LSM}
ROOT=${ROOT:-/mnt/nas/wlx/Memory/outputs/m2ba_a1_resume_equivalence_20260814}
AUTHORIZATION=${A1_RESUME_AUTHORIZATION_FILE:-}
if [ -z "$AUTHORIZATION" ] || [ ! -s "$AUTHORIZATION" ]; then
  echo "A1_TRAINING_BLOCKED: reviewed resume-test authorization artifact missing"
  exit 0
fi
if [ -e "$ROOT" ]; then
  echo "BLOCKED_RESUME_EQUIVALENCE: output root already exists; use a fresh root"
  exit 0
fi
mkdir -p "$ROOT"
cd "$REPO" || { echo "REPO_NOT_FOUND=$REPO"; exit 0; }

python tools/preflight/m2ba_a1/train_exploratory.py --repo "$REPO" \
  --output "$ROOT/uninterrupted" --resume-test-mode --stop-after 20 --authorization "$AUTHORIZATION"
U_EXIT=$?
python tools/preflight/m2ba_a1/train_exploratory.py --repo "$REPO" \
  --output "$ROOT/resumed" --resume-test-mode --stop-after 10 --authorization "$AUTHORIZATION"
S1_EXIT=$?
python tools/preflight/m2ba_a1/train_exploratory.py --repo "$REPO" \
  --output "$ROOT/resumed" --resume-test-mode --stop-after 20 \
  --authorization "$AUTHORIZATION" \
  --resume "$ROOT/resumed/checkpoints/step_0010.pt"
S2_EXIT=$?
printf '{"uninterrupted":%s,"split_1_10":%s,"resume_11_20":%s}\n' \
  "$U_EXIT" "$S1_EXIT" "$S2_EXIT" > "$ROOT/process_exits.json"

python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
import torch
root=Path(sys.argv[1]); report={"schema_version":"m2ba_a1_resume_equivalence_v1"}
def equal(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if hasattr(a,"shape") and hasattr(b,"shape"): return bool((a==b).all())
    return a==b
try:
    exits=json.loads((root/"process_exits.json").read_text())
    exits_ok=all(value==0 for value in exits.values())
    reports=[json.loads((root/name/"exploratory_report.json").read_text()) for name in ("uninterrupted","resumed")]
    reports_ok=reports[0].get("status")=="A1_RESUME_TEST_SEGMENT_COMPLETE" and reports[0].get("completed_optimizer_step")==20 and reports[1].get("status")=="A1_RESUME_TEST_SEGMENT_COMPLETE" and reports[1].get("completed_optimizer_step")==20
    left=torch.load(root/"uninterrupted/checkpoints/step_0020.pt",map_location="cpu",weights_only=False)
    right=torch.load(root/"resumed/checkpoints/step_0020.pt",map_location="cpu",weights_only=False)
    fields=("adapter_state","optimizer_state","rng_state","health_state","sampler_state","canonical_config","config_sha256","fixture_manifest_sha256","runtime_contract_sha256","base_checkpoint_inventory_fingerprint","repo_commit","repo_dirty","trainable_inventory","trainable_inventory_sha256","lr_scheduler_state","lr_scheduler_config","grad_scaler_state","gradient_accumulation","microstep","run_validity")
    field_checks={field:equal(left[field],right[field]) for field in fields}
    normalize=lambda row:{k:v for k,v in row.items() if k not in {"step_time_seconds","gpu_memory_allocated_bytes"}}
    left_sequence=[normalize(x) for x in left["loss_log_tail"]]
    right_sequence=[normalize(x) for x in right["loss_log_tail"]]
    sequence_steps=[x.get("step") for x in left_sequence]
    sequence_ok=sequence_steps==list(range(1,21)) and equal(left_sequence,right_sequence)
    def chain(root_dir):
        result=[]
        for step in (0,10,20):
            pt=root_dir/"checkpoints"/f"step_{step:04d}.pt"
            sc=json.loads(pt.with_suffix(".pt.sha256.json").read_text())
            result.append({"step":step,"schema":sc.get("schema_version"),"validity":sc.get("run_validity"),"parent_step":sc.get("parent_completed_optimizer_step"),"sha_matches":sc.get("sha256")==__import__("hashlib").sha256(pt.read_bytes()).hexdigest()})
        return result
    left_chain,right_chain=chain(root/"uninterrupted"),chain(root/"resumed")
    expected_chain=[{"step":0,"schema":"m2ba_a1_checkpoint_file_v1","validity":"VALID_A1_RUN","parent_step":None,"sha_matches":True},{"step":10,"schema":"m2ba_a1_checkpoint_file_v1","validity":"VALID_A1_RUN","parent_step":0,"sha_matches":True},{"step":20,"schema":"m2ba_a1_checkpoint_file_v1","validity":"VALID_A1_RUN","parent_step":10,"sha_matches":True}]
    chain_ok=left_chain==expected_chain and right_chain==expected_chain
    passed=exits_ok and reports_ok and all(field_checks.values()) and sequence_ok and chain_ok
    report.update({"status":"RESUME_EQUIVALENCE_CANDIDATE" if passed else "RESUME_EQUIVALENCE_GAP","process_exits":exits,"process_exits_ok":exits_ok,"reports_success":reports_ok,"field_checks":field_checks,"full_step_1_20_sequence_equal":sequence_ok,"sequence_steps":sequence_steps,"checkpoint_chain_semantics_equal":chain_ok,"uninterrupted_chain":left_chain,"resumed_chain":right_chain,"steps":[1,20],"split_at":10})
except Exception as exc: report.update({"status":"BLOCKED_RESUME_EQUIVALENCE","error":f"{type(exc).__name__}: {exc}"})
tmp=root/"resume_equivalence.json.tmp"; final=root/"resume_equivalence.json"
tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); tmp.replace(final); print(json.dumps(report,indent=2,sort_keys=True))
PY
CMP_EXIT=$?
echo "UNINTERRUPTED_EXIT=$U_EXIT SPLIT1_EXIT=$S1_EXIT SPLIT2_EXIT=$S2_EXIT COMPARE_EXIT=$CMP_EXIT"
echo "A1_RESUME_EQUIVALENCE_DIAGNOSTIC_COMPLETE"
