#!/usr/bin/env python3
"""Read-only, fail-closed M2-B-A A0/A1 fixture evidence probe."""
from __future__ import annotations

import argparse, ast, hashlib, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIXTURES = (
 ("TRAIN","Ep000027_p0007_77s_86s_two_windows_revisit","side_alley_return",(280,405)),
 ("UNTRAINED_DEV","Ep000027_p0007_26s_35s_fwd_back_two_windows","arch_return",(272,405)),
 ("EMPTY_REJECT_SAFETY","Ep000027_p0001_75s_87s_lookback_path","lookback_non_strict_probe",(176,224)),
)
SMALL_LIMIT=64*1024*1024

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def stable(v:Any)->str: return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(path:Path, text:str):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(text); os.replace(tmp,path)
def dump(path:Path,v:Any): atomic(path,json.dumps(v,indent=2,sort_keys=True)+"\n")
def cmd(args,cwd=None):
 try:return subprocess.run(args,cwd=cwd,text=True,capture_output=True,check=False)
 except OSError as e:return type('R',(),{'returncode':127,'stdout':'','stderr':str(e)})()
def jload(p): return json.loads(p.read_text())
def first_existing(xs): return [p for p in xs if p.is_file()]

def resolve_manifest(cid:str, dirs:list[Path])->dict[str,Any]:
 """Resolve path aliases by content, while failing closed on content conflicts."""
 matches=[]; seen=set()
 for path in first_existing([d/f'{cid}.json' for d in dirs]):
  key=str(path)
  if key not in seen: matches.append(path); seen.add(key)
 groups:dict[str,list[Path]]={}
 for path in matches: groups.setdefault(sha(path),[]).append(path)
 result={
  'candidate_paths':[str(x) for x in matches],
  'content_groups':[{'sha256':digest,'paths':[str(x) for x in paths]}
                    for digest,paths in groups.items()],
  'selected_path':None,'selected_sha256':None,'alias_paths':[],
  'status':'missing' if not matches else 'content_conflict',
 }
 if len(groups)==1:
  digest,aliases=next(iter(groups.items()))
  # matches already follows the explicit directory-priority order.
  result.update(status='resolved',selected_path=str(aliases[0]),
                selected_sha256=digest,alias_paths=[str(x) for x in aliases])
 return result

def resolve_memory(role:str, frames:Any, first_visit:tuple[Any,Any], query_start:Any)->dict[str,Any]:
 values=list(frames) if isinstance(frames,list) else []
 if role=='EMPTY_REJECT_SAFETY':
  return {'candidate_full_frames':values,'selected_full_frames':None,
          'selection_status':'not_consumed',
          'selection_reason':'not consumed by empty/reject fixture','valid':True}
 fs,fe=first_visit
 integers=all(isinstance(value,int) and not isinstance(value,bool) for value in values)
 unique=len(set(values))==len(values) if integers else False
 nonempty=bool(values)
 before_query=integers and isinstance(query_start,int) and all(value<query_start for value in values)
 inside_first=integers and isinstance(fs,int) and isinstance(fe,int) and all(fs<=value<fe for value in values)
 valid=nonempty and integers and unique and before_query and inside_first
 return {'candidate_full_frames':values,
         'selected_full_frames':values if valid else None,
         'selection_status':'selected' if valid else 'ambiguous',
         'selection_reason':'entire deterministic ordered Phase 1 memory frame set' if valid else
                            'positive oracle fixture requires a nonempty ordered set of unique integer frames inside first_visit and before query_start',
         'checks':{'nonempty':nonempty,'all_integers':integers,'unique':unique,
                   'all_before_query':before_query,'all_inside_first_visit':inside_first},
         'valid':valid}

def source_contract(repo:Path):
 out={'status':'STATIC_ONLY','evidence':{},'unresolved':[]}
 files={'model':repo/'refs/lingbot-world/wan/modules/model.py','config':repo/'refs/lingbot-world/wan/configs/wan_i2v_A14B.py','run':repo/'src/pipeline/v7/phase1/run.py'}
 for k,p in files.items():
  if not p.is_file(): out['unresolved'].append(f'missing:{p}'); continue
  t=p.read_text(errors='replace'); out['evidence'][k]={'path':str(p),'sha256':sha(p)}
  if k=='model':
   out.update({'wan_class':'wan.modules.model.WanModel','block0':'model.blocks[0]',
    'block0_self_attention':'model.blocks[0].self_attn','pre_head_hook_candidate':'model.head (forward_pre_hook required)',
    'qkvo_attribute_candidates':{x:bool(re.search(rf'self\.{x}\s*=',t)) for x in ('q','k','v','o')}})
  if k=='config':
   for name in ('vae_stride','patch_size'):
    m=re.search(rf'{name}\s*=\s*(\([^\n]+\))',t)
    out[name]=list(ast.literal_eval(m.group(1))) if m else None
 return out

def planner(total,support,context=81,seam=8):
 start,end=support; seg=[]
 if start:seg.append((0,start,False))
 seg.append((start,end,True))
 if end<total:seg.append((end,total,False))
 plans=[]; cap=context-2*seam
 for a,b,sup in seg:
  cur=a
  while cur<b:
   own=min(b,cur+cap); ss=min(max(0,cur-seam),max(0,total-context)); src=list(range(ss,min(total,ss+context)))
   src += [src[-1]]*(context-len(src)); plans.append({'window_index':len(plans),'source_frame_index':src,'owned_half_open':[cur,own],'support':sup});cur=own
 return plans

def select_prov(cands,event):
 valid=[]
 for p in cands:
  try:v=jload(p)
  except Exception:continue
  if v.get('case_id')!=event[0] or v.get('event_id')!=event[1]:continue
  video=Path(v.get('video','')); idx=p.parent/'run_index_entry.json'
  valid.append((int(v.get('config',{}).get('num_inference_steps',0) if isinstance(v.get('config'),dict) else 0),video.is_file() and idx.is_file(),str(p),v))
 valid.sort(key=lambda x:(x[1],x[0],x[2]),reverse=True)
 return valid

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',default=os.environ.get('REPO','/mnt/nas/wlx/Memory/projects/Lingbot_LSM')); ap.add_argument('--cases',default=os.environ.get('CASES','/mnt/h20/135/Memory-world/inference_data/revisit_ep027_manual_v2_5clip_selected')); ap.add_argument('--ckpt',default=os.environ.get('CKPT','/mnt/h20/135/lingbot-models/lingbot-world-base-act')); ap.add_argument('--output',default=os.environ.get('PROBE_OUT','/mnt/nas/wlx/Memory/outputs/m2ba_a01_fixture_freeze_20260811')); ap.add_argument('--manifest-dir',action='append'); a=ap.parse_args()
 repo,cases,ckpt,out=map(Path,(a.repo,a.cases,a.ckpt,a.output)); out.mkdir(parents=True,exist_ok=True)
 unresolved=[]; ambiguities=[]; fps=[]
 git_head=cmd(['git','rev-parse','HEAD'],repo); git_status=cmd(['git','status','--short'],repo)
 env={'python':sys.version,'platform':platform.platform(),'repo_exists':repo.is_dir(),'cases_exists':cases.is_dir(),'checkpoint_exists':ckpt.is_dir(),'git_head':git_head.stdout.strip(),'git_status_short':git_status.stdout}
 try:
  import torch
  env.update(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
 except Exception as e: env['torch_error']=repr(e)
 # Authority order is explicit. src/pipeline/v7/phase1/manifests is intentionally
 # excluded because it contains null/TODO templates, not runnable fixtures.
 mdirs=[Path(x) for x in (a.manifest_dir or [])]+[
  Path('/mnt/nas/wlx/Memory/outputs/phase1_three_arm_pilot_20260803/manifests'),
  Path('/mnt/workspace/wlx/Memory/outputs/phase1_three_arm_pilot_20260803/manifests'),
  repo/'src/scripts/v7/dev/manifests',
 ]
 prov_roots=[Path('/mnt/nas/wlx/Memory/outputs'),Path('/mnt/workspace/wlx/Memory/outputs')]
 provs=[]
 for root in prov_roots:
  if root.is_dir(): provs.extend(root.rglob('provenance.json'))
 fixtures=[]
 for role,cid,eid,expected in FIXTURES:
  resolution=resolve_manifest(cid,mdirs)
  f={'role':role,'case_id':cid,'event_id':eid,'candidate_support_half_open':list(expected),
     'manifest_candidates':resolution['candidate_paths'],
     'manifest_alias_paths':resolution['alias_paths'],
     'manifest_content_groups':resolution['content_groups'],
     'manifest_selected_path':resolution['selected_path'],
     'manifest_selected_sha256':resolution['selected_sha256'],'validation':{}}
  if resolution['status']!='resolved':
   kind='manifest_missing' if resolution['status']=='missing' else 'manifest_content_conflict'
   ambiguities.append({'fixture':cid,'kind':kind,'candidates':resolution['candidate_paths'],
                       'content_groups':resolution['content_groups']})
   unresolved.append(f'{cid}:{kind}');fixtures.append(f);continue
  mp=Path(resolution['selected_path']); m=jload(mp); ev=[x for x in m.get('revisit_events',[]) if x.get('event_id')==eid]
  if len(ev)!=1: unresolved.append(f'{cid}:event_not_unique');fixtures.append(f);continue
  e=ev[0]; support=(e.get('query_start'),e.get('query_end')); first=m.get('first_visit',{}); target=(support[0]+support[1]-1)//2 if all(isinstance(x,int) for x in support) and support[0]<support[1] else None
  first_interval=(first.get('start'),first.get('end'))
  memory=resolve_memory(role,e.get('memory_frame_indices'),first_interval,support[0])
  case=cases/cid; inputs=[]
  for n in ('poses.npy','action.npy','intrinsics.npy','ground_truth_full.mp4','prompt.txt','image.jpg','image.png'):
   p=case/n
   if p.is_file(): inputs.append(p)
  for p in [mp,*inputs]: fps.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
  plans=planner(int(m['total_frames']),support) if target is not None else []
  owners=[p for p in plans if p['owned_half_open'][0]<=target<p['owned_half_open'][1]]
  local=None
  if len(owners)==1:
   inds=owners[0]['source_frame_index']; local=[i for i,x in enumerate(inds) if x==target]
  pv=select_prov(provs,(cid,eid)); selected=pv[0][3] if pv else None
  f.update(manifest_path=str(mp),manifest_sha256=resolution['selected_sha256'],total_frames=m.get('total_frames'),fps=m.get('fps'),support_full_half_open=list(support),first_visit_full_half_open=[first.get('start'),first.get('end')],target_full_frame=target,memory_candidate_full_frames=memory['candidate_full_frames'],memory_selected_full_frames=memory['selected_full_frames'],memory_selection_status=memory['selection_status'],memory_selection_reason=memory['selection_reason'],memory_selection_checks=memory.get('checks'),memory_source='manifest.revisit_events[].memory_frame_indices (the complete ordered tuple consumed by Phase 1)',input_paths=[str(x) for x in inputs],planner_windows=plans,target_window_id=owners[0]['window_index'] if len(owners)==1 else None,target_window_local_indices=local,provenance_candidates=[x[2] for x in pv],selected_provenance_path=pv[0][2] if pv else None,selected_provenance=selected,frame_to_token_mapping={'status':'BLOCKED_GPU_RUNTIME','reason':'requires actual VAE/model hidden runtime shape'},validation={'support_matches_candidate':support==expected,'target_inside_support':target is not None and support[0]<=target<support[1],'manifest_unique_logical_content':True,'manifest_alias_count':len(resolution['alias_paths']),'event_unique':True,'memory_contract_valid':memory['valid'],'inputs_required_present':all((case/n).is_file() for n in ('poses.npy','action.npy','intrinsics.npy','ground_truth_full.mp4'))})
  if support!=expected: unresolved.append(f'{cid}:support_candidate_mismatch')
  if not memory['valid']: unresolved.append(f'{cid}:memory_frame_set_invalid')
  fixtures.append(f)
 # checkpoint inventory, content hashes only for small metadata/config files
 inv=[]
 if ckpt.is_dir():
  for p in sorted(x for x in ckpt.rglob('*') if x.is_file()):
   rel=str(p.relative_to(ckpt)); st=p.stat(); row={'path':rel,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
   if st.st_size<=SMALL_LIMIT and p.suffix.lower() in {'.json','.txt','.yaml','.yml','.index'}: row['sha256']=sha(p)
   inv.append(row)
 ckfp=stable(inv)
 static=source_contract(repo)
 runtime={'status':'BLOCKED_MINIMAL_FORWARD_API_MISSING','reason':'Repository exposes generation pipeline but no independently verified no-sampling teacher-forward fixture API. Static inspection cannot prove clean t=0, hidden layout, determinism, or frame-to-token mapping. No GPU model load was attempted.','static_contract':static,'required_runtime_evidence':['actual block-0 input hook','actual pre-head input hook','clean t=0 semantics','repeat determinism','full-frame to latent/patch/token mapping']}
 static_ok=repo.is_dir() and cases.is_dir() and ckpt.is_dir() and len(fixtures)==3 and not unresolved and all(x.get('validation',{}).get('inputs_required_present') for x in fixtures)
 status='BLOCKED_MINIMAL_FORWARD_API_MISSING' if static_ok else 'BLOCKED_STATIC_FACTS'
 frozen={'schema_version':'m2ba_a01_fixture_freeze_v1','status':status,'generated_at_utc':utc(),'repo':str(repo),'repo_commit':env.get('git_head'),'repo_dirty_preexisting':bool(env.get('git_status_short')),'cases_root':str(cases),'checkpoint_root':str(ckpt),'checkpoint_fingerprint_kind':'stable_inventory(path,bytes,mtime_ns)+small_config_sha256','checkpoint_fingerprint':ckfp,'environment':env,'fixtures':fixtures,'unresolved':unresolved,'blockers':[runtime['status']]}
 dump(out/'frozen_fixture_manifest.json',frozen);dump(out/'runtime_contract.json',runtime);dump(out/'file_fingerprints.json',{'inputs':fps,'checkpoint_inventory':inv,'checkpoint_fingerprint':ckfp});dump(out/'ambiguities.json',{'items':ambiguities})
 summary=f"""# M2-B-A A0/A1 fixture freeze probe\n\n- Final status: `{status}`\n- FIXTURE_STATIC_GATE: `{'PASS' if static_ok else 'FAIL'}`\n- WAN_RUNTIME_CONTRACT_GATE: `BLOCKED`\n- Repository: `{repo}` @ `{env.get('git_head','')}`\n- Checkpoint fingerprint ({frozen['checkpoint_fingerprint_kind']}): `{ckfp}`\n\n## Fixtures\n\n"""
 for x in fixtures: summary+=f"- {x['role']}: `{x['case_id']}` / `{x['event_id']}`; support={x.get('support_full_half_open')}; memory_candidates={x.get('memory_candidate_full_frames')}; memory_selected_set={x.get('memory_selected_full_frames')}; target={x.get('target_full_frame')}; window={x.get('target_window_id')}; local={x.get('target_window_local_indices')}\n"
 summary+="\n## Runtime blocker\n\n"+runtime['reason']+"\n\nNo repository, data, manifest, checkpoint, or existing Phase 1 output was modified.\n"
 atomic(out/'probe_summary.md',summary)
 generated=['frozen_fixture_manifest.json','runtime_contract.json','file_fingerprints.json','ambiguities.json','probe_summary.md']
 atomic(out/'generated_files.txt',''.join(str(out/x)+'\n' for x in generated))
 sums=[]
 for x in generated+['generated_files.txt']: sums.append(f"{sha(out/x)}  {x}\n")
 atomic(out/'SHA256SUMS',''.join(sums)); print(summary); return 0 if static_ok else 2
if __name__=='__main__': raise SystemExit(main())
