"""Probe existing joint-trained assembly encoders; a reference, not a theoretical upper bound."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from features import metrics
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;COMMON=ROOT/'source_ipc_baseline_20261025'
sys.path.insert(0,str(COMMON/'archive/Experiment'))
p=COMMON/'archive/Experiment/final_train_seeded.py';spec=importlib.util.spec_from_file_location('original_encoder',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def main():
 if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
 torch.set_num_threads(4);data={s:np.load(BASE/f'{s}_cache.npz',allow_pickle=False) for s in ['train','val','test']};results={}
 for seed in [42,43,44]:
  checkpoint=ROOT/'ia_bpe_validation_search_20261026/trials/trial_07/checkpoint.pt' if seed==42 else ROOT/f'ia_bpe_tuned_multiseed_20261027/runs/IA_seed{seed}/checkpoint.pt'
  c=torch.load(checkpoint,map_location='cpu',weights_only=True);assert c['step']==9762
  model=mod.Encoder(254).cuda();model.load_state_dict(c['asm_enc']);model.eval()
  def encode(tokens):
   out=[]
   with torch.inference_mode():
    for i in range(0,len(tokens),16):out.append(model(torch.from_numpy(tokens[i:i+16].astype(np.int64)).cuda()).cpu().numpy())
   return np.concatenate(out).astype(np.float64)
  x=encode(data['train']['asm_tokens']);y=np.log1p(data['train']['ipc'][data['train']['token_indices']]);xm=x.mean(0);ym=y.mean();xc=x-xm;coef=np.linalg.solve(xc.T@xc+np.eye(128),xc.T@(y-ym))
  result={'checkpoint':str(checkpoint),'step':c['step'],'n_probe_train':len(x),'ridge_alpha':1,'embedding_standardization':False,'description':'Frozen assembly branch from joint-trained IA model; IPC supervision enters this branch via joint representation learning. Not an assembly-only trained model or theoretical upper bound.'}
  for split in ['val','test']:
   z=encode(data[split]['asm_tokens']);pred=(z-xm)@coef+ym;result[split]=metrics(data[split]['ipc'],pred)
   with (BASE/f'assembly_probe_seed{seed}_{split}_predictions.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['id','ipc_llvm_mca','pred_log1p','pred_ipc']);w.writerows(zip(data[split]['ids'],data[split]['ipc'],pred,np.expm1(pred)))
  results[str(seed)]=result;(BASE/'assembly_probe_results.json').write_text(json.dumps(results,indent=2));print('ASSEMBLY_PROBE',seed,result['test'],flush=True)
  del c,model;torch.cuda.empty_cache()
 summary={s:{m:{'mean':float(np.mean([r[s][m] for r in results.values()])),'sample_sd':float(np.std([r[s][m] for r in results.values()],ddof=1))} for m in ['r2_log1p','r2_raw','mae_ipc']} for s in ['val','test']}
 (BASE/'assembly_probe_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
