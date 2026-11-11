"""Dispatch the versioned implementations; never overwrite archived reference results."""
import argparse,json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];E=P/'experiments'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('experiment',choices=['joint','source_only','alignment_only','asm_only','mixed_view','features','pretrained','speed','hardware_prepare','hardware_measure','hardware_predict','hardware_analyze']);ap.add_argument('--model',choices=['IA','BPE'],default='IA');ap.add_argument('--seed',type=int,choices=[42,43,44],default=42);ap.add_argument('--index',type=int,default=0);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();cmd=[]
 def add(name,*args):cmd.append([sys.executable,str(E/name),*map(str,args)])
 if a.experiment=='joint':
  if a.seed==42:
   pr=json.loads((E/'ia_bpe_validation_search_20261026/promotions.json').read_text())['promotions'];slot=next(i for i,v in enumerate(pr) if v['trial_id']==(7 if a.model=='IA' else 15));add('ia_bpe_validation_search_20261026/train_search.py','--stage','full','--slot',slot)
  else:add('ia_bpe_tuned_multiseed_20261027/train_fixed.py','--model',a.model,'--seed',a.seed)
 elif a.experiment in ['source_only','alignment_only']:add('supervision_ablations_20261027/train.py','--model',a.experiment,'--seed',a.seed)
 elif a.experiment=='asm_only':add('ipc_validation_20261030/asm_only_train.py',a.seed)
 elif a.experiment=='mixed_view':
  if not (E/'ipc_validation_20261030/strip_train_tokens.npy').exists():add('ipc_validation_20261030/aug_prepare.py')
  add('ipc_validation_20261030/aug_train.py',a.seed,.5)
 elif a.experiment=='features':add('ipc_validation_20261030/prepare.py');add('ipc_validation_20261030/features.py')
 elif a.experiment=='pretrained':
  if not 0<=a.index<12:raise ValueError('Pretrained index must be0..11: model,condition,seed.')
  add('pretrained_equal_steps_20261108/experiment.py','final','--index',a.index)
 elif a.experiment=='speed':cmd.append([sys.executable,str(P/'tools/benchmark_inference.py')])
 else:
  names={'speed':'speed_robustness.py','hardware_prepare':'hw_prepare.py','hardware_measure':'run_hw.py','hardware_predict':'predict_hw.py','hardware_analyze':'hw_analyze.py'};add('ipc_validation_20261030/'+names[a.experiment])
 if a.experiment in ['joint','source_only','alignment_only']:
  if a.experiment=='joint':
   ck=E/f'ia_bpe_validation_search_20261026/trials/trial_{7 if a.model=="IA" else 15:02d}/checkpoint.pt' if a.seed==42 else E/f'ia_bpe_tuned_multiseed_20261027/runs/{a.model}_seed{a.seed}/checkpoint.pt'
  else:ck=E/f'supervision_ablations_20261027/runs/{a.experiment}_seed{a.seed}/checkpoint.pt'
  cmd.append([sys.executable,str(P/'tools/evaluate_compact.py'),'--checkpoint',str(ck),'--output',str(ck.parent/'replicated_evaluation')])
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 for c in cmd:
  print(' '.join(c),flush=True)
  if not a.dry_run:
   if a.experiment in ['joint','source_only','alignment_only','pretrained']:
    installed=[p for p in E.rglob('checkpoint.pt') if p.is_symlink() and (P/'checkpoints') in p.resolve().parents]
    if installed:raise RuntimeError('Evaluation-only checkpoint links are installed. Retrain in a fresh package copy without install_checkpoints.py.')
   subprocess.run(c,cwd=P,env=env,check=True)
if __name__=='__main__':main()
