"""Source-only pretrained baselines; validation-only LR selection."""
import os
os.environ.setdefault('HF_HOME','/proj/berzelius-2026-145/users/x_liryi/pretrained_ipc_20261027/hf')
import argparse,csv,json,math,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import RobertaModel,RobertaTokenizer
B=Path(__file__).resolve().parent
R=B.parent; C=R/'source_ipc_baseline_20261025'
OLD=R/'pretrained_ipc_20261027'
BUDGET_SECONDS=1800.
MODELS={'codebert':'microsoft/codebert-base','unixcoder':'microsoft/unixcoder-base'}
LRS=[1e-5,2e-5,5e-5]
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def encoder(name):return RobertaModel.from_pretrained(OLD/'models'/name,add_pooling_layer=False).cuda()
def embed(net,ids):
 mask=ids.ne(1);z=net(input_ids=ids,attention_mask=mask).last_hidden_state
 return (z.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
def evaluate(net,head,name,run,test=False):
 net.eval();head.eval();d=OLD/'data'/name
 ix=np.load(C/'source_only_data/probe_train_indices.npy');tr=np.load(d/'train.npy',mmap_mode='r');labels=np.load(d/'train_ipc.npy')
 def collect(data,indices):
  zs=[];ps=[]
  with torch.inference_mode():
   for i in range(0,len(indices),16):
    x=torch.tensor(np.array(data[indices[i:i+16]],dtype=np.int64),device='cuda')
    z=embed(net,x);p=head(z).squeeze(-1)
    zs.append(z.float().cpu().numpy());ps.append(p.float().cpu().numpy())
  return np.concatenate(zs).astype(float),np.concatenate(ps).astype(float)
 x,_=collect(tr,ix);y=np.log1p(labels[ix].astype(float));xm=x.mean(0);ym=y.mean();xc=x-xm;coef=np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(y-ym))
 results={}
 for split in (['test'] if test else ['val']):
  actual=np.load(d/f'{split}_ipc.npy').astype(float)
  if split=='test':actual=np.load(C/'probe_embeddings_seed42.npz')['test_ipc'].astype(float)
  z,direct=collect(np.load(d/f'{split}.npy',mmap_mode='r'),np.arange(len(actual)));pl=(z-xm)@coef+ym;pred=np.expm1(pl)
  def r2(a,b):return float(1-((a-b)**2).sum()/((a-a.mean())**2).sum())
  results[split]={'probe_r2_log1p':r2(np.log1p(actual),pl),'probe_r2_raw_ipc':r2(actual,pred),'probe_mae_ipc':float(abs(actual-pred).mean()),'direct_head_r2_raw_ipc':r2(actual,np.expm1(direct)),'n':len(actual)}
  with (run/f'{split}_predictions.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['id','ipc_llvm_mca','probe_ipc','probe_log1p','direct_head_ipc']);w.writerows(zip(json.loads((d/f'{split}_ids.json').read_text()),actual,pred,pl,np.expm1(direct)))
 dump(run/('test_metrics.json' if test else 'validation_metrics.json'),results)
 return results
def train(name,lr,seed,test=False):
 run=B/'runs'/f'{name}_lr{lr:g}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 net=encoder(name);head=torch.nn.Linear(768,1).cuda();params=list(net.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
 data=np.load(OLD/'data'/name/'train.npy',mmap_mode='r');labels=np.load(OLD/'data'/name/'train_ipc.npy')
 gen=torch.Generator().manual_seed(seed);perm=torch.randperm(len(data),generator=gen);pos=0;step=0;elapsed=0.;warmup_elapsed=None
 ck=run/'checkpoint.pt'
 if ck.exists():
  c=torch.load(ck,map_location='cpu',weights_only=True)
  if c['plan_sha256']!=hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest():raise RuntimeError('Frozen protocol changed')
  net.load_state_dict(c['encoder']);head.load_state_dict(c['head']);opt.load_state_dict(c['optimizer']);step=c['step'];pos=c['position'];perm=c['permutation'];gen.set_state(c['sampler_rng']);torch.set_rng_state(c['cpu_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);elapsed=c['elapsed'];warmup_elapsed=c['warmup_elapsed']
 dump(run/'config.json',{'model':MODELS[name],'seed':seed,'lr':lr,'budget_seconds':BUDGET_SECONDS,'timer':'training wall time incl periodic checkpoints; excludes model loading, tokenization and evaluation','effective_batch':256,'microbatch':16,'target':'log1p IPC','loss':'unweighted MSE','pooling':'masked mean, full 768 dimensions','warmup':200,'scheduler':'cosine over remaining wall time after 200-update warmup; floor 0.1','clip':.5,'weight_decay':.01,'source_parameters':sum(p.numel() for p in net.parameters()),'assembly_used':False,'evaluation_precision':'float32','plan_sha256':hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest()})
 start=time.monotonic();net.train();head.train()
 def save(status,loss_value):
  torch.cuda.synchronize();used=elapsed+time.monotonic()-start
  state={'encoder':net.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict(),'step':step,'position':pos,'permutation':perm,'sampler_rng':gen.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'elapsed':used,'warmup_elapsed':warmup_elapsed,'budget_seconds':BUDGET_SECONDS,'plan_sha256':hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest()}
  tmp=run/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(ck)
  dump(run/'status.json',{'step':step,'status':status,'training_seconds':used,'budget_seconds':BUDGET_SECONDS,'loss':loss_value})
 while elapsed+time.monotonic()-start<BUDGET_SECONDS:
  if pos+256>len(data):perm=torch.randperm(len(data),generator=gen);pos=0
  ix=perm[pos:pos+256].numpy();pos+=256;opt.zero_grad(set_to_none=True);loss_value=0
  for off in range(0,256,16):
   j=ix[off:off+16];ids=torch.tensor(np.array(data[j],dtype=np.int64),device='cuda');target=torch.tensor(np.log1p(labels[j]),device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.mse_loss(head(embed(net,ids)).squeeze(-1),target)/16
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();loss_value+=float(loss.detach())
  torch.nn.utils.clip_grad_norm_(params,.5)
  used=elapsed+time.monotonic()-start
  if step<200:multiplier=step/200
  else:
   if warmup_elapsed is None:warmup_elapsed=used
   progress=min(1.,max(0.,(used-warmup_elapsed)/max(1.,BUDGET_SECONDS-warmup_elapsed)))
   multiplier=max(.1,.5*(1+math.cos(math.pi*progress)))
  for group in opt.param_groups:group['lr']=lr*multiplier
  opt.step();step+=1
  if step==1 or step%100==0:print(name,seed,lr,step,loss_value,flush=True)
  if step==1 or step%500==0:save('training',loss_value)
 save('trained',loss_value if 'loss_value' in locals() else None)
 if not (run/'validation_metrics.json').exists():evaluate(net,head,name,run)
 if test and not (run/'test_metrics.json').exists():evaluate(net,head,name,run,True)
 if test:aggregate()
 print('COMPLETE',run,flush=True)
def aggregate():
 import fcntl
 with (B/'aggregate.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  selected=json.loads((B/'selected.json').read_text());summary={};rows=[]
  for name,lr in selected.items():
   metrics=[]
   for seed in [42,43,44]:
    run=B/'runs'/f'{name}_lr{lr:g}_seed{seed}';path=run/'test_metrics.json'
    if not path.exists():return
    m=json.loads(path.read_text())['test'];metrics.append(m);rows.append(dict(model=name,seed=seed,lr=lr,**m))
   summary[name]={k:{'mean':float(np.mean([m[k] for m in metrics])),'sample_sd':float(np.std([m[k] for m in metrics],ddof=1))} for k in metrics[0] if k!='n'}
  dump(B/'summary.json',{'budget_seconds':BUDGET_SECONDS,'selected_lrs':selected,'seeds':[42,43,44],'summary':summary,'per_seed':rows})
  with (B/'per_seed.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def select():
 selected={}
 for name in MODELS:
  scores=[(json.loads((B/'runs'/f'{name}_lr{lr:g}_seed42'/'validation_metrics.json').read_text())['val']['probe_r2_log1p'],lr) for lr in LRS]
  selected[name]=max(scores,key=lambda x:(x[0],-x[1]))[1]
 dump(B/'selected.json',selected)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','select','final','preflight']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
 if a.stage=='preflight':
  for name in MODELS:
   assert (OLD/'models'/name/'model.safetensors').exists()
   assert np.load(OLD/'data'/name/'train.npy',mmap_mode='r').shape==(309913,512)
   assert np.array_equal(np.load(OLD/'data'/name/'train_ipc.npy'),np.load(C/'source_only_data/train_ipc.npy'))
  print('PREFLIGHT_OK',flush=True)
 elif a.stage=='select':select()
 elif a.stage=='train':train(list(MODELS)[a.index//3],LRS[a.index%3],42)
 else:
  name=list(MODELS)[a.index//3];train(name,json.loads((B/'selected.json').read_text())[name],42+a.index%3,True)
