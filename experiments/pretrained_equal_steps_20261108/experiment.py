"""Equal-update CodeBERT/UniXcoder IPC-only and assembly-supervised experiments."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,os,random,sys,time
import numpy as np
import torch
from transformers import RobertaModel
from gradient_cache import backward
B=Path(__file__).resolve().parent;E=B.parent;ROOT=E.parent;C=E/'source_ipc_baseline_20261025';PRE=E/'pretrained_ipc_20261027';IA=E/'ia_fixed_joint_ablation_20261026';PLAN=json.loads((B/'protocol.json').read_text());PH=hashlib.sha256((B/'protocol.json').read_bytes()).hexdigest()
path=C/'archive/Experiment/final_train_seeded.py';sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('compact',path);arch=importlib.util.module_from_spec(spec);spec.loader.exec_module(arch)

def dump(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def emb(net,ids):
 mask=ids.ne(1);z=net(input_ids=ids,attention_mask=mask).last_hidden_state
 return (z.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
def runpath(model,kind,lr,slot,seed):return B/'runs'/f'{model}_{kind}_lr{lr:g}_aw{slot}_seed{seed}'
def preflight():
 source=json.loads((C/'source_only_data/manifest.json').read_text());assembly=json.loads((IA/'assembly_data/manifest.json').read_text());assert source['splits']['train']['input_sha256']==assembly['splits']['train']['input_sha256']
 ix=np.load(C/'source_only_data/probe_train_indices.npy');assert len(set(ix.tolist()))==5000
 assert hashlib.sha256((C/'source_only_data/probe_train_indices.npy').read_bytes()).hexdigest()==PLAN['probe_indices_sha256']
 assert np.load(IA/'assembly_data/train_asm_ia_fixed.npy',mmap_mode='r').shape==(309913,512)
 for name in PLAN['models']:
  d=PRE/'data'/name;assert np.load(d/'train.npy',mmap_mode='r').shape==(309913,512)
  assert np.array_equal(np.load(d/'train_ipc.npy'),np.load(C/'source_only_data/train_ipc.npy'))
  assert json.loads((d/'val_ids.json').read_text())==json.loads((E/'ia_bpe_validation_search_20261026/validation_data/ids.json').read_text())
  assert json.loads((d/'test_ids.json').read_text())==np.load(C/'probe_embeddings_seed42.npz')['test_ids'].tolist()
  assert (PRE/'models'/name/'model.safetensors').exists()
 print('PREFLIGHT_PASS',flush=True)

def networks(name,kind,seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
 net=RobertaModel.from_pretrained(PRE/'models'/name,add_pooling_layer=False).cuda();head=torch.nn.Linear(768,1).cuda()
 projection=torch.nn.Linear(768,128,bias=False).cuda() if kind=='joint' else None;assembly=arch.Encoder(254).cuda() if kind=='joint' else None
 return net,head,projection,assembly

def evaluate(net,head,name,out,split):
 out.mkdir(parents=True,exist_ok=True);net.eval();head.eval();d=PRE/'data'/name;idx=np.load(C/'source_only_data/probe_train_indices.npy');train=np.load(d/'train.npy',mmap_mode='r');labels=np.load(d/'train_ipc.npy')
 def collect(data,ix):
  zz=[];pp=[]
  with torch.inference_mode():
   for start in range(0,len(ix),16):
    ids=torch.tensor(np.array(data[ix[start:start+16]],dtype=np.int64),device='cuda');z=emb(net,ids);p=head(z).squeeze(-1);zz.append(z.cpu().numpy());pp.append(p.cpu().numpy())
  return np.concatenate(zz).astype(np.float64),np.concatenate(pp).astype(np.float64)
 x,_=collect(train,idx);y=np.log1p(labels[idx].astype(np.float64));xm=x.mean(0);ym=y.mean();xc=x-xm;coef=np.linalg.solve(xc.T@xc+np.eye(768),xc.T@(y-ym))
 actual=np.load(d/f'{split}_ipc.npy').astype(float)
 if split=='test':actual=np.load(C/'probe_embeddings_seed42.npz')['test_ipc'].astype(float)
 z,direct=collect(np.load(d/f'{split}.npy',mmap_mode='r'),np.arange(len(actual)));pl=(z-xm)@coef+ym;pred=np.expm1(pl)
 assert np.isfinite(pl).all() and np.isfinite(pred).all()
 def r2(a,b):return float(1-np.square(a-b).sum()/np.square(a-a.mean()).sum())
 result={'probe_r2_log1p':r2(np.log1p(actual),pl),'probe_r2_raw_ipc':r2(actual,pred),'probe_mae_ipc':float(abs(actual-pred).mean()),'direct_head_r2_raw_ipc':r2(actual,np.expm1(direct)),'n':len(actual)}
 with (out/f'{split}_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','ipc_llvm_mca','probe_ipc','probe_log1p','direct_head_ipc']);w.writerows(zip(json.loads((d/f'{split}_ids.json').read_text()),actual,pred,pl,np.expm1(direct)))
 np.savez(out/'readout.npz',coef=coef,x_mean=xm,y_mean=ym,probe_indices=idx);dump(out/f'{split}_metrics.json',{split:result});print('EVALUATED',split,json.dumps(result),flush=True);return result

def train(name,kind,lr,slot,seed,stage):
 smoke=stage=='smoke';steps=2 if smoke else PLAN['screen_steps'] if stage=='screen' else PLAN['updates'];weight=PLAN['alignment_weights'][slot] if kind=='joint' else 0.
 run=B/'smoke'/f'{name}_{kind}' if smoke else runpath(name,kind,lr,slot,seed);run.mkdir(parents=True,exist_ok=True)
 net,head,proj,asm=networks(name,kind,seed);params=list(net.parameters())+list(head.parameters())+(list(proj.parameters())+list(asm.parameters()) if asm is not None else []);opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
 data=np.load(PRE/f'data/{name}/train.npy',mmap_mode='r');a_data=np.load(IA/'assembly_data/train_asm_ia_fixed.npy',mmap_mode='r') if asm is not None else None;labels=np.load(PRE/f'data/{name}/train_ipc.npy')
 gen=torch.Generator().manual_seed(seed);perm=torch.randperm(len(data),generator=gen);pos=0;step=0;elapsed=0.;checkpoint=run/'checkpoint.pt';out=run/f'eval_{stage}'
 if checkpoint.exists():
  c=torch.load(checkpoint,map_location='cpu',weights_only=True);assert c['protocol_sha256']==PH
  for module,key in [(net,'encoder'),(head,'head'),(proj,'alignment_projection'),(asm,'asm_enc')]:
   if module is not None:module.load_state_dict(c[key])
  opt.load_state_dict(c['optimizer']);step=c['step'];pos=c['position'];perm=c['permutation'];gen.set_state(c['sampler_rng']);torch.set_rng_state(c['cpu_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);elapsed=c['training_seconds'];del c
  if step>=steps and (smoke or (out/'val_metrics.json').exists()):print('ALREADY_COMPLETE',run,stage,flush=True);return
 config={'model':name,'kind':kind,'lr':lr,'alignment_weight':weight,'seed':seed,'updates':PLAN['updates'],'batch':256,'microbatch':16,'source_parameters':sum(p.numel() for p in net.parameters()),'source_dimensions':768,'train_only_alignment_dimensions':128 if asm is not None else None,'protocol_sha256':PH}
 dump(run/'config.json',config)
 for module in [net,head,proj,asm]:
  if module is not None:module.train()
 start=time.monotonic();last={}
 def save(status):
  torch.cuda.synchronize();used=elapsed+time.monotonic()-start
  state={'encoder':net.state_dict(),'head':head.state_dict(),'alignment_projection':proj.state_dict() if proj is not None else None,'asm_enc':asm.state_dict() if asm is not None else None,'optimizer':opt.state_dict(),'step':step,'position':pos,'permutation':perm,'sampler_rng':gen.get_state(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'training_seconds':used,'protocol_sha256':PH}
  torch.save(state,run/'checkpoint.tmp');(run/'checkpoint.tmp').replace(checkpoint);dump(run/'status.json',{'step':step,'stage':stage,'target_steps':steps,'status':status,'training_seconds':used,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2,**last})
 while step<steps:
  if pos+256>len(data):perm=torch.randperm(len(data),generator=gen);pos=0
  ix=perm[pos:pos+256].numpy();pos+=256;opt.zero_grad(set_to_none=True);batches=[]
  for off in range(0,256,16):
   j=ix[off:off+16];sid=torch.tensor(np.array(data[j],dtype=np.int64),device='cuda');aid=torch.tensor(np.array(a_data[j],dtype=np.int64),device='cuda') if a_data is not None else None;batches.append((sid,aid))
  target=torch.tensor(np.log1p(labels[ix]),device='cuda');last=backward(lambda x:emb(net,x),asm,head,proj,batches,target,weight)
  norm=torch.nn.utils.clip_grad_norm_(params,.5,error_if_nonfinite=True)
  mult=step/200 if step<200 else max(.1,.5*(1+math.cos(math.pi*(step-200)/9800)))
  for group in opt.param_groups:group['lr']=lr*mult
  opt.step();step+=1;last['gradient_norm']=float(norm)
  if step==1 or step%100==0:print(json.dumps({'model':name,'kind':kind,'seed':seed,'lr':lr,'weight':weight,'step':step,'elapsed':elapsed+time.monotonic()-start,**last}),flush=True)
  if not smoke and step%500==0:save('training')
 save('smoke_passed' if smoke else 'trained')
 if not smoke:evaluate(net,head,name,out,'val')
 print('TRAIN_COMPLETE',run,stage,flush=True)

def rank_ipc_screen():
 result={}
 for name in PLAN['models']:
  scores=[{'lr':lr,'validation':json.loads((runpath(name,'ipc',lr,0,42)/'eval_screen/val_metrics.json').read_text())['val']['probe_r2_log1p']} for lr in PLAN['learning_rates']]
  result[name]=sorted(scores,key=lambda x:(-x['validation'],x['lr']))[:2]
 dump(B/'ipc_promotions.json',result)
def freeze_ipc():
 result={}
 for name,candidates in json.loads((B/'ipc_promotions.json').read_text()).items():
  scores=[{'lr':x['lr'],'validation':json.loads((runpath(name,'ipc',x['lr'],0,42)/'eval_full/val_metrics.json').read_text())['val']['probe_r2_log1p']} for x in candidates];result[name]=max(scores,key=lambda x:(x['validation'],-x['lr']))
 dump(B/'selected_ipc.json',{'selection':'seed42 full9762-update validation only; LR shared by IPC-only and joint for each source encoder','models':result,'protocol_sha256':PH})
def rank_joint_screen():
 result={};selected=json.loads((B/'selected_ipc.json').read_text())['models']
 for name in PLAN['models']:
  lr=selected[name]['lr'];scores=[{'slot':i,'alignment_weight':w,'validation':json.loads((runpath(name,'joint',lr,i,42)/'eval_screen/val_metrics.json').read_text())['val']['probe_r2_log1p']} for i,w in enumerate(PLAN['alignment_weights'])];result[name]=sorted(scores,key=lambda x:(-x['validation'],x['alignment_weight']))[:2]
 dump(B/'joint_promotions.json',result)
def freeze_joint():
 result={};ip=json.loads((B/'selected_ipc.json').read_text())['models']
 for name,candidates in json.loads((B/'joint_promotions.json').read_text()).items():
  lr=ip[name]['lr'];scores=[{'slot':x['slot'],'alignment_weight':x['alignment_weight'],'validation':json.loads((runpath(name,'joint',lr,x['slot'],42)/'eval_full/val_metrics.json').read_text())['val']['probe_r2_log1p']} for x in candidates];result[name]=dict(lr=lr,**max(scores,key=lambda x:(x['validation'],-x['alignment_weight'])))
 dump(B/'selected_joint.json',{'models':result,'selection':'seed42 full9762-update validation only; frozen before all final test evaluation','protocol_sha256':PH})
def final(index):
 name=PLAN['models'][index//6];kind=['ipc','joint'][(index%6)//3];seed=PLAN['seeds'][index%3];lr=json.loads((B/'selected_ipc.json').read_text())['models'][name]['lr'];slot=json.loads((B/'selected_joint.json').read_text())['models'][name]['slot'] if kind=='joint' else 0
 train(name,kind,lr,slot,seed,'full');run=runpath(name,kind,lr,slot,seed);out=run/'eval_full'
 if (out/'test_metrics.json').exists():return
 net,head,_,_=networks(name,'ipc',seed);c=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True);net.load_state_dict(c['encoder']);head.load_state_dict(c['head']);del c;evaluate(net,head,name,out,'test');print('FINAL_COMPLETE',name,kind,seed,flush=True)
def aggregate():
 rows=[];summary={};ip=json.loads((B/'selected_ipc.json').read_text())['models'];joint=json.loads((B/'selected_joint.json').read_text())['models']
 for name in PLAN['models']:
  for kind in ['ipc','joint']:
   rr=[];lr=ip[name]['lr'];slot=joint[name]['slot'] if kind=='joint' else 0
   for seed in PLAN['seeds']:
    run=runpath(name,kind,lr,slot,seed);m=json.loads((run/'eval_full/test_metrics.json').read_text())['test'];st=json.loads((run/'status.json').read_text());assert st['step']==9762;row={'model':name,'kind':kind,'seed':seed,'lr':lr,'alignment_weight':PLAN['alignment_weights'][slot] if kind=='joint' else 0.,'step':st['step'],'training_seconds':st['training_seconds'],**m};rr.append(row);rows.append(row)
   summary[f'{name}_{kind}']={k:{'mean':float(np.mean([r[k] for r in rr])),'sample_sd':float(np.std([r[k] for r in rr],ddof=1))} for k in ['probe_r2_log1p','probe_r2_raw_ipc','probe_mae_ipc']}
 dump(B/'summary.json',{'protocol':PLAN,'summary':summary,'per_seed':rows,'paired_joint_minus_ipc_r2':{n:[next(r for r in rows if r['model']==n and r['kind']=='joint' and r['seed']==s)['probe_r2_log1p']-next(r for r in rows if r['model']==n and r['kind']=='ipc' and r['seed']==s)['probe_r2_log1p'] for s in PLAN['seeds']] for n in PLAN['models']}})
 with (B/'per_seed.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('ALL12_FINAL_RUNS_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','smoke','ipc_screen','ipc_rank','ipc_promote','ipc_select','joint_screen','joint_rank','joint_promote','joint_select','final','aggregate']);ap.add_argument('--index',type=int,default=0);a=ap.parse_args();i=a.index
 if a.stage=='preflight':preflight()
 elif a.stage=='smoke':
  preflight()
  for name in PLAN['models']:
   for kind in ['ipc','joint']:train(name,kind,5e-5,1 if kind=='joint' else 0,42,'smoke')
 elif a.stage=='ipc_screen':train(PLAN['models'][i//3],'ipc',PLAN['learning_rates'][i%3],0,42,'screen')
 elif a.stage=='ipc_rank':rank_ipc_screen()
 elif a.stage=='ipc_promote':
  name=PLAN['models'][i//2];lr=json.loads((B/'ipc_promotions.json').read_text())[name][i%2]['lr'];train(name,'ipc',lr,0,42,'full')
 elif a.stage=='ipc_select':freeze_ipc()
 elif a.stage=='joint_screen':
  name=PLAN['models'][i//3];lr=json.loads((B/'selected_ipc.json').read_text())['models'][name]['lr'];train(name,'joint',lr,i%3,42,'screen')
 elif a.stage=='joint_rank':rank_joint_screen()
 elif a.stage=='joint_promote':
  name=PLAN['models'][i//2];lr=json.loads((B/'selected_ipc.json').read_text())['models'][name]['lr'];slot=json.loads((B/'joint_promotions.json').read_text())[name][i%2]['slot'];train(name,'joint',lr,slot,42,'full')
 elif a.stage=='joint_select':freeze_joint()
 elif a.stage=='final':final(i)
 else:aggregate()
