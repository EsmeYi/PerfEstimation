"""Validation-only equal-budget hyperparameter search, with resumable promotion."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('IPC_PROJECT_ROOT','/proj/berzelius-2026-145/users/x_liryi'))
COMMON=ROOT/'source_ipc_baseline_20261025'
ASSEMBLY={'IA':(ROOT/'ia_fixed_joint_ablation_20261026/assembly_data/train_asm_ia_fixed.npy',254),
          'BPE':(ROOT/'bpe_joint_ablation_20261025/assembly_data/train_asm_bpe.npy',16384)}
PLAN=json.loads((BASE/'search_plan.json').read_text())

def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def trial_config(index):
    if not 0<=index<16:raise ValueError('Bad trial ID')
    model=PLAN['models'][index//8]
    within=index%8
    return {'trial_id':index,'model':model,'target':PLAN['targets'][within//4],
            'base_weight':PLAN['base_weights'][within%4],'seed':PLAN['seed']}

def load_encoder_class():
    path=COMMON/'archive/Experiment/final_train_seeded.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('original_training',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.Encoder

def load_inputs(model_name):
    data=COMMON/'source_only_data'
    source=np.load(data/'train_tokens.npy',mmap_mode='r')
    labels=np.load(data/'train_ipc.npy',mmap_mode='r')
    asm=np.load(ASSEMBLY[model_name][0],mmap_mode='r')
    probe_indices=np.load(data/'probe_train_indices.npy',allow_pickle=False)
    validation=np.load(BASE/'validation_data/source_tokens.npy',mmap_mode='r')
    validation_labels=np.load(BASE/'validation_data/ipc.npy',allow_pickle=False)
    if source.shape!=asm.shape or source.shape[1]!=512:raise ValueError('Wrong paired shapes')
    if len(set(probe_indices.tolist()))!=5000:raise ValueError('Invalid training probe selection')
    source_manifest=json.loads((data/'manifest.json').read_text())
    assembly_manifest=json.loads((ASSEMBLY[model_name][0].parent/'manifest.json').read_text())
    val_manifest=json.loads((BASE/'validation_data/manifest.json').read_text())
    if source_manifest['splits']['train']['input_sha256']!=assembly_manifest['splits']['train']['input_sha256']:
        raise ValueError('Assembly/source training data mismatch')
    if source_manifest['fingerprint']['tokenizer_sha256']!=val_manifest['tokenizer_sha256']:
        raise ValueError('Validation/source tokenizer mismatch')
    return source,asm,labels,probe_indices,validation,validation_labels

def evaluate(src,head,inputs,target,run,stage):
    source,_,labels,indices,validation,val_labels=inputs
    src.eval();head.eval()
    def collect(values,ix):
        zs=[];ps=[]
        with torch.inference_mode():
            for off in range(0,len(ix),16):
                ids=torch.from_numpy(np.array(values[ix[off:off+16]],dtype=np.int64)).cuda()
                z=src(ids);zs.append(z.cpu().numpy());ps.append(head(z).squeeze(-1).cpu().numpy())
        return np.concatenate(zs).astype(np.float64),np.concatenate(ps).astype(np.float64)
    x,_=collect(source,indices);z,direct=collect(validation,np.arange(len(validation)))
    y=np.log1p(np.array(labels[indices],dtype=np.float64));actual=val_labels.astype(np.float64)
    xm=x.mean(0);ym=y.mean();xc=x-xm
    coef=np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(y-ym))
    pred=(z-xm)@coef+ym
    raw_direct=direct if target=='raw' else np.expm1(direct)
    def r2(a,b):return float(1-((a-b)**2).sum()/((a-a.mean())**2).sum())
    metrics={'val_probe_r2_log1p':r2(np.log1p(actual),pred),
             'val_probe_r2_raw':r2(actual,np.expm1(pred)),
             'val_probe_mae_ipc':float(np.abs(actual-np.expm1(pred)).mean()),
             'val_direct_head_r2_raw':r2(actual,raw_direct),'n_validation':len(actual),
             'n_probe_train':len(indices),'stage':stage,'test_evaluated':False}
    if not all(math.isfinite(metrics[k]) for k in metrics if k.startswith('val_')):
        raise RuntimeError('Nonfinite validation result')
    np.savez_compressed(run/(stage+'_validation_predictions.npz'),ipc=actual,
        probe_log1p=pred,direct_ipc=raw_direct,ridge_coef=coef,ridge_intercept=ym-xm@coef)
    write_json(run/(stage+'_metrics.json'),metrics)
    print('VALIDATION '+json.dumps(metrics),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['screen','full'],required=True)
    parser.add_argument('--trial',type=int);parser.add_argument('--slot',type=int)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.stage=='full':
        if args.slot is None:raise ValueError('Promotion slot required')
        trial=json.loads((BASE/'promotions.json').read_text())['promotions'][args.slot]['trial_id']
    else:trial=args.trial
    config=trial_config(trial)
    inputs=load_inputs(config['model'])
    source,assembly,labels,_,_,_=inputs
    raw=np.asarray(labels,dtype=np.float64)
    transformed=raw if config['target']=='raw' else np.log1p(raw)
    ratio=float(raw.var()/transformed.var())
    config.update(effective_lambda=config['base_weight']*ratio,variance_calibration=ratio,
        raw_label_variance=float(raw.var()),target_label_variance=float(transformed.var()),
        temperature=.05,label_smoothing=.05,batch_size=256,seed=42,
        scheduler_total_steps=10000,warmup=200,ridge_alpha=1,test_evaluated=False,
        validation_manifest_sha256=hashlib.sha256((BASE/'validation_data/manifest.json').read_bytes()).hexdigest())
    if args.preflight:
        load_encoder_class();print('PREFLIGHT_OK '+json.dumps(config),flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('Slurm GPU allocation required')
    run=BASE/'trials'/f'trial_{trial:02d}';run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    Encoder=load_encoder_class()
    src=Encoder(16384).cuda();asm=Encoder(ASSEMBLY[config['model']][1]).cuda();head=torch.nn.Linear(128,1).cuda()
    params=list(src.parameters())+list(asm.parameters())+list(head.parameters())
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    def factor(step):return step/200 if step<200 else max(.1,.5*(1+math.cos(math.pi*(step-200)/9800)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    generator=torch.Generator().manual_seed(42);permutation=torch.randperm(len(source),generator=generator).numpy()
    step=0;position=0;previous=0.
    checkpoint=run/'checkpoint.pt'
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if c['trial_id']!=trial:raise RuntimeError('Checkpoint trial mismatch')
        src.load_state_dict(c['src_enc']);asm.load_state_dict(c['asm_enc']);head.load_state_dict(c['hw_head'])
        opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
        permutation=c['permutation'].numpy();position=c['position'];step=c['step'];previous=c['training_seconds']
        generator.set_state(c['sampler_rng']);torch.set_rng_state(c['cpu_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
    target_steps=PLAN['screen_steps'] if args.stage=='screen' else PLAN['full_steps']
    config.update(stage=args.stage,target_steps=target_steps,torch=torch.__version__,precision='CUDA bf16',
                  parameters=sum(p.numel() for p in params))
    write_json(run/(args.stage+'_config.json'),config);print(json.dumps(config),flush=True)
    src.train();asm.train();head.train();start=time.monotonic();start_step=step
    def save(status,loss):
        torch.cuda.synchronize();elapsed=previous+time.monotonic()-start
        state={'trial_id':trial,'src_enc':src.state_dict(),'asm_enc':asm.state_dict(),'hw_head':head.state_dict(),
            'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'permutation':torch.from_numpy(permutation.copy()),
            'position':position,'step':step,'sampler_rng':generator.get_state(),'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'training_seconds':elapsed}
        tmp=run/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(checkpoint)
        rate=(time.monotonic()-start)/max(1,step-start_step)
        write_json(run/'status.json',{'stage':args.stage,'status':status,'step':step,'target_steps':target_steps,
            'loss':loss,'seconds_per_step':rate,'remaining_seconds':max(0,target_steps-step)*rate})
    last=0.
    while step<target_steps:
        if position+256>len(permutation):permutation=torch.randperm(len(source),generator=generator).numpy();position=0
        ix=permutation[position:position+256];position+=256
        ids=torch.from_numpy(np.array(source[ix],dtype=np.int64)).cuda()
        asm_ids=torch.from_numpy(np.array(assembly[ix],dtype=np.int64)).cuda()
        ipc=torch.from_numpy(np.array(labels[ix],dtype=np.float32)).cuda()
        regression_target=ipc if config['target']=='raw' else torch.log1p(ipc)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            z=src(ids);a=asm(asm_ids)
            scores=F.normalize(z,dim=-1)@F.normalize(a,dim=-1).T/.05
            truth=torch.arange(len(z),device='cuda')
            contrast=(F.cross_entropy(scores,truth,label_smoothing=.05)+F.cross_entropy(scores.T,truth,label_smoothing=.05))/2
            mse=F.mse_loss(head(z).squeeze(-1),regression_target)
            loss=contrast+config['effective_lambda']*mse
        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step();scheduler.step();step+=1;last=float(loss.detach().cpu())
        if step==1 or step%250==0:print(f'trial={trial} step={step}/{target_steps} loss={last:.6f}',flush=True)
        if step==1 or step%500==0 or step==target_steps:save('training',last)
    save('evaluating_validation',last)
    evaluate(src,head,inputs,config['target'],run,args.stage)
    save('complete',last)

if __name__=='__main__':main()
