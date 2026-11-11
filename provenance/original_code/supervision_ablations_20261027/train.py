"""Fixed validation-selected configuration; independent seed confirmation."""
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
HPO=ROOT/'ia_bpe_validation_search_20261026'
PLAN=json.loads((HPO/'search_plan.json').read_text())
FROZEN=json.loads((ROOT/'ia_bpe_tuned_multiseed_20261027/frozen_configs.json').read_text())

def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

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
    validation=np.load(HPO/'validation_data/source_tokens.npy',mmap_mode='r')
    validation_labels=np.load(HPO/'validation_data/ipc.npy',allow_pickle=False)
    if source.shape!=asm.shape or source.shape[1]!=512:raise ValueError('Wrong paired shapes')
    if len(set(probe_indices.tolist()))!=5000:raise ValueError('Invalid training probe selection')
    source_manifest=json.loads((data/'manifest.json').read_text())
    assembly_manifest=json.loads((ASSEMBLY[model_name][0].parent/'manifest.json').read_text())
    val_manifest=json.loads((HPO/'validation_data/manifest.json').read_text())
    if source_manifest['splits']['train']['input_sha256']!=assembly_manifest['splits']['train']['input_sha256']:
        raise ValueError('Assembly/source training data mismatch')
    if source_manifest['fingerprint']['tokenizer_sha256']!=val_manifest['tokenizer_sha256']:
        raise ValueError('Validation/source tokenizer mismatch')
    return source,asm,labels,probe_indices,validation,validation_labels

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['source_only','alignment_only'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();args.stage='full'
    config=dict(FROZEN['models']['IA']);config.update(seed=args.seed,ablation=args.model,ipc_supervision=args.model=='source_only',assembly_alignment=args.model=='alignment_only')
    trial=config['trial_id']
    inputs=load_inputs(config['model'])
    source,assembly,labels,_,_,_=inputs
    raw=np.asarray(labels,dtype=np.float64)
    transformed=raw if config['target']=='raw' else np.log1p(raw)
    ratio=float(raw.var()/transformed.var())
    config.update(effective_lambda=config['base_weight']*ratio,variance_calibration=ratio,
        raw_label_variance=float(raw.var()),target_label_variance=float(transformed.var()),
        temperature=.05,label_smoothing=.05,batch_size=256,seed=args.seed,
        scheduler_total_steps=10000,warmup=200,ridge_alpha=1,test_evaluated=False,
        validation_manifest_sha256=hashlib.sha256((HPO/'validation_data/manifest.json').read_bytes()).hexdigest())
    config.update(active_ipc_weight=config['effective_lambda'] if args.model=='source_only' else 0.,active_alignment_weight=1. if args.model=='alignment_only' else 0.)
    if args.preflight:
        load_encoder_class();print('PREFLIGHT_OK '+json.dumps(config),flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('Slurm GPU allocation required')
    run=BASE/'runs'/f'{args.model}_seed{args.seed}';run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=False
    Encoder=load_encoder_class()
    src=Encoder(16384).cuda()
    asm=Encoder(254).cuda() if args.model=='alignment_only' else None
    head=torch.nn.Linear(128,1).cuda() if args.model=='source_only' else None
    params=list(src.parameters())+(list(asm.parameters()) if asm is not None else [])+(list(head.parameters()) if head is not None else [])
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    def factor(step):return step/200 if step<200 else max(.1,.5*(1+math.cos(math.pi*(step-200)/9800)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    generator=torch.Generator().manual_seed(args.seed);permutation=torch.randperm(len(source),generator=generator).numpy()
    step=0;position=0;previous=0.
    checkpoint=run/'checkpoint.pt'
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if c['trial_id']!=trial or c['seed']!=args.seed:raise RuntimeError('Checkpoint trial mismatch')
        src.load_state_dict(c['src_enc'])
        if c['ablation']!=args.model:raise RuntimeError('Wrong ablation checkpoint')
        if asm is not None:asm.load_state_dict(c['asm_enc'])
        if head is not None:head.load_state_dict(c['hw_head'])
        opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler'])
        permutation=c['permutation'].numpy();position=c['position'];step=c['step'];previous=c['training_seconds']
        generator.set_state(c['sampler_rng']);torch.set_rng_state(c['cpu_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
    target_steps=FROZEN['models']['IA']['target_steps']
    config.update(stage=args.stage,target_steps=target_steps,torch=torch.__version__,precision='CUDA bf16',
                  parameters=sum(p.numel() for p in params))
    write_json(run/(args.stage+'_config.json'),config);print(json.dumps(config),flush=True)
    src.train()
    if asm is not None:asm.train()
    if head is not None:head.train()
    start=time.monotonic();start_step=step
    def save(status,loss):
        torch.cuda.synchronize();elapsed=previous+time.monotonic()-start
        state={'trial_id':trial,'seed':args.seed,'src_enc':src.state_dict(),'ablation':args.model,'asm_enc':asm.state_dict() if asm is not None else None,'hw_head':head.state_dict() if head is not None else None,
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
        asm_ids=torch.from_numpy(np.array(assembly[ix],dtype=np.int64)).cuda() if asm is not None else None
        regression_target=torch.log1p(torch.from_numpy(np.array(labels[ix],dtype=np.float32)).cuda()) if head is not None else None
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            z=src(ids)
            if asm is not None:
                a=asm(asm_ids)
                scores=F.normalize(z,dim=-1)@F.normalize(a,dim=-1).T/.05
                truth=torch.arange(len(z),device='cuda')
                loss=(F.cross_entropy(scores,truth,label_smoothing=.05)+F.cross_entropy(scores.T,truth,label_smoothing=.05))/2
            else:
                loss=config['effective_lambda']*F.mse_loss(head(z).squeeze(-1),regression_target)
        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step();scheduler.step();step+=1;last=float(loss.detach().cpu())
        if step==1 or step%250==0:print(f'trial={trial} step={step}/{target_steps} loss={last:.6f}',flush=True)
        if step==1 or step%500==0 or step==target_steps:save('training',last)
    save('trained',last)
    print('TRAINING_COMPLETE; fixed validation/test evaluation follows in separate process',flush=True)

if __name__=='__main__':main()
