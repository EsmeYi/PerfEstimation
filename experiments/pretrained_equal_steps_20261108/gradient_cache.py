"""Full logical-batch gradients with exact dropout replay, for IPC-only and joint training."""
import torch
import torch.nn.functional as F

def backward(source_forward,assembly_forward,head,projection,batches,target,weight,amp=True):
 device=target.device;cuda=device.type=='cuda';chunks=[];source=[];assembly=[]
 for sid,aid in batches:
  rng=(torch.get_rng_state(),torch.cuda.get_rng_state(device) if cuda else None)
  with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
   z=source_forward(sid);a=assembly_forward(aid).float() if assembly_forward is not None else None
  chunks.append((sid,aid,rng));source.append(z)
  if a is not None:assembly.append(a)
 z=torch.cat(source).detach().requires_grad_();a=torch.cat(assembly).detach().requires_grad_() if assembly else None
 with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
  mse=F.mse_loss(head(z).squeeze(-1),target)
  projected=F.normalize(projection(z).float(),dim=-1) if a is not None else None
 if a is not None:
  scores=projected@F.normalize(a,dim=-1).T/.05;truth=torch.arange(len(target),device=device)
  align=(F.cross_entropy(scores,truth,label_smoothing=.05)+F.cross_entropy(scores.T,truth,label_smoothing=.05))/2
 else:align=mse.new_zeros(())
 loss=mse+weight*align
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
 loss.backward();zg=z.grad.detach();ag=a.grad.detach() if a is not None else None;offset=0
 for sid,aid,(cpu_rng,cuda_rng) in chunks:
  with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if cuda else []):
   torch.set_rng_state(cpu_rng)
   if cuda:torch.cuda.set_rng_state(cuda_rng,device)
   with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
    zz=source_forward(sid);aa=assembly_forward(aid).float() if assembly_forward is not None else None
   n=len(sid)
   if aa is None:zz.backward(zg[offset:offset+n])
   else:torch.autograd.backward([zz,aa],[zg[offset:offset+n],ag[offset:offset+n]])
  offset+=len(sid)
 return {'loss':float(loss.detach()),'ipc_mse':float(mse.detach()),'alignment':float(align.detach())}
