"""Compare actual cached gradients and RNG state against ordinary full-graph autograd."""
import json,torch
import torch.nn.functional as F
from gradient_cache import backward

def main():
 torch.set_num_threads(2);torch.manual_seed(19);results=[]
 for joint in [False,True]:
  src=torch.nn.Sequential(torch.nn.Linear(12,16),torch.nn.Dropout(.25),torch.nn.Linear(16,12));asm=torch.nn.Sequential(torch.nn.Linear(12,16),torch.nn.Dropout(.25),torch.nn.Linear(16,8)) if joint else None
  head=torch.nn.Linear(12,1);proj=torch.nn.Linear(12,8,bias=False) if joint else None
  params=list(src.parameters())+list(head.parameters())+(list(asm.parameters())+list(proj.parameters()) if joint else [])
  batches=[(torch.randn(8,12),torch.randn(8,12) if joint else None) for _ in range(4)];y=torch.randn(32);weight=.17 if joint else 0;start=torch.get_rng_state();zs=[];aa=[]
  for x,a in batches:
   zs.append(src(x))
   if joint:aa.append(asm(a))
  z=torch.cat(zs);loss=F.mse_loss(head(z).squeeze(-1),y)
  if joint:
   score=F.normalize(proj(z).float(),dim=-1)@F.normalize(torch.cat(aa).float(),dim=-1).T/.05;truth=torch.arange(32)
   loss=loss+weight*(F.cross_entropy(score,truth,label_smoothing=.05)+F.cross_entropy(score.T,truth,label_smoothing=.05))/2
  loss.backward();reference=[p.grad.clone() for p in params];end=torch.get_rng_state()
  for p in params:p.grad=None
  torch.set_rng_state(start);m=backward(src,asm,head,proj,batches,y,weight,amp=False)
  for p,r in zip(params,reference):assert torch.allclose(p.grad,r,rtol=1e-4,atol=2e-6),float((p.grad-r).abs().max())
  assert torch.equal(torch.get_rng_state(),end),'Replay changed RNG stream'
  results.append({'joint':joint,'max_gradient_error':max(float((p.grad-r).abs().max()) for p,r in zip(params,reference)),'dropout_rng_equal':True})
 print('GRADIENT_EQUIVALENCE_PASS',json.dumps(results))
if __name__=='__main__':main()
