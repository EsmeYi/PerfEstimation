"""Same-device source-encoder/linear-readout timing; separate host tokenization."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from evaluate import P,load,test_tokens,tokenizer
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['compact_ia_joint','codebert_ipc','unixcoder_ipc']);ap.add_argument('--repetitions',type=int,default=100);ap.add_argument('--warmup',type=int,default=20);ap.add_argument('--output',type=Path,default=P/'generated/efficiency.json');a=ap.parse_args();torch.set_num_threads(4)
 if not torch.cuda.is_available():raise SystemExit('This synchronized GPU benchmark requires CUDA.')
 report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'precision':'float32','sequence_length':512,'scope':'encoder+dimension-matched scalar linear readout; excludes tokenization and host/device transfers','weights':'final available checkpoints; archived reference measurements used identical architectures with earlier IPC-only pretrained weights','models':{}}
 for model in a.models:
  net,encode=load(model,42,'cuda');data,_,_=test_tokens(model);dim=128 if model.startswith('compact') else 768;w=torch.ones(dim,device='cuda')/dim;b=torch.zeros((),device='cuda');timing={}
  with torch.inference_mode():
   for batch in [1,16]:
    xx=[torch.tensor(np.array(data[i:i+batch],dtype=np.int64),device='cuda') for i in range(0,batch*16,batch)]
    def predict(x):return encode(x)@w+b
    for i in range(a.warmup):predict(xx[i%len(xx)])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for i in range(a.repetitions):
     start=time.perf_counter();out=predict(xx[i%len(xx)]);torch.cuda.synchronize();times.append(time.perf_counter()-start)
    timing[str(batch)]={'mean_ms':float(np.mean(times)*1000),'p50_ms':float(np.median(times)*1000),'p95_ms':float(np.percentile(times,95)*1000),'functions_per_second':float(batch/np.mean(times)),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    del xx,out
  tokenize,_=tokenizer(model);texts=[]
  with (P/'data/raw/test.jsonl').open() as f:
   for line in f:
    texts.append(json.loads(line)['source'])
    if len(texts)==256:break
  for text in texts[:10]:tokenize(text)
  times=[]
  for text in texts:
   start=time.perf_counter();tokenize(text);times.append(time.perf_counter()-start)
  report['models'][model]={'parameters':sum(p.numel() for p in net.parameters()),'gpu':timing,'tokenization_mean_ms':float(np.mean(times)*1000)}
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(model,json.dumps(timing),flush=True)
  del net,encode,w,b;torch.cuda.empty_cache()
if __name__=='__main__':main()
