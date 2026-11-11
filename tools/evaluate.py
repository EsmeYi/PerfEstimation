"""Portable frozen-model evaluation, with archived Ridge coefficients and no training."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,sys
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import RobertaConfig,RobertaModel,RobertaTokenizer
from metrics import score
P=Path(__file__).resolve().parents[1]
MODELS=['compact_ipc','compact_bpe_joint','compact_ia_joint','codebert_ipc','codebert_joint','unixcoder_ipc','unixcoder_joint']
def checkpoint(model,seed):
 if model.startswith('compact'):
  family={'compact_ipc':'source_only','compact_bpe_joint':'BPE','compact_ia_joint':'IA'}[model];return P/f'checkpoints/{family}_seed{seed}.pt'
 return P/f'checkpoints/pretrained/{model}_seed{seed}.pt'
def load(model,seed,device,checkpoint_override=None):
 path=checkpoint_override or checkpoint(model,seed)
 if not path.exists():raise FileNotFoundError(f'{path}: run tools/download_artifacts.py --model {model} --seed {seed}')
 state=torch.load(path,map_location='cpu',weights_only=True)
 if model.startswith('compact'):
  folder=P/'experiments/source_ipc_baseline_20261025/archive/Experiment';sys.path.insert(0,str(folder));spec=importlib.util.spec_from_file_location('compact_encoder',folder/'final_train_seeded.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  net=module.Encoder(16384);net.load_state_dict(state['src_enc']);net=net.to(device).eval()
  def encode(x):return net(x)
 else:
  name=model.split('_')[0];net=RobertaModel(RobertaConfig.from_pretrained(P/f'data/pretrained_metadata/{name}',local_files_only=True),add_pooling_layer=False);net.load_state_dict(state['encoder']);net=net.to(device).eval()
  def encode(x):
   mask=x.ne(1);z=net(input_ids=x,attention_mask=mask).last_hidden_state
   return (z.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
 del state
 return net,encode
def tokenizer(model):
 name='compact' if model.startswith('compact') else model.split('_')[0]
 tk=Tokenizer.from_file(str(P/'data/tokenizers/bpe_tokenizer.json')) if name=='compact' else RobertaTokenizer.from_pretrained(P/f'data/pretrained_metadata/{name}',local_files_only=True)
 def tokenize(text):
  if name=='compact':return [2]+tk.encode(text).ids[:510]+[3]
  if name=='unixcoder':return tk.convert_tokens_to_ids([tk.cls_token,'<encoder-only>',tk.sep_token]+tk.tokenize(text)[:508]+[tk.sep_token])
  return tk.encode(text,truncation=True,max_length=512)
 return tokenize,0 if name=='compact' else 1
def test_tokens(model,raw_dir=None):
 name='compact' if model.startswith('compact') else model.split('_')[0];cache=P/'generated/full_cache';cache.mkdir(parents=True,exist_ok=True);path=cache/f'{name}.npy';source=(raw_dir or P/'data/raw')/'test.jsonl'
 if not source.exists():raise FileNotFoundError(f'{source}: download the data archive first.')
 rows=[json.loads(l) for l in source.open()];assert len(rows)==17301
 ids=[r['id'] for r in rows];expected=json.loads((P/'results/expanded/dedup_masks.json').read_text());assert ids==[r['id'] for r in expected]
 labels=np.array([r['ipc'] for r in rows],dtype=np.float64)
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  return h.hexdigest()
 tkpaths=[P/'data/tokenizers/bpe_tokenizer.json'] if name=='compact' else list((P/f'data/pretrained_metadata/{name}').glob('*.json'))
 fingerprint={'raw_sha256':digest(source),'tokenizer':{p.name:digest(p) for p in sorted(tkpaths)},'max_length':512,'model':name}
 mp=path.with_suffix('.json')
 if path.exists() and mp.exists() and json.loads(mp.read_text())==fingerprint:
  data=np.load(path,mmap_mode='r');assert data.shape==(17301,512);return data,ids,labels
 temporary=path.with_name(f'{name}.{os.getpid()}.tmp.npy')
 tokenize,pad=tokenizer(model);data=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.int32,shape=(len(rows),512));data[:]=pad
 for i,row in enumerate(rows):
  seq=tokenize(row['source']);data[i,:len(seq)]=seq
 data.flush();del data;temporary.replace(path)
 mt=mp.with_name(mp.name+f'.{os.getpid()}.tmp');mt.write_text(json.dumps(fingerprint,indent=2));mt.replace(mp)
 return np.load(path,mmap_mode='r'),ids,labels
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--seed',type=int,choices=[42,43,44],default=42);ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--scope',choices=['original','full','filtered'],default='full');ap.add_argument('--limit',type=int);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--raw-dir',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args();torch.set_num_threads(4)
 net,encode=load(a.model,a.seed,a.device,a.checkpoint);data,ids,labels=test_tokens(a.model,a.raw_dir);record=P/f'results/expanded/{a.model}_seed{a.seed}';rr=np.load(record/'readout.npz')
 count=2000 if a.scope=='original' else len(ids)
 if a.limit:count=min(count,a.limit)
 chunks=[]
 with torch.inference_mode():
  for start in range(0,count,a.batch_size):
   x=torch.tensor(np.array(data[start:min(count,start+a.batch_size)],dtype=np.int64),device=a.device);chunks.append(encode(x).cpu().numpy())
 z=np.concatenate(chunks).astype(np.float64);pred=(z-rr['x_mean'])@rr['coef']+rr['y_mean'];assert np.isfinite(pred).all()
 original=list(csv.DictReader((record/'test_predictions.csv').open()))
 labels[:min(count,2000)]=[float(r['ipc_llvm_mca']) for r in original[:min(count,2000)]]
 delta=float(np.max(abs(pred[:min(count,2000)]-np.array([float(r['probe_log1p']) for r in original[:min(count,2000)]]))))
 rows=[{'id':ids[i],'ipc_llvm_mca':labels[i],'probe_log1p':pred[i],'probe_ipc':float(np.expm1(pred[i]))} for i in range(count)]
 if a.scope=='filtered':
  masks={r['id']:r for r in json.loads((P/'results/expanded/dedup_masks.json').read_text())};rows=[r for r in rows if '0.9' not in masks[r['id']]['near_matches']]
 out=a.output or P/f'generated/evaluation/{a.model}_seed{a.seed}_{a.scope}';out.mkdir(parents=True,exist_ok=True)
 with (out/'test_predictions.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'model':a.model,'seed':a.seed,'scope':a.scope,'n':len(rows),'device':a.device,'torch':torch.__version__,'max_abs_original_log_prediction_difference':delta,'metrics':score(rows),'precision':'float32 source encoder; float64 archived Ridge'};(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
