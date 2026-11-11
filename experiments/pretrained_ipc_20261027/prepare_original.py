"""Source-only pretrained baselines; validation-only LR selection."""
import os
os.environ.setdefault('HF_HOME',str(__import__('pathlib').Path(__file__).resolve().parent/'hf'))
import argparse,csv,json,math,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import RobertaModel,RobertaTokenizer
B=Path(__file__).resolve().parent
R=B.parent; C=R/'source_ipc_baseline_20261025'
MODELS={'codebert':'microsoft/codebert-base','unixcoder':'microsoft/unixcoder-base'}
LRS=[1e-5,2e-5,5e-5]
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def prepare(name, tokenize_only=False):
 d=B/'data'/name;d.mkdir(parents=True,exist_ok=True)
 metadata=R.parent/'data/pretrained_metadata'/name
 provenance=R.parent/'provenance/pretrained_initialization'/name/'pretrained_config.json'
 commit=json.loads(provenance.read_text()).get('commit') if provenance.exists() else None
 tok=RobertaTokenizer.from_pretrained(metadata,local_files_only=True)
 if not tokenize_only:
  net=RobertaModel.from_pretrained(MODELS[name],revision=commit);net.save_pretrained(B/'models'/name,safe_serialization=True);tok.save_pretrained(B/'models'/name)
  dump(d/'pretrained_config.json',{'model':MODELS[name],'commit':commit,'tokenizer':'archived native','representation':'masked mean 768'})
  del net
 for split,n in [('train',309913),('val',2000),('test',2000)]:
  out=d/f'{split}.npy'
  if out.exists():continue
  arr=np.lib.format.open_memmap(d/f'{split}.tmp.npy',mode='w+',dtype=np.int32,shape=(n,512));arr[:]=tok.pad_token_id
  labels=[];ids=[];h=hashlib.sha256()
  with (B/'raw'/f'{split}.jsonl').open('rb') as f:
   for line in f:
    if not line.strip():continue
    h.update(line);r=json.loads(line);text=r['source']
    # Official UniXcoder encoder-only sequence format; CodeBERT standard BOS/EOS.
    if name=='unixcoder':
     tokens=[tok.cls_token,'<encoder-only>',tok.sep_token]+tok.tokenize(text)[:508]+[tok.sep_token]
     seq=tok.convert_tokens_to_ids(tokens)
    else:seq=tok.encode(text,truncation=True,max_length=512)
    arr[len(ids),:len(seq)]=seq;labels.append(r['ipc']);ids.append(r['id'])
    if len(ids)==n:break
  assert len(ids)==n
  if split=='train':
   expected_manifest=json.loads((C/'source_only_data/manifest.json').read_text())
   assert h.hexdigest()==expected_manifest['splits']['train']['input_sha256']
  elif split=='val':assert ids==json.loads((R/'ia_bpe_validation_search_20261026/validation_data/ids.json').read_text())
  else:assert ids==np.load(C/'probe_embeddings_seed42.npz')['test_ids'].tolist()
  arr.flush();del arr;(d/f'{split}.tmp.npy').replace(out)
  np.save(d/f'{split}_ipc.npy',np.array(labels,dtype=np.float32));dump(d/f'{split}_ids.json',ids);dump(d/f'{split}_manifest.json',{'n':n,'input_sha256':h.hexdigest(),'model':MODELS[name],'max_length':512})
  expected=np.load(C/'source_only_data'/('train_ipc.npy' if split=='train' else 'test_ipc.npy')) if split!='val' else np.load(R/'ia_bpe_validation_search_20261026/validation_data/ipc.npy')
  assert np.array_equal(np.asarray(labels,dtype=np.float32),expected)
 print('PREPARED',name,flush=True)
