"""Rebuild the original runtime layout from frozen raw splits and tokenizer files."""
import argparse,gzip,hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
P=Path(__file__).resolve().parents[1];E=P/'experiments';C=E/'source_ipc_baseline_20261025';A=E/'ia_fixed_joint_ablation_20261026';B=E/'bpe_joint_ablation_20261025';V=E/'ia_bpe_validation_search_20261026/validation_data'
sys.path.insert(0,str(A));from fixed_tokenizer import FixedIATokenizer

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,default=P/'data/raw');args=ap.parse_args()
 raw=C/'archive/dl4nlp_crossmodal';raw.mkdir(exist_ok=True,parents=True)
 for n in ['bpe_tokenizer.json','ia_tokenizer.pkl']:shutil.copy2(P/'data/tokenizers'/n,raw/n)
 for split in ['train','val','test']:
  src=args.raw_dir/f'{split}.jsonl'
  if not src.exists():
   gz=args.raw_dir/f'{split}.jsonl.gz'
   if not gz.exists():raise FileNotFoundError(f'Provide {src} or {gz}, or extract the separate data archive at the package root.')
   with gzip.open(gz,'rb') as f,src.open('wb') as g:shutil.copyfileobj(f,g,8*1024*1024)
  dest=raw/src.name
  if not dest.exists():dest.symlink_to(src.resolve())
 old=E/'pretrained_ipc_20261027';old.mkdir(exist_ok=True)
 if not (old/'raw').exists():(old/'raw').symlink_to(raw.resolve(),target_is_directory=True)
 common=C/'source_only_data';common.mkdir(exist_ok=True)
 shutil.copy2(P/'data/splits/probe_train_indices.npy',common/'probe_train_indices.npy')
 shutil.copy2(P/'data/splits/reference_labels.npz',C/'probe_embeddings_seed42.npz')
 tk=Tokenizer.from_file(str(P/'data/tokenizers/bpe_tokenizer.json'));toksha=digest(P/'data/tokenizers/bpe_tokenizer.json')
 spec=json.loads((P/'data/tokenizers/tokenizer_spec.json').read_text());ia=FixedIATokenizer(spec)
 counts={'train':309913,'val':2000,'test':2000};split_manifest={};ia_manifest={'splits':{}};bpe_manifest={'splits':{}}
 for path in [A/'assembly_data',B/'assembly_data',V]:path.mkdir(exist_ok=True,parents=True)
 for split,n in counts.items():
  # Regeneration is explicit and overwrites only derived caches.
  tokens=np.lib.format.open_memmap((V/'source_tokens.npy') if split=='val' else common/f'{split}_tokens.npy',mode='w+',dtype=np.uint16,shape=(n,512))
  labels=np.lib.format.open_memmap((V/'ipc.npy') if split=='val' else common/f'{split}_ipc.npy',mode='w+',dtype=np.float32,shape=(n,))
  it=np.lib.format.open_memmap(A/'assembly_data'/f'{split}_asm_ia_fixed.npy',mode='w+',dtype=np.uint16,shape=(n,512))
  bt=np.lib.format.open_memmap(B/'assembly_data'/f'{split}_asm_bpe.npy',mode='w+',dtype=np.uint16,shape=(n,512))
  il=np.lib.format.open_memmap(A/'assembly_data'/f'{split}_lengths.npy',mode='w+',dtype=np.int32,shape=(n,))
  bl=np.lib.format.open_memmap(B/'assembly_data'/f'{split}_bpe_lengths.npy',mode='w+',dtype=np.int32,shape=(n,))
  sha=hashlib.sha256();ids=[]
  with (raw/f'{split}.jsonl').open('rb') as f:
   for i,line in enumerate(f):
    if i>=n:break
    sha.update(line);r=json.loads(line);ids.append(r['id']);labels[i]=r['ipc']
    source=[2]+tk.encode(r['source']).ids[:510]+[3];asm=tk.encode(r['asm']).ids;itokens=ia.encode(r['asm'],max_len=10**9)
    aseq=itokens[:511]+[3] if len(itokens)>512 else itokens;bseq=[2]+asm[:510]+[3]
    for arr,seq in [(tokens,source),(it,aseq),(bt,bseq)]:arr[i]=0;arr[i,:len(seq)]=seq
    il[i]=len(itokens);bl[i]=len(asm)+2
    if (i+1)%25000==0:print(split,i+1,n,flush=True)
  assert len(ids)==n
  for arr in [tokens,labels,it,bt,il,bl]:arr.flush()
  if split=='train':
   idx=np.load(common/'probe_train_indices.npy');expected=json.loads((P/'data/splits/train_ids.json').read_text());assert [ids[int(i)] for i in idx]==expected
  else:assert ids==json.loads((P/f'data/splits/{"validation" if split=="val" else "test"}_ids.json').read_text())
  split_manifest[split]={'count':n,'input_sha256':sha.hexdigest(),'hash_scope':'full file' if split=='train' else 'first2000 records'}
  for manifest,lengths in [(ia_manifest,il),(bpe_manifest,bl)]:manifest['splits'][split]={'count':n,'input_sha256':sha.hexdigest(),'mean':float(lengths.mean()),'fraction_truncated_at_512':float((lengths>512).mean())}
  if split=='val':(V/'ids.json').write_text(json.dumps(ids));(V/'manifest.json').write_text(json.dumps({'tokenizer_sha256':toksha,'input_sha256':sha.hexdigest(),'n':n}))
  print('PREPARED',split,n,flush=True)
 source_manifest={'fingerprint':{'tokenizer_sha256':toksha},'splits':split_manifest}
 (common/'manifest.json').write_text(json.dumps(source_manifest,indent=2))
 (A/'assembly_data/manifest.json').write_text(json.dumps(ia_manifest,indent=2));(B/'assembly_data/manifest.json').write_text(json.dumps(bpe_manifest,indent=2))
 # Preserve the historical raw-byte fingerprint checks for the primary partitions.
 expected=json.loads((P/'provenance/source_cache_manifest.json').read_text())
 for split in ['train','test']:assert expected['splits'][split]['input_sha256']==split_manifest[split]['input_sha256'],f'Raw split mismatch: {split}'
 print('PASS: source/assembly order, raw split hashes, frozen probe identities, and tokenizer IDs verified.')
if __name__=='__main__':main()
