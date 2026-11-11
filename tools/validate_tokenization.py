"""Re-encode test inputs and compare exact array bytes to original experimental caches."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'experiments/ia_fixed_joint_ablation_20261026'))
from fixed_tokenizer import FixedIATokenizer
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,default=P/'data/raw');a=ap.parse_args()
 tk=Tokenizer.from_file(str(P/'data/tokenizers/bpe_tokenizer.json'));ia=FixedIATokenizer(json.loads((P/'data/tokenizers/tokenizer_spec.json').read_text()));arrs={k:np.zeros((2000,512),dtype=np.uint16) for k in ['source','IA','BPE']};ids=[]
 with (a.raw_dir/'test.jsonl').open() as f:
  for i,line in enumerate(f):
   if i==2000:break
   r=json.loads(line);ids.append(r['id']);it=ia.encode(r['asm'],max_len=10**9)
   seqs=[[2]+tk.encode(r['source']).ids[:510]+[3],it[:511]+[3] if len(it)>512 else it,[2]+tk.encode(r['asm']).ids[:510]+[3]]
   for k,seq in zip(arrs,seqs):arrs[k][i,:len(seq)]=seq
 assert ids==json.loads((P/'data/splits/test_ids.json').read_text())
 refs=json.loads((P/'provenance/test_token_cache_hashes.json').read_text())
 for k,v in arrs.items():assert hashlib.sha256(v.tobytes()).hexdigest()==refs[k]['array_bytes_sha256'],k
 print('PASS: all 2000 test source/IA/BPE sequences exactly match the original cache bytes.')
if __name__=='__main__':main()
