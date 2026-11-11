"""Cache transparent features and original IA tokens; never use measured target-derived fields."""
import csv,hashlib,importlib.util,json,re,sys,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=Path('/proj/berzelius-2026-145/users/x_liryi')
COMMON=ROOT/'source_ipc_baseline_20261025'
RAW=ROOT/"pretrained_ipc_20261027/raw"
IDX=np.load(COMMON/'source_only_data/probe_train_indices.npy');selected=set(IDX.tolist())
ref=np.load(COMMON/'probe_embeddings_seed42.npz',allow_pickle=False)
val_ids=json.loads((ROOT/'ia_bpe_validation_search_20261026/validation_data/ids.json').read_text())
with (ROOT/'ia_fixed_joint_ablation_20261026/tokenizer_spec.json').open() as f:spec=json.load(f)
p=ROOT/'ia_fixed_joint_ablation_20261026/fixed_tokenizer.py'
s=importlib.util.spec_from_file_location('fixed',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);tok=m.FixedIATokenizer(spec)
COMMENT=re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',re.S)
INSTRUCTION=re.compile(r'^\s*([A-Za-z][\w.]*)\s*(.*)$')
GROUPS={
 'move':r'^(?:v?mov|lea)', 'load_store':r'^(?:v?mov|push|pop)', 'branch':r'^j',
 'call':r'^call', 'return':r'^ret', 'compare':r'^(?:cmp|test|u?comi|v?u?comi)',
 'add_sub':r'^(?:v?(?:p?add|p?sub)|inc|dec|adc|sbb)', 'multiply':r'^(?:i?mul|v?(?:p?mul|fm))',
 'divide':r'^(?:i?div|v?div|sqrt|v?sqrt)', 'logic':r'^(?:and|or|xor|not|v?p(?:and|or|xor))',
 'shift':r'^(?:sh[lr]|sa[lr]|ro[lr]|v?ps[lr])', 'conditional_move':r'^cmov',
 'setcc':r'^set', 'vector':r'^v', 'scalar_fp':r'^(?:add|sub|mul|div|sqrt|u?comi)[sp][sd]$',
 'stack':r'^(?:push|pop|leave|enter)', 'convert':r'^(?:v?cvt)', 'nop':r'^nop'}
GROUPS={k:re.compile(v) for k,v in GROUPS.items()}
SRC_NAMES=['src_lines','src_chars','src_identifiers','src_numbers','src_loops','src_branches','src_calls','src_pointer_stars','src_arithmetic','src_array_access','src_return']
ASM_NAMES=['asm_instructions','asm_chars','asm_memory_operands','asm_register_operands','asm_immediates']+['asm_'+x for x in GROUPS]+['asm_fraction_'+x for x in GROUPS]
NAMES=SRC_NAMES+ASM_NAMES

def features(row):
 text=COMMENT.sub(' ',row['source']);ident=re.findall(r'\b[A-Za-z_]\w*\b',text)
 src=[sum(bool(x.strip()) for x in text.splitlines()),len(text),len(ident),len(re.findall(r'\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)\b',text)),sum(x in {'for','while','do'} for x in ident),sum(x in {'if','else','switch','case'} for x in ident),len(re.findall(r'\b(?!(?:if|for|while|switch|sizeof)\b)[A-Za-z_]\w*\s*\(',text)),text.count('*'),len(re.findall(r'[+*/%\-]',text)),text.count('['),ident.count('return')]
 ins=[]
 for line in row['asm'].splitlines():
  line=line.split('#',1)[0].strip()
  if not line or line.startswith('.') or ':' in line:continue
  m=INSTRUCTION.fullmatch(line)
  if m:ins.append((m[1].lower(),m[2]))
 ops=' '.join(x[1] for x in ins);n=len(ins)
 counts=[sum(bool(p.match(op)) for op,_ in ins) for p in GROUPS.values()]
 asm=[n,len(ops),ops.count('('),len(re.findall(r'%[A-Za-z0-9]+',ops)),ops.count('$')]+counts+[x/max(n,1) for x in counts]
 return src+asm

def main():
 manifest={'raw':str(RAW),'feature_names':NAMES,'excluded_fields':['ipc','total_cycles','block_rthroughput','ir'],'training_probe_indices':IDX.tolist(),'features_note':'AT&T instruction parser ignores labels/directives/comments; memory operands use parentheses; opcode categories include suffixes; source comments and literals removed.'}
 for split in ['train','val','test']:
  out=BASE/f'{split}_cache.npz'
  if out.exists():print('CACHE_EXISTS',split,flush=True);continue
  xs=[];ys=[];ids=[];ts=[];ti=[];sha=hashlib.sha256();start=time.monotonic()
  with (RAW/f'{split}.jsonl').open('rb') as f:
   for i,line in enumerate(f):
    sha.update(line);row=json.loads(line)
    xs.append(features(row));ys.append(float(row['ipc']));ids.append(row['id'])
    if split!='train' or i in selected:
     seq=tok.encode(row['asm']);pad=np.zeros(512,np.uint16);pad[:len(seq)]=seq;ts.append(pad);ti.append(i)
    if (i+1)%50000==0:print(split,i+1,round(time.monotonic()-start,1),flush=True)
    if split!='train' and len(ids)==2000:break
  if split=='train':
   assert len(ids)==309913
   assert np.array(ids)[IDX].tolist()==ref['train_ids'].tolist()
   positions={ix:j for j,ix in enumerate(ti)};ts=[ts[positions[int(ix)]] for ix in IDX];ti=IDX.tolist()
  elif split=='val':assert ids==val_ids
  else:assert ids==ref['test_ids'].tolist()
  np.savez_compressed(out,x=np.asarray(xs,np.float64),ipc=np.array(ys,np.float64),ids=np.array(ids),asm_tokens=np.array(ts),token_indices=np.array(ti))
  manifest[split]={'n':len(ids),'sha256':sha.hexdigest(),'hash_scope':'full file' if split=='train' else 'first2000 records','n_asm_tokens':len(ts),'seconds':time.monotonic()-start}
  print('PREPARED',split,manifest[split],flush=True)
 (BASE/'feature_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
