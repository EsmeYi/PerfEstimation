"""Audit exact and near duplicates against encoder training, independent of predictions.

Prefix retrieval is complete for hashed-shingle Jaccard >= .8: an overlap
of ceil(.8*|test|) must intersect any prefix of |test|-ceil(.8*|test|)+1.
Candidate length bounds and exact intersections verify the actual Jaccard.
No approximate nearest-neighbor retrieval or model results are used.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math,re,tarfile,time
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'generated/dedup'
LEX=re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|(?:u8|[LuU])?"(?:\\.|[^"\\])*"|(?:[LuU])?\'(?:\\.|[^\'\\])*\'|[A-Za-z_][A-Za-z_0-9]*|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?(?:[eEpP][+-]?\d+)?)[uUlLfF]*|>>=|<<=|\.\.\.|->|\+\+|--|&&|\|\||<=|>=|==|!=|<<|>>|[+*/%&|^!-]=|[^\s]')
KEYWORDS=set('auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local __attribute__ __extension__ __inline__ __restrict__ __volatile__'.split())
MASK=(1<<64)-1;BASE=1000003;POWER=pow(BASE,4,1<<64);TOKEN_HASH={}
def tokenize(text):
 raw=[t for t in LEX.findall(text) if not t.startswith(('//','/*'))]
 names={};norm=[]
 for t in raw:
  if t in KEYWORDS:pass
  elif re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',t):t=names.setdefault(t,f'ID{len(names)}')
  elif '"' in t:t='STR'
  elif "'" in t:t='CHAR'
  elif t[0].isdigit():t='NUM'
  norm.append(t)
 return raw,norm
def signature(text):
 raw,norm=tokenize(text)
 exact=hashlib.sha256(' '.join(raw).encode()).hexdigest()
 normalized=hashlib.sha256(' '.join(norm).encode()).hexdigest()
 if len(norm)<5:return exact,normalized,{int.from_bytes(hashlib.blake2b(' '.join(norm).encode(),digest_size=8).digest(),'little')},len(norm)
 vals=[]
 for t in norm:
  if t not in TOKEN_HASH:TOKEN_HASH[t]=int.from_bytes(hashlib.blake2b(t.encode(),digest_size=8).digest(),'little')
  vals.append(TOKEN_HASH[t])
 h=0
 for v in vals[:5]:h=(h*BASE+v)&MASK
 out={h}
 for old,new in zip(vals,vals[5:]):
  h=((h-old*POWER)*BASE+new)&MASK;out.add(h)
 return exact,normalized,out,len(norm)
def records(archive,split):
 if archive.is_dir():
  with (archive/f'{split}.jsonl').open() as f:
   for line in f:yield json.loads(line)
  return
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   if member.name==f'data/raw/{split}.jsonl':
    for line in tar.extractfile(member):yield json.loads(line)
    return
 raise FileNotFoundError(split)
def audit(archive):
 start=time.monotonic();test=[];freq=Counter();exact_index=defaultdict(list);norm_index=defaultdict(list)
 for row in records(archive,'test'):
  ex,no,sh,n=signature(row['source']);i=len(test)
  test.append({'id':row['id'],'ipc':row['ipc'],'exact':ex,'normalized':no,'shingles':sh,'tokens':n,'matches':{},'exact_match':None,'normalized_match':None})
  freq.update(sh);exact_index[ex].append(i);norm_index[no].append(i)
 assert len(test)==17301
 prefix=defaultdict(list)
 for i,r in enumerate(test):
  sh=r['shingles'];k=len(sh)-math.ceil(.8*len(sh))+1
  for h in sorted(sh,key=lambda h:(freq[h],h))[:k]:prefix[h].append(i)
 print('TEST_INDEX_READY',len(test),'prefix keys',len(prefix),flush=True)
 candidates_checked=0;train_n=0;train_ids=hashlib.sha256()
 for row in records(archive,'train'):
  train_n+=1;train_ids.update((row['id']+'\n').encode())
  ex,no,sh,n=signature(row['source']);tid=row['id']
  for i in exact_index.get(ex,[]):test[i]['exact_match']=tid
  for i in norm_index.get(no,[]):
   r=test[i];r['normalized_match']=tid
   for threshold in [.8,.9,.95]:r['matches'].setdefault(str(threshold),{'train_id':tid,'jaccard':1.0,'kind':'normalized_exact'})
  candidates=set()
  for h in sh:candidates.update(prefix.get(h,()))
  for i in candidates:
   r=test[i]
   if '.95' in r['matches'] or '0.95' in r['matches']:continue
   ts=r['shingles'];a,b=len(ts),len(sh)
   if min(a,b)<.8*max(a,b):continue
   inter=len(ts & sh)
   if inter<math.ceil(.8*(a+b)/1.8):continue
   candidates_checked+=1;sim=inter/(a+b-inter)
   for threshold in [.8,.9,.95]:
    if sim>=threshold:r['matches'].setdefault(str(threshold),{'train_id':tid,'jaccard':sim,'kind':'fivegram'})
  if train_n%10000==0:
   status={'train_processed':train_n,'test_n':len(test),'near90':sum('0.9' in r['matches'] for r in test),'elapsed_seconds':time.monotonic()-start}
   (B/'dedup_progress.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
 assert train_n==309913
 rows=[]
 for i,r in enumerate(test):
  rows.append({'index':i,'id':r['id'],'ipc':r['ipc'],'exact_train_match':r['exact_match'],'normalized_train_match':r['normalized_match'],'token_count':r['tokens'],'shingle_count':len(r['shingles']),'near_matches':r['matches']})
 result={'protocol':json.loads((ROOT/'configs/expanded_evaluation.json').read_text()),'test_n':len(rows),'train_n':train_n,'elapsed_seconds':time.monotonic()-start,'summary':{}}
 for name,subset in [('original2000',rows[:2000]),('full17301',rows)]:
  result['summary'][name]={'n':len(subset),'exact_duplicates':sum(r['exact_train_match'] is not None for r in subset),'normalized_duplicates':sum(r['normalized_train_match'] is not None for r in subset),'retained_after_near_filter':{str(t):sum(str(t) not in r['near_matches'] for r in subset) for t in [.8,.9,.95]}}
 (B/'dedup_masks.json').write_text(json.dumps(rows,indent=2)+'\n');(B/'dedup_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2),flush=True)
def check():
 a='int sum(int x, int y) { return x+y; }';b='int add(int a, int b) { /* hi */ return a+b; }';c='int sum(int x, int y) { return x*y; }'
 aa=signature(a);bb=signature(b);cc=signature(c)
 assert aa[0]!=bb[0] and aa[1]==bb[1] and aa[2]==bb[2]
 assert aa[1]!=cc[1] and aa[2]!=cc[2]
 # Exhaustively verify prefix retrieval on all nonempty subsets of a small universe.
 sets=[{j for j in range(8) if mask&(1<<j)} for mask in range(1,256)]
 for a in sets:
  prefix=set(sorted(a)[:len(a)-math.ceil(.8*len(a))+1])
  for b in sets:
   if len(a&b)/len(a|b)>=.8:assert prefix&b
 print('NORMALIZATION_AND_PREFIX_RETRIEVAL_CHECKS_PASS',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);ap.add_argument('--raw-dir',type=Path);ap.add_argument('--output',type=Path,default=B);ap.add_argument('--check',action='store_true');args=ap.parse_args()
 B=args.output;B.mkdir(parents=True,exist_ok=True)
 if args.check:check()
 if args.archive or args.raw_dir:audit(args.raw_dir or args.archive)
