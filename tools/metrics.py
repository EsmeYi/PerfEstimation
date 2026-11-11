"""Reference metrics using only Python's standard library; average ranks handle ties."""
import math,statistics
def ranks(values):
 order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);i=0
 while i<len(values):
  j=i+1
  while j<len(values) and values[order[j]]==values[order[i]]:j+=1
  rank=(i+j-1)/2
  for k in order[i:j]:out[k]=rank
  i=j
 return out
def correlation(a,b):
 am=statistics.mean(a);bm=statistics.mean(b);den=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
 return sum((x-am)*(y-bm) for x,y in zip(a,b))/den if den else None
def score(rows):
 if len(rows)<2:raise ValueError('At least two observations are required.')
 actual=[float(r['ipc_llvm_mca']) for r in rows];pred=[float(r.get('probe_log1p',r.get('pred_log1p'))) for r in rows]
 if not all(math.isfinite(x) for x in actual+pred):raise ValueError('Non-finite prediction or label.')
 target=[math.log1p(x) for x in actual];raw=[math.expm1(x) for x in pred]
 def r2(a,b):
  mean=statistics.mean(a);den=sum((x-mean)**2 for x in a)
  return 1-sum((x-y)**2 for x,y in zip(a,b))/den if den else None
 return {'r2_log1p':r2(target,pred),'r2_raw':r2(actual,raw),'mae_ipc':statistics.mean(abs(x-y) for x,y in zip(actual,raw)),'spearman':correlation(ranks(actual),ranks(pred))}
def aggregate(values):
 if len(values)<2:raise ValueError('Multiple seeds required for sample SD.')
 return {k:{'mean':statistics.mean(v[k] for v in values),'sample_sd':statistics.stdev(v[k] for v in values)} for k in values[0]}
