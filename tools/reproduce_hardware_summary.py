"""Recompute exploratory point correlations from archived measurements; no perf access required."""
import json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def ranks(x):
 order=sorted(range(len(x)),key=x.__getitem__);out=[0.]*len(x);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  for k in order[i:j]:out[k]=(i+j-1)/2
  i=j
 return out
def pearson(x,y):
 a=statistics.mean(x);b=statistics.mean(y);den=math.sqrt(sum((v-a)**2 for v in x)*sum((v-b)**2 for v in y));return sum((u-a)*(v-b) for u,v in zip(x,y))/den
rows=[json.loads(l) for l in (P/'results/supplement/hw_merged.jsonl').open()];stable=[r for r in rows if r['hw_ipc_spread']<.1]
subsets={'all_stable':stable,'at_least_10_instructions':[r for r in stable if r['static_instr']>=10],'branch_free_at_least_10':[r for r in stable if r['static_instr']>=10 and not r['has_branch']]};out={}
for name,rs in subsets.items():
 out[name]={'n':len(rs)}
 for k in ['label_ipc','pred_source_model_ext','static_instr']:
  good=[r for r in rs if math.isfinite(r[k])];out[name][k]=pearson(ranks([r[k] for r in good]),ranks([r['hw_ipc'] for r in good]))
assert len(rows)==2137 and len(stable)==2131
assert abs(out['all_stable']['label_ipc']-.6524307773882477)<1e-10
(P/'generated').mkdir(exist_ok=True);(P/'generated/hardware_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
