"""Validation-selected Ridge and nonlinear count baselines; equal 5000-label and full-train settings."""
import csv,json,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
BASE=Path(__file__).resolve().parent

def metrics(y,p):
 t=np.log1p(y);raw=np.expm1(p)
 return {'r2_log1p':float(1-np.square(t-p).sum()/np.square(t-t.mean()).sum()),'r2_raw':float(1-np.square(y-raw).sum()/np.square(y-y.mean()).sum()),'mae_ipc':float(abs(y-raw).mean())}

def main():
 data={s:np.load(BASE/f'{s}_cache.npz',allow_pickle=False) for s in ['train','val','test']}
 names=json.loads((BASE/'feature_manifest.json').read_text())['feature_names'];results={}
 groups={'source_counts':[i for i,x in enumerate(names) if x.startswith('src_')], 'assembly_length':[names.index('asm_instructions')], 'assembly_counts':[i for i,x in enumerate(names) if x.startswith('asm_')], 'source_and_assembly_counts':list(range(len(names)))}
 for budget,ix in [('probe5000',data['train']['token_indices']),('full_train',np.arange(len(data['train']['ipc'])))]:
  for group,cols in groups.items():
   x=data['train']['x'][ix][:,cols];y=np.log1p(data['train']['ipc'][ix]);v=data['val']['x'][:,cols];yt=data['val']['ipc'];candidates=[]
   for alpha in [.01,.1,1.,10.,100.,1000.]:
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x,y);score=metrics(yt,model.predict(v))['r2_log1p'];candidates.append((score,'ridge',{'alpha':alpha},model))
   for leaves in [15,31]:
    model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=leaves,l2_regularization=1.,learning_rate=.1,early_stopping=False,random_state=42);model.fit(x,y);score=metrics(yt,model.predict(v))['r2_log1p'];candidates.append((score,'hist_gradient_boosting',{'max_leaf_nodes':leaves,'max_iter':200,'l2_regularization':1.},model))
   for kind in ['ridge','hist_gradient_boosting']:
    selected=max((r for r in candidates if r[1]==kind),key=lambda z:z[0]);key=f'{budget}_{group}_{kind}';model=selected[3]
    result={'n_train':len(ix),'features':[names[i] for i in cols],'selection':'highest validation r2_log1p, no test selection','selected':selected[2],'validation_candidates':[{'family':r[1],'config':r[2],'r2_log1p':r[0]} for r in candidates if r[1]==kind]}
    for split in ['val','test']:
     d=data[split];p=model.predict(d['x'][:,cols]);result[split]=metrics(d['ipc'],p)
     with (BASE/f'{key}_{split}_predictions.csv').open('w') as f:
      w=csv.writer(f);w.writerow(['id','ipc_llvm_mca','pred_log1p','pred_ipc']);w.writerows(zip(d['ids'],d['ipc'],p,np.expm1(p)))
    results[key]=result;(BASE/'feature_results.json').write_text(json.dumps(results,indent=2));print(key,result['test'],flush=True)
if __name__=='__main__':main()
