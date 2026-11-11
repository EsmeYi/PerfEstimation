"""All test functions and all three seeds; common transformed axes and count scale."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'generated/.mpl-cache'))
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'generated';(BASE/'figures').mkdir(exist_ok=True,parents=True)
ia=json.loads((ROOT/'results/joint/summary.json').read_text())['summary']['IA']['test']['tuned']
ab=json.loads((ROOT/'results/supervision/summary.json').read_text())
models=[('IPC only','7.52M',ab['source_only'],[ROOT/f'results/supervision/source_only_seed{seed}/ablation_test_predictions.csv' for seed in [42,43,44]]),('Alignment only','7.52M',ab['alignment_only'],[ROOT/f'results/supervision/alignment_only_seed{seed}/ablation_test_predictions.csv' for seed in [42,43,44]]),('Joint (IA)','7.52M',ia,[ROOT/f'results/joint/IA_seed{seed}/tuned_test_predictions.csv' for seed in [42,43,44]])]
arrays=[];sources=[];reference=None
for model,params,metrics,paths in models:
 xs=[];ys=[]
 for path in paths:
  rows=list(csv.DictReader(path.open()));ids=[r['id'] for r in rows]
  assert len(rows)==2000 and len(set(ids))==2000
  x=np.array([math.log1p(float(r['ipc_llvm_mca'])) for r in rows]);y=np.array([float(r['probe_log1p']) for r in rows])
  assert np.isfinite(x).all() and np.isfinite(y).all()
  if reference is None:reference=(ids,x.copy())
  assert ids==reference[0] and np.array_equal(x,reference[1])
  xs.extend(x);ys.extend(y);sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 arrays.append((np.array(xs),np.array(ys)))
low=math.floor(min(0.,*[float(np.min(a)) for pair in arrays for a in pair])*2)/2
high=math.ceil(max(float(np.max(a)) for pair in arrays for a in pair)*2)/2
with plt.rc_context({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42}):
 fig,axes=plt.subplots(1,3,figsize=(7.16,2.72),layout='constrained',sharex=True,sharey=True)
 bins=[]
 for i,(ax,(name,params,m,_),(x,y)) in enumerate(zip(axes,models,arrays)):
  h=ax.hexbin(x,y,gridsize=35,extent=(low,high,low,high),mincnt=1,cmap='viridis',linewidths=0)
  assert int(h.get_array().sum())==len(x), 'Prediction pairs dropped by binning'
  bins.append(h);ax.plot([low,high],[low,high],ls='--',color='#ba3b25',lw=.9)
  ax.set(xlim=(low,high),ylim=(low,high),xlabel='Analyzed log(1 + IPC)',title=f'({chr(97+i)}) {name}');ax.set_aspect('equal')
  r=m['probe_r2_log1p'];sd=r.get('sample_sd',r.get('sd'))
  ax.text(.04,.96,f"$R^2$ = {r['mean']:.3f} ± {sd:.3f}",transform=ax.transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
  ax.tick_params(length=3,width=.6)
  for spine in ax.spines.values():spine.set_linewidth(.6)
 axes[0].set_ylabel('Predicted log(1 + IPC)')
 vmax=max(float(h.get_array().max()) for h in bins)
 for h in bins:h.set_norm(LogNorm(vmin=1,vmax=vmax))
 cb=fig.colorbar(bins[0],ax=axes,shrink=.78,pad=.025,aspect=23);cb.set_label('Function–seed pairs');cb.locator=LogLocator(base=10);cb.update_ticks()
 fig.savefig(BASE/'figures/ablation_prediction.pdf',facecolor='white')
 fig.savefig(BASE/'figures/ablation_prediction.png',dpi=400,facecolor='white')
 plt.close(fig)
manifest={'sources':sources,'models':[x[0] for x in models],'functions_per_seed':2000,'seeds':[42,43,44],'prediction_pairs_per_panel':6000,'axes':'log1p actual IPC and unmodified probe log1p prediction','axis_limits':[low,high],'binning':'hexbin gridsize35 common extent','density':'logarithmic count, same scale in all panels','count_range':[1,vmax],'exclusions':None,'aggregation':'density counts only; no averaging predictions','uncertainty':'sample SD over individual-seed R2; no pooled R2','width_inches':7.16,'height_inches':2.72,'png_dpi':400,'matplotlib':matplotlib.__version__,'numpy':np.__version__}
(BASE/'figures/ablation_prediction_manifest.json').write_text(json.dumps(manifest,indent=2))
print('ALL_18000_PREDICTIONS_INCLUDED; COMMON_FUNCTION_IDS_VERIFIED')
