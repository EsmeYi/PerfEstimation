"""Recompute final paper results from archived predictions; standard library only."""
from pathlib import Path
import argparse,csv,json
from metrics import score,aggregate
P=Path(__file__).resolve().parents[1];SEEDS=[42,43,44]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=P/'generated');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 ids=json.loads((P/'data/splits/test_ids.json').read_text());reference=None;checked=0
 def read(path,n=2000):
  nonlocal reference,checked
  rows=list(csv.DictReader(path.open()));assert len(rows)==n and len({r['id'] for r in rows})==n,path
  assert [r['id'] for r in rows[:2000]]==ids,path
  labels=[float(r['ipc_llvm_mca']) for r in rows[:2000]]
  if reference is None:reference=labels
  assert labels==reference,path
  checked+=1;return rows
 original={}
 for name in ['IA','BPE']:
  original[name]=aggregate([score(read(P/f'results/joint/{name}_seed{s}/tuned_test_predictions.csv')) for s in SEEDS])
 for name in ['source_only','alignment_only']:
  original[name]=aggregate([score(read(P/f'results/supervision/{name}_seed{s}/ablation_test_predictions.csv')) for s in SEEDS])
 for name in ['codebert_ipc','codebert_joint','unixcoder_ipc','unixcoder_joint']:
  original[name]=aggregate([score(read(P/f'results/pretrained/{name}_seed{s}/test_predictions.csv')) for s in SEEDS])
 masks={r['id']:r for r in json.loads((P/'results/expanded/dedup_masks.json').read_text())};assert len(masks)==17301
 names=['compact_ipc','compact_bpe_joint','compact_ia_joint','codebert_ipc','codebert_joint','unixcoder_ipc','unixcoder_joint'];expanded={};per_seed=[];full_ids=None;full_labels=None
 scopes={'all':None,'near08':'0.8','near09':'0.9','near095':'0.95'}
 for name in names:
  values={k:[] for k in scopes}
  for seed in SEEDS:
   rows=read(P/f'results/expanded/{name}_seed{seed}/test_predictions.csv',17301)
   ii=[r['id'] for r in rows];yy=[float(r['ipc_llvm_mca']) for r in rows]
   if full_ids is None:full_ids=ii;full_labels=yy
   assert ii==full_ids and yy==full_labels
   for scope,threshold in scopes.items():
    rr=rows if threshold is None else [r for r in rows if threshold not in masks[r['id']]['near_matches']]
    m=score(rr);values[scope].append(m);per_seed.append({'condition':name,'seed':seed,'scope':scope,'n':len(rr),**m})
  expanded[name]={scope:{'n':next(r['n'] for r in per_seed if r['condition']==name and r['scope']==scope),'n_seeds':3,**aggregate(v)} for scope,v in values.items()}
 expected=json.loads((P/'results/expanded/full_filtered_summary.json').read_text())['summary']
 for name in names:
  for scope in scopes:
   for key in ['r2_log1p','mae_ipc','spearman']:
    for statistic in ['mean','sample_sd']:
     assert abs(expanded[name][scope][key][statistic]-expected[name][scope][key][statistic])<1e-9,(name,scope,key,statistic)
 assert expanded[names[0]]['near09']['n']==12600
 sup=P/'results/supplement';features={}
 for name in ['full_train_source_counts_ridge','probe5000_source_counts_hist_gradient_boosting','full_train_source_counts_hist_gradient_boosting','full_train_assembly_length_hist_gradient_boosting','full_train_assembly_counts_hist_gradient_boosting']:
  features[name]=score(read(sup/(name+'_test_predictions.csv')))
 features['assembly_transformer']=aggregate([score(read(sup/f'asm_only/seed{s}_test_predictions.csv')) for s in SEEDS])
 mixed=[json.loads((sup/f'aug_runs/seed{s}_p0.5_metrics.json').read_text()) for s in SEEDS]
 mixed={k:aggregate([r['ridge_on_full'][k] for r in mixed]) for k in ['full_all2000','full_on_strippable_subset','stripped_on_strippable_subset']}
 original_views=json.loads((sup/'speed_robustness.json').read_text())['robustness_per_seed']
 mixed['complete_only_training']={k:aggregate([original_views[str(seed)][k] for seed in SEEDS]) for k in ['full','stripped']}
 output={'original2000':original,'expanded':expanded,'feature_and_assembly_references':features,'mixed_view':mixed,'inference':json.loads((P/'results/expanded/efficiency_summary.json').read_text()),'checks':{'prediction_files_recomputed':checked,'full_test_n':17301,'filtered_n':12600,'all_common_ids_and_labels_verified':True,'seeds':SEEDS,'metric_space':'R2 log1p IPC;MAE IPC;Spearman static IPC','aggregation':'mean and sample SD over seeds; no pooled score','mixed_view':'historical per-seed metrics; per-example files not saved'}}
 (a.output/'tables.json').write_text(json.dumps(output,indent=2)+'\n')
 with (a.output/'expanded_per_seed.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
 lines=['# Recomputed final results','','All values are mean ± sample SD across seeds42/43/44. Filtering: normalized token5gram Jaccard>=0.9.','','| Encoder / supervision | Full17301 R² | Filtered12600 R² | Full MAE | Filtered MAE | Full Spearman | Filtered Spearman |','|---|---:|---:|---:|---:|---:|---:|']
 def fmt(v):return f'{v["mean"]:.6f} ± {v["sample_sd"]:.6f}'
 for name in names:
  d=expanded[name];lines.append('| '+name+' | '+' | '.join(fmt(d[scope][metric]) for metric in ['r2_log1p','mae_ipc','spearman'] for scope in ['all','near09'])+' |')
 lines+=['','## Original2000 controlled evaluation','','| Configuration | R² | MAE |','|---|---:|---:|']
 for name,d in original.items():lines.append(f'| {name} | {fmt(d["r2_log1p"])} | {fmt(d["mae_ipc"])} |')
 (a.output/'tables.md').write_text('\n'.join(lines)+'\n')
 print(f'PASS: {checked} prediction files; original/full IDs and labels; all thresholds and three-seed statistics verified.')
 print('Wrote',a.output/'tables.md')
if __name__=='__main__':main()
