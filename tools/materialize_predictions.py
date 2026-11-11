"""Copy archived per-function predictions to runtime paths expected by the hardware analysis."""
from pathlib import Path
import shutil
P=Path(__file__).resolve().parents[1];E=P/'experiments'
for seed in [42,43,44]:
 target=E/f'ia_bpe_tuned_multiseed_20261027/runs/IA_seed{seed}/tuned_test_predictions.csv';target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(P/f'results/joint/IA_seed{seed}/tuned_test_predictions.csv',target)
sup=P/'results/supplement';out=E/'ipc_validation_20261030'
for f in sup.rglob('*'):
 if f.is_file() and (f.suffix=='.csv' or f.name in ['hw_measurements.jsonl','hw_candidates.jsonl','hw_source_preds.csv']):
  t=out/f.relative_to(sup);t.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(f,t)
print('Materialized archived predictions; no models trained or selected.')
