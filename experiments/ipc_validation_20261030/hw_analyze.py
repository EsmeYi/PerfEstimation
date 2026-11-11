"""Compare hardware-measured IPC with llvm-mca labels and learned predictions."""
import csv, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, PATH=f'{os.environ.get("LLVM_BIN", str(ROOT / "llvm18-env/bin"))}:{os.environ["PATH"]}')
rng = np.random.default_rng(0)

def mca(cpu, idx):
    f = BASE / 'hw' / f'f{idx:05d}' / 'f.s'
    out = subprocess.run(['llvm-mca', f'-mcpu={cpu}', str(f)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=ENV).stdout
    m = re.search(r'IPC:\s+([\d.]+)', out)
    return float(m.group(1)) if m else np.nan

def load_preds(path, col):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r['id']] = float(r[col])
    return d

def boot(x, y, fn, n=2000):
    idx = rng.integers(0, len(x), (n, len(x)))
    v = [fn(x[i], y[i]) for i in idx]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def row(name, h, p):
    ok = np.isfinite(p)
    h, p = h[ok], p[ok]
    sp = spearmanr(h, p)[0]; pe = pearsonr(np.log1p(h), np.log1p(p))[0]
    lo, hi = boot(h, p, lambda a, b: spearmanr(a, b)[0], 1000)
    return {'predictor': name, 'n': int(len(h)), 'spearman': float(sp), 'spearman_ci95': [lo, hi], 'pearson_log1p': float(pe),
            'mae_ipc': float(np.abs(h - p).mean())}

def main():
    recs = [json.loads(l) for l in (BASE / 'hw_measurements.jsonl').open()]
    with ThreadPoolExecutor(8) as ex:
        z4 = list(ex.map(lambda r: mca('znver4', r['idx']), recs))
        z2 = list(ex.map(lambda r: mca('znver2', r['idx']), recs))
    for r, a, b in zip(recs, z4, z2):
        r['mca_znver4_ipc'] = a; r['mca_znver2_ipc'] = b
    tuned = [load_preds(ROOT / f'ia_bpe_tuned_multiseed_20261027/runs/IA_seed{s}/tuned_test_predictions.csv', 'probe_ipc') for s in (42, 43, 44)]
    gbm = load_preds(BASE / 'full_train_source_and_assembly_counts_hist_gradient_boosting_test_predictions.csv', 'pred_ipc')
    gbm_asm = load_preds(BASE / 'full_train_assembly_counts_hist_gradient_boosting_test_predictions.csv', 'pred_ipc')
    src_gbm = load_preds(BASE / 'full_train_source_counts_hist_gradient_boosting_test_predictions.csv', 'pred_ipc')
    asm_only = [load_preds(p, 'pred_ipc') for p in sorted((BASE / 'asm_only').glob('seed*_test_predictions.csv'))] if (BASE / 'asm_only').exists() else []
    for r in recs:
        i = r['id']
        r['pred_source_model'] = float(np.mean([t[i] for t in tuned])) if i in tuned[0] else np.nan
        r['pred_source_counts_gbm'] = src_gbm.get(i, np.nan)
        r['pred_asm_counts_gbm'] = gbm_asm.get(i, np.nan)
        r['pred_asm_only_model'] = float(np.mean([t[i] for t in asm_only])) if asm_only and i in asm_only[0] else np.nan
    ext = BASE / 'hw_source_preds.csv'
    if ext.exists():  # frozen paper checkpoints applied to every measurable test function
        with ext.open() as f:
            by_idx = {int(r['idx']): float(r['pred_mean']) for r in csv.DictReader(f)}
        for r in recs:
            r['pred_source_model_ext'] = by_idx.get(r['idx'], np.nan)
    else:
        for r in recs:
            r['pred_source_model_ext'] = np.nan
    (BASE / 'hw_merged.jsonl').write_text('\n'.join(json.dumps(r) for r in recs))
    A = lambda k, rs: np.array([r[k] for r in rs], float)
    quality = [r for r in recs if r['hw_ipc_spread'] < 0.10]
    out = {'n_measured': len(recs), 'n_stable_spread_lt_10pct': len(quality),
           'variant_counts': {str(v): sum(r['variant'] == v for r in recs) for v in (0, 1)},
           'label_reproduced_fraction': float(np.mean([r['label_reproduced'] for r in recs])),
           'subsets': {}}
    subsets = {'all_stable': quality,
               'static_instr_ge_10': [r for r in quality if r['static_instr'] >= 10],
               'straight_line_no_branch': [r for r in quality if not r['has_branch']],
               'with_branches': [r for r in quality if r['has_branch']],
               'first2000_test_with_model_preds': [r for r in quality if np.isfinite(r['pred_source_model'])],
               'first2000_and_instr_ge_10': [r for r in quality if np.isfinite(r['pred_source_model']) and r['static_instr'] >= 10]}
    for name, rs in subsets.items():
        if len(rs) < 20:
            out['subsets'][name] = {'n': len(rs)}; continue
        h = A('hw_ipc', rs)
        entries = [row('llvm-mca skylake (paper label)', h, A('label_ipc', rs)), row('llvm-mca znver4 (matches host)', h, A('mca_znver4_ipc', rs)),
                   row('llvm-mca znver2', h, A('mca_znver2_ipc', rs)), row('static instruction count', h, A('static_instr', rs))]
        for k, label in [('pred_source_model_ext', 'source model (IA joint, mean of 3 seeds)'), ('pred_source_counts_gbm', 'source-count GBM'),
                         ('pred_asm_counts_gbm', 'assembly-count GBM'), ('pred_asm_only_model', 'assembly-only transformer')]:
            p = A(k, rs)
            if np.isfinite(p).sum() >= 20:
                entries.append(row(label, h, p))
        out['subsets'][name] = {'n': len(rs), 'hw_ipc_mean': float(h.mean()), 'hw_ipc_sd': float(h.std()), 'entries': entries}
    (BASE / 'hw_analysis.json').write_text(json.dumps(out, indent=2))
    for name, s in out['subsets'].items():
        print('==', name, 'n =', s['n'])
        for e in s.get('entries', []):
            print(f"   {e['predictor']:45s} n={e['n']:5d} spearman={e['spearman']:.3f} [{e['spearman_ci95'][0]:.3f},{e['spearman_ci95'][1]:.3f}] pearson(log)={e['pearson_log1p']:.3f} MAE={e['mae_ipc']:.3f}")

if __name__ == '__main__':
    main()
