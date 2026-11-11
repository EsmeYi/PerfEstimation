"""Source-model predictions (frozen tuned IA joint checkpoints, 5000-example Ridge readout) for every
hardware-measurable test function, not only the 2,000 used in the paper's tables."""
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from speed_robustness import BASE, RAW, bpe, encode, fit_ridge, load_model

def main():
    cands = [json.loads(l) for l in (BASE / 'hw_candidates.jsonl').open()]
    want = {c['idx'] for c in cands}
    src = {}
    with RAW.open() as f:
        for i, line in enumerate(f):
            if i in want:
                src[i] = json.loads(line)['source']
    idxs = sorted(src)
    toks = bpe.batch([src[i] for i in idxs], 512)
    gpu = torch.device('cuda'); preds = []
    for seed in (42, 43, 44):
        enc = load_model(seed, gpu); xm, ym, coef = fit_ridge(enc, gpu)
        preds.append(np.expm1((encode(enc, toks, gpu) - xm) @ coef + ym))
        del enc; torch.cuda.empty_cache()
    preds = np.stack(preds)
    with (BASE / 'hw_source_preds.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['idx', 'pred_seed42', 'pred_seed43', 'pred_seed44', 'pred_mean'])
        for k, i in enumerate(idxs):
            w.writerow([i, *preds[:, k], preds[:, k].mean()])
    print('wrote', len(idxs))

if __name__ == '__main__':
    main()
