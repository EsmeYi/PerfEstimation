"""Joint IA model (paper's frozen recipe) trained with declaration-stripped sources mixed in (p=0.5 per example).

Everything else matches the paper's joint model: assembly partner, InfoNCE tau=.05, log1p regression with
lambda=5.99, 9,762 updates, batch 256, AdamW, same schedule. Evaluated on complete and declaration-stripped sources.
"""
import csv, importlib.util, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / 'source_ipc_baseline_20261025'
ASM = ROOT / 'ia_fixed_joint_ablation_20261026/assembly_data'
LAMBDA = 5.990544826588272; STEPS = 9762; P_STRIP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

def load_encoder_class():
    p = COMMON / 'archive/Experiment/final_train_seeded.py'
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location('original_training', p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Encoder

def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())

def strip_preamble(s):
    i = s.find('__attribute__((used))')
    return s[i:] if i >= 0 else None

def main():
    seed = int(sys.argv[1])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.backends.cudnn.benchmark = False
    Encoder = load_encoder_class()
    src_tok = np.load(COMMON / 'source_only_data/train_tokens.npy', mmap_mode='r')
    asm = np.load(ASM / 'train_asm_ia_fixed.npy', mmap_mode='r')
    labels = np.load(COMMON / 'source_only_data/train_ipc.npy', mmap_mode='r')
    st = np.load(BASE / 'strip_train_tokens.npy', mmap_mode='r'); has = np.load(BASE / 'strip_train_has.npy')
    probe_idx = np.load(COMMON / 'source_only_data/probe_train_indices.npy')
    src = Encoder(16384).cuda(); asm_enc = Encoder(254).cuda(); head = torch.nn.Linear(128, 1).cuda()
    params = list(src.parameters()) + list(asm_enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=.001, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / 200 if s < 200 else max(.1, .5 * (1 + math.cos(math.pi * (s - 200) / 9800))))
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(len(asm), generator=g).numpy(); pos = 0
    rng = np.random.default_rng(seed); t0 = time.monotonic()
    for m in (src, asm_enc, head): m.train()
    for step in range(1, STEPS + 1):
        if pos + 256 > len(perm):
            perm = torch.randperm(len(asm), generator=g).numpy(); pos = 0
        ix = np.sort(perm[pos:pos + 256]); pos += 256
        use = (rng.random(len(ix)) < P_STRIP) & has[ix]
        s = np.where(use[:, None], np.array(st[ix], dtype=np.int64), np.array(src_tok[ix], dtype=np.int64))
        ids = torch.from_numpy(s).cuda(); a_ids = torch.from_numpy(np.array(asm[ix], dtype=np.int64)).cuda()
        y = torch.log1p(torch.from_numpy(np.array(labels[ix], dtype=np.float32)).cuda())
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z = src(ids); a = asm_enc(a_ids)
            sc = F.normalize(z, dim=-1) @ F.normalize(a, dim=-1).T / .05; tr = torch.arange(len(z), device='cuda')
            loss = (F.cross_entropy(sc, tr, label_smoothing=.05) + F.cross_entropy(sc.T, tr, label_smoothing=.05)) / 2 + LAMBDA * F.mse_loss(head(z).squeeze(-1), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, .5); opt.step(); sched.step()
        if step == 1 or step % 500 == 0:
            print(f'seed={seed} step={step}/{STEPS} loss={loss.item():.4f} t={time.monotonic() - t0:.0f}s', flush=True)
    src.eval()
    def encode(tokens):
        out = []
        with torch.inference_mode():
            for i in range(0, len(tokens), 16):
                out.append(src(torch.from_numpy(np.array(tokens[i:i + 16], dtype=np.int64)).cuda()).cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    def fit(x, y):
        xm = x.mean(0); ym = y.mean(); xc = x - xm
        return xm, ym, np.linalg.solve(xc.T @ xc + np.eye(128), xc.T @ (y - ym))
    pi = np.sort(probe_idx); order = np.argsort(np.argsort(probe_idx))
    yp = np.log1p(np.asarray(labels[probe_idx], dtype=np.float64))
    xf = encode(src_tok[pi])[order]; xs = encode(st[pi])[order]; hp = has[probe_idx]
    ridge = {'ridge_on_full': fit(xf, yp), 'ridge_on_mixed': fit(np.concatenate([xf, xs[hp]]), np.concatenate([yp, yp[hp]]))}
    rows = [json.loads(l) for _, l in zip(range(2000), open(ROOT / 'pretrained_ipc_20261027/raw/test.jsonl'))]
    y_test = np.array([r['ipc'] for r in rows]); strips = [strip_preamble(r['source']) for r in rows]
    keep = np.array([s is not None for s in strips])
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(COMMON / 'archive/dl4nlp_crossmodal/bpe_tokenizer.json'))
    stoks = np.zeros((2000, 512), np.int64)
    for k, s in enumerate(strips):
        if s is not None:
            ids = [2] + tk.encode(s).ids[:510] + [3]; stoks[k, :len(ids)] = ids
    zf = encode(np.load(COMMON / 'source_only_data/test_tokens.npy', mmap_mode='r')); zs = encode(stoks)
    res = {'seed': seed, 'p_strip': P_STRIP, 'steps': STEPS, 'n_test_full': 2000, 'n_test_strippable': int(keep.sum()), 'train_seconds': time.monotonic() - t0}
    t = np.log1p(y_test)
    for name, (xm, ym, coef) in ridge.items():
        pf = (zf - xm) @ coef + ym; ps = (zs - xm) @ coef + ym
        res[name] = {'full_all2000': {'r2_log1p': r2(t, pf), 'mae_ipc': float(abs(y_test - np.expm1(pf)).mean())},
                     'full_on_strippable_subset': {'r2_log1p': r2(t[keep], pf[keep]), 'mae_ipc': float(abs(y_test[keep] - np.expm1(pf[keep])).mean())},
                     'stripped_on_strippable_subset': {'r2_log1p': r2(t[keep], ps[keep]), 'mae_ipc': float(abs(y_test[keep] - np.expm1(ps[keep])).mean())}}
        out_csv = BASE / 'aug_runs'; out_csv.mkdir(exist_ok=True)
        for condition, prediction, mask in [('full', pf, np.ones(len(pf), dtype=bool)), ('stripped', ps, keep)]:
            with (out_csv / f'seed{seed}_p{P_STRIP}_{name}_{condition}_predictions.csv').open('w') as handle:
                writer=csv.writer(handle);writer.writerow(['id','ipc_llvm_mca','probe_log1p','probe_ipc'])
                writer.writerows((rows[i]['id'],y_test[i],prediction[i],float(np.expm1(prediction[i]))) for i in np.flatnonzero(mask))
    print(json.dumps(res, indent=1), flush=True)
    out = BASE / 'aug_runs'; out.mkdir(exist_ok=True)
    torch.save({'src_enc': src.state_dict(), 'asm_enc': asm_enc.state_dict(), 'hw_head': head.state_dict(), 'step': STEPS, 'seed': seed, 'p_strip': P_STRIP}, out / f'seed{seed}_p{P_STRIP}_checkpoint.pt')
    (out / f'seed{seed}_p{P_STRIP}_metrics.json').write_text(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
