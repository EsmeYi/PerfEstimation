"""Assembly-only IPC regressor: same encoder, IA tokens, optimizer, schedule, update budget and Ridge readout as the
source-only ablation, but the input is the compiled assembly. A trained reference for how far assembly is from source."""
import csv, importlib.util, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = Path('/proj/berzelius-2026-145/users/x_liryi')
COMMON = ROOT / 'source_ipc_baseline_20261025'
ASM = ROOT / 'ia_fixed_joint_ablation_20261026/assembly_data'
LAMBDA = 5.990544826588272  # validation-selected weight from the paper's joint model (Eq. 3)
STEPS = 9762

def load_encoder_class():
    p = COMMON / 'archive/Experiment/final_train_seeded.py'
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location('original_training', p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Encoder

def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())

def main():
    seed = int(sys.argv[1])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.backends.cudnn.benchmark = False
    Encoder = load_encoder_class()
    asm = np.load(ASM / 'train_asm_ia_fixed.npy', mmap_mode='r')
    labels = np.load(COMMON / 'source_only_data/train_ipc.npy', mmap_mode='r')
    probe_idx = np.load(COMMON / 'source_only_data/probe_train_indices.npy')
    test_asm = np.load(ASM / 'test_asm_ia_fixed.npy', mmap_mode='r')
    ref = np.load(COMMON / 'probe_embeddings_seed42.npz', allow_pickle=False)
    test_ipc = ref['test_ipc'].astype(np.float64); test_ids = ref['test_ids'].tolist()
    assert len(test_asm) == 2000 == len(test_ipc)
    enc = Encoder(254).cuda(); head = torch.nn.Linear(128, 1).cuda()
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=.001, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / 200 if s < 200 else max(.1, .5 * (1 + math.cos(math.pi * (s - 200) / 9800))))
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(len(asm), generator=g).numpy(); pos = 0
    enc.train(); head.train(); t0 = time.monotonic()
    for step in range(1, STEPS + 1):
        if pos + 256 > len(perm):
            perm = torch.randperm(len(asm), generator=g).numpy(); pos = 0
        ix = np.sort(perm[pos:pos + 256]); pos += 256
        ids = torch.from_numpy(np.array(asm[ix], dtype=np.int64)).cuda()
        y = torch.log1p(torch.from_numpy(np.array(labels[ix], dtype=np.float32)).cuda())
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = LAMBDA * F.mse_loss(head(enc(ids)).squeeze(-1), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, .5); opt.step(); sched.step()
        if step == 1 or step % 500 == 0:
            print(f'seed={seed} step={step}/{STEPS} loss={loss.item():.5f} t={time.monotonic() - t0:.0f}s', flush=True)
    enc.eval(); head.eval()
    def encode(tokens):
        zs, ps = [], []
        with torch.inference_mode():
            for i in range(0, len(tokens), 16):
                z = enc(torch.from_numpy(np.array(tokens[i:i + 16], dtype=np.int64)).cuda())
                zs.append(z.cpu().numpy()); ps.append(head(z).squeeze(-1).cpu().numpy())
        return np.concatenate(zs).astype(np.float64), np.concatenate(ps).astype(np.float64)
    x, _ = encode(asm[np.sort(probe_idx)])
    order = np.argsort(np.argsort(probe_idx))  # restore probe order after sorted gather
    x = x[order]
    yp = np.log1p(np.asarray(labels[probe_idx], dtype=np.float64)); xm = x.mean(0); ym = yp.mean(); xc = x - xm
    coef = np.linalg.solve(xc.T @ xc + np.eye(128), xc.T @ (yp - ym))
    z, direct = encode(test_asm)
    pred = (z - xm) @ coef + ym
    res = {'seed': seed, 'steps': STEPS, 'params_encoder': sum(p.numel() for p in enc.parameters()),
           'test_probe': {'r2_log1p': r2(np.log1p(test_ipc), pred), 'r2_raw': r2(test_ipc, np.expm1(pred)), 'mae_ipc': float(abs(test_ipc - np.expm1(pred)).mean())},
           'test_direct_head_r2_log1p': r2(np.log1p(test_ipc), direct), 'train_seconds': time.monotonic() - t0}
    print(json.dumps(res), flush=True)
    out = BASE / 'asm_only'; out.mkdir(exist_ok=True)
    (out / f'seed{seed}_metrics.json').write_text(json.dumps(res, indent=2))
    with (out / f'seed{seed}_test_predictions.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['id', 'ipc_llvm_mca', 'pred_log1p', 'pred_ipc']); w.writerows(zip(test_ids, test_ipc, pred, np.expm1(pred)))

if __name__ == '__main__':
    main()
