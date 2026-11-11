"""(1) Cost of compile + llvm-mca vs the source-only predictor; (2) behaviour on code that does not compile.

Uses the frozen tuned IA joint checkpoints (seeds 42-44) with the same 5000-example Ridge readout as the paper.
Nothing here trains or selects anything on the test split.
"""
import importlib.util, json, os, re, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = Path('/proj/berzelius-2026-145/users/x_liryi')
COMMON = ROOT / 'source_ipc_baseline_20261025'
HPO = ROOT / 'ia_bpe_validation_search_20261026'
LLVM = ROOT / 'llvm18-env/bin'
RAW = ROOT / 'pretrained_ipc_20261027/raw/test.jsonl'
ENV = dict(os.environ, PATH=f'{LLVM}:{os.environ["PATH"]}')
N_SPEED = 300

sys.path.insert(0, str(COMMON / 'archive/Experiment'))
spec = importlib.util.spec_from_file_location('original_training', COMMON / 'archive/Experiment/final_train_seeded.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
from prepare import BPETokenizer
bpe = BPETokenizer(); bpe.load(COMMON / 'archive/dl4nlp_crossmodal/bpe_tokenizer.json')

def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())

def load_model(seed, device):
    path = HPO / 'trials/trial_07/checkpoint.pt' if seed == 42 else ROOT / f'ia_bpe_tuned_multiseed_20261027/runs/IA_seed{seed}/checkpoint.pt'
    c = torch.load(path, map_location='cpu', weights_only=True)
    assert c['step'] == 9762
    enc = original.Encoder(16384); enc.load_state_dict(c['src_enc']); enc.eval()
    return enc.to(device)

def encode(enc, tokens, device, bs=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(tokens), bs):
            out.append(enc(tokens[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def fit_ridge(enc, device):
    idx = np.load(COMMON / 'source_only_data/probe_train_indices.npy')
    tok = np.load(COMMON / 'source_only_data/train_tokens.npy', mmap_mode='r')
    y = np.log1p(np.asarray(np.load(COMMON / 'source_only_data/train_ipc.npy', mmap_mode='r')[idx], dtype=np.float64))
    x = encode(enc, torch.from_numpy(np.array(tok[idx], dtype=np.int64)), device)
    xm, ym = x.mean(0), y.mean(); xc = x - xm
    return xm, ym, np.linalg.solve(xc.T @ xc + np.eye(x.shape[1]), xc.T @ (y - ym))

def strip_preamble(src):
    i = src.find('__attribute__((used))')
    return src[i:] if i >= 0 else None

def compiles(src):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 's.c'; p.write_text(src)
        r = subprocess.run(['clang', '-fsyntax-only', '-w', str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
        return r.returncode == 0

def main():
    rows = []
    with RAW.open() as f:
        for i, line in enumerate(f):
            if i >= 2000: break
            rows.append(json.loads(line))
    y = np.array([r['ipc'] for r in rows])
    gpu = torch.device('cuda')
    res = {'n_test': len(rows)}

    # ---- (2) robustness to missing declarations (non-compilable snippets) ----
    stripped = [strip_preamble(r['source']) for r in rows]
    keep = [i for i, s in enumerate(stripped) if s]
    res['stripped_available'] = len(keep)
    with ThreadPoolExecutor(8) as ex:
        full_ok = list(ex.map(compiles, [rows[i]['source'] for i in keep]))
        strip_ok = list(ex.map(compiles, [stripped[i] for i in keep]))
    res['compile_rate_full_source'] = float(np.mean(full_ok))
    res['compile_rate_without_preamble'] = float(np.mean(strip_ok))
    tok_full = bpe.batch([rows[i]['source'] for i in keep], 512)
    tok_strip = bpe.batch([stripped[i] for i in keep], 512)
    res['mean_bpe_len_full'] = float((tok_full != 0).sum(1).float().mean()); res['mean_bpe_len_stripped'] = float((tok_strip != 0).sum(1).float().mean())
    yk = y[keep]; per_seed = {}
    for seed in (42, 43, 44):
        enc = load_model(seed, gpu); xm, ym, coef = fit_ridge(enc, gpu)
        pf = (encode(enc, tok_full, gpu) - xm) @ coef + ym
        ps = (encode(enc, tok_strip, gpu) - xm) @ coef + ym
        t = np.log1p(yk)
        per_seed[seed] = {'full': {'r2_log1p': r2(t, pf), 'mae_ipc': float(abs(yk - np.expm1(pf)).mean())},
                          'stripped': {'r2_log1p': r2(t, ps), 'mae_ipc': float(abs(yk - np.expm1(ps)).mean())},
                          'corr_full_vs_stripped_pred': float(np.corrcoef(pf, ps)[0, 1])}
        del enc; torch.cuda.empty_cache()
    res['robustness_per_seed'] = per_seed
    print(json.dumps(res, indent=1), flush=True)

    # ---- (1) cost of compile + llvm-mca vs prediction ----
    sample = rows[:N_SPEED]; t_cc, t_mca = [], []
    with tempfile.TemporaryDirectory() as d:
        for r in sample:
            p = Path(d) / 's.c'; p.write_text(r['source']); s = Path(d) / 's.s'
            a = time.perf_counter(); q = subprocess.run(['clang', '-O3', '-S', '-w', '-o', str(s), str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV); b = time.perf_counter()
            if q.returncode: continue
            subprocess.run(['llvm-mca', '-mcpu=skylake', str(s)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV); c = time.perf_counter()
            t_cc.append(b - a); t_mca.append(c - b)
    ms = lambda v: {'mean': 1e3 * statistics.mean(v), 'median': 1e3 * statistics.median(v), 'p95': 1e3 * sorted(v)[int(.95 * len(v))]}
    speed = {'n': len(t_cc), 'clang_O3_S_ms': ms(t_cc), 'llvm_mca_ms': ms(t_mca), 'compile_plus_mca_ms': ms([x + y_ for x, y_ in zip(t_cc, t_mca)])}
    enc = load_model(42, torch.device('cpu')); xm, ym, coef = fit_ridge(enc, torch.device('cpu'))
    for threads in (1, 8):
        torch.set_num_threads(threads); lat = []
        for k, r in enumerate(sample[:100]):
            a = time.perf_counter(); ids = bpe.batch([r['source']], 512)
            with torch.inference_mode(): z = enc(ids).numpy().astype(np.float64)
            _ = float(np.expm1((z[0] - xm) @ coef + ym)); lat.append(time.perf_counter() - a)
        speed[f'model_cpu_{threads}thread_batch1_ms'] = ms(lat[10:])
    del enc
    enc = load_model(42, gpu); srcs = [r['source'] for r in rows]
    a = time.perf_counter(); toks = bpe.batch(srcs, 512); t_tok = time.perf_counter() - a
    torch.cuda.synchronize(); a = time.perf_counter()
    encode(enc, toks, gpu, bs=256); torch.cuda.synchronize(); t_fwd = time.perf_counter() - a
    speed['model_gpu_a100_batch256'] = {'tokenize_ms_per_function': 1e3 * t_tok / len(srcs), 'encode_ms_per_function': 1e3 * t_fwd / len(srcs), 'functions_per_second_encode_only': len(srcs) / t_fwd}
    speed['cpu'] = subprocess.run('lscpu | grep "Model name"', shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    res['speed'] = speed
    (BASE / 'speed_robustness.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(speed, indent=1))

if __name__ == '__main__':
    main()
