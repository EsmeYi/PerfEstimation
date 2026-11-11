"""Select self-contained test functions, recover their C signatures, and cache llvm-mca labels.

A function is measurable on hardware only if its stored assembly assembles to an object with no undefined
symbols (no calls, no globals) and its source gives a classifiable signature. The assembly measured is the
stored assembly, so the instruction stream is the one the llvm-mca label was computed from.
"""
import json, re, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'pretrained_ipc_20261027/raw/test.jsonl'
LLVM = Path(os.environ.get('LLVM_BIN',str(ROOT / 'llvm18-env/bin')))
WORK = BASE / 'hw'
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 17301
ENV = dict(os.environ, PATH=f'{LLVM}:{os.environ["PATH"]}')

def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=ENV, **kw)

def classify(t):
    t = t.replace('const', '').replace('volatile', '').replace('restrict', '').strip()
    if '(' in t:  # function pointer
        return 'P'
    if '*' in t or '[' in t:
        return 'P'
    if t in ('float',):
        return 'f'
    if t in ('double',):
        return 'd'
    if re.fullmatch(r'(unsigned |signed )?(char|short|int|long|long long|_Bool|__int128)|unsigned|signed|size_t', t) or t.startswith('enum'):
        return 'I' if '__int128' not in t else None
    return None  # structs by value, long double, vectors, ...

def signature(src, name, d):
    p = d / 's.c'
    p.write_text(src)
    r = run(['clang', '-fsyntax-only', '-w', '-Xclang', '-ast-dump=json', '-Xclang', f'-ast-dump-filter={name}', str(p)])
    out = r.stdout
    dec = json.JSONDecoder()
    i = out.find('{')
    while i != -1 and i < len(out):
        try:
            obj, j = dec.raw_decode(out, i)
        except ValueError:
            break
        if obj.get('kind') == 'FunctionDecl' and obj.get('name') == name:
            params = [x for x in obj.get('inner', []) if x.get('kind') == 'ParmVarDecl']
            if '...' in obj.get('type', {}).get('qualType', ''):
                return None
            cls = [classify(x['type'].get('desugaredQualType') or x['type']['qualType']) for x in params]
            if any(c is None for c in cls):
                return None
            return ''.join(cls)
        i = out.find('{', j)
    return None

def process(item):
    idx, row = item
    name_list = re.findall(r'\.type\s+([\w$.@]+),@function', row['asm'])
    if len(name_list) != 1:
        return idx, None, 'not_single_function'
    name = name_list[0]
    if re.search(r'\bcall|@PLT|@GOT|@TPOFF|\bjmpq?\s+\*|\.quad\s+\.|\.long\s+\.L', row['asm']):
        return idx, None, 'external_or_indirect'
    d = WORK / f'f{idx:05d}'
    d.mkdir(parents=True, exist_ok=True)
    asm = row['asm'] + f'\n\t.globl\t{name}\n'
    (d / 'f.s').write_text(asm)
    r = run(['clang', '-c', '-x', 'assembler', '-o', str(d / 'f.o'), str(d / 'f.s')])
    if r.returncode:
        return idx, None, 'assemble_fail'
    u = [x for x in run(['llvm-nm', '-u', str(d / 'f.o')]).stdout.split('\n') if x.strip() and '_GLOBAL_OFFSET_TABLE_' not in x]
    if u:
        return idx, None, 'undefined_symbols'
    sig = signature(row['source'], name, d)
    if sig is None:
        return idx, None, 'signature'
    if sig.count('I') + sig.count('P') > 6 or sig.count('f') + sig.count('d') > 8:
        return idx, None, 'too_many_args'
    rec = {'idx': idx, 'id': row['id'], 'name': name, 'sig': sig, 'label_ipc': row['ipc'], 'label_cycles': row['total_cycles']}
    for cpu in ('skylake', 'znver2'):
        m = run(['llvm-mca', f'-mcpu={cpu}', str(d / 'f.s')]).stdout
        g = lambda k: float(re.search(k + r':\s+([\d.]+)', m).group(1))
        rec[f'mca_{cpu}_ipc'] = g('IPC'); rec[f'mca_{cpu}_cycles'] = g('Total Cycles'); rec[f'mca_{cpu}_rthru'] = g('Block RThroughput')
        rec['static_instr'] = g('Instructions') / g('Iterations')
    rec['label_reproduced'] = abs(rec['mca_skylake_ipc'] - row['ipc']) < 0.011
    return idx, rec, 'ok'

def main():
    WORK.mkdir(exist_ok=True)
    rows = []
    with RAW.open() as f:
        for i, line in enumerate(f):
            if i >= NMAX:
                break
            rows.append((i, json.loads(line)))
    stats = {}
    out = []
    with ThreadPoolExecutor(int(os.environ.get('NJOBS', '8'))) as ex:
        for idx, rec, why in ex.map(process, rows):
            stats[why] = stats.get(why, 0) + 1
            if rec:
                out.append(rec)
    with (BASE / 'hw_candidates.jsonl').open('w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print('considered', len(rows), 'selected', len(out), stats)
    print('in_first_2000', sum(r['idx'] < 2000 for r in out), 'label_reproduced', sum(r['label_reproduced'] for r in out))

if __name__ == '__main__':
    main()
