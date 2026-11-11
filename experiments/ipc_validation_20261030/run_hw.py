"""Build one harness per candidate and measure user-mode cycles/instructions with perf_event_open."""
import json, os, re, subprocess, sys, statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
WORK = BASE / 'hw'
TEMPLATE = (BASE / 'harness.c.in').read_text()
UNROLL = 4
CPU = sorted(os.sched_getaffinity(0))[0]

def build(rec):
    d = WORK / f'f{rec["idx"]:05d}'
    ints = ptrs = 0
    proto, args = [], []
    for c in rec['sig']:
        if c == 'I':
            proto.append('long'); args.append('4L')
        elif c == 'P':
            proto.append('void *'); args.append(f'(void *)(b + {ptrs} * 262144 + 4096)'); ptrs += 1
        elif c == 'f':
            proto.append('float'); args.append('1.5f')
        else:
            proto.append('double'); args.append('1.5')
    call = '        acc += %s(%s);' % (rec['name'], ', '.join(args))
    src = (TEMPLATE.replace('@NAME@', rec['name']).replace('@PROTO@', ', '.join(proto) or 'void')
           .replace('@CALLS@', '\n'.join([call] * UNROLL)).replace('@UNROLL@', str(UNROLL)))
    (d / 'h.c').write_text(src)
    r = subprocess.run(['/usr/bin/gcc', '-O1', '-w', str(d / 'h.c'), str(d / 'f.o'), '-o', str(d / 'h')],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return d, r.returncode == 0

def measure(d, variant):
    try:
        r = subprocess.run(['taskset', '-c', str(CPU), str(d / 'h'), str(variant)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    if r.returncode != 0:
        return None, f'rc{r.returncode}'
    reps = [tuple(int(x) for x in l.split()) for l in r.stdout.strip().split('\n')]
    return reps, 'ok'

def main():
    cands = [json.loads(l) for l in (BASE / 'hw_candidates.jsonl').open()]
    out = (BASE / 'hw_measurements.jsonl').open('w')
    stats = {}
    for k, rec in enumerate(cands):
        d, ok = build(rec)
        if not ok:
            stats['build_fail'] = stats.get('build_fail', 0) + 1
            continue
        res = None
        for variant in (0, 1):
            reps, why = measure(d, variant)
            if reps:
                res = (variant, reps)
                break
        if not res:
            stats[why] = stats.get(why, 0) + 1
            continue
        variant, reps = res
        ipcs = [i / c for _, _, c, i in reps if c > 0]
        n = reps[0][1]
        med = statistics.median(ipcs)
        rec.update({'variant': variant, 'hw_ipc': med, 'hw_ipc_spread': (max(ipcs) - min(ipcs)) / med,
                    'hw_instr_per_call': statistics.median(i / n for _, n, _, i in reps),
                    'hw_cycles_per_call': statistics.median(c / n for _, n, c, _ in reps),
                    'has_branch': bool(re.search(r'^\s*j', (d / 'f.s').read_text(), re.M)), 'calls_per_rep': n,
                    'cpu': CPU})
        out.write(json.dumps(rec) + '\n'); out.flush()
        stats['measured'] = stats.get('measured', 0) + 1
        if k % 100 == 0:
            print(k, len(cands), stats, flush=True)
    print('done', stats)

if __name__ == '__main__':
    main()
