#!/usr/bin/env python3
"""Self / inclusive time from tools/prof/sprof.so samples.
usage: an.py EXE [SPROF_OUT] [N] [--thread main|other|all]"""
import sys, subprocess, collections, bisect
args = [a for a in sys.argv[1:] if not a.startswith('--')]
thr = 'all'
for a in sys.argv[1:]:
    if a.startswith('--thread='):
        thr = a.split('=')[1]
exe = args[0]; out = args[1] if len(args) > 1 else '/tmp/sprof.out'
N = int(args[2]) if len(args) > 2 else 30
base = None; samples = []
for l in open(out):
    if l.startswith('M '):
        f = l.split()
        if base is None and f[-1].endswith(exe.split('/')[-1]):
            lo = int(f[1].split('-')[0], 16); off = int(f[3], 16)
            base = lo - off
    elif l.startswith('S '):
        f = l.split()
        if thr == 'main' and f[1] != '1': continue
        if thr == 'other' and f[1] != '0': continue
        samples.append([int(x, 16) for x in f[2:]])
# symbol table
syms = []
for l in subprocess.run(['nm', '-C', '--defined-only', exe], capture_output=True, text=True).stdout.splitlines():
    p = l.split(None, 2)
    if len(p) == 3 and p[1] in 'tTwW':
        syms.append((int(p[0], 16), p[2]))
syms.sort(); addrs = [s[0] for s in syms]
def name(a):
    a -= base
    i = bisect.bisect_right(addrs, a) - 1
    return syms[i][1] if i >= 0 else '?'
cache = {}
def nm(a):
    if a not in cache: cache[a] = name(a)
    return cache[a]
selfc = collections.Counter(); incl = collections.Counter()
for s in samples:
    if not s: continue
    selfc[nm(s[0])] += 1
    for fn in set(nm(a) for a in s):
        incl[fn] += 1
T = len(samples)
print('%d samples (%s)' % (T, thr))
print('--- self'); [print('%6.1f%%  %s' % (100.0 * c / T, f)) for f, c in selfc.most_common(N)]
print('--- inclusive'); [print('%6.1f%%  %s' % (100.0 * c / T, f)) for f, c in incl.most_common(N)]
# --callers=FN : immediate callers of FN (by self samples)
for a in sys.argv[1:]:
    if a.startswith('--callers='):
        fn = a.split('=', 1)[1]; c = collections.Counter()
        for s in samples:
            ns = [nm(x) for x in s]
            for i, x in enumerate(ns[:-1]):
                if x == fn:
                    c[ns[i + 1] + ' <- ' + (ns[i + 2] if i + 2 < len(ns) else '')] += 1
                    break
        print('--- callers of', fn)
        for f, k in c.most_common(N): print('%6.1f%%  %s' % (100.0 * k / T, f))
# --children=FN : inclusive time of FN's immediate callees
for a in sys.argv[1:]:
    if a.startswith('--children='):
        fn = a.split('=', 1)[1]; c = collections.Counter(); tot = 0
        for s in samples:
            ns = [nm(x) for x in s]
            if fn not in ns: continue
            tot += 1
            i = len(ns) - 1 - ns[::-1].index(fn)   # outermost occurrence
            c[ns[i - 1] if i > 0 else '(self)'] += 1
        print('--- children of %s (%.1f%%)' % (fn, 100.0 * tot / T))
        for f, k in c.most_common(N): print('%6.1f%%  %s' % (100.0 * k / T, f))
