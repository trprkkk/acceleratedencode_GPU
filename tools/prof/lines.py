#!/usr/bin/env python3
"""Self samples of one function by source line (addr2line).
usage: lines.py EXE SPROF_OUT FUNCTION [N]"""
import sys, subprocess, collections, bisect
exe, out, fn = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 25
base = None; pcs = []
for l in open(out):
    if l.startswith('M '):
        f = l.split()
        if base is None and f[-1].endswith(exe.split('/')[-1]):
            base = int(f[1].split('-')[0], 16) - int(f[3], 16)
    elif l.startswith('S '):
        pcs.append(int(l.split()[2], 16))
syms = []
for l in subprocess.run(['nm', '--defined-only', exe], capture_output=True, text=True).stdout.splitlines():
    p = l.split(None, 2)
    if len(p) == 3 and p[1] in 'tTwW':
        syms.append((int(p[0], 16), p[2]))
syms.sort(); addrs = [s[0] for s in syms]
sel = []
for a in pcs:
    a -= base; i = bisect.bisect_right(addrs, a) - 1
    if i >= 0 and syms[i][1] == fn:
        sel.append(a)
c = collections.Counter(sel)
outl = subprocess.run(['addr2line', '-e', exe] + [hex(a) for a in c],
                      capture_output=True, text=True).stdout.splitlines()
lines = collections.Counter()
for (a, n), ln in zip(c.items(), outl):
    lines[ln.split('/')[-1].split(' ')[0]] += n
for ln, n in lines.most_common(N):
    print(n, ln)
print('total', len(sel))
