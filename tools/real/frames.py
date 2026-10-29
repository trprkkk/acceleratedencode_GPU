#!/usr/bin/env python3
"""Per-picture comparison of H.264 streams of the same source.

usage: tools/real/frames.py src.yuv w h a.264 [b.264 ...]

For every picture in display order: the type, the coded size in bytes, the
mean QP of its macroblocks and the luma PSNR, one column group per stream,
then the totals.  Needs numpy (PYTHONPATH=/tmp/dl/ext) and FFMPEG / ffprobe
(ffmpeg -debug qp gives the per-macroblock QPs).
"""
import math
import os
import re
import subprocess
import sys

import numpy as np

FF = os.environ.get('FFMPEG', '/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
src, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fs = w * h * 3 // 2
S = np.fromfile(src, np.uint8)


def analyse(f):
    p = subprocess.run([FF, '-v', 'debug', '-threads', '1', '-debug', 'qp', '-i', f, '-f', 'rawvideo', '-'],
                       capture_output=True, check=True)
    R = np.frombuffer(p.stdout, np.uint8).reshape(-1, fs)
    n = len(R)
    psnr = []
    for k in range(n):
        a = S[k * fs:k * fs + w * h].astype(np.float64)
        b = R[k, :w * h].astype(np.float64)
        mse = ((a - b) ** 2).mean()
        psnr.append(99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse))
    # -debug qp: a "New frame, type: X" line, then rows of 2-digit QPs (decode order)
    frames = []
    for line in p.stderr.decode(errors='replace').splitlines():
        m = re.search(r'New frame, type: (\w)', line)
        if m:
            frames.append([m.group(1), []])
            continue
        m = re.search(r'\]\s+(\d+)((?:\s*\d\d)+)\s*$', line)
        if frames and m and len(frames[-1][1]) < h // 16:
            row = m.group(2).replace(' ', '')
            qs = [int(row[i:i + 2]) for i in range(0, len(row), 2)]
            if len(qs) == w // 16:
                frames[-1][1].append(qs)
    return n, psnr, frames


streams = sys.argv[4:]
res = [analyse(f) for f in streams]
# sizes per decode-order picture via the encoder-independent NAL split
def nal_sizes(f):
    d = open(f, 'rb').read()
    starts = [m.start() for m in re.finditer(b'\x00\x00\x01', d)]
    sizes, cur = [], 0
    for i, s0 in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(d)
        t = d[s0 + 3] & 31
        if t in (1, 5):
            sizes.append(e - s0)
    return sizes


sizes = [nal_sizes(f) for f in streams]
# map decode order -> display order: POC isn't parsed; assume the qp-debug
# frames list is in decode order and pair it with sizes, then sort by the
# decoder's output order which -debug gives no handle on -- so print decode
# order for types/QP/size and display order for PSNR separately.
n = min(r[0] for r in res)
print('decode order: type size meanQP per stream')
for k in range(n):
    cols = []
    for (cnt, ps, fr), sz in zip(res, sizes):
        if k < len(fr):
            t, rows = fr[k]
            q = np.mean(rows) if rows else float('nan')
            cols.append('%s %6d %5.1f' % (t, sz[k] if k < len(sz) else -1, q))
    print('%3d  ' % k + '   |   '.join(cols))
print('display order PSNR')
for k in range(n):
    print('%3d  ' % k + '  '.join('%6.2f' % r[1][k] for r in res))
for f, (cnt, ps, fr), sz in zip(streams, res, sizes):
    print('%s: %d B, mean PSNR %.3f' % (f, sum(sz), np.mean(ps[:n])))
