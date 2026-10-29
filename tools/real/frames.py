#!/usr/bin/env python3
"""Per-picture comparison of H.264 streams of the same source.

usage: tools/real/frames.py src.yuv w h a.264 [b.264 ...]

For every picture in display order: the slice type, the coded size in bytes
(slice NAL units), the mean QP of its macroblocks (ffmpeg -debug qp, which
reports skipped macroblocks at the QP before them) and the luma PSNR, one
column group per stream; then per-type totals.  The display order comes from
the slice headers (pic_order_cnt_type 0 only).  Needs numpy
(PYTHONPATH=/tmp/dl/ext) and FFMPEG.
"""
import math
import os
import re
import subprocess
import sys

import numpy as np

FF = os.environ.get('FFMPEG', '/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


class Bits:
    def __init__(self, b):
        # strip emulation prevention
        out = bytearray()
        z = 0
        for c in b:
            if z >= 2 and c == 3:
                z = 0
                continue
            out.append(c)
            z = z + 1 if c == 0 else 0
        self.b = bytes(out)
        self.p = 0

    def u(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.b[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
        return (1 << z) - 1 + self.u(z)


def pictures(f):
    """[(poc_key, type, size)] in decode order."""
    d = open(f, 'rb').read()
    starts = [m.start() + 3 for m in re.finditer(b'\x00\x00\x01', d)]
    sps = {}
    pics = []
    base = 0
    for i, s0 in enumerate(starts):
        e = starts[i + 1] - 3 if i + 1 < len(starts) else len(d)
        nal = d[s0:e]
        t = nal[0] & 31
        b = Bits(nal[1:64])
        if t == 7:
            prof = b.u(8); b.u(16); b.ue()
            if prof in (100, 110, 122, 244, 44, 83, 86, 118, 128):
                cf = b.ue()
                if cf == 3:
                    b.u(1)
                b.ue(); b.ue(); b.u(1)
                if b.u(1):
                    raise SystemExit('scaling matrices not supported')
            sps['lfn'] = b.ue() + 4
            sps['poct'] = b.ue()
            if sps['poct'] == 0:
                sps['lpoc'] = b.ue() + 4
        elif t in (1, 5):
            first_mb = b.ue()
            st = b.ue() % 5
            b.ue()
            b.u(sps['lfn'])
            if t == 5:
                b.ue()
            poc = b.u(sps['lpoc']) if sps.get('poct') == 0 else len(pics)
            if first_mb == 0:
                if t == 5:
                    base = (pics[-1][0] + 1) if pics else 0
                pics.append([base * 0 + poc + (base << 16), 'PBI'[st] if st < 3 else '?', 0])
            pics[-1][2] += e - s0 + 3
    return pics


def analyse(f, S, w, h):
    fs = w * h * 3 // 2
    p = subprocess.run([FF, '-v', 'debug', '-threads', '1', '-debug', 'qp', '-i', f,
                        '-f', 'rawvideo', '-'], capture_output=True, check=True)
    R = np.frombuffer(p.stdout, np.uint8).reshape(-1, fs)
    psnr = []
    for k in range(len(R)):
        a = S[k * fs:k * fs + w * h].astype(np.float64)
        mse = ((a - R[k, :w * h].astype(np.float64)) ** 2).mean()
        psnr.append(99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse))
    # -debug qp prints the pictures in output (display) order
    qps = []
    cur = None
    for line in p.stderr.decode(errors='replace').splitlines():
        if 'New frame, type:' in line:
            cur = []
            qps.append(cur)
            continue
        m = re.search(r'\]\s+(\d+) (\d+)\s*$', line)
        if cur is not None and m and len(m.group(2)) == 2 * (w // 16):
            row = m.group(2)
            cur.extend(int(row[i:i + 2]) for i in range(0, len(row), 2))
    pics = sorted(pictures(f), key=lambda x: x[0])
    return [(pics[k][1], pics[k][2], float(np.mean(qps[k])) if k < len(qps) and qps[k] else float('nan'),
             psnr[k]) for k in range(min(len(pics), len(psnr)))]


def main():
    src, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    S = np.fromfile(src, np.uint8)
    streams = sys.argv[4:]
    res = [analyse(f, S, w, h) for f in streams]
    n = min(len(r) for r in res)
    print('pic  ' + '   |   '.join('%-24s' % os.path.basename(f)[:24] for f in streams))
    for k in range(n):
        print('%3d  ' % k + '   |   '.join('%s %6d %5.1f %6.2f' % r[k] for r in res))
    for f, r in zip(streams, res):
        tot = sum(x[1] for x in r[:n])
        mse = np.mean([10 ** (-x[3] / 10) for x in r[:n]])
        line = '%s: %d B, PSNR %.3f (global)' % (os.path.basename(f), tot, -10 * math.log10(mse))
        for t in 'IPB':
            sel = [x for x in r[:n] if x[0] == t]
            if sel:
                line += '   %s: %d x %6.0f B %5.1f qp %.2f dB' % (
                    t, len(sel), np.mean([x[1] for x in sel]), np.nanmean([x[2] for x in sel]),
                    np.mean([x[3] for x in sel]))
        print(line)


main()
