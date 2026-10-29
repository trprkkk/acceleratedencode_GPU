#!/usr/bin/env python3
"""BD-rate of one aegenc configuration against another (ours vs ours).

    ./tools/abbd.py --clip clip.yuv 320x192 [--clip ...] -n 30 \
        [-q 22,25,28,31,34,37] [-p 3] --a "--ref 1" --b "--ref 2"

Leading NAME=value words in --a/--b are set in that side's environment, so an
AEG_* experiment switch can be compared against the build's default:
--a "" --b "AEG_FOO=1 --bframes 3".

Both configurations are encoded at the same QPs on the same source; the
PSNR is measured from the stream *decoded by ffmpeg* (`--ffmpeg`, default
$FFMPEG or `ffmpeg`), so B-picture streams are compared in display order and
nothing depends on the encoder's own reconstruction.  Negative = B needs fewer
bits than A at matched PSNR.  Prints the encode time of each side too, because
every quality knob in this encoder is a quality-for-time trade.

No reference encoder is involved, which is the point: it isolates the effect of
one option and runs on a machine with nothing but the tree and a decoder.
"""
import argparse
import math
import os
import subprocess
import sys
import time


def psnr(a, b, fs):
    m = min(len(a), len(b)) // fs
    if m <= 0:
        return float('nan')
    tot = 0.0
    try:
        import numpy as np
        A = np.frombuffer(a[:m * fs], dtype=np.uint8).reshape(m, fs).astype(np.int32)
        B = np.frombuffer(b[:m * fs], dtype=np.uint8).reshape(m, fs).astype(np.int32)
        mse = ((A - B) ** 2).mean(axis=1)
        if os.environ.get('ABBD_GLOBAL'):
            # PSNR of the whole sequence's MSE: a picture coded losslessly
            # (a black frame) cannot dominate it the way it dominates the
            # per-picture average, where it counts as ~100+ dB
            return float(10 * np.log10(255.0 * 255.0 / max(mse.mean(), 1e-12)))
        return float(np.mean(10 * np.log10(255.0 * 255.0 / np.maximum(mse, 1e-12))))
    except ImportError:
        pass
    for i in range(m):
        se = 0
        o = i * fs
        for j in range(fs):
            d = a[o + j] - b[o + j]
            se += d * d
        tot += 10 * math.log10(255 * 255 / max(se / fs, 1e-12))
    return tot / m


def encode(args, extra, clip, w, h, q, tag):
    out = '/tmp/abbd_%s.264' % tag
    dec = '/tmp/abbd_%s.yuv' % tag
    # leading NAME=value tokens are environment variables for this side only
    # (the AEG_* experiment switches), the rest are aegenc options; BIN=path
    # runs that side with another build (e.g. a worktree of the last commit)
    env = dict(os.environ)
    toks = extra.split()
    binary = args.bin
    while toks and '=' in toks[0] and not toks[0].startswith('-'):
        k, v = toks.pop(0).split('=', 1)
        if k == 'BIN':
            binary = v
        else:
            env[k] = v
    t0 = time.time()
    subprocess.run([binary, '-w', str(w), '-h', str(h), '-i', clip, '-o', out,
                    '-q', str(q), '-n', str(args.n), '-p', str(args.preset)] +
                   toks, check=True, capture_output=True, env=env)
    dt = time.time() - t0
    subprocess.run([args.ffmpeg, '-v', 'error', '-y', '-i', out, '-f', 'rawvideo',
                    '-pix_fmt', 'yuv420p', dec], check=True)
    fs = w * h * 3 // 2
    src = open(clip, 'rb').read()[:args.n * fs]
    return os.path.getsize(out) * 8, psnr(src, open(dec, 'rb').read(), fs), dt


def bd(ra, pa, rb, pb):
    """Bjontegaard rate difference by piecewise-linear log-rate interpolation
    over the overlapping PSNR span (21 samples)."""
    def curve(r, p):
        pts = sorted(zip(p, [math.log(x) for x in r]))
        return pts

    def interp(pts, x):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
        return None

    A, B = curve(ra, pa), curve(rb, pb)
    lo = max(A[0][0], B[0][0])
    hi = min(A[-1][0], B[-1][0])
    if hi <= lo:
        return float('nan')
    s = 0.0
    n = 21
    for i in range(n):
        x = lo + (hi - lo) * i / (n - 1)
        s += interp(B, x) - interp(A, x)
    return (math.exp(s / n) - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', nargs=2, action='append', required=True,
                    metavar=('YUV', 'WxH'))
    ap.add_argument('-n', type=int, default=30)
    ap.add_argument('-q', default='22,25,28,31,34,37')
    ap.add_argument('-p', '--preset', type=int, default=3)
    ap.add_argument('--a', default='')
    ap.add_argument('--b', default='')
    ap.add_argument('--bin', default='./build/aegenc')
    ap.add_argument('--ffmpeg', default=os.environ.get('FFMPEG', 'ffmpeg'))
    args = ap.parse_args()
    qs = [int(x) for x in args.q.split(',')]
    res = []
    tta = ttb = 0.0
    for clip, wh in args.clip:
        w, h = (int(x) for x in wh.lower().split('x'))
        ra, pa, rb, pb = [], [], [], []
        ta = tb = 0.0
        for q in qs:
            r1, p1, t1 = encode(args, args.a, clip, w, h, q, 'a')
            r2, p2, t2 = encode(args, args.b, clip, w, h, q, 'b')
            ra.append(r1); pa.append(p1); rb.append(r2); pb.append(p2)
            ta += t1; tb += t2
            print('  %-28s q%-2d  A %8d b %6.3f dB   B %8d b %6.3f dB' %
                  (os.path.basename(clip), q, r1, p1, r2, p2))
        d = bd(ra, pa, rb, pb)
        res.append(d)
        tta += ta
        ttb += tb
        print('%-30s BD-rate B vs A: %+6.2f %%   time A %.2fs B %.2fs (%+.0f %%)' %
              (os.path.basename(clip), d, ta, tb, (tb / ta - 1) * 100 if ta else 0))
    print('mean BD-rate: %+.2f %%   total time A %.1fs B %.1fs (%+.1f %%)' %
          (sum(res) / len(res), tta, ttb, (ttb / tta - 1) * 100 if tta else 0))


if __name__ == '__main__':
    sys.exit(main())
