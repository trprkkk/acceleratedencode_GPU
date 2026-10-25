#!/usr/bin/env python3
"""BD-rate of two encoders over a set of QPs, with the overlap made explicit.

    ./tools/bdrate.py --clip w x h [--clip ...] [-q 22,25,28,31,34,37]
                      [--preset 2] [--extra "--trellis 1"] [--bin ./build/aegenc]
                      [--xq N] [--x264 /path/to/x264 | --ffmpeg /path/to/ffmpeg]

Prints, per clip, the rate and PSNR of every QP for both encoders and the
BD-rate (Bjontegaard, cubic-free log-rate integration over 21 samples) of ours
against the reference.

The number is only meaningful where the two curves *overlap*: the comparison
range is the intersection of the two PSNR spans, and a sample outside either
span is skipped.  When one encoder's curve stops well short of the other's the
remaining samples are a one-sided comparison and the BD-rate moves for reasons
that have nothing to do with quality -- which is how a deadzone change can look
like 45 points on a still clip.  This tool therefore prints the overlap and
marks the result `partial` when it is less than the full span of either curve,
so the reader can see how much of the curve the number came from.

Why a script and not a one-liner: the pair of encoders is compared at matched
PSNR, so both have to be run at the *same* QPs on the *same* source, and the
PSNR has to come from the same reconstruction convention (this script uses the
decoded stream for both, never `aegenc`'s own PSNR summary, which is about
0.9 dB optimistic relative to a per-plane comparison of the YUV files).
"""
import argparse
import json
import math
import os
import subprocess
import sys

build = os.environ.get('BUILD', './build')


def psnr(a, b, fs):
    """Mean per-frame PSNR of two I420 files over whole frames."""
    m = min(len(a), len(b)) // fs
    if m <= 0:
        return float('nan')
    total = 0.0
    for i in range(m):
        off = i * fs
        se = 0
        for j in range(fs):
            d = a[off + j] - b[off + j]
            se += d * d
        total += 10 * math.log10(255 * 255 / max(se / fs, 1e-12))
    return total / m


def decode(ffmpeg, path, w, h, out):
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'h264', '-i', path,
                    '-f', 'rawvideo', '-pix_fmt', 'yuv420p', out], check=True)
    return open(out, 'rb').read()


CACHE = '/tmp/aegenc_bdref.json'


def cache_load():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {}


def cache_save(c):
    try:
        json.dump(c, open(CACHE, 'w'))
    except Exception:
        pass


def ref_point(args, clip, q, w, h, n, cache):
    """Rate and PSNR of the reference encoder, remembered across runs.

    The reference does not depend on what is being changed in our encoder, so
    re-running it for every preset, every experiment and every tool run is the
    same encode repeated; it is by far the slowest part of a sweep."""
    st = os.stat(clip)
    key = '%s:%d:%d:%dx%d:%d:%s:%s:%s' % (os.path.abspath(clip), st.st_size,
                                          int(st.st_mtime), w, h, q,
                                          args.xpreset, args.xopts,
                                          'x264' if args.x264 else 'ffmpeg')
    if key in cache:
        return tuple(cache[key])
    fs = w * h * 3 // 2
    src = open(clip, 'rb').read()
    rb = x264(args, clip, q, w, h)
    rp = psnr(src, decode(args.ffmpeg, '/tmp/bd_ref.264', w, h, '/tmp/bd_ref.yuv'), fs)
    cache[key] = [rb, rp]
    cache_save(cache)
    return rb, rp


def ours(args, clip, q, w, h, n):
    dst = '/tmp/bd_ours.264'
    subprocess.run([args.bin, '-w', str(w), '-h', str(h), '-i', clip, '-o', dst,
                    '-q', str(q), '-n', str(n), '-p', str(args.preset),
                    '--dump-recon', '/tmp/bd_ours.rec'] + args.extra.split(),
                   check=True, capture_output=True)
    return os.path.getsize(dst) * 8 / 1000.0


def x264(args, clip, q, w, h):
    dst = '/tmp/bd_ref.264'
    if args.x264:
        subprocess.run([args.x264, '--input-res', '%dx%d' % (w, h), '--fps', '30',
                        '-q', str(q), '-o', dst, '--quiet'] +
                       args.xopts.split() + [clip], check=True)
    else:
        subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error',
                        '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                        '-s', '%dx%d' % (w, h), '-r', '30', '-i', clip,
                        '-c:v', 'libx264', '-preset', args.xpreset] +
                       (['-x264-params', args.xopts] if args.xopts else []) +
                       ['-qp', str(q), '-f', 'h264', '-y', dst], check=True)
    return os.path.getsize(dst) * 8 / 1000.0


def bdrate(ours_pts, ref_pts):
    """BD-rate of ours against the reference over the shared PSNR span.

    The standard sign: negative means ours needs *fewer* bits than the
    reference at matched PSNR, i.e. negative is better."""
    def interp(pts, p):
        pts = sorted(pts, key=lambda t: t[1])
        for i in range(len(pts) - 1):
            if pts[i][1] <= p <= pts[i + 1][1]:
                x0, y0 = math.log(pts[i][0]), pts[i][1]
                x1, y1 = math.log(pts[i + 1][0]), pts[i + 1][1]
                if y1 == y0:
                    return x0
                return x0 + (x1 - x0) * (p - y0) / (y1 - y0)
        return None
    lo = max(min(y for _, y in ours_pts), min(y for _, y in ref_pts))
    hi = min(max(y for _, y in ours_pts), max(y for _, y in ref_pts))
    if hi <= lo:
        return float('nan'), 0.0, (hi, lo)
    diffs = []
    for i in range(21):
        p = lo + (hi - lo) * i / 20
        a, b = interp(ours_pts, p), interp(ref_pts, p)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return float('nan'), 0.0, (lo, hi)
    ours_span = max(y for _, y in ours_pts) - min(y for _, y in ours_pts)
    ref_span = max(y for _, y in ref_pts) - min(y for _, y in ref_pts)
    cover = (hi - lo) / max(ours_span, ref_span)
    return math.exp(sum(diffs) / len(diffs)) - 1, cover, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', nargs=4, action='append', metavar=('PATH', 'W', 'H', 'N'),
                    required=True)
    ap.add_argument('-q', '--qps', default='22,25,28,31,34,37')
    ap.add_argument('--preset', type=int, default=2)
    ap.add_argument('--extra', default='', help='extra aegenc arguments')
    ap.add_argument('--xopts', default='', help='x264 parameters (or -x264-params)')
    ap.add_argument('--xpreset', default='medium')
    ap.add_argument('--x264', default='', help='use an x264 binary instead of ffmpeg')
    ap.add_argument('--bin', default=os.path.join(build, 'aegenc'))
    ap.add_argument('--nocache', action='store_true',
                    help='re-run the reference encoder instead of reusing '
                         '/tmp/aegenc_bdref.json')
    ap.add_argument('--ffmpeg',
                    default=os.environ.get('FFMPEG',
                                           '/tmp/dl/ext/imageio_ffmpeg/binaries/'
                                           'ffmpeg-linux-x86_64-v7.0.2'))
    a = ap.parse_args()
    a.extra = a.extra.strip()
    a.xopts = a.xopts.strip()
    qs = [int(x) for x in a.qps.split(',')]

    cache = {} if a.nocache else cache_load()
    for path, w, h, n in a.clip:
        w, h, n = int(w), int(h), int(n)
        fs = w * h * 3 // 2
        src = open(path, 'rb').read()
        o_pts, r_pts = [], []
        print('== %s %dx%d %d frames, preset %d %s' %
              (path, w, h, n, a.preset, a.extra))
        for q in qs:
            ob = ours(a, path, q, w, h, n)
            orec = open('/tmp/bd_ours.rec', 'rb').read()
            rb, rp = ref_point(a, path, q, w, h, n, cache) if cache is not None \
                else (x264(a, path, q, w, h), float('nan'))
            op = psnr(src, orec, fs)
            o_pts.append((ob, op))
            r_pts.append((rb, rp))
            print('   qp %2d  ours %8.1f kbit %7.3f dB   ref %8.1f kbit %7.3f dB'
                  % (q, ob, op, rb, rp))
        d, cover, (lo, hi) = bdrate(o_pts, r_pts)
        note = '' if cover > 0.98 else '  (partial: %.0f%% of the span)' % (100 * cover)
        print('   BD-rate %+6.2f%% over %.2f..%.2f dB: ours spends %5.1f%% %s '
              'bits than the reference at matched PSNR%s'
              % (100 * d, lo, hi, 100 * abs(d), 'more' if d >= 0 else 'fewer',
                 note))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
