#!/usr/bin/env python3
"""The comparison against x264 (and x265): every preset against every preset.

    tools/x264cmp.py run    [--only REGEX] [--clips main,val] [--mode qp|crf|10]
    tools/x264cmp.py report [--clips main,val] [--mode qp|crf|10]
    tools/x264cmp.py speed  [--only REGEX] [--threads 1|0] [--runs 3]

`run` encodes the rate-quality curve of each configuration on each clip once
and caches every point (bytes, PSNR, SSIM, CPU and wall time) in $X264CMP
(default /tmp/x264cmp.json), keyed on the clip, the value and, for aegenc, the
binary's hash -- so `report` can pair any two configurations without
re-encoding either, and a rebuilt aegenc is re-measured automatically.

Conventions (the same as tools/bdrate.py, so the numbers line up with the rest
of docs/STATUS.md):
  * PSNR is the *global* PSNR of the sequence: the MSE over every sample of
    every plane of every frame (BD_GLOBAL=1 in bdrate.py), on the decoded
    stream (FFmpeg) against the source.  SSIM is FFmpeg's ssim filter "All",
    in dB (-10 log10(1 - SSIM)).
  * BD-rate integrates log-rate linearly over the PSNR span both curves cover
    (21 samples).  Negative = aegenc needs fewer bits.  `ovl` is the shared
    span as a fraction of the smaller curve's span; under 0.5 the cell is
    marked with `*`.
  * x264 and x265 run with --tune psnr (psy-rd off, AQ off), as aegenc does
    at constant QP (AQ only acts with --crf / rate control), so a PSNR
    comparison is the fair one.  Everything else is the encoders' defaults.
  * Quality runs are single-threaded (aegenc -t 1, x264/x265 threads 1),
    which is also the setting where x264's output does not depend on the
    machine.  Speed is measured separately (`speed`, long clips): on 30
    pictures FFmpeg's start-up is a large part of an x264 run.
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = os.environ.get('FFMPEG', '/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
AEG = os.environ.get('AEGENC', os.path.join(ROOT, 'build', 'aegenc'))
CACHE = os.environ.get('X264CMP', '/tmp/x264cmp.json')
REAL = os.environ.get('REAL', '/tmp/real')
REAL2 = os.environ.get('REAL2', '/tmp/real2')
TMP = tempfile.mkdtemp(prefix='x264cmp_')

try:
    import numpy as np
except ImportError:
    sys.path.insert(0, '/tmp/dl/ext')
    import numpy as np

CLIPS = {
    'main': [('cock', REAL + '/cock.yuv', 640, 352), ('vtest', REAL + '/vtest.yuv', 384, 288),
             ('mega', REAL + '/mega.yuv', 352, 256), ('short', REAL + '/short.yuv', 320, 240),
             ('tree', REAL + '/tree.yuv', 320, 240)],
    'val': [('cock2', REAL2 + '/cock.yuv', 640, 352), ('vtest2', REAL2 + '/vtest.yuv', 384, 288),
            ('mega2', REAL2 + '/mega.yuv', 352, 256), ('tree2', REAL2 + '/tree.yuv', 320, 240),
            ('vtest3', REAL2 + '/vtest2.yuv', 384, 288), ('mega3', REAL2 + '/mega2.yuv', 352, 256)],
    # tools/real/longclips.sh: 240-300 pictures, for GOP-scale behaviour and speed
    'long': [('cockL', REAL + '/cockL.yuv', 640, 352), ('vtestL', REAL + '/vtestL.yuv', 384, 288),
             ('megaL', REAL + '/megaL.yuv', 352, 256)],
}
QPS = [22, 25, 28, 31, 34, 37]
CRFS = [18, 21, 24, 27, 30, 33]
X264_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium',
                'slow', 'slower', 'veryslow']
X265_PRESETS = ['fast', 'medium', 'slow']


def configs():
    """name -> (encoder, preset, extra).  'aeg' presets are 0..4."""
    c = {}
    for p in range(5):
        c['aeg-p%d' % p] = ('aeg', p, [])
    for p in X264_PRESETS:
        c['x264-' + p] = ('x264', p, ['-tune', 'psnr'])
    # x264 turns AQ off by itself at constant QP, so the old reference
    # (medium, psy=0) is this same encoder there -- measured identical
    for p in X265_PRESETS:
        c['x265-' + p] = ('x265', p, ['-tune', 'psnr'])
    return c


def sha(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:12]


AEG_ID = None


def cache_load():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {}


def cache_save(c):
    tmp = '%s.%d' % (CACHE, os.getpid())
    json.dump(c, open(tmp, 'w'), indent=0, sort_keys=True)
    os.replace(tmp, CACHE)


def src10(path, w, h):
    """The 10-bit source both encoders get in --mode 10: the 8-bit clip
    shifted left by two, as yuv420p10le -- so neither encoder's (or
    swscale's) up-conversion is part of the comparison."""
    out = path[:-4] + '.p10.yuv'
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        (np.fromfile(path, np.uint8).astype('<u2') << 2).tofile(out)
    return out


def timed(cmd):
    r0 = os.times()
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t1 = time.perf_counter()
    r1 = os.times()
    return (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system), t1 - t0


def encode_cmd(cfg, clip, w, h, n, mode, v, dst, threads=1):
    enc, preset, extra = cfg
    ten = mode == '10'
    if enc == 'aeg':
        cmd = [AEG, '-w', str(w), '-h', str(h), '-i', clip, '-o', dst, '-n', str(n),
               '-p', str(preset), '-t', str(threads)]
        if ten:
            cmd += ['--bit-depth', '10', '--input-depth', '10']
        if mode == 'crf':
            cmd += ['--crf', str(v), '--aq', '0']
        else:
            cmd += ['-q', str(v)]
        return cmd + extra
    lib = 'libx264' if enc == 'x264' else 'libx265'
    pf = 'yuv420p10le' if ten else 'yuv420p'
    cmd = [FF, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', pf,
           '-s', '%dx%d' % (w, h), '-r', '30', '-i', clip, '-frames:v', str(n),
           '-c:v', lib, '-preset', preset] + extra
    if threads == 1:
        if enc == 'x264':
            cmd += ['-threads', '1']
        else:
            cmd += ['-x265-params', 'pools=1:frame-threads=1:wpp=0:log-level=error']
    elif enc == 'x265':
        cmd += ['-x265-params', 'log-level=error']
    if ten:
        cmd += ['-pix_fmt', 'yuv420p10le']
    if mode == 'crf':
        cmd += ['-crf', str(v)]
    else:
        # x264/x265 count a 10-bit QP from QP' (QpBdOffset 12 below aegenc's)
        cmd += ['-qp', str(v + 12 if ten else v)]
    return cmd + ['-f', 'h264' if enc == 'x264' else 'hevc', dst]


def measure(cfg, clip, w, h, n, mode, v):
    enc = cfg[0]
    ten = mode == '10'
    dst = os.path.join(TMP, 'o.' + ('265' if enc == 'x265' else '264'))
    if os.path.exists(dst):
        os.remove(dst)
    cpu, wall = timed(encode_cmd(cfg, clip, w, h, n, mode, v, dst))
    bits = os.path.getsize(dst) * 8
    pf = 'yuv420p10le' if ten else 'yuv420p'
    dec = os.path.join(TMP, 'd.yuv')
    subprocess.run([FF, '-v', 'error', '-y', '-i', dst, '-f', 'rawvideo', '-pix_fmt', pf, dec],
                   check=True)
    dt = '<u2' if ten else np.uint8
    a = np.fromfile(clip, dt).astype(np.int64)
    b = np.fromfile(dec, dt).astype(np.int64)
    fs = w * h * 3 // 2
    m = min(len(a), len(b)) // fs
    if m != n:
        raise RuntimeError('%s: decoded %d of %d frames' % (clip, m, n))
    a, b = a[:m * fs], b[:m * fs]
    peak = 1023 if ten else 255
    d = (a - b).reshape(m, fs)
    mse = float((d * d).mean())
    mse_y = float((d[:, :w * h] ** 2).mean())
    psnr = 10 * math.log10(peak * peak / max(mse, 1e-12))
    psnr_y = 10 * math.log10(peak * peak / max(mse_y, 1e-12))
    s = subprocess.run([FF, '-hide_banner', '-f', 'rawvideo', '-pix_fmt', pf, '-s', '%dx%d' % (w, h),
                        '-i', dec, '-f', 'rawvideo', '-pix_fmt', pf, '-s', '%dx%d' % (w, h),
                        '-i', clip, '-frames:v', str(n), '-lavfi', 'ssim', '-f', 'null', '-'],
                       check=True, capture_output=True, text=True).stderr
    ssim = float(re.search(r'All:([0-9.]+)', s).group(1))
    return {'kbps': bits * 30 / n / 1000.0, 'psnr': psnr, 'psnr_y': psnr_y,
            'ssim_db': -10 * math.log10(max(1 - ssim, 1e-12)), 'cpu': cpu, 'wall': wall, 'n': n}


def clip_list(names):
    out = []
    for s in names.split(','):
        out += CLIPS[s]
    return out


def key(name, cfg, clip, mode, v):
    st = os.stat(clip)
    k = '%s|%s|%d|%d|%s|%s' % (name, os.path.abspath(clip), st.st_size, int(st.st_mtime), mode, v)
    if cfg[0] == 'aeg':
        k += '|' + AEG_ID
    return k


def values(mode):
    return CRFS if mode == 'crf' else QPS


def nframes(clip, w, h, ten):
    return os.path.getsize(clip) // (w * h * 3 // 2 * (2 if ten else 1))


def cmd_run(a):
    cache = cache_load()
    cf = configs()
    names = [k for k in cf if re.search(a.only, k)]
    for name in names:
        for tag, clip, w, h in clip_list(a.clips):
            if a.mode == '10':
                clip = src10(clip, w, h)
            n = nframes(clip, w, h, a.mode == '10')
            for v in values(a.mode):
                k = key(name, cf[name], clip, a.mode, v)
                if k in cache:
                    continue
                cache[k] = measure(cf[name], clip, w, h, n, a.mode, v)
                cache_save(cache)
                r = cache[k]
                print('%-18s %-7s %s %2s %9.1f kb/s %6.2f dB %6.2f dB(ssim) %6.2fs' %
                      (name, tag, a.mode, v, r['kbps'], r['psnr'], r['ssim_db'], r['cpu']),
                      flush=True)


def bd(ours, ref):
    """bdrate.py's BD-rate: piecewise-linear log-rate over the shared span."""
    def interp(pts, p):
        pts = sorted(pts, key=lambda t: t[1])
        for i in range(len(pts) - 1):
            if pts[i][1] <= p <= pts[i + 1][1]:
                x0, y0 = math.log(pts[i][0]), pts[i][1]
                x1, y1 = math.log(pts[i + 1][0]), pts[i + 1][1]
                return x0 if y1 == y0 else x0 + (x1 - x0) * (p - y0) / (y1 - y0)
        return None
    lo = max(min(y for _, y in ours), min(y for _, y in ref))
    hi = min(max(y for _, y in ours), max(y for _, y in ref))
    span = min(max(y for _, y in ours) - min(y for _, y in ours),
               max(y for _, y in ref) - min(y for _, y in ref))
    if hi <= lo:
        return float('nan'), 0.0
    d = []
    for i in range(21):
        p = lo + (hi - lo) * i / 20
        x, y = interp(ours, p), interp(ref, p)
        if x is not None and y is not None:
            d.append(x - y)
    return (math.exp(sum(d) / len(d)) - 1) * 100, (hi - lo) / span


def curves(cache, name, cfg, clips, mode):
    out = {}
    for tag, clip, w, h in clips:
        if mode == '10':
            clip = clip[:-4] + '.p10.yuv'
        pts = []
        for v in values(mode):
            try:
                pts.append(cache[key(name, cfg, clip, mode, v)])
            except (KeyError, FileNotFoundError):
                pts = None
                break
        if pts:
            out[tag] = pts
    return out


def cmd_report(a):
    cache = cache_load()
    cf = configs()
    clips = clip_list(a.clips)
    tags = [c[0] for c in clips]
    cv = {n: curves(cache, n, cf[n], clips, a.mode) for n in cf}
    cv = {n: c for n, c in cv.items() if len(c) == len(clips)}
    ours = [n for n in cv if n.startswith('aeg')]
    refs = [n for n in cv if not n.startswith('aeg')]
    metric = a.metric

    def pair(o, r):
        res = {}
        for t in tags:
            po = [(p['kbps'], p[metric]) for p in cv[o][t]]
            pr = [(p['kbps'], p[metric]) for p in cv[r][t]]
            res[t] = bd(po, pr)
        return res

    print('### BD-rate (%s), mean over %d clips (%s), mode %s\n' % (metric, len(tags), a.clips, a.mode))
    print('| aegenc \\ ref | ' + ' | '.join(refs) + ' |')
    print('|---|' + '---|' * len(refs))
    for o in ours:
        cells = []
        for r in refs:
            pr = pair(o, r)
            m = statistics.mean(v[0] for v in pr.values())
            low = min(v[1] for v in pr.values())
            cells.append('%+.1f%%%s' % (m, '*' if low < 0.5 else ''))
        print('| %s | %s |' % (o, ' | '.join(cells)))
    if a.detail:
        for o, r in [x.split(':') for x in a.detail.split(',')]:
            pr = pair(o, r)
            print('\n### %s vs %s per clip (%s)\n' % (o, r, metric))
            print('| ' + ' | '.join(tags) + ' | mean |')
            print('|' + '---|' * (len(tags) + 1))
            print('| ' + ' | '.join('%+.2f%%%s' % (pr[t][0], '*' if pr[t][1] < 0.5 else '')
                                     for t in tags) +
                  ' | %+.2f%% |' % statistics.mean(v[0] for v in pr.values()))


def cmd_speed(a):
    """Wall-clock fps on the long clips, median of --runs, all configs at the
    given thread setting (0 = each encoder's default: aegenc frame threads,
    x264/x265 their own thread pools).  FFmpeg's own share (reading the raw
    file and its start-up) is measured with -c:v rawvideo -f null and taken
    off the x264/x265 times."""
    cf = configs()
    names = [k for k in cf if re.search(a.only, k)]
    clips = [(c.split(':')[0], c.split(':')[1], int(c.split(':')[2]), int(c.split(':')[3]))
             for c in a.long.split(',')]
    print('| config | ' + ' | '.join('%s fps' % c[0] for c in clips) + ' |')
    print('|---|' + '---|' * len(clips))
    base = {}
    for tag, clip, w, h in clips:
        n = nframes(clip, w, h, False)
        ts = []
        for _ in range(a.runs):
            ts.append(timed([FF, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt',
                             'yuv420p', '-s', '%dx%d' % (w, h), '-r', '30', '-i', clip,
                             '-c:v', 'rawvideo', '-f', 'null', '-'])[1])
        base[tag] = statistics.median(ts)
    for name in names:
        cells = []
        for tag, clip, w, h in clips:
            n = nframes(clip, w, h, False)
            ts = []
            for _ in range(a.runs):
                dst = os.path.join(TMP, 's.out')
                ts.append(timed(encode_cmd(cf[name], clip, w, h, n, 'qp', a.qp, dst,
                                           threads=a.threads))[1])
            t = statistics.median(ts) - (base[tag] if cf[name][0] != 'aeg' else 0)
            cells.append('%.1f' % (n / t))
        print('| %s | %s |' % (name, ' | '.join(cells)), flush=True)


def main():
    global AEG_ID
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report', 'speed'])
    ap.add_argument('--only', default='.')
    ap.add_argument('--clips', default='main')
    ap.add_argument('--mode', default='qp', choices=['qp', 'crf', '10'])
    ap.add_argument('--metric', default='psnr', choices=['psnr', 'psnr_y', 'ssim_db'])
    ap.add_argument('--detail', default='')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--qp', type=int, default=27)
    ap.add_argument('--long', default='')
    a = ap.parse_args()
    AEG_ID = sha(AEG)
    {'run': cmd_run, 'report': cmd_report, 'speed': cmd_speed}[a.cmd](a)


if __name__ == '__main__':
    main()
