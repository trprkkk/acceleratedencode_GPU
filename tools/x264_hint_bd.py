#!/usr/bin/env python3
"""BD-rate for x264 fork hint modes.

Run with Windows Python: py -3 tools/x264_hint_bd.py --hint-mode hex ...
Hints are generated with the same x264.exe and --threads 1.
"""
import argparse
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from bdrate import bdrate

BUILD = os.path.join(ROOT, 'build', 'work')
X264 = os.path.join(ROOT, 'forks', 'x264', 'x264.exe')
FF = ['wsl.exe', '-d', 'Ubuntu', '--', 'ffmpeg']


def wsl_path(p):
    return p.replace('Q:\\', '/mnt/q/').replace('\\', '/')


def decode(path, out):
    subprocess.run(FF + ['-v', 'error', '-y', '-i', wsl_path(path),
                         '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                         wsl_path(out)], check=True)


def scores(dec_path, src, w, h):
    d = np.fromfile(dec_path, dtype=np.uint8)
    s = np.fromfile(src, dtype=np.uint8)
    fs = w * h * 3 // 2
    n = min(len(d), len(s)) // fs
    ps = []
    ss = []
    for k in range(n):
        a = s[k*fs:k*fs + w*h].reshape(h, w).astype(np.float64)
        b = d[k*fs:k*fs + w*h].reshape(h, w).astype(np.float64)
        ps.append(10 * math.log10(255.0**2 / max(((a-b)**2).mean(), 1e-12)))
        gh, gw = h // 8, w // 8
        aa = a[:gh*8, :gw*8].reshape(gh, 8, gw, 8)
        bb = b[:gh*8, :gw*8].reshape(gh, 8, gw, 8)
        ma = aa.mean(axis=(1,3)); mb = bb.mean(axis=(1,3))
        va = ((aa-ma[:,None,:,None])**2).mean(axis=(1,3))
        vb = ((bb-mb[:,None,:,None])**2).mean(axis=(1,3))
        cov = ((aa-ma[:,None,:,None])*(bb-mb[:,None,:,None])).mean(axis=(1,3))
        c1 = (0.01*255)**2; c2 = (0.03*255)**2
        ss.append((((2*ma*mb+c1)*(2*cov+c2)) /
                   ((ma*ma+mb*mb+c1)*(va+vb+c2))).mean())
    return statistics.mean(ps), statistics.mean(ss)


def run(args, out):
    t0 = time.perf_counter()
    subprocess.run([X264, '--input-res', '1920x1080', '--fps', '30',
                    '--frames', str(args.frames), '--threads', '1',
                    '--preset', 'medium', '--crf', str(args.crf),
                    '--me', args.me, '--subme', str(args.subme),
                    '-o', out] + args.extra.split() + [args.y4m],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def make_hint(args):
    dump = os.path.join(BUILD, 'hint_%s.dump' % args.hint_mode)
    if os.path.exists(dump):
        os.remove(dump)
    env = os.environ.copy()
    env['X264_MV_DUMP'] = dump
    cmd = [X264, '--input-res', '1920x1080', '--fps', '30', '--frames',
           str(args.frames), '--threads', '1', '--preset', 'medium',
           '--crf', str(args.crf_hint), '--me', args.hint_mode,
           '--subme', '9', '-o', os.path.join(BUILD, 'hint_src.264'),
           args.y4m]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='cockL')
    ap.add_argument('--y4m', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--crfs', default='23,27,31')
    ap.add_argument('--modes', default='skip,seed,limit:2,limit:4')
    ap.add_argument('--hint-mode', default='hex', choices=('hex', 'esa'))
    ap.add_argument('--me', default='hex')
    ap.add_argument('--subme', type=int, default=7)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--crf-hint', type=int, default=27)
    ap.add_argument('--extra', default='')
    a = ap.parse_args()
    hint = make_hint(a)
    rows = {}
    for crf in [int(x) for x in a.crfs.split(',')]:
        a.crf = crf
        for rep in range(a.reps):
            order = ['base'] + a.modes.split(',')
            if rep % 2:
                order = order[1:] + order[:1]
            for mode in order:
                env = os.environ.copy()
                extra = []
                if mode != 'base':
                    env['X264_MV_HINT'] = hint
                    env['X264_MV_HINT_MODE'] = mode
                out = os.path.join(BUILD, 'bd_%s_%d_%s.264'
                                   % (a.label, crf, mode.replace(':', '_')))
                old = os.environ.copy()
                os.environ.update(env)
                sec = run(a, out)
                os.environ.clear(); os.environ.update(old)
                dec = os.path.join(BUILD, 'bd_dec.yuv')
                decode(out, dec)
                p, s = scores(dec, a.src, 1920, 1080)
                size = os.path.getsize(out)
                rows.setdefault((mode, crf), []).append((sec, size, p, s))
                print('%s %s crf%d rep%d %.3fs %d PSNR %.4f SSIM %.6f'
                      % (a.label, mode, crf, rep, sec, size, p, s),
                      flush=True)
    md = ['# x264 hint BD-rate: %s (%s hints)' % (a.label, a.hint_mode),
          '', 'Modes vs base, CRF %s, %d reps.' % (a.crfs, a.reps), '',
          '| mode | CRF | time s | bytes | PSNR | SSIM |',
          '|---|---:|---:|---:|---:|---:|']
    curves = {}
    for k, v in sorted(rows.items()):
        t = statistics.median(x[0] for x in v)
        b = statistics.median(x[1] for x in v)
        p = statistics.median(x[2] for x in v)
        s = statistics.median(x[3] for x in v)
        curves.setdefault(k[0], []).append((b, p, s, t))
        md.append('| %s | %d | %.3f | %d | %.4f | %.6f |'
                  % (k[0], k[1], t, b, p, s))
    base = curves['base']
    md += ['', '| mode | BD PSNR | BD SSIM | speed vs base |',
           '|---|---:|---:|---:|']
    for mode, c in curves.items():
        if mode == 'base':
            continue
        bp, _, _ = bdrate([(x[0], x[1]) for x in c],
                          [(x[0], x[1]) for x in base])
        bs, _, _ = bdrate([(x[0], -10*math.log10(max(1e-9, 1-x[2]))) for x in c],
                          [(x[0], -10*math.log10(max(1e-9, 1-x[2]))) for x in base])
        sp = statistics.median(x[3] for x in base) / statistics.median(x[3] for x in c)
        md.append('| %s | %+.3f%% | %+.3f%% | %.2fx |'
                  % (mode, bp*100, bs*100, sp))
    path = os.path.join(ROOT, 'docs', 'X264_BD_%s_%s.md'
                        % (a.label, a.hint_mode))
    open(path, 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print(path)


if __name__ == '__main__':
    main()
