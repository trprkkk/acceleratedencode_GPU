#!/usr/bin/env python3
"""Our bit excess over x264 at equal PSNR, at one QP.

usage: tools/real/gap.py clip w h "<x264 params>" "<ours>" [qp]

x264 (medium, psy=0, CQP, scene cuts off) is encoded at qp-2 and qp; its
slope (dB per log-bit) between the two converts our PSNR difference at `qp`
into bits.  `clip` is a name in $REAL (default /tmp/real) or a path; the first
30 frames are used.  "<x264 params>" is appended to the x264 parameter string
(start it with ':'), "<ours>" is an optional run of AEG_*=value assignments
followed by aegenc options (BIN=path selects another binary).  PL=0/1/2/3
selects the PSNR plane: Y (default), U, V, or all samples.  Needs numpy
(PYTHONPATH=/tmp/dl/ext) and FFMPEG.
"""
import math
import os
import subprocess
import sys

import numpy as np

REAL = os.environ.get('REAL', '/tmp/real')
FF = os.environ.get('FFMPEG', '/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

clip, w, h, xp, op = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
q = int(sys.argv[6]) if len(sys.argv) > 6 else 30
path = clip if os.path.exists(clip) else os.path.join(REAL, clip + '.yuv')
fs = w * h * 3 // 2
src = np.fromfile(path, np.uint8)
n = min(30, len(src) // fs)
pl = int(os.environ.get('PL', '0'))
sl = [slice(0, w * h), slice(w * h, w * h * 5 // 4), slice(w * h * 5 // 4, fs), slice(0, fs)][pl]
src = src[:n * fs].reshape(-1, fs)[:, sl].astype(np.float64)
tmp = '/tmp/gap_%d' % os.getpid()


def measure(f):
    r = subprocess.run([FF, '-v', 'error', '-i', f, '-f', 'rawvideo', '-'],
                       capture_output=True, check=True).stdout
    rec = np.frombuffer(r, np.uint8).reshape(-1, fs)[:, sl].astype(np.float64)
    return os.path.getsize(f), 10 * math.log10(255 * 255 / ((src - rec) ** 2).mean())


def x264(qq):
    subprocess.run([FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                    '-s', '%dx%d' % (w, h), '-i', path, '-frames', str(n),
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-x264-params', 'qp=%d:psy=0:scenecut=0%s' % (qq, xp),
                    '-f', 'h264', tmp + 'x.264'], check=True)
    return measure(tmp + 'x.264')


x1, p1 = x264(q - 2)
x2, p2 = x264(q)
env = dict(os.environ)
binary = os.path.join(ROOT, 'build', 'aegenc')
toks = op.split()
while toks and '=' in toks[0] and not toks[0].startswith('-'):
    k, v = toks.pop(0).split('=', 1)
    if k == 'BIN':
        binary = v
    else:
        env[k] = v
subprocess.run([binary, '-w', str(w), '-h', str(h), '-i', path, '-o', tmp + 'o.264',
                '-n', str(n), '-p', '3', '-q', str(q), '--scenecut', '0'] + toks,
               check=True, capture_output=True, env=env)
o, po = measure(tmp + 'o.264')
for f in (tmp + 'x.264', tmp + 'o.264'):
    os.remove(f)
k = (p1 - p2) / math.log(x1 / x2)
eq = x2 * math.exp((po - p2) / k)
print('gap %+.2f%%  (x264 %d B %.3f dB, ours %d B %.3f dB)' % (100 * (o / eq - 1), x2, p2, o, po))
