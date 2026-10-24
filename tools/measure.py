#!/usr/bin/env python3
"""Rate/quality of one encode: bytes, fps and plane PSNR against the source.

    ./tools/measure.py <w> <h> <frames> <qp> <partitions> <clip.yuv> [extra aegenc args...]

Pass `--bitrate N` as an extra argument to measure the ABR controller instead of
a constant QP.  Set FFMPEG=... to choose the decoder used for the comparison.
"""
import math
import os
import subprocess
import sys
import time

args = sys.argv[1:]
w, h, n, qp, p = (int(x) for x in args[:5])
clip = args[5]
extra = args[6:]
ffmpeg = os.environ.get(
    'FFMPEG', '/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')

t0 = time.time()
subprocess.run(['./build/aegenc', '-w', str(w), '-h', str(h), '-q', str(qp),
                '-n', str(n), '-i', clip, '-o', '/tmp/mm.264',
                '--partitions', str(p), '-v', '0'] + extra,
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
elapsed = time.time() - t0
subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', '/tmp/mm.264', '-f',
                'rawvideo', '-pix_fmt', 'yuv420p', '/tmp/mmd.yuv'], check=True)

src = open(clip, 'rb').read()
dec = open('/tmp/mmd.yuv', 'rb').read()
fs, ya, ua = w * h * 3 // 2, w * h, w * h // 4
out, total = [], 0
for name, a0, a1 in (('Y', 0, ya), ('U', ya, ya + ua), ('V', ya + ua, fs)):
    mse = sum(sum((a - b) ** 2 for a, b in zip(src[f * fs + a0:f * fs + a1],
                                               dec[f * fs + a0:f * fs + a1]))
              for f in range(n)) / (n * (a1 - a0))
    total += mse * (a1 - a0)
    out.append('%s=%.2f' % (name, 10 * math.log10(255 * 255 / mse)))
print('q%d p%2d %s bytes=%8d fps=%6.1f %s all=%.2f'
      % (qp, p, ' '.join(extra).ljust(16), os.path.getsize('/tmp/mm.264'),
         n / elapsed, ' '.join(out), 10 * math.log10(255 * 255 / (total / fs))))
