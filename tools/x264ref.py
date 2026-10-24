#!/usr/bin/env python3
"""Same rate/quality measurement for x264, as the reference point.

    ./tools/x264ref.py <w> <h> <frames> <qp> <clip.yuv> [x264 extra args...]

Uses `--bitrate` mode (2-pass off, 1-pass ABR) when a rate is given so the two
encoders can be compared at the same rate rather than at the same QP.
"""
import math
import os
import subprocess
import sys

w, h, n, qp = (int(x) for x in sys.argv[1:5])
clip = sys.argv[5]
extra = sys.argv[6:]
x264 = os.environ.get('X264', '/tmp/x264b/x264')
ffmpeg = os.environ.get(
    'FFMPEG', '/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')

cmd = [x264, '--quiet', '--frames', str(n), '--profile', 'baseline',
       '-o', '/tmp/x.264', '--input-res', '%dx%d' % (w, h),
       '--input-csp', 'i420']
# constant QP unless the caller asked for a rate (--bitrate) or a QP itself
if not any(a in ('--bitrate', '-B', '-q', '--qp') for a in extra):
    cmd += ['-q', str(qp)]
if not any(a.startswith('--keyint') or a == '-I' for a in extra):
    cmd += ['--keyint', '250']
cmd += extra
cmd.append(clip)
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', '/tmp/x.264', '-f',
                'rawvideo', '-pix_fmt', 'yuv420p', '/tmp/xd.yuv'], check=True)
src = open(clip, 'rb').read()
dec = open('/tmp/xd.yuv', 'rb').read()
fs, ya, ua = w * h * 3 // 2, w * h, w * h // 4
out, total = [], 0
for name, a0, a1 in (('Y', 0, ya), ('U', ya, ya + ua), ('V', ya + ua, fs)):
    mse = sum(sum((a - b) ** 2 for a, b in zip(src[f * fs + a0:f * fs + a1],
                                               dec[f * fs + a0:f * fs + a1]))
              for f in range(n)) / (n * (a1 - a0))
    total += mse * (a1 - a0)
    out.append('%s=%.2f' % (name, 10 * math.log10(255 * 255 / mse)))
print('x264 %-14s bytes=%8d %s all=%.2f'
      % (' '.join(extra), os.path.getsize('/tmp/x.264'), ' '.join(out),
         10 * math.log10(255 * 255 / (total / fs))))
