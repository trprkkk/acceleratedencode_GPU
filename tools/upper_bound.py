#!/usr/bin/env python3
"""Where does the encode time go, and what is the ceiling for a GPU?

Prints the encoder's own stage accounting for a run, and the ceiling: what
the wall time would be with the motion search free (an infinitely fast GPU),
which is the largest honest answer to "what can offloading buy".
"""
import subprocess, sys, os, re

R = '/mnt/q/Downloads/HybridEncode/repo/'
B = R + 'build-work/'
W, H, N = 1280, 720, 30
SRC = '/tmp/aegbench/src_1280x720_30.yuv'
PRESET = sys.argv[1] if len(sys.argv) > 1 else '4'
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = B + ':/usr/local/cuda-12.1/targets/x86_64-linux/lib'


def run(extra):
    out = subprocess.run(
        [B + 'aegenc', '-w', str(W), '-h', str(H), '-n', str(N),
         '-i', SRC, '-o', '/tmp/ub.264', '-p', PRESET, '-q', '27'] + extra,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return out.stdout.decode('utf-8', 'replace')


def ms(log, key):
    m = re.search(re.escape(key) + r'\s+([0-9.]+) ms', log)
    return float(m.group(1)) if m else 0.0


base = run([])
print('preset %s, QP 27, %dx%d, %d frames' % (PRESET, W, H, N))
print('  fps: %s' % re.search(r'([0-9.]+) fps', base).group(1))
est = ms(base, 'stage motion-est')
inter = ms(base, 'stage inter-pred')
tr = ms(base, 'stage transform')
iq = ms(base, 'stage iquant-idct')
dbk = ms(base, 'stage deblock')
psnr = ms(base, 'stage psnr')
print('  motion-est     %8.1f ms' % est)
print('  inter-pred     %8.1f ms' % inter)
if tr:
    print('  transform      %8.1f ms' % tr)
if iq:
    print('  iquant-idct    %8.1f ms' % iq)
print('  deblock        %8.1f ms' % dbk)
print('  psnr           %8.1f ms' % psnr)
known = est + inter + tr + iq + dbk + psnr
print('  (stages total  %8.1f ms)' % known)
print()
for label, extra in (
        ('no subpel (--subme 0)', ['--subme', '0']),
        ('one reference (--ref 1)', ['--ref', '1']),
        ('16x16 only (--partitions 1)', ['--partitions', '1']),
        ('range 16 (--range 16)', ['--range', '16']),
):
    log = run(extra)
    print('  %-30s fps %-6s motion-est %8.1f ms' %
          (label, re.search(r'([0-9.]+) fps', log).group(1),
           ms(log, 'stage motion-est')))
print()
print('=== the ceiling: what a free motion search would buy (Amdahl) ===')
fps = float(re.search(r'([0-9.]+) fps', base).group(1))
wall = 1000.0 * N / fps
free = wall - (est + inter)
print('  wall            %8.1f ms' % wall)
print('  minus ME+inter  %8.1f ms   ->  %.1f fps ceiling'
      % (free, 1000.0 * N / free if free > 0 else 0))
half = wall - 0.5 * (est + inter)
print('  minus half of ME %7.1f ms  ->  %.1f fps' % (half, 1000.0 * N / half))
print('  x264 medium, the same clip:     4.29 fps')
