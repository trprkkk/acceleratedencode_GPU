#!/usr/bin/env python3
"""Measure x264/x265 search/RDO/lookahead contribution by settings only.

Usage:
  tools/contrib_probe.py --encoder x265 --y4m FILE --label cockL \
      --frames 240 --presets medium,slow --crf 27 --reps 3
  tools/contrib_probe.py --encoder x264 --y4m FILE --label cockL \
      --size 640x352 --frames 240 --presets medium,slow --crf 27 --reps 3
"""
import argparse
import os
import statistics
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build', 'work')

V_X265 = [
    ('base', []),
    ('me_dia', ['--me', 'dia']),
    ('me_full', ['--me', 'full']),
    ('subme0', ['--subme', '0']),
    ('rd1', ['--rd', '1']),
    ('no_cutree', ['--no-cutree']),
    ('no_lookahead', ['--no-cutree', '--rc-lookahead', '5']),
    ('min_search', ['--me', 'dia', '--subme', '0']),
    ('all_off', ['--me', 'dia', '--subme', '0', '--rd', '1',
                 '--no-cutree', '--rc-lookahead', '5']),
]
V_X264 = [
    ('base', []),
    ('me_dia', ['--me', 'dia']),
    ('me_esa', ['--me', 'esa']),
    ('subme0', ['--subme', '0']),
    ('trellis0', ['--trellis', '0']),
    ('no_mbtree', ['--no-mbtree']),
    ('no_lookahead', ['--no-mbtree', '--rc-lookahead', '0']),
    ('min_search', ['--me', 'dia', '--subme', '0']),
    ('all_off', ['--me', 'dia', '--subme', '0', '--trellis', '0',
                 '--no-mbtree', '--rc-lookahead', '0']),
]


def run(cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', choices=('x264', 'x265'), required=True)
    ap.add_argument('--y4m', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--size', default='640x352')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--presets', default='medium')
    ap.add_argument('--crf', type=int, default=27)
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    variants = V_X265 if a.encoder == 'x265' else V_X264
    rows = []
    for preset in a.presets.split(','):
        for name, extra in variants:
            for rep in range(a.reps):
                out = os.path.join(BUILD, 'contrib_%s.hevc' % a.encoder)
                if a.encoder == 'x265':
                    cmd = ['x265', '--input', a.y4m, '--preset', preset,
                           '--crf', str(a.crf), '--frames', str(a.frames),
                           '--no-info', '-o', out] + extra
                else:
                    cmd = ['x264', '--input-res', a.size, '--fps', '30',
                           '--frames', str(a.frames), '--preset', preset,
                           '--crf', str(a.crf), '-o', out] + extra + [a.y4m]
                sec = run(cmd)
                size = os.path.getsize(out)
                rows.append((a.label, preset, name, rep, round(sec, 4), size))
                print('%s %s %-12s rep%d %.3f' % (a.label, preset, name,
                                                    rep, sec), flush=True)
    csv = os.path.join(BUILD, 'contrib_%s_%s.csv' % (a.encoder, a.label))
    with open(csv, 'w') as f:
        f.write('clip,preset,variant,rep,seconds,bytes\n')
        for r in rows:
            f.write('%s,%s,%s,%d,%.4f,%d\n' % r)
    print('\nMedian seconds:')
    for preset in a.presets.split(','):
        base = statistics.median([r[4] for r in rows
                                  if r[1] == preset and r[2] == 'base'])
        for name, _ in variants:
            vals = [r[4] for r in rows
                    if r[1] == preset and r[2] == name]
            med = statistics.median(vals)
            print('%s %s %-12s %.3f  (%.3fx)'
                  % (a.label, preset, name, med, base / med))


if __name__ == '__main__':
    main()
