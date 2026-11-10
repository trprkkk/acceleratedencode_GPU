#!/usr/bin/env python3
"""Measure x265 analysis save/load upper bound (CPU ceiling proxy).

Usage:
  tools/ceiling_x265.py --y4m FILE --label NAME --frames N \
      --presets medium,slow --crfs 23,27,31 --reps 3

Each point runs one save and three load modes (refine 0/1/3), each `reps`
times.  Prints a CSV to build/work/ceiling_x265.csv and a median summary.
The load bitstreams should be byte-identical to save; the script checks MD5.
"""
import argparse
import hashlib
import os
import statistics
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build', 'work')
X265 = '/usr/bin/x265'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def mode_args(mode, afile):
    a = ['--analysis-reuse-file', afile, '--analysis-reuse-level', '5']
    if mode == 'save':
        return ['--analysis-reuse-mode', 'save'] + a
    a = ['--analysis-reuse-mode', 'load'] + a
    if mode == 'load0':
        return a + ['--refine-inter', '0', '--refine-intra', '0',
                    '--no-refine-mv']
    if mode == 'load1':
        return a + ['--refine-inter', '1', '--refine-intra', '1',
                    '--no-refine-mv']
    return a + ['--refine-inter', '3', '--refine-intra', '3', '--refine-mv']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--y4m', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--presets', default='medium')
    ap.add_argument('--crfs', default='27')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    presets = a.presets.split(',')
    crfs = [int(x) for x in a.crfs.split(',')]
    rows = []
    out_csv = a.out or os.path.join(BUILD, 'ceiling_%s.csv' % a.label)
    os.makedirs(BUILD, exist_ok=True)
    for preset in presets:
        for crf in crfs:
            md5s = {}
            for rep in range(a.reps):
                for mode in ('save', 'load0', 'load1', 'load3'):
                    afile = os.path.join(BUILD, 'x265_analysis.dat')
                    out = os.path.join(BUILD, 'x265_ceiling.hevc')
                    cmd = [X265, '--input', a.y4m, '--preset', preset,
                           '--crf', str(crf), '--frames', str(a.frames),
                           '--no-info', '-o', out]
                    cmd += mode_args(mode, afile)
                    elapsed = run(cmd)
                    size = os.path.getsize(out)
                    digest = md5(out)
                    md5s.setdefault(mode, set()).add(digest)
                    rows.append((a.label, preset, crf, mode, rep,
                                 round(elapsed, 4), size, digest))
                    print('%s %s crf%d %-5s rep%d %.3fs %d'
                          % (a.label, preset, crf, mode, rep, elapsed, size),
                          flush=True)
            if any(len(v) != 1 for v in md5s.values()) or len(
                    set().union(*md5s.values())) != 1:
                print('WARNING: %s %s crf%d bitstreams differ'
                      % (a.label, preset, crf))
    with open(out_csv, 'w') as f:
        f.write('clip,preset,crf,mode,rep,seconds,bytes,md5\n')
        for r in rows:
            f.write('%s,%s,%d,%s,%d,%.4f,%d,%s\n' % r)
    print('\nMedian seconds:')
    for preset in presets:
        for crf in crfs:
            line = []
            for mode in ('save', 'load0', 'load1', 'load3'):
                vals = [r[5] for r in rows
                        if r[1] == preset and r[2] == crf and r[3] == mode]
                line.append('%s=%.3f' % (mode, statistics.median(vals)))
            print('%s %s crf%d  %s' % (a.label, preset, crf, ' '.join(line)))


if __name__ == '__main__':
    main()
