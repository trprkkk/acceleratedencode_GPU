#!/usr/bin/env python3
"""GO-gate measurement: base vs all_off, interleaved and repeated.

Runs 1080p long clips with the real x264/x265 binaries.  The primary KPI is
the within-run ratio base/all_off; absolute times are environment-dependent.

Example:
  tools/go_gate.py --encoder x264 --crfs 23,27,31 --reps 3
  tools/go_gate.py --encoder x265 --crfs 23,27,31 --reps 3

Outputs build/work/go_gate_<encoder>.csv and docs/GO_GATE_<encoder>.md.
"""
import argparse
import os
import statistics
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build', 'work')
CLIPS = [
    ('cockL', 'cockL_1080_120.y4m'),
    ('vtestL', 'vtestL_1080_120.y4m'),
    ('megaL', 'megaL_1080_120.y4m'),
    ('treeL', 'treeL_1080_120.y4m'),
]
ALL_OFF = {
    'x264': ['--me', 'dia', '--subme', '0', '--trellis', '0',
             '--no-mbtree', '--rc-lookahead', '0'],
    'x265': ['--me', 'dia', '--subme', '0', '--rd', '1',
             '--no-cutree', '--rc-lookahead', '5'],
}


def run(cmd, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            return time.perf_counter() - t0
        except subprocess.CalledProcessError as e:
            last = e
            time.sleep(1)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', choices=('x264', 'x265'), required=True)
    ap.add_argument('--crfs', default='23,27,31')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--presets', default='medium,slow')
    ap.add_argument('--frames', type=int, default=120)
    a = ap.parse_args()
    crfs = [int(x) for x in a.crfs.split(',')]
    rows = []
    for clip, fn in CLIPS:
        y4m = os.path.join(BUILD, fn)
        if not os.path.exists(y4m):
            raise SystemExit('missing %s' % y4m)
        for preset in a.presets.split(','):
            for crf in crfs:
                for rep in range(a.reps):
                    order = ('base', 'all_off') if rep % 2 == 0 else (
                        'all_off', 'base')
                    for mode in order:
                        out = os.path.join(BUILD, 'go_gate.%s' % a.encoder)
                        if a.encoder == 'x264':
                            cmd = ['x264', '--input-res', '1920x1080',
                                   '--fps', '30', '--frames', str(a.frames),
                                   '--preset', preset, '--crf', str(crf),
                                   '-o', out]
                            if mode == 'all_off':
                                cmd += ALL_OFF['x264']
                        else:
                            cmd = ['x265', '--input', y4m, '--preset', preset,
                                   '--crf', str(crf), '--frames',
                                   str(a.frames), '--no-info', '-o', out]
                            if mode == 'all_off':
                                cmd += ALL_OFF['x265']
                        if a.encoder == 'x264':
                            cmd += [y4m]
                        sec = run(cmd)
                        size = os.path.getsize(out)
                        rows.append((clip, preset, crf, rep, mode,
                                     round(sec, 4), size))
                        print('%s %s crf%d %-7s rep%d %.3fs'
                              % (clip, preset, crf, mode, rep, sec),
                              flush=True)
    csv = os.path.join(BUILD, 'go_gate_%s.csv' % a.encoder)
    with open(csv, 'w') as f:
        f.write('clip,preset,crf,rep,mode,seconds,bytes\n')
        for r in rows:
            f.write('%s,%s,%d,%d,%s,%.4f,%d\n' % r)
    md = ['# GO gate: %s' % a.encoder, '',
          '1080p 120f, base vs all_off, 3 reps, interleaved order.',
          'Primary KPI is the paired ratio; environment load is high.', '',
          '| clip | preset | CRF | base s | all_off s | ratio |',
          '|---|---|---:|---:|---:|---:|']
    by = {}
    for clip, _ in CLIPS:
        for preset in a.presets.split(','):
            for crf in crfs:
                bs = [r[5] for r in rows
                      if r[0] == clip and r[1] == preset and r[2] == crf
                      and r[4] == 'base']
                os_ = [r[5] for r in rows
                       if r[0] == clip and r[1] == preset and r[2] == crf
                       and r[4] == 'all_off']
                b, o = statistics.median(bs), statistics.median(os_)
                ratio = b / o
                by.setdefault((preset, crf), []).append(ratio)
                md.append('| %s | %s | %d | %.3f | %.3f | **%.2fx** |'
                          % (clip, preset, crf, b, o, ratio))
    md += ['', '## Gate summary', '',
           '| preset | CRF | mean | worst |', '|---|---:|---:|---:|']
    for (preset, crf), v in sorted(by.items()):
        md.append('| %s | %d | %.2fx | %.2fx |'
                  % (preset, crf, statistics.mean(v), min(v)))
    gate = {'medium': 1.2, 'slow': 1.4} if a.encoder == 'x264' else {
        'medium': 1.15, 'slow': 1.5}
    md += ['', 'Gate: ' + ', '.join('%s >= %.2fx' % (k, v)
                                    for k, v in gate.items()), '']
    for (preset, crf), v in sorted(by.items()):
        ok = statistics.mean(v) >= gate.get(preset, 0) and min(v) >= gate.get(
            preset, 0)
        md.append('- %s CRF%d: **%s** (mean %.2fx, worst %.2fx)'
                  % (preset, crf, 'PASS' if ok else 'FAIL',
                     statistics.mean(v), min(v)))
    md += ['', 'Absolute times are wall-clock, high system load; use ratios.']
    path = os.path.join(ROOT, 'docs', 'GO_GATE_%s.md' % a.encoder)
    open(path, 'w').write('\n'.join(md) + '\n')
    print(path)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
