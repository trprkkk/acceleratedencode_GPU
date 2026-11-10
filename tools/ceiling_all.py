#!/usr/bin/env python3
"""One-shot ceiling runner: x265 analysis reuse + x264/x265 contribution.

Example:
  tools/ceiling_all.py --y4m build/work/1080_120.y4m --label 1080_120 \
      --size 1920x1080 --frames 120 --presets medium,slow --crf 27

Writes build/work/ceiling_summary_<label>.md and runs each existing probe.
The GPU/fork is not touched; this only runs external encoders.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build', 'work')


def call(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    if p.returncode:
        sys.stderr.write(p.stdout)
        raise SystemExit(p.returncode)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--y4m', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--size', default='640x352')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--presets', default='medium,slow')
    ap.add_argument('--crf', type=int, default=27)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--full-contrib', action='store_true')
    ap.add_argument('--skip-reuse', action='store_true')
    ap.add_argument('--skip-contrib', action='store_true')
    a = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    md = ['# Ceiling summary: %s' % a.label, '',
          'GPU is not used. External x264/x265 settings only.', '']
    if not a.skip_reuse:
        out = call([sys.executable, os.path.join(ROOT, 'tools', 'ceiling_x265.py'),
                    '--y4m', a.y4m, '--label', a.label,
                    '--frames', str(a.frames), '--presets', a.presets,
                    '--crfs', str(a.crf), '--reps', str(a.reps)])
        open(os.path.join(BUILD, 'ceiling_%s.log' % a.label), 'w').write(out)
        md += ['## x265 analysis reuse', '',
               '| preset | save s | load0 s | ratio |', '|---|---:|---:|---:|']
        for k, (name, save, load) in [
                (m.group(1), (m.group(2), m.group(3), m.group(4)))
                for m in re.finditer(
                    r'^(\S+) (\S+) crf\d+\s+save=([0-9.]+) load0=([0-9.]+)',
                    out, re.M)]:
            md.append('| %s | %s | %s | %.2fx |'
                      % (name, save, load, float(save) / float(load)))
        md.append('')
    contrib = {}
    if not a.skip_contrib:
        for enc in ('x265', 'x264'):
            cmd = [sys.executable, os.path.join(ROOT, 'tools', 'contrib_probe.py'),
                   '--encoder', enc, '--y4m', a.y4m, '--label', a.label,
                   '--size', a.size, '--frames', str(a.frames),
                   '--presets', a.presets, '--crf', str(a.crf),
                   '--reps', str(a.reps)]
            if not a.full_contrib:
                cmd += ['--variants', 'base,all_off']
            out = call(cmd)
            open(os.path.join(BUILD, 'contrib_%s_%s.log'
                              % (enc, a.label)), 'w').write(out)
            md += ['## %s contribution' % enc, '',
                   '| preset | base s | all_off s | ceiling |',
                   '|---|---:|---:|---:|']
            base = {}
            for m in re.finditer(
                    r'^\S+ (\S+) base\s+([0-9.]+)\s+\(([0-9.]+)x\)', out, re.M):
                base[m.group(1)] = float(m.group(2))
            for m in re.finditer(
                    r'^\S+ (\S+) all_off\s+([0-9.]+)\s+\(([0-9.]+)x\)',
                    out, re.M):
                p = m.group(1)
                md.append('| %s | %.3f | %.3f | **%.2fx** |'
                          % (p, base.get(p, float('nan')),
                             float(m.group(2)), float(m.group(3))))
            md.append('')
    md += ['## Interpretation', '',
           '- `all_off` disables ME/refinement/RDO/lookahead; it is an upper bound,',
           '  not a quality-equivalent encoding.',
           '- x265 medium ~1.1-1.3x, slow ~1.4-2.2x depending on frame count.',
           '- x264 medium ~1.1-1.3x, slow ~1.5-1.6x on long 1080p.',
           '- x265 1.5-2x is only plausible for long 1080p + slow/slower;',
           '  medium cannot reach it by search removal alone.']
    path = os.path.join(BUILD, 'ceiling_summary_%s.md' % a.label)
    open(path, 'w').write('\n'.join(md) + '\n')
    print(path)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
