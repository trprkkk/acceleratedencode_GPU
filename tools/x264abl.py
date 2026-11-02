#!/usr/bin/env python3
"""Ablations against x264: which x264 tool accounts for a gap on which clips.

    tools/x264abl.py run    CONFIG... [--clips SET]
    tools/x264abl.py report BASE CONFIG... [--clips SET] [--metric psnr|ssim_db]

A CONFIG is NAME=SPEC, where SPEC is
    aeg:P[:extra aegenc args, separated by spaces or commas]
    x264:PRESET[:x264-params]          (always with --tune psnr)
e.g.  x264vs=x264:veryslow  vs-ref3=x264:veryslow:ref=3  p4=aeg:4
      p4b2=aeg:4:--bframes,2   p4x=aeg:4:AEG_I16N=3  (NAME=VALUE: an environment variable)
The same measurement as tools/x264cmp.py (constant QP 22-37, global PSNR of
the decoded stream, FFmpeg SSIM, single-threaded; BD-rate of every config
against BASE, negative = fewer bits than BASE), cached in $X264ABL (default
/tmp/x264abl.json) keyed on the spec and, for aegenc, the binary's hash.

Clip sets: the x264cmp sets (main, val, long) and 'ct' = the clips where
aegenc trailed x264 veryslow (cock, cock2, tree, tree2) plus 'vt' = where it
led (vtest, vtest2, vtest3); a comma list mixes them.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x264cmp as X  # noqa: E402

X.CACHE = os.environ.get('X264ABL', '/tmp/x264abl.json')
SETS = dict(X.CLIPS)
_by = {c[0]: c for s in X.CLIPS.values() for c in s}
SETS['ct'] = [_by[t] for t in ('cock', 'cock2', 'tree', 'tree2')]
SETS['vt'] = [_by[t] for t in ('vtest', 'vtest2', 'vtest3')]


MODE = 'qp'


def points():
    """QP 22-37, or with --mode crf CRF 18-33 (both encoders then run their
    macroblock tree; aegenc with --aq 0, x264 with --tune psnr)."""
    return X.CRFS if MODE == 'crf' else X.QPS


def clips(names):
    out = []
    for s in names.split(','):
        out += SETS[s] if s in SETS else [_by[s]]
    return out


def parse(spec):
    name, s = spec.split('=', 1)
    parts = s.split(':', 2)
    enc = parts[0]
    if enc == 'aeg':
        extra = parts[2].replace(',', ' ').split() if len(parts) > 2 and parts[2] else []
        return name, ('aeg', int(parts[1]), extra), s
    extra = ['-tune', 'psnr']
    if len(parts) > 2 and parts[2]:
        extra += ['-x264-params', parts[2]]
    return name, ('x264', parts[1], extra), s


def key(spec, cfg, clip, v):
    st = os.stat(clip)
    k = '%s|%s|%d|%d|%s|%s' % (spec, os.path.abspath(clip), st.st_size, int(st.st_mtime),
                               MODE, v)
    if cfg[0] == 'aeg':
        k += '|' + X.AEG_ID
    return k


def measure_env(cfg, clip, w, h, n, v):
    """An aegenc token NAME=VALUE (no leading '-') is an environment
    variable for that encode rather than an argument (the encoder's AEG_*
    measurement knobs)."""
    env = {}
    if cfg[0] == 'aeg':
        args = []
        for t in cfg[2]:
            if not t.startswith('-') and '=' in t:
                kk, vv = t.split('=', 1)
                env[kk] = vv
            else:
                args.append(t)
        cfg = (cfg[0], cfg[1], args)
    old = {kk: os.environ.get(kk) for kk in env}
    os.environ.update(env)
    try:
        return X.measure(cfg, clip, w, h, n, MODE, v)
    finally:
        for kk, vv in old.items():
            if vv is None:
                del os.environ[kk]
            else:
                os.environ[kk] = vv


def cmd_run(a):
    cache = X.cache_load()
    for spec in a.configs:
        name, cfg, s = parse(spec)
        for tag, clip, w, h in clips(a.clips):
            n = X.nframes(clip, w, h, False)
            for v in points():
                k = key(s, cfg, clip, v)
                if k in cache:
                    continue
                cache[k] = measure_env(cfg, clip, w, h, n, v)
                X.cache_save(cache)
                r = cache[k]
                print('%-12s %-7s %2d %9.1f kb/s %6.2f dB %6.2fs' %
                      (name, tag, v, r['kbps'], r['psnr'], r['cpu']), flush=True)


def cmd_report(a):
    cache = X.cache_load()
    cl = clips(a.clips)
    tags = [c[0] for c in cl]

    def curve(spec):
        name, cfg, s = parse(spec)
        out = {}
        for tag, clip, w, h in cl:
            out[tag] = [(cache[key(s, cfg, clip, v)]['kbps'], cache[key(s, cfg, clip, v)][a.metric])
                        for v in points()]
        return name, out

    bname, base = curve(a.configs[0])
    print('BD-rate (%s) against %s; negative = fewer bits\n' % (a.metric, bname))
    print('| config | ' + ' | '.join(tags) + ' | mean | cpu s |')
    print('|---|' + '---|' * (len(tags) + 2))
    for spec in a.configs[1:]:
        try:
            name, c = curve(spec)
        except KeyError:
            print('| %s | not measured (run it first) |' % spec.split('=', 1)[0])
            continue
        _, cfg, s = parse(spec)
        r = [X.bd(c[t], base[t]) for t in tags]
        cpu = sum(cache[key(s, cfg, clip, v)]['cpu'] for _, clip, _, _ in cl for v in points())
        print('| %s | %s | %+.2f%% | %.0f |' % (
            name, ' | '.join('%+.2f%s' % (x[0], '*' if x[1] < 0.5 else '') for x in r),
            statistics.mean(x[0] for x in r), cpu), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--clips', default='ct,vt')
    ap.add_argument('--mode', default='qp', choices=['qp', 'crf'])
    ap.add_argument('--metric', default='psnr', choices=['psnr', 'psnr_y', 'ssim_db'])
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    X.AEG_ID = os.environ.get('AEGENC_ID') or X.sha(X.AEG)
    {'run': cmd_run, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
