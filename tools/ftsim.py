#!/usr/bin/env python3
"""Replay a serial encode's timeline on N simulated cores (frame threads).

    AEG_FTTRACE=t.trace aegenc ... -t 1 --frame-threads 1 -o x.264
    tools/ftsim.py t.trace [--cores 1,2,4,8] [--workers N] [--ghz 2.6]

The trace (src/encoder.c, ftt_on) lists every picture body in coding order
(B type / E), the reference rows each block row waits for (W slot rows),
the rows it publishes (P slot rows) and the lookahead searches done on the
main thread (L cycles), with rdtsc times.  The replay follows the frame
threads' rules:

  * the main thread does each picture's prologue (the gap between two
    bodies in the trace, without the lookahead's searches) and then hands
    the picture to a free worker (auto: one more than the cores, at most
    FT_MAXW), no more than FT_Q - 1 pictures past the oldest one still
    coding (the packet queue);
  * a picture's block rows run in order; before a row it waits until each
    reference has published the rows it needs; its publishes release the
    pictures that wait for it;
  * a P picture whose serial run waited for the whole reference at its
    start (weighted prediction) starts speculatively instead and waits for
    the whole reference only before it publishes its last rows (ft_spec,
    ft_wp_check);
  * the lookahead's searches run on helper threads as background load, and
    whatever runs (workers, the main thread, the lookahead) shares the
    cores equally.

It prints the predicted wall time and speed-up for each core count, and
where the time goes: the share of picture time spent waiting for reference
rows, the main thread's share, and how full the cores were.
"""
import argparse
import sys


def parse(path):
    pics, la_in_gap, gaps = [], [], []
    cur = None
    last_t = None
    last_e = None
    la_acc = 0
    owner = {}
    for line in open(path):
        f = line.split()
        if not f:
            continue
        k = f[0]
        if k == 'L':
            la_acc += int(f[1])
            continue
        if k == 'B':
            t = int(f[2])
            gaps.append(0 if last_e is None else t - last_e)
            la_in_gap.append(la_acc)
            la_acc = 0
            cur = {'type': int(f[1]), 'ev': [], 'spec': False, 'idx': len(pics)}
            last_t = t
            continue
        if k == 'E':
            t = int(f[1])
            cur['ev'].append(('c', t - last_t))
            pics.append(cur)
            last_e = t
            cur = None
            continue
        t = int(f[3])
        s, rows = int(f[1]), int(f[2])
        cur['ev'].append(('c', t - last_t))
        last_t = t
        if k == 'P':
            owner[s] = cur['idx']
            cur['ev'].append(('p', rows))
        elif k == 'W':
            o = owner.get(s)
            if o is None or o == cur['idx']:
                continue
            cur['ev'].append(('w', o, rows))
    # weighted prediction: a whole-reference wait before the first row
    # becomes a wait before the last publish
    for p in pics:
        ev = p['ev']
        first_p = next((i for i, e in enumerate(ev) if e[0] == 'p'), len(ev))
        mh = max((e[1] for e in ev if e[0] == 'p'), default=0)
        full = [i for i, e in enumerate(ev[:first_p])
                if e[0] == 'w' and e[2] >= mh and p['type'] == 1]
        if not full:
            continue
        w = ev[full[0]]
        del ev[full[0]]
        last_p = max(i for i, e in enumerate(ev) if e[0] == 'p')
        ev.insert(last_p, ('w', w[1], w[2]))
        p['spec'] = True
    main = [max(0, g - l) for g, l in zip(gaps, la_in_gap)]
    la_total = sum(la_in_gap) + la_acc
    return pics, main, la_total


FT_Q = 16                       # the packet queue (encoder.c)
FT_MAXW = 8                     # the most workers (encoder.c)


def simulate(pics, main, la_total, cores, workers):
    n = len(pics)
    pub = [0] * n               # rows published
    done = [False] * n
    pos = [0] * n               # event index
    rem = [0.0] * n             # remaining cycles of the current segment
    started = [False] * n
    start_t = [0.0] * n
    end_t = [0.0] * n
    wait_cyc = [0.0] * n
    main_i = 0                  # picture whose prologue the main thread runs
    main_rem = float(main[0]) if n else 0.0
    main_busy_cyc = 0.0
    la_rem = float(la_total)
    t = 0.0
    busy_area = 0.0

    def load(i):
        while pos[i] < len(pics[i]['ev']):
            e = pics[i]['ev'][pos[i]]
            if e[0] == 'c':
                if rem[i] <= 0:
                    rem[i] = float(e[1])
                if rem[i] > 0:
                    return 'run'
                pos[i] += 1
                rem[i] = 0
                continue
            if e[0] == 'p':
                pub[i] = max(pub[i], e[1])
                pos[i] += 1
                continue
            if e[0] == 'w':
                o, rows = e[1], e[2]
                if done[o] or pub[o] >= rows:
                    pos[i] += 1
                    continue
                return 'wait'
        done[i] = True
        return 'done'

    active = []                 # started, not done
    state = {}

    def oldest():
        return min(active) if active else main_i
    while True:
        # dispatch: the main thread finished picture main_i's prologue
        while (main_i < n and main_rem <= 0 and len(active) < workers and
               main_i - oldest() < FT_Q - 1):
            i = main_i
            started[i] = True
            start_t[i] = t
            active.append(i)
            main_i += 1
            main_rem = float(main[main_i]) if main_i < n else 0.0
        # settle states (publishes may release waiters)
        changed = True
        while changed:
            changed = False
            for i in list(active):
                st = load(i)
                if st == 'done':
                    end_t[i] = t
                    active.remove(i)
                    state.pop(i, None)
                    changed = True
                elif state.get(i) != st:
                    state[i] = st
                    changed = True
        if main_i >= n and not active:
            break
        run = [i for i in active if state[i] == 'run']
        main_run = main_i < n and main_rem > 0
        main_blocked = main_i < n and main_rem <= 0   # waiting for a worker
        la_run = la_rem > 0
        nrun = len(run) + (1 if main_run else 0) + (1 if la_run else 0)
        if nrun == 0:
            if main_blocked and len(active) < workers and \
                    main_i - oldest() < FT_Q - 1:
                continue        # a worker was freed: dispatch
            if main_blocked and not active:
                raise RuntimeError('deadlock')
            # everything waits: impossible unless a dependency is broken
            raise RuntimeError('stuck at t=%g' % t)
        speed = min(1.0, cores / nrun)
        # the lookahead may use several helpers: it takes the share of up to
        # min(4, cores) threads when cores are idle
        la_threads = 1
        if la_run and len(run) + (1 if main_run else 0) < cores:
            la_threads = min(4, cores - len(run) - (1 if main_run else 0))
        dt = float('inf')
        for i in run:
            dt = min(dt, rem[i] / speed)
        if main_run:
            dt = min(dt, main_rem / speed)
        if la_run:
            dt = min(dt, la_rem / (speed * la_threads))
        for i in active:
            if state[i] == 'wait':
                wait_cyc[i] += dt
        for i in run:
            rem[i] -= dt * speed
            if rem[i] <= 1e-6:
                rem[i] = 0
                pos[i] += 1
        if main_run:
            main_rem -= dt * speed
            main_busy_cyc += dt * speed
        if la_run:
            la_rem -= dt * speed * la_threads
        busy_area += dt * min(cores, len(run) + (1 if main_run else 0) +
                              (la_threads if la_run else 0))
        t += dt
    body = sum(end_t[i] - start_t[i] for i in range(n))
    return {
        'wall': t,
        'wait_share': sum(wait_cyc) / body if body else 0,
        'main_share': main_busy_cyc / t if t else 0,
        'util': busy_area / (t * cores) if t else 0,
    }


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--cores', default='1,2,4,8')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--ghz', type=float, default=2.6)
    a = ap.parse_args()
    pics, main, la = parse(a.trace)
    work = sum(e[1] for p in pics for e in p['ev'] if e[0] == 'c')
    serial = work + sum(main) + la
    hz = a.ghz * 1e9
    nspec = sum(1 for p in pics if p['spec'])
    print('%d pictures (%d P with speculative weights): bodies %.2f s, '
          'main thread %.2f s, lookahead %.2f s, serial %.2f s' %
          (len(pics), nspec, work / hz, sum(main) / hz, la / hz, serial / hz))
    print('%5s %7s %9s %8s %10s %10s %6s' %
          ('cores', 'workers', 'wall s', 'speedup', 'row waits', 'main busy',
           'util'))
    for c in [int(x) for x in a.cores.split(',')]:
        w = a.workers or min(c + 1, FT_MAXW)
        r = simulate(pics, main, la, c, w)
        print('%5d %7d %9.2f %7.2fx %9.1f%% %9.1f%% %5.0f%%' %
              (c, w, r['wall'] / hz, serial / r['wall'], 100 * r['wait_share'],
               100 * r['main_share'], 100 * r['util']))


if __name__ == '__main__':
    sys.exit(main_())
