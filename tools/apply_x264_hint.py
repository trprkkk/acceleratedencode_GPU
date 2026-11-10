#!/usr/bin/env python3
"""Minimal x264 MV dump/hint patch applicator.

This does not vendor x264.  Run it inside a checkout of x264 0.164:

    python3 tools/apply_x264_hint.py --src forks/x264

It inserts two calls into encoder/me.c and adds encoder/aeg_mv_hint.{c,h}.
The patch is env-gated: unset X264_MV_HINT / X264_MV_DUMP means stock x264.
"""
import argparse
import os
import re
import shutil

HEADER = r'''#ifndef AEG_MV_HINT_H
#define AEG_MV_HINT_H

#include "common/common.h"
#include "me.h"

void aeg_mv_hint_pre(x264_me_t *m);
void aeg_mv_hint_post(const x264_me_t *m);

#endif
'''

SOURCE = r'''/* GPL-2.0-or-later, part of the x264 fork.  Minimal external MV seed/dump. */
#include "common/common.h"
#include "aeg_mv_hint.h"
#include <stdio.h>
#include <stdlib.h>

static FILE *in_file, *out_file;

void aeg_mv_hint_pre(x264_me_t *m)
{
    const char *ev = getenv("X264_MV_HINT");
    int x, y;
    if (!ev)
        return;
    if (!in_file)
        in_file = fopen(ev, "r");
    if (!in_file || fscanf(in_file, "%d %d", &x, &y) != 2)
        return;
    m->mv[0] = (int16_t)x;
    m->mv[1] = (int16_t)y;
}

void aeg_mv_hint_post(const x264_me_t *m)
{
    const char *ev = getenv("X264_MV_DUMP");
    if (!ev)
        return;
    if (!out_file)
        out_file = fopen(ev, "a");
    if (!out_file)
        return;
    fprintf(out_file, "%d %d %d %d\n", m->mv[0], m->mv[1], m->cost,
            m->i_pixel);
    fflush(out_file);
}
'''


def find_matching(src, open_pos):
    depth = 0
    in_str = in_chr = in_line = in_block = False
    i = open_pos
    while i < len(src):
        c = src[i]
        n = src[i + 1] if i + 1 < len(src) else ''
        if in_line:
            if c == '\n':
                in_line = False
        elif in_block:
            if c == '*' and n == '/':
                in_block = False
                i += 1
        elif in_str:
            if c == '\\':
                i += 1
            elif c == '"':
                in_str = False
        elif in_chr:
            if c == '\\':
                i += 1
            elif c == "'":
                in_chr = False
        elif c == '/' and n == '/':
            in_line = True
            i += 1
        elif c == '/' and n == '*':
            in_block = True
            i += 1
        elif c == '"':
            in_str = True
        elif c == "'":
            in_chr = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    me = os.path.join(a.src, 'encoder', 'me.c')
    if not os.path.exists(me):
        raise SystemExit('not an x264 tree: %s' % me)
    if a.revert:
        bak = me + '.aeg.bak'
        if os.path.exists(bak):
            shutil.copyfile(bak, me)
        for f in ('aeg_mv_hint.c', 'aeg_mv_hint.h'):
            p = os.path.join(a.src, 'encoder', f)
            if os.path.exists(p):
                os.remove(p)
        return
    src = open(me, encoding='utf-8', errors='replace').read()
    if 'aeg_mv_hint_pre' in src:
        raise SystemExit('already patched')
    shutil.copyfile(me, me + '.aeg.bak')
    fn = re.search(r'void\s+x264_me_search_ref\s*\([^;]*?\)\s*\{', src, re.S)
    if not fn:
        raise SystemExit('x264_me_search_ref body not found')
    open_pos = src.index('{', fn.start())
    close_pos = find_matching(src, open_pos)
    if close_pos < 0:
        raise SystemExit('could not find function end')
    src = src[:open_pos + 1] + '\n    aeg_mv_hint_pre(m);' + src[open_pos + 1:]
    close_pos = find_matching(src, open_pos)
    src = src[:close_pos] + '\n    aeg_mv_hint_post(m);\n' + src[close_pos:]
    # include the header after the first include block
    inc = re.search(r'^#include[^\n]*\n', src, re.M)
    if inc:
        src = src[:inc.end()] + '#include "aeg_mv_hint.h"\n' + src[inc.end():]
    open(me, 'w', encoding='utf-8').write(src)
    open(os.path.join(a.src, 'encoder', 'aeg_mv_hint.h'), 'w').write(HEADER)
    open(os.path.join(a.src, 'encoder', 'aeg_mv_hint.c'), 'w').write(SOURCE)
    print('patched %s' % me)


if __name__ == '__main__':
    main()
