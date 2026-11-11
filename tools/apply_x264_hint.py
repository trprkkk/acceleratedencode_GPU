#!/usr/bin/env python3
"""Apply/revert the minimal x264 MV hint/dump patch.

    python3 tools/apply_x264_hint.py --src forks/x264
    python3 tools/apply_x264_hint.py --src forks/x264 --revert

The patch is env-gated:
  X264_MV_HINT=<file>   seed candidate MV per x264_me_search_ref call
  X264_MV_DUMP=<file>   dump best mv/cost/i_pixel per call
Unset both => stock x264 behavior and byte-identical output.
"""
import argparse
import os
import re
import shutil

HEADER = r'''#ifndef AEG_MV_HINT_H
#define AEG_MV_HINT_H

#include "common/common.h"
#include "me.h"

int aeg_mv_hint_get( int *x, int *y );
void aeg_mv_hint_post( const x264_t *h, const x264_me_t *m );

#endif
'''

SOURCE = r'''/* GPL-2.0-or-later.  Minimal external MV seed/dump for the x264 fork. */
#include "common/common.h"
#include "aeg_mv_hint.h"
#include <stdio.h>
#include <stdlib.h>

static FILE *hint_file;
static FILE *dump_file;

int aeg_mv_hint_get( int *x, int *y )
{
    const char *ev = getenv( "X264_MV_HINT" );
    if( !ev )
        return 0;
    if( !hint_file )
        hint_file = fopen( ev, "r" );
    if( !hint_file )
        return 0;
    return fscanf( hint_file, "%d %d", x, y ) == 2;
}

void aeg_mv_hint_post( const x264_t *h, const x264_me_t *m )
{
    const char *ev = getenv( "X264_MV_DUMP" );
    if( !ev )
        return;
    if( !dump_file )
        dump_file = fopen( ev, "a" );
    if( !dump_file )
        return;
    fprintf( dump_file, "%d %d %d %d\n", m->mv[0], m->mv[1], m->cost,
             m->i_pixel );
    fflush( dump_file );
}
'''

HINT = '''    {
        int hx, hy;
        if( aeg_mv_hint_get( &hx, &hy ) && CHECK_MVRANGE( hx, hy ) )
            COST_MV( hx, hy );
    }
'''


def matching_brace(src, pos):
    depth = 0
    i = pos
    while i < len(src):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
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
    mk = os.path.join(a.src, 'Makefile')
    if not os.path.exists(me):
        raise SystemExit('not an x264 tree: %s' % me)
    if a.revert:
        for p in (me, mk):
            bak = p + '.aeg.bak'
            if os.path.exists(bak):
                shutil.copyfile(bak, p)
        for f in ('aeg_mv_hint.c', 'aeg_mv_hint.h'):
            p = os.path.join(a.src, 'encoder', f)
            if os.path.exists(p):
                os.remove(p)
        print('reverted')
        return
    src = open(me, encoding='utf-8', errors='replace').read()
    if 'aeg_mv_hint_get' in src:
        raise SystemExit('already patched')
    shutil.copyfile(me, me + '.aeg.bak')
    shutil.copyfile(mk, mk + '.aeg.bak')

    if '#include "aeg_mv_hint.h"' not in src:
        inc = src.find('#include "me.h"')
        if inc < 0:
            raise SystemExit('#include "me.h" not found')
        end = src.find('\n', inc) + 1
        src = src[:end] + '#include "aeg_mv_hint.h"\n' + src[end:]

    sw = src.find('switch( h->mb.i_me_method )')
    if sw < 0:
        raise SystemExit('me method switch not found')
    src = src[:sw] + HINT + '\n' + src[sw:]

    fn = src.find('void x264_me_search_ref(')
    if fn < 0:
        raise SystemExit('x264_me_search_ref not found')
    op = src.find('{', fn)
    cl = matching_brace(src, op)
    if cl < 0:
        raise SystemExit('function end not found')
    src = src[:cl] + '    aeg_mv_hint_post( h, m );\n' + src[cl:]
    open(me, 'w', encoding='utf-8').write(src)

    mksrc = open(mk, encoding='utf-8', errors='replace').read()
    mksrc = mksrc.replace('encoder/me.c encoder/ratecontrol.c',
                          'encoder/me.c encoder/aeg_mv_hint.c '
                          'encoder/ratecontrol.c', 1)
    open(mk, 'w', encoding='utf-8').write(mksrc)

    open(os.path.join(a.src, 'encoder', 'aeg_mv_hint.h'), 'w').write(HEADER)
    open(os.path.join(a.src, 'encoder', 'aeg_mv_hint.c'), 'w').write(SOURCE)
    print('patched %s' % me)


if __name__ == '__main__':
    main()
