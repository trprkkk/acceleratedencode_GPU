#!/usr/bin/env python3
"""Token-similarity scan of this encoder's C functions against another C
code base (x264, FFmpeg, ...), for the provenance notes (docs/PROVENANCE.md).

    tools/simscan.py OTHER_SRC_DIR [--k 20] [--min-tokens 60] [--top 40]
                     [--raw] [--files src/*.c cli/aegenc.c]

Both sides are tokenised with comments and string contents removed.  By
default identifiers become one token "I" and numbers "N", so the scan looks
at the *shape* of the code, not at names; --raw keeps identifiers and
numbers, which finds code that was copied with its names.  Every run of k
consecutive tokens (a shingle) of an aeg function is looked up in the other
code base; a function's score is the share of its shingles found there, with
the other side's file:line of the most matched function.

This is the measure MOSS-like tools use.  A high score means the token
stream is the same over long stretches -- a reason to look at the pair, not a
finding by itself (short arithmetic kernels, butterflies and table lookups
that the standard fixes come out alike whoever writes them).  A low score
does not show independent creation either: an algorithm can be followed
closely in different code.  It is not a legal judgement.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

KEYWORDS = set("""auto break case char const continue default do double else enum
extern float for goto if inline int long register restrict return short signed
sizeof static struct switch typedef union unsigned void volatile while _Bool
uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t size_t
pixel""".split())

TOK = re.compile(r"""
    (?P<id>[A-Za-z_]\w*) |
    (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|\d+\.?\d*(?:[eE][-+]?\d+)?[uUlLfF]*|\.\d+) |
    (?P<op><<=|>>=|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^!~<>=?:;,.(){}\[\]])
""", re.X)


def strip(src):
    """Comments and string/char literal contents out, newlines kept."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * src.count("\n", i, j))
            i = j
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out).replace("\\\n", " \n")


def tokens(src, raw):
    """[(token, line)] of stripped source."""
    res = []
    for ln, line in enumerate(src.split("\n"), 1):
        if line.lstrip().startswith("#") and not line.lstrip().startswith("#define"):
            continue                      # includes, conditionals
        for m in TOK.finditer(line):
            t = m.group(0)
            if m.group("id"):
                if t == "define":
                    continue
                t = t if (raw or t in KEYWORDS) else "I"
            elif m.group("num"):
                t = t if raw else "N"
            res.append((t, ln))
    return res


def functions(path, raw):
    """Top-level brace bodies that follow a ')' -- functions (and the
    function-like macros of x264's DITHER_PLANE kind): [(name, line, toks)]."""
    src = strip(open(path, encoding="utf-8", errors="replace").read())
    names = tokens(src, True)
    toks = tokens(src, raw)
    res, depth, start = [], 0, None
    for i, (t, ln) in enumerate(names):
        if t == "{":
            if depth == 0 and i > 0 and names[i - 1][0] == ")":
                # name: the identifier before the matching '('
                d, j = 0, i - 1
                while j >= 0:
                    if names[j][0] == ")":
                        d += 1
                    elif names[j][0] == "(":
                        d -= 1
                        if d == 0:
                            break
                    j -= 1
                name = names[j - 1][0] if j > 0 else "?"
                start = (i, name, ln)
            depth += 1
        elif t == "}":
            depth = max(0, depth - 1)
            if depth == 0 and start:
                s, name, ln0 = start
                lo = max(0, s - 40)       # the signature too
                res.append((name, ln0, [x[0] for x in toks[lo:i + 1]]))
                start = None
    return res


def shingles(ts, k):
    return {hash(tuple(ts[i:i + k])) for i in range(len(ts) - k + 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("other")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min-tokens", type=int, default=60)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--files", nargs="*")
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = a.files or sorted(glob.glob(os.path.join(here, "src", "*.c")) +
                              [os.path.join(here, "cli", "aegenc.c")])
    other = [p for p in glob.glob(os.path.join(a.other, "**", "*.[ch]"),
                                  recursive=True)]
    index = defaultdict(set)               # shingle -> {other function id}
    ofuncs = []
    for p in other:
        for name, ln, ts in functions(p, a.raw):
            fid = len(ofuncs)
            ofuncs.append("%s:%d %s" % (os.path.relpath(p, a.other), ln, name))
            for h in shingles(ts, a.k):
                index[h].add(fid)
    rows = []
    for p in files:
        for name, ln, ts in functions(p, a.raw):
            if len(ts) < a.min_tokens:
                continue
            sh = shingles(ts, a.k)
            if not sh:
                continue
            hit, per = 0, defaultdict(int)
            for h in sh:
                if h in index:
                    hit += 1
                    for fid in index[h]:
                        per[fid] += 1
            if hit:
                best = max(per, key=per.get)
                rows.append((hit / len(sh), len(ts), "%s:%d %s" %
                             (os.path.relpath(p, here), ln, name),
                             ofuncs[best], per[best] / len(sh)))
    rows.sort(reverse=True)
    print("k=%d, %s tokens, %d other functions; share of shingles found:"
          % (a.k, "raw" if a.raw else "normalised", len(ofuncs)))
    for score, n, me, them, sbest in rows[:a.top]:
        print("%5.1f%%  %5d tok  %-48s  best: %s (%.1f%%)"
              % (100 * score, n, me, them, 100 * sbest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
