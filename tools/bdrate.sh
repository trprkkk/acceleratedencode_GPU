#!/bin/bash
# Quality against x264 and x265 at matched rate: each encoder is run at several
# quality points, every stream is decoded with ffmpeg and scored against the
# source, and Bjontegaard's delta-rate says how many percent more (or fewer)
# bits one encoder needs for the same quality.
#
#   usage: bdrate.sh [clip] [frames] [w] [h] [preset] [extra aegenc args...]
#
# The number that matters is the last one: aegenc against x264 (and x265),
# negative = aegenc needs fewer bits for the same quality.
set -u
CLIP="${1:-/tmp/aegbench/src_1280x720_30.yuv}"
N="${2:-30}"
W="${3:-1280}"
H="${4:-720}"
PRESET="${5:-4}"
shift 5 2>/dev/null || shift $#
AEG=/mnt/q/Downloads/HybridEncode/repo/build-work/aegenc
WORK=/tmp/aegbd
export LD_LIBRARY_PATH="/mnt/q/Downloads/HybridEncode/repo/build-work:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
rm -rf "$WORK"; mkdir -p "$WORK"
# the clip's own size, so a mismatched width/height fails loudly instead of
# scoring garbage
case "$CLIP" in */*) ;; *) CLIP="$PWD/$CLIP";; esac
if ! [ -s "$CLIP" ]; then
    echo "no clip at $CLIP" >&2
    exit 1
fi
want=$((W * H * 3 / 2 * N))
got=$(stat -c %s "$CLIP")
if [ "$got" -ne "$want" ]; then
    echo "$CLIP is $got bytes, but ${W}x${H} $N frames of yuv420p is $want" >&2
    echo "  (pass the clip the script generated, or the right width/height)" >&2
    exit 2
fi

psnr_of() {  # stream -> "kbits/s dB"
  local f="$1" dec="$WORK/d.yuv" q
  rm -f "$dec"
  ffmpeg -v error -y -vsync 0 -i "$f" -f rawvideo -pix_fmt yuv420p "$dec" 2>/dev/null
  q=$(ffmpeg -hide_banner -f rawvideo -s ${W}x${H} -pix_fmt yuv420p -i "$CLIP" \
        -f rawvideo -s ${W}x${H} -pix_fmt yuv420p -i "$dec" -lavfi psnr \
        -f null - 2>&1 | grep -o 'average:[0-9.]*' | head -1 | cut -d: -f2)
  [ -z "$q" ] && q=99.0
  echo "$(python3 -c "print('%.1f' % ($(stat -c %s "$f") * 8 / 1000.0 / $N * 30))") $q"
}

# "rate dB" lines, one per quality point
: > "$WORK/aeg.txt"; : > "$WORK/x264.txt"; : > "$WORK/x265.txt"
echo "### $(basename "$CLIP") ${W}x${H} ${N} frames, aegenc preset ${PRESET}${*:+ $*}"
for q in 20 26 32 38 44; do
  # aegenc at constant QP (its rate control is compared in the separate sweep)
  f="$WORK/aeg_$q.264"
  "$AEG" -w "$W" -h "$H" -n "$N" -i "$CLIP" -o "$f" -p "$PRESET" -q "$q" "$@" \
      >/dev/null 2>&1
  echo "$(psnr_of "$f")" >> "$WORK/aeg.txt"
  f="$WORK/x264_$q.264"
  x264 --qp "$q" --preset medium --input-res ${W}x${H} --fps 30 \
       --keyint 250 --min-keyint 25 --bframes 3 --ref 4 -o "$f" "$CLIP" \
       >/dev/null 2>&1
  echo "$(psnr_of "$f")" >> "$WORK/x264.txt"
  f="$WORK/x265_$q.hevc"
  x265 --qp "$q" -p 4 --input-res ${W}x${H} --fps 30 --keyint 250 \
       --min-keyint 25 --bframes 3 --ref 4 -o "$f" "$CLIP" >/dev/null 2>&1
  echo "$(psnr_of "$f")" >> "$WORK/x265.txt"
  printf '   qp %-3s aegenc %-16s x264 %-16s x265 %-16s\n' "$q" \
      "$(tail -1 "$WORK/aeg.txt")" "$(tail -1 "$WORK/x264.txt")" \
      "$(tail -1 "$WORK/x265.txt")"
done
echo
python3 - "$WORK/aeg.txt" "$WORK/x264.txt" "$WORK/x265.txt" <<'PY'
import sys

def read(p):
    d = []
    for line in open(p):
        a, b = line.split()
        d.append((float(a), float(b)))
    return sorted(d)

def bd_rate(a, b):       # % more bits b needs for a's quality (negative = b wins)
    # Bjontegaard's delta rate: the average of the log-rate difference over the
    # PSNR range both share, from cubic splines of log(rate) against PSNR
    def spline(d):
        n = len(d)
        xs = [p[1] for p in d]
        ys = [__import__('math').log(p[0]) for p in d]
        m = [0.0] * n
        for i in range(1, n - 1):
            m[i] = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1])
        def f(x):
            for i in range(n - 1):
                if xs[i] <= x <= xs[i + 1]:
                    h = xs[i + 1] - xs[i]
                    t = (x - xs[i]) / h
                    a = ys[i] * (1 - t) + ys[i + 1] * t
                    # cubic Hermite on the x-y spline, y = ys
                    y0, y1 = ys[i], ys[i + 1]
                    d0, d1 = m[i] * h, m[i + 1] * h
                    return (y0 * (2 * t ** 3 - 3 * t ** 2 + 1)
                            + d0 * (t ** 3 - 2 * t ** 2 + t)
                            + y1 * (-2 * t ** 3 + 3 * t ** 2)
                            + d1 * (t ** 3 - t ** 2))
            return ys[0] if x < xs[0] else ys[-1]
        return f, xs[0], xs[-1]
    fa, lo_a, hi_a = spline(a)
    fb, lo_b, hi_b = spline(b)
    lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
    if hi <= lo:
        return float('nan')
    n = 100
    integ = sum(fb(lo + (hi - lo) * k / n) - fa(lo + (hi - lo) * k / n)
                for k in range(n + 1)) / (n + 1)
    return (__import__('math').exp(integ) - 1) * 100.0

aeg, x264, x265 = read(sys.argv[1]), read(sys.argv[2]), read(sys.argv[3])
print("  average PSNR range: aegenc %.2f..%.2f dB, x264 %.2f..%.2f, x265 %.2f..%.2f"
      % (aeg[0][1], aeg[-1][1], x264[0][1], x264[-1][1], x265[0][1], x265[-1][1]))
for name, other in (("x264", x264), ("x265", x265)):
    r = bd_rate(aeg, other)
    if r == r:
        print("  BD-rate: %-6s needs %+7.2f%% of the bits aegenc needs for the "
              "same quality -> %s" % (name, r,
              "aegenc is better" if r < 0 else "aegenc needs more bits"))
    else:
        print("  BD-rate against %s: the quality ranges do not overlap" % name)
PY
