#!/bin/sh
# robust_sweep.sh - extreme and hostile inputs for the rate control and the
# two-pass statistics, best run on a sanitizer build:
#
#   CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
#   LDFLAGS="-fsanitize=address,undefined" make BUILD=build-asan build-asan/aegenc
#   FFMPEG=/path/to/ffmpeg sh tools/robust_sweep.sh build-asan/aegenc [workdir]
#
# Every line: the case, the exit status, sanitizer reports, decoder errors and
# the rate.  A refused input is exit 1 or 2 with a reason on stderr; nothing
# may report a sanitizer error or write a stream the decoder complains about.
A=${1:?usage: robust_sweep.sh <aegenc> [workdir]}
W=${2:-/tmp/aeg_sweep}
G=$(dirname "$A")/genyuv
[ -x "$G" ] || G=./build/genyuv
FF=${FFMPEG:-ffmpeg}
A=$(cd "$(dirname "$A")" && pwd)/$(basename "$A")
G=$(cd "$(dirname "$G")" && pwd)/$(basename "$G")
mkdir -p "$W" && cd "$W" || exit 1
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1}
fails=0
"$G" 176 144 40 m1.yuv 1 >/dev/null
"$G" 176 144 40 m6.yuv 6 >/dev/null
"$G" 16 16 10 t16.yuv 1 >/dev/null
"$G" 2 2 10 t2.yuv 0 >/dev/null

run() {
    name=$1; shift
    rc=0; "$A" "$@" -v 1 >"$name.out" 2>"$name.err" || rc=$?
    san=$(grep -c "runtime error\|AddressSanitizer\|LeakSanitizer" "$name.err")
    out=""; prev=""; dec=""
    for a in "$@"; do [ "$prev" = "-o" ] && out=$a; prev=$a; done
    if [ $rc = 0 ] && [ -n "$out" ] && [ -s "$out" ] && command -v "$FF" >/dev/null 2>&1; then
        "$FF" -v error -y -i "$out" -f null - 2>"$name.dec"
        [ -s "$name.dec" ] && dec=" DECODE-ERROR"
    fi
    [ "$san" != 0 ] && fails=1
    [ -n "$dec" ] && fails=1
    [ $rc -gt 2 ] && fails=1
    st=$( [ "$san" != 0 ] && echo " SANITIZER($san)" )
    echo "$name: exit $rc$st$dec $(grep -o 'bitrate [0-9.]* kb/s' "$name.out" | head -1)"
}

S="-w 176 -h 144 -n 40 -i m1.yuv -o a.264"
run abr1      $S --bitrate 1
run abr1e5    $S --bitrate 100000
run abr5_p    $S --bitrate 5 --bframes 0
run held      -w 176 -h 144 -n 40 -i m6.yuv -o a.264 --bitrate 50
run qp30      $S --bitrate 300 --qpmin 30 --qpmax 30
run qpmin51   $S --bitrate 3000 --qpmin 51
run qpmax0    $S --bitrate 10 --qpmax 0
run crf0      $S --crf 0
run crf51     $S --crf 51
run crf05_p   $S --crf 0.5 --bframes 0
run vbv1      $S --bitrate 300 --vbv-maxrate 300 --vbv-bufsize 1
run vbvcap    $S --bitrate 1000 --vbv-maxrate 100 --vbv-bufsize 50
run crfvbv    $S --crf 10 --vbv-maxrate 50 --vbv-bufsize 10
run hrdtiny   $S --bitrate 20 --vbv-maxrate 20 --vbv-bufsize 5 --nal-hrd
run n1        -w 176 -h 144 -n 1 -i m1.yuv -o a.264 --bitrate 300
run t16       -w 16 -h 16 -n 10 -i t16.yuv -o a.264 --crf 23 -p 4
run t2vbv     -w 2 -h 2 -n 10 -i t2.yuv -o a.264 --bitrate 5 --vbv-maxrate 5 --vbv-bufsize 2 --nal-hrd
run slices    $S --bitrate 300 -t 3
run intra     $S --bitrate 1000 -g 1 --bframes 0
run ftcrf     $S --crf 23 --frame-threads 2 -p 3
run cavlc     $S --bitrate 300 --no-cabac
run qc100_p   $S --bitrate 300 --qcomp 100 --bframes 0

P="-w 176 -h 144 -n 40 -i m1.yuv"
"$A" $P -o p.264 --pass 1 --stats b.stats --bitrate 200 -v 0
"$A" $P -o p.264 --pass 1 --stats p.stats --bitrate 200 --bframes 0 -v 0
run 2p_short  -w 176 -h 144 -n 10 -i m1.yuv -o a.264 --pass 2 --stats b.stats --bitrate 200
run 2p_bmis   $P -o a.264 --pass 2 --stats b.stats --bitrate 200 --bframes 0
run 2p_pmis   $P -o a.264 --pass 2 --stats p.stats --bitrate 200
run 2p_g1     $P -o a.264 --pass 2 --stats b.stats --bitrate 200 -g 1
run 2p_vbv    $P -o a.264 --pass 2 --stats b.stats --bitrate 200 --vbv-maxrate 200 --vbv-bufsize 50
printf 'aeg-stats 2\n' > hdr.stats
run 2p_hdr    $P -o a.264 --pass 2 --stats hdr.stats --bitrate 200
{ head -5 b.stats; echo "5 Q 1 2 3"; } > t.stats
run 2p_type   $P -o a.264 --pass 2 --stats t.stats --bitrate 200
awk 'NR==1{print;next}{$5=0;print}' b.stats > z.stats
run 2p_zero   $P -o a.264 --pass 2 --stats z.stats --bitrate 200
awk 'NR==1{print;next}{$3=999;print}' b.stats > q.stats
run 2p_qp     $P -o a.264 --pass 2 --stats q.stats --bitrate 200
awk 'NR==1{print;next}NR%2==0{print}' b.stats > g.stats
run 2p_gaps   $P -o a.264 --pass 2 --stats g.stats --bitrate 200
{ head -3 b.stats; echo "99999999 P 20 20 1000"; } > f.stats
run 2p_far    $P -o a.264 --pass 2 --stats f.stats --bitrate 200
[ $fails = 0 ] && echo "sweep: clean" || echo "sweep: FAILED"
exit $fails
