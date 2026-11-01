#!/bin/sh
# identity.sh NEW OLD [aegenc args...]
#
# Byte-compare the streams of two aegenc binaries over fourteen configurations
# (constant QP at presets 2-4, no B pictures, CRF, CRF + AQ, ABR, VBV, CAVLC,
# slices, slice threads, NAL HRD + AUD, a short GOP, the info SEI).  The gate
# for a change that must not change the 8-bit output: build the old commit in
# a worktree, then
#
#   git worktree add /tmp/old <commit> && make -C /tmp/old -s
#   tools/identity.sh ./build/aegenc /tmp/old/build/aegenc
#
# Extra arguments go to every run (e.g. --bit-depth 10, or --level 40 when the
# automatic level choice changed between the two).  The clips are the ones
# tools/real/fetch.sh writes (CLIPS, default /tmp/real).  Exit status 1 if any
# configuration differs.
N=$1; O=$2
[ -n "$N" ] && [ -n "$O" ] || { echo "usage: $0 NEW OLD [aegenc args...]" >&2; exit 2; }
shift 2
CLIPS=${CLIPS:-/tmp/real}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0
run() { name=$1; shift
    $N "$@" -v 0 -o "$T/n_$name.264" 2>/dev/null
    $O "$@" -v 0 -o "$T/o_$name.264" 2>/dev/null
    if [ -s "$T/n_$name.264" ] && cmp -s "$T/n_$name.264" "$T/o_$name.264"; then
        echo "same $name"
    else
        echo "DIFF $name"; fail=1
    fi
}
V="-i $CLIPS/vtest.yuv -w 384 -h 288 -n 30"
C="-i $CLIPS/cock.yuv -w 640 -h 352 -n 20"
run cqp_p2 $V -p 2 -q 26 "$@"
run cqp_p3 $V -p 3 -q 24 "$@"
run cqp_p4 $C -p 4 -q 28 "$@"
run bf0    $V -p 3 --bframes 0 -q 26 "$@"
run crf    $V -p 3 --crf 23 "$@"
run crfaq  $C -p 3 --crf 22 --aq 10 "$@"
run abr    $V -p 3 --bitrate 300 "$@"
run vbv    $V -p 2 --bitrate 300 --vbv-maxrate 300 --vbv-bufsize 600 "$@"
run cavlc  $V -p 2 --no-cabac -q 25 "$@"
run slices $C -p 2 --slices 3 -q 26 "$@"
run t2     $C -p 3 -t 2 -q 26 "$@"
run hrd    $V -p 2 --bitrate 300 --vbv-maxrate 300 --vbv-bufsize 600 --nal-hrd --aud "$@"
run g5     $V -p 2 -g 5 -q 26 "$@"
run info   $V -p 1 -q 30 --info-sei "$@"
exit $fail
