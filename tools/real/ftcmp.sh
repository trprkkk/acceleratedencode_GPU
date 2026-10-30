#!/bin/bash
# Frame threads must write the same bytes as the serial encoder: every option
# line of tools/real/ftopts.txt (or stdin with "-") on each clip, coded with
# frame threads (default, or AEG_FTW=n workers) and with --frame-threads 1.
#   tools/real/ftcmp.sh [-]     env: B=binary CLIPS="cock short" N=frames
#                               FTW=workers FTOPT="extra opts for the FT run"
REAL=${REAL:-/tmp/real}
cd "$(dirname "$0")/../.."
B=${B:-./build/aegenc}
T=${TMPDIR:-/tmp}/aeg_ftcmp; mkdir -p $T
declare -A W=([cock]=640x352 [vtest]=384x288 [mega]=352x256 [short]=320x240 [tree]=320x240 [fade]=384x288)
src=tools/real/ftopts.txt; [ "$1" = - ] && src=/dev/stdin
bad=0; tot=0
while read -r opts; do
 [ -z "$opts" ] || [ "${opts:0:1}" = "#" ] && continue
 for n in ${CLIPS:-cock short}; do
  IFS=x read w h <<<"${W[$n]}"
  tot=$((tot+1))
  AEG_FTW=1 $B -w $w -h $h -n ${N:-30} -i $REAL/$n.yuv -o $T/s.264 $opts --frame-threads 1 >/dev/null 2>&1
  ${FTW:+env AEG_FTW=$FTW} $B -w $w -h $h -n ${N:-30} -i $REAL/$n.yuv -o $T/f.264 $opts $FTOPT >$T/log 2>&1
  cmp -s $T/s.264 $T/f.264 || { echo "DIFF $n $opts"; grep -i fail $T/log; bad=$((bad+1)); }
 done
done < $src
echo "ft identical $((tot-bad))/$tot"
[ $bad = 0 ]
