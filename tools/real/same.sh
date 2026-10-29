#!/bin/bash
# Speed work guard: the encoder under test must write the same bytes as a
# reference binary, on every real clip and preset, and this prints how long
# each took.  usage: tools/real/same.sh /path/to/ref/aegenc ["extra opts"]
#   PRESETS="0 1 2 3 4"  QPS="24 32"  CLIPS="cock vtest ..."  (env overrides)
REAL=${REAL:-/tmp/real}
cd "$(dirname "$0")/../.."
REF=$1; opts=$2
T=${TMPDIR:-/tmp}/aeg_same; mkdir -p $T
declare -A W=([cock]=640x352 [vtest]=384x288 [mega]=352x256 [short]=320x240 [tree]=320x240)
bad=0; tot=0; ta=0; tb=0
for p in ${PRESETS:-0 1 2 3 4}; do
 pa=0; pb=0
 for n in ${CLIPS:-cock vtest mega short tree}; do
  IFS=x read w h <<<"${W[$n]}"
  for q in ${QPS:-24 32}; do
   tot=$((tot+1))
   s=$(date +%s%N)
   $REF -w $w -h $h -n 30 -i $REAL/$n.yuv -o $T/ref.264 -p $p -q $q $opts >/dev/null 2>&1
   m=$(date +%s%N)
   ./build/aegenc -w $w -h $h -n 30 -i $REAL/$n.yuv -o $T/new.264 -p $p -q $q $opts >/dev/null 2>&1
   f=$(date +%s%N)
   pa=$((pa+m-s)); pb=$((pb+f-m))
   cmp -s $T/ref.264 $T/new.264 || { echo "DIFF p$p $n q$q"; bad=$((bad+1)); }
  done
 done
 ta=$((ta+pa)); tb=$((tb+pb))
 python3 -c "print('preset $p: ref %.2fs new %.2fs  x%.2f' % ($pa/1e9, $pb/1e9, $pa/max($pb,1)))"
done
python3 -c "print('total: ref %.2fs new %.2fs  x%.2f   identical %d/%d' % ($ta/1e9, $tb/1e9, $ta/max($tb,1), $tot-$bad, $tot))"
