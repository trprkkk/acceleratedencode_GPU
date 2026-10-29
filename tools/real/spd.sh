#!/bin/bash
# Wall time of two encoders on one clip, alternating runs (noise guard).
# usage: tools/real/spd.sh A_BIN B_BIN ["opts"] [runs]   env CLIP=/tmp/cock180.yuv WH=640x352 N=90
A=$1; B=$2; opts=${3:--p 3 -q 28}; runs=${4:-3}
CLIP=${CLIP:-/tmp/cock180.yuv}; IFS=x read w h <<<"${WH:-640x352}"; N=${N:-90}
for i in $(seq $runs); do
 for X in A B; do
  bin=${!X}; s=$(date +%s%N)
  $bin -w $w -h $h -n $N -i $CLIP -o /tmp/spd_$X.264 $opts >/dev/null 2>&1
  e=$(date +%s%N); echo "$X $(( (e-s)/1000000 ))"
 done
done | awk '{t[$1]+=$2; n[$1]++; v[$1]=v[$1]" "$2} END {for (k in t) printf "%s mean %.0f ms (%s)\n", k, t[k]/n[k], v[k]; printf "B/A %.3f\n", t["B"]/t["A"]}'
cmp -s /tmp/spd_A.264 /tmp/spd_B.264 && echo identical || echo DIFFERENT
