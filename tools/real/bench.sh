#!/bin/bash
# Two builds against each other: min / mean CPU seconds (user+sys of the child)
# over alternating runs, then whether the two streams are byte-identical.
#   tools/real/bench.sh BIN_A BIN_B ["aegenc opts"] [runs]
#   env CLIP=yuv (/tmp/cock180.yuv, 640x352)  N=frames (60)
A=$1; B=$2; opts=${3:--p 3 -q 28 -t 1}; n=${4:-3}
CLIP=${CLIP:-/tmp/cock180.yuv}; N=${N:-60}; T=${TMPDIR:-/tmp}
for i in $(seq $n); do for X in A B; do bin=${!X}
 python3 -c "
import sys,resource,subprocess
subprocess.run(sys.argv[2:],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
r=resource.getrusage(resource.RUSAGE_CHILDREN); print(sys.argv[1], r.ru_utime+r.ru_stime)" $X $bin -w 640 -h 352 -n $N -i $CLIP -o $T/bench_$X.264 $opts
done; done | awk '{t=$2; if(!($1 in m)||t<m[$1])m[$1]=t; s[$1]+=t; c[$1]++} END{printf "A min %.2f mean %.2f | B min %.2f mean %.2f | B/A(min) %.3f\n", m["A"], s["A"]/c["A"], m["B"], s["B"]/c["B"], m["B"]/m["A"]}'
cmp -s $T/bench_A.264 $T/bench_B.264 && echo identical || echo DIFFERENT
