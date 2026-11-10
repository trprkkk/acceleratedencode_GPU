#!/bin/bash
# The same api_check-under-ASan loop on the untouched baseline: does the
# dynamic linker crash there too?
set -u
O=/mnt/q/Downloads/HybridEncode/baseline-repo
D="$O/build-asan"
make -C "$O" BUILD="$D" OPENCL=1 CUDA=1 \
    CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c11 -Wall -Wextra -Iinclude -Isrc" \
    LDFLAGS="-fsanitize=address,undefined" "$D/api_check" >/dev/null 2>&1
export LD_LIBRARY_PATH="$D:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
export ASAN_OPTIONS=detect_leaks=0
for i in 1 2 3 4 5 6; do
  t0=$(date +%s)
  out=$(timeout 900 "$D/api_check" 2>&1)
  rc=$?
  echo "baseline run $i: rc=$rc  $(( $(date +%s) - t0 )) s  last: \
$(printf '%s\n' "$out" | tail -1 | cut -c1-40)"
done
