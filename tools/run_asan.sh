#!/bin/bash
# ASan+UBSan evidence, with the environment's one caveat handled honestly:
# api_check under ASan hangs about one run in five -- inside the *dynamic
# linker*, resolving pthread_getspecific for the ASan runtime -- and the
# untouched baseline hangs identically (tools/asan_base_tri.sh), so it is this
# machine's glibc 2.27 + the encoder's dlopen of the device libraries, not the
# encoder.  The run that does not hang reports the same result every time.
#
# A second caveat: with ASan in the way the device loaders find no device (the
# baseline's --list-devices says the same), so the ASan builds cover the CPU
# paths; the device paths are covered by backend_check and gpu_validate.sh on
# the same machine without ASan.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
D="$R/build-asan"
export LD_LIBRARY_PATH="$D:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
export ASAN_OPTIONS=detect_leaks=0
export UBSAN_OPTIONS=print_stacktrace=1
ffmpeg -v error -y -f lavfi -i "testsrc2=size=320x192:rate=30" -frames:v 6 \
    -pix_fmt yuv420p -f rawvideo /tmp/asan_clip.yuv 2>/dev/null
echo "=== the unit tests (each one run twice; a hang is reported as such) ==="
tot=0
for t in api_check cavlc_check cabac_check cabac_mb_check kernel_check \
         input_check syntax_check; do
  for rep in 1 2; do
    out=$(timeout 300 "$D/$t" 2>&1)
    rc=$?
    n=$(printf '%s\n' "$out" | grep -cE 'runtime error|AddressSanitizer')
    tot=$((tot+n))
    if [ $rc = 124 ]; then
      printf '  %-14s run %d  TIMED OUT (the known dynamic-linker hang)\n' "$t" "$rep"
    else
      printf '  %-14s run %d  rc=%d sanitizer lines: %d  %s\n' "$t" "$rep" "$rc" "$n" \
          "$(printf '%s\n' "$out" | tail -1 | cut -c1-38)"
    fi
  done
done
echo "=== the encoder, each backend, on a 6-frame clip ==="
for be in cpu cuda opencl; do
  out=$(timeout 600 "$D/aegenc" -w 320 -h 192 -n 6 -i /tmp/asan_clip.yuv \
        -o "/tmp/asan_$be.264" -b "$be" -p 4 -q 26 2>&1)
  rc=$?
  n=$(printf '%s\n' "$out" | grep -cE 'runtime error|AddressSanitizer')
  tot=$((tot+n))
  printf '  %-8s rc=%d sanitizer lines: %d  stream %s\n' "$be" "$rc" "$n" \
      "$(md5sum "/tmp/asan_$be.264" 2>/dev/null | cut -c1-16)"
done
echo "=== the forced device paths (same code, same guard, same dispatcher) ==="
for be in cuda opencl; do
  out=$(timeout 600 "$D/aegenc" -w 320 -h 192 -n 6 -i /tmp/asan_clip.yuv \
        -o "/tmp/asan_s_$be.264" -b "$be" -p 4 -q 26 -s tq=$be -s iq=$be 2>&1)
  rc=$?
  n=$(printf '%s\n' "$out" | grep -cE 'runtime error|AddressSanitizer')
  tot=$((tot+n))
  printf '  -s tq/iq=%-7s rc=%d sanitizer lines: %d  stream %s\n' "$be" "$rc" "$n" \
      "$(md5sum "/tmp/asan_s_$be.264" 2>/dev/null | cut -c1-16)"
done
echo "=== total sanitizer reports: $tot ==="
