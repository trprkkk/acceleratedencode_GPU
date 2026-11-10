#!/bin/bash
# One ASan encode through the OpenCL backend: with ASan in the way the device
# loaders find no device (the baseline says the same), so what happens?
set -u
R=/mnt/q/Downloads/HybridEncode/repo
D="$R/build-asan"
export LD_LIBRARY_PATH="$D:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
export ASAN_OPTIONS=detect_leaks=0
ffmpeg -v error -y -f lavfi -i "testsrc2=size=320x192:rate=30" -frames:v 6 \
    -pix_fmt yuv420p -f rawvideo /tmp/asan_clip.yuv 2>/dev/null
t0=$(date +%s)
"$D/aegenc" -w 320 -h 192 -n 6 -i /tmp/asan_clip.yuv -o /tmp/asan_ocl.264 \
    -b opencl -p 4 -q 26 > /tmp/asan_ocl.log 2>&1
echo "rc=$? after $(( $(date +%s) - t0 )) s"
tail -3 /tmp/asan_ocl.log
ls -la /tmp/asan_ocl.264 2>&1 | tail -1
grep -cE 'runtime error|AddressSanitizer' /tmp/asan_ocl.log
