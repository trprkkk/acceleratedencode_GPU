#!/bin/sh
# CI helper: require the OpenCL device (PoCL) to enumerate, and when it does
# not, say why in a form that survives without the job log -- GitHub
# "::error::" annotations, which the checks API returns even where the log
# archive cannot be downloaded.
#
#   . tools/pocl_env.sh && sh tools/ci_device_check.sh ./build/aegenc
#
# Prints the device list and exits 0 when an opencl device is listed.
BIN=${1:-./build/aegenc}
out=${TMPDIR:-/tmp}/aeg_devices.$$
"$BIN" --list-devices >"$out" 2>&1
rc=$?
cat "$out"
if [ $rc -eq 0 ] && grep -q opencl "$out"; then
    rm -f "$out"
    exit 0
fi

# one line per annotation: newlines become " | ", and the length is capped
ann() {
    printf '::error title=%s::%s\n' "$1" \
        "$(printf '%s' "$2" | tr '\n' '|' | sed 's/|/ | /g' | cut -c1-3000)"
}
ann "no OpenCL device (rc $rc)" "$(cat "$out")"
ann "runner CPU" "$(grep -m1 'model name' /proc/cpuinfo) / $(nproc) cpus / \
flags: $(grep -m1 '^flags' /proc/cpuinfo | grep -o 'avx[0-9a-z_]*' | sort -u | tr '\n' ' ')"
ann "environment" "AEG_OPENCL_LIB=${AEG_OPENCL_LIB:-} PYOPENCL_HOME=${PYOPENCL_HOME:-} \
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} icd: $(ls "${PYOPENCL_HOME:-.}"/.libs 2>&1 | tr '\n' ' ')"
if [ -n "${PYOPENCL_HOME:-}" ]; then
    ann "pyopencl enumeration" "$(PYTHONPATH=$(dirname "$PYOPENCL_HOME") \
        python3 -c 'import pyopencl as cl
for p in cl.get_platforms():
    print(p.name, p.version, [d.name for d in p.get_devices()])' 2>&1 | tail -n 15)"
fi
ann "POCL_DEBUG tail" "$(POCL_DEBUG=all "$BIN" --list-devices 2>&1 | tail -n 25)"
rm -f "$out"
exit 1
