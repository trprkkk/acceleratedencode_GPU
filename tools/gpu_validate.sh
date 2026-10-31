#!/bin/sh
# Verify on a machine with a GPU that every batched (device) code path is
# correct, and measure what it buys.
#
#   tools/gpu_validate.sh clip.yuv W H FRAMES QP [extra aegenc args]
#
# Build first with the backend you want to test:
#
#   make OPENCL=1        # or: make CUDA=1
#
# (and run build/backend_check first: it checks every device operation on its
# own, which says *which* one is wrong when a run below differs)
#
# The clip is encoded several times:
#
#   cpu         -b cpu: the reference stream, everything on the CPU
#   auto        the dispatcher free to choose (what a user gets)
#   coefbatch   AEG_COEF_BATCH=1: tq_batch + iq_idct_batch over the sixteen
#               luma blocks of one inter candidate
#   rdbatch     AEG_RD_BATCH=1: the same two kernels over *every* inter
#               candidate of a macroblock at once, decided afterwards
#   search      AEG_SEED=1 AEG_PARFULL=1: the whole-frame search rounds
#               (sad_batch for the integer rounds, satd_batch for the
#               fractional ones; the per-macroblock subpel walk is skipped)
#   all         all of the above at once
#
# Set BE=opencl (or BE=cuda) to add one more run that *forces* the two
# coefficient stages and the interpolation planes through that backend with
# `-s tq=... -s iq=... -s inter=...`.  The
# dispatcher is free not to pick those stages (it may measure the CPU as
# faster), in which case `auto` is silently all-CPU; that forced run is the one
# that tells whether the batched form pays for itself on the device.
#
# Two checks, and they are different on purpose:
#
#   * cpu, auto, coefbatch, rdbatch must produce the *same bitstream*.  Moving a
#     coefficient stage to a device can only change the speed: the kernels are
#     integer arithmetic and bit-identical to the CPU reference.
#   * search and all use a different motion search (whole-frame rounds instead
#     of the serial walk), so their stream legitimately differs from cpu's --
#     but not from `searchcpu`, the same search with the CPU implementations
#     of sad_batch / satd_batch (-b cpu): the device must reproduce that byte
#     for byte.
#
# And for every run, the encoder's own reconstruction has to match an
# independent decode of its stream sample for sample (--dump-recon against
# ffmpeg; the dump is in coding order and the decode in display order, so the
# pictures are compared as a set of per-frame hashes); a mismatch would mean
# the encoder and the decoder disagree, which drifts on every following
# picture.
#
# No GPU at hand?  Any OpenCL implementation runs the device paths, a CPU one
# included: see tools/pocl_env.sh.
#
# Exit status: 0 on success, 1 on any mismatch or decode failure.
set -u

CLIP=${1:-}
W=${2:-}
H=${3:-}
FRAMES=${4:-}
QP=${5:-}
if [ -z "$CLIP" ] || [ -z "$QP" ]; then
    echo "usage: tools/gpu_validate.sh clip.yuv W H FRAMES QP [extra aegenc args]" >&2
    exit 2
fi
shift 5

BIN=${BIN:-./build/aegenc}
FFMPEG=${FFMPEG:-ffmpeg}
TMP=${TMP:-${TMPDIR:-/tmp}/aeg_gpu_validate.$$}
mkdir -p "$TMP" || exit 1
trap 'rm -rf "$TMP"' EXIT

echo "== aeg GPU path validation"
echo "   binary $BIN, clip $CLIP ${W}x${H}, $FRAMES frames, QP $QP"
"$BIN" --list-devices || exit 1

fail=0

# run NAME [ENV=VAL ...] -- aegenc-args: encode, keep the stream and the
# reconstruction, print the fps and the stage split from -v 2
run()
{
    name=$1
    shift
    envs=""
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        envs="$envs $1"
        shift
    done
    [ $# -gt 0 ] && shift
    # shellcheck disable=SC2086
    env $envs "$BIN" -w "$W" -h "$H" -n "$FRAMES" -q "$QP" -i "$CLIP" \
        -o "$TMP/$name.264" --dump-recon "$TMP/$name.yuv" -v 2 "$@" \
        >"$TMP/$name.log" 2>"$TMP/$name.err"
    if [ $? -ne 0 ]; then
        echo "   $name: ENCODE FAILED"
        sed -n 1,5p "$TMP/$name.err" >&2
        fail=1
        return 1
    fi
    fps=$(sed -n 's/.*(\([0-9.]*\) fps).*/\1/p' "$TMP/$name.log" | tail -1)
    gpu=$(sed -n 's/^ *stage \([a-z-]*\) *[0-9.]* ms *\([a-z]*\).*/\1(\2) /p' \
          "$TMP/$name.log" | tr -d '\n' | tr -s ' ')
    bytes=$(wc -c <"$TMP/$name.264" | tr -d ' ')
    psnr=$(sed -n 's/.*PSNR \([0-9.]*\) dB.*/\1/p' "$TMP/$name.log" | tail -1)
    printf '   %-10s %8s fps %7s B %6s dB  %s\n' \
        "$name" "${fps:-?}" "$bytes" "${psnr:-?}" "$gpu"
    return 0
}

run cpu  -- -b cpu "$@" || fail=1
run auto -- "$@" || fail=1
run coefbatch AEG_COEF_BATCH=1 -- "$@" || fail=1
run rdbatch AEG_RD_BATCH=1 -- "$@" || fail=1
run search AEG_SEED=1 AEG_PARFULL=1 -- "$@" || fail=1
# the same whole-frame search with the CPU implementations of its operations:
# the device must reproduce it byte for byte
run searchcpu AEG_SEED=1 AEG_PARFULL=1 -- -b cpu "$@" || fail=1
if [ -n "${BE:-}" ]; then
    run stageforce -- -s "tq=$BE" -s "iq=$BE" -s "inter=$BE" "$@" || fail=1
fi
run all AEG_SEED=1 AEG_PARFULL=1 AEG_COEF_BATCH=1 AEG_RD_BATCH=1 -- "$@" \
    || fail=1

echo "== bitstream: the coefficient stages may not change a single byte"
same="auto coefbatch rdbatch"
[ -n "${BE:-}" ] && same="$same stageforce"
for name in $same; do
    if [ ! -f "$TMP/$name.264" ] || [ ! -f "$TMP/cpu.264" ]; then
        echo "   $name: no stream to compare"
        fail=1
        continue
    fi
    if cmp -s "$TMP/cpu.264" "$TMP/$name.264"; then
        echo "   $name: identical to the CPU stream"
    else
        n=$(cmp -l "$TMP/cpu.264" "$TMP/$name.264" 2>/dev/null | wc -l | tr -d ' ')
        echo "   $name: DIFFERS from the CPU stream ($n bytes)"
        fail=1
    fi
done
# search/all are a different motion search from the serial one, so they are
# compared with that search run on the CPU backend instead
for name in search all; do
    if [ ! -f "$TMP/$name.264" ] || [ ! -f "$TMP/searchcpu.264" ]; then
        echo "   $name: no stream to compare"
        fail=1
        continue
    fi
    if cmp -s "$TMP/searchcpu.264" "$TMP/$name.264"; then
        echo "   $name: identical to the same search on the CPU backend"
    else
        echo "   $name: DIFFERS from the same search on the CPU backend"
        fail=1
    fi
done

# frame_hashes FILE: sorted md5 of every W x H I420 picture in FILE
frame_hashes()
{
    rm -rf "$TMP/split"
    mkdir -p "$TMP/split"
    split -b $((W * H * 3 / 2)) -d -a 5 "$1" "$TMP/split/f." || return 1
    md5sum "$TMP/split"/f.* | cut -c1-32 | sort
}

echo "== reconstruction: the encoder's samples against an independent decode"
for name in cpu auto coefbatch rdbatch search searchcpu all stageforce; do
    [ -f "$TMP/$name.264" ] || continue
    if ! "$FFMPEG" -v error -y -i "$TMP/$name.264" -f rawvideo -pix_fmt yuv420p \
            "$TMP/$name.dec.yuv" 2>"$TMP/$name.dec.log"; then
        echo "   $name: DECODE FAILED"
        sed -n 1,3p "$TMP/$name.dec.log"
        fail=1
        continue
    fi
    if [ -s "$TMP/$name.dec.log" ]; then
        echo "   $name: the decoder logged an error"
        sed -n 1,3p "$TMP/$name.dec.log"
        fail=1
        continue
    fi
    # --dump-recon writes coding order, the decoder display order: with
    # B-frames the two differ, so compare the pictures as a multiset of
    # per-frame hashes (any wrong sample still changes a hash)
    if [ "$(frame_hashes "$TMP/$name.yuv")" = \
         "$(frame_hashes "$TMP/$name.dec.yuv")" ]; then
        echo "   $name: reconstruction == decode"
    else
        echo "   $name: RECONSTRUCTION DIFFERS from the decode"
        fail=1
    fi
done

if [ $fail -ne 0 ]; then
    echo "== FAIL"
    exit 1
fi
echo "== PASS: every device path matches the CPU reference"
