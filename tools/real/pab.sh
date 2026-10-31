#!/bin/bash
# Ours against ours, one thread, global PSNR: each argument is one B side
# (AEG_* assignments first, then aegenc options) against the build's default.
#   P=preset (4)  CL=clips file (/tmp/real/clips.txt; /tmp/real2/clips.txt is
#   the validation set from fetch.sh)  BIN=binary (./build/aegenc)
cd "$(dirname "$0")/../.."
export PYTHONPATH=${PYTHONPATH:-/tmp/dl/ext} ABBD_GLOBAL=1
export FFMPEG=${FFMPEG:-/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
for b in "$@"; do
  echo "== $b"
  python3 tools/abbd.py $(cat ${CL:-/tmp/real/clips.txt}) -p ${P:-4} \
    --bin ${BIN:-./build/aegenc} --a "AEG_X=0 -t 1" --b "$b -t 1" 2>&1 |
    grep -E 'BD-rate' | cut -c1-150
done
