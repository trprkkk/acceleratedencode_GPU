#!/bin/bash
# Ours against ours on the five real clips: tools/real/ab.sh "<A>" "<B>"
# (each side: AEG_* environment assignments followed by aegenc options).
# ABBD_GLOBAL=1: PSNR of the sequence MSE instead of the per-picture mean
# (a black picture coded losslessly otherwise counts as 100+ dB).
REAL=${REAL:-/tmp/real}
cd "$(dirname "$0")/../.."
export PYTHONPATH=${PYTHONPATH:-/tmp/dl/ext}
export FFMPEG=${FFMPEG:-/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
python3 tools/abbd.py $(cat $REAL/clips.txt) --a "$1" --b "$2" | grep -E 'BD-rate' | cut -c1-120
