#!/bin/bash
# BD-rate of aegenc against libx264 on the five real clips (tools/real/fetch.sh).
# usage: [ENV=...] tools/real/bdx.sh "<aegenc opts>" "<x264 params>" [preset]
# Both with scene cuts off.  x264's medium preset with psy=0 is the fair
# PSNR reference ("" for x264's defaults, which trade PSNR for detail).
REAL=${REAL:-/tmp/real}
cd "$(dirname "$0")/../.."
FF=${FFMPEG:-/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
xo=${2-psy=0}
python3 tools/bdrate.py --ffmpeg "$FF" \
  --clip $REAL/cock.yuv 640 352 30 --clip $REAL/vtest.yuv 384 288 30 \
  --clip $REAL/mega.yuv 352 256 30 --clip $REAL/short.yuv 320 240 30 \
  --clip $REAL/tree.yuv 320 240 30 --preset ${3:-3} \
  --extra "$1 --scenecut 0" --xopts "${xo:+$xo:}scenecut=0" 2>&1 |
  grep -E 'BD-rate' | awk '{print $2; s+=$2} END {printf "mean %+.2f%%\n", s/NR}' |
  tr '\n' ' '; echo
