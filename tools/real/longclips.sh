#!/bin/bash
# Long clips for tools/x264cmp.py (--clips long, and `speed`): run after
# tools/real/fetch.sh.  cockL 640x352 240 pictures, vtestL 384x288 300,
# megaL 352x256 240, and hdL: cockatoo scaled up to 1280x720, 120 pictures
# (speed only -- upscaled content is smoother than real 720p).
set -e
REAL=${REAL:-/tmp/real}; DL=${DL:-/tmp/dl}
FF=$DL/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
cv() { "$FF" -v error -y -i "$DL/src/$1" -vf "$3" -frames:v $4 -pix_fmt yuv420p -f rawvideo "$REAL/$2.yuv"; }
cv cockatoo.mp4 cockL scale=640:352 240
cv vtest.avi    vtestL scale=384:288 300
cv Megamind.avi megaL "scale=360:264,crop=352:256" 240
cv cockatoo.mp4 hdL   scale=1280:720 120
ls -l "$REAL"/*L.yuv
