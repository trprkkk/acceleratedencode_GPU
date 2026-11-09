#!/bin/bash
# Long clips for tools/x264cmp.py (--clips long, and `speed`): run after
# tools/real/fetch.sh.  cockL 640x352 240 pictures, vtestL 384x288 300,
# megaL 352x256 240, and hdL: cockatoo scaled up to 1280x720, 120 pictures
# (speed only -- upscaled content is smoother than real 720p).
set -e
REAL=${REAL:-/tmp/real}; DL=${DL:-/tmp/dl}
FF=$DL/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
# an existing clip is kept: rewriting it would change its mtime (part of the
# x264cmp cache key) and corrupt any encode reading it meanwhile
cv() { [ -s "$REAL/$2.yuv" ] && return 0
       "$FF" -v error -y -i "$DL/src/$1" -vf "$3" -frames:v $4 -pix_fmt yuv420p -f rawvideo "$REAL/$2.yuv.tmp"
       mv "$REAL/$2.yuv.tmp" "$REAL/$2.yuv"; }
cv cockatoo.mp4 cockL scale=640:352 240
cv vtest.avi    vtestL scale=384:288 300
cv Megamind.avi megaL "scale=360:264,crop=352:256" 240
cv cockatoo.mp4 hdL   scale=1280:720 120
# validation set (--clips long2): other content of the same sources, coded at
# native resolution (cockatoo and Megamind cropped, not scaled), vtest's
# later minute and tree in full
cv cockatoo.mp4 cockL2  "crop=640:352:520:300" 240
cv vtest.avi    vtestL2 "select=gte(n\\,400),setpts=N/TB,crop=384:288:300:200" 300
cv Megamind.avi megaL2  "crop=352:256:200:150" 240
cv tree.avi     treeL   "format=yuv420p" 240
ls -l "$REAL"/*L.yuv
# a fixed mtime (part of the tools/x264cmp.py cache key): the clips are
# deterministic, so a cache survives their re-creation
touch -d @1700000000 "$REAL"/*.yuv ${REAL2:-/tmp/real2}/*.yuv 2>/dev/null || true
