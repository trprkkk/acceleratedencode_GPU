#!/bin/bash
# Fetch the five real test clips (public sample videos from the imageio and
# OpenCV repositories) and convert them to 30-frame I420 files in $REAL
# (default /tmp/real), plus an ffmpeg binary (imageio-ffmpeg) in $DL.
#   cock 640x352 (cockatoo.mp4), vtest 384x288 (vtest.avi), mega 352x256
#   (Megamind.avi), short 320x240 (realshort.mp4), tree 320x240 (tree.avi)
set -e
REAL=${REAL:-/tmp/real}
DL=${DL:-/tmp/dl}
FF=$DL/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
[ -x "$FF" ] || pip install -q --target "$DL/ext" imageio-ffmpeg numpy
mkdir -p "$REAL" "$DL/src"
get() {  # repo path
    [ -s "$DL/src/$(basename "$2")" ] ||
        gh api -H "Accept: application/vnd.github.raw" "repos/$1/contents/$2" \
            > "$DL/src/$(basename "$2")"
}
get imageio/imageio-binaries images/cockatoo.mp4
get imageio/imageio-binaries images/realshort.mp4
get opencv/opencv samples/data/vtest.avi
get opencv/opencv samples/data/Megamind.avi
get opencv/opencv samples/data/tree.avi
cv() { "$FF" -v error -y -i "$DL/src/$1" $3 -frames:v 30 -pix_fmt yuv420p -f rawvideo "$REAL/$2.yuv"; }
cv cockatoo.mp4  cock  "-vf scale=640:352"
cv vtest.avi     vtest "-vf scale=384:288"
cv Megamind.avi  mega  "-vf scale=360:264,crop=352:256"
cv realshort.mp4 short ""
cv tree.avi      tree  ""
# fade.yuv: vtest fading to half brightness over 20 frames and back (the
# weighted-prediction test; not part of clips.txt)
PYTHONPATH=$DL/ext python3 - "$REAL" <<'PY'
import sys, numpy as np
real = sys.argv[1]; w, h = 384, 288; fs = w * h * 3 // 2
d = np.fromfile(real + '/vtest.yuv', np.uint8)
out = []
for k in range(30):
    f = d[k * fs:(k + 1) * fs].astype(np.float32)
    g = 1.0 - 0.025 * k if k < 20 else 0.5 + 0.04 * (k - 20)
    y = f[:w * h] * g; c = (f[w * h:] - 128) * g + 128
    out.append(np.clip(np.concatenate([y, c]) + 0.5, 0, 255).astype(np.uint8))
np.concatenate(out).tofile(real + '/fade.yuv')
PY
echo "--clip $REAL/cock.yuv 640x352 --clip $REAL/vtest.yuv 384x288 --clip $REAL/mega.yuv 352x256 --clip $REAL/short.yuv 320x240 --clip $REAL/tree.yuv 320x240" > "$REAL/clips.txt"
echo "FFMPEG=$FF"
