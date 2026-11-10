#!/bin/bash
# CRF over a long clip, on the encoder's own reconstruction dump, reordered
# into display order with --dump-order (the encoder's recon is in coding
# order, and with B pictures that is not the display order -- comparing the
# one with the other scores the reordering, not the encoder).
#
#   usage: crf_long.sh [frames] [crf]
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
W=1280; H=720; N=${1:-300}; CRF=${2:-24}
WORK=/tmp/aegcrf
rm -rf "$WORK"; mkdir -p "$WORK"
SRC="$WORK/clip.yuv"
[ -s "$SRC" ] || ffmpeg -v error -y -f lavfi -i "testsrc2=size=${W}x${H}:rate=30" \
    -frames:v "$N" -pix_fmt yuv420p -f rawvideo "$SRC"
echo "### ${W}x${H} ${N} frames, CRF $CRF: PSNR of the reconstruction against"
echo "    the source, per quarter of 75 frames"
for p in 2 4; do
  "$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o "$WORK/aeg$p.264" -p "$p" \
      --crf "$CRF" --dump-recon "$WORK/aeg$p.recon" \
      --dump-order "$WORK/aeg$p.order" > "$WORK/aeg$p.log" 2>&1
  printf '  aegenc preset %-3s ' "$p"
  python3 - "$SRC" "$WORK/aeg$p.recon" "$WORK/aeg$p.order" "$W" "$H" "$N" <<'PY'
import os, subprocess, sys
src, rec, orderf, W, H, N = sys.argv[1:7]
W, H, N = int(W), int(H), int(N)
fs = W * H * 3 // 2
A = open(src, "rb").read()
B = open(rec, "rb").read()
reported = [int(l) for l in open(orderf)]
n = min(N, len(B) // fs, len(A) // fs)
disp = [0] * n
for k in range(min(n, len(reported))):
    if 0 <= reported[k] < n:
        disp[reported[k]] = k
per = (n + 3) // 4
out = []
for q in range(4):
    idx = [d for d in range(q * per, min((q + 1) * per, n))]
    if not idx:
        break
    pa, pb = "/tmp/cq_s.yuv", "/tmp/cq_r.yuv"
    open(pa, "wb").write(b"".join(A[d * fs:(d + 1) * fs] for d in idx))
    open(pb, "wb").write(
        b"".join(B[disp[d] * fs:(disp[d] + 1) * fs] for d in idx))
    r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "rawvideo", "-s",
                        "%dx%d" % (W, H), "-pix_fmt", "yuv420p", "-i", pa,
                        "-f", "rawvideo", "-s", "%dx%d" % (W, H),
                        "-pix_fmt", "yuv420p", "-i", pb, "-lavfi", "psnr",
                        "-f", "null", "-"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    o = r.stderr.decode("utf-8", "replace")
    line = [l for l in o.splitlines() if "average:" in l]
    db = line[-1].split("average:")[1].split()[0] if line else "n/a"
    out.append("%s dB" % db)
print("  ".join(out))
PY
done
x264 --crf "$CRF" --preset medium --input-res ${W}x${H} --fps 30 \
     --keyint 250 --bframes 3 --ref 4 -o "$WORK/x264.264" "$SRC" >/dev/null 2>&1
ffmpeg -v error -y -i "$WORK/x264.264" -f rawvideo -pix_fmt yuv420p \
    "$WORK/x264.recon" 2>/dev/null
printf '  x264 medium             '
python3 - "$SRC" "$WORK/x264.recon" "$WORK/x264.order" "$W" "$H" "$N" <<'PY'
import os, subprocess, sys
src, rec, orderf, W, H, N = sys.argv[1:7]
W, H, N = int(W), int(H), int(N)
fs = W * H * 3 // 2
A = open(src, "rb").read()
B = open(rec, "rb").read()
n = min(N, len(B) // fs, len(A) // fs)
per = (n + 3) // 4
out = []
for q in range(4):
    idx = list(range(q * per, min((q + 1) * per, n)))
    if not idx:
        break
    pa, pb = "/tmp/cx_s.yuv", "/tmp/cx_r.yuv"
    open(pa, "wb").write(b"".join(A[d * fs:(d + 1) * fs] for d in idx))
    open(pb, "wb").write(b"".join(B[d * fs:(d + 1) * fs] for d in idx))
    r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "rawvideo", "-s",
                        "%dx%d" % (W, H), "-pix_fmt", "yuv420p", "-i", pa,
                        "-f", "rawvideo", "-s", "%dx%d" % (W, H),
                        "-pix_fmt", "yuv420p", "-i", pb, "-lavfi", "psnr",
                        "-f", "null", "-"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    o = r.stderr.decode("utf-8", "replace")
    line = [l for l in o.splitlines() if "average:" in l]
    db = line[-1].split("average:")[1].split()[0] if line else "n/a"
    out.append("%s dB" % db)
print("  ".join(out))
PY
echo "    (x264's stream decodes in display order, so it needs no reorder)"
