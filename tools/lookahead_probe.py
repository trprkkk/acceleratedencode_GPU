#!/usr/bin/env python3
"""lookahead_probe.py - estimate the x264 lookahead/slicetype cost share.

Runs a small set of encoder configurations (3 times each, median) and prints
the timings.  Intended for the 640p cockL clip.

usage: py -3 tools/lookahead_probe.py <x264.exe> <in.yuv> <w> <h> <frames>
"""
import subprocess, sys, time, statistics, os

X264, IN, W, H, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
OUT = os.path.join(os.path.dirname(IN), "la_probe.264")
CFG = {
    "base": [],
    "no-mbtree": ["--no-mbtree"],
    "lookahead-0": ["--rc-lookahead", "0"],
    "lookahead-10": ["--rc-lookahead", "10"],
    "no-scenecut": ["--no-scenecut"],
}
for preset in ("medium", "slow"):
    for name, extra in CFG.items():
        ts, sizes = [], []
        for _ in range(3):
            cmd = [X264, "--input-res", f"{W}x{H}", "--fps", "30", "--frames", N,
                   "--threads", "4", "--preset", preset, "--crf", "27",
                   "--me", "hex", "--subme", "7", "-o", OUT, IN] + extra
            t = time.perf_counter()
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ts.append(time.perf_counter() - t)
            sizes.append(os.path.getsize(OUT))
        print(f"{preset:6s} {name:13s} median={statistics.median(ts):7.3f}s "
              f"spread={max(ts)-min(ts):.3f}s size={sizes[0]}")
