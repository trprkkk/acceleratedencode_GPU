#!/usr/bin/env python3
"""x265_probe.py - verify that x265's analysis save/load is lossless.

Run this once a forked x265 binary exists.  It checks:
  1. baseline bitstream MD5
  2. --analysis-save writes per-frame files
  3. --analysis-load with --refine-inter 0 reproduces the same MD5

usage: py -3 tools/x265_probe.py <x265.exe> <in.yuv> <w> <h> <frames> <workdir>
"""
import hashlib, os, subprocess, sys

X265, IN, W, H, N, WD = sys.argv[1:7]
os.makedirs(WD, exist_ok=True)
PREFIX = os.path.join(WD, "analysis")


def run(out, extra):
    cmd = [X265, "--input", IN, "--input-res", f"{W}x{H}", "--fps", "30",
           "--frames", N, "--preset", "slow", "--crf", "27", "-o", out] + extra
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


base = os.path.join(WD, "base.264")
run(base, [])
base_md5 = md5(base)
print("base", base_md5)

run(os.path.join(WD, "saved.264"), ["--analysis-save", PREFIX])
files = sorted(f for f in os.listdir(WD) if f.startswith("analysis"))
print("analysis files", len(files), files[:3])

run(os.path.join(WD, "loaded.264"),
    ["--analysis-load", PREFIX, "--refine-inter", "0"])
loaded_md5 = md5(os.path.join(WD, "loaded.264"))
print("loaded", loaded_md5, "same", loaded_md5 == base_md5)

run(os.path.join(WD, "refined.264"),
    ["--analysis-load", PREFIX, "--refine-inter", "3"])
print("refine3", md5(os.path.join(WD, "refined.264")))
