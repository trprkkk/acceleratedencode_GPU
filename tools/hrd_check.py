#!/usr/bin/env python3
"""Check the HRD timing an aegenc --nal-hrd stream signals.

    hrd_check.py <ffmpeg> <stream.264> <order file (--dump-order)>

Reads the buffering period / picture timing SEI through ffmpeg's
trace_headers bitstream filter and checks, picture by picture in coding
order, that the CPB removal times increase, every dpb_output_delay is
non-negative, and the output times are the display order at two clock ticks
per frame (time_scale = 2 * fps): t_out = T0 + 2 * display_index.
Exit status 0 when every picture agrees."""
import re
import subprocess
import sys

ff, stream, orderf = sys.argv[1:4]
txt = subprocess.run([ff, "-v", "info", "-i", stream, "-c", "copy", "-bsf:v",
                      "trace_headers", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
order = [int(x) for x in open(orderf)]
pics, cur = [], None
for line in txt.splitlines():
    if "Packet:" in line:
        cur = {"bp": None, "cpb": None, "dpb": None}
        pics.append(cur)
    m = re.search(r"initial_cpb_removal_delay\[0\]\s+\d+ = (\d+)", line)
    if m and cur is not None:
        cur["bp"] = int(m.group(1))
    m = re.search(r"cpb_removal_delay\s+\d+ = (\d+)", line)
    if m and cur is not None and "initial" not in line:
        cur["cpb"] = int(m.group(1))
    m = re.search(r"dpb_output_delay\s+\d+ = (\d+)", line)
    if m and cur is not None and "length" not in line:
        cur["dpb"] = int(m.group(1))
ok = len(pics) == len(order) and len(pics) > 0
if not ok:
    print("   hrd: %d packets, %d in the order file  FAIL" % (len(pics), len(order)))
t_bp = None
prev_tr = None
t0 = None
bps = 0
for k, (p, d) in enumerate(zip(pics, order)):
    if p["cpb"] is None or p["dpb"] is None:
        print("   hrd: picture %d has no picture timing SEI  FAIL" % k)
        ok = False
        break
    if p["bp"] is not None:
        bps += 1
        t_r = 0 if t_bp is None else t_bp + p["cpb"]
        t_bp = t_r
    else:
        if t_bp is None:
            print("   hrd: first picture without a buffering period  FAIL")
            ok = False
            break
        t_r = t_bp + p["cpb"]
    if prev_tr is not None and t_r <= prev_tr:
        print("   hrd: removal time of picture %d does not increase  FAIL" % k)
        ok = False
    prev_tr = t_r
    t_o = t_r + p["dpb"]
    if t0 is None:
        t0 = t_o - 2 * d
    if t_o != t0 + 2 * d:
        print("   hrd: picture %d (display %d) output at %d, expected %d  FAIL"
              % (k, d, t_o, t0 + 2 * d))
        ok = False
if ok:
    print("   hrd: %d pictures, %d buffering periods, output in display order"
          % (len(pics), bps))
sys.exit(0 if ok else 1)
