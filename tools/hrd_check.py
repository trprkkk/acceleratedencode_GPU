#!/usr/bin/env python3
"""Check the HRD an aegenc --nal-hrd stream signals.

    hrd_check.py <ffmpeg (unused, kept for the old interface)> <stream.264>
                 <order file (--dump-order)>

Parses the byte stream itself: the SPS's VUI (timing, NAL HRD parameters,
pic_struct_present_flag), the access units (their sizes, start codes
included) and their buffering period / picture timing SEI.  Then, access unit
by access unit in coding order:

* timing: the CPB removal times increase, every dpb_output_delay is
  non-negative, and the output times are the display order at two clock
  ticks per frame (time_scale = 2 * fps) -- a field picture lasts one tick,
  the second field of a frame (pic_struct 1/2) is output a tick after the
  first: t_out = T0 + 2 * display_index + field;
* the CPB (Annex C.1, the signalled schedule): initial arrival times (VBR:
  no earlier than the removal time minus the initial delay (+ offset after
  the first access unit of a buffering period); CBR: back to back), final
  arrival no later than the removal time (no underflow), and the fullness
  just before every removal within the CPB size (no overflow).

Exit status 0 when everything agrees."""
import sys


class Bits:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def u(self, n):
        v = 0
        for _ in range(n):
            byte = self.d[self.p >> 3] if (self.p >> 3) < len(self.d) else 0
            v = (v << 1) | ((byte >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
            if z > 31:
                raise ValueError("bad ue(v)")
        return (1 << z) - 1 + self.u(z)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def rbsp(nal):
    out = bytearray()
    zeros = 0
    for b in nal:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def split_nals(data):
    """[(offset of the start code incl. leading zero bytes, payload)]"""
    starts = []
    i, n = 0, len(data)
    while i + 3 <= n:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            s = i
            while s > 0 and data[s - 1] == 0 and (not starts or s - 1 >= starts[-1][1]):
                s -= 1
            starts.append((s, i + 3))
            i += 3
        else:
            i += 1
    nals = []
    for k, (s, p) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else n
        nals.append((s, data[p:e]))
    return nals


def hrd_params(b):
    cnt = b.ue() + 1
    rs, cs = b.u(4), b.u(4)
    sched = []
    for _ in range(cnt):
        rate = (b.ue() + 1) << (6 + rs)
        size = (b.ue() + 1) << (4 + cs)
        cbr = b.u(1)
        sched.append((rate, size, cbr))
    il, cl, dl, tl = b.u(5) + 1, b.u(5) + 1, b.u(5) + 1, b.u(5)
    return {"sched": sched, "init_len": il, "cpb_len": cl, "dpb_len": dl}


def parse_sps(r):
    b = Bits(r)
    prof = b.u(8)
    b.u(16)
    b.ue()
    if prof in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        cf = b.ue()
        if cf == 3:
            b.u(1)
        b.ue()
        b.ue()
        b.u(1)
        if b.u(1):
            raise ValueError("scaling matrices in the SPS: not handled")
    b.ue()                                  # log2_max_frame_num_minus4
    pt = b.ue()
    if pt == 0:
        b.ue()
    elif pt == 1:
        b.u(1)
        b.se()
        b.se()
        for _ in range(b.ue()):
            b.se()
    b.ue()
    b.u(1)
    b.ue()
    b.ue()
    if not b.u(1):                          # frame_mbs_only_flag
        b.u(1)
    b.u(1)
    if b.u(1):
        b.ue(); b.ue(); b.ue(); b.ue()
    v = {"nal": None, "vcl": None, "pic_struct": 0, "units": 0, "scale": 0}
    if not b.u(1):
        return v
    if b.u(1):
        if b.u(8) == 255:
            b.u(32)
    if b.u(1):
        b.u(1)
    if b.u(1):
        b.u(4)
        if b.u(1):
            b.u(24)
    if b.u(1):
        b.ue()
        b.ue()
    if b.u(1):
        v["units"], v["scale"] = b.u(32), b.u(32)
        b.u(1)
    if b.u(1):
        v["nal"] = hrd_params(b)
    if b.u(1):
        v["vcl"] = hrd_params(b)
    if v["nal"] or v["vcl"]:
        b.u(1)
    v["pic_struct"] = b.u(1)
    return v


def parse_sei(r, sps, au):
    b = Bits(r)
    h = sps["nal"] or sps["vcl"]
    while (b.p >> 3) < len(r) - 1:
        t = 0
        while True:
            x = b.u(8)
            t += x
            if x != 255:
                break
        size = 0
        while True:
            x = b.u(8)
            size += x
            if x != 255:
                break
        end = b.p + 8 * size
        if t == 0 and h:
            b.ue()
            au["bp"] = (b.u(h["init_len"]), b.u(h["init_len"]))
        elif t == 1:
            if h:
                au["cpb"] = b.u(h["cpb_len"])
                au["dpb"] = b.u(h["dpb_len"])
            if sps["pic_struct"]:
                au["ps"] = b.u(4)
        b.p = end


def main():
    stream, orderf = sys.argv[2], sys.argv[3]
    data = open(stream, "rb").read()
    order = [int(x) for x in open(orderf)]
    sps = None
    aus = []
    cur = None
    prev_vcl = False
    for off, nal in split_nals(data):
        if not nal:
            continue
        t = nal[0] & 0x1f
        vcl = t in (1, 5)
        new = False
        if t == 9:
            new = True
        elif t in (6, 7, 8) and (prev_vcl or cur is None):
            new = True
        elif vcl and (cur is None or prev_vcl):
            # a slice starting a picture (first_mb_in_slice 0) after slices
            new = cur is None or Bits(rbsp(nal[1:4])).ue() == 0
        if new:
            cur = {"off": off, "bp": None, "cpb": None, "dpb": None, "ps": None}
            aus.append(cur)
        if t == 7 and sps is None:
            sps = parse_sps(rbsp(nal[1:]))
        elif t == 6 and sps is not None:
            parse_sei(rbsp(nal[1:]), sps, cur)
        prev_vcl = vcl
    for k, a in enumerate(aus):
        a["bits"] = 8 * ((aus[k + 1]["off"] if k + 1 < len(aus) else len(data)) - a["off"])

    ok = True
    h = sps and (sps["nal"] or sps["vcl"])
    if not h:
        print("   hrd: no HRD parameters in the SPS  FAIL")
        sys.exit(1)
    rate, size, cbr = h["sched"][0]
    tc = sps["units"] / sps["scale"]

    # frames / fields: pair the field access units, give each its frame
    frames = 0
    fld = []
    pend = False
    for a in aus:
        if a["ps"] in (1, 2):
            if pend:
                fld.append((frames - 1, 1))
                pend = False
            else:
                fld.append((frames, 0))
                frames += 1
                pend = True
        else:
            fld.append((frames, 0))
            frames += 1
    if frames != len(order) or not aus:
        print("   hrd: %d frames (%d access units), %d in the order file  FAIL"
              % (frames, len(aus), len(order)))
        sys.exit(1)

    t_bp = None
    prev_tr = None
    t0 = None
    bps = 0
    tr = []
    for k, a in enumerate(aus):
        fi, f = fld[k]
        d = order[fi]
        if a["cpb"] is None or a["dpb"] is None:
            print("   hrd: access unit %d has no picture timing SEI  FAIL" % k)
            sys.exit(1)
        if a["bp"] is not None:
            bps += 1
            t_r = 0 if t_bp is None else t_bp + a["cpb"]
            t_bp = t_r
        else:
            if t_bp is None:
                print("   hrd: first access unit without a buffering period  FAIL")
                sys.exit(1)
            t_r = t_bp + a["cpb"]
        if prev_tr is not None and t_r <= prev_tr:
            print("   hrd: removal time of access unit %d does not increase  FAIL" % k)
            ok = False
        prev_tr = t_r
        tr.append(t_r)
        t_o = t_r + a["dpb"]
        if t0 is None:
            t0 = t_o - 2 * d - f
        if t_o != t0 + 2 * d + f:
            print("   hrd: access unit %d (display %d, field %d) output at %d, "
                  "expected %d  FAIL" % (k, d, f, t_o, t0 + 2 * d + f))
            ok = False

    # the CPB (C.1): removal times in seconds from the first buffering period
    first_bp = aus[0]["bp"]
    base = first_bp[0] / 90000.0
    t_rem = [base + tc * x for x in tr]
    t_ai, t_af = [], []
    bp = first_bp
    worst = 0.0
    for k, a in enumerate(aus):
        if a["bp"] is not None:
            bp = a["bp"]
        if k == 0:
            ai = 0.0
        elif cbr:
            ai = t_af[-1]
        else:
            if a["bp"] is not None:
                early = t_rem[k] - bp[0] / 90000.0
            else:
                early = t_rem[k] - (bp[0] + bp[1]) / 90000.0
            ai = max(t_af[-1], early)
        af = ai + a["bits"] / rate
        t_ai.append(ai)
        t_af.append(af)
        if af > t_rem[k] + 1e-9:
            if ok or worst == 0.0:
                print("   hrd: CPB underflow at access unit %d: arrives %.6f s, "
                      "removed %.6f s  FAIL" % (k, af, t_rem[k]))
            worst = max(worst, af - t_rem[k])
            ok = False
    removed = 0
    peak = 0.0
    for k in range(len(aus)):
        t = t_rem[k]
        arrived = 0.0
        for m in range(len(aus)):
            if t_ai[m] >= t:
                break
            arrived += min(aus[m]["bits"], (min(t, t_af[m]) - t_ai[m]) * rate)
        full = arrived - removed
        peak = max(peak, full)
        if full > size + 1.0:
            print("   hrd: CPB overflow before access unit %d: %.0f > %d bits  FAIL"
                  % (k, full, size))
            ok = False
            break
        removed += aus[k]["bits"]
    if ok:
        nf = sum(1 for a in aus if a["ps"] in (1, 2))
        print("   hrd: %d access units (%d fields), %d buffering periods, output "
              "in display order, CPB peak %.0f%% of %d bits"
              % (len(aus), nf, bps, 100.0 * peak / size, size))
    sys.exit(0 if ok else 1)


main()
