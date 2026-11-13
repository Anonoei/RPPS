"""What is in an MPEG transport stream: the programs (PAT / PMT), their streams, and the ATSC channel names (PSIP virtual channel table)

    python projects/atsc/ts_info.py data/atsc/ch35.ts

PAT (PID 0) lists the programs and their PMT PIDs; each PMT lists the elementary streams (MPEG-2 video 0x02, H.264 0x1B, AC-3 audio 0x81, ...). The PSIP base PID 0x1FFB carries the terrestrial
virtual channel table (table id 0xC8): major.minor channel number, short name (UTF-16, 7 characters), program number, source id. Sections that were cut off by the capture's length, or whose CRC-32 fails (damaged packets), are skipped.
Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
from collections import defaultdict

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import rpps as rp

_CRC32 = rp.link.load("crc", "crc32_mpeg2")._enc # MPEG-2 sections end in a CRC-32 (not reflected): the CRC over a whole intact section is 0

STREAM = {0x01: "MPEG-1 video", 0x02: "MPEG-2 video", 0x03: "MPEG-1 audio", 0x04: "MPEG-2 audio", 0x0F: "AAC", 0x11: "AAC-LATM", 0x1B: "H.264 video", 0x24: "H.265 video", 0x81: "AC-3 audio", 0x87: "E-AC-3 audio"}


def sections(data, wanted):
    """{pid: [section bytes]} for the PIDs in wanted, reassembled from the TS packets"""
    buf = {}; out = defaultdict(list)
    for i in range(0, len(data) - 187, 188):
        p = data[i:i + 188]
        if p[0] != 0x47:
            continue
        pid = ((p[1] & 0x1F) << 8) | p[2]
        if pid not in wanted or not (p[3] & 0x10):
            continue
        off = 4 + (1 + p[4] if p[3] & 0x20 else 0)
        pl = p[off:]
        if p[1] & 0x40: # a section starts here
            pl = pl[1 + pl[0]:] if pl else b""
            buf[pid] = bytearray(pl)
        elif pid in buf:
            buf[pid] += pl
        else:
            continue
        b = buf.get(pid)
        while b is not None and len(b) >= 3:
            n = 3 + (((b[1] & 0x0F) << 8) | b[2])
            if len(b) < n:
                break
            sec = bytes(b[:n]); del b[:n]
            if _CRC32.compute(sec) == 0: # damaged sections (an uncorrectable Reed-Solomon block) are dropped
                out[pid].append(sec)
            if not b or b[0] == 0xFF:
                buf.pop(pid, None); break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); a = ap.parse_args(argv)
    data = open(a.file, "rb").read()
    sec = sections(data, {0, 0x1FFB})
    progs = {}
    for s in sec.get(0, []):
        if s[0] == 0 and len(s) > 12:
            for k in range(8, len(s) - 4, 4):
                n = (s[k] << 8) | s[k + 1]; pid = ((s[k + 2] & 0x1F) << 8) | s[k + 3]
                if n:
                    progs[n] = pid
    print(f"PAT: {len(progs)} programs: " + ", ".join(f"{n} (PMT {pid:#x})" for n, pid in sorted(progs.items())))
    pm = sections(data, set(progs.values()))
    for n, pid in sorted(progs.items()):
        for s in pm.get(pid, [])[:1]:
            if s[0] != 2:
                continue
            pil = ((s[10] & 0x0F) << 8) | s[11]; k = 12 + pil; es = []
            while k + 5 <= len(s) - 4:
                t = s[k]; ep = ((s[k + 1] & 0x1F) << 8) | s[k + 2]; il = ((s[k + 3] & 0x0F) << 8) | s[k + 4]
                es.append(f"{STREAM.get(t, hex(t))} (PID {ep:#x})"); k += 5 + il
            print(f"  program {n}: " + ", ".join(es))
    seen = {}
    for s in sec.get(0x1FFB, []):
        if s[0] in (0xC8, 0xC9) and len(s) > 14:
            k = 10; nch = s[9]
            for _ in range(nch):
                if k + 32 > len(s) - 4:
                    break
                name = s[k:k + 14].decode("utf-16-be", "replace").rstrip("\x00")
                major = ((s[k + 14] & 0x0F) << 6) | (s[k + 15] >> 2); minor = ((s[k + 15] & 0x03) << 8) | s[k + 16]
                prog = (s[k + 24] << 8) | s[k + 25]; src = (s[k + 28] << 8) | s[k + 29]; dl = ((s[k + 30] & 0x03) << 8) | s[k + 31]
                seen[(major, minor)] = (name, prog, src); k += 32 + dl
    if seen:
        print("PSIP virtual channels:")
        for (maj, mnr), (name, prog, src) in sorted(seen.items()):
            print(f"  {maj}.{mnr}  {name:8s}  program {prog}, source id {src}")
    else:
        print("no PSIP virtual channel table in this capture (it repeats every few hundred ms: capture longer)")


if __name__ == "__main__":
    main()
