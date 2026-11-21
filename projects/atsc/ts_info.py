"""What is in an MPEG transport stream: the programs (PAT / PMT), their streams, the ATSC channel names (PSIP virtual channel table) and DVB service information (NIT / SDT)

    python projects/atsc/ts_info.py data/atsc/ch35.ts

PAT (PID 0) lists the programs and their PMT PIDs; each PMT lists the elementary streams (MPEG-2 video 0x02, H.264 0x1B, AC-3 audio 0x81, ...). The PSIP base PID 0x1FFB carries the terrestrial
virtual channel table (table id 0xC8): major.minor channel number, short name (UTF-16, 7 characters), program number, source id. DVB (EN 300 468): the NIT
(PID 0x10) names the network and, per transport stream, where it is sent (satellite delivery descriptor 0x43: frequency, orbital position, polarization,
symbol rate, modulation); the SDT (PID 0x11) names each service and its provider (service descriptor 0x48). Sections that were cut off by the capture's length, or whose CRC-32 fails (damaged packets), are skipped.
Meant to be upstreamed into RPPS.
"""
import argparse
import os
import sys
from collections import defaultdict

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import rpps as rp

_CRC32 = rp.link.load("crc/crc32_mpeg2").encoder # MPEG-2 sections end in a CRC-32 (not reflected): the CRC over a whole intact section is 0

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


def _bcd(b):
    """Packed BCD bytes as an integer"""
    return int("".join(f"{x:02x}" for x in b))

def _text(b):
    """A DVB string (EN 300 468 Annex A): a leading byte under 0x20 picks the character table, the rest Latin"""
    if b and b[0] < 0x20:
        b = b[1:] if b[0] != 0x10 else b[3:]
    return bytes(c for c in b if c >= 0x20 and not 0x80 <= c < 0xA0).decode("latin-1")

POL = ("H", "V", "L", "R")
DVB_MOD = {0: "auto", 1: "QPSK", 2: "8PSK", 3: "16QAM"}
SERVICE = {1: "TV", 2: "radio", 0x0A: "radio (advanced)", 0x0C: "data", 0x11: "HD TV (MPEG-2)", 0x16: "TV (H.264 SD)",
           0x19: "TV (H.264 HD)", 0x1F: "TV (HEVC)"}

def nit(secs):
    """The NIT: (network name, {(tsid, onid): delivery}) from network_information_sections (0x40 actual, 0x41 other)"""
    name, ts = None, {}
    for s in secs:
        if s[0] not in (0x40, 0x41) or len(s) < 16:
            continue
        nl = ((s[8] & 0x0F) << 8) | s[9]; k = 10
        for tag, body in _descriptors(s[k:k + nl]):
            if tag == 0x40:
                name = _text(body)
        k += nl
        tl = ((s[k] & 0x0F) << 8) | s[k + 1]; k += 2; end = min(k + tl, len(s) - 4)
        while k + 6 <= end:
            tsid = (s[k] << 8) | s[k + 1]; onid = (s[k + 2] << 8) | s[k + 3]; dl = ((s[k + 4] & 0x0F) << 8) | s[k + 5]
            for tag, d in _descriptors(s[k + 6:k + 6 + dl]):
                if tag == 0x43 and len(d) >= 11: # satellite delivery system
                    freq = _bcd(d[0:4])*10e3 # GHz with 5 decimals: units of 10 kHz
                    orb = _bcd(d[4:6])/10; east = bool(d[6] & 0x80)
                    sr = _bcd(d[7:11])//10*100 # Msym/s with 4 decimals... in units of 100 sym/s
                    fec = d[10] & 0x0F
                    ts[(tsid, onid)] = {"freq_hz": freq, "orbit": f"{orb:.1f}{'E' if east else 'W'}", "pol": POL[(d[6] >> 5) & 3],
                                        "s2": bool(d[6] & 0x04), "mod": DVB_MOD[d[6] & 3], "symbol_rate": sr,
                                        "fec": {1: "1/2", 2: "2/3", 3: "3/4", 4: "5/6", 5: "7/8", 6: "8/9", 7: "3/5", 8: "4/5", 9: "9/10"}.get(fec, fec),
                                        "table": "actual" if s[0] == 0x40 else "other"}
                elif tag == 0x44 and len(d) >= 11: # cable
                    ts.setdefault((tsid, onid), {"cable_freq_hz": _bcd(d[0:4])*100})
                elif tag == 0x5A and len(d) >= 4: # terrestrial
                    ts.setdefault((tsid, onid), {"terr_freq_hz": int.from_bytes(d[0:4], "big")*10})
            k += 6 + dl
    return name, ts

def sdt(secs):
    """The SDT: {(tsid, onid, service id): (provider, name, type)} from service_description_sections (0x42 actual, 0x46 other)"""
    out = {}
    for s in secs:
        if s[0] not in (0x42, 0x46) or len(s) < 15:
            continue
        tsid = (s[3] << 8) | s[4]; onid = (s[8] << 8) | s[9]; k = 11
        while k + 5 <= len(s) - 4:
            sid = (s[k] << 8) | s[k + 1]; dl = ((s[k + 3] & 0x0F) << 8) | s[k + 4]
            for tag, d in _descriptors(s[k + 5:k + 5 + dl]):
                if tag == 0x48 and len(d) >= 3:
                    pl = d[1]; prov = _text(d[2:2 + pl]); nl = d[2 + pl]; name = _text(d[3 + pl:3 + pl + nl])
                    out[(tsid, onid, sid)] = (prov, name, SERVICE.get(d[0], hex(d[0])), "actual" if s[0] == 0x42 else "other")
            k += 5 + dl
    return out

def _descriptors(b):
    k = 0
    while k + 2 <= len(b):
        tag, n = b[k], b[k + 1]
        yield tag, b[k + 2:k + 2 + n]
        k += 2 + n

def dvb_si(data):
    """Print the DVB network (NIT) and services (SDT); returns (network name, transport streams, services)"""
    sec = sections(data, {0x10, 0x11})
    name, ts = nit(sec.get(0x10, []))
    sv = sdt(sec.get(0x11, []))
    if name or ts:
        print(f"NIT: network {name!r}, {len(ts)} transport streams")
        for (tsid, onid), d in sorted(ts.items()):
            if "freq_hz" in d:
                print(f"  ts {tsid} onid {onid} ({d['table']}): {d['freq_hz']/1e6:.2f} MHz {d['pol']} at {d['orbit']}, "
                      f"{'DVB-S2' if d['s2'] else 'DVB-S'} {d['mod']} {d['symbol_rate']/1e6:.4f} MBd FEC {d['fec']}")
            else:
                print(f"  ts {tsid} onid {onid}: {d}")
    else:
        print("no NIT in this capture (it repeats every <=10 s)")
    if sv:
        print(f"SDT: {len(sv)} services")
        for (tsid, onid, sid), (prov, nm, typ, tab) in sorted(sv.items()):
            print(f"  ts {tsid} onid {onid} service {sid} ({tab}): {nm!r} from {prov!r}, {typ}")
    else:
        print("no SDT in this capture (it repeats every <=2 s)")
    return name, ts, sv


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
    dvb_si(data)


if __name__ == "__main__":
    main()
