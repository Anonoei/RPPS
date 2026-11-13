"""802.11 MAC frame parsing for what a passive scan sees: management frames (beacon, probe response/request, ...), control and data headers

IEEE 802.11-2020 clause 9. Frame control (2 bytes, little endian): protocol version (2 bits), type (2: 0 management, 1 control, 2 data), subtype (4), flags.
A beacon (type 0, subtype 8) is sent by every access point about every 102.4 ms: timestamp (8), beacon interval (2, in 1024 us units), capabilities (2) and
information elements (id, length, data): 0 SSID, 1 supported rates, 3 DS channel, 7 country, 45 HT capabilities, 48 RSN (WPA2/3), 61 HT operation,
127 extended capabilities, 191 VHT capabilities, 221 vendor (WPA1 = 00:50:F2:01, WPS = 00:50:F2:04), 255 extension (HE = 35).
"""
import struct

MGMT = {0: "assoc req", 1: "assoc resp", 2: "reassoc req", 3: "reassoc resp", 4: "probe req", 5: "probe resp", 8: "beacon", 9: "ATIM", 10: "disassoc", 11: "auth", 12: "deauth", 13: "action"}
CTRL = {7: "ctrl wrapper", 8: "block ack req", 9: "block ack", 10: "ps-poll", 11: "RTS", 12: "CTS", 13: "ACK", 14: "CF-end"}
DATA = {0: "data", 4: "null", 8: "QoS data", 12: "QoS null"}
CIPHER = {1: "WEP40", 2: "TKIP", 4: "CCMP", 5: "WEP104", 8: "GCMP", 9: "GCMP-256"}
AKM = {1: "802.1X", 2: "PSK", 3: "FT-802.1X", 4: "FT-PSK", 5: "802.1X-SHA256", 6: "PSK-SHA256", 8: "SAE", 9: "FT-SAE", 18: "OWE"}

def mac(b):
    return ":".join(f"{v:02x}" for v in b)

def parse(frame):
    """Header of any frame; for beacons and probe responses also the elements. Returns a dict (or None if too short)."""
    if len(frame) < 10:
        return None
    fc = struct.unpack("<H", frame[:2])[0]
    ver, typ, sub = fc & 3, (fc >> 2) & 3, (fc >> 4) & 15
    d = dict(type=typ, subtype=sub, flags=fc >> 8, retry=bool(fc & 0x0800), protected=bool(fc & 0x4000))
    if typ == 0:
        d["kind"] = MGMT.get(sub, f"mgmt {sub}")
        if len(frame) >= 24:
            d.update(da=mac(frame[4:10]), sa=mac(frame[10:16]), bssid=mac(frame[16:22]))
        if sub in (5, 8) and len(frame) >= 36 + 4:
            ts, interval, cap = struct.unpack("<QHH", frame[24:36])
            d.update(timestamp=ts, interval_tu=interval, capabilities=cap, privacy=bool(cap & 0x10), ess=bool(cap & 1))
            d["ie"] = elements(frame[36:-4])
            summarize(d)
        elif sub == 4 and len(frame) >= 28:
            d["ie"] = elements(frame[24:-4])
            d["ssid"] = next((v.decode("utf8", "replace") for i, v in d["ie"] if i == 0), None)
    elif typ == 1:
        d["kind"] = CTRL.get(sub, f"ctrl {sub}")
    else:
        d["kind"] = DATA.get(sub, f"data {sub}")
    return d

def elements(b):
    out = []; i = 0
    while i + 2 <= len(b):
        n = b[i + 1]
        if i + 2 + n > len(b):
            break
        out.append((b[i], bytes(b[i + 2:i + 2 + n])))
        i += 2 + n
    return out

def rsn(v):
    """RSN element: group cipher, pairwise ciphers, AKM suites"""
    try:
        gc = CIPHER.get(v[5], v[5]); n = struct.unpack("<H", v[6:8])[0]
        pc = [CIPHER.get(v[11 + 4*i], v[11 + 4*i]) for i in range(n)]
        o = 8 + 4*n; m = struct.unpack("<H", v[o:o + 2])[0]
        ak = [AKM.get(v[o + 5 + 4*i], v[o + 5 + 4*i]) for i in range(m)]
        return gc, pc, ak
    except (IndexError, struct.error):
        return None

def summarize(d):
    ssid, chan, sec, std = None, None, [], []
    for i, v in d["ie"]:
        if i == 0:
            ssid = v.decode("utf8", "replace") if v else ""
        elif i == 3 and v:
            chan = v[0]
        elif i == 48:
            r = rsn(v)
            if r:
                sec.append("WPA2/3 " + "+".join(map(str, r[2])) + " " + "/".join(map(str, r[1])))
        elif i == 221 and v[:4] == b"\x00\x50\xf2\x01":
            sec.append("WPA1")
        elif i == 45:
            std.append("n")
        elif i == 191:
            std.append("ac")
        elif i == 255 and v[:1] == b"\x23":
            std.append("ax")
    if not sec:
        sec = ["open" if not d.get("privacy") else "WEP/unknown"]
    d.update(ssid=ssid, channel=chan, security=", ".join(sec), standards=("b/g" + "".join("/" + s for s in std)))
