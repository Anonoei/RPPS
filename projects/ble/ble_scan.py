"""Scan the three Bluetooth LE advertising channels and summarize what is advertising

    python projects/ble/ble_scan.py                    # capture channels 37, 38, 39 (6 s each) and report
    python projects/ble/ble_scan.py --seconds 20 --keep captures/
    python projects/ble/ble_scan.py --files "captures/ble_{}.cf64"      # saved captures (4 MS/s, centred on the channel)
    python projects/ble/ble_scan.py --hide-names --addresses            # hide local names / show full addresses

A HackRF cannot follow connections (they hop over 37 data channels) but advertising is on three fixed channels: 2402, 2426 and 2480 MHz. Each channel is captured at
20 MS/s with the highest non-clipping gain (strong Wi-Fi next to channel 38 clips easily) and decimated to 4 MS/s; ble_adv.py finds packets (access address, dewhitening, CRC-24).
Advertisers are shown by kind of address and a 4-hex-digit hash of the address (the address itself only with --addresses); local names are shown unless --hide-names
(they are sometimes people's names). Only advertising traffic is read; nothing is decoded from connections.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from rpps.signal.sample import resample_poly

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import ble_adv
import ble_parse

FREQ = {37: 2402e6, 38: 2426e6, 39: 2480e6}

def analyse(x, channel, fs=4e6):
    p = np.abs(x.astype(np.complex128))**2
    floor = float(np.percentile(p[:len(p)//64*64].reshape(-1, 64).mean(1), 20))
    out = []; n_aa = 0
    xc = x.astype(np.complex128)
    for s, e in ble_adv.bursts(xc, fs):
        r = ble_adv.decode_burst(xc[max(0, s - 16):e + 16], channel, fs)
        if not r:
            continue
        n_aa += 1
        if r["crc_ok"]:
            r["snr"] = 10*np.log10(np.mean(p[s:e])/floor); r["t"] = s/fs
            r["parsed"] = ble_parse.parse(r)
            out.append(r)
    return out, n_aa

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--files", help="saved captures instead of the radio; {} is replaced by the channel (37/38/39)")
    ap.add_argument("--keep", help="save the 4 MS/s captures here as ble_<channel>.cf64")
    ap.add_argument("--hide-names", action="store_true", help="do not print local names (shown by default)")
    ap.add_argument("--hide-addresses", action="store_true", help="print full device addresses")
    a = ap.parse_args(argv)
    sdr = None
    if not a.files:
        import radio_io
        sdr = radio_io.open_radio(); radio_io.start_rx(sdr)
    packets = []; per_ch = {}
    try:
        for ch in (37, 38, 39):
            if a.files:
                x = np.fromfile(a.files.format(ch), np.complex64); g = None; clip = 0.0
            else:
                sdr.set_rx_freq(FREQ[ch], flush=True)
                x20, g, clip = radio_io.capture_clean(sdr, 20e6, a.seconds, check_seconds=1.0)
                x = resample_poly(x20.astype(np.complex64), 1, 5).astype(np.complex64)
                if a.keep:
                    x.tofile(os.path.join(a.keep, f"ble_{ch}.cf64"))
            pk, n_aa = analyse(x, ch)
            dur = len(x)/4e6
            per_ch[ch] = (len(pk), n_aa, dur, g, clip)
            for r in pk:
                r["channel"] = ch
            packets += pk
            print(f"channel {ch} ({FREQ[ch]/1e6:.0f} MHz): gain {g}, clipped {clip:.0e}, {dur:.1f} s: {len(pk)} packets with a valid CRC-24 ({n_aa} access addresses found)", flush=True)
    finally:
        if sdr:
            sdr.stop_rx(); sdr.close()
    if not packets:
        print("no packets"); return
    types = Counter(r["parsed"]["kind"] for r in packets)
    print(f"\n{len(packets)} packets: " + ", ".join(f"{k} {v}" for k, v in types.most_common()))
    devs = defaultdict(lambda: dict(n=0, snr=-99, kinds=Counter(), labels=Counter(), channels=set(), name=None, ak=None, full=None))
    for r in packets:
        d = r["parsed"]
        if "addr" not in d:
            continue
        key = (bytes(d["addr"]), d["addr_kind"])
        v = devs[key]; v["n"] += 1; v["snr"] = max(v["snr"], r["snr"]); v["kinds"][d["kind"]] += 1
        v["labels"].update(d["labels"]); v["channels"].add(r["channel"]); v["ak"] = d["addr_kind"]; v["full"] = ble_parse.addr_str(d["addr"])
        v["name"] = v["name"] or d["name"]
    ak = Counter(v["ak"] for v in devs.values())
    print(f"\n{len(devs)} distinct advertiser addresses ({dict(ak)}); resolvable-private addresses rotate every ~15 minutes, so this overcounts devices")
    print(f"named: {sum(1 for v in devs.values() if v['name'])} of them carry a local name" + ("" if a.hide_names else " (shown in the table)"))
    lab = Counter()
    for v in devs.values():
        for l in set(v["labels"]):
            lab[l] += 1
    print("what they say they are (number of addresses): " + ", ".join(f"{k} {n}" for k, n in lab.most_common(14)))
    print(f"\n  {'id':4s} {'address kind':24s} {'pkts':>5s} {'dB':>5s}  {'channels':8s} {'types':28s} labels" + ("   address" if not a.hide_addresses else "") + ("" if a.hide_names else "   name"))
    for key, v in sorted(devs.items(), key=lambda kv: -kv[1]["snr"])[:25]:
        row = (f"  {ble_parse.short_id(key[0]):4s} {v['ak']:24s} {v['n']:5d} {v['snr']:5.1f}  {','.join(map(str, sorted(v['channels']))):8s} "
               f"{','.join(k for k, _ in v['kinds'].most_common(2)):28s} {', '.join(l for l, _ in v['labels'].most_common(3))}")
        if not a.hide_addresses: row += "   " + v["full"]
        if not a.hide_names and v["name"]: row += "   " + repr(v["name"])
        print(row)
    return packets, devs

if __name__ == "__main__":
    main()
