"""Scan the 2.4 / 5 GHz Wi-Fi channels and list the access points heard

    python projects/wifi/wifi_scan.py                       # channels 1-13 of 2.4 GHz, 2 s each
    python projects/wifi/wifi_scan.py --channels 1 6 11 --seconds 5
    python projects/wifi/wifi_scan.py --band5 --channels 36 40 44 48   # 5 GHz (OFDM only)
    python projects/wifi/wifi_scan.py --files data/wifi_ch{}.cf64      # analyse saved captures (channel number substituted)

For each channel: the highest gain that does not clip over 0.3 s (bursty traffic: the SNR-against-an-empty-band search of gain_search.py has nothing to
measure), a 2 s capture at 20 MS/s, burst detection, then both receivers: 802.11b DSSS (wifi_dsss.py: 1 and 2 Mbit/s, beacons are sent at 1 Mbit/s by most
2.4 GHz access points) and 802.11a/g OFDM (wifi_ofdm.py: required on 5 GHz). Only frames whose CRC-32 is valid count. The result is a table of access points
from their beacons and probe responses (BSSID, SSID, channel, security, standards, strength relative to the noise floor) and a count of what else was on the air.
Only access points are listed: client devices' addresses are neither collected nor printed. Strength is uncalibrated (dB over this capture's noise floor).
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import radio_io
import wifi_dsss
import wifi_mac
import wifi_ofdm


def freq_mhz(ch, band5=False):
    return 5000 + 5*ch if band5 else (2484 if ch == 14 else 2407 + 5*ch)

def capture(sdr, ch, seconds, band5=False):
    f = freq_mhz(ch, band5)*1e6
    sdr.set_rx_freq(f, flush=True)
    x, g, clip = radio_io.capture_clean(sdr, 20e6, seconds)
    return x.astype(np.complex64), g, clip

def analyse(x, fs=20e6):
    """All frames with a valid FCS in a capture: list of dicts (phy, rate, frame, snr_db, t)"""
    p = np.abs(x.astype(np.complex128))**2
    floor = float(np.percentile(p[:len(p)//100*100].reshape(-1, 100).mean(1), 20))
    x128 = x.astype(np.complex128)
    out = []; n_bursts = 0
    y = wifi_dsss.to_44(x128, fs)
    sc = 44e6/fs
    for s, e in wifi_dsss.bursts(x128, fs):
        n_bursts += 1
        r = wifi_dsss.decode_burst(y, int(s*sc), int(e*sc))
        if r and r.get("fcs_ok"):
            out.append(dict(phy="DSSS", rate=r["rate"], frame=r["frame"], snr=10*np.log10(np.mean(p[s:e])/floor), t=s/fs))
    for s, e in wifi_ofdm.bursts(x128, fs):
        r = wifi_ofdm.receive(x128, s, e, fs)
        if r and r.get("fcs_ok"):
            out.append(dict(phy="OFDM", rate=r["rate"], frame=r["frame"], snr=10*np.log10(np.mean(p[s:e])/floor), t=s/fs))
    return out, n_bursts, 10*np.log10(floor)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, nargs="+", help="channel numbers (default: 2.4 GHz 1-13)")
    ap.add_argument("--band5", action="store_true", help="channels are 5 GHz channels")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--files", help="saved captures instead of the radio; {} is replaced by the channel number")
    ap.add_argument("--keep", help="save the captures here as wifi_ch<N>.cf64")
    a = ap.parse_args(argv)
    chans = a.channels or list(range(1, 14))
    sdr = None
    if not a.files:
        sdr = radio_io.open_radio(); radio_io.start_rx(sdr)
    aps = {}
    kinds = Counter(); per_chan = {}
    try:
        for ch in chans:
            if a.files:
                x = np.fromfile(a.files.format(ch), np.complex64); g = None; clip = 0.0
            else:
                x, g, clip = capture(sdr, ch, a.seconds, a.band5)
                if a.keep:
                    x.tofile(os.path.join(a.keep, f"wifi_ch{ch}.cf64"))
            frames, nb, floor = analyse(x)
            c = Counter()
            for fr in frames:
                d = wifi_mac.parse(fr["frame"])
                if not d:
                    continue
                c[d["kind"]] += 1
                if d["kind"] in ("beacon", "probe resp") and d.get("bssid"):
                    key = d["bssid"]
                    cur = aps.get(key)
                    entry = dict(ssid=d.get("ssid"), channel=d.get("channel"), security=d.get("security"), standards=d.get("standards"), seen={ch: 1}, snr=fr["snr"], phy=fr["phy"], rate=fr["rate"],
                                 heard_on=ch, beacons=1 if d["kind"] == "beacon" else 0, interval=d.get("interval_tu"))
                    if cur:
                        cur["seen"][ch] = cur["seen"].get(ch, 0) + 1
                        cur["beacons"] += entry["beacons"]
                        if fr["snr"] > cur["snr"]:
                            cur.update(snr=fr["snr"], heard_on=ch)
                        cur["ssid"] = cur["ssid"] or entry["ssid"]
                    else:
                        aps[key] = entry
            per_chan[ch] = (nb, len(frames), dict(c), g, clip, floor)
            kinds.update(c)
            print(f"ch {ch:3d} ({freq_mhz(ch, a.band5)} MHz): gain {g}, {nb} DSSS-length bursts, {len(frames)} frames with a valid FCS: {dict(c)}", flush=True)
    finally:
        if sdr:
            sdr.stop_rx(); sdr.close()
    print(f"\n{len(aps)} access points (strength = dB over the capture's noise floor, uncalibrated):")
    print(f"  {'BSSID':17s} {'ch':>3s} {'heard on':>8s} {'dB':>5s}  {'security':28s} {'standards':14s} SSID")
    for b, v in sorted(aps.items(), key=lambda kv: -kv[1]["snr"]):
        print(f"  {b:17s} {str(v['channel']):>3s} {v['heard_on']:>8d} {v['snr']:5.1f}  {v['security']:28s} {v['standards']:14s} {v['ssid']!r}  ({v['phy']} {v['rate']} Mbit/s, {v['beacons']} beacons)")
    print(f"\nframe types heard (valid FCS): {dict(kinds)}")
    return aps, per_chan

if __name__ == "__main__":
    main()
