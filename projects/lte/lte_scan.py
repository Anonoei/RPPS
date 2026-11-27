"""Scan for LTE downlink carriers, identify each cell, and measure how busy it is

    python projects/lte/lte_scan.py --scan                      # sweep the default bands, find carriers, identify each (PCI, MIB, SIB1)
    python projects/lte/lte_scan.py --scan --traffic            # ... and lock to each and measure the load
    python projects/lte/lte_scan.py --carrier 885.0 --bw 10 --traffic --seconds 2     # skip the sweep: one known carrier
    python projects/lte/lte_scan.py --scan --range 1930 1995 --keep captures/         # a different range, keep the captures

Stages (each one prints what it found; every stage after the first needs the one before):
  1. sweep   : 20 MS/s tiles over each range, gain stepped down per tile until nothing clips; stitched power spectrum
  2. detect  : LTE carriers are flat plateaus of 4.5 / 9 / 13.5 / 18 MHz (occupied width of 5 / 10 / 15 / 20 MHz carriers) with steep edges; find them
  3. capture : per carrier, the gain search of gain_search.py, then a capture at 20 MS/s resampled to the LTE rate
               (1.92, 3.84, 7.68, 15.36, 23.04 or 30.72 MS/s), with the carrier 2 MHz off the HackRF's centre so its DC spike misses the sync signals
  4. identify: cell search, MIB and SIB1 of the strongest cell (lte_sib, over rpps.grid.lte): PCI, FDD or TDD, bandwidth, antenna ports, band, PLMN,
               tracking area, cell identity; the PCIs of any other cells on the carrier
  5. traffic : lock (carrier offset and subframe timing from the cell search) and, for every subframe of the capture, measure
               - reference-signal power per RE and the power of every PRB's data REs: a PRB is *active* when its data power is above 0.12 x the CRS power
               - CFI (how many symbols the control channel takes: it grows with the number of scheduled users)
               - PDCCH grants addressed to the public RNTIs: SI-RNTI (system information) and P-RNTI (a paging message in that subframe)
             Only these aggregates are reported. User data and paging records are not decoded (they carry subscriber identifiers).
References: TS 36.211 (PSS/SSS/CRS/PDCCH), 36.101 (bands), 36.331 (SIB1). The ratio threshold and the load model are in LTE.md.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import types
from collections import Counter

import numpy as np
from rpps.signal.sample import resample_poly

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _paths  # noqa: E402,F401
import gain_search
import radio_io
import lte_sib
import rpps as rp
from rpps.grid import lte
from rpps.grid.lte import phy

DEFAULT_RANGES = [(728.0, 756.0), (869.0, 894.0), (1930.0, 1995.0), (2110.0, 2155.0)] # MHz: bands 12/13/17, 5, 2, 4/66 downlink (North America)
CLASSES = [ # (name, occupied MHz, n_rb, LTE sample rate MHz, nominal bandwidth MHz)
    (1.4, 1.08, 6, 1.92), (3, 2.7, 15, 3.84), (5, 4.5, 25, 7.68), (10, 9.0, 50, 15.36), (15, 13.5, 75, 23.04), (20, 18.0, 100, 30.72)]
CAPTURE_FS = 20e6

# ---------------------------------------------------------------------------------------------------------------- the radio
def open_radio():
    return radio_io.open_radio(cf=100e6, fs=CAPTURE_FS)

set_gain, grab, clipped = radio_io.set_gain, radio_io.grab, radio_io.clipped

def quick_gain(sdr):
    """Highest gain that does not clip (a detection setting; identification then runs the full search)"""
    return sdr.find_gain(count=2**18, max_clip=1e-4, confirm=0)

def sweep(sdr, lo, hi, step=16e6, nfft=2**14):
    """Stitched power spectrum (Hz, dB) of lo..hi [Hz]; the DC spike of every tile masked"""
    frs, dbs, gains = [], [], []
    radio_io.start_rx(sdr)
    cf = lo + step/2
    while cf - step/2 < hi:
        sdr.set_rx_freq(cf, flush=True)
        g = quick_gain(sdr); gains.append(g)
        x = grab(sdr, 2**21)
        seg = x[:len(x)//nfft*nfft].reshape(-1, nfft)*np.hanning(nfft)
        P = np.mean(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1))**2, 0)
        fr = np.fft.fftshift(np.fft.fftfreq(nfft, 1/CAPTURE_FS)) + cf
        keep = (np.abs(fr - cf) < step/2) & (np.abs(fr - cf) > 150e3)
        frs.append(fr[keep]); dbs.append(10*np.log10(P[keep] + 1e-20))
        cf += step
    sdr.stop_rx()
    return np.concatenate(frs), np.concatenate(dbs), gains

# ---------------------------------------------------------------------------------------------------------------- detection
def detect(fr, db, min_excess=6.0, min_width=2.4e6, bin_hz=100e3):
    """Plateaus of an LTE width. Returns a list of dicts (lo, hi, centre, level, cls) [Hz, dB]."""
    from rpps.math import percentile_filter
    edges = np.arange(fr.min(), fr.max() + bin_hz, bin_hz)
    idx = np.digitize(fr, edges)
    p = np.array([10*np.log10(np.mean(10**(db[idx == i]/10))) if np.any(idx == i) else np.nan for i in range(1, len(edges))])
    ctr = edges[:-1] + bin_hz/2
    ok = ~np.isnan(p)
    p = np.interp(ctr, ctr[ok], p[ok])
    floor = percentile_filter(p, 15, size=int(24e6/bin_hz), mode="nearest")
    ex = p - floor
    on = ex > min_excess
    # close gaps of up to 3 bins (the masked DC and dips), then runs
    for i in range(1, len(on) - 3):
        if not on[i] and on[i - 1] and on[i:i + 4].any():
            on[i] = True
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i)*bin_hz >= min_width:
                runs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for i, j in runs:
        lo, hi = ctr[i] - bin_hz/2, ctr[j - 1] + bin_hz/2
        out.append(dict(lo=lo, hi=hi, width=hi - lo, level=float(ex[i:j].mean())))
    return out

def classify(width_hz):
    """Nearest LTE bandwidth class by occupied width (within 1.5 MHz); None if the plateau is not LTE-shaped"""
    best = min(CLASSES, key=lambda c: abs(c[1]*1e6 - width_hz))
    return best if abs(best[1]*1e6 - width_hz) < 1.5e6 else None

def candidates(runs):
    """Carrier guesses (centre MHz on the 100 kHz raster, class). A plateau wider than the biggest class (neighbouring carriers merge) is cut into
    10 MHz slices from its lower edge; unmatched widths are tried as the smaller classes at both edges."""
    out = []
    for r in runs:
        c = classify(r["width"])
        if c:
            out.append((round((r["lo"] + r["hi"])/2/1e5)/10, c, r))
        elif r["width"] > 19.5e6:
            k = 0
            while r["lo"] + (k + 1)*9.9e6 <= r["hi"] + 1e6:
                out.append((round((r["lo"] + 4.95e6 + k*9.9e6)/1e5)/10, CLASSES[3], r)); k += 1
    return out

# ---------------------------------------------------------------------------------------------------------------- capture
def pick_geometry(carrier_mhz, cls, psd):
    """Where to centre the capture and which band to call empty for the gain search, from the sweep's spectrum (Hz, dB) when there is one.
    The carrier goes 2 MHz (1.5 for 20 MHz carriers) to one side of the HackRF's centre so its DC spike misses the sync signals; the side is the one that
    leaves the strongest neighbour outside the baseband filter (about +-7.5 MHz). The 'empty' band is the quietest 0.6 MHz of the sweep inside the capture."""
    mag = 1.5e6 if cls[2] >= 100 else 2.0e6
    c = carrier_mhz*1e6
    half = cls[1]*1e6/2
    if psd is None:
        return c + mag, (c + mag + CAPTURE_FS/2*0.93 - 0.4e6, c + mag + CAPTURE_FS/2*0.93)
    fr, db = psd
    best = None
    for sign in (+1, -1):
        cf = c + sign*mag
        w = (fr > cf - 7.5e6) & (fr < cf + 7.5e6)
        out = w & (np.abs(fr - c) > half + 0.3e6)
        neigh = float(np.percentile(db[out], 98)) if out.any() else -200.0
        if best is None or neigh < best[0]:
            best = (neigh, cf)
    cf = best[1]
    quiet = None
    for f0 in np.arange(cf - 8.5e6, cf + 8.5e6, 0.2e6):
        if abs(f0 + 0.3e6 - c) < half + 0.5e6 or abs(f0 + 0.3e6 - cf) < 0.5e6:
            continue
        m = (fr >= f0) & (fr < f0 + 0.6e6)
        if m.sum() < 5:
            continue
        v = float(np.mean(db[m]))
        if quiet is None or v < quiet[0]:
            quiet = (v, f0)
    vac = (quiet[1], quiet[1] + 0.6e6) if quiet else (cf + CAPTURE_FS/2*0.93 - 0.4e6, cf + CAPTURE_FS/2*0.93)
    return cf, vac

def capture(sdr, carrier_mhz, cls, seconds, tune=True, keep=None, psd=None):
    """Gain search, then capture and resample. Returns (path, fs, cf, gain, clipped fraction). The radio streams only during the capture."""
    fs = cls[3]*1e6
    cf, vac = pick_geometry(carrier_mhz, cls, psd)
    sig = (carrier_mhz*1e6 - cls[1]*1e6/2*0.9, carrier_mhz*1e6 + cls[1]*1e6/2*0.9)
    radio_io.start_rx(sdr)
    try:
        sdr.set_rx_freq(cf, flush=True)
        if tune:
            g, _ = gain_search.search(sdr, cf, CAPTURE_FS, sig, vac, verbose=False)
        else:
            g = quick_gain(sdr)
        set_gain(sdr, g)
        x = grab(sdr, int(CAPTURE_FS*seconds))
    finally:
        sdr.stop_rx()
    clip = clipped(x)
    from fractions import Fraction
    fr = Fraction(fs/CAPTURE_FS).limit_denominator(1000)
    y = resample_poly(x, fr.numerator, fr.denominator).astype(np.complex64)
    path = os.path.join(keep or "/tmp", f"lte_{carrier_mhz:.1f}.cf64")
    y.tofile(path)
    return path, fs, cf, g, clip

# ---------------------------------------------------------------------------------------------------------------- identify
def sib1_fields(msg):
    cai = msg["cellAccessRelatedInfo"]
    plmn = cai["plmn-IdentityList"][0]["plmn-Identity"]
    eci = int.from_bytes(cai["cellIdentity"][0], "big") >> 4
    tac = int.from_bytes(cai["trackingAreaCode"][0], "big")
    return dict(mcc="".join(map(str, plmn.get("mcc", []))), mnc="".join(map(str, plmn["mnc"])), tac=tac, eci=eci, enb=eci >> 8, cell=eci & 255,
                band=msg["freqBandIndicator"], barred=cai["cellBarred"])

def parse_sib1_basic(payload):
    """Without an ASN.1 library: the first fields of SIB1 by hand (UPER). Returns the same dict as sib1_fields, or None if the layout is not what is expected.
    Layout (36.331): CHOICE bits (c1, SIB1) 2 | optional bitmap p-Max, tdd-Config, nonCriticalExtension 3 | csg-Identity present 1 | PLMN list length-1 (3 bits) |
    per PLMN: mcc present 1, 3 x 4 bits, mnc length-2 (1 bit), digits 4 bits each, reserved-for-operator 1 | TAC 16 | cell identity 28 | barred 1 |
    intra-frequency reselection 1 | CSG indication 1 | q-RxLevMinOffset present 1 | q-RxLevMin 6 | (offset 3) | freqBandIndicator 6 (value-1)"""
    bits = "".join(f"{b:08b}" for b in payload)
    p = 0
    def take(n):
        nonlocal p
        v = int(bits[p:p + n], 2); p += n
        return v
    try:
        if take(2) != 0b01:
            return None
        bm = take(3); csg_id = take(1)
        n_plmn = take(3) + 1
        plmn = None
        for i in range(n_plmn):
            has_mcc = take(1)
            mcc = "".join(str(take(4)) for _ in range(3)) if has_mcc else (plmn or {}).get("mcc", "")
            mnc_len = take(1) + 2
            mnc = "".join(str(take(4)) for _ in range(mnc_len))
            take(1)
            plmn = plmn or dict(mcc=mcc, mnc=mnc)
        tac = take(16); eci = take(28); barred = ("barred", "notBarred")[take(1)]
        take(1); take(1)
        if csg_id:
            take(27)
        has_off = take(1) # cellSelectionInfo: presence bit of q-RxLevMinOffset first
        take(6)
        if has_off:
            take(3)
        if bm & 0b100:
            take(6) # p-Max
        band = take(6) + 1
        return dict(mcc=plmn["mcc"], mnc=plmn["mnc"], tac=tac, eci=eci, enb=eci >> 8, cell=eci & 255, band=band, barred=barred)
    except (ValueError, IndexError):
        return None

def identify(path, fs, cf, carrier, asn=None, quiet=True):
    """Cell search, MIB and SIB1 of the strongest cell of a capture. Returns a dict (None when there is no cell; "error" where
    receiving it failed). `res` and `x` keep what the traffic stage needs."""
    buf = io.StringIO()
    try:
        x = np.fromfile(path, np.complex64)
        offset = (carrier*1e6 - cf)
        found = lte.search(x, fs, offset=offset, span=150e3)
        if not found:
            return None
        with contextlib.redirect_stdout(buf if quiet else sys.stdout):
            res = lte_sib.receive(x, fs, offset, 150e3, found[0]["pci"])
            codec = None
            if asn:
                import lte_rrc
                codec = lte_rrc.load(asn)
            lte_sib.report(res, codec, si=False)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {str(e)[:80]}", log=buf.getvalue())
    c = found[0]
    info = dict(pci=c["pci"], duplex=c["duplex"], n_rb=res["n_rb"], ports=res["ports"], offset=c["offset"],
                carrier_mhz=(cf + c["offset"])/1e6, others=[o["pci"] for o in found[1:]])
    info["valid"] = res["mibs"][0] > 0 # (a cell: the SSS agreed across occurrences, and the MIB's CRC passes)
    if "sib1_msg" in res:
        info["sib1"] = sib1_fields(res["sib1_msg"])
    elif res["sib1"]:
        info["sib1"] = parse_sib1_basic(res["sib1"][0])
    ones = [r for r in res["subframes"] if lte_sib.is_sib1(r)]
    info["sib1_ok"] = sum(bool(r["payload"]) for r in ones)
    info["sib1_tried"] = len(ones)
    info.update(res=res, x=x, fs=fs)
    return info

# ---------------------------------------------------------------------------------------------------------------- traffic
def grid_frames(info):
    """(header, G, H) of every subframe of the identified cell"""
    d = rp.Data(info["x"], Fs=info["fs"], flush=True)
    d = d + lte.LteCell(offset=info["offset"], span=10e3, pci=info["pci"]) + lte.LteGrid(info["n_rb"])
    y = np.asarray(d.data)
    F = lte.frame_len(info["n_rb"])
    for i in range(len(y)//F):
        yield lte.unpack(y[i*F:(i + 1)*F])

def downlink(G, pci, n_rb, sf):
    """Whether a subframe carries the cell's reference signals (a TDD cell's uplink subframes do not): neighbouring ones
    agree in phase once their values are taken off"""
    h = G[0, (phy.crs_shift(0, 2*sf, 0) + pci) % 6::6]*np.conj(phy.crs_values(pci, 2*sf, 0, n_rb))
    return abs(np.sum(h[1:]*np.conj(h[:-1])))/max(float(np.sum(np.abs(h)**2)), 1e-30) > 0.4

def traffic(info, seconds=None, threshold=0.12):
    """Lock to the identified cell and measure the load of every downlink subframe in the capture."""
    res = info["res"]
    pci, n_rb, ports = info["pci"], res["n_rb"], res["ports"]
    ks = np.arange(12*n_rb)
    crs_mask = (ks - pci % 6) % 3 == 0
    centre_prbs = set(range(n_rb//2 - 3, n_rb//2 + 3))
    k_dc = 6*n_rb - info["offset"]/15e3 # the radio's DC spike sits at the capture centre: a constant fake 'active' PRB if left in
    dc_prbs = {int(k_dc//12) + d for d in (-1, 0, 1)}
    mib = res["mib"]
    pcfich = lte.PCFICH(ports=ports)
    rows, cfis, grants, mi = [], [], Counter(), {}
    for head, G, H in grid_frames(info):
        sf = head["sf"]
        if not downlink(G, pci, n_rb, sf):
            continue
        P = np.abs(G)**2
        crs = np.mean([P[l][crs_mask] for l in (0, 4, 7, 11)])
        prb = np.zeros(n_rb)
        for q in range(n_rb):
            vals = []
            for l in range(3, 14):
                sl = slice(12*q, 12*q + 12)
                m = ~crs_mask[sl] if l in (4, 7, 8, 11) else np.ones(12, bool)
                vals.append(P[l][sl][m])
            prb[q] = np.mean(np.concatenate(vals))
        # CFI and the grants to public RNTIs (paging can only be in subframes 0, 4, 5, 9 of FDD; 0, 1, 5, 6 of TDD)
        cfi = pcfich.read(head, G, H)
        paging = (0, 1, 5, 6) if head["duplex"] == "tdd" else (0, 4, 5, 9)
        pdcch = lte.PDCCH(rntis=[phy.SI_RNTI] + ([phy.P_RNTI] if sf in paging else []), ports=ports, cfi=cfi,
                          phich_ng=mib["phich_ng"], phich_duration=mib["phich_duration"])
        for _, rnti, _ in pdcch.find(head, G, H, mi):
            grants["paging" if rnti == phy.P_RNTI else "system information"] += 1
        cfis.append(cfi)
        rows.append(dict(sf=sf, crs=crs, prb=prb, cfi=cfi))
    if not rows:
        return None
    crs_all = np.array([r["crs"] for r in rows])
    rho = np.array([r["prb"]/r["crs"] for r in rows]) # (subframes, PRBs)
    use = [q for q in range(n_rb) if q not in centre_prbs and q not in dc_prbs]
    active = rho[:, use] > threshold
    return dict(subframes=len(rows), seconds=len(rows)/1000, util=float(active.mean()), per_sf=active.mean(1), rho=rho, active=active,
                rho_db_pct={p: float(10*np.log10(np.percentile(rho[:, use], p) + 1e-12)) for p in (10, 50, 90)},
                crs_db=float(10*np.log10(np.median(crs_all))), cfi=Counter(cfis), grants=grants, rows=rows)

SPARK = " ▁▂▃▄▅▆▇█"
def spark(v, width=60):
    v = np.asarray(v, float)
    n = max(1, len(v)//width)
    c = v[:len(v)//n*n].reshape(-1, n).mean(1)
    return "".join(SPARK[min(8, int(round(x*8)))] for x in c)

def report_traffic(t, label):
    print(f"  traffic over {t['seconds']:.2f} s ({t['subframes']} subframes), CRS level {t['crs_db']:.1f} dB:")
    print(f"    PRB utilisation (data power > 0.12 x reference power, outside the 6 central PRBs and the DC spike's): {100*t['util']:.1f}% mean; "
          f"per-subframe 10/50/90th percentile {100*np.percentile(t['per_sf'], 10):.0f}/{100*np.percentile(t['per_sf'], 50):.0f}/{100*np.percentile(t['per_sf'], 90):.0f}%")
    print(f"    PRB data power relative to reference: 10/50/90th percentile {t['rho_db_pct'][10]:.1f} / {t['rho_db_pct'][50]:.1f} / {t['rho_db_pct'][90]:.1f} dB  "
          f"(idle PRBs sit far below, loaded ones near -3..-6 dB)")
    print(f"    load over time (each char ~{max(1, t['subframes']//60)} ms): [{spark(t['per_sf'])}]")
    tot = sum(t["cfi"].values())
    print(f"    control channel width (CFI) {dict(sorted((k, round(100*v/tot)) for k, v in t['cfi'].items()))} % of subframes, mean {sum(k*v for k, v in t['cfi'].items())/tot:.2f} symbols")
    print(f"    public-RNTI grants seen: {dict(t['grants'])}  (paging occasions with a message: {t['grants'].get('paging', 0)} in {t['seconds']:.1f} s)")

def heatmap(t, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(10*np.log10(t["rho"].T + 1e-6), aspect="auto", origin="lower", vmin=-25, vmax=3, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("subframe (ms)"); ax.set_ylabel("PRB"); ax.set_title(title)
    fig.colorbar(im, label="PRB data power / reference power [dB]")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------- main
def describe(cand, info):
    s = f"{cand:8.1f} MHz"
    if info is None or "error" in info:
        return s + f": {'no cell (' + info['error'] + ')' if info else 'no cell found'}"
    sb = info.get("sib1")
    if not info["valid"]:
        return s + f": synchronization signals of PCI {info['pci']} but no MIB decoded (too weak)"
    s += (f": PCI {info['pci']}, {info['duplex'].upper()}, {info['n_rb']} RB, {info['ports']} ports, "
          f"carrier {info['carrier_mhz']:.4f} MHz")
    if sb:
        s += f", band {sb['band']}, PLMN {sb['mcc']}-{sb['mnc']}, TAC {sb['tac']:#06x}, eNB {sb['enb']} cell {sb['cell']}"
    s += f"; SIB1 {info['sib1_ok']}/{info['sib1_tried']}"
    if info["others"]:
        s += f"; also on the carrier: PCI {', '.join(map(str, info['others']))} (lte_sib.py --pci)"
    return s

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="sweep and detect carriers")
    ap.add_argument("--range", nargs=2, type=float, action="append", metavar=("LO", "HI"), help="sweep this range [MHz] (repeatable; default: bands 12/13, 5, 2, 4)")
    ap.add_argument("--carrier", type=float, action="append", help="a carrier's centre [MHz]; skips the sweep (repeatable)")
    ap.add_argument("--bw", type=float, default=10, help="its bandwidth [MHz] (1.4, 3, 5, 10, 15, 20) (default 10)")
    ap.add_argument("--traffic", action="store_true", help="lock to each identified cell and measure the load")
    ap.add_argument("--seconds", type=float, default=1.0, help="capture length per carrier (default 1; 0.5 is enough to identify)")
    ap.add_argument("--asn", help="ASN.1 module from lte_rrc.py (needs asn1tools) to decode SIB1; without it the front of SIB1 is read by hand")
    ap.add_argument("--keep", help="keep the captures in this directory (default /tmp)")
    ap.add_argument("--no-tune", action="store_true", help="skip the per-carrier gain search (quick gain only)")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--file", help="analyse a saved capture (complex64) instead of using the radio; needs --fs [MS/s], --cf [MHz] and --carrier-nominal")
    ap.add_argument("--fs", type=float); ap.add_argument("--cf", type=float); ap.add_argument("--carrier-nominal", type=float)
    a = ap.parse_args(argv)
    if a.file:
        info = identify(a.file, a.fs*1e6, a.cf*1e6, a.carrier_nominal, asn=a.asn)
        print(describe(a.carrier_nominal, info))
        if a.traffic and info and "error" not in info and info["valid"]:
            t = traffic(info)
            report_traffic(t, a.carrier_nominal)
            if a.keep:
                heatmap(t, os.path.join(a.keep, f"traffic_{a.carrier_nominal:.1f}.png"), f"{a.carrier_nominal:.1f} MHz")
        return
    if not a.scan and not a.carrier:
        ap.error("give --scan, --carrier or --file")
    sdr = open_radio()
    cands = []
    psd = None
    try:
        if a.carrier:
            cls = min(CLASSES, key=lambda c: abs(c[0] - a.bw))
            cands = [(c, cls, None) for c in a.carrier]
        else:
            allruns = []
            psds = []
            for lo, hi in (a.range or DEFAULT_RANGES):
                print(f"sweeping {lo:.0f}-{hi:.0f} MHz ...", flush=True)
                fr, db, gains = sweep(sdr, lo*1e6, hi*1e6)
                runs = detect(fr, db)
                psds.append((fr, db))
                print(f"  {len(gains)} tiles, gains used {sorted(set(gains))}; {len(runs)} plateau(s): "
                      + ", ".join(f"{r['lo']/1e6:.1f}-{r['hi']/1e6:.1f} MHz (+{r['level']:.0f} dB)" for r in runs))
                allruns += runs
            cands = candidates(allruns)
            psd = (np.concatenate([p[0] for p in psds]), np.concatenate([p[1] for p in psds]))
            print(f"\n{len(cands)} candidate carrier(s): " + ", ".join(f"{c:.1f} MHz ({k[0]} MHz)" for c, k, _ in cands))
        results = []
        for carrier, cls, run in cands:
            print(f"\n=== {carrier:.1f} MHz, {cls[0]} MHz class ===", flush=True)
            path, fs, cf, g, clip = capture(sdr, carrier, cls, a.seconds, tune=not a.no_tune, keep=a.keep, psd=psd)
            print(f"  gain {g:.2f}, clipped {clip:.0e}; capture {a.seconds:g} s at {fs/1e6:g} MS/s, centre {cf/1e6:.2f} MHz")
            info = identify(path, fs, cf, carrier, asn=a.asn)
            print("  " + describe(carrier, info))
            entry = dict(carrier=carrier, bw=cls[0], gain=g, clip=clip)
            if info and "error" not in info and info["valid"]:
                entry.update(pci=info["pci"], n_rb=info["n_rb"], ports=info["ports"], carrier_mhz=info["carrier_mhz"], sib1=info.get("sib1"))
                if a.traffic:
                    t = traffic(info)
                    if t:
                        report_traffic(t, carrier)
                        if a.keep:
                            hp = os.path.join(a.keep, f"traffic_{carrier:.1f}.png")
                            heatmap(t, hp, f"{carrier:.1f} MHz, PCI {info['pci']}: PRB power over time")
                            print(f"    heat map: {hp}")
                        entry["traffic"] = dict(util=t["util"], seconds=t["seconds"], cfi=dict(t["cfi"]), grants=dict(t["grants"]))
                info.pop("res", None); info.pop("x", None)
            results.append(entry)
        print("\nSummary:")
        for e in results:
            sb = e.get("sib1")
            print(f"  {e['carrier']:8.1f} MHz {e['bw']:>4} MHz: " + (f"PCI {e['pci']}, {e['ports']} ports"
                  + (f", band {sb['band']}, {sb['mcc']}-{sb['mnc']}, eNB {sb['enb']}/{sb['cell']}" if sb else "")
                  + (f", load {100*e['traffic']['util']:.0f}%" if 'traffic' in e else "") if 'pci' in e else "no cell"))
        if a.json:
            json.dump(results, open(a.json, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    finally:
        sdr.close()

if __name__ == "__main__":
    main()
