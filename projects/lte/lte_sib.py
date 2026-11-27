"""LTE system information of every cell on a carrier: cell search -> MIB -> CFI -> DCI (SI-RNTI) -> PDSCH -> SIB1, SI messages

    python projects/lte/lte_sib.py data/lte_b13.cf64 --fs 15.36 --cf 753 --carrier 751.0
    python projects/lte/lte_sib.py cap.cf64 --fs 30.72 --cf 3589.9 --carrier 3589.9 --pci 179

The receiver is rpps.grid.lte (the profile lte/downlink): FDD or TDD, 1, 2 or 4 transmit antenna ports, system
information scheduled by DCI 1A (localized or distributed) or 1C. This finds the cells on the carrier (`rp.grid.lte.search`:
each one's synchronization signals are subtracted before the next is looked for), runs the profile on each and reports
what it read. SIB1 is sent in subframe 5 of every even radio frame (36.331 5.2.1.2); the other messages in the windows
SIB1 lists. A weaker cell may be found and still not decode under a stronger one.
"""
import argparse
import pprint
from collections import Counter

import numpy as np

import rpps as rp
from rpps.grid import lte
from rpps.grid.lte import phy

BLOCK = 262144

def n_rb_of(fs):
    """The resource blocks that fill the FFT of this sample rate"""
    nfft = int(round(fs/phy.SCS))
    if nfft not in phy.N_RB:
        raise ValueError(f"{fs/1e6:g} MS/s is not an LTE rate (1.92, 3.84, 7.68, 15.36, 23.04, 30.72)")
    return phy.N_RB[nfft]

def run(x, fs, offset, span, pci, n_rb):
    """The messages of lte/downlink on one cell"""
    rx = rp.profile.load("lte/downlink", n_rb=n_rb, offset=offset, span=span, pci=pci).rx()
    msgs = []
    for d in rp.source.Memory(x, Fs=fs)(BLOCK):
        rx(d)
        msgs += rx.take_messages()
    return msgs

def receive(x, fs, offset=0.0, span=50e3, pci=-1):
    """One cell of a capture (pci -1: the strongest), its carrier `offset` Hz from the centre. Returns a dict: pci,
    duplex, ports, n_rb, sfn (of the first MIB), mibs (decoded, tried), subframes (the SI-RNTI's: sfn, sf, cfi, dci,
    tbs, payload), sib1 (the distinct SIB1 payloads, most seen first)"""
    n_rb = n_rb_of(fs)
    msgs = run(x, fs, offset, span, pci, n_rb)
    good = [m for m in msgs if m.kind == "pbch" and m.crc_ok]
    if good and good[0].fields["n_rb"] not in (None, n_rb): # a narrower carrier than the rate holds
        n_rb = good[0].fields["n_rb"]
        msgs = run(x, fs, offset, span, pci, n_rb)
        good = [m for m in msgs if m.kind == "pbch" and m.crc_ok]
    tried = [m for m in msgs if m.kind == "pbch"]
    out = {"pci": pci if pci >= 0 else None, "duplex": None, "ports": None, "n_rb": None, "sfn": None, "mib": None,
           "mibs": (len(good), len(tried)), "subframes": [], "sib1": []}
    if tried:
        out["pci"] = tried[0].fields["pci"]
    if not good:
        return out
    first = good[0]
    out.update(duplex=first.fields["duplex"], ports=first.fields["ports"], n_rb=n_rb, sfn=first.fields["sfn"], mib=first.fields)
    frame0 = first.fields["subframe"]//10
    cfi = {m.fields["subframe"]: m.fields["cfi"] for m in msgs if m.kind == "pcfich"}
    blocks = {m.fields["subframe"]: m for m in msgs if m.kind == "pdsch"}
    seen = {}
    for m in msgs:
        s = m.fields.get("subframe")
        if m.kind != "pdcch" or s in seen: # (the first DCI of a subframe is the one received)
            continue
        b = blocks.get(s)
        sfn = (first.fields["sfn"] + s//10 - frame0) % 1024
        seen[s] = {"sfn": sfn, "sf": m.fields["sf"], "cfi": cfi.get(s), "dci": m.fields,
                   "tbs": None if b is None else b.fields["tbs"], "payload": b.payload if b is not None and b.crc_ok else None}
    out["subframes"] = [seen[s] for s in sorted(seen)]
    payloads = Counter(r["payload"] for r in out["subframes"] if r["payload"] and is_sib1(r))
    out["sib1"] = [p for p, _ in payloads.most_common()]
    return out

def special(r, s):
    """A DCI of TDD's special subframe 1, whose PDSCH rpps.grid.lte does not receive"""
    return r["duplex"] == "tdd" and s["sf"] == 1

def is_sib1(r):
    return r["sf"] == 5 and r["sfn"] % 2 == 0

PERIOD = {"rf8": 8, "rf16": 16, "rf32": 32, "rf64": 64, "rf128": 128, "rf256": 256, "rf512": 512}
WINDOW = {"ms1": 1, "ms2": 2, "ms5": 5, "ms10": 10, "ms15": 15, "ms20": 20, "ms40": 40}

def si_windows(sib1):
    """The system-information windows (36.331 5.2.3): message n (0-based in schedulingInfoList) with window w starts at x = n w; the frames are
    SFN mod T = floor(x/10), the first subframe x mod 10. Returns a list of (n, T, w, frame residue, first subframe, sibs)."""
    w = WINDOW[sib1["si-WindowLength"]]
    out = []
    for n, si in enumerate(sib1["schedulingInfoList"]):
        x = n*w
        out.append((n, PERIOD[si["si-Periodicity"]], w, x//10, x % 10, si["sib-MappingInfo"]))
    return out

def in_window(sfn, sf, win):
    n, T, w, res, a, _ = win
    t = sfn*10 + sf
    for base in range((sfn//T - 1)*T, (sfn//T + 2)*T, T):
        t0 = (base + res)*10 + a
        if t0 <= t < t0 + w:
            return True
    return False

def report(r, codec=None, si=True):
    """Print what `receive` read of a cell; with an ASN.1 codec (lte_rrc.load) the messages decoded, into r"""
    got, tried = r["mibs"]
    if not got:
        print(f"\nPCI {r['pci']}: no MIB decoded ({tried} tried)")
        return
    print(f"\nPCI {r['pci']}: {r['duplex'].upper()}, {r['n_rb']} RB, {r['ports']} port(s), MIB {got}/{tried}, first SFN {r['sfn']}, "
          f"PHICH Ng {r['mib']['phich_ng']:g}{' extended' if r['mib']['phich_duration'] else ''}")
    for s in r["subframes"]:
        d = s["dci"]
        if special(r, s):
            print(f"  SFN {s['sfn']:4d} sf {s['sf']}: CFI {s['cfi']}, {d['format']} in a special subframe (DwPTS): not covered")
            continue
        what = (f"{d['format']}{' distributed' if d.get('distributed') else ''} mcs {d.get('mcs')} TBS {s['tbs']}"
                if s["tbs"] is not None else f"{d['format']} (no transport block received)")
        print(f"  SFN {s['sfn']:4d} sf {s['sf']}: CFI {s['cfi']}, {what}: CRC24 {'OK' if s['payload'] else 'fail'}")
    ones = [s for s in r["subframes"] if is_sib1(s)]
    print(f"  CFI votes {dict(Counter(s['cfi'] for s in r['subframes']))}; SIB1: CRC ok in {sum(bool(s['payload']) for s in ones)}/{len(ones)} DCIs")
    for pl in r["sib1"]:
        print(f"  SIB1 payload (in {sum(s['payload'] == pl for s in ones)} subframes): {pl.hex()}")
    others = Counter(s["payload"] for s in r["subframes"] if s["payload"] and not is_sib1(s))
    if codec is None or not r["sib1"]:
        for pl, n in others.items():
            print(f"  other message (in {n} subframes): {pl.hex()}")
        return
    import lte_rrc
    sib1 = lte_rrc.decode_bcch_dl_sch(codec, r["sib1"][0])["message"][1][1]
    print("\nSIB1:"); pprint.pprint(sib1, width=140)
    r["sib1_msg"] = sib1
    if not si:
        return
    wins = si_windows(sib1)
    print("\nSI windows:", [(n + 1, f"every {T} frames", f"{w} ms", sibs) for n, T, w, _, _, sibs in wins])
    for pl, n in others.items():
        s = next(s for s in r["subframes"] if s["payload"] == pl)
        tag = next((f"SI message {w[0] + 1}" for w in wins if in_window(s["sfn"], s["sf"], w)), "outside the SI windows")
        try:
            m = lte_rrc.decode_bcch_dl_sch(codec, pl)
            print(f"\n{tag} (in {n} subframes, first SFN {s['sfn']} sf {s['sf']}):"); pprint.pprint(m, width=140)
        except Exception as e:
            print(f"\n{tag}: ASN.1 decode failed ({str(e)[:80]}); bytes {pl.hex()}")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="complex64 capture at an LTE rate")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate [MS/s]")
    ap.add_argument("--cf", type=float, required=True, help="Capture centre [MHz]")
    ap.add_argument("--carrier", type=float, required=True, help="Where the carrier's centre should be [MHz]")
    ap.add_argument("--span", type=float, default=150.0, help="Search +-this [kHz] around it (default 150: the 100 kHz raster plus a clock error)")
    ap.add_argument("--pci", type=int, action="append", help="receive this cell (repeatable; default: every cell found)")
    ap.add_argument("--asn", help="ASN.1 module built by lte_rrc.py (needs asn1tools): decode the messages and read the SI schedule from SIB1")
    ap.add_argument("--no-si", action="store_true", help="with --asn: SIB1 only")
    a = ap.parse_args(argv)
    fs, offset, span = a.fs*1e6, (a.carrier - a.cf)*1e6, a.span*1e3
    x = np.fromfile(a.path, np.complex64)
    print(f"{a.path}: {len(x)/fs:.2f} s at {a.fs:g} MS/s, centre {a.cf:g} MHz")
    found = lte.search(x, fs, offset=offset, span=span)
    for c in found:
        print(f"  cell {c['pci']:3d}: {c['duplex'].upper()}, carrier {(a.cf*1e6 + c['offset'])/1e6:.5f} MHz, {c['level_db']:+.1f} dB, "
              f"SSS {c['sss']:.2f}")
    pcis = a.pci or [c["pci"] for c in found]
    if not pcis:
        print("  no cell found")
    codec = None
    if a.asn:
        import lte_rrc
        codec = lte_rrc.load(a.asn)
    cells = []
    for pci in pcis:
        r = receive(x, fs, offset, span, pci)
        report(r, codec, not a.no_si)
        cells.append(r)
    if len(cells) > 1:
        print("\nSummary:")
        for r in cells:
            ones = [s for s in r["subframes"] if is_sib1(s)]
            print(f"  PCI {r['pci']:3d}: " + (f"{r['duplex'].upper()}, {r['n_rb']} RB, {r['ports']} port(s), SIB1 "
                  f"{sum(bool(s['payload']) for s in ones)}/{len(ones)}" if r["mibs"][0] else "found, MIB not decoded"))
    return dict(found=found, cells=cells)

if __name__ == "__main__":
    main()
